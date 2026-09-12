"""Alias-free synthesis network (StyleGAN3-T class).

Layer schedule: input Fourier field at 4x4 (content band = input Nyquist),
then a geometric progression of cutoff frequencies (doubling per stage,
cycles/image) up to the output Nyquist, with `num_critical` final layers
operating at the target sampling rate without resampling. Each non-critical
layer: modulated conv -> optional noise -> [up 2x with anti-imaging filter]
-> bias -> leaky ReLU (continuous domain) -> [low-pass at the output cutoff].
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..ops.filtered_lrelu import filtered_lrelu
from ..ops.filters import design_kaiser_filter
from .layers import ModulatedConv2D
from .synthesis_input import SynthesisInput


def channels_at(res: int, base: int = 32768, max_ch: int = 512) -> int:
    return min(base // res, max_ch)


class SynthesisLayer(tf.keras.layers.Layer):
    def __init__(self, w_dim: int, in_ch: int, out_ch: int,
                 in_res: int, out_res: int, c_in: float, c_out: float,
                 stopband_factor: float, critical: bool,
                 noise_mode: str = "random", name: str = "syn_layer"):
        super().__init__(name=name)
        self.in_res, self.out_res = in_res, out_res
        self.critical = critical
        self.noise_mode = noise_mode
        self.conv = ModulatedConv2D(out_ch, 3, demodulate=True, gain=1.0,
                                    name="conv")
        self.affine = tf.keras.layers.Dense(in_ch, name="affine")
        self.affine.build([None, w_dim])
        self.affine.set_weights([np.zeros((w_dim, in_ch), np.float32),
                                 np.ones(in_ch, np.float32)])
        self.bias = self.add_weight(name="bias", shape=[out_ch],
                                    initializer=tf.zeros_initializer())
        self.noise_strength = self.add_weight(name="noise_strength", shape=[],
                                              initializer=tf.zeros_initializer())
        self.const_noise = self.add_weight(
            name="const_noise", shape=[1, in_res, in_res, 1], trainable=False,
            initializer=tf.random_normal_initializer(0.0, 1.0))
        if not critical:
            half_in = c_in * (stopband_factor - 1.0) / 2.0
            half_out = c_out * (stopband_factor - 1.0) / 2.0
            self.fu = design_kaiser_filter(2.0 * c_in / out_res, half_in / out_res,
                                           sampling_rate=1.0)
            self.fd = design_kaiser_filter(c_out / out_res, half_out / out_res,
                                           sampling_rate=1.0)
        else:
            self.fu = self.fd = None

    def call(self, x, w, noise_mode=None, training=None):
        styles = self.affine(w)
        x = self.conv(x, styles)
        mode = noise_mode or self.noise_mode
        if mode != "none":
            noise = (tf.random.normal(tf.shape(x)) if mode == "random"
                     else self.const_noise)
            x = x + noise * tf.cast(self.noise_strength, x.dtype)
        if not self.critical:
            return filtered_lrelu(x, fu=self.fu, fd=self.fd, up=2, down=1,
                                  bias=self.bias)
        return filtered_lrelu(x, fu=None, fd=None, up=1, down=1, bias=self.bias)


class SynthesisNetwork(tf.keras.layers.Layer):
    def __init__(self, cfg: dict, name: str = "synthesis"):
        super().__init__(name=name)
        res = int(cfg["img_resolution"])
        base, maxc = cfg["channels_base"], cfg["channels_max"]
        self.cfg = cfg
        # cutoff schedule in cycles/image: geometric, doubling per stage
        num_up = int(round(np.log2(res))) - 2                 # 4 -> res
        c_sched = [cfg["first_cutoff_cycles"] * (2.0 ** i) for i in range(num_up + 1)]
        c_sched = [min(c, res / 2.0) for c in c_sched]

        in_bandwidth = c_sched[0] / 4.0                       # cycles/unit, span [-1,1]
        self.input_layer = SynthesisInput(cfg["w_dim"], channels_at(4, base, maxc),
                                    size=4, bandwidth=in_bandwidth, res=res)
        self.layers, self.num_ws = [], 1
        c_in, r_in = c_sched[0], 4
        for i in range(num_up):
            r_out = r_in * 2
            c_out = c_sched[i + 1]
            crit = False
            layer = SynthesisLayer(cfg["w_dim"], channels_at(r_in, base, maxc),
                                   channels_at(r_out, base, maxc),
                                   r_in, r_out, c_in, c_out,
                                   cfg["stopband_factor"], crit,
                                   noise_mode=cfg["noise_mode"],
                                   name=f"layer{i}")
            self.layers.append(layer)
            self.num_ws += 1
            c_in, r_in = c_out, r_out
        for j in range(int(cfg["num_critical"])):
            layer = SynthesisLayer(cfg["w_dim"], channels_at(r_in, base, maxc),
                                   channels_at(r_in, base, maxc),
                                   r_in, r_in, c_in, c_in,
                                   cfg["stopband_factor"], True,
                                   noise_mode=cfg["noise_mode"],
                                   name=f"critical{j}")
            self.layers.append(layer)
            self.num_ws += 1
        self.torgb = ModulatedConv2D(cfg["img_channels"], 1, demodulate=False,
                                     gain=1.0, name="to_rgb")
        self.torgb_affine = tf.keras.layers.Dense(channels_at(res, base, maxc),
                                                  name="to_rgb_affine")
        self.torgb_affine.build([None, cfg["w_dim"]])
        self.torgb_affine.set_weights([
            np.zeros((cfg["w_dim"], channels_at(res, base, maxc)), np.float32),
            np.ones(channels_at(res, base, maxc), np.float32)])
        self.torgb_bias = self.add_weight(
            name="to_rgb_bias", shape=[cfg["img_channels"]],
            initializer=tf.zeros_initializer())
        self.num_ws += 1

    def call(self, ws, shift=(0.0, 0.0), noise_mode=None, training=None):
        x = self.input_layer(ws[:, 0], shift=shift)
        for i, layer in enumerate(self.layers):
            x = layer(x, ws[:, i + 1], noise_mode=noise_mode, training=training)
        styles = self.torgb_affine(ws[:, -1])
        x = self.torgb(x, styles) + tf.cast(self.torgb_bias, x.dtype)
        return tf.cast(x, tf.float32)
