"""Residual convolutional discriminator with filtered downsampling.

Architecture (StyleGAN2/3 style): from_rgb 1x1 at full resolution, then a
chain of residual blocks (3x3 conv -> LReLU -> 3x3 conv -> LReLU -> filtered
downsample, with a 1x1 + downsample skip), minibatch-stddev epilogue at 4x4,
final conv and single logit. Convs use equalized parameterization.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..generator.layers import EqualizedConv2D, EqualizedDense
from ..ops.filters import binomial_filter
from ..ops.upfirdn import downsample2d


def _channels(res: int, base: int, maxc: int) -> int:
    return min(base // res, maxc)


def lrelu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


class MinibatchStd(tf.keras.layers.Layer):
    """Concatenates the across-batch std (averaged over channels) as a feature map."""

    def __init__(self, group_size: int = 4, name: str = "minibatch_std"):
        super().__init__(name=name)
        self.group_size = group_size

    def call(self, x):
        B = tf.shape(x)[0]
        gs = tf.minimum(self.group_size, B)
        y = tf.reshape(x, [gs, -1, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]])
        y = tf.cast(y, tf.float32)
        var = tf.math.reduce_variance(y, axis=0)
        std = tf.sqrt(tf.maximum(var, 0.0) + 1e-14)
        std = tf.reduce_mean(std, axis=-1, keepdims=True)      # (B/gs,H,W,1)
        std = tf.tile(std, [gs, 1, 1, 1])
        return tf.cast(tf.concat([x, std], axis=-1), x.dtype)


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, in_ch: int, out_ch: int, name: str = "d_block"):
        super().__init__(name=name)
        self.conv0 = EqualizedConv2D(out_ch, 3, gain=np.sqrt(2), name="conv0")
        self.conv1 = EqualizedConv2D(out_ch, 3, gain=np.sqrt(2), name="conv1")
        self.skip = EqualizedConv2D(out_ch, 1, gain=1.0, name="skip")
        self.filt = binomial_filter(5)

    def call(self, x):
        y = lrelu(self.conv0(x))
        y = lrelu(self.conv1(y))
        y = downsample2d(y, self.filt)
        s = downsample2d(self.skip(x), self.filt)
        return y + s


class Discriminator(tf.keras.Model):
    def __init__(self, cfg: dict, name: str = "discriminator"):
        super().__init__(name=name)
        res = int(cfg["img_resolution"])
        base, maxc = cfg["d_channels_base"], cfg["d_channels_max"]
        self.from_rgb = EqualizedConv2D(_channels(res, base, maxc), 1,
                                        gain=1.0, name="from_rgb")
        self.blocks = []
        r = res
        while r > 4:
            self.blocks.append(DiscriminatorBlock(_channels(r, base, maxc),
                                                  _channels(r // 2, base, maxc),
                                                  name=f"block_{r//2}"))
            r //= 2
        self.mbstd = MinibatchStd(cfg["mbstd_group_size"])
        self.conv_final = EqualizedConv2D(_channels(4, base, maxc), 3,
                                          gain=np.sqrt(2), name="conv_final")
        self.fc = EqualizedDense(_channels(4, base, maxc), gain=np.sqrt(2),
                                 name="fc")
        self.logit = EqualizedDense(1, gain=1.0, name="logit")

    def call(self, x, training=None):
        x = lrelu(self.from_rgb(x))
        for block in self.blocks:
            x = block(x)
        x = self.mbstd(x)
        x = lrelu(self.conv_final(x))
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = lrelu(self.fc(x))
        return self.logit(x)   # (B, 1), float32
