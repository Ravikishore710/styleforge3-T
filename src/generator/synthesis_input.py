# Band-limited Fourier synthesis input with a learned affine transform
from __future__ import annotations

import numpy as np
import tensorflow as tf


class SynthesisInput(tf.keras.layers.Layer):
    def __init__(self, w_dim: int, channels: int, size: int,
                 bandwidth: float, res: int = 64, name: str = "synthesis_input"):
        super().__init__(name=name)
        self.channels = channels
        self.size = size
        self.res = int(res)
        self.bandwidth = float(bandwidth)

        rng = np.random.default_rng(0)
        freqs = rng.standard_normal((channels, 2)).astype(np.float32)
        radii = np.sqrt((freqs ** 2).sum(axis=1, keepdims=True))
        freqs = freqs / (radii * np.exp(radii ** 2 / 4.0))   # disc, |f| <= 1
        freqs *= self.bandwidth
        phases = (rng.uniform(-0.5, 0.5, size=channels)).astype(np.float32)

        self.freqs = self.add_weight(shape=freqs.shape, trainable=False,
                                     name="freqs",
                                     initializer=tf.constant_initializer(freqs))
        self.phases = self.add_weight(shape=phases.shape, trainable=False,
                                      name="phases",
                                      initializer=tf.constant_initializer(phases))
        self.weight = self.add_weight(
            shape=[channels, channels], name="weight",
            initializer=tf.random_normal_initializer(0.0, 1.0 / np.sqrt(channels)))
        self.affine = tf.keras.layers.Dense(4, name="affine")
        # identity transform at init: weight 0, bias [1, 0, 0, 0]
        self.affine.build([None, w_dim])
        self.affine.set_weights([np.zeros((w_dim, 4), np.float32),
                                 np.array([1.0, 0.0, 0.0, 1.0], np.float32)])
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        grid = np.stack([(xx + 0.5) / size * 2.0 - 1.0,
                         (yy + 0.5) / size * 2.0 - 1.0], axis=-1).reshape(-1, 2)
        self.grid = self.add_weight(shape=grid.shape, trainable=False, name="grid",
                                    initializer=tf.constant_initializer(grid))

    def call(self, w, shift=(0.0, 0.0)):
        B = tf.shape(w)[0]
        w = tf.cast(w, tf.float32)
        t = tf.cast(self.affine(w), tf.float32)                                # (B,4)
        t = tf.reshape(t, [B, 2, 2])
        f = tf.matmul(t, tf.cast(self.freqs, tf.float32), transpose_b=True)   # (B,2,C)
        coords = tf.cast(self.grid, tf.float32)
        try:
            sy, sx = float(shift[0]), float(shift[1])
        except Exception:
            sy, sx = 0.0, 0.0
        if sy != 0.0 or sx != 0.0:
            pixel = 2.0 / float(self.res)                                      # unit length per pixel
            coords = coords - tf.constant([sx * pixel, sy * pixel], tf.float32)[None, :]
        ang = 2.0 * np.pi * tf.matmul(coords, f)                              # (B,N,C)
        ang = ang + tf.cast(self.phases, tf.float32)[None, None, :]
        x = tf.sin(ang)                                                       # (B,N,C)
        x = tf.matmul(x, tf.cast(self.weight, tf.float32))                    # channel mixing
        x = tf.reshape(x, [B, self.size, self.size, self.channels])
        return tf.cast(x, self.compute_dtype)
