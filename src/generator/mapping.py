"""z -> w mapping network (StyleGAN3 default depth: 2)."""
from __future__ import annotations

import tensorflow as tf

from .layers import EqualizedDense


def lrelu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


class MappingNetwork(tf.keras.layers.Layer):
    def __init__(self, z_dim: int = 512, w_dim: int = 512, depth: int = 2,
                 lr_mult: float = 0.01, num_ws: int = 1, name: str = "mapping"):
        super().__init__(name=name)
        self.num_ws = num_ws
        self.layers = [EqualizedDense(w_dim, lr_mult=lr_mult, activation=lrelu,
                                      name=f"map_{i}") for i in range(depth)]

    def call(self, z):
        z = tf.cast(z, tf.float32)
        # normalize latents to hypersphere (StyleGAN convention)
        z = z * tf.math.rsqrt(tf.reduce_mean(tf.square(z), axis=1, keepdims=True) + 1e-8)
        w = z
        for layer in self.layers:
            w = layer(w)
        if self.num_ws > 1:
            w = tf.tile(w[:, None, :], [1, self.num_ws, 1])
        return w
