"""Equalized-parameterization layers and style-modulated convolution."""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _he_std(fan_in: float, gain: float = 1.0) -> float:
    return gain / np.sqrt(fan_in)


class EqualizedDense(tf.keras.layers.Layer):
    """Dense with equalized learning rate: kernel ~ N(0,1), runtime scale
    lr_mult * gain / sqrt(fan_in) (both init scale and effective LR)."""

    def __init__(self, units: int, lr_mult: float = 1.0, gain: float = 1.0,
                 activation=None, name: str = "eq_dense"):
        super().__init__(name=name)
        self.units = units
        self.lr_mult = lr_mult
        self.gain = gain
        self.activation = activation

    def build(self, input_shape):
        fan_in = int(input_shape[-1])
        self.scale = np.float32(self.lr_mult * self.gain / np.sqrt(fan_in))
        self.kernel = self.add_weight(
            name="kernel", shape=[fan_in, self.units],
            initializer=tf.random_normal_initializer(0.0, 1.0), trainable=True)
        self.bias = self.add_weight(
            name="bias", shape=[self.units],
            initializer=tf.zeros_initializer(), trainable=True)
        super().build(input_shape)

    def call(self, x):
        k = tf.cast(self.kernel, tf.float32) * self.scale
        y = tf.matmul(tf.cast(x, tf.float32), k)
        b = tf.cast(self.bias, tf.float32) * np.float32(self.lr_mult)
        y = tf.cast(y + b, self.compute_dtype)
        if self.activation is not None:
            y = self.activation(y)
        return y


class EqualizedConv2D(tf.keras.layers.Layer):
    """Conv2D with equalized parameterization (same scheme as EqualizedDense)."""

    def __init__(self, filters: int, kernel_size: int = 3, stride: int = 1,
                 lr_mult: float = 1.0, gain: float = 1.0, name: str = "eq_conv"):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.lr_mult = lr_mult
        self.gain = gain

    def build(self, input_shape):
        k = self.kernel_size
        fan_in = k * k * int(input_shape[-1])
        self.scale = np.float32(self.lr_mult * self.gain / np.sqrt(fan_in))
        self.kernel = self.add_weight(
            name="kernel", shape=[k, k, int(input_shape[-1]), self.filters],
            initializer=tf.random_normal_initializer(0.0, 1.0), trainable=True)
        self.bias = self.add_weight(name="bias", shape=[self.filters],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super().build(input_shape)

    def call(self, x):
        w = tf.cast(self.kernel, tf.float32) * self.scale
        pad = (self.kernel_size - 1) // 2
        x_pad = tf.pad(tf.cast(x, tf.float32), [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        y = tf.nn.conv2d(x_pad, w, strides=[1, self.stride, self.stride, 1],
                         padding="VALID")
        b = tf.cast(self.bias, tf.float32) * np.float32(self.lr_mult)
        return tf.cast(y + b, self.compute_dtype)


class ModulatedConv2D(tf.keras.layers.Layer):
    """Style-modulated 3x3 (or 1x1) convolution with optional demodulation.

    styles: (B, in_channels) per-sample scale factors (~1 at init).
    Demodulation rescales each output channel so that the style magnitude
    does not propagate (StyleGAN2 'demodulate').
    """

    def __init__(self, out_channels: int, kernel_size: int = 3,
                 demodulate: bool = True, lr_mult: float = 1.0,
                 gain: float = 1.0, name: str = "mod_conv"):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.lr_mult = lr_mult
        self.gain = gain

    def build(self, input_shape):
        k = self.kernel_size
        self.fan_in = k * k * int(input_shape[-1])
        self.scale = np.float32(self.lr_mult * self.gain / np.sqrt(self.fan_in))
        self.kernel = self.add_weight(
            name="kernel", shape=[k, k, int(input_shape[-1]), self.out_channels],
            initializer=tf.random_normal_initializer(0.0, 1.0), trainable=True)
        super().build(input_shape)

    def call(self, x, styles):
        B = tf.shape(x)[0]
        w = tf.cast(self.kernel, tf.float32) * self.scale
        styles = tf.cast(styles, tf.float32)
        w = w[None] * styles[:, None, None, :, None]           # (B,k,k,cin,cout)
        if self.demodulate:
            demod = tf.math.rsqrt(tf.reduce_sum(tf.square(w), axis=[1, 2, 3]) + 1e-8)
            w = w * demod[:, None, None, None, :]
        pad = (self.kernel_size - 1) // 2
        x_pad = tf.pad(tf.cast(x, tf.float32), [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        out = tf.map_fn(
            lambda elem: tf.nn.conv2d(elem[0][None], elem[1], strides=[1, 1, 1, 1], padding="VALID")[0],
            (x_pad, w),
            fn_output_signature=tf.float32,
            parallel_iterations=16
        )
        return tf.cast(out, self.compute_dtype)
