"""Filtered resampling: upsample (zero-stuff) -> FIR conv -> decimate.

Separable application of a symmetric 1-D FIR filter along rows and columns
via depthwise conv2d. Everything runs in float32 for numerical stability and
casts back to the input dtype on return.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


_FILTER_CACHE: dict = {}


def _get_separable_filters(f, cin: int):
    f1 = np.asarray(f, dtype=np.float32)
    if f1.ndim == 2:
        f1 = f1[0]
    key = (tuple(f1.tolist()), cin)
    if key not in _FILTER_CACHE:
        kh = len(f1)
        px = (kh - 1) // 2
        arr_x = np.tile(f1[None, :, None, None], [1, 1, cin, 1])
        arr_y = np.tile(f1[:, None, None, None], [1, 1, cin, 1])
        _FILTER_CACHE[key] = (arr_x, arr_y, px)
    arr_x, arr_y, px = _FILTER_CACHE[key]
    return tf.constant(arr_x, dtype=tf.float32), tf.constant(arr_y, dtype=tf.float32), px


def _upsample_zero_stuff(x: tf.Tensor, up: int) -> tf.Tensor:
    b = tf.shape(x)[0]
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = tf.reshape(x, [b, h, 1, w, 1, c])
    x = tf.pad(x, [[0, 0], [0, 0], [0, up - 1], [0, 0], [0, up - 1], [0, 0]])
    return tf.reshape(x, [b, h * up, w * up, c])


def upfirdn2d(x: tf.Tensor, f, up: int = 1, down: int = 1,
              gain: float = 1.0) -> tf.Tensor:
    """Upsample by `up` (zero-stuffing), convolve with separable FIR `f`,
    decimate by `down`. Output length along each axis: floor((H*up)/down)."""
    in_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    cin = int(x.shape[-1])
    filtx, filty, px = _get_separable_filters(f, cin)

    if up > 1:
        x = _upsample_zero_stuff(x, up)

    # 1D horizontal pass
    xp = tf.pad(x, [[0, 0], [0, 0], [px, px], [0, 0]])
    x = tf.nn.depthwise_conv2d(xp, filtx, strides=[1, 1, 1, 1], padding="VALID")

    # 1D vertical pass
    yp = tf.pad(x, [[0, 0], [px, px], [0, 0], [0, 0]])
    x = tf.nn.depthwise_conv2d(yp, filty, strides=[1, 1, 1, 1], padding="VALID")

    if down > 1:
        x = x[:, ::down, ::down, :]
    if gain != 1.0:
        x = x * np.float32(gain)
    return tf.cast(x, in_dtype)


def upsample2d(x: tf.Tensor, f, gain: float = 4.0) -> tf.Tensor:
    """2x upsampling with anti-imaging filter and energy gain up^2."""
    return upfirdn2d(x, f, up=2, down=1, gain=gain)


def downsample2d(x: tf.Tensor, f, gain: float = 1.0) -> tf.Tensor:
    """2x downsampling with anti-alias filter."""
    return upfirdn2d(x, f, up=1, down=2, gain=gain)
