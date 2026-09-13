# Continuous-domain leaky ReLU with band limiting (StyleGAN3-T core op)
from __future__ import annotations

import tensorflow as tf

from .upfirdn import upfirdn2d


def filtered_lrelu(x: tf.Tensor, fu=None, fd=None, up: int = 1, down: int = 1,
                   bias=None, slope: float = 0.2, gain: float = 1.0) -> tf.Tensor:
    in_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    if fu is not None:
        x = upfirdn2d(x, fu, up=up, down=1, gain=float(up) ** 2)
    if bias is not None:
        x = x + tf.cast(tf.reshape(bias, [1, 1, 1, -1]), tf.float32)
    x = tf.nn.leaky_relu(x, alpha=slope)
    if fd is not None:
        x = upfirdn2d(x, fd, up=1, down=down, gain=1.0)
    x = x * np_float32(gain)
    return tf.cast(x, in_dtype)


def np_float32(v: float) -> float:
    import numpy as np
    return float(np.float32(v))
