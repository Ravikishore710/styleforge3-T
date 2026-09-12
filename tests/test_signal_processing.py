import numpy as np
import tensorflow as tf

from src.ops.filtered_lrelu import filtered_lrelu
from src.ops.filters import binomial_filter, design_kaiser_filter
from src.ops.upfirdn import downsample2d, upfirdn2d, upsample2d


def test_upfirdn_upsample_shape_and_energy():
    x = tf.ones([2, 16, 16, 3])
    y = upsample2d(x, binomial_filter(5))
    assert y.shape == (2, 32, 32, 3)
    # interior pixels have DC gain 1.0
    assert np.allclose(y[:, 4:-4, 4:-4, :].numpy(), 1.0, atol=1e-5)


def test_upfirdn_downsample_shape():
    x = tf.random.normal([2, 16, 16, 3])
    y = downsample2d(x, binomial_filter(5))
    assert y.shape == (2, 8, 8, 3)


def test_upfirdn_identity_passthrough():
    x = tf.random.normal([1, 8, 8, 3])
    y = upfirdn2d(x, [1.0], up=1, down=1)
    np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-6)


def test_separable_matches_2d_kernel():
    f = binomial_filter(5)
    x = tf.random.normal([1, 12, 12, 2])
    y1 = upfirdn2d(x, f)                       # 1-D separable path
    f2 = np.outer(f, f)[:, :, None, None].astype(np.float32)
    f2 = np.tile(f2, [1, 1, 2, 1])
    xp = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])
    y2 = tf.nn.depthwise_conv2d(xp, f2, strides=[1, 1, 1, 1], padding="VALID")
    np.testing.assert_allclose(y1.numpy(), y2.numpy(), atol=1e-5)


def test_filtered_lrelu_constant_signal():
    # on a constant field, filtering is identity (DC gain 1), so the result is
    # exactly leaky_relu of the constant + bias
    c = tf.constant([[[[2.0], [-3.0]]]])
    b = tf.constant([0.5, -1.0])
    y = filtered_lrelu(c, fu=None, fd=None, bias=b, slope=0.2)
    expect = tf.nn.leaky_relu(c + b[None, None, None, :], alpha=0.2)
    np.testing.assert_allclose(y.numpy(), expect.numpy(), atol=1e-6)


def test_filtered_lrelu_reduces_aliasing_vs_naive():
    # high-frequency sine pushed through a nonlinearity: the filtered version
    # must carry far less energy above the output Nyquist than naive
    # downsample(relu(x))
    n = 256
    t = np.arange(n, dtype=np.float32)
    s = np.sin(2 * np.pi * 0.45 * t)[None, :, None, None]
    x = s * np.ones([1, 1, n, 4], dtype=np.float32)  # (1, n, n, 4) pattern
    x = tf.constant(x)
    fd = design_kaiser_filter(0.25, 0.05, sampling_rate=1.0)
    y_filtered = filtered_lrelu(x, fu=None, fd=fd, down=2, bias=None)
    y_naive = tf.nn.leaky_relu(x, alpha=0.2)[:, ::2, ::2, :]

    e_filt = float(tf.reduce_mean(tf.square(y_filtered)))
    e_naive = float(tf.reduce_mean(tf.square(y_naive)))
    assert e_filt < e_naive * 0.5


def test_filtered_lrelu_dtype_preserved():
    x = tf.random.normal([1, 8, 8, 2])
    y = filtered_lrelu(x, fu=None, fd=None, bias=tf.zeros([2]))
    assert y.dtype == x.dtype
