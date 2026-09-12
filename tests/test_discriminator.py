import numpy as np
import tensorflow as tf

from src.config import load_config
from src.discriminator.discriminator import Discriminator, MinibatchStd


def _cfg(res=64):
    return load_config(None, img_resolution=res)


def test_minibatch_std_appends_channel():
    mb = MinibatchStd(group_size=4)
    x = tf.random.normal([4, 4, 4, 8])
    y = mb(x)
    assert y.shape == (4, 4, 4, 9)


def test_minibatch_std_identical_images():
    # identical images -> zero std map -> same score as a single image
    mb = MinibatchStd(group_size=4)
    one = tf.random.normal([1, 4, 4, 8])
    many = tf.tile(one, [4, 1, 1, 1])
    assert np.allclose(mb(many).numpy()[0, ..., -1], 0.0, atol=1e-6)


def test_discriminator_output_shape_and_gradients():
    cfg = _cfg(64)
    D = Discriminator(cfg)
    x = tf.random.normal([4, 64, 64, 3])
    with tf.GradientTape() as tape:
        logit = D(x, training=True)
        loss = tf.reduce_mean(logit)
    assert logit.shape == (4, 1)
    grads = tape.gradient(loss, D.trainable_variables)
    assert all(g is not None for g in grads)


def test_discriminator_score_depends_on_batch():
    # with non-identical images, minibatch std makes scores batch-dependent
    cfg = _cfg(64)
    D = Discriminator(cfg)
    _ = D(tf.zeros([4, 64, 64, 3]))
    x = tf.random.normal([4, 64, 64, 3])
    solo = D(x[:1], training=False).numpy()[0, 0]
    in_batch = D(x, training=False).numpy()[0, 0]
    assert not np.isclose(solo, in_batch)


def test_discriminator_downsamples_to_4x4():
    cfg = _cfg(64)
    D = Discriminator(cfg)
    _ = D(tf.zeros([1, 64, 64, 3]))
    h = 64
    for _ in D.blocks:
        h //= 2
    assert h == 4
