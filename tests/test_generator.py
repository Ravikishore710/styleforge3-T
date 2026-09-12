import numpy as np
import tensorflow as tf

from src.config import load_config
from src.generator.generator import Generator
from src.generator.mapping import MappingNetwork
from src.generator.synthesis import SynthesisNetwork, channels_at


def _cfg(res=64, **kw):
    return load_config(None, img_resolution=res, **kw)


def test_mapping_shapes_and_tiling():
    m = MappingNetwork(num_ws=5)
    ws = m(tf.random.normal([4, 512]))
    assert ws.shape == (4, 5, 512)


def test_mapping_z_normalization():
    m = MappingNetwork(num_ws=1)
    z = tf.random.normal([16, 512]) * 10.0
    _ = m(z)  # must not overflow; hypersphere scaling caps RMS at sqrt(512)


def test_channels_schedule():
    assert channels_at(4) == 512
    assert channels_at(1024) == 32


def test_generator_forward_shapes_and_range():
    cfg = _cfg(64)
    G = Generator(cfg)
    img = G(tf.random.normal([2, 512]), training=False)
    assert img.shape == (2, 64, 64, 3)
    assert tf.math.is_finite(img).numpy().all()
    assert float(tf.reduce_max(tf.abs(img))) < 1e4  # sane initial magnitude


def test_generator_ws_count_matches():
    cfg = _cfg(64)
    G = Generator(cfg)
    expected = G.synthesis.num_ws
    _, ws = G(tf.random.normal([1, 512]), return_ws=True)
    assert ws.shape[1] == expected


def test_generator_seed_determinism():
    cfg = _cfg(64, noise_mode="const")
    G = Generator(cfg)
    _ = G(tf.zeros([1, 512]))  # build
    z = np.random.default_rng(0).standard_normal([2, 512]).astype(np.float32)
    a = G(tf.constant(z), noise_mode="const", training=False).numpy()
    b = G(tf.constant(z), noise_mode="const", training=False).numpy()
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_generator_batch_independence():
    cfg = _cfg(64, noise_mode="const")
    G = Generator(cfg)
    _ = G(tf.zeros([1, 512]))
    z = np.random.default_rng(1).standard_normal([4, 512]).astype(np.float32)
    full = G(tf.constant(z), noise_mode="const", training=False).numpy()
    single = G(tf.constant(z[:1]), noise_mode="const", training=False).numpy()
    np.testing.assert_allclose(full[:1], single, atol=1e-5)


def test_generator_noise_modes():
    cfg = _cfg(64)
    G = Generator(cfg)
    z = tf.zeros([2, 512])
    for mode in ("random", "const", "none"):
        img = G(z, noise_mode=mode, training=False)
        assert img.shape == (2, 64, 64, 3)


def test_generator_shift_moves_content():
    tf.random.set_seed(42)
    cfg = _cfg(64, noise_mode="const")
    G = Generator(cfg)
    _ = G(tf.zeros([1, 512]))
    z = np.random.default_rng(3).standard_normal([1, 512]).astype(np.float32)
    base = G(tf.constant(z), noise_mode="const", training=False).numpy()
    shifted = G(tf.constant(z), noise_mode="const", shift=(8.0, 0.0),
                training=False).numpy()
    rolled = np.roll(base, 8, axis=1)
    corr_base = np.abs(base - rolled).mean()
    corr_shift = np.abs(shifted - rolled).mean()
    assert corr_shift < corr_base   # grid-shifted gen matches rolled image better


def test_gradient_flows_through_synthesis():
    cfg = _cfg(64)
    G = Generator(cfg)
    with tf.GradientTape() as tape:
        img = G(tf.random.normal([2, 512]), training=True)
        loss = tf.reduce_mean(tf.square(img))
    grads = tape.gradient(loss, G.trainable_variables)
    assert all(g is not None for g in grads)
    assert all(tf.reduce_all(tf.math.is_finite(g)).numpy() for g in grads)
