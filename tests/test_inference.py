import numpy as np
import tensorflow as tf

from src.config import load_config
from src.generator.generator import Generator
from src.inference.interpolation import interpolate_z, slerp
from src.inference.truncation import apply_truncation
from src.inference.sampling import generate, seeded_z


def _G():
    cfg = load_config(None, img_resolution=64, noise_mode="const")
    G = Generator(cfg)
    _ = G(tf.zeros([1, 512]))
    return G, cfg


def test_seeded_z_deterministic():
    a = seeded_z(4, 512, seed=42)
    b = seeded_z(4, 512, seed=42)
    np.testing.assert_array_equal(a, b)
    c = seeded_z(4, 512, seed=43)
    assert not np.allclose(a, c)


def test_generate_seeded_reproducible():
    G, cfg = _G()
    z = seeded_z(2, 512, 7)
    a = generate(G, z, noise_mode="const")
    b = generate(G, z, noise_mode="const")
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert a.shape == (2, 64, 64, 3)


def test_truncation_pulls_toward_mean():
    rng = np.random.default_rng(0)
    ws = rng.standard_normal([100, 10, 512]).astype(np.float32)
    w_mean = ws.mean(axis=0, keepdims=True)
    out = apply_truncation(ws, w_mean, psi=0.0)
    np.testing.assert_allclose(out, np.broadcast_to(w_mean, ws.shape), atol=1e-6)
    out1 = apply_truncation(ws, w_mean, psi=1.0)
    np.testing.assert_allclose(out1, ws, atol=1e-6)
    out5 = apply_truncation(ws, w_mean, psi=0.5)
    assert np.abs(out5 - w_mean).mean() < np.abs(ws - w_mean).mean()


def test_generate_truncation_changes_images():
    G, cfg = _G()
    _ = G(tf.zeros([1, 512]))
    G.synthesis.layers[0].affine.kernel.assign(
        tf.random.normal(G.synthesis.layers[0].affine.kernel.shape, stddev=0.1))
    z = seeded_z(2, 512, 3)
    w_mean = np.zeros([1, 1, cfg["w_dim"]], np.float32)
    a = generate(G, z, noise_mode="const", truncation_psi=1.0, w_mean=w_mean)
    b = generate(G, z, noise_mode="const", truncation_psi=0.3, w_mean=w_mean)
    assert not np.allclose(a, b)


def test_slerp_endpoints_and_midpoint():
    z1 = np.array([1.0, 0.0]); z2 = np.array([0.0, 1.0])
    np.testing.assert_allclose(slerp(0.0, z1, z2), z1, atol=1e-6)
    np.testing.assert_allclose(slerp(1.0, z1, z2), z2, atol=1e-6)
    mid = slerp(0.5, z1, z2)
    assert np.isclose(np.linalg.norm(mid), 1.0, atol=1e-6)


def test_interpolate_z_shapes():
    z1 = np.zeros(512, np.float32); z2 = np.ones(512, np.float32)
    out = interpolate_z(z1, z2, steps=8, spherical=False)
    assert out.shape == (8, 512)
    np.testing.assert_allclose(out[0], z1)
    np.testing.assert_allclose(out[-1], z2)
