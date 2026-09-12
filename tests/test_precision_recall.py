import numpy as np

from src.evaluation.precision_recall import precision_recall


def test_perfect_when_fake_equals_real():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((400, 16)).astype(np.float64)
    p, r = precision_recall(feats, feats.copy(), k=3)
    assert p > 99.0 and r > 99.0


def test_recall_collapses_on_mode_collapse():
    rng = np.random.default_rng(1)
    real = rng.standard_normal((400, 16)).astype(np.float64)
    fake = np.tile(real[:1], (400, 1))          # single mode
    p, r = precision_recall(real, fake, k=3)
    assert r < 10.0         # almost no real modes covered (<10% vs 100% perfect)
    assert p > 50.0         # the one fake mode is a real mode
