import numpy as np

from src.ops.filters import (binomial_filter, clamp_cutoff, design_kaiser_filter)


def test_binomial_dc_gain_and_symmetry():
    f = binomial_filter(5)
    assert np.isclose(f.sum(), 1.0)
    assert np.allclose(f, f[::-1])
    assert np.allclose(f, np.array([1, 4, 6, 4, 1]) / 16)


def test_kaiser_dc_gain_is_one():
    for cutoff, hw in [(0.2, 0.05), (0.4, 0.05), (0.1, 0.02)]:
        f = design_kaiser_filter(cutoff, hw, sampling_rate=1.0)
        assert abs(f.sum() - 1.0) < 1e-5
        assert len(f) % 2 == 1


def test_kaiser_attenuates_high_frequencies():
    fs = 1.0
    f = design_kaiser_filter(0.15, 0.05, sampling_rate=fs)
    n = 4096
    t = np.arange(n)
    low = np.sin(2 * np.pi * 0.05 * t)      # inside passband
    high = np.sin(2 * np.pi * 0.45 * t)     # deep in stopband
    def resp(x):
        return np.convolve(x, f, mode="same")
    rl = np.sqrt(np.mean(resp(low) ** 2)) / np.sqrt(np.mean(low ** 2))
    rh = np.sqrt(np.mean(resp(high) ** 2)) / np.sqrt(np.mean(high ** 2))
    assert rl > 0.9
    assert rh < 0.1


def test_clamp_cutoff_respects_nyquist():
    c, hw = clamp_cutoff(0.6, 0.2, sampling_rate=1.0)   # nyquist 0.5
    assert c + hw <= 0.5
