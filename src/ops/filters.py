# FIR filter design: Kaiser-windowed low-pass filters and binomial filters
from __future__ import annotations

import numpy as np

ATTENUATION_DB = 40.0
MAX_TAPS = 25


def _kaiser_beta(attenuation_db: float) -> float:
    a = float(attenuation_db)
    if a > 50.0:
        return 0.1102 * (a - 8.7)
    if a >= 21.0:
        return 0.5842 * (a - 21.0) ** 0.4 + 0.07886 * (a - 21.0)
    return 0.0


def _kaiser_window(numtaps: int, beta: float) -> np.ndarray:
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    r = 2.0 * n / (numtaps - 1)
    arg = np.clip(1.0 - r * r, 0.0, None)
    return np.i0(beta * np.sqrt(arg)) / np.i0(beta)


def clamp_cutoff(cutoff: float, half_width: float, sampling_rate: float) -> tuple[float, float]:
    # Keep cutoff + half_width below Nyquist of the given sampling rate
    nyq = sampling_rate / 2.0
    cutoff = min(float(cutoff), nyq - 1e-3)
    half_width = min(float(half_width), max(nyq - cutoff - 1e-3, 1e-4))
    return cutoff, half_width


def design_kaiser_filter(cutoff: float, half_width: float,
                         sampling_rate: float,
                         attenuation_db: float = ATTENUATION_DB,
                         max_taps: int = MAX_TAPS) -> np.ndarray:
    # Symmetric Kaiser low-pass FIR, DC gain 1. Identity if 1 tap suffices
    cutoff, half_width = clamp_cutoff(cutoff, half_width, sampling_rate)
    width = max(2.0 * half_width / sampling_rate, 1e-6)
    # numtaps estimate from Kaiser formulas (A in dB, dw normalized to fs).
    dw = 2.0 * np.pi * width
    numtaps = int(np.ceil((attenuation_db - 8.0) / (2.285 * dw))) + 1
    numtaps = min(numtaps + (numtaps % 2 == 0), max_taps)
    if numtaps % 2 == 0:
        numtaps += 1
    if numtaps <= 1:
        return np.ones(1, dtype=np.float32)
    n = np.arange(numtaps, dtype=np.float64) - (numtaps - 1) / 2.0
    fc = cutoff / sampling_rate
    h = np.sinc(2.0 * fc * n) * 2.0 * fc
    h *= _kaiser_window(numtaps, _kaiser_beta(attenuation_db))
    h /= h.sum()
    return h.astype(np.float32)


def lowpass_kaiser(cutoff: float, half_width: float, sampling_rate: float) -> np.ndarray:
    # Alias of design_kaiser_filter with AliasForge defaults
    return design_kaiser_filter(cutoff, half_width, sampling_rate)


def binomial_filter(taps: int = 5) -> np.ndarray:
    # Binomial low-pass (taps=5 -> [1,4,6,4,1]/16), DC gain 1
    from math import comb
    f = np.array([comb(taps - 1, k) for k in range(taps)], dtype=np.float64)
    return (f / f.sum()).astype(np.float32)
