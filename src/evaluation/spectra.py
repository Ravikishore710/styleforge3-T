"""Radial frequency spectra: connects generated images to alias-free claims."""
from __future__ import annotations

import numpy as np


def radial_spectrum(images: np.ndarray) -> np.ndarray:
    """Mean radially-averaged power spectrum, (res//2,) float64."""
    x = np.mean(images.astype(np.float64), axis=-1)
    f = np.fft.fftshift(np.fft.fft2(x - x.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))
    p = np.abs(f) ** 2
    res = x.shape[1]
    yy, xx = np.mgrid[0:res, 0:res]
    r = np.sqrt((yy - res // 2) ** 2 + (xx - res // 2) ** 2).astype(int)
    r = np.clip(r, 0, res // 2 - 1)
    out = np.zeros(res // 2)
    count = np.zeros(res // 2)
    for i in range(res // 2):
        mask = r == i
        out[i] = p[:, mask].mean()
        count[i] = 1
    return out


def spectral_stats(real_images: np.ndarray, fake_images: np.ndarray) -> dict:
    pr = radial_spectrum(real_images)
    pf = radial_spectrum(fake_images)
    lf, rf = np.log(pr + 1e-12), np.log(pf + 1e-12)
    slope_r = np.polyfit(np.arange(len(lf)), lf, 1)[0]
    slope_f = np.polyfit(np.arange(len(rf)), rf, 1)[0]
    return {"real_log_slope": float(slope_r), "fake_log_slope": float(slope_f),
            "spectral_l2": float(np.linalg.norm(pr / pr.sum() - pf / pf.sum()))}
