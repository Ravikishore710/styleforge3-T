"""Latent interpolation: linear in w and spherical in z."""
from __future__ import annotations

import numpy as np


def slerp(t: float, z1: np.ndarray, z2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    z1n = z1 / (np.linalg.norm(z1) + eps)
    z2n = z2 / (np.linalg.norm(z2) + eps)
    omega = np.arccos(np.clip(np.sum(z1n * z2n), -1.0, 1.0))
    if omega < eps:
        return (1 - t) * z1 + t * z2
    so = np.sin(omega)
    return (np.sin((1 - t) * omega) / so) * z1 + (np.sin(t * omega) / so) * z2


def interpolate_z(z1: np.ndarray, z2: np.ndarray, steps: int,
                  spherical: bool = True) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    if spherical:
        return np.stack([slerp(float(t), z1, z2) for t in ts])
    return np.stack([(1 - t) * z1 + t * z2 for t in ts])


def interpolate_ws(ws1: np.ndarray, ws2: np.ndarray,
                   steps: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None, None]
    return (1 - ts) * ws1[None] + ts * ws2[None]
