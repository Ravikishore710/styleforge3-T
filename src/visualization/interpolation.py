"""Interpolation visualizations (rows saved as PNG)."""
from __future__ import annotations

import numpy as np

from ..inference.sampling import generate, save_grid
from ..inference.interpolation import interpolate_z


def interpolation_row(G_ema, cfg: dict, seed_a: int = 0, seed_b: int = 1,
                      steps: int = 8, spherical: bool = True,
                      out_path: str = "outputs/interpolations/row.png"):
    z_dim = int(cfg["z_dim"])
    rng = np.random.default_rng(0)
    za = rng.standard_normal([z_dim], np.float32)
    rng = np.random.default_rng(1)
    zb = rng.standard_normal([z_dim], np.float32)
    zs = interpolate_z(za, zb, steps, spherical=spherical)
    imgs = generate(G_ema, zs, noise_mode="const")
    return save_grid(imgs, out_path, cols=steps)
