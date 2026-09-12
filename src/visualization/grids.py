"""Sample grids from the EMA generator."""
from __future__ import annotations

import numpy as np

from ..inference.sampling import generate, save_grid, seeded_z


def sample_grid(G_ema, cfg: dict, n: int = 64, seed: int = 0,
                truncation_psi: float = 1.0, w_mean=None,
                out_path: str = "outputs/samples/grid.png"):
    z = seeded_z(n, int(cfg["z_dim"]), seed)
    imgs = generate(G_ema, z, noise_mode="const",
                    truncation_psi=truncation_psi, w_mean=w_mean)
    return save_grid(imgs, out_path)
