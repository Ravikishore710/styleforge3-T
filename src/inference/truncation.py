# Truncation trick: w' = w_mean + psi * (w - w_mean)
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def estimate_w_mean(G_ema, z_dim: int, num: int = 10000,
                    batch: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(0, num, batch):
        z = rng.standard_normal([batch, z_dim], dtype=np.float32)
        ws = G_ema.mapping(tf.constant(z), training=False)
        means.append(tf.reduce_mean(ws[:, 0], axis=0).numpy())
    return np.mean(means, axis=0).astype(np.float32)


def cache_w_mean(G_ema, cfg: dict, path: str | None = None,
                 num: int = 10000) -> np.ndarray:
    path = path or str(Path(cfg["output_dir"]) / "w_mean.npy")
    if Path(path).exists():
        return np.load(path)
    w_mean = estimate_w_mean(G_ema, cfg["z_dim"], num=num,
                             seed=int(cfg["seed"]))
    np.save(path, w_mean)
    return w_mean


def apply_truncation(ws: np.ndarray, w_mean: np.ndarray,
                     psi: float) -> np.ndarray:
    return w_mean + psi * (ws - w_mean)
