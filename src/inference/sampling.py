"""Latent sampling engine: z ~ N(0,I), seeded determinism, image export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def random_z(batch: int, z_dim: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal([batch, z_dim], dtype=np.float32)


def seeded_z(batch: int, z_dim: int, seed: int) -> np.ndarray:
    return random_z(batch, z_dim, seed)


@tf.function
def _generate(G_ema, z, w_mean, psi, noise_mode):
    ws = G_ema.mapping(z)
    w_mean = tf.cast(w_mean, ws.dtype)
    psi = tf.cast(psi, ws.dtype)
    ws = w_mean + psi * (ws - w_mean)
    return G_ema.synthesis(ws, noise_mode=noise_mode)


def generate(G_ema, z: np.ndarray, noise_mode: str = "const",
             truncation_psi: float = 1.0, w_mean: np.ndarray | None = None):
    """Generate images from latents with optional truncation."""
    z = tf.constant(np.asarray(z, np.float32))
    if truncation_psi != 1.0:
        if w_mean is None:
            from .truncation import cache_w_mean, estimate_w_mean
            cfg = getattr(G_ema, "cfg", None)
            if cfg is not None and "output_dir" in cfg:
                w_mean = cache_w_mean(G_ema, cfg, num=2000)
            else:
                z_dim = int(z.shape[-1])
                w_mean = estimate_w_mean(G_ema, z_dim, num=2000)
        w_mean = tf.constant(np.asarray(w_mean, np.float32))
    else:
        w_dim = int(G_ema.cfg["w_dim"]) if hasattr(G_ema, "cfg") and "w_dim" in G_ema.cfg else 512
        w_mean = tf.zeros([w_dim], tf.float32)
    psi = tf.constant(float(truncation_psi), tf.float32)
    return _generate(G_ema, z, w_mean, psi, noise_mode).numpy()


def generate_batch(G_ema, batch: int, z_dim: int, seed: int,
                   **kwargs) -> tuple[np.ndarray, np.ndarray]:
    z = seeded_z(batch, z_dim, seed)
    return generate(G_ema, z, **kwargs), z


def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip((img + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)
    return img


def save_images(images: np.ndarray, out_dir: str, prefix: str = "img") -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = str(Path(out_dir) / f"{prefix}_{i:05d}.png")
        Image.fromarray(_to_uint8(img)).save(p)
        paths.append(p)
    return paths


def save_grid(images: np.ndarray, out_path: str, cols: int = 8) -> str:
    images = _to_uint8(images)
    n, h, w, _ = images.shape
    rows = int(np.ceil(n / cols))
    grid = np.zeros([rows * h, cols * w, 3], np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    return out_path
