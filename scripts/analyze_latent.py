"""Latent-space analysis: truncation sweep + interpolation sweeps."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.generator.generator import Generator
from src.inference.sampling import generate, save_grid, seeded_z
from src.inference.truncation import cache_w_mean, apply_truncation
from src.training.checkpoint import CheckpointManager
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tf.keras.mixed_precision.set_global_policy("float32")
    G_ema = Generator(cfg)
    _ = G_ema(tf.zeros([1, int(cfg["z_dim"])]))
    ckpt = CheckpointManager(cfg["output_dir"], G_ema, G_ema, G_ema, None, None,
                             tf.Variable(0, dtype=tf.int64),
                             tf.Variable(0, dtype=tf.int64))
    ckpt.restore(); ckpt.restore_ema_into(G_ema)
    w_mean = cache_w_mean(G_ema, cfg)
    out = Path(cfg["output_dir"]) / "interpolations"

    # truncation sweep
    z = seeded_z(16, int(cfg["z_dim"]), args.seed)
    rows = [generate(G_ema, z, noise_mode="const", truncation_psi=p, w_mean=w_mean)
            for p in (1.0, 0.8, 0.6, 0.4, 0.2)]
    print(save_grid(np.concatenate(rows), str(out / "psi_sweep.png"), cols=16))

    # interpolation sweeps
    for k in range(args.pairs):
        za = seeded_z(1, int(cfg["z_dim"]), 1000 + k)[0]
        zb = seeded_z(1, int(cfg["z_dim"]), 2000 + k)[0]
        ts = np.linspace(0, 1, args.steps, dtype=np.float32)
        zs = np.stack([(1 - t) * za + t * zb for t in ts])
        imgs = generate(G_ema, zs, noise_mode="const")
        print(save_grid(imgs, str(out / f"interp_{k}.png"), cols=args.steps))


if __name__ == "__main__":
    main()
