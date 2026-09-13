"""Inference: seeded samples, truncation grids, interpolations.

Usage:
  python scripts/generate.py --config configs/ffhq_256.yaml --num-images 16 --seed 42
  python scripts/generate.py --config configs/ffhq_256.yaml --truncation-grid
  python scripts/generate.py --config configs/ffhq_256.yaml --interpolate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from src.config import load_config
from src.generator.generator import Generator
from src.inference.sampling import generate, save_grid, save_images, seeded_z
from src.inference.truncation import cache_w_mean
from src.inference.interpolation import interpolate_z
from src.training.checkpoint import CheckpointManager


def load_G_ema(cfg):
    tf.keras.mixed_precision.set_global_policy("float32")
    G_ema = Generator(cfg)
    _ = G_ema(tf.zeros([1, int(cfg["z_dim"])]))
    from src.training.ema import EMA
    ema = EMA(G_ema)
    ckpt = CheckpointManager(cfg["output_dir"], G_ema, None, None, None, None,
                             tf.Variable(0, dtype=tf.int64),
                             tf.Variable(0, dtype=tf.int64),
                             ema_state=ema)
    ckpt.restore()
    ckpt.restore_ema_into(G_ema)
    return G_ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=16)
    ap.add_argument("--truncation", type=float, default=None)
    ap.add_argument("--noise-mode", default="const",
                    choices=["const", "random", "none"])
    ap.add_argument("--truncation-grid", action="store_true")
    ap.add_argument("--interpolate", action="store_true")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.output_dir or str(Path(cfg["output_dir"]) / "samples")
    G_ema = load_G_ema(cfg)
    psi = args.truncation if args.truncation is not None else cfg["truncation_psi"]
    w_mean = cache_w_mean(G_ema, cfg) if psi != 1.0 else None

    if args.truncation_grid:
        z = seeded_z(args.num_images, int(cfg["z_dim"]), args.seed)
        rows = [generate(G_ema, z, noise_mode="const",
                         truncation_psi=p, w_mean=w_mean)
                for p in (1.0, 0.7, 0.5, 0.3)]
        save_grid(np.concatenate(rows),
                  str(Path(out) / "truncation_grid.png"),
                  cols=args.num_images)
        print("saved truncation grid")
        return
    if args.interpolate:
        za = seeded_z(1, int(cfg["z_dim"]), args.seed)[0]
        zb = seeded_z(1, int(cfg["z_dim"]), args.seed + 1)[0]
        zs = interpolate_z(za, zb, args.steps, spherical=True)
        imgs = generate(G_ema, zs, noise_mode=args.noise_mode,
                        truncation_psi=psi, w_mean=w_mean)
        print(save_grid(imgs, str(Path(out) / "interpolation.png"),
                        cols=args.steps))
        return

    z = seeded_z(args.num_images, int(cfg["z_dim"]), args.seed)
    imgs = generate(G_ema, z, noise_mode=args.noise_mode,
                    truncation_psi=psi, w_mean=w_mean)
    print(save_grid(imgs, str(Path(out) / f"seeded_{args.seed}.png"),
                    cols=8))
    print(f"saved {len(save_images(imgs, out, 'sample'))} images to {out}")


if __name__ == "__main__":
    main()
