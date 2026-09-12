"""Train AliasForge from random initialization.

Usage:
  python scripts/train.py --config configs/ffhq_64.yaml
  python scripts/train.py --config configs/ffhq_256.yaml --resume
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

from src.config import load_config, save_config
from src.data.ffhq import build_dataset
from src.discriminator.discriminator import Discriminator
from src.generator.generator import Generator
from src.training.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tf.random.set_seed(int(cfg["seed"]))
    if int(cfg["mixed_precision"]):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    dataset, num_images = build_dataset(cfg)
    print(f"[data] {num_images} images at {cfg['img_resolution']}px")

    G = Generator(cfg)
    D = Discriminator(cfg)
    G_ema = Generator(cfg)
    z = tf.zeros([2, int(cfg["z_dim"])])
    _ = G(z); _ = G_ema(z); _ = D(tf.zeros([2, cfg["img_resolution"],
                                            cfg["img_resolution"], 3]))
    for v_s, v_t in zip(G.trainable_variables, G_ema.trainable_variables):
        v_t.assign(v_s)
    print(f"[model] G params: {G.count_params():,} | D params: {D.count_params():,}")

    trainer = Trainer(cfg, G, D, G_ema, dataset)
    result = trainer.fit(resume=args.resume)
    print(result)
    save_config(cfg, str(Path(cfg["output_dir"]) / "config_resolved.yaml"))


if __name__ == "__main__":
    main()
