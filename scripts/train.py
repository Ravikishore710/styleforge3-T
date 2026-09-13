# Train StyleForge3-T-draft from random initialization
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
    ap.add_argument("--resume-dir", default=None, help="Directory or checkpoint path to resume from")
    ap.add_argument("--data-dir", default=None, help="Override data_dir")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    ap.add_argument("--train-kimg", type=int, default=None, help="Override train_kimg")
    ap.add_argument("--tick-kimg", type=int, default=None, help="Override tick_kimg")
    ap.add_argument("--snap-kimg", type=int, default=None, help="Override snap_kimg")
    ap.add_argument("--gamma", type=float, default=None, help="Override R1 gamma regularization weight")
    ap.add_argument("--g-lr", type=float, default=None, help="Override G learning rate")
    ap.add_argument("--d-lr", type=float, default=None, help="Override D learning rate")
    ap.add_argument("--github-repo", default=None, help="Override GitHub repo for backup (owner/repo)")
    ap.add_argument("--drive-dir", default=None, help="Google Drive directory for backup")
    ap.add_argument("--no-backup", action="store_true", help="Disable automatic cloud backups")
    args = ap.parse_args()

    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.train_kimg:
        overrides["train_kimg"] = args.train_kimg
    if args.tick_kimg:
        overrides["tick_kimg"] = args.tick_kimg
    if args.snap_kimg:
        overrides["snap_kimg"] = args.snap_kimg
    if args.gamma is not None:
        overrides["gamma"] = args.gamma
    if args.g_lr is not None:
        overrides["G_lr"] = args.g_lr
    if args.d_lr is not None:
        overrides["D_lr"] = args.d_lr
    if args.resume_dir:
        overrides["resume_dir"] = args.resume_dir
    if args.github_repo:
        overrides["github_repo"] = args.github_repo
    if args.drive_dir:
        overrides["drive_dir"] = args.drive_dir
    if args.no_backup:
        overrides["auto_backup"] = False
    cfg = load_config(args.config, **overrides)
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
    result = trainer.fit(resume=args.resume or bool(args.resume_dir))
    print(result)
    save_config(cfg, str(Path(cfg["output_dir"]) / "config_resolved.yaml"))


if __name__ == "__main__":
    main()
