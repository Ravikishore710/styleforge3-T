# Configuration schema and YAML loading
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    # experiment
    "experiment": "ffhq",
    "seed": 0,
    "output_dir": "outputs",
    # data
    "data_dir": "data/ffhq",
    "img_resolution": 256,
    "img_channels": 3,
    "mirror": 1,
    "max_images": None,
    # generator (StyleGAN3-T class)
    "z_dim": 512,
    "w_dim": 512,
    "mapping_depth": 2,
    "mapping_lr_mult": 0.01,
    "channels_base": 32768,
    "channels_max": 512,
    "num_critical": 2,
    "first_cutoff_cycles": 2.0,
    "stopband_factor": 1.25,
    "noise_mode": "random",
    # discriminator
    "d_channels_base": 32768,
    "d_channels_max": 512,
    "mbstd_group_size": 4,
    # training
    "batch_size": 32,
    "train_kimg": 1600,
    "tick_kimg": 4,
    "G_lr": 0.0025,
    "D_lr": 0.002,
    "beta1": 0.9,
    "beta2": 0.99,
    "epsilon": 1e-8,
    "gamma": 32.8,
    "r1_interval": 16,
    "ema_kimg": 10,
    "ema_rampup": 0.05,
    "mixed_precision": 1,
    "snap_kimg": 50,
    # evaluation
    "eval_images": 10000,
    "fid_images": 10000,
    "pr_images": 10000,
    "eq_images": 256,
    "eq_batch": 16,
    "truncation_psi": 1.0,
}


def load_config(path: str | None = None, **overrides) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg.update(user)
    cfg.update(overrides)
    cfg["_path"] = str(Path(path).resolve()) if path else None
    return cfg


def save_config(cfg: dict, path: str) -> None:
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
