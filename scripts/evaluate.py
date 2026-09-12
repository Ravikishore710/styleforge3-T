"""Full evaluation: FID, precision/recall, equivariance, spectra, diversity.

Usage:
  python scripts/evaluate.py --config configs/ffhq_256.yaml --fid-images 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from src.config import load_config
from src.data.ffhq import build_dataset
from src.evaluation.diversity import diversity_score
from src.evaluation.equivariance import equivariance_metrics
from src.evaluation.fid import compute_fid, generate_np, inception_features
from src.evaluation.precision_recall import precision_recall
from src.evaluation.spectra import spectral_stats
from src.generator.generator import Generator
from src.inference.sampling import seeded_z
from src.training.checkpoint import CheckpointManager
from src.visualization.spectra import plot_spectra


def load_G_ema(cfg):
    tf.keras.mixed_precision.set_global_policy("float32")
    G_ema = Generator(cfg)
    _ = G_ema(tf.zeros([1, int(cfg["z_dim"])]))
    ckpt = CheckpointManager(cfg["output_dir"], G_ema, G_ema, G_ema, None, None,
                             tf.Variable(0, dtype=tf.int64),
                             tf.Variable(0, dtype=tf.int64))
    ckpt.restore()
    ckpt.restore_ema_into(G_ema)
    return G_ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fid-images", type=int, default=None)
    ap.add_argument("--pr-images", type=int, default=None)
    ap.add_argument("--skip-eq", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    G_ema = load_G_ema(cfg)
    out_dir = Path(cfg["output_dir"]) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_fid = args.fid_images or int(cfg["fid_images"])
    ds, num = build_dataset(cfg, batch_size=50, shuffle=False)
    it = iter(ds.repeat())

    fid = compute_fid(G_ema, it, cfg, num=n_fid, seed=args.seed)
    print(f"[fid] {fid:.3f} ({n_fid} images)")

    n_pr = args.pr_images or int(cfg["pr_images"])
    rng = np.random.default_rng(args.seed)
    z = rng.standard_normal([n_pr, int(cfg["z_dim"])], dtype=np.float32)
    fakes = generate_np(G_ema, z, cfg)
    reals = []
    while sum(len(r) for r in reals) < n_pr:
        reals.append(next(it).numpy())
    reals = np.concatenate(reals, 0)[:n_pr]
    p, r = precision_recall(inception_features(reals), inception_features(fakes))
    print(f"[precision] {p:.2f}  [recall] {r:.2f}")

    div = diversity_score(fakes[:64])
    spec = spectral_stats(reals[:200], fakes[:200])
    plot_spectra(reals[:200], fakes[:200],
                 str(Path(cfg["output_dir"]) / "spectra" / "spectra.png"))

    results = {"fid": fid, "precision": p, "recall": r, **div, **spec,
               "num_images": num}
    if not args.skip_eq:
        results.update(equivariance_metrics(G_ema, cfg, num=int(cfg["eq_images"]),
                                            seed=args.seed))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
