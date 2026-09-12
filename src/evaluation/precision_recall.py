"""Improved precision & recall for generative models (Kynkaanniemi et al. 2019)."""
from __future__ import annotations

import numpy as np


def _radii_and_hits(feats_other: np.ndarray, feats_own: np.ndarray,
                    k: int = 3, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """radii: distance to k-th NN within own set; hits: count of other-set
    points inside each own-set sphere."""
    n = len(feats_own)
    radii = np.zeros(n, np.float64)
    hits = np.zeros(n, np.int64)
    for i in range(0, n, chunk):
        d = np.linalg.norm(feats_own[i:i + chunk, None] - feats_own[None], axis=-1)
        part = np.partition(d, k, axis=1)
        radii[i:i + chunk] = part[:, k]
        d2 = np.linalg.norm(feats_own[i:i + chunk, None] - feats_other[None], axis=-1)
        hits[i:i + chunk] = (d2 <= radii[i:i + chunk, None]).sum(axis=1)
    return radii, hits


def precision_recall(real_feats: np.ndarray, fake_feats: np.ndarray,
                     k: int = 3) -> tuple[float, float]:
    half_r = len(real_feats) // 2
    half_f = len(fake_feats) // 2
    _, hits_f = _radii_and_hits(real_feats[:half_r], fake_feats[:half_f], k)
    precision = float(np.mean(hits_f > 0)) * 100.0
    _, hits_r = _radii_and_hits(fake_feats[:half_f], real_feats[:half_r], k)
    recall = float(np.mean(hits_r > 0)) * 100.0
    return precision, recall
