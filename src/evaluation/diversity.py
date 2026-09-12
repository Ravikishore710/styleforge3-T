"""Latent diversity: normalized pairwise VGG distance statistics."""
from __future__ import annotations

import numpy as np

from .equivariance import _feats


def diversity_score(images: np.ndarray) -> dict:
    f = _feats(images)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    sim = f @ f.T
    n = len(f)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(np.clip(2.0 - 2.0 * sim[iu], 0, None))
    return {"diversity_mean": float(d.mean()), "diversity_std": float(d.std())}
