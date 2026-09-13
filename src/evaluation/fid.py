# Fréchet Inception Distance with InceptionV3 features (scipy sqrtm)
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy import linalg

_inception = None


def _inception_model():
    global _inception
    if _inception is None:
        _inception = tf.keras.applications.InceptionV3(
            include_top=False, pooling="avg", weights="imagenet")
    return _inception


def inception_features(images: np.ndarray, batch: int = 50) -> np.ndarray:
    # images in [-1,1], any square HxW; resized to 299 internally
    model = _inception_model()
    feats = []
    for i in range(0, len(images), batch):
        x = images[i:i + batch].astype(np.float32)
        x = tf.image.resize(x, [299, 299], method="bilinear", antialias=True)
        x = tf.keras.applications.inception_v3.preprocess_input((x + 1.0) * 127.5)
        f = model(x, training=False)
        feats.append(tf.cast(f, tf.float64).numpy())
    return np.concatenate(feats, axis=0)


def _stats(feats: np.ndarray):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def fid_from_features(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    mu1, s1 = _stats(real_feats)
    mu2, s2 = _stats(fake_feats)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


def compute_fid(G_ema, dataset_iter, cfg: dict, num: int | None = None,
                batch: int = 50, seed: int = 0) -> float:
    n = num or int(cfg["fid_images"])
    z_dim = int(cfg["z_dim"])
    rng = np.random.default_rng(seed)
    fakes, reals = [], []
    done = 0
    while done < n:
        b = min(batch, n - done)
        z = rng.standard_normal([b, z_dim], dtype=np.float32)
        fakes.append(generate_np(G_ema, z, cfg))
        reals.append(next(dataset_iter).numpy()[:b])
        done += b
    real_feats = inception_features(np.concatenate(reals, 0))
    fake_feats = inception_features(np.concatenate(fakes, 0))
    return fid_from_features(real_feats, fake_feats)


def generate_np(G_ema, z: np.ndarray, cfg: dict, batch: int = 25) -> np.ndarray:
    out = []
    for i in range(0, len(z), batch):
        out.append(G_ema(tf.constant(z[i:i + batch].astype(np.float32)),
                         noise_mode="const", training=False).numpy())
    return np.concatenate(out, 0)
