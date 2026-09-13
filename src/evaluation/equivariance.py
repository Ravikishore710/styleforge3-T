# StyleGAN3-style equivariance metrics (translation int/frac, rotation)
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy import ndimage

_vgg = None


def _vgg():
    global _vgg
    if _vgg is None:
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet",
                                           input_shape=(None, None, 3))
        _vgg = tf.keras.Model(base.input, base.get_layer("block3_pool").output)
    return _vgg


def _feats(images: np.ndarray) -> np.ndarray:
    x = tf.cast(images, tf.float32)
    x = (x + 1.0) / 2.0 * 255.0
    return _vgg()(x, training=False).numpy().reshape(len(images), -1)


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa, fb = _feats(a), _feats(b)
    return np.linalg.norm(fa - fb, axis=1) / np.linalg.norm(fa, axis=1)


def equivariance_metrics(G_ema, cfg: dict, num: int = 64, batch: int = 16,
                         seed: int = 0, rotate_deg: float = 5.0) -> dict:
    rng = np.random.default_rng(seed)
    z_dim = int(cfg["z_dim"])
    eqt_int, eqt_frac, eqr = [], [], []
    for _ in range(0, num, batch):
        z = rng.standard_normal([batch, z_dim], dtype=np.float32)
        base = G_ema(tf.constant(z), noise_mode="const", training=False).numpy()
        # integer translation
        k = int(rng.integers(1, 4))
        shifted_img = np.roll(base, (k, k), axis=(1, 2))
        shifted_gen = G_ema(tf.constant(z), noise_mode="const",
                            shift=(float(k), float(k)),
                            training=False).numpy()
        eqt_int.append(_dist(shifted_img, shifted_gen))
        # fractional translation
        frac = float(rng.uniform(0.25, 0.75))
        rolled = _frac_roll(base, frac)
        gen = G_ema(tf.constant(z), noise_mode="const",
                    shift=(frac, frac), training=False).numpy()
        eqt_frac.append(_dist(rolled, gen))
        # rotation
        rot = np.stack([ndimage.rotate(im, rotate_deg, reshape=False, order=1,
                                       mode="nearest") for im in base])
        eqr.append(_dist(base, rot))
    return {"eqt50k_int": float(np.mean(np.concatenate(eqt_int))),
            "eqt50k_frac": float(np.mean(np.concatenate(eqt_frac))),
            "eqr50k": float(np.mean(np.concatenate(eqr)))}


def _frac_roll(x: np.ndarray, shift: float) -> np.ndarray:
    lo = int(np.floor(shift))
    frac = shift - lo
    a = np.roll(x, (lo, lo), axis=(1, 2))
    b = np.roll(x, (lo + 1, lo + 1), axis=(1, 2))
    return a * (1 - frac) + b * frac
