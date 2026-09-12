import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.config import load_config
from src.data.ffhq import build_dataset
from src.evaluation.spectra import radial_spectrum, spectral_stats


@pytest.fixture()
def tiny_image_folder(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(12):
        arr = rng.integers(0, 255, (48, 48, 3), np.uint8)
        Image.fromarray(arr).save(d / f"{i:03d}.png")
    return str(d)


def test_dataset_shapes_range_and_repeatability(tiny_image_folder):
    cfg = load_config(None, img_resolution=32, data_dir=tiny_image_folder,
                      batch_size=4, mirror=0)
    ds, n = build_dataset(cfg, shuffle=False)
    batch = next(iter(ds))
    assert n == 12
    assert batch.shape == (4, 32, 32, 3)
    assert float(tf.reduce_min(batch)) >= -1.0
    assert float(tf.reduce_max(batch)) <= 1.0
    first = next(iter(build_dataset(cfg, shuffle=False)[0]))[0]
    assert np.array_equal(first, batch[0])   # deterministic order


def test_dataset_mirror_augmentation(tiny_image_folder):
    cfg0 = load_config(None, img_resolution=32, data_dir=tiny_image_folder,
                       batch_size=4, mirror=0)
    cfg1 = load_config(None, img_resolution=32, data_dir=tiny_image_folder,
                       batch_size=4, mirror=1)
    ds0, _ = build_dataset(cfg0, shuffle=False)
    ds1, _ = build_dataset(cfg1, shuffle=False)
    seen0 = np.concatenate([b.numpy() for b in ds0], 0)
    seen1 = np.concatenate([b.numpy() for b in ds1], 0)
    for s0, s1 in zip(seen0, seen1):
        is_same = np.allclose(s0, s1, atol=1e-5)
        is_flipped = np.allclose(s0[:, ::-1, :], s1, atol=1e-5)
        assert is_same or is_flipped


def test_radial_spectrum_shape_and_power():
    imgs = np.random.default_rng(0).standard_normal((4, 32, 32, 3)).astype(np.float32)
    s = radial_spectrum(imgs)
    assert s.shape == (16,)
    assert np.all(s >= 0)


def test_spectral_stats_compare_identical():
    rng = np.random.default_rng(0)
    imgs = rng.standard_normal((8, 32, 32, 3)).astype(np.float32)
    stats = spectral_stats(imgs, imgs.copy())
    assert stats["spectral_l2"] < 1e-6
