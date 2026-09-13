# FFHQ dataset pipeline: scan -> decode -> validate -> resize -> [-1,1]
from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def scan_images(data_dir: str) -> list[str]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"data dir {data_dir} not found. Run scripts/prepare_ffhq.py first.")
    files = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"no images found under {data_dir}")
    return files


def _load(path, res: int):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)          # [0,1]
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    img = tf.cond(tf.logical_or(h != res, w != res),
                  lambda: tf.image.resize(img, [res, res], antialias=True),
                  lambda: img)
    return img * 2.0 - 1.0                                       # [-1,1]


def build_dataset(cfg: dict, batch_size: int | None = None,
                  mirror: bool | None = None, shuffle: bool = True):
    files = scan_images(cfg["data_dir"])
    if cfg.get("max_images"):
        files = files[: int(cfg["max_images"])]
    res = int(cfg["img_resolution"])
    bs = batch_size or int(cfg["batch_size"])
    do_mirror = cfg["mirror"] if mirror is None else mirror

    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(len(files), reshuffle_each_iteration=True)
    ds = ds.map(lambda p: _load(p, res), num_parallel_calls=tf.data.AUTOTUNE)
    if do_mirror:
        ds = ds.map(lambda x: tf.image.random_flip_left_right(x),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE), len(files)


def write_manifest(cfg: dict, path: str) -> dict:
    files = scan_images(cfg["data_dir"])
    info = {"data_dir": cfg["data_dir"], "num_images": len(files),
            "resolution": int(cfg["img_resolution"])}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info
