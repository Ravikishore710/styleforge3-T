import shutil
import zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.config import load_config
from src.generator.generator import Generator
from src.training.backup import (create_checkpoint_zip, get_github_token,
                                 get_repo_slug, trigger_backup)
from src.training.checkpoint import CheckpointManager
from src.training.ema import EMA


def _G():
    cfg = load_config(None, img_resolution=64)
    G = Generator(cfg)
    _ = G(tf.zeros([1, 512]))
    return G


def test_create_checkpoint_zip(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint").write_text('model_checkpoint_path: "ckpt-1"\n')
    (ckpt_dir / "ckpt-1.index").write_text("index_data")
    (ckpt_dir / "ckpt-1.data-00000-of-00001").write_text("tensor_data")

    zip_path = tmp_path / "backup.zip"
    out = create_checkpoint_zip(ckpt_dir, zip_path, prefix="ckpt-1")
    assert out.exists()
    with zipfile.ZipFile(out, "r") as zf:
        names = zf.namelist()
        assert "checkpoint" in names
        assert "ckpt-1.index" in names
        assert "ckpt-1.data-00000-of-00001" in names


def test_checkpoint_restore_from_dir(tmp_path):
    # Create checkpoint in source dir
    src_dir = tmp_path / "previous_notebook_output"
    G = _G()
    D = tf.keras.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(1)])
    G_ema = _G()
    nimg = tf.Variable(66000, dtype=tf.int64)
    step = tf.Variable(4125, dtype=tf.int64)
    g_opt = tf.keras.optimizers.Adam(0.0025)
    d_opt = tf.keras.optimizers.Adam(0.002)
    ema = EMA(G)

    cm_src = CheckpointManager(str(src_dir), G, D, G_ema, g_opt, d_opt, nimg, step, ema_state=ema)
    cm_src.save(kimg=66.0)

    # Now simulate a fresh session with a new empty output dir
    dst_dir = tmp_path / "fresh_session_output"
    nimg_new = tf.Variable(0, dtype=tf.int64)
    step_new = tf.Variable(0, dtype=tf.int64)
    cm_dst = CheckpointManager(str(dst_dir), G, D, G_ema, g_opt, d_opt, nimg_new, step_new, ema_state=ema)

    # Restore passing the previous output checkpoints dir
    src_ckpts = src_dir / "checkpoints"
    ok = cm_dst.restore(resume_dir=src_ckpts)
    assert ok
    assert int(nimg_new) == 66000
    assert int(step_new) == 4125

    # Check that subsequent save becomes ckpt-2
    next_path = cm_dst.save(kimg=80.0)
    assert "ckpt-2" in str(next_path)


def test_backup_drive_trigger(tmp_path):
    out_dir = tmp_path / "run"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "checkpoint").write_text("model_checkpoint_path: ckpt-1\n")
    (ckpt_dir / "ckpt-1.index").write_text("data")
    (ckpt_dir / "ckpt-1.data-00000-of-00001").write_text("data")

    sample_img = tmp_path / "sample.png"
    sample_img.write_text("fake_png")

    drive_backup_dir = tmp_path / "google_drive_mount"
    trigger_backup(
        output_dir=str(out_dir),
        kimg=60.0,
        latest_ckpt_prefix=str(ckpt_dir / "ckpt-1"),
        sample_path=sample_img,
        drive_dir=str(drive_backup_dir),
        async_mode=False,
    )
    assert drive_backup_dir.exists()
    assert (drive_backup_dir / "checkpoint_kimg_0060.zip").exists()
    assert (drive_backup_dir / "sample.png").exists()


def test_token_and_repo_slug():
    repo = get_repo_slug()
    assert "/" in repo
