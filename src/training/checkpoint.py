"""Checkpointing: models + optimizers + training state, with resume support."""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf


class CheckpointManager:
    def __init__(self, output_dir: str, G, D, G_ema, g_opt, d_opt,
                 nimg: tf.Variable, step: tf.Variable, ema_state=None):
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {"G": G, "D": D, "G_ema": G_ema, "g_opt": g_opt, "d_opt": d_opt,
                   "nimg": nimg, "step": step}
        payload = {k: v for k, v in payload.items() if v is not None}
        self.ema_vars = None
        if ema_state is not None:
            self.ema_vars = ema_state.shadow
        for i, v in enumerate(self.ema_vars or []):
            payload[f"ema_{i}"] = v
        self.ckpt = tf.train.Checkpoint(**payload)
        self.manager = tf.train.CheckpointManager(self.ckpt, str(ckpt_dir),
                                                  max_to_keep=20)
    @property
    def latest(self):
        return self.manager.latest_checkpoint

    def save(self, kimg: float):
        path = self.manager.save()
        print(f"[ckpt] saved kimg={kimg:.0f} -> {path}")
        return path

    def restore(self) -> bool:
        if self.latest is None:
            return False
        self.ckpt.restore(self.latest).expect_partial()
        print(f"[ckpt] restored {self.latest}")
        return True

    def restore_ema_into(self, G_ema) -> None:
        if self.ema_vars is None:
            return
        if not getattr(G_ema, "built", False):
            cfg = getattr(G_ema, "cfg", None)
            z_dim = int(cfg["z_dim"]) if cfg and "z_dim" in cfg else 512
            _ = G_ema(tf.zeros([1, z_dim]))
        if len(self.ema_vars) != len(G_ema.trainable_variables):
            print(f"[warn] EMA var mismatch: {len(self.ema_vars)} in ckpt vs {len(G_ema.trainable_variables)} in model")
        for s, v in zip(self.ema_vars, G_ema.trainable_variables):
            v.assign(tf.convert_to_tensor(s))
