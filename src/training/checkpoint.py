# Checkpointing: models + optimizers + training state, with resume support
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
        self.ckpt_dir = ckpt_dir
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

    def restore(self, resume_dir: str | Path | None = None) -> bool:
        if (resume_dir is None or resume_dir == "auto") and self.latest is None:
            # Check if running in Kaggle and there's a previous session checkpoint in /kaggle/input
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                candidates = list(kaggle_input.glob("**/ckpt-*.index"))
                if candidates:
                    # Sort candidates so the latest ckpt number is chosen
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    auto_dir = candidates[0].parent
                    print(f"[ckpt] auto-detected Kaggle input checkpoint directory: {auto_dir}")
                    resume_dir = auto_dir

        if resume_dir:
            res_path = Path(resume_dir)
            if res_path.exists():
                src_dir = res_path if res_path.is_dir() else res_path.parent
                if src_dir.resolve() != self.ckpt_dir.resolve():
                    print(f"[ckpt] importing checkpoints from {src_dir} into {self.ckpt_dir}")
                    import shutil
                    for f in src_dir.iterdir():
                        if f.is_file() and (f.name.startswith("ckpt-") or f.name == "checkpoint"):
                            dest = self.ckpt_dir / f.name
                            if not dest.exists():
                                shutil.copy2(f, dest)
                    # refresh manager
                    self.manager = tf.train.CheckpointManager(self.ckpt, str(self.ckpt_dir), max_to_keep=20)

        target = self.latest
        if target is None and resume_dir:
            target = tf.train.latest_checkpoint(str(resume_dir))

        if target is None:
            print("[ckpt] no checkpoint found to restore")
            return False
        self.ckpt.restore(target).expect_partial()
        print(f"[ckpt] restored {target}")
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
