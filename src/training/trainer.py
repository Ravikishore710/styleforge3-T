"""Adversarial training engine: AMP, lazy R1, EMA, checkpointing, logging.

Training is measured in kimg (thousands of real images seen), matching GAN
literature convention. One 'step' = one G update + one D update on a batch.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..losses.losses import (adjust_betas_for_lazy_reg, compute_r1,
                             d_logistic_loss, g_logistic_loss, should_apply_r1)
from .checkpoint import CheckpointManager
from .ema import EMA


class Trainer:
    def __init__(self, cfg: dict, G, D, G_ema, dataset):
        self.cfg = cfg
        self.G, self.D, self.G_ema = G, D, G_ema
        if not getattr(G, "built", False):
            _ = G(tf.zeros([1, int(cfg["z_dim"])]))
        if not getattr(G_ema, "built", False):
            _ = G_ema(tf.zeros([1, int(cfg["z_dim"])]))
        if not getattr(D, "built", False):
            _ = D(tf.zeros([1, int(cfg["img_resolution"]), int(cfg["img_resolution"]), int(cfg["img_channels"])]))
        self.dataset = dataset.repeat() if hasattr(dataset, "repeat") else dataset
        self.batch = int(cfg["batch_size"])
        self.cur_nimg = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)

        b1, b2, eps = adjust_betas_for_lazy_reg(cfg["beta1"], cfg["beta2"],
                                                cfg["epsilon"],
                                                int(cfg["r1_interval"]))
        self.g_opt = tf.keras.optimizers.Adam(cfg["G_lr"], beta_1=b1, beta_2=b2,
                                              epsilon=eps)
        self.d_opt = tf.keras.optimizers.Adam(cfg["D_lr"], beta_1=b1, beta_2=b2,
                                              epsilon=eps)
        if int(cfg["mixed_precision"]):
            self.g_opt = tf.keras.mixed_precision.LossScaleOptimizer(self.g_opt)
            self.d_opt = tf.keras.mixed_precision.LossScaleOptimizer(self.d_opt)
        self.g_opt.build(self.G.trainable_variables)
        self.d_opt.build(self.D.trainable_variables)
        self.ema = EMA(G, cfg["ema_kimg"], cfg["ema_rampup"])
        self.ckpt = CheckpointManager(cfg["output_dir"], G, D, G_ema,
                                      self.g_opt, self.d_opt, self.cur_nimg,
                                      self.step, ema_state=self.ema)
        self.log_path = Path(cfg["output_dir"]) / "logs" / "log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary = tf.summary.create_file_writer(
            str(Path(cfg["output_dir"]) / "logs" / "tensorboard"))
        self._nan_seen = False

    # ---- compiled steps ----
    @tf.function
    def g_step(self, z):
        with tf.GradientTape() as tape:
            fake = self.G(z, training=True)
            loss = g_logistic_loss(self.D(fake, training=True))
            loss = self.g_opt.get_scaled_loss(loss) if hasattr(self.g_opt, "get_scaled_loss") else loss
        grads = tape.gradient(loss, self.G.trainable_variables)
        if hasattr(self.g_opt, "get_unscaled_gradients"):
            grads = self.g_opt.get_unscaled_gradients(grads)
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, self.G.trainable_variables)]
        self.g_opt.apply_gradients(zip(grads, self.G.trainable_variables))
        return loss

    @tf.function
    def d_step(self, real, z):
        with tf.GradientTape() as tape:
            fake = self.G(z, training=False)
            real_logits = self.D(real, training=True)
            fake_logits = self.D(fake, training=True)
            loss = d_logistic_loss(real_logits, fake_logits)
            if hasattr(self.d_opt, "get_scaled_loss"):
                loss = self.d_opt.get_scaled_loss(loss)
        grads = tape.gradient(loss, self.D.trainable_variables)
        if hasattr(self.d_opt, "get_unscaled_gradients"):
            grads = self.d_opt.get_unscaled_gradients(grads)
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, self.D.trainable_variables)]
        self.d_opt.apply_gradients(zip(grads, self.D.trainable_variables))
        return loss, tf.reduce_mean(real_logits), tf.reduce_mean(fake_logits)

    @tf.function
    def d_reg_step(self, real):
        with tf.GradientTape() as tape:
            r1 = compute_r1(self.D, real, float(self.cfg["gamma"]),
                            int(self.cfg["r1_interval"]))
            if hasattr(self.d_opt, "get_scaled_loss"):
                r1 = self.d_opt.get_scaled_loss(r1)
        grads = tape.gradient(r1, self.D.trainable_variables)
        if hasattr(self.d_opt, "get_unscaled_gradients"):
            grads = self.d_opt.get_unscaled_gradients(grads)
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, self.D.trainable_variables)]
        self.d_opt.apply_gradients(zip(grads, self.D.trainable_variables))
        return r1

    # ---- logging ----
    def _log_tick(self, stats: dict):
        stats = {"kimg": float(self.cur_nimg) / 1000.0, "step": int(self.step), **stats}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        with self.summary.as_default():
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    tf.summary.scalar(k, v, step=int(self.cur_nimg))
        return stats

    def _check_finite(self, *vals) -> bool:
        ok = all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in vals if v is not None)
        if not ok:
            self._nan_seen = True
        return ok

    # ---- main loop ----
    def fit(self, resume: bool = False, verbose: bool = True) -> dict:
        if resume:
            self.ckpt.restore()
        target_nimg = int(self.cfg["train_kimg"]) * 1000
        tick_nimg = int(self.cfg["tick_kimg"]) * 1000
        snap_nimg = int(self.cfg["snap_kimg"]) * 1000
        interval = int(self.cfg["r1_interval"])
        rng = np.random.default_rng(int(self.cfg["seed"]))
        t_start = time.time()
        t_last_tick, n_last_tick = t_start, int(self.cur_nimg)
        next_tick = int(self.cur_nimg) + tick_nimg
        next_snap = int(self.cur_nimg) + snap_nimg
        loss_g = loss_d = r1_val = None

        def fmt_time(sec):
            m, s = divmod(int(sec), 60)
            h, m = divmod(m, 60)
            return f"{h}h {m:02d}m" if h > 0 else f"{m}m {s:02d}s"

        it = iter(self.dataset)
        while int(self.cur_nimg) < target_nimg:
            try:
                real = next(it)
            except StopIteration:
                it = iter(self.dataset)
                real = next(it)
            real = tf.cast(real, tf.float32)
            z = tf.constant(rng.standard_normal([self.batch, int(self.cfg["z_dim"])],
                                                dtype=np.float32))
            loss_g = self.g_step(z)
            if should_apply_r1(int(self.step), interval):
                loss_d, r1_val = None, None
                r1_val = self.d_reg_step(real)
                loss_d, real_scr, fake_scr = self.d_step(real, z)
            else:
                r1_val = None
                loss_d, real_scr, fake_scr = self.d_step(real, z)
            self.ema.update(self.G, float(self.cur_nimg), float(self.batch))
            self.cur_nimg.assign_add(self.batch)
            self.step.assign_add(1)

            if not self._check_finite(loss_g, loss_d, r1_val):
                print(f"[warn] non-finite loss at kimg={float(self.cur_nimg)/1000:.1f}")
            n = int(self.cur_nimg)
            if n >= next_tick:
                t_now = time.time()
                tick_dur = max(t_now - t_last_tick, 1e-6)
                cur_speed = round((n - n_last_tick) / tick_dur, 1)
                t_last_tick, n_last_tick = t_now, n
                elapsed = t_now - t_start
                eta = (target_nimg - n) / max(cur_speed, 1e-6)

                next_tick += tick_nimg
                stats = self._log_tick({
                    "loss_G": float(loss_g), "loss_D": float(loss_d),
                    "r1": float(r1_val) if r1_val is not None else -1.0,
                    "D_real": float(real_scr), "D_fake": float(fake_scr),
                    "ema_decay": self.ema._decay(float(n), float(self.batch)),
                    "imgs_per_s": cur_speed,
                    "elapsed_sec": round(elapsed, 1),
                    "eta_sec": round(eta, 1),
                })
                if verbose:
                    print(f"kimg {stats['kimg']:>7.0f} | G {stats['loss_G']:.3f} | "
                          f"D {stats['loss_D']:.3f} | R1 {stats['r1']:.4f} | "
                          f"Dreal {stats['D_real']:.3f} Dfake {stats['D_fake']:.3f} | "
                          f"{stats['imgs_per_s']:.0f} img/s | elapsed: {fmt_time(elapsed)} | ETA: {fmt_time(eta)}")
            if n >= next_snap:
                next_snap += snap_nimg
                self.save()

        self.ema.copy_to(self.G_ema)
        self.save(final=True)
        return {"final_kimg": float(self.cur_nimg) / 1000.0,
                "nan_seen": self._nan_seen}

    def save(self, final: bool = False):
        path = self.ckpt.save(float(self.cur_nimg) / 1000.0)
        if final:
            print(f"[ckpt] final saved: {path}")
        return path
