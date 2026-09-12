"""Exponential moving average of generator weights (the inference generator)."""
from __future__ import annotations

import numpy as np
import tensorflow as tf


class EMA:
    def __init__(self, generator: tf.keras.Model, ema_kimg: float = 10.0,
                 rampup: float = 0.05):
        self.ema_kimg = float(ema_kimg)
        self.rampup = float(rampup)
        if not getattr(generator, "built", False):
            cfg = getattr(generator, "cfg", None)
            z_dim = int(cfg["z_dim"]) if cfg and "z_dim" in cfg else 512
            _ = generator(tf.zeros([1, z_dim]))
        self.shadow = [tf.Variable(tf.convert_to_tensor(v), trainable=False,
                                   name=f"ema/{v.name.split(':')[0]}")
                       for v in generator.trainable_variables]

    def _decay(self, cur_nimg: float, batch_size: float) -> float:
        ema_nimg = min(self.ema_kimg * 1000.0, cur_nimg * self.rampup)
        ema_nimg = max(ema_nimg, 1e-8)
        return 0.5 ** (batch_size / ema_nimg)

    def update(self, generator: tf.keras.Model, cur_nimg: float,
               batch_size: float) -> float:
        decay = self._decay(cur_nimg, batch_size)
        for s, v in zip(self.shadow, generator.trainable_variables):
            s.assign(s * decay + tf.convert_to_tensor(v) * (1.0 - decay))
        return decay

    def copy_to(self, generator: tf.keras.Model) -> None:
        for s, v in zip(self.shadow, generator.trainable_variables):
            v.assign(tf.convert_to_tensor(s))

    def state_dict(self):
        return {"shadow": [tf.convert_to_tensor(s) for s in self.shadow],
                "ema_kimg": self.ema_kimg, "rampup": self.rampup}

    def load_state_dict(self, state) -> None:
        for s, v in zip(self.shadow, state["shadow"]):
            s.assign(v)
