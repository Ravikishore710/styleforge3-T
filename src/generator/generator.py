"""Full generator: MappingNetwork -> SynthesisNetwork."""
from __future__ import annotations

import tensorflow as tf

from .mapping import MappingNetwork
from .synthesis import SynthesisNetwork


class Generator(tf.keras.Model):
    def __init__(self, cfg: dict, name: str = "generator"):
        super().__init__(name=name)
        self.cfg = cfg
        self.synthesis = SynthesisNetwork(cfg)
        self.mapping = MappingNetwork(cfg["z_dim"], cfg["w_dim"],
                                      cfg["mapping_depth"],
                                      cfg["mapping_lr_mult"],
                                      num_ws=self.synthesis.num_ws)

    def call(self, z, shift=(0.0, 0.0), noise_mode=None, training=False,
             return_ws=False):
        ws = self.mapping(z)
        img = self.synthesis(ws, shift=shift, noise_mode=noise_mode,
                             training=training)
        if return_ws:
            return img, ws
        return img
