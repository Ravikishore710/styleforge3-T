import numpy as np
import tensorflow as tf

from src.config import load_config
from src.generator.generator import Generator
from src.training.ema import EMA


def _G():
    cfg = load_config(None, img_resolution=64)
    G = Generator(cfg)
    _ = G(tf.zeros([1, 512]))
    return G


def test_ema_tracks_source():
    G = _G()
    ema = EMA(G)
    src = [v.numpy().copy() for v in G.trainable_variables]
    for v in G.trainable_variables:
        v.assign(v + 1.0)
    decay = ema.update(G, cur_nimg=100000.0, batch_size=64.0)
    after = ema.shadow[0].numpy()
    before = src[0]
    src0 = G.trainable_variables[0].numpy()
    assert 0.0 < decay < 1.0
    assert np.allclose(after, before * decay + src0 * (1 - decay), atol=1e-6)


def test_ema_decay_bounds_and_rampup():
    G = _G()
    ema = EMA(G, ema_kimg=10, rampup=0.05)
    d_early = ema._decay(cur_nimg=1000.0, batch_size=64.0)     # rampup dominates
    d_late = ema._decay(cur_nimg=2_000_000.0, batch_size=64.0)
    assert 0.0 < d_early < d_late < 1.0


def test_ema_copy_to_restores():
    G = _G()
    ema = EMA(G)
    for v in G.trainable_variables:
        v.assign(v * 0.0)
    assert all(np.allclose(v.numpy(), 0.0) for v in G.trainable_variables)
    ema.copy_to(G)
    assert any(not np.allclose(v.numpy(), 0.0) for v in G.trainable_variables)


def test_checkpoint_roundtrip(tmp_path):
    from src.training.checkpoint import CheckpointManager
    G = _G()
    D = tf.keras.Sequential([tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(1)])
    G_ema = _G()
    for v_s, v_t in zip(G.trainable_variables, G_ema.trainable_variables):
        v_t.assign(v_s)
    nimg = tf.Variable(12345, dtype=tf.int64)
    step = tf.Variable(7, dtype=tf.int64)
    g_opt = tf.keras.optimizers.Adam(0.0025)
    d_opt = tf.keras.optimizers.Adam(0.002)
    ema = EMA(G)
    cm = CheckpointManager(str(tmp_path), G, D, G_ema, g_opt, d_opt, nimg, step,
                           ema_state=ema)
    cm.save(kimg=12.3)
    assert cm.latest is not None
    nimg2, step2 = tf.Variable(0, dtype=tf.int64), tf.Variable(0, dtype=tf.int64)
    cm2 = CheckpointManager(str(tmp_path), G, D, G_ema, g_opt, d_opt, nimg2, step2,
                            ema_state=ema)
    assert cm2.restore()
    assert int(nimg2) == 12345 and int(step2) == 7
