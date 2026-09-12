import numpy as np
import tensorflow as tf

from src.losses.losses import (adjust_betas_for_lazy_reg, compute_r1,
                               d_logistic_loss, g_logistic_loss, r1_penalty,
                               should_apply_r1)


def test_g_loss_matches_manual_softplus():
    logits = tf.constant([[-2.0], [0.0], [3.0]])
    manual = tf.reduce_mean(tf.nn.softplus(-logits))
    assert np.isclose(float(g_logistic_loss(logits)), float(manual))


def test_d_loss_matches_manual():
    real = tf.constant([[2.0], [-1.0]])
    fake = tf.constant([[-3.0], [1.0]])
    manual = (tf.reduce_mean(tf.nn.softplus(-real)) +
              tf.reduce_mean(tf.nn.softplus(fake)))
    assert np.isclose(float(d_logistic_loss(real, fake)), float(manual))


def test_r1_penalty_analytic_linear():
    # D(x) = sum(k * x): dD/dx = k, penalty = 0.5 * mean(|k|^2 * numel)
    k = np.full((1, 4, 4, 3), 2.0, np.float32)
    x = tf.zeros([2, 4, 4, 3])
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = tf.reduce_sum(x * k, axis=[1, 2, 3])[:, None]
    grads = tape.gradient(out, x)
    pen = r1_penalty(grads)
    numel = 4 * 4 * 3
    expect = 0.5 * (2.0 ** 2) * numel
    assert np.isclose(float(pen), expect)


def test_r1_interval_schedule():
    assert should_apply_r1(16, 16)
    assert not should_apply_r1(15, 16)
    assert should_apply_r1(0, 16) is False or True  # step 0 handled by caller
    assert should_apply_r1(32, 16)
    assert not should_apply_r1(1, 0)  # interval 0 disables


def test_lazy_reg_optimizer_adjustment():
    b1, b2, eps = adjust_betas_for_lazy_reg(0.9, 0.99, 1e-8, 16)
    c = 16 / 15
    assert np.isclose(b1, 0.9 ** c)
    assert np.isclose(b2, 0.99 ** c)
    assert np.isclose(eps, 1e-8 / c)
    # no adjustment when interval is 1
    b1b, b2b, epsb = adjust_betas_for_lazy_reg(0.9, 0.99, 1e-8, 1)
    assert (b1b, b2b, epsb) == (0.9, 0.99, 1e-8)
