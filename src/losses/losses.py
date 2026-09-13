# Non-saturating logistic GAN losses + R1 gradient penalty (lazy)
from __future__ import annotations

import tensorflow as tf


def g_logistic_loss(fake_logits: tf.Tensor) -> tf.Tensor:
    # E_z[ softplus(-D(G(z))) ] -- non-saturating generator loss
    return tf.reduce_mean(tf.nn.softplus(-tf.cast(fake_logits, tf.float32)))


def d_logistic_loss(real_logits: tf.Tensor, fake_logits: tf.Tensor) -> tf.Tensor:
    # E_x[softplus(-D(x))] + E_z[softplus(D(G(z)))]
    real = tf.reduce_mean(tf.nn.softplus(-tf.cast(real_logits, tf.float32)))
    fake = tf.reduce_mean(tf.nn.softplus(tf.cast(fake_logits, tf.float32)))
    return real + fake


def r1_penalty(grads: tf.Tensor) -> tf.Tensor:
    # 0.5 * E_x ||dD(x)/dx||^2, summed over H, W, C per sample
    grads = tf.cast(grads, tf.float32)
    per_sample = tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])
    return 0.5 * tf.reduce_mean(per_sample)


def compute_r1(discriminator, real_img, gamma: float, r1_interval: int) -> tf.Tensor:
    # Compute the (lazy-interval-compensated) R1 contribution to the D loss
    real_img = tf.identity(real_img)
    with tf.GradientTape() as tape:
        tape.watch(real_img)
        logits = discriminator(real_img, training=True)
        logits = tf.reduce_sum(tf.cast(logits, tf.float32))
    grads = tape.gradient(logits, real_img)
    return float(gamma) * float(r1_interval) * r1_penalty(grads)


def should_apply_r1(step: int, interval: int) -> bool:
    # Lazy R1 schedule: fire on intervals of D steps (step counted in D iters)
    return interval > 0 and (step % interval == 0)


def adjust_betas_for_lazy_reg(beta1: float, beta2: float, eps: float,
                              interval: int):
    # StyleGAN lazy-regularization optimizer adjustment
    if interval <= 1:
        return beta1, beta2, eps
    c = interval / (interval - 1.0)
    return beta1 ** c, beta2 ** c, eps / c
