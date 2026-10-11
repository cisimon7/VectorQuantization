"""PatchGAN objectives and the shared warm-up gate."""

import jax
import jax.numpy as jnp

from vqgan.config import LossConfig


def adversarial_active(step, config: LossConfig):
    return (step >= config.discriminator_start) & (config.adversarial_weight > 0)


def generator_adversarial_loss(fake_logits):
    return -jnp.mean(fake_logits.astype(jnp.float32))


def discriminator_hinge_loss(real_logits, fake_logits):
    real = jnp.mean(jax.nn.relu(1 - real_logits.astype(jnp.float32)))
    fake = jnp.mean(jax.nn.relu(1 + fake_logits.astype(jnp.float32)))
    return 0.5 * (real + fake)
