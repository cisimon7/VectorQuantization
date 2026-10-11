"""Discriminator objective; generated images cannot send gradients to the generator."""

import jax
import jax.numpy as jnp

from .adversarial import discriminator_hinge_loss


def discriminator_loss(discriminator, images, reconstructed):
    real = jax.vmap(discriminator)(jax.lax.stop_gradient(images))
    fake = jax.vmap(discriminator)(jax.lax.stop_gradient(reconstructed))
    loss = discriminator_hinge_loss(real, fake)
    metrics = {
        "discriminator/total": loss,
        "discriminator/real_logits": jnp.mean(real),
        "discriminator/fake_logits": jnp.mean(fake),
    }
    return loss, metrics
