"""Readable composition of the generator objective, independent of optimization."""

import equinox as eqx
import jax
import jax.numpy as jnp

from vqgan.config import LossConfig
from vqgan.models.blocks import conv

from .adversarial import adversarial_active, generator_adversarial_loss
from .autoencoder import autoencoder_terms, reconstruction_terms
from .feature_matching import discriminator_feature_matching_loss
from .perceptual import PerceptualLoss


def adaptive_weight(output_layer, features, images, discriminator, config, perceptual_fn):
    """Balance gradient norms at the decoder's last convolution.

    Reuse already-computed features: the encoder/decoder are not rerun. This
    coefficient is stop-gradient, so training does not use second derivatives.
    """
    features = jax.lax.stop_gradient(features)

    def project(weight):
        layer = eqx.tree_at(lambda m: m.weight, output_layer, weight)
        return jax.vmap(lambda x: conv(layer, x))(features)

    def reconstruction_objective(weight):
        total, _ = reconstruction_terms(images, project(weight), config, perceptual_fn)
        return total

    def adversarial_objective(weight):
        return generator_adversarial_loss(jax.vmap(discriminator)(project(weight)))

    rec_grad = jax.grad(reconstruction_objective)(output_layer.weight)
    adv_grad = jax.grad(adversarial_objective)(output_layer.weight)
    ratio = jnp.linalg.norm(rec_grad) / (jnp.linalg.norm(adv_grad) + config.adaptive_weight_eps)
    return jax.lax.stop_gradient(jnp.clip(ratio, 0, config.adaptive_weight_max))


def generator_loss(
    model,
    discriminator,
    images,
    step,
    config: LossConfig,
    perceptual_fn: PerceptualLoss | None = None,
):
    """Return (total, (scalar metrics, detached reconstructions)).

    Only the first argument is differentiated by the trainer. The discriminator
    participates in backpropagation to images, but its parameters are frozen.
    """
    if discriminator is None:
        raise ValueError("VQGAN loss requires a discriminator; use vqvae_loss for VQVAE")
    terms = autoencoder_terms(model, images, config, perceptual_fn)
    reconstructed = terms.reconstructed

    def adversarial_branch(_):
        if config.feature_matching_weight > 0:
            _, real_features = jax.vmap(discriminator.logits_and_features)(
                jax.lax.stop_gradient(images)
            )
            fake_logits, fake_features = jax.vmap(discriminator.logits_and_features)(reconstructed)
            feature_matching = discriminator_feature_matching_loss(
                real_features,
                fake_features,
                normalize=config.normalize_feature_matching,
            )
        else:
            fake_logits = jax.vmap(discriminator)(reconstructed)
            feature_matching = jnp.zeros((), jnp.float32)
        adversarial = generator_adversarial_loss(fake_logits)
        scale = (
            adaptive_weight(
                model.decoder.output, terms.features, images, discriminator, config, perceptual_fn
            )
            if config.adaptive_adversarial_weight
            else jnp.ones((), jnp.float32)
        )
        return adversarial, config.adversarial_weight * scale, feature_matching

    adversarial, weight, feature_matching = jax.lax.cond(
        adversarial_active(step, config),
        adversarial_branch,
        lambda _: (
            jnp.zeros((), jnp.float32),
            jnp.zeros((), jnp.float32),
            jnp.zeros((), jnp.float32),
        ),
        None,
    )

    total = (
        terms.total
        + weight * adversarial
        + config.feature_matching_weight * feature_matching
    )
    metrics = {
        **terms.metrics,
        "generator/total": total,
        "generator/adversarial": adversarial,
        "generator/adversarial_weight": weight,
        "generator/feature_matching": feature_matching,
    }
    return total, (metrics, jax.lax.stop_gradient(reconstructed))
