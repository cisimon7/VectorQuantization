"""Shared reconstruction and VQ objective, independent of adversarial training."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from vqgan.config import LossConfig

from .perceptual import PerceptualLoss, perceptual_loss
from .quantization import codebook_metrics, quantization_losses
from .reconstruction import reconstruction_loss
from .structural import laplacian_pyramid_loss, multi_scale_ssim_loss


class AutoencoderTerms(NamedTuple):
    total: jax.Array
    metrics: dict[str, jax.Array]
    reconstructed: jax.Array
    features: jax.Array


def reconstruction_terms(images, reconstructed, config, perceptual_fn):
    pixel = reconstruction_loss(images, reconstructed, kind=config.reconstruction)
    perceptual = perceptual_loss(
        images, reconstructed, perceptual_fn, weight=config.perceptual_weight
    )
    ms_ssim = (
        multi_scale_ssim_loss(images, reconstructed)
        if config.ms_ssim_weight > 0
        else jnp.zeros((), jnp.float32)
    )
    laplacian = (
        laplacian_pyramid_loss(images, reconstructed)
        if config.laplacian_weight > 0
        else jnp.zeros((), jnp.float32)
    )
    total = (
        config.reconstruction_weight * pixel
        + config.perceptual_weight * perceptual
        + config.ms_ssim_weight * ms_ssim
        + config.laplacian_weight * laplacian
    )
    return total, (pixel, perceptual, ms_ssim, laplacian)


def autoencoder_terms(
    model, images, config: LossConfig, perceptual_fn: PerceptualLoss | None = None
) -> AutoencoderTerms:
    """One forward pass and the common objective for both model comparisons."""
    encoded, quantized = jax.vmap(model.encode)(images)
    features = jax.vmap(model.decode_features)(quantized.quantized)
    reconstructed = jax.vmap(model.decoder.project)(features)
    reconstruction, (pixel, perceptual, ms_ssim, laplacian) = reconstruction_terms(
        images, reconstructed, config, perceptual_fn
    )
    vq = quantization_losses(encoded, quantized.embeddings)
    total = (
        reconstruction
        + config.codebook_weight * vq.codebook
        + config.commitment_weight * vq.commitment
    )
    metrics = {
        "generator/total": total,
        "generator/reconstruction": pixel,
        "generator/perceptual": perceptual,
        "generator/ms_ssim": ms_ssim,
        "generator/laplacian": laplacian,
        "generator/codebook": vq.codebook,
        "generator/commitment": vq.commitment,
        **codebook_metrics(quantized.indices, model.config.codebook_size),
    }
    return AutoencoderTerms(total, metrics, reconstructed, features)


def vqvae_loss(model, images, config: LossConfig, perceptual_fn: PerceptualLoss | None = None):
    """VQVAE = reconstruction + codebook + commitment (+ optional perceptual loss).

    There is no discriminator argument or adversarial computation in this loss.
    The generator/* metric names are shared with VQGAN for direct comparison.
    """
    terms = autoencoder_terms(model, images, config, perceptual_fn)
    return terms.total, (terms.metrics, jax.lax.stop_gradient(terms.reconstructed))
