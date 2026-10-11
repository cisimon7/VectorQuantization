"""Pretrained-free multi-scale structural reconstruction losses."""

from math import log2

import jax
import jax.numpy as jnp


def _gaussian_kernel(channels: int, *, size: int = 7, sigma: float = 1.5):
    coordinates = jnp.arange(size, dtype=jnp.float32) - (size - 1) / 2
    kernel_1d = jnp.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= jnp.sum(kernel_1d)
    kernel_2d = jnp.outer(kernel_1d, kernel_1d)
    return jnp.broadcast_to(kernel_2d[None, None], (channels, 1, size, size))


def _blur(images):
    kernel = _gaussian_kernel(images.shape[1])
    return jax.lax.conv_general_dilated(
        images,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=images.shape[1],
    )


def _downsample(images):
    return jax.lax.reduce_window(
        images,
        0.0,
        jax.lax.add,
        window_dimensions=(1, 1, 2, 2),
        window_strides=(1, 1, 2, 2),
        padding="VALID",
    ) / 4


def _ssim_components(target, prediction):
    target_mean = _blur(target)
    prediction_mean = _blur(prediction)
    target_variance = jnp.maximum(_blur(target**2) - target_mean**2, 0)
    prediction_variance = jnp.maximum(
        _blur(prediction**2) - prediction_mean**2, 0
    )
    covariance = _blur(target * prediction) - target_mean * prediction_mean
    c1 = (0.01 * 2) ** 2
    c2 = (0.03 * 2) ** 2
    luminance = (2 * target_mean * prediction_mean + c1) / (
        target_mean**2 + prediction_mean**2 + c1
    )
    contrast_structure = (2 * covariance + c2) / (
        target_variance + prediction_variance + c2
    )
    axes = (1, 2, 3)
    return jnp.mean(luminance * contrast_structure, axis=axes), jnp.mean(
        contrast_structure, axis=axes
    )


def multi_scale_ssim_loss(target, prediction, *, max_scales: int = 5):
    """Return `1 - MS-SSIM` for NCHW images in the `[-1, 1]` range."""
    target = target.astype(jnp.float32)
    prediction = prediction.astype(jnp.float32)
    scales = min(max_scales, max(1, int(log2(min(target.shape[2:])))))
    standard_weights = jnp.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])[:scales]
    weights = standard_weights / jnp.sum(standard_weights)
    contrast_scores = []
    for scale in range(scales):
        similarity, contrast_structure = _ssim_components(target, prediction)
        if scale == scales - 1:
            final_similarity = jnp.clip(similarity, 1e-6, 1)
        else:
            contrast_scores.append(jnp.clip(contrast_structure, 1e-6, 1))
            target = _downsample(target)
            prediction = _downsample(prediction)
    score = final_similarity ** weights[-1]
    for scale, contrast in enumerate(contrast_scores):
        score *= contrast ** weights[scale]
    return 1 - jnp.mean(score)


def laplacian_pyramid_loss(target, prediction, *, max_levels: int = 4):
    """Match Gaussian residual bands at multiple spatial resolutions."""
    target = target.astype(jnp.float32)
    prediction = prediction.astype(jnp.float32)
    levels = min(max_levels, max(1, int(log2(min(target.shape[2:])))))
    losses = []
    for level in range(levels):
        target_low = _blur(target)
        prediction_low = _blur(prediction)
        losses.append(
            jnp.mean(jnp.abs((target - target_low) - (prediction - prediction_low)))
        )
        if level < levels - 1:
            target = _downsample(target_low)
            prediction = _downsample(prediction_low)
    return jnp.mean(jnp.stack(losses))
