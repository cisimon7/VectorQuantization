"""Frozen perceptual-network adapters used by the autoencoder objectives."""

from typing import Protocol

import jax
import jax.numpy as jnp


class PerceptualLoss(Protocol):
    def __call__(self, target: jax.Array, prediction: jax.Array) -> jax.Array:
        """Accept NCHW images; return a scalar or per-example losses.

        Keep feature-network parameters frozen, but preserve gradients through
        prediction. Normalization expected by the feature network belongs here.
        """
        ...


def load_jaxlpips(network: str = "vgg16") -> PerceptualLoss:
    """Load frozen pretrained jaxlpips weights and adapt NCHW to NHWC.

    Project images already use LPIPS' expected ``[-1, 1]`` range. The returned
    callable remains differentiable with respect to ``prediction``; the feature
    network is captured as a fixed loss dependency and is never optimized.
    """
    if network not in ("alexnet", "vgg16"):
        raise ValueError("jaxlpips network must be 'alexnet' or 'vgg16'")
    try:
        from jaxlpips import LPIPS
    except ImportError as error:
        raise ImportError(
            "jaxlpips is required for perceptual training; install '.[perceptual]'"
        ) from error

    model = LPIPS(pretrained_network=network)

    def loss(target: jax.Array, prediction: jax.Array) -> jax.Array:
        target_nhwc = jnp.moveaxis(jax.lax.stop_gradient(target), 1, -1)
        prediction_nhwc = jnp.moveaxis(prediction, 1, -1)
        return model(target_nhwc, prediction_nhwc)

    return loss


def perceptual_loss(target, prediction, fn: PerceptualLoss | None, *, weight: float):
    if weight == 0:
        return jnp.zeros((), dtype=jnp.float32)
    if fn is None:
        raise ValueError("a perceptual_loss callable is required when perceptual_weight > 0")
    return jnp.mean(fn(target.astype(jnp.float32), prediction.astype(jnp.float32)))
