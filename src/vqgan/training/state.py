"""Explicit models and optimizer states, with no hidden mutation."""

from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from vqgan.config import OptimizerConfig
from vqgan.models import VQVAE, Discriminator


class Optimizers(NamedTuple):
    generator: optax.GradientTransformation
    discriminator: optax.GradientTransformation | None


class TrainState(eqx.Module):
    generator: VQVAE
    discriminator: Discriminator | None
    generator_opt_state: Any
    discriminator_opt_state: Any
    step: jax.Array

    @property
    def mode(self) -> Literal["vqgan", "vqvae"]:
        return "vqvae" if self.discriminator is None else "vqgan"


def make_optimizers(
    generator: OptimizerConfig = OptimizerConfig(),
    discriminator: OptimizerConfig | None = None,
    *,
    mode: Literal["vqgan", "vqvae"] = "vqgan",
) -> Optimizers:
    """VQVAE creates only the autoencoder optimizer; VQGAN also creates PatchGAN's."""
    if mode not in ("vqgan", "vqvae"):
        raise ValueError("mode must be vqgan or vqvae")
    if mode == "vqvae" and discriminator is not None:
        raise ValueError("VQVAE mode does not accept a discriminator optimizer configuration")

    def build(config):
        return optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adamw(
                config.learning_rate,
                b1=config.beta1,
                b2=config.beta2,
                weight_decay=config.weight_decay,
            ),
        )

    return Optimizers(
        build(generator), build(discriminator or generator) if mode == "vqgan" else None
    )


def create_train_state(
    generator: VQVAE, discriminator: Discriminator | None, optimizers: Optimizers
) -> TrainState:
    if (discriminator is None) != (optimizers.discriminator is None):
        raise ValueError(
            "discriminator and its optimizer must both be present for VQGAN, "
            "or both absent for VQVAE"
        )
    if (
        discriminator is not None
        and generator.config.in_channels != discriminator.config.in_channels
    ):
        raise ValueError("generator and discriminator must use the same input channels")
    return TrainState(
        generator,
        discriminator,
        optimizers.generator.init(eqx.filter(generator, eqx.is_inexact_array)),
        (
            optimizers.discriminator.init(eqx.filter(discriminator, eqx.is_inexact_array))
            if discriminator is not None
            else None
        ),
        jnp.zeros((), dtype=jnp.int32),
    )
