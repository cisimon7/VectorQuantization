"""Compiled VQVAE or VQGAN updates; objective equations live in vqgan.losses."""

import equinox as eqx
import jax
import jax.numpy as jnp

from vqgan.config import LossConfig
from vqgan.losses import PerceptualLoss, discriminator_loss, generator_loss, vqvae_loss
from vqgan.losses.adversarial import adversarial_active

from .state import Optimizers, TrainState


def _update(model, gradients, opt_state, optimizer):
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    return eqx.apply_updates(model, updates), opt_state


def make_train_step(
    optimizers: Optimizers,
    config: LossConfig = LossConfig(),
    *,
    perceptual_loss: PerceptualLoss | None = None,
    donate_state: bool = False,
):
    """Build once, call repeatedly with fixed batch shapes.

    If donate_state=True, the old state must not be reused after calling step.
    Image buffers are never donated and may be reused by the caller.
    The presence of a discriminator optimizer selects VQGAN; otherwise VQVAE.
    """
    if config.perceptual_weight > 0 and perceptual_loss is None:
        raise ValueError("provide perceptual_loss when perceptual_weight > 0")
    is_gan = optimizers.discriminator is not None
    generator_grad = eqx.filter_value_and_grad(
        generator_loss if is_gan else vqvae_loss, has_aux=True
    )
    discriminator_grad = (
        eqx.filter_value_and_grad(discriminator_loss, has_aux=True) if is_gan else None
    )

    def compiled(images, state):
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("training images must be a nonempty NCHW batch")
        if is_gan != (state.discriminator is not None):
            raise ValueError("training state and optimizer modes must match")
        if is_gan:
            (_, (metrics, reconstructed)), gradients = generator_grad(
                state.generator, state.discriminator, images, state.step, config, perceptual_loss
            )
        else:
            (_, (metrics, reconstructed)), gradients = generator_grad(
                state.generator, images, config, perceptual_loss
            )
        generator, generator_opt_state = _update(
            state.generator, gradients, state.generator_opt_state, optimizers.generator
        )

        # A static Python branch: no discriminator modules, tracing, forward pass,
        # gradients, or Adam moments exist in the compiled VQVAE executable.
        if not is_gan:
            return TrainState(generator, None, generator_opt_state, None, state.step + 1), metrics

        discriminator_params, discriminator_static = eqx.partition(
            state.discriminator, eqx.is_array
        )

        def train_discriminator(operand):
            params, opt_state = operand
            discriminator = eqx.combine(params, discriminator_static)
            (_, disc_metrics), gradients = discriminator_grad(discriminator, images, reconstructed)
            discriminator, opt_state = _update(
                discriminator, gradients, opt_state, optimizers.discriminator
            )
            return eqx.filter(discriminator, eqx.is_array), opt_state, disc_metrics

        def skip_discriminator(operand):
            discriminator, opt_state = operand
            metrics = {
                name: jnp.zeros((), jnp.float32)
                for name in (
                    "discriminator/total",
                    "discriminator/real_logits",
                    "discriminator/fake_logits",
                )
            }
            return discriminator, opt_state, metrics

        discriminator_params, discriminator_opt_state, disc_metrics = jax.lax.cond(
            adversarial_active(state.step, config),
            train_discriminator,
            skip_discriminator,
            (discriminator_params, state.discriminator_opt_state),
        )
        discriminator = eqx.combine(discriminator_params, discriminator_static)
        new_state = TrainState(
            generator, discriminator, generator_opt_state, discriminator_opt_state, state.step + 1
        )
        return new_state, {**metrics, **disc_metrics}

    # Put images first so Equinox's all-except-first donation protects the batch.
    compiled = eqx.filter_jit(compiled, donate="all-except-first" if donate_state else "none")

    def step(state: TrainState, images):
        return compiled(images, state)

    return step
