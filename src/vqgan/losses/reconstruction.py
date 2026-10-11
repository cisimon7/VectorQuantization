"""Pixel-space reconstruction objective."""

import jax.numpy as jnp


def reconstruction_loss(target, prediction, *, kind="l1"):
    difference = prediction.astype(jnp.float32) - target.astype(jnp.float32)
    if kind == "l1":
        return jnp.mean(jnp.abs(difference))
    if kind == "l2":
        return jnp.mean(jnp.square(difference))
    raise ValueError("reconstruction kind must be l1 or l2")
