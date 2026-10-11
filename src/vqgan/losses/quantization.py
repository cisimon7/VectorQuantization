"""The two VQ objectives have intentionally different gradient destinations."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class QuantizationLosses(NamedTuple):
    codebook: jax.Array
    commitment: jax.Array


def quantization_losses(encoded, embeddings) -> QuantizationLosses:
    encoded, embeddings = encoded.astype(jnp.float32), embeddings.astype(jnp.float32)
    # Move codebook vectors towards encoder outputs; do not train the encoder here.
    codebook = jnp.mean(jnp.square(embeddings - jax.lax.stop_gradient(encoded)))
    # Encourage encoder outputs to commit to their chosen codebook vectors.
    commitment = jnp.mean(jnp.square(encoded - jax.lax.stop_gradient(embeddings)))
    return QuantizationLosses(codebook, commitment)


def codebook_metrics(indices, size):
    counts = jnp.bincount(indices.reshape(-1), length=size).astype(jnp.float32)
    probabilities = counts / jnp.maximum(jnp.sum(counts), 1)
    entropy = -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, 1e-10)))
    return {
        "codebook/perplexity": jnp.exp(entropy),
        "codebook/usage": jnp.mean((counts > 0).astype(jnp.float32)),
    }
