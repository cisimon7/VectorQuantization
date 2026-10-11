"""Exact nearest-neighbour VQ without a full token-by-codebook distance matrix."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp


class QuantizerOutput(NamedTuple):
    quantized: jax.Array  # Straight-through values (CHW).
    embeddings: jax.Array  # Raw selected embeddings, for codebook loss.
    indices: jax.Array  # HW integer codes.


def nearest_indices(tokens, codebook, *, token_chunk_size: int, codebook_chunk_size: int):
    """Bound distance storage by the chunk sizes; ties choose the first code.

    Assignment is non-differentiable and runs in float32 even for bf16 activations.
    Both scans emit only integer indices, never distances or one-hot encodings.
    """
    tokens = jax.lax.stop_gradient(tokens).astype(jnp.float32)
    codebook = jax.lax.stop_gradient(codebook).astype(jnp.float32)
    n, dim = tokens.shape
    size = codebook.shape[0]
    tc, cc = min(token_chunk_size, n), min(codebook_chunk_size, size)
    tokens = jnp.pad(tokens, ((0, (-n) % tc), (0, 0))).reshape(-1, tc, dim)
    codebook = jnp.pad(codebook, ((0, (-size) % cc), (0, 0))).reshape(-1, cc, dim)
    offsets = jnp.arange(codebook.shape[0], dtype=jnp.int32) * cc
    norms = jnp.sum(jnp.square(codebook), axis=-1)

    def assign_chunk(_, token_block):
        token_norm = jnp.sum(jnp.square(token_block), axis=-1, keepdims=True)

        def search(best, block):
            best_distance, best_index = best
            vectors, vector_norm, offset = block
            distances = token_norm + vector_norm[None, :] - 2 * (token_block @ vectors.T)
            valid = offset + jnp.arange(cc) < size
            distances = jnp.where(valid[None, :], distances, jnp.inf)
            index = jnp.argmin(distances, axis=-1)
            distance = jnp.take_along_axis(distances, index[:, None], axis=1)[:, 0]
            better = distance < best_distance
            return (
                jnp.where(better, distance, best_distance),
                jnp.where(better, offset + index, best_index),
            ), None

        initial = (jnp.full((tc,), jnp.inf), jnp.zeros((tc,), dtype=jnp.int32))
        (_, indices), _ = jax.lax.scan(search, initial, (codebook, norms, offsets))
        return None, indices

    _, indices = jax.lax.scan(assign_chunk, None, tokens)
    return indices.reshape(-1)[:n]


class VectorQuantizer(eqx.Module):
    codebook: jax.Array
    token_chunk_size: int = eqx.field(static=True)
    codebook_chunk_size: int = eqx.field(static=True)

    def __init__(self, size, dim, *, token_chunk_size=256, codebook_chunk_size=512, key):
        if min(size, dim, token_chunk_size, codebook_chunk_size) <= 0:
            raise ValueError("codebook and chunk dimensions must be positive")
        self.codebook = jax.random.uniform(key, (size, dim), minval=-1 / size, maxval=1 / size)
        self.token_chunk_size = token_chunk_size
        self.codebook_chunk_size = codebook_chunk_size

    def __call__(self, z):
        if z.ndim != 3 or z.shape[0] != self.codebook.shape[1] or min(z.shape[1:]) <= 0:
            raise ValueError("quantizer input must be nonempty CHW with embedding_dim channels")
        dim, height, width = z.shape
        tokens = z.transpose(1, 2, 0).reshape(-1, dim)
        indices = nearest_indices(
            tokens,
            self.codebook,
            token_chunk_size=self.token_chunk_size,
            codebook_chunk_size=self.codebook_chunk_size,
        )
        embeddings = self.codebook[indices].reshape(height, width, dim).transpose(2, 0, 1)
        # Decoder gradients go to the encoder, never directly to the codebook.
        values = embeddings.astype(z.dtype)
        # Subtract identical values to avoid cancellation between large encoder
        # activations and small embeddings, especially with bfloat16.
        quantized = jax.lax.stop_gradient(values) + (z - jax.lax.stop_gradient(z))
        return QuantizerOutput(quantized, embeddings, indices.reshape(height, width))

    def lookup(self, indices):
        """Decode an HW code grid into CHW embeddings (indices must be in range)."""
        if indices.ndim != 2 or not jnp.issubdtype(indices.dtype, jnp.integer):
            raise ValueError("indices must be a 2D integer grid")
        indices = eqx.error_if(
            indices,
            jnp.any((indices < 0) | (indices >= self.codebook.shape[0])),
            "code indices must be within the codebook",
        )
        return self.codebook[indices].transpose(2, 0, 1)
