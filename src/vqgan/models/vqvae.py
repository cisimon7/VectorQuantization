"""Shared VQ autoencoder used by VQVAE and adversarial VQGAN training."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from vqgan.config import ModelConfig

from .autoencoder import Decoder, Encoder
from .blocks import conv
from .quantizer import QuantizerOutput, VectorQuantizer


class VQVAEOutput(NamedTuple):
    reconstruction: jax.Array
    encoded: jax.Array
    embeddings: jax.Array
    indices: jax.Array


class VQVAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    quantizer: VectorQuantizer
    pre_quant: eqx.nn.Conv2d
    post_quant: eqx.nn.Conv2d
    config: ModelConfig = eqx.field(static=True)

    def __init__(self, config: ModelConfig = ModelConfig(), *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.encoder = Encoder(config, key=k1)
        self.decoder = Decoder(config, key=k2)
        self.quantizer = VectorQuantizer(
            config.codebook_size,
            config.embedding_dim,
            token_chunk_size=config.token_chunk_size,
            codebook_chunk_size=config.codebook_chunk_size,
            key=k3,
        )
        self.pre_quant = eqx.nn.Conv2d(config.latent_channels, config.embedding_dim, 1, key=k4)
        self.post_quant = eqx.nn.Conv2d(config.embedding_dim, config.latent_channels, 1, key=k5)
        self.config = config

    def encode(self, image) -> tuple[jax.Array, QuantizerOutput]:
        if image.ndim != 3 or image.shape[0] != self.config.in_channels:
            raise ValueError("image must have shape (in_channels, height, width); batch with vmap")
        factor = self.config.downsample_factor
        if any(size <= 0 or size % factor for size in image.shape[1:]):
            raise ValueError(f"image height and width must be positive multiples of {factor}")
        image = image.astype(getattr(jnp, self.config.compute_dtype))
        z = conv(self.pre_quant, self.encoder(image))
        return z, self.quantizer(z)

    def decode_features(self, quantized):
        quantized = quantized.astype(getattr(jnp, self.config.compute_dtype))
        return self.decoder.features(conv(self.post_quant, quantized))

    def decode(self, quantized):
        return self.decoder.project(self.decode_features(quantized))

    def decode_codes(self, indices):
        return self.decode(self.quantizer.lookup(indices))

    def __call__(self, image) -> VQVAEOutput:
        encoded, q = self.encode(image)
        return VQVAEOutput(self.decode(q.quantized), encoded, q.embeddings, q.indices)
