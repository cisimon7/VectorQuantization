"""Stateless PatchGAN with GroupNorm, so no mutable batch statistics are needed."""

import equinox as eqx
import jax
import jax.numpy as jnp

from vqgan.config import DiscriminatorConfig

from .blocks import conv, group_norm, normalize


class Discriminator(eqx.Module):
    layers: tuple[eqx.nn.Conv2d, ...]
    norms: tuple[eqx.nn.GroupNorm | None, ...]
    output: eqx.nn.Conv2d
    config: DiscriminatorConfig = eqx.field(static=True)

    def __init__(self, config: DiscriminatorConfig = DiscriminatorConfig(), *, key):
        keys = jax.random.split(key, config.num_layers + 2)
        layers, norms = [], []
        previous = config.in_channels
        for i in range(config.num_layers + 1):
            channels = min(config.base_channels * 2**i, config.max_channels)
            layers.append(
                eqx.nn.Conv2d(
                    previous,
                    channels,
                    4,
                    stride=2 if i < config.num_layers else 1,
                    padding=1,
                    key=keys[i],
                )
            )
            norms.append(None if i == 0 else group_norm(channels, config.norm_groups))
            previous = channels
        self.layers, self.norms = tuple(layers), tuple(norms)
        self.output = eqx.nn.Conv2d(previous, 1, 4, padding=1, key=keys[-1])
        self.config = config

    @property
    def minimum_image_size(self):
        return 3 * 2**self.config.num_layers

    def logits_and_features(self, image):
        """Return PatchGAN logits and post-activation features from every block."""
        if image.ndim != 3 or image.shape[0] != self.config.in_channels:
            raise ValueError("discriminator image must have shape (in_channels, height, width)")
        if min(image.shape[1:]) < self.minimum_image_size:
            raise ValueError(
                f"discriminator requires image dimensions >= {self.minimum_image_size}"
            )
        x = image.astype(getattr(jnp, self.config.compute_dtype))
        features = []
        for layer, norm in zip(self.layers, self.norms, strict=True):
            x = conv(layer, x)
            if norm is not None:
                x = normalize(norm, x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
            features.append(x)
        return conv(self.output, x).astype(jnp.float32), tuple(features)

    def __call__(self, image):
        return self.logits_and_features(image)[0]
