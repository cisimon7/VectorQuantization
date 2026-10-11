"""Convolutional encoder and decoder; no training objectives live here."""

import equinox as eqx
import jax

from vqgan.config import ModelConfig

from .blocks import Bottleneck, Stage, conv, group_norm, normalize, run_block


class Encoder(eqx.Module):
    input: eqx.nn.Conv2d
    stages: tuple[Stage, ...]
    bottleneck: Bottleneck
    norm: eqx.nn.GroupNorm
    output: eqx.nn.Conv2d
    remat: bool = eqx.field(static=True)

    def __init__(self, config: ModelConfig, *, key):
        channels = config.channels
        keys = jax.random.split(key, len(channels) + 3)
        self.input = eqx.nn.Conv2d(config.in_channels, channels[0], 3, padding=1, key=keys[0])
        self.stages = tuple(
            Stage(
                channels[max(i - 1, 0)],
                ch,
                config.num_res_blocks,
                config.norm_groups,
                resample=i < len(channels) - 1,
                up=False,
                key=keys[i + 1],
            )
            for i, ch in enumerate(channels)
        )
        self.bottleneck = Bottleneck(
            channels[-1],
            config.norm_groups,
            config.bottleneck_attention,
            config.attention_heads,
            key=keys[-2],
        )
        self.norm = group_norm(channels[-1], config.norm_groups)
        self.output = eqx.nn.Conv2d(
            channels[-1], config.latent_channels, 3, padding=1, key=keys[-1]
        )
        self.remat = config.remat

    def __call__(self, x):
        x = conv(self.input, x)
        for stage in self.stages:
            x = run_block(stage, x, remat=self.remat)
        x = run_block(self.bottleneck, x, remat=self.remat)
        return conv(self.output, jax.nn.silu(normalize(self.norm, x)))


class Decoder(eqx.Module):
    input: eqx.nn.Conv2d
    bottleneck: Bottleneck
    stages: tuple[Stage, ...]
    norm: eqx.nn.GroupNorm
    output: eqx.nn.Conv2d
    remat: bool = eqx.field(static=True)

    def __init__(self, config: ModelConfig, *, key):
        channels = config.channels[::-1]
        keys = jax.random.split(key, len(channels) + 3)
        self.input = eqx.nn.Conv2d(config.latent_channels, channels[0], 3, padding=1, key=keys[0])
        self.bottleneck = Bottleneck(
            channels[0],
            config.norm_groups,
            config.bottleneck_attention,
            config.attention_heads,
            key=keys[1],
        )
        self.stages = tuple(
            Stage(
                channels[max(i - 1, 0)],
                ch,
                config.num_res_blocks,
                config.norm_groups,
                resample=i < len(channels) - 1,
                up=True,
                key=keys[i + 2],
            )
            for i, ch in enumerate(channels)
        )
        self.norm = group_norm(channels[-1], config.norm_groups)
        self.output = eqx.nn.Conv2d(channels[-1], config.in_channels, 3, padding=1, key=keys[-1])
        self.remat = config.remat

    def features(self, x):
        x = run_block(self.bottleneck, conv(self.input, x), remat=self.remat)
        for stage in self.stages:
            x = run_block(stage, x, remat=self.remat)
        return jax.nn.silu(normalize(self.norm, x))

    def project(self, features):
        # Unbounded output, as in VQGAN; clip only for visualization.
        return conv(self.output, features)

    def __call__(self, x):
        return self.project(self.features(x))
