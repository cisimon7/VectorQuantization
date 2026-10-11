"""Shared CHW building blocks. Parameters stay float32 in mixed precision."""

from math import gcd

import equinox as eqx
import jax
import jax.numpy as jnp


def conv(layer: eqx.nn.Conv2d, x: jax.Array) -> jax.Array:
    """Cast convolution parameters for computation, preserving master weights."""
    layer = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(x.dtype) if eqx.is_inexact_array(leaf) else leaf, layer
    )
    return layer(x)


def normalize(layer: eqx.nn.GroupNorm, x: jax.Array) -> jax.Array:
    return layer(x.astype(jnp.float32)).astype(x.dtype)


def group_norm(channels: int, groups: int) -> eqx.nn.GroupNorm:
    return eqx.nn.GroupNorm(gcd(channels, groups), channels, eps=1e-6)


class ResidualBlock(eqx.Module):
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    shortcut: eqx.nn.Conv2d | None

    def __init__(self, in_channels: int, out_channels: int, groups: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm1 = group_norm(in_channels, groups)
        self.norm2 = group_norm(out_channels, groups)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, 3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, 3, padding=1, key=k2)
        self.shortcut = (
            eqx.nn.Conv2d(in_channels, out_channels, 1, key=k3)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x):
        h = conv(self.conv1, jax.nn.silu(normalize(self.norm1, x)))
        h = conv(self.conv2, jax.nn.silu(normalize(self.norm2, h)))
        return h + (x if self.shortcut is None else conv(self.shortcut, x))


class AttentionBlock(eqx.Module):
    """Spatial multi-head attention, used only at the bottleneck."""

    norm: eqx.nn.GroupNorm
    qkv: eqx.nn.Conv2d
    projection: eqx.nn.Conv2d
    heads: int = eqx.field(static=True)

    def __init__(self, channels: int, groups: int, heads: int, *, key):
        k1, k2 = jax.random.split(key)
        self.norm = group_norm(channels, groups)
        self.qkv = eqx.nn.Conv2d(channels, 3 * channels, 1, key=k1)
        self.projection = eqx.nn.Conv2d(channels, channels, 1, key=k2)
        self.heads = heads

    def __call__(self, x):
        channels, height, width = x.shape
        qkv = conv(self.qkv, normalize(self.norm, x))
        q, k, v = qkv.reshape(3, self.heads, channels // self.heads, height * width)
        # JAX attention uses (sequence, heads, head_dim), without a batch axis here.
        q, k, v = (a.transpose(2, 0, 1) for a in (q, k, v))
        h = jax.nn.dot_product_attention(q, k, v)
        h = h.transpose(1, 2, 0).reshape(channels, height, width)
        return x + conv(self.projection, h)


class Bottleneck(eqx.Module):
    first: ResidualBlock
    attention: AttentionBlock | None
    second: ResidualBlock

    def __init__(self, channels, groups, attention, heads, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.first = ResidualBlock(channels, channels, groups, key=k1)
        self.attention = AttentionBlock(channels, groups, heads, key=k2) if attention else None
        self.second = ResidualBlock(channels, channels, groups, key=k3)

    def __call__(self, x):
        x = self.first(x)
        if self.attention is not None:
            x = self.attention(x)
        return self.second(x)


class Resample(eqx.Module):
    convolution: eqx.nn.Conv2d
    up: bool = eqx.field(static=True)

    def __init__(self, channels, *, up, key):
        self.up = up
        self.convolution = eqx.nn.Conv2d(
            channels, channels, 3, stride=1 if up else 2, padding=1, key=key
        )

    def __call__(self, x):
        if self.up:
            x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
        return conv(self.convolution, x)


class Stage(eqx.Module):
    blocks: tuple[ResidualBlock, ...]
    resample: Resample | None

    def __init__(self, in_channels, out_channels, count, groups, *, resample, up, key):
        keys = jax.random.split(key, count + 1)
        self.blocks = tuple(
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, groups, key=k)
            for i, k in enumerate(keys[:-1])
        )
        self.resample = Resample(out_channels, up=up, key=keys[-1]) if resample else None

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x if self.resample is None else self.resample(x)


def run_block(block, x, *, remat):
    """Recompute block activations during backward rather than retaining them."""
    return eqx.filter_checkpoint(block)(x) if remat else block(x)
