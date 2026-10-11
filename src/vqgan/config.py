"""Immutable configuration; shape-affecting options are static under JIT."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal


def _positive(**values):
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _nonnegative(**values):
    for name, value in values.items():
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    base_channels: int = 128
    channel_multipliers: tuple[int, ...] = (1, 1, 2, 4)
    num_res_blocks: int = 2
    latent_channels: int = 256
    embedding_dim: int = 256
    codebook_size: int = 1024
    norm_groups: int = 32
    bottleneck_attention: bool = True
    attention_heads: int = 1
    remat: bool = True
    compute_dtype: Literal["float32", "bfloat16"] = "float32"
    token_chunk_size: int = 256
    codebook_chunk_size: int = 512

    def __post_init__(self):
        _positive(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            num_res_blocks=self.num_res_blocks,
            latent_channels=self.latent_channels,
            embedding_dim=self.embedding_dim,
            codebook_size=self.codebook_size,
            norm_groups=self.norm_groups,
            attention_heads=self.attention_heads,
            token_chunk_size=self.token_chunk_size,
            codebook_chunk_size=self.codebook_chunk_size,
        )
        if not self.channel_multipliers:
            raise ValueError("channel_multipliers must not be empty")
        for multiplier in self.channel_multipliers:
            _positive(channel_multiplier=multiplier)
        if self.channels[-1] % self.attention_heads:
            raise ValueError("bottleneck channels must be divisible by attention_heads")
        if self.compute_dtype not in ("float32", "bfloat16"):
            raise ValueError("compute_dtype must be float32 or bfloat16")

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(self.base_channels * m for m in self.channel_multipliers)

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.channel_multipliers) - 1)


@dataclass(frozen=True)
class DiscriminatorConfig:
    in_channels: int = 3
    base_channels: int = 64
    num_layers: int = 3
    max_channels: int = 512
    norm_groups: int = 32
    compute_dtype: Literal["float32", "bfloat16"] = "float32"

    def __post_init__(self):
        _positive(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            num_layers=self.num_layers,
            max_channels=self.max_channels,
            norm_groups=self.norm_groups,
        )
        if self.compute_dtype not in ("float32", "bfloat16"):
            raise ValueError("compute_dtype must be float32 or bfloat16")


@dataclass(frozen=True)
class LossConfig:
    reconstruction: Literal["l1", "l2"] = "l1"
    reconstruction_weight: float = 1.0
    codebook_weight: float = 1.0
    commitment_weight: float = 0.25
    perceptual_weight: float = 0.0
    feature_matching_weight: float = 0.0
    ms_ssim_weight: float = 0.0
    laplacian_weight: float = 0.0
    adversarial_weight: float = 1.0
    discriminator_start: int = 10_000
    adaptive_adversarial_weight: bool = True
    adaptive_weight_max: float = 10_000.0
    adaptive_weight_eps: float = 1e-4
    normalize_feature_matching: bool = False

    def __post_init__(self):
        if self.reconstruction not in ("l1", "l2"):
            raise ValueError("reconstruction must be l1 or l2")
        _nonnegative(
            reconstruction_weight=self.reconstruction_weight,
            codebook_weight=self.codebook_weight,
            commitment_weight=self.commitment_weight,
            perceptual_weight=self.perceptual_weight,
            feature_matching_weight=self.feature_matching_weight,
            ms_ssim_weight=self.ms_ssim_weight,
            laplacian_weight=self.laplacian_weight,
            adversarial_weight=self.adversarial_weight,
            adaptive_weight_max=self.adaptive_weight_max,
        )
        if not isinstance(self.normalize_feature_matching, bool):
            raise ValueError("normalize_feature_matching must be a boolean")
        if not isinstance(self.discriminator_start, int) or self.discriminator_start < 0:
            raise ValueError("discriminator_start must be a nonnegative integer")
        if not isfinite(self.adaptive_weight_eps) or self.adaptive_weight_eps <= 0:
            raise ValueError("adaptive_weight_eps must be finite and positive")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.9
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    def __post_init__(self):
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        _nonnegative(weight_decay=self.weight_decay)
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("Adam betas must lie in [0, 1)")
