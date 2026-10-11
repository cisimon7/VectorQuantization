"""Backward-compatible names for the shared autoencoder; GAN training adds PatchGAN."""

from .vqvae import VQVAE as VQGAN
from .vqvae import VQVAEOutput as VQGANOutput

__all__ = ["VQGAN", "VQGANOutput"]
