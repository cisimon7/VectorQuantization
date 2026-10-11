"""Feature matching using the VQGAN discriminator as the feature extractor."""

import jax
import jax.numpy as jnp


def discriminator_feature_matching_loss(real_features, fake_features, *, normalize=False):
    """Mean L1 distance across discriminator blocks with a detached real target."""
    if not real_features or len(real_features) != len(fake_features):
        raise ValueError("real and fake discriminator features must have matching layers")
    def prepare(features):
        features = features.astype(jnp.float32)
        if normalize:
            norm = jnp.sqrt(jnp.sum(features**2, axis=1, keepdims=True) + 1e-10)
            features = features / norm
        return features

    losses = [
        jnp.mean(
            jnp.abs(
                jax.lax.stop_gradient(prepare(real))
                - prepare(fake)
            )
        )
        for real, fake in zip(real_features, fake_features, strict=True)
    ]
    return jnp.mean(jnp.stack(losses))
