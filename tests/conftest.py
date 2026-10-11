import jax
import pytest

from vqgan import VQGAN, Discriminator, DiscriminatorConfig, ModelConfig


@pytest.fixture(scope="session")
def tiny_config():
    return ModelConfig(
        base_channels=4,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        latent_channels=4,
        embedding_dim=3,
        codebook_size=7,
        norm_groups=2,
        attention_heads=2,
        token_chunk_size=7,
        codebook_chunk_size=3,
    )


@pytest.fixture(scope="session")
def model(tiny_config):
    return VQGAN(tiny_config, key=jax.random.PRNGKey(0))


@pytest.fixture(scope="session")
def discriminator():
    config = DiscriminatorConfig(base_channels=4, num_layers=1, max_channels=8, norm_groups=2)
    return Discriminator(config, key=jax.random.PRNGKey(1))


@pytest.fixture(scope="session")
def images():
    return jax.random.uniform(jax.random.PRNGKey(2), (2, 3, 8, 8), minval=-1, maxval=1)
