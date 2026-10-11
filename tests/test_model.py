from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vqgan import VQGAN, ModelConfig


def test_forward_and_code_decode(model, images):
    output = eqx.filter_jit(lambda m, x: jax.vmap(m)(x))(model, images)
    assert output.reconstruction.shape == images.shape
    assert output.indices.shape == (2, 4, 4)
    decoded = eqx.filter_jit(lambda m, codes: jax.vmap(m.decode_codes)(codes))(
        model, output.indices
    )
    np.testing.assert_allclose(decoded, output.reconstruction, atol=2e-5)
    assert jnp.all(jnp.isfinite(decoded))


def test_bfloat16_has_float32_master_weights(tiny_config, images):
    model = VQGAN(replace(tiny_config, compute_dtype="bfloat16"), key=jax.random.PRNGKey(0))
    assert all(
        x.dtype == jnp.float32 for x in jax.tree_util.tree_leaves(model) if eqx.is_inexact_array(x)
    )
    output = eqx.filter_jit(model)(images[0])
    assert output.reconstruction.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(output.reconstruction))


def test_remat_preserves_gradients(tiny_config, images):
    model = VQGAN(tiny_config, key=jax.random.PRNGKey(0))
    plain = VQGAN(replace(tiny_config, remat=False), key=jax.random.PRNGKey(0))
    objective = eqx.filter_jit(
        eqx.filter_value_and_grad(lambda m: jnp.mean(m(images[0]).reconstruction ** 2))
    )
    value, gradients = objective(model)
    plain_value, plain_gradients = objective(plain)
    np.testing.assert_allclose(value, plain_value, rtol=1e-5, atol=1e-6)
    for a, b in zip(
        jax.tree_util.tree_leaves(gradients),
        jax.tree_util.tree_leaves(plain_gradients),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_invalid_shapes_fail_early(model):
    with pytest.raises(ValueError, match="multiples"):
        model(jnp.ones((3, 7, 8)))
    with pytest.raises(ValueError, match="shape"):
        model(jnp.ones((1, 3, 8, 8)))


def test_invalid_config():
    with pytest.raises(ValueError):
        ModelConfig(channel_multipliers=())
    with pytest.raises(ValueError):
        ModelConfig(codebook_size=0)
