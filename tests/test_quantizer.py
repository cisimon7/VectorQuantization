import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vqgan.models.quantizer import VectorQuantizer, nearest_indices


@pytest.mark.parametrize("token_chunk,code_chunk", [(1, 1), (7, 3), (256, 512)])
def test_chunked_search_matches_dense(token_chunk, code_chunk):
    k1, k2 = jax.random.split(jax.random.PRNGKey(5))
    tokens = jax.random.normal(k1, (23, 5))
    codebook = jax.random.normal(k2, (11, 5))
    expected = jnp.argmin(
        jnp.sum((tokens[:, None, :] - codebook[None, :, :]) ** 2, axis=-1), axis=-1
    )
    actual = jax.jit(
        lambda t, c: nearest_indices(
            t, c, token_chunk_size=token_chunk, codebook_chunk_size=code_chunk
        )
    )(tokens, codebook)
    np.testing.assert_array_equal(actual, expected)


def test_ties_and_padding():
    tokens = jnp.array([[0.0], [2.0], [3.0]])
    codebook = jnp.array([[1.0], [1.0], [4.0]])
    actual = nearest_indices(tokens, codebook, token_chunk_size=2, codebook_chunk_size=2)
    np.testing.assert_array_equal(actual, [0, 0, 2])


def test_straight_through_routes_gradients():
    quantizer = VectorQuantizer(5, 2, key=jax.random.PRNGKey(0))
    z = jnp.ones((2, 2, 3))
    output = quantizer(z)
    np.testing.assert_allclose(output.quantized, output.embeddings, atol=1e-7)
    encoder_grad = jax.grad(lambda x: jnp.sum(quantizer(x).quantized))(z)
    np.testing.assert_array_equal(encoder_grad, jnp.ones_like(z))
    codebook_grad = eqx.filter_grad(lambda q: jnp.sum(q(z).quantized))(quantizer)
    np.testing.assert_array_equal(codebook_grad.codebook, jnp.zeros_like(quantizer.codebook))


def test_lookup_checks_codes():
    quantizer = VectorQuantizer(5, 2, key=jax.random.PRNGKey(0))
    with pytest.raises((eqx.EquinoxRuntimeError, RuntimeError)):
        quantizer.lookup(jnp.array([[5]]))


def test_bfloat16_quantized_values_do_not_cancel():
    quantizer = VectorQuantizer(5, 2, key=jax.random.PRNGKey(0))
    z = jnp.full((2, 2, 3), 100, dtype=jnp.bfloat16)
    output = quantizer(z)
    np.testing.assert_array_equal(output.quantized, output.embeddings.astype(jnp.bfloat16))
