import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vqgan import LossConfig, make_optimizers, make_train_step
from vqgan.losses import (
    discriminator_feature_matching_loss,
    discriminator_hinge_loss,
    discriminator_loss,
    generator_adversarial_loss,
    laplacian_pyramid_loss,
    multi_scale_ssim_loss,
    quantization_losses,
    reconstruction_loss,
)
from vqgan.losses.perceptual import perceptual_loss


def test_quantization_gradient_destinations():
    encoded, embeddings = jnp.ones((2, 3)), jnp.zeros((2, 3))
    code_grads = jax.grad(lambda z, e: quantization_losses(z, e).codebook, argnums=(0, 1))(
        encoded, embeddings
    )
    commit_grads = jax.grad(lambda z, e: quantization_losses(z, e).commitment, argnums=(0, 1))(
        encoded, embeddings
    )
    np.testing.assert_array_equal(code_grads[0], jnp.zeros_like(encoded))
    assert jnp.linalg.norm(code_grads[1]) > 0
    assert jnp.linalg.norm(commit_grads[0]) > 0
    np.testing.assert_array_equal(commit_grads[1], jnp.zeros_like(embeddings))


def test_objective_values():
    assert reconstruction_loss(jnp.zeros(2), jnp.array([1, 3]), kind="l1") == 2
    assert reconstruction_loss(jnp.zeros(2), jnp.array([1, 3]), kind="l2") == 5
    assert discriminator_hinge_loss(jnp.array([2.0]), jnp.array([-2.0])) == 0
    assert discriminator_hinge_loss(jnp.array([0.0]), jnp.array([0.0])) == 1
    assert generator_adversarial_loss(jnp.array([2.0, 4.0])) == -3


def test_discriminator_detaches_fake_images(discriminator, images):
    gradient = jax.grad(lambda fake: discriminator_loss(discriminator, images, fake)[0])(images)
    np.testing.assert_array_equal(gradient, jnp.zeros_like(images))


def test_discriminator_feature_matching_detaches_real_features(discriminator, images):
    def features(batch):
        return jax.vmap(discriminator.logits_and_features)(batch)[1]

    real_features = features(images)
    assert discriminator_feature_matching_loss(real_features, real_features) == 0
    perturbed = images + 0.1
    fake_features = features(perturbed)
    assert discriminator_feature_matching_loss(real_features, fake_features) > 0

    real_grad, fake_grad = jax.grad(
        lambda real, fake: discriminator_feature_matching_loss(features(real), features(fake)),
        argnums=(0, 1),
    )(images, perturbed)
    np.testing.assert_array_equal(real_grad, jnp.zeros_like(images))
    assert jnp.linalg.norm(fake_grad) > 0


def test_normalized_feature_matching_ignores_feature_magnitude():
    real = (jnp.arange(48, dtype=jnp.float32).reshape(1, 3, 4, 4) + 1,)
    scaled = (real[0] * 2,)
    assert discriminator_feature_matching_loss(real, scaled) > 0
    np.testing.assert_allclose(
        discriminator_feature_matching_loss(real, scaled, normalize=True),
        0,
        atol=1e-7,
    )


def test_structural_losses_are_zero_for_identity_and_differentiable(images):
    assert multi_scale_ssim_loss(images, images) < 1e-6
    assert laplacian_pyramid_loss(images, images) == 0
    prediction = images + 0.1
    assert multi_scale_ssim_loss(images, prediction) > 0
    assert laplacian_pyramid_loss(images, prediction) > 0
    ms_ssim_grad = jax.grad(lambda x: multi_scale_ssim_loss(images, x))(prediction)
    laplacian_grad = jax.grad(lambda x: laplacian_pyramid_loss(images, x))(prediction)
    assert jnp.all(jnp.isfinite(ms_ssim_grad))
    assert jnp.linalg.norm(ms_ssim_grad) > 0
    assert jnp.linalg.norm(laplacian_grad) > 0


def test_perceptual_extension_is_disabled_and_validated():
    def forbidden(*args):
        raise AssertionError("disabled perceptual function was called")

    images = jnp.zeros((1, 3, 2, 2))
    assert perceptual_loss(images, images, forbidden, weight=0) == 0
    with pytest.raises(ValueError, match="perceptual_loss"):
        make_train_step(make_optimizers(), LossConfig(perceptual_weight=1))

    def fn(a, b):
        return jnp.mean((a - b) ** 2, axis=(1, 2, 3))

    grad = jax.grad(lambda x: perceptual_loss(images, x, fn, weight=1))(jnp.ones_like(images))
    assert jnp.linalg.norm(grad) > 0
