from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vqgan import VQGAN, VQVAE, LossConfig, create_train_state, make_optimizers, make_train_step
from vqgan.training import load_checkpoint, read_checkpoint_configs, save_checkpoint


def assert_tree_equal(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


def changed(a, b):
    return any(
        not np.array_equal(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True)
    )


def test_warmup_then_adversarial_update_and_resume(model, discriminator, images, tmp_path):
    optimizers = make_optimizers()
    initial = create_train_state(model, discriminator, optimizers)
    step = make_train_step(optimizers, LossConfig(discriminator_start=1))
    warm, metrics = step(initial, images)
    assert int(warm.step) == 1
    assert changed(initial.generator, warm.generator)
    assert_tree_equal(initial.discriminator, warm.discriminator)
    assert_tree_equal(initial.discriminator_opt_state, warm.discriminator_opt_state)
    assert metrics["generator/adversarial_weight"] == 0
    assert metrics["discriminator/total"] == 0

    trained, metrics = step(warm, images)
    assert int(trained.step) == 2
    assert changed(warm.discriminator, trained.discriminator)
    assert changed(warm.generator.quantizer.codebook, trained.generator.quantizer.codebook)
    assert metrics["generator/adversarial_weight"] > 0
    assert all(jnp.isfinite(v) for v in metrics.values())
    # Explicitly verify the composition of the reported generator loss.
    expected = (
        metrics["generator/reconstruction"]
        + metrics["generator/codebook"]
        + 0.25 * metrics["generator/commitment"]
        + metrics["generator/adversarial_weight"] * metrics["generator/adversarial"]
    )
    np.testing.assert_allclose(metrics["generator/total"], expected, rtol=1e-5)

    path = tmp_path / "state.eqx"
    save_checkpoint(path, trained)
    restored = load_checkpoint(path, initial)
    assert_tree_equal(restored, trained)
    continued, _ = step(trained, images)
    resumed, _ = step(restored, images)
    assert_tree_equal(continued, resumed)


def test_zero_adversarial_weight_keeps_discriminator_frozen(model, discriminator, images):
    optimizers = make_optimizers()
    initial = create_train_state(model, discriminator, optimizers)
    step = make_train_step(
        optimizers,
        LossConfig(adversarial_weight=0, discriminator_start=0, adaptive_adversarial_weight=False),
    )
    trained, metrics = step(initial, images)
    assert_tree_equal(initial.discriminator, trained.discriminator)
    assert_tree_equal(initial.discriminator_opt_state, trained.discriminator_opt_state)
    assert metrics["generator/adversarial"] == 0


def test_perceptual_callback_used_in_training(model, discriminator, images):
    optimizers = make_optimizers()
    state = create_train_state(model, discriminator, optimizers)
    step = make_train_step(
        optimizers,
        LossConfig(perceptual_weight=0.5, adversarial_weight=0),
        perceptual_loss=lambda target, pred: jnp.mean((target - pred) ** 2),
    )
    _, metrics = step(state, images)
    assert metrics["generator/perceptual"] > 0


def test_feature_matching_used_after_discriminator_warmup(model, discriminator, images):
    optimizers = make_optimizers()
    state = create_train_state(model, discriminator, optimizers)
    config = LossConfig(
        feature_matching_weight=0.5,
        discriminator_start=0,
        adaptive_adversarial_weight=False,
    )
    _, metrics = make_train_step(optimizers, config)(state, images)
    assert metrics["generator/feature_matching"] > 0
    expected = (
        metrics["generator/reconstruction"]
        + metrics["generator/codebook"]
        + 0.25 * metrics["generator/commitment"]
        + metrics["generator/adversarial"]
        + 0.5 * metrics["generator/feature_matching"]
    )
    np.testing.assert_allclose(metrics["generator/total"], expected, rtol=1e-5)


def test_pretrained_free_structural_objective(model, discriminator, images):
    optimizers = make_optimizers()
    state = create_train_state(model, discriminator, optimizers)
    config = LossConfig(
        feature_matching_weight=0.5,
        normalize_feature_matching=True,
        ms_ssim_weight=0.2,
        laplacian_weight=0.1,
        discriminator_start=0,
        adaptive_adversarial_weight=False,
    )
    _, metrics = make_train_step(optimizers, config)(state, images)
    assert metrics["generator/feature_matching"] > 0
    assert metrics["generator/ms_ssim"] > 0
    assert metrics["generator/laplacian"] > 0
    assert all(jnp.isfinite(value) for value in metrics.values())


def test_donation_preserves_input_batch(model, discriminator, images):
    optimizers = make_optimizers()

    # Copy parameter buffers so donating does not invalidate session fixtures.
    def copy(m):
        return jax.tree_util.tree_map(lambda x: x.copy() if eqx.is_array(x) else x, m)

    state = create_train_state(copy(model), copy(discriminator), optimizers)
    step = make_train_step(optimizers, LossConfig(adversarial_weight=0), donate_state=True)
    trained, _ = step(state, images)
    assert int(trained.step) == 1
    assert jnp.all(jnp.isfinite(images))


def test_checkpoint_checks_config(model, discriminator, tmp_path):
    optimizers = make_optimizers()
    state = create_train_state(model, discriminator, optimizers)
    path = tmp_path / "state.eqx"
    save_checkpoint(path, state)
    different = VQGAN(replace(model.config, remat=False), key=jax.random.PRNGKey(0))
    template = create_train_state(different, discriminator, optimizers)
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, template)


def test_vqvae_update_without_any_discriminator(model, tmp_path):
    optimizers = make_optimizers(mode="vqvae")
    initial = create_train_state(model, None, optimizers)
    assert optimizers.discriminator is None
    assert initial.discriminator is None
    assert initial.discriminator_opt_state is None
    assert initial.mode == "vqvae"
    assert VQGAN is VQVAE  # A single implementation and parameter layout.
    # This image is smaller than PatchGAN's minimum: even tracing D would fail.
    images = jax.random.uniform(jax.random.PRNGKey(12), (2, 3, 4, 4), minval=-1, maxval=1)
    config = LossConfig(
        reconstruction_weight=2,
        codebook_weight=0.5,
        commitment_weight=0.75,
        adversarial_weight=17,
        discriminator_start=0,
    )
    step = make_train_step(optimizers, config)
    trained, metrics = step(initial, images)
    assert int(trained.step) == 1
    assert changed(initial.generator, trained.generator)
    assert changed(initial.generator.quantizer.codebook, trained.generator.quantizer.codebook)
    assert trained.discriminator is None
    assert trained.discriminator_opt_state is None
    assert "generator/adversarial" not in metrics
    assert not any(name.startswith("discriminator/") for name in metrics)
    expected = (
        2 * metrics["generator/reconstruction"]
        + 0.5 * metrics["generator/codebook"]
        + 0.75 * metrics["generator/commitment"]
    )
    np.testing.assert_allclose(metrics["generator/total"], expected, rtol=1e-5)
    assert all(jnp.isfinite(value) for value in metrics.values())

    path = tmp_path / "vqvae.eqx"
    save_checkpoint(path, trained)
    config_read, disc_config = read_checkpoint_configs(path)
    assert config_read == model.config
    assert disc_config is None
    restored = load_checkpoint(path, initial)
    assert_tree_equal(restored, trained)
    continued, _ = step(trained, images)
    resumed, _ = step(restored, images)
    assert_tree_equal(continued, resumed)


def test_vqvae_and_vqgan_share_baseline_updates(model, discriminator, images):
    vae_optimizers = make_optimizers(mode="vqvae")
    gan_optimizers = make_optimizers(mode="vqgan")
    vae = create_train_state(model, None, vae_optimizers)
    gan = create_train_state(model, discriminator, gan_optimizers)
    config = LossConfig(discriminator_start=100)
    vae, vae_metrics = make_train_step(vae_optimizers, config)(vae, images)
    gan, gan_metrics = make_train_step(gan_optimizers, config)(gan, images)
    for a, b in zip(
        jax.tree_util.tree_leaves(vae.generator),
        jax.tree_util.tree_leaves(gan.generator),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    for name in vae_metrics:
        np.testing.assert_allclose(vae_metrics[name], gan_metrics[name], rtol=1e-5)


def test_invalid_training_mode_combinations(model, discriminator, images):
    vae_optimizers = make_optimizers(mode="vqvae")
    gan_optimizers = make_optimizers()
    with pytest.raises(ValueError, match="mode"):
        make_optimizers(mode="other")
    with pytest.raises(ValueError, match="both"):
        create_train_state(model, None, gan_optimizers)
    with pytest.raises(ValueError, match="both"):
        create_train_state(model, discriminator, vae_optimizers)
    state = create_train_state(model, None, vae_optimizers)
    with pytest.raises(ValueError, match="modes must match"):
        make_train_step(gan_optimizers)(state, images)


def test_checkpoint_rejects_cross_mode_restore(model, discriminator, tmp_path):
    vae = create_train_state(model, None, make_optimizers(mode="vqvae"))
    gan = create_train_state(model, discriminator, make_optimizers())
    path = tmp_path / "vqvae.eqx"
    save_checkpoint(path, vae)
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, gan)


def test_legacy_gan_checkpoint_is_still_readable(model, discriminator, tmp_path):
    import json

    state = create_train_state(model, discriminator, make_optimizers())
    path = tmp_path / "vqgan.eqx"
    save_checkpoint(path, state)
    with path.open("rb") as stream:
        metadata = json.loads(stream.readline())
        payload = stream.read()
    metadata["format_version"] = 1
    del metadata["mode"]
    legacy = tmp_path / "legacy.eqx"
    legacy.write_bytes((json.dumps(metadata) + "\n").encode() + payload)
    assert_tree_equal(load_checkpoint(legacy, state), state)
    assert read_checkpoint_configs(legacy) == (model.config, discriminator.config)
