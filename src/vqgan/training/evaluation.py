"""Deterministic reconstruction metrics for validation and held-out testing."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


@eqx.filter_jit
def reconstruction_metrics(model, images):
    predictions = jax.vmap(model)(images).reconstruction.astype(jnp.float32)
    difference = predictions - images.astype(jnp.float32)
    axes = (1, 2, 3)
    mse = jnp.mean(jnp.square(difference), axis=axes)
    return {
        "l1": jnp.mean(jnp.abs(difference), axis=axes),
        "mse": mse,
        "psnr": 10 * jnp.log10(4 / jnp.maximum(mse, 1e-10)),
    }


@eqx.filter_jit
def reconstruct(model, images):
    """Reconstruct a fixed-shape batch for external perceptual metrics."""
    return jax.vmap(model)(images).reconstruction.astype(jnp.float32)


def evaluate_reconstruction(model, dataset, *, batch_size=8, max_batches=None):
    totals = {name: 0.0 for name in ("l1", "mse", "psnr")}
    count = 0
    for images, valid in dataset.evaluation_batches(batch_size, max_batches=max_batches):
        metrics = jax.device_get(reconstruction_metrics(model, jnp.asarray(images)))
        for name in totals:
            totals[name] += float(np.sum(metrics[name][:valid], dtype=np.float64))
        count += valid
    if count == 0:
        raise ValueError("evaluation dataset must not be empty")
    return {**{name: total / count for name, total in totals.items()}, "num_images": count}


def evaluate_perceptual(
    models, dataset, *, batch_size=8, kid_subsets=50, kid_subset_size=100, seed=0
):
    """Evaluate paired LPIPS/SSIM and distributional rFID/KID with frozen networks.

    PyTorch metric dependencies are imported lazily so ordinary JAX training and
    pixel-only evaluation do not require the optional ``metrics`` dependencies.
    Inception real features are computed once and retained across all models.
    """
    try:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
    except ImportError as error:
        raise ImportError(
            "install the optional metrics dependencies with pip install -e '.[metrics]'"
        ) from error

    if not models:
        raise ValueError("provide at least one model")
    if min(batch_size, kid_subsets, kid_subset_size) <= 0:
        raise ValueError("batch size and KID settings must be positive")
    kid_subset_size = min(kid_subset_size, len(dataset))
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    fid = FrechetInceptionDistance(reset_real_features=False, normalize=False)
    kid = KernelInceptionDistance(
        subsets=kid_subsets,
        subset_size=kid_subset_size,
        reset_real_features=False,
        normalize=False,
    )

    def clipped_float(images):
        if isinstance(images, torch.Tensor):
            return images.detach().to(torch.float32).clamp(-1, 1)
        return torch.from_numpy(np.array(images, copy=True)).to(torch.float32).clamp(-1, 1)

    def uint8_images(images):
        return ((clipped_float(images) + 1) * 127.5).round().to(torch.uint8)

    for images, valid in dataset.evaluation_batches(batch_size):
        real = uint8_images(images[:valid])
        fid.update(real, real=True)
        kid.update(real, real=True)

    results = {}
    for name, model in models.items():
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False)
        ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
        for images, valid in dataset.evaluation_batches(batch_size):
            prediction = np.asarray(jax.device_get(reconstruct(model, jnp.asarray(images))))[:valid]
            target = clipped_float(images[:valid])
            prediction = clipped_float(prediction)
            fid.update(uint8_images(prediction), real=False)
            kid.update(uint8_images(prediction), real=False)
            lpips.update(prediction, target)
            ssim.update(prediction, target)
        torch.manual_seed(seed)
        kid_mean, kid_std = kid.compute()
        results[name] = {
            "ssim": float(ssim.compute()),
            "lpips_alex": float(lpips.compute()),
            "rfid": float(fid.compute()),
            "kid_mean": float(kid_mean),
            "kid_std": float(kid_std),
        }
        fid.reset()
        kid.reset()
    return results
