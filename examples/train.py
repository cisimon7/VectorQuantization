"""Synthetic smoke test or training on prepared, memory-mapped image datasets."""

import argparse
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from vqgan import (
    VQVAE,
    Discriminator,
    DiscriminatorConfig,
    LossConfig,
    ModelConfig,
    create_train_state,
    make_optimizers,
    make_train_step,
)
from vqgan.data import ImageDataset
from vqgan.losses import load_jaxlpips
from vqgan.training import (
    evaluate_reconstruction,
    load_checkpoint,
    read_checkpoint_configs,
    save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, help="NCHW uint8 or normalized float .npy images")
    source.add_argument("--dataset-dir", type=Path, help="Prepared directory containing train.npy")
    parser.add_argument(
        "--model",
        choices=("vqgan", "vqvae"),
        help="Training objective; default vqgan, or inferred when resuming",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--disc-start", type=int, help="Default: 1000 for data, 2 for synthetic")
    parser.add_argument(
        "--perceptual-weight",
        type=float,
        default=0.0,
        help="Frozen jaxlpips loss weight; 0 disables perceptual loss",
    )
    parser.add_argument(
        "--perceptual-network",
        choices=("vgg16", "alexnet"),
        default="vgg16",
        help="Pretrained jaxlpips feature network",
    )
    parser.add_argument(
        "--feature-matching-weight",
        type=float,
        default=0.0,
        help="L1 weight for frozen real/fake discriminator features; VQGAN only",
    )
    parser.add_argument(
        "--normalize-feature-matching",
        action="store_true",
        help="Channel-normalize discriminator features before matching",
    )
    parser.add_argument(
        "--ms-ssim-weight",
        type=float,
        default=0.0,
        help="Pretrained-free multi-scale structural similarity loss weight",
    )
    parser.add_argument(
        "--laplacian-weight",
        type=float,
        default=0.0,
        help="Pretrained-free multi-scale Laplacian detail loss weight",
    )
    parser.add_argument("--base-channels", type=int, help="Default: 64 for data, 32 for synthetic")
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument(
        "--val-batches",
        type=int,
        default=4,
        help="Validation batches per check; standalone evaluation uses the full split",
    )
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.image_size, args.val_every, args.val_batches) <= 0:
        parser.error("steps, batch size, image size, and validation settings must be positive")
    if args.perceptual_weight < 0:
        parser.error("perceptual weight must be nonnegative")
    if args.feature_matching_weight < 0:
        parser.error("feature matching weight must be nonnegative")
    if args.ms_ssim_weight < 0 or args.laplacian_weight < 0:
        parser.error("MS-SSIM and Laplacian weights must be nonnegative")

    path = args.dataset_dir / "train.npy" if args.dataset_dir else args.data
    data = ImageDataset(path) if path else None
    validation = ImageDataset(args.dataset_dir / "validation.npy") if args.dataset_dir else None
    if data is not None and (data.images.shape[1] != 3 or len(data) < args.batch_size):
        parser.error("data must be an NCHW array with three channels and at least one full batch")

    # Small example defaults; ModelConfig() supplies the larger architecture.
    dtype = "bfloat16" if args.bf16 else "float32"
    base_channels = args.base_channels or (64 if data is not None else 32)
    config = ModelConfig(
        base_channels=base_channels,
        channel_multipliers=(1, 2, 2, 4) if data is not None else (1, 2),
        num_res_blocks=1,
        latent_channels=128 if data is not None else 32,
        embedding_dim=128 if data is not None else 32,
        codebook_size=1024 if data is not None else 128,
        compute_dtype=dtype,
    )
    mode = args.model or "vqgan"
    disc_config = (
        DiscriminatorConfig(
            base_channels=base_channels,
            num_layers=3 if data is not None else 2,
            compute_dtype=dtype,
        )
        if mode == "vqgan"
        else None
    )
    if args.resume:
        config, disc_config = read_checkpoint_configs(args.resume)
        saved_mode = "vqvae" if disc_config is None else "vqgan"
        if args.model is not None and args.model != saved_mode:
            parser.error(f"--model {args.model} does not match the {saved_mode} checkpoint")
        mode = saved_mode
    if args.feature_matching_weight > 0 and mode != "vqgan":
        parser.error("feature matching requires --model vqgan")
    disc_start = (
        args.disc_start if args.disc_start is not None else (1000 if data is not None else 2)
    )
    loss_config = LossConfig(
        discriminator_start=disc_start,
        perceptual_weight=args.perceptual_weight,
        feature_matching_weight=args.feature_matching_weight,
        normalize_feature_matching=args.normalize_feature_matching,
        ms_ssim_weight=args.ms_ssim_weight,
        laplacian_weight=args.laplacian_weight,
    )
    perceptual_fn = (
        load_jaxlpips(args.perceptual_network) if args.perceptual_weight > 0 else None
    )
    generator_key, disc_key, data_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    optimizers = make_optimizers(mode=mode)
    state = create_train_state(
        VQVAE(config, key=generator_key),
        Discriminator(disc_config, key=disc_key) if disc_config is not None else None,
        optimizers,
    )
    if args.resume:
        state = load_checkpoint(args.resume, state)
    train_step = make_train_step(
        optimizers,
        loss_config,
        perceptual_loss=perceptual_fn,
        donate_state=True,
    )
    rng = np.random.default_rng(args.seed)
    synthetic = None
    if data is None:
        synthetic = jax.random.uniform(
            data_key, (args.batch_size, 3, args.image_size, args.image_size), minval=-1, maxval=1
        )

    for i in range(args.steps):
        images = (
            synthetic
            if data is None
            else jnp.asarray(
                data.batch(
                    rng.choice(len(data), args.batch_size, replace=False),
                    rng=rng,
                    horizontal_flip=not args.no_augment,
                )
            )
        )
        start = perf_counter()
        state, metrics = train_step(state, images)
        metrics = jax.device_get(metrics)  # Synchronize only when logging.
        discriminator_log = f"d={metrics['discriminator/total']:.4f} " if mode == "vqgan" else ""
        print(
            f"model={mode} step={int(state.step)} loss={metrics['generator/total']:.4f} "
            + discriminator_log
            + f"perplexity={metrics['codebook/perplexity']:.1f} "
            f"seconds={perf_counter() - start:.3f}" + (" (includes compilation)" if i == 0 else "")
        )
        if validation is not None and int(state.step) % args.val_every == 0:
            result = evaluate_reconstruction(
                state.generator,
                validation,
                batch_size=args.batch_size,
                max_batches=args.val_batches,
            )
            print(f"validation: {result}")
    if args.checkpoint:
        save_checkpoint(args.checkpoint, state)


if __name__ == "__main__":
    main()
