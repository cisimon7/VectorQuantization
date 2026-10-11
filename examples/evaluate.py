"""Evaluate one checkpoint or compare two checkpoints on validation/test data."""

import argparse
import json
from pathlib import Path

import jax

from vqgan import VQVAE, Discriminator, create_train_state, make_optimizers
from vqgan.data import ImageDataset
from vqgan.training import (
    evaluate_perceptual,
    evaluate_reconstruction,
    load_checkpoint,
    read_checkpoint_configs,
)


def load_generator(path: Path):
    config, disc_config = read_checkpoint_configs(path)
    mode = "vqvae" if disc_config is None else "vqgan"
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    template = create_train_state(
        VQVAE(config, key=k1),
        Discriminator(disc_config, key=k2) if disc_config is not None else None,
        make_optimizers(mode=mode),
    )
    state = load_checkpoint(path, template)
    return mode, state.generator, int(state.step)


def emphasized(value, *, best):
    text = f"{value:.6f}"
    return f"**{text}**" if best else text


def markdown_report(report):
    models = report["models"]
    lower = ("l1", "mse", "lpips_alex", "rfid", "kid_mean")
    higher = ("psnr", "ssim")
    best = {metric: min(models, key=lambda name: models[name][metric]) for metric in lower}
    best.update({metric: max(models, key=lambda name: models[name][metric]) for metric in higher})
    rows = [
        "# Reconstruction evaluation",
        "",
        f"Split: **{report['split']}** · Images: **{report['num_images']}**",
        "",
        (
            "| Model | Step | L1 ↓ | MSE ↓ | PSNR dB ↑ | SSIM ↑ | "
            "LPIPS-Alex ↓ | rFID ↓ | KID ×1000 ↓ |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in models.items():
        values = [
            emphasized(metrics[metric], best=best[metric] == name)
            for metric in ("l1", "mse", "psnr", "ssim", "lpips_alex", "rfid")
        ]
        kid = metrics["kid_mean"] * 1000
        kid_std = metrics["kid_std"] * 1000
        kid_text = f"{kid:.4f} ± {kid_std:.4f}"
        if best["kid_mean"] == name:
            kid_text = f"**{kid_text}**"
        rows.append(
            f"| {name.upper()} | {metrics['step']} | "
            + " | ".join(values)
            + f" | {kid_text} |"
        )
    rows.extend(
        [
            "",
            "## Protocol",
            "",
            (
                "- L1, MSE, and PSNR use unclipped model outputs; they measure exact "
                "reconstruction fidelity."
            ),
            (
                "- SSIM and LPIPS use paired images clipped to `[-1, 1]`. LPIPS uses "
                "frozen pretrained AlexNet features."
            ),
            (
                "- rFID and KID compare clipped reconstruction and real-image distributions "
                "using frozen Inception-v3 features."
            ),
            (
                f"- KID uses {report['kid_subsets']} subsets of "
                f"{report['kid_subset_size']} images with seed {report['seed']}."
            ),
            (
                "- FID is sample-size biased; with this small test split, interpret rFID "
                "together with KID and paired LPIPS."
            ),
            "- Bold values are best within this report. Lower is better except for PSNR and SSIM.",
            "",
            "## Checkpoints",
            "",
        ]
    )
    rows.extend(
        f"- **{name.upper()}**: `{metrics['checkpoint']}`" for name, metrics in models.items()
    )
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compare-checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kid-subsets", type=int, default=50)
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional JSON metric report")
    parser.add_argument("--readme", type=Path, help="Optional Markdown comparison report")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.readme and not args.compare_checkpoint:
        parser.error("--readme requires --compare-checkpoint")

    dataset = ImageDataset(args.dataset_dir / f"{args.split}.npy")
    first_mode, first_model, first_step = load_generator(args.checkpoint)
    if args.compare_checkpoint is None:
        metrics = evaluate_reconstruction(first_model, dataset, batch_size=args.batch_size)
        report = {
            "model": first_mode,
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "dataset_dir": str(args.dataset_dir),
            "step": first_step,
            **metrics,
        }
    else:
        second_mode, second_model, second_step = load_generator(args.compare_checkpoint)
        if first_mode == second_mode:
            parser.error("comparison checkpoints must have different model modes")
        models = {first_mode: first_model, second_mode: second_model}
        paths = {first_mode: args.checkpoint, second_mode: args.compare_checkpoint}
        steps = {first_mode: first_step, second_mode: second_step}
        pixel = {
            name: evaluate_reconstruction(model, dataset, batch_size=args.batch_size)
            for name, model in models.items()
        }
        perceptual = evaluate_perceptual(
            models,
            dataset,
            batch_size=args.batch_size,
            kid_subsets=args.kid_subsets,
            kid_subset_size=args.kid_subset_size,
            seed=args.seed,
        )
        report = {
            "split": args.split,
            "dataset_dir": str(args.dataset_dir),
            "num_images": len(dataset),
            "seed": args.seed,
            "kid_subsets": args.kid_subsets,
            "kid_subset_size": min(args.kid_subset_size, len(dataset)),
            "models": {
                name: {
                    "checkpoint": str(paths[name]),
                    "step": steps[name],
                    **pixel[name],
                    **perceptual[name],
                }
                for name in models
            },
        }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.readme:
        args.readme.parent.mkdir(parents=True, exist_ok=True)
        args.readme.write_text(markdown_report(report))
        print(f"Wrote comparison report: {args.readme}")


if __name__ == "__main__":
    main()
