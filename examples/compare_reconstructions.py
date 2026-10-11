"""Create a multi-sample reconstruction comparison PDF."""

import argparse
from io import BytesIO
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from vqgan import VQVAE, Discriminator, create_train_state, make_optimizers
from vqgan.data import ImageDataset
from vqgan.training import load_checkpoint, read_checkpoint_configs


def load_generator(path: Path, expected_mode: str):
    config, disc_config = read_checkpoint_configs(path)
    mode = "vqvae" if disc_config is None else "vqgan"
    if mode != expected_mode:
        raise ValueError(f"Expected a {expected_mode} checkpoint, got {mode}: {path}")
    generator_key, discriminator_key = jax.random.split(jax.random.PRNGKey(0))
    template = create_train_state(
        VQVAE(config, key=generator_key),
        Discriminator(disc_config, key=discriminator_key) if disc_config is not None else None,
        make_optimizers(mode=mode),
    )
    return load_checkpoint(path, template).generator


def to_image(array: np.ndarray) -> Image.Image:
    """Convert normalized CHW model data to a clipped RGB Pillow image."""
    pixels = ((np.clip(array, -1, 1) + 1) * 127.5).round().astype(np.uint8)
    return Image.fromarray(pixels.transpose(1, 2, 0), mode="RGB")


def image_reader(array: np.ndarray) -> ImageReader:
    """Encode one source tile losslessly instead of rasterizing the full PDF page."""
    stream = BytesIO()
    to_image(array).save(stream, format="PNG")
    stream.seek(0)
    return ImageReader(stream)


def wrap_title(title: str, *, max_width: float, font: str, size: float) -> list[str]:
    """Wrap a PDF heading on word boundaries within one image-column width."""
    lines, current = [], []
    for word in title.split():
        candidate = " ".join((*current, word))
        if current and stringWidth(candidate, font, size) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def write_pdf(path: Path, columns, indices, *, rows_per_page: int):
    """Write vector text and losslessly embedded source-resolution image tiles."""
    tile_width = 108.0
    tile_height = tile_width * columns[0][1].shape[2] / columns[0][1].shape[3]
    title_font, title_size, title_leading = "Helvetica-Bold", 10.0, 12.0
    wrapped_titles = [
        wrap_title(title, max_width=tile_width, font=title_font, size=title_size)
        for title, _ in columns
    ]
    margin, gap, row_label = 28.0, 14.0, 14.0
    header = max(30.0, max(map(len, wrapped_titles)) * title_leading + 8.0)
    page_width = 2 * margin + len(columns) * tile_width + (len(columns) - 1) * gap
    document = None
    for start in range(0, len(indices), rows_per_page):
        count = min(rows_per_page, len(indices) - start)
        page_height = 2 * margin + header + count * (row_label + tile_height + gap) - gap
        if document is None:
            document = canvas.Canvas(
                str(path), pagesize=(page_width, page_height), pageCompression=1
            )
            document.setTitle(path.stem)
        else:
            document.setPageSize((page_width, page_height))
        document.setFont(title_font, title_size)
        for column, lines in enumerate(wrapped_titles):
            x = margin + column * (tile_width + gap) + tile_width / 2
            y = page_height - margin - title_size
            for line in lines:
                document.drawCentredString(x, y, line)
                y -= title_leading
        document.setFont("Helvetica", 8)
        for page_row in range(count):
            sample = start + page_row
            row_top = page_height - margin - header - page_row * (row_label + tile_height + gap)
            document.drawString(margin, row_top - 9, f"test index {int(indices[sample])}")
            image_y = row_top - row_label - tile_height
            for column, (_, values) in enumerate(columns):
                x = margin + column * (tile_width + gap)
                document.drawImage(
                    image_reader(values[sample]),
                    x,
                    image_y,
                    width=tile_width,
                    height=tile_height,
                    preserveAspectRatio=True,
                    mask="auto",
                )
        document.showPage()
    document.save()


@eqx.filter_jit
def reconstruct_batch(model, images):
    return jax.vmap(model)(images).reconstruction


def reconstruct(model, images, *, batch_size: int):
    """Run bounded-size batches so a large comparison does not multiply device memory."""
    batches = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        count = len(batch)
        if count < batch_size:
            batch = np.concatenate((batch, np.repeat(batch[-1:], batch_size - count, axis=0)))
        output = reconstruct_batch(model, jnp.asarray(batch))[:count]
        batches.append(np.asarray(jax.device_get(output)))
    return np.concatenate(batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--vqgan-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--vqgan-lpips-checkpoint",
        type=Path,
        help="Optional perceptual VQGAN checkpoint shown as a fourth column",
    )
    parser.add_argument(
        "--vqgan-feature-matching-checkpoint",
        type=Path,
        help="Optional feature-matching VQGAN checkpoint shown as another column",
    )
    parser.add_argument(
        "--vqgan-structural-checkpoint",
        type=Path,
        help="Optional normalized-feature/MS-SSIM/Laplacian VQGAN checkpoint",
    )
    parser.add_argument("--output", type=Path, default=Path("comparison.pdf"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.num_samples, args.rows_per_page, args.batch_size) <= 0:
        parser.error("num samples, rows per page, and batch size must be positive")

    dataset = ImageDataset(args.dataset_dir / f"{args.split}.npy")
    if args.num_samples > len(dataset):
        parser.error(f"num samples cannot exceed the {len(dataset)} images in the split")
    indices = np.sort(
        np.random.default_rng(args.seed).choice(len(dataset), args.num_samples, replace=False)
    )
    test = dataset.batch(indices)
    vqvae = load_generator(args.vqvae_checkpoint, "vqvae")
    vqgan = load_generator(args.vqgan_checkpoint, "vqgan")
    vqvae_images = reconstruct(vqvae, test, batch_size=args.batch_size)
    vqgan_images = reconstruct(vqgan, test, batch_size=args.batch_size)
    columns = [(args.split.title(), test), ("VQVAE", vqvae_images), ("VQGAN", vqgan_images)]
    if args.vqgan_lpips_checkpoint is not None:
        vqgan_lpips = load_generator(args.vqgan_lpips_checkpoint, "vqgan")
        vqgan_lpips_images = reconstruct(vqgan_lpips, test, batch_size=args.batch_size)
        columns.append(("VQGAN + LPIPS", vqgan_lpips_images))
    if args.vqgan_feature_matching_checkpoint is not None:
        vqgan_feature_matching = load_generator(
            args.vqgan_feature_matching_checkpoint, "vqgan"
        )
        feature_matching_images = reconstruct(
            vqgan_feature_matching, test, batch_size=args.batch_size
        )
        columns.append(("VQGAN + Feature Matching", feature_matching_images))
    if args.vqgan_structural_checkpoint is not None:
        vqgan_structural = load_generator(args.vqgan_structural_checkpoint, "vqgan")
        structural_images = reconstruct(vqgan_structural, test, batch_size=args.batch_size)
        columns.append(("VQGAN + Structural Losses", structural_images))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_pdf(
        args.output,
        columns,
        indices,
        rows_per_page=args.rows_per_page,
    )
    pages = (len(indices) + args.rows_per_page - 1) // args.rows_per_page
    print(f"Saved {len(indices)} comparisons across {pages} page(s): {args.output}")


if __name__ == "__main__":
    main()
