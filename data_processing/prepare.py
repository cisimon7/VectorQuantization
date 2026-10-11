"""Deterministic, class-stratified preparation into memory-mapped uint8 arrays."""

import json
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps
from scipy.io import loadmat

from .download import IMAGE_COUNT, SOURCES, archive_members, check_sources, sha256

SPLITS = ("train", "validation", "test")
RATIOS = np.array([0.8, 0.1, 0.1])
NUM_CLASSES = 102


def stratified_split(labels, seed=0):
    """Allocate 80/10/10 per class by largest remainder; keep every image once."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a nonempty vector")
    rng = np.random.default_rng(seed)
    groups = {split: [] for split in SPLITS}
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label) + 1  # Oxford image IDs are one-based.
        if len(ids) < 3:
            raise ValueError("each class needs at least three images")
        rng.shuffle(ids)
        ideal = len(ids) * RATIOS
        counts = np.floor(ideal).astype(int)
        order = np.argsort(-(ideal - counts), kind="stable")
        counts[order[: len(ids) - counts.sum()]] += 1
        if np.any(counts == 0):
            raise ValueError("each class must have an image in each split")
        chunks = np.split(ids, np.cumsum(counts)[:-1])
        for split, chunk in zip(SPLITS, chunks, strict=True):
            groups[split].extend(chunk.tolist())
    for split in SPLITS:
        groups[split] = np.array(groups[split], dtype=np.int32)
        rng.shuffle(groups[split])
    return groups


def preprocess_image(path: Path | BinaryIO, size: int):
    """EXIF-correct, RGB, aspect-preserving Lanczos resize and center crop."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = size / min(width, height)
        image = image.resize(
            (max(size, round(width * scale)), max(size, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        left, top = (image.width - size) // 2, (image.height - size) // 2
        image = image.crop((left, top, left + size, top + size))
        return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def official_assignments(raw: Path):
    split_ids = loadmat(raw / "setid.mat")
    assignments = {}
    for key, split in (("trnid", "train"), ("valid", "validation"), ("tstid", "test")):
        for image_id in split_ids[key].reshape(-1):
            image_id = int(image_id)
            if image_id in assignments:
                raise ValueError("Official splits contain overlapping IDs")
            assignments[image_id] = split
    if set(assignments) != set(range(1, IMAGE_COUNT + 1)):
        raise ValueError("Official splits do not cover the expected image IDs")
    return assignments


def processed_directory(root, size=128, seed=0):
    return Path(root) / "processed" / f"{size}_seed{seed}"


def prepare(root: Path, *, size=128, seed=0):
    if size <= 0 or size % 8:
        raise ValueError("image size must be a positive multiple of 8")
    raw = root / "raw"
    check_sources(raw)
    destination = processed_directory(root, size, seed)
    if destination.exists():
        print(f"Prepared directory exists; verifying {destination}", flush=True)
        verify(destination)
        return destination
    labels = loadmat(raw / "imagelabels.mat")["labels"].reshape(-1).astype(np.int32) - 1
    if labels.shape != (IMAGE_COUNT,) or set(labels.tolist()) != set(range(NUM_CLASSES)):
        raise ValueError("Expected 8189 labels with 102 zero-based categories")
    original = official_assignments(raw)
    split_ids = stratified_split(labels, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(dir=destination.parent, prefix=".prepare-") as temporary,
        tarfile.open(raw / "102flowers.tgz", "r:*") as archive,
    ):
        members = archive_members(archive)
        staging = Path(temporary) / destination.name
        staging.mkdir()
        metadata = {
            "format_version": 1,
            "dataset": "oxford_flowers102",
            "image_size": size,
            "seed": seed,
            "num_images": IMAGE_COUNT,
            "num_classes": NUM_CLASSES,
            "layout": "NCHW",
            "dtype": "uint8",
            "normalization": "float32(x) / 127.5 - 1",
            "split_policy": "custom class-stratified 80/10/10; largest remainder per class",
            "preprocessing": "EXIF transpose; RGB; resize shorter edge with Lanczos; center crop",
            "sources": SOURCES,
            "splits": {},
            "files": {},
        }
        arrays = {}
        rows_by_split = {}
        locations = {}
        for split, ids in split_ids.items():
            name = f"{split}.npy"
            arrays[split] = np.lib.format.open_memmap(
                staging / name, mode="w+", dtype=np.uint8, shape=(len(ids), 3, size, size)
            )
            rows = []
            for index, image_id in enumerate(ids):
                image_id = int(image_id)
                locations[image_id] = (split, index)
                rows.append(
                    {
                        "image_id": image_id,
                        "filename": f"image_{image_id:05d}.jpg",
                        "label": int(labels[image_id - 1]),
                        "official_split": original[image_id],
                    }
                )
            rows_by_split[split] = rows
            metadata["splits"][split] = {
                "count": len(ids),
                "images": name,
                "labels": f"{split}_labels.npy",
                "manifest": f"{split}.json",
            }
        # The archive is gzip-compressed and its members are not ordered by image ID.
        # Preserve the dict's archive insertion order to avoid repeatedly decompressing
        # from the beginning while still writing each image to its randomized split row.
        for decoded, (filename, member) in enumerate(members.items(), start=1):
            image_id = int(filename.removeprefix("image_").removesuffix(".jpg"))
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable image: {filename}")
            split, index = locations[image_id]
            with source:
                arrays[split][index] = preprocess_image(source, size)
            if decoded % 500 == 0 or decoded == IMAGE_COUNT:
                print(f"decoded: {decoded}/{IMAGE_COUNT}", flush=True)
        for array in arrays.values():
            array.flush()
        del arrays
        for split, ids in split_ids.items():
            np.save(staging / f"{split}_labels.npy", labels[ids - 1], allow_pickle=False)
            (staging / f"{split}.json").write_text(
                json.dumps(rows_by_split[split], indent=2) + "\n"
            )
        for path in staging.iterdir():
            metadata["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        (staging / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
        verify(staging)
        staging.rename(destination)
    print(f"Prepared data ready: {destination}", flush=True)
    return destination


def verify(directory: Path):
    """Check content hashes, shapes, labels, complete coverage and disjoint splits."""
    metadata = json.loads((directory / "dataset.json").read_text())
    if metadata["format_version"] != 1 or metadata["layout"] != "NCHW":
        raise ValueError("Unsupported dataset format")
    for name, expected in metadata["files"].items():
        path = directory / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f"Invalid or missing dataset file: {name}")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    all_ids = []
    for split in SPLITS:
        record = metadata["splits"][split]
        images = np.load(directory / record["images"], mmap_mode="r", allow_pickle=False)
        labels = np.load(directory / record["labels"], allow_pickle=False)
        rows = json.loads((directory / record["manifest"]).read_text())
        expected_shape = (record["count"], 3, metadata["image_size"], metadata["image_size"])
        if images.shape != expected_shape or images.dtype != np.uint8:
            raise ValueError(f"Invalid image array for {split}")
        if len(rows) != len(images) or labels.shape != (len(images),):
            raise ValueError(f"Manifest/label length mismatch for {split}")
        if not np.array_equal(labels, [row["label"] for row in rows]):
            raise ValueError(f"Manifest/label values mismatch for {split}")
        if set(labels.tolist()) != set(range(metadata["num_classes"])):
            raise ValueError(f"Missing categories in {split}")
        all_ids.extend(row["image_id"] for row in rows)
    if len(all_ids) != metadata["num_images"] or sorted(all_ids) != list(
        range(1, metadata["num_images"] + 1)
    ):
        raise ValueError("Splits overlap or fail to cover all images")
    print(
        "Verified hashes, array shapes, class coverage, and disjoint splits: "
        + ", ".join(f"{s}={metadata['splits'][s]['count']}" for s in SPLITS),
        flush=True,
    )
    return metadata
