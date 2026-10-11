"""Download and prepare COCO 2017 Captions for CLIP-style training.

Usage:
    python -m data_processing.coco_clip download --root /path/to/coco_clip
    python -m data_processing.coco_clip prepare --root /path/to/coco_clip
    python -m data_processing.coco_clip verify --root /path/to/coco_clip
"""

import argparse
import json
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

SOURCES = {
    "train2017.zip": "https://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "https://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": (
        "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ),
}
SPLITS = {
    "train": {
        "directory": "train2017",
        "annotations": "captions_train2017.json",
        "images": 118_287,
        "captions": 591_753,
    },
    "validation": {
        "directory": "val2017",
        "annotations": "captions_val2017.json",
        "images": 5_000,
        "captions": 25_014,
    },
}


def _validate_member(name: str, expected_prefix: str):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    if path.parts[0] != expected_prefix:
        raise ValueError(f"Unexpected archive path: {name}")


def _extract_images(archive_path: Path, images_root: Path, directory: str):
    destination = images_root / directory
    if destination.is_dir():
        count = sum(1 for path in destination.iterdir() if path.suffix.lower() == ".jpg")
        expected = next(
            record["images"]
            for record in SPLITS.values()
            if record["directory"] == directory
        )
        if count != expected:
            raise ValueError(f"Expected {expected} images in {destination}, found {count}")
        print(f"Verified existing image directory: {destination}", flush=True)
        return
    images_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        for member in members:
            _validate_member(member.filename, directory)
            if PurePosixPath(member.filename).suffix.lower() != ".jpg" or member.file_size <= 0:
                raise ValueError(f"Unexpected image archive entry: {member.filename}")
        archive.extractall(images_root)
    print(f"Extracted {len(members)} images: {destination}", flush=True)


def _extract_caption_annotations(archive_path: Path, annotations_root: Path):
    annotations_root.mkdir(parents=True, exist_ok=True)
    wanted = {record["annotations"] for record in SPLITS.values()}
    with zipfile.ZipFile(archive_path) as archive:
        available = {}
        for member in archive.infolist():
            if member.is_dir():
                continue
            _validate_member(member.filename, "annotations")
            name = PurePosixPath(member.filename).name
            if name in wanted:
                available[name] = member
        if set(available) != wanted:
            raise ValueError("Caption archive does not contain both 2017 caption files")
        for name, member in available.items():
            destination = annotations_root / name
            if destination.is_file() and destination.stat().st_size == member.file_size:
                continue
            partial = destination.with_suffix(".json.part")
            with archive.open(member) as source, partial.open("wb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
            partial.replace(destination)
    print(f"Prepared caption annotations: {annotations_root}", flush=True)


def download(root: Path):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        destination = raw / name
        partial = raw / f"{name}.part"
        if destination.is_file():
            with zipfile.ZipFile(destination) as archive:
                bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt existing archive member {bad}: {destination}")
            print(f"Verified existing archive: {destination}", flush=True)
            continue
        print(f"Downloading {url}", flush=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--show-error",
                "--retry",
                "5",
                "--connect-timeout",
                "30",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
        with zipfile.ZipFile(partial) as archive:
            bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Downloaded archive has a corrupt member: {bad}")
        partial.replace(destination)
        print(f"ZIP CRC verified: {destination}", flush=True)


def _write_manifest(root: Path, split: str, record: dict):
    annotations_path = root / "annotations" / record["annotations"]
    payload = json.loads(annotations_path.read_text())
    images = {int(image["id"]): image for image in payload["images"]}
    captions = defaultdict(list)
    for annotation in payload["annotations"]:
        captions[int(annotation["image_id"])].append(annotation["caption"].strip())
    if len(images) != record["images"] or len(payload["annotations"]) != record["captions"]:
        raise ValueError(f"Unexpected COCO {split} annotation counts")
    if set(images) != set(captions):
        raise ValueError(f"Some COCO {split} images have no captions")

    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    destination = manifests / f"{split}.jsonl"
    partial = manifests / f"{split}.jsonl.part"
    caption_total = 0
    with partial.open("w") as stream:
        for image_id in sorted(images):
            image = images[image_id]
            image_path = Path("images") / record["directory"] / image["file_name"]
            absolute_image = root / image_path
            if not absolute_image.is_file() or absolute_image.stat().st_size == 0:
                raise ValueError(f"Missing or empty COCO image: {absolute_image}")
            image_captions = captions[image_id]
            caption_total += len(image_captions)
            stream.write(
                json.dumps(
                    {
                        "image_id": image_id,
                        "image": image_path.as_posix(),
                        "width": int(image["width"]),
                        "height": int(image["height"]),
                        "captions": image_captions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if caption_total != record["captions"]:
        raise ValueError(f"Unexpected COCO {split} caption count")
    partial.replace(destination)
    return {
        "images": len(images),
        "captions": caption_total,
        "manifest": str(destination.relative_to(root)),
    }


def prepare(root: Path):
    raw = root / "raw"
    for name in SOURCES:
        if not (raw / name).is_file():
            raise ValueError(f"Missing {raw / name}; run download first")
    images_root = root / "images"
    _extract_images(raw / "train2017.zip", images_root, "train2017")
    _extract_images(raw / "val2017.zip", images_root, "val2017")
    _extract_caption_annotations(raw / "annotations_trainval2017.zip", root / "annotations")
    split_metadata = {
        split: _write_manifest(root, split, record) for split, record in SPLITS.items()
    }
    metadata = {
        "format_version": 1,
        "dataset": "coco_2017_captions",
        "root": ".",
        "image_format": "original JPEG",
        "manifest_format": "JSONL; one image per row with all captions",
        "training_policy": "sample one caption per image per epoch or flatten captions",
        "preprocessing": "deferred to training: RGB decode, random resized crop, normalization",
        "sources": SOURCES,
        "splits": split_metadata,
        "terms": "https://cocodataset.org/#termsofuse",
    }
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    verify(root)


def verify(root: Path):
    metadata = json.loads((root / "dataset.json").read_text())
    if metadata.get("format_version") != 1 or metadata.get("dataset") != "coco_2017_captions":
        raise ValueError("Unsupported COCO CLIP dataset metadata")
    for split, expected in SPLITS.items():
        manifest = root / metadata["splits"][split]["manifest"]
        image_count = caption_count = 0
        with manifest.open() as stream:
            for line in stream:
                row = json.loads(line)
                image = root / row["image"]
                if not image.is_file() or image.stat().st_size == 0 or not row["captions"]:
                    raise ValueError(f"Invalid manifest row for {split}: {row.get('image_id')}")
                image_count += 1
                caption_count += len(row["captions"])
        if image_count != expected["images"] or caption_count != expected["captions"]:
            raise ValueError(f"Manifest count mismatch for {split}")
        print(f"Verified {split}: images={image_count}, captions={caption_count}", flush=True)
    print(f"COCO CLIP data ready: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        download(args.root)
    elif args.command == "prepare":
        prepare(args.root)
    else:
        verify(args.root)


if __name__ == "__main__":
    main()
