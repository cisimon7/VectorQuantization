"""Download verified Oxford sources and validate images while keeping them packed."""

import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
# Published TensorFlow Datasets checksums; no TensorFlow dependency is needed.
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/
# oxford_flowers102/checksums.tsv
SOURCES = {
    "102flowers.tgz": {
        "bytes": 344862509,
        "sha256": "2d01ecc807db462958cfe3d92f57a8c252b4abd240eb955770201e45f783b246",
    },
    "imagelabels.mat": {
        "bytes": 502,
        "sha256": "4903e94206bac23bf772aadf06451916df56b58fc483a62db32a97b82656651d",
    },
    "setid.mat": {
        "bytes": 14989,
        "sha256": "46b8678f91fd95d3c8f4feab80d271a6c834a1dd896fe29fd3e6ad9ce5c8dccd",
    },
}
IMAGE_COUNT = 8189


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sources(raw: Path):
    for name, expected in SOURCES.items():
        path = raw / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing or incomplete source: {path}; run download first")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")


def archive_members(archive):
    """Validate the complete archive index without expanding thousands of files."""
    members = {}
    for member in archive:
        if member.isdir():
            continue
        if not member.isfile() or not re.fullmatch(r"jpg/image_\d{5}\.jpg", member.name):
            raise ValueError(f"Unexpected archive entry: {member.name}")
        name = Path(member.name).name
        if name in members or member.size <= 0:
            raise ValueError(f"Duplicate or empty archive image: {name}")
        members[name] = member
    expected = {f"image_{i:05d}.jpg" for i in range(1, IMAGE_COUNT + 1)}
    if set(members) != expected:
        raise ValueError(f"Expected {IMAGE_COUNT} images in the archive")
    return members


def download(root: Path):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCES.items():
        path = raw / name
        if path.exists():
            if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"Existing source failed verification: {path}")
            print(f"Verified existing {name}", flush=True)
            continue
        partial = raw / f"{name}.part"
        print(f"Downloading {BASE_URL + name}", flush=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                BASE_URL + name,
            ],
            check=True,
        )
        if partial.stat().st_size != expected["bytes"] or sha256(partial) != expected["sha256"]:
            raise ValueError(f"Downloaded source failed verification: {partial}")
        partial.rename(path)
        print(f"SHA-256 verified: {name}", flush=True)
    with tarfile.open(raw / "102flowers.tgz", "r:*") as archive:
        archive_members(archive)
    print(f"Verified {IMAGE_COUNT} archive images; originals remain packed", flush=True)
    (raw / "sources.json").write_text(
        json.dumps(
            {name: {"url": BASE_URL + name, **record} for name, record in SOURCES.items()}, indent=2
        )
        + "\n"
    )
