"""CLI: python -m data_processing.flowers102 {download,prepare,verify} [--root PATH]."""

import argparse
import json
from pathlib import Path

from .download import download


def main():
    local = Path(__file__).with_name("dataset.local.json")
    defaults = json.loads(local.read_text()) if local.exists() else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare", "verify"))
    parser.add_argument("--root", type=Path, default=defaults.get("root"))
    parser.add_argument("--size", type=int, default=defaults.get("image_size", 128))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    args = parser.parse_args()
    if args.root is None:
        parser.error("pass --root or configure data_processing/dataset.local.json")
    if args.command == "download":
        download(args.root)
    else:
        from .prepare import prepare, processed_directory, verify

        if args.command == "prepare":
            prepare(args.root, size=args.size, seed=args.seed)
        else:
            verify(processed_directory(args.root, args.size, args.seed))


if __name__ == "__main__":
    main()
