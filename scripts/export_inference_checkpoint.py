#!/usr/bin/env python3
"""Create a minimal, sanitized inference checkpoint for model release."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_meta(checkpoint: dict) -> dict:
    """Keep only inference metadata that is safe and useful to publish."""
    source_meta = checkpoint.get("meta") or {}
    dataset_meta = source_meta.get("dataset_meta")
    if not isinstance(dataset_meta, dict):
        raise KeyError("meta.dataset_meta missing from source checkpoint")
    return {"dataset_meta": dataset_meta}


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"state_dict missing from {args.source}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {args.output}")
    slim = {
        "meta": public_meta(checkpoint),
        "state_dict": checkpoint["state_dict"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        temporary = Path(directory) / args.output.name
        torch.save(slim, temporary)
        os.replace(temporary, args.output)
    print(f"{args.output} {args.output.stat().st_size} bytes")
    print(f"sha256 {sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
