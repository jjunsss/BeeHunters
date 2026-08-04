#!/usr/bin/env python3
"""Build the exact keyframe-only and original train+valid COCO files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "data/extracted/BuzzSet_challenge"
ANN_ROOT = ROOT / "data/extracted/testphase/BuzzSpot_testphase/annotations"
EXPECTED = {
    "train": (5275, 10884),
    "valid": (932, 1116),
    "test_testphase": (4763, 0),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def keyframe_only(coco: dict[str, Any]) -> dict[str, Any]:
    images = [image for image in coco["images"] if image.get("is_keyframe", False)]
    image_ids = {int(image["id"]) for image in images}
    output = dict(coco)
    output["images"] = images
    output["annotations"] = [
        annotation
        for annotation in coco.get("annotations", [])
        if int(annotation["image_id"]) in image_ids
    ]
    return output


def merge_sources(
    sources: list[tuple[str, Path, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    merged: dict[str, Any] = {
        "info": {"description": "BuzzSpot merged training source"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    category_signature: str | None = None
    next_image_id = 1
    next_annotation_id = 1
    source_images: Counter[str] = Counter()
    source_annotations: Counter[str] = Counter()
    category_annotations: Counter[int] = Counter()

    for name, path, prefix in sources:
        coco = read_json(path)
        signature = json.dumps(coco.get("categories", []), sort_keys=True)
        if category_signature is None:
            category_signature = signature
            merged["categories"] = coco.get("categories", [])
        elif signature != category_signature:
            raise ValueError(f"category mismatch: {path}")

        image_id_map: dict[int, int] = {}
        for image in coco.get("images", []):
            source_id = int(image["id"])
            file_name = str(image["file_name"]).strip("/")
            if prefix and not file_name.startswith(f"{prefix}/"):
                file_name = f"{prefix}/{file_name}"
            new_image = dict(image)
            new_image.update(
                id=next_image_id,
                file_name=file_name,
                source_dataset=name,
                source_image_id=source_id,
            )
            image_id_map[source_id] = next_image_id
            merged["images"].append(new_image)
            source_images[name] += 1
            next_image_id += 1

        for annotation in coco.get("annotations", []):
            source_image_id = int(annotation["image_id"])
            if source_image_id not in image_id_map:
                raise ValueError(f"orphan annotation in {path}: {annotation.get('id')}")
            new_annotation = dict(annotation)
            new_annotation.update(
                id=next_annotation_id,
                image_id=image_id_map[source_image_id],
                source_dataset=name,
                source_annotation_id=int(annotation["id"]),
            )
            merged["annotations"].append(new_annotation)
            source_annotations[name] += 1
            category_annotations[int(annotation["category_id"])] += 1
            next_annotation_id += 1

    missing = [
        image["file_name"]
        for image in merged["images"]
        if not (IMAGE_ROOT / image["file_name"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing; first: {missing[0]}")

    manifest = {
        "images": len(merged["images"]),
        "annotations": len(merged["annotations"]),
        "per_source_images": dict(source_images),
        "per_source_annotations": dict(source_annotations),
        "per_category_annotations": dict(sorted(category_annotations.items())),
        "missing_images": 0,
    }
    return merged, manifest


def main() -> int:
    for split, expected in EXPECTED.items():
        source = ANN_ROOT / f"{split}.json"
        output = ANN_ROOT / f"{split}_keyframes.json"
        keyframes = keyframe_only(read_json(source))
        counts = (len(keyframes["images"]), len(keyframes.get("annotations", [])))
        if counts != expected:
            raise ValueError(f"{split} count mismatch: {counts} != {expected}")
        write_json(output, keyframes)
        print(f"{output}: {counts[0]} images, {counts[1]} annotations")

    merged, manifest = merge_sources(
        [
            ("train_keyframes", ANN_ROOT / "train_keyframes.json", "train"),
            ("valid_keyframes", ANN_ROOT / "valid_keyframes.json", "valid"),
        ]
    )
    if (manifest["images"], manifest["annotations"]) != (6207, 12000):
        raise ValueError(f"train+valid count mismatch: {manifest}")
    write_json(ANN_ROOT / "trainval_keyframes.json", merged)
    write_json(ANN_ROOT / "trainval_keyframes_manifest.json", manifest)
    print("trainval_keyframes.json: 6207 images, 12000 annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
