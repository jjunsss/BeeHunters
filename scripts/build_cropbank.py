#!/usr/bin/env python3
"""Build the class-aware object CropBank and its train+valid COCO file."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from prepare_annotations import merge_sources, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "data/extracted/BuzzSet_challenge"
ANN_ROOT = ROOT / "data/extracted/testphase/BuzzSpot_testphase/annotations"


def crop_window(
    bbox: list[float], width: int, height: int, size: int, rng: random.Random
) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = map(float, bbox)
    center_x = x + box_width / 2 + rng.uniform(-0.15 * size, 0.15 * size)
    center_y = y + box_height / 2 + rng.uniform(-0.15 * size, 0.15 * size)
    crop_width, crop_height = min(size, width), min(size, height)
    left = max(0, min(width - crop_width, round(center_x - crop_width / 2)))
    top = max(0, min(height - crop_height, round(center_y - crop_height / 2)))
    return left, top, left + crop_width, top + crop_height


def remap_annotations(
    annotations: list[dict[str, Any]], window: tuple[int, int, int, int]
) -> list[dict[str, Any]]:
    left, top, right, bottom = window
    output: list[dict[str, Any]] = []
    for annotation in annotations:
        x, y, width, height = map(float, annotation["bbox"])
        x1, y1 = max(x, left), max(y, top)
        x2, y2 = min(x + width, right), min(y + height, bottom)
        if x2 <= x1 or y2 <= y1:
            continue
        retained = (x2 - x1) * (y2 - y1)
        if retained / max(1.0, width * height) < 0.35:
            continue
        item = dict(annotation)
        item.update(
            bbox=[x1 - left, y1 - top, x2 - x1, y2 - y1],
            area=retained,
            source_annotation_id=int(annotation["id"]),
        )
        output.append(item)
    return output


def annotation_weight(annotation: dict[str, Any]) -> float:
    multipliers = {1: 0.75, 2: 10.0, 3: 3.0, 4: 10.0}
    x, y, width, height = map(float, annotation["bbox"])
    area = float(annotation.get("area", width * height))
    attributes = annotation.get("attributes") or {}
    weight = multipliers[int(annotation["category_id"])]
    if area < 32 * 32:
        weight *= 2.0
    if int(attributes.get("blur", 0)) > 0:
        weight *= 1.5
    if int(attributes.get("occluded", 0)) >= 2:
        weight *= 1.5
    return weight


def build_cropbank(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"CropBank output must be empty: {output_dir}. Move or clear it explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = ANN_ROOT / "train_keyframes.json"
    coco = read_json(source_path)
    images = {int(image["id"]): image for image in coco["images"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    candidates = [a for a in coco["annotations"] if int(a.get("iscrowd", 0)) == 0]
    weights = [annotation_weight(annotation) for annotation in candidates]

    rng = random.Random(20260628)
    next_image_id = max(images) + 1
    next_annotation_id = max(int(a["id"]) for a in candidates) + 1
    output_images: list[dict[str, Any]] = []
    output_annotations: list[dict[str, Any]] = []
    class_counts: Counter[int] = Counter()

    for index in range(6000):
        selected = rng.choices(candidates, weights=weights, k=1)[0]
        source_image = images[int(selected["image_id"])]
        source_file = IMAGE_ROOT / "train" / str(source_image["file_name"])
        size = rng.randint(192, 384)
        with Image.open(source_file) as image_file:
            image = image_file.convert("RGB")
            window = crop_window(
                selected["bbox"],
                int(source_image["width"]),
                int(source_image["height"]),
                size,
                rng,
            )
            crop = image.crop(window)

        remapped = remap_annotations(by_image[int(source_image["id"])], window)
        if not remapped:
            continue
        crop_path = output_dir / f"crop_{index:06d}_ann{int(selected['id'])}.jpg"
        crop.save(crop_path, quality=95)
        file_name = crop_path.resolve().relative_to(IMAGE_ROOT.resolve()).as_posix()
        image_id = next_image_id
        next_image_id += 1
        output_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": crop.width,
                "height": crop.height,
                "source_image_id": int(source_image["id"]),
                "source_annotation_id": int(selected["id"]),
                "generated_type": "mosaic_cropbank",
                "crop_window_xyxy": list(window),
            }
        )
        for annotation in remapped:
            item = dict(annotation)
            item.update(
                id=next_annotation_id,
                image_id=image_id,
                generated_type="mosaic_cropbank_object",
            )
            next_annotation_id += 1
            output_annotations.append(item)
            class_counts[int(item["category_id"])] += 1

    info = dict(coco.get("info", {}))
    info.update(
        source_annotation=source_path.relative_to(ROOT).as_posix(),
        augmentation="mosaic_cropbank_no_mixup",
        crop_size=256,
        crop_size_range=[192, 384],
        crop_count_requested=6000,
        category_multipliers={"1": 0.75, "2": 10.0, "3": 3.0, "4": 10.0},
        generated_counts={"original": 0, "cropbank": len(output_images)},
        generated_annotation_class_counts={
            str(key): value for key, value in sorted(class_counts.items())
        },
    )
    output = {
        "info": info,
        "licenses": coco.get("licenses", []),
        "images": output_images,
        "annotations": output_annotations,
        "categories": coco.get("categories", []),
    }
    manifest = {
        "num_images": len(output_images),
        "num_annotations": len(output_annotations),
        "info": info,
    }
    return output, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        default=IMAGE_ROOT / "train_mosaic_cropbank_c256",
    )
    args = parser.parse_args()

    cropbank, cropbank_manifest = build_cropbank(args.output_image_dir)
    cropbank_path = ANN_ROOT / "train_keyframes_mosaic_cropbank_only_c256.json"
    write_json(cropbank_path, cropbank)
    write_json(
        ANN_ROOT / "train_keyframes_mosaic_cropbank_only_c256_manifest.json",
        cropbank_manifest,
    )
    if (len(cropbank["images"]), len(cropbank["annotations"])) != (5993, 6508):
        raise ValueError("CropBank counts differ from the final run")

    merged, merged_manifest = merge_sources(
        [
            ("train_mosaic_cropbank_only_c256", cropbank_path, ""),
            ("valid_keyframes", ANN_ROOT / "valid_keyframes.json", "valid"),
        ]
    )
    if (merged_manifest["images"], merged_manifest["annotations"]) != (6925, 7624):
        raise ValueError("merged CropBank counts differ from the final run")
    write_json(ANN_ROOT / "trainval_keyframes_mosaic_cropbank_only_c256.json", merged)
    write_json(
        ANN_ROOT / "trainval_keyframes_mosaic_cropbank_only_c256_manifest.json",
        merged_manifest,
    )
    print("CropBank: 5993 images, 6508 annotations")
    print("CropBank + valid: 6925 images, 7624 annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
