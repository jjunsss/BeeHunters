#!/usr/bin/env python3
"""Run the final checkpoint once per FinalTest keyframe and package predictions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF = (
    ROOT
    / "data/extracted/testphase/BuzzSpot_testphase/annotations/"
    "test_testphase_keyframes.json"
)
DEFAULT_IMAGE_ROOT = (
    ROOT / "data/extracted/testphase/BuzzSpot_testphase/test_testphase"
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def make_chunks(reference: dict[str, Any], output: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    annotations = reference.get("annotations", [])
    for index in range(count):
        images = reference["images"][index::count]
        image_ids = {int(image["id"]) for image in images}
        chunk = dict(reference)
        chunk["images"] = images
        chunk["annotations"] = [
            annotation
            for annotation in annotations
            if int(annotation["image_id"]) in image_ids
        ]
        path = output / "chunk_refs" / f"chunk_{index:02d}.json"
        write_json(path, chunk)
        paths.append(path)
    return paths


def convert_predictions(path: Path, allowed_ids: set[int]) -> list[dict[str, Any]]:
    raw = read_json(path)
    output: list[dict[str, Any]] = []
    for item in raw:
        image_id = int(item["image_id"])
        category_id = int(item["category_id"])
        x, y, width, height = map(float, item["bbox"])
        if image_id not in allowed_ids or category_id not in {1, 2, 3, 4}:
            continue
        if width <= 0 or height <= 0:
            continue
        output.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, width, height],
                "score": float(item["score"]),
            }
        )
    return output


def environment(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    paths = [str(ROOT / "external/mmdetection/projects/CO-DETR"), str(ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def manifest_path(path: Path) -> str:
    """Record a useful path without publishing a machine-specific prefix."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/buzzspot/stage3_raw_cooldown_2e.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/train_mmdet/buzzspot_stage3_raw_cooldown_2e/epoch_2.pth"
        ),
    )
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/submissions/buzzspot_final_single_model",
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    for path in (args.config, args.checkpoint, args.ref, args.image_root):
        if not path.exists():
            raise FileNotFoundError(path)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one device")

    reference = read_json(args.ref)
    allowed_ids = {int(image["id"]) for image in reference["images"]}
    if len(allowed_ids) != 4763:
        raise ValueError(f"expected 4763 FinalTest keyframes, found {len(allowed_ids)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_refs = make_chunks(reference, args.output_dir, len(gpus))

    processes: list[tuple[subprocess.Popen[str], Any, Path]] = []
    for index, (gpu, chunk_ref) in enumerate(zip(gpus, chunk_refs)):
        chunk_dir = args.output_dir / "chunks" / f"chunk_{index:02d}"
        prefix = chunk_dir / "pred"
        log_path = chunk_dir / "inference.log"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(ROOT / "external/mmdetection/tools/test.py"),
            str(args.config),
            str(args.checkpoint),
            "--work-dir",
            str(chunk_dir / "work_dir"),
            "--cfg-options",
            f"test_evaluator.outfile_prefix={prefix}",
            f"test_evaluator.ann_file={chunk_ref.resolve()}",
            "test_evaluator.format_only=True",
            "model.backbone.init_cfg=None",
            "test_dataloader.dataset.data_root=",
            f"test_dataloader.dataset.ann_file={chunk_ref.resolve()}",
            f"test_dataloader.dataset.data_prefix.img={args.image_root.resolve()}/",
            f"test_dataloader.batch_size={args.batch_size}",
            f"test_dataloader.num_workers={args.num_workers}",
        ]
        log = log_path.open("w", encoding="utf-8")
        log.write(" ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((process, log, log_path))

    failures: list[str] = []
    for process, log, log_path in processes:
        return_code = process.wait()
        log.close()
        if return_code:
            failures.append(f"{log_path} (exit {return_code})")
    if failures:
        raise RuntimeError("inference failed: " + ", ".join(failures))

    predictions: list[dict[str, Any]] = []
    for index in range(len(gpus)):
        predictions.extend(
            convert_predictions(
                args.output_dir / "chunks" / f"chunk_{index:02d}" / "pred.bbox.json",
                allowed_ids,
            )
        )
    predictions.sort(
        key=lambda item: (
            item["image_id"],
            item["category_id"],
            -item["score"],
        )
    )
    predicted_ids = {item["image_id"] for item in predictions}
    if predicted_ids != allowed_ids:
        raise ValueError(f"prediction coverage mismatch: {len(predicted_ids)}/4763")

    predictions_path = args.output_dir / "predictions.json"
    write_json(predictions_path, predictions)
    zip_path = args.output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(predictions_path, arcname="predictions.json")
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "config": manifest_path(args.config),
            "checkpoint": manifest_path(args.checkpoint),
            "reference": manifest_path(args.ref),
            "gpus": gpus,
            "keyframes": len(allowed_ids),
            "predictions": len(predictions),
            "inference": "single_model_single_pass_no_tta_no_fusion",
        },
    )
    print(f"predictions: {predictions_path}")
    print(f"submission: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
