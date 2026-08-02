# Reproduction

Run every command from the repository root. The commands below reproduce the
final method; they do not promise a bitwise-identical checkpoint across CUDA,
GPU, and distributed-training implementations.

## 1. Environment

The recorded run used Python 3.10.9, PyTorch 2.0.1+cu118, torchvision
0.15.2+cu118, OpenMIM 0.3.9, MMCV 2.0.1, MMDetection 3.2.0, MMEngine
0.10.7, NumPy 1.26.4, OpenCV 4.11.0.86, and Pillow 12.2.0.
MMDetection's CO-DETR project files are pinned to commit
`cfd5d3a985b0249de009b67d04f37263e11cdf3d`.

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

The original training used NVIDIA A100 80GB GPUs with batch size one per GPU.

## 2. Data

Place the two organizer archives under `data/raw/`. The helper below downloads
the same files from the organizer's recorded locations and supports resume:

```bash
bash scripts/download_data.sh
```

Expected sizes:

| Archive | Bytes | SHA-256 | Role |
|---|---:|---|---|
| `BuzzSet_challenge.zip` | 35,261,367,058 | `964ad8ef...3a4453` | train/valid images |
| `BuzzSet_challenge_testphase.tar` | 19,557,529,600 | `24cf5885...31e846` | corrected annotations and FinalTest |

The complete hashes are recorded in `scripts/download_data.sh` and
`data/README.md`. Both archive size and SHA-256 are checked before extraction.

Extract them and generate keyframe-only COCO files:

```bash
bash scripts/prepare_data.sh
```

This command verifies 5,275 train, 932 validation, and 4,763 FinalTest
keyframes. It then creates the 6,207-image, 12,000-box raw train+valid source.
Extraction completion markers are written only after each archive command and
its required paths succeed, so an interrupted extraction is not silently
accepted on the next run.

Build the class-aware CropBank in an empty output directory:

```bash
python scripts/build_cropbank.py
```

The fixed parameters are 6,000 requested crops, size 192--384, seed 20260628,
class multipliers `[0.75, 10, 3, 10]`, retained-area threshold 0.35, and JPEG
quality 95. Seven requests retain no box, so the result contains 5,993 crop
images and 6,508 annotations. Adding the 932 former-validation keyframes gives
6,925 images and 7,624 annotations for stage 2.

`CropBank` names this generated object-centered image source. An older local
path used the tag `tCropBank`; it was not a separate augmentation. Mosaic is
used only in stage 2. Stages 1 and 3 read raw train+valid keyframes with the
standard Co-DINO resize/crop transforms and do not use Mosaic.

The script refuses a non-empty CropBank directory. This prevents stale JPEGs
from earlier runs from entering the public data path.

A clean end-to-end audit on 2026-08-02 regenerated all 5,993 referenced JPEGs
byte-for-byte, produced zero missing or unreferenced files, and matched the
canonical COCO `licenses/images/annotations/categories` fields exactly. The
merged CropBank+valid source also matched at 6,925 images and 7,624 annotations.

## 3. Initialization checkpoint

```bash
bash scripts/download_checkpoints.sh pretrained
```

The script downloads the public Objects365-to-COCO Co-DINO/Swin-L checkpoint
and verifies SHA-256
`614254c94b57acbff6a4448f7aa5d6315f8483b115438796e2606ce7a62712fe`.

## 4. Training

```bash
bash scripts/train.sh 1 0,1,2,3
bash scripts/train.sh 2 0,1,2,3
bash scripts/train.sh 3 0,1,2
```

| Stage | Data | Epochs | LR | Output |
|---|---|---:|---:|---|
| 1 | raw train+valid | 12 | `1e-4` | `outputs/train_mmdet/buzzspot_stage1_trainval_12e/epoch_12.pth` |
| 2 | CropBank Mosaic, `p=.85` | 3 | `1e-5` | `outputs/train_mmdet/buzzspot_stage2_cropbank_mosaic_3e/epoch_3.pth` |
| 3 | raw train+valid | 2 | `5e-6` | `outputs/train_mmdet/buzzspot_stage3_raw_cooldown_2e/epoch_2.pth` |

The winning artifact used four GPUs for cooldown epoch 1, then resumed
`epoch_1.pth` on three GPUs after an interrupted epoch 2. The clean command
above reproduces the 2E method on three GPUs, not the interruption history.

## 5. Final inference

```bash
python scripts/infer.py --gpus 0,1,2
```

The script runs one checkpoint once on each of the 4,763 FinalTest keyframes,
merges GPU shards, checks coverage, and writes:

```text
outputs/submissions/buzzspot_final_single_model/predictions.json
outputs/submissions/buzzspot_final_single_model/submission.zip
outputs/submissions/buzzspot_final_single_model/run_manifest.json
```

The ZIP contains only `predictions.json`. The procedure uses no TTA, WBF,
checkpoint ensemble, or auxiliary classifier at inference.

The public conversion path was checked against all three archived July shards:
all 1,314,945 predictions matched exactly and covered 4,763/4,763 keyframes.

## 6. Final checkpoint artifact

The full training checkpoint is 2.83GB because it includes optimizer state.
Create the inference-only artifact with:

```bash
python scripts/export_inference_checkpoint.py \
  outputs/train_mmdet/buzzspot_stage3_raw_cooldown_2e/epoch_2.pth \
  checkpoints/buzzspot_codino_etf_final_inference.pth
```

The verified release export is 943,564,029 bytes with SHA-256
`12705c0395abc7dd1d6937555e344e63d5f203e65b22d0b4e0d9f96590dfe31f`.
The export keeps only `state_dict` and the four-class `dataset_meta`; training
paths, configuration text, optimizer state, timestamps, and experiment names
are removed. Its machine-readable identity is in
`checkpoints/manifest.json`. Publish it as a GitHub Release asset rather than a
Git or Git LFS object, then add the asset URL to that manifest and the release
notes.
Users can then run:

```bash
FINAL_CHECKPOINT_URL='<url>' bash scripts/download_checkpoints.sh final
python scripts/infer.py \
  --checkpoint checkpoints/buzzspot_codino_etf_final_inference.pth \
  --gpus 0,1,2
```

The archived full winning checkpoint has SHA-256
`057722df93fa011b980154acf94fa89e670b86486e6aa2bd3ce922309169b87f`.

## 7. Lightweight verification

```bash
PYTHONPATH=external/mmdetection/projects/CO-DETR:. \
  python -m unittest discover -s tests -v
```

These tests check the fixed simplex geometry, keyframe filtering, FinalTest
sharding, and validation of the challenge's one-based category IDs.
