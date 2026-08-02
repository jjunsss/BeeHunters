# BeeHunters

Official implementation of **Where Is the Bee? Detecting Tiny Pollinators with
a Collaborative-Head Transformer**.

[![Public repository checks](https://github.com/jjunsss/BeeHunters/actions/workflows/quality.yml/badge.svg)](https://github.com/jjunsss/BeeHunters/actions/workflows/quality.yml)

This repository reproduces the single-model system used for our rank-1 entry
in the CVPPA@ECCV 2026 BuzzSpot challenge. The final checkpoint reached
`0.5061609965` mAP@[.5:.95] on FinalTest without test-time augmentation or
model fusion.

The paper and arXiv sources are managed separately. This repository contains
only the executable training and inference path.

## Method at a glance

```text
Objects365-to-COCO Co-DINO/Swin-L initialization
  -> 12 epochs on 6,207 raw train+valid keyframes
  -> 3 epochs on class-aware CropBank Mosaic
  -> 2 epochs on raw train+valid keyframes at low learning rate
  -> one checkpoint, one inference pass
```

The detector keeps Co-DINO's standard prediction path. During training, the
matched positive decoder queries receive an additional class-weighted loss
toward fixed simplex ETF targets.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/buzzspot/` | Final 12E -> 3E -> 2E configuration chain |
| `projects/buzzspot/` | Fixed-ETF loss for matched positive queries |
| `scripts/` | Environment, data, checkpoint, training, and inference tools |
| `docs/REPRODUCTION.md` | Full commands, expected counts, hashes, and output paths |
| `docs/REPOSITORY_MAP.md` | Public-release versus local-artifact boundaries |
| `docs/GITHUB_RELEASE_CHECKLIST.md` | Remaining gates before the first public push |
| `checkpoints/manifest.json` | Machine-readable checkpoint identities |
| `tests/` | Data, ETF geometry, and inference utility checks |

The tracked release is intentionally small. Datasets, checkpoints, experiment
outputs, historical code, and paper-working files stay outside Git history.
See [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) for the exact boundary.

## Quick start

Run all commands from the repository root. The recorded environment uses
Python 3.10 and CUDA 11.8.

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/download_checkpoints.sh pretrained
```

Download the organizer archives, then generate the keyframe annotations and
the deterministic CropBank:

```bash
bash scripts/download_data.sh
bash scripts/prepare_data.sh
python scripts/build_cropbank.py
```

Train the three stages:

```bash
bash scripts/train.sh 1 0,1,2,3
bash scripts/train.sh 2 0,1,2,3
bash scripts/train.sh 3 0,1,2
```

Create the FinalTest submission:

```bash
python scripts/infer.py --gpus 0,1,2
```

Alternatively, download the released inference-only checkpoint and skip
training:

```bash
FINAL_CHECKPOINT_URL='<released-checkpoint-url>' \
  bash scripts/download_checkpoints.sh final
python scripts/infer.py \
  --checkpoint checkpoints/buzzspot_codino_etf_final_inference.pth \
  --gpus 0,1,2
```

See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) before running the full
pipeline. The dataset archives total about 55 GB and training was performed on
NVIDIA A100 80 GB GPUs.

## Verification

After environment setup:

```bash
PYTHONPATH=external/mmdetection/projects/CO-DETR:. \
  python -m unittest discover -s tests -v
python -m pip check
```

The repository-level release check does not require CUDA or third-party Python
packages:

```bash
python scripts/check_public_repo.py
```

The code package and CI checks are prepared, but publication still requires a
repository license and a public URL for the final inference checkpoint. Track
those two release gates in
[`docs/GITHUB_RELEASE_CHECKLIST.md`](docs/GITHUB_RELEASE_CHECKLIST.md).

## Scope

This release contains only the final train+valid recipe. The historical June
28 WBF submission, TTA runs, pseudo-label runs, detector sweeps, datasets,
training outputs, and paper-working files are intentionally excluded.

The implementation builds on MMDetection and its CO-DETR project. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream and dataset
boundaries.
