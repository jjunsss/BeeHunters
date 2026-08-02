# BeeHunters

![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)
[![CVPPA@ECCV 2026](https://img.shields.io/badge/CVPPA%40ECCV%202026-1st%20Place-gold.svg)](https://www.codabench.org/competitions/16441/)

Official implementation of **Where's the Bee? Detecting Tiny Pollinators with
a Collaborative-Head Transformer**.

This repository reproduces our 1st-place single-model solution to the BuzzSpot
Challenge at CVPPA@ECCV 2026. The final checkpoint achieved `0.5061609965`
mAP@[.5:.95] on FinalTest without test-time augmentation or model fusion.

## Method

BeeHunters uses Co-DINO with a Swin-L backbone and an auxiliary class-weighted
fixed-simplex ETF loss on matched positive decoder queries. Training follows
three phases:

| Phase | Training source | Epochs | Learning rate |
|---|---|---:|---:|
| 1 | Raw train+valid keyframes | 12 | `1e-4` |
| 2 | Class-aware CropBank Mosaic | 3 | `1e-5` |
| 3 | Raw-keyframe cooldown | 2 | `5e-6` |

## Setup

The recorded environment uses Python 3.10 and CUDA 11.8. Run all commands from
the repository root.

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/download_checkpoints.sh pretrained
```

Download and prepare the BuzzSpot data:

```bash
bash scripts/download_data.sh
bash scripts/prepare_data.sh
python scripts/build_cropbank.py
```

The organizer archives total about 55 GB. The download scripts verify their
sizes and SHA-256 values automatically.

## Training

Run the three phases in order on four NVIDIA A100 80 GB GPUs.

```bash
bash scripts/train.sh 1 0,1,2,3
bash scripts/train.sh 2 0,1,2,3
bash scripts/train.sh 3 0,1,2,3
```

## Inference

Create the FinalTest submission with a single inference pass:

```bash
python scripts/infer.py --gpus 0
```

The inference-only checkpoint will be linked here after its public release.

## Acknowledgements

Thanks to the following open-source projects for making this work possible:

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [CO-DETR](https://github.com/Sense-X/Co-DETR)
- [BuzzSpot DevKit](https://github.com/lereiss/buzzspot-devkit)
