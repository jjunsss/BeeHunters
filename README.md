# BeeHunters

![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)
[![CVPPA@ECCV 2026](https://img.shields.io/badge/CVPPA%40ECCV%202026-1st%20Place-gold.svg)](https://www.codabench.org/competitions/16441/)

Official implementation of **Where is the Bee? Detecting Tiny Pollinators with
a Collaborative-Head Transformer**.

This repository reproduces our 1st-place single-model solution to the BuzzSpot
Challenge at CVPPA@ECCV 2026. The final checkpoint achieved `0.5061609965`
mAP@[.5:.95] on FinalTest.

## Method

BeeHunters uses [Co-DINO with a Swin-L backbone](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py),
initialized from the [Objects365-to-COCO checkpoint](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth), and adds an auxiliary
class-weighted fixed-simplex ETF loss adapted from [NC-FSCIL](https://openreview.net/forum?id=y5W8tpojhtJ)
on matched positive decoder queries. Training follows
three phases:

| Phase | Training source | Epochs | Learning rate |
|---|---|---:|---:|
| 1 | Raw train+valid keyframes | 12 | `1e-4` |
| 2 | Class-aware Mosaic | 3 | `1e-5` |
| 3 | Raw-keyframe cooldown | 2 | `5e-6` |

## Setup

The recorded environment uses Python 3.10 and CUDA 11.8. Run all commands from the repository root.

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/download_checkpoints.sh pretrained
```

Download and prepare the [BuzzSpot data](https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/e5fb8e49-cbdf-4846-af7d-044a92ef7fae):

```bash
bash scripts/download_data.sh
bash scripts/prepare_data.sh
python scripts/build_cropbank.py
```

The organizer archives total about 55 GB. 

## Training

Run the three phases in order.

1. **Phase 1: Raw-keyframe training (12 epochs).** Start from the pretrained
   Co-DINO checkpoint and train on the combined train and validation keyframes.

   ```bash
   bash scripts/train.sh 1 0,1,2,3
   ```

2. **Phase 2: Mosaic fine-tuning (3 epochs).** Resume Phase 1 and fine-tune on class-aware Mosaic samples.

   ```bash
   bash scripts/train.sh 2 0,1,2,3
   ```

3. **Phase 3: Raw-keyframe cooldown (2 epochs).** Resume Phase 2 and return to the raw train and validation keyframes with a `5e-6` learning rate.

   ```bash
   bash scripts/train.sh 3 0,1,2,3
   ```

## Inference

Create the FinalTest submission with a single inference pass:

```bash
python scripts/infer.py --gpus 0
```

We provide a full trained checkpoint at here. [TBD]

## Acknowledgements

Thanks to the following open-source projects for making this work possible:

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [CO-DETR](https://github.com/Sense-X/Co-DETR)
- [BuzzSpot DevKit](https://github.com/lereiss/buzzspot-devkit)
