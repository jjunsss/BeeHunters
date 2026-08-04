# Reproduction

Run every command from the repository root. The recorded software stack is
Python 3.10, CUDA 11.8, PyTorch 2.0.1, MMCV 2.0.1, MMEngine 0.10.7, and
MMDetection 3.2.0.

## 1. Environment and data

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/download_checkpoints.sh pretrained
bash scripts/download_data.sh
bash scripts/prepare_data.sh
python scripts/build_cropbank.py
```

The prepared sources contain 6,207 original train+valid keyframes and 5,993
generated object-crop images.

## 2. Training

The clean public replay uses PyTorch distributed data parallel with one image
per GPU:

```bash
bash scripts/train.sh 1 0,1,2,3
bash scripts/train.sh 2 0,1,2,3
bash scripts/train.sh 3 0,1,2,3
```

Expected checkpoints:

```text
outputs/train_mmdet/buzzspot_stage1_trainval_12e/epoch_12.pth
outputs/train_mmdet/buzzspot_stage2_cropbank_mosaic_3e/epoch_3.pth
outputs/train_mmdet/buzzspot_stage3_raw_cooldown_2e/epoch_2.pth
```

Runtime provenance of the scored checkpoint differs only in device count: its
third-stage epoch 1 ran on four A100 80 GB GPUs, training was interrupted, and
epoch 2 resumed from `epoch_1.pth` on three GPUs. The released inference-only
checkpoint is the canonical way to reproduce the submitted predictions.

## 3. FinalTest inference

Download the released checkpoint and run FinalTest inference:

```bash
FINAL_CHECKPOINT_URL='https://github.com/jjunsss/BeeHunters/releases/download/v1.0.0/buzzspot_codino_etf_final_inference.pth' \
  bash scripts/download_checkpoints.sh final
python scripts/infer.py --gpus 0,1,2
```

To infer with a newly trained checkpoint instead:

```bash
python scripts/infer.py \
  --checkpoint outputs/train_mmdet/buzzspot_stage3_raw_cooldown_2e/epoch_2.pth \
  --gpus 0,1,2
```

Expected outputs:

```text
outputs/submissions/buzzspot_final_single_model/predictions.json
outputs/submissions/buzzspot_final_single_model/submission.zip
outputs/submissions/buzzspot_final_single_model/run_manifest.json
```

Successful packaging requires predictions for all 4,763 FinalTest keyframes.
The organizer's hidden-label evaluation of this checkpoint is
`0.5061609965` mAP@[.5:.95], rank 1; this score cannot be recomputed locally
without the hidden annotations.
