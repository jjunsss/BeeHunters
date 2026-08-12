#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:?usage: scripts/train.sh STAGE GPU_IDS}"
GPUS="${2:?usage: scripts/train.sh STAGE GPU_IDS}"
PORT="${MASTER_PORT:-29500}"

case "$STAGE" in
  1)
    CONFIG='configs/buzzspot/stage1_trainval_12e.py'
    REQUIRED='checkpoints/co_dino_5scale_swin_large_16e_o365tococo.pth'
    DATA_REQUIRED='data/extracted/testphase/BuzzSpot_testphase/annotations/trainval_keyframes.json'
    ;;
  2)
    CONFIG='configs/buzzspot/stage2_cropbank_mosaic_3e.py'
    REQUIRED='outputs/train_mmdet/buzzspot_stage1_trainval_12e/epoch_12.pth'
    DATA_REQUIRED='data/extracted/testphase/BuzzSpot_testphase/annotations/trainval_keyframes_mosaic_cropbank_only_c256.json'
    ;;
  3)
    CONFIG='configs/buzzspot/stage3_raw_cooldown_2e.py'
    REQUIRED='outputs/train_mmdet/buzzspot_stage2_cropbank_mosaic_3e/epoch_3.pth'
    DATA_REQUIRED='data/extracted/testphase/BuzzSpot_testphase/annotations/trainval_keyframes.json'
    ;;
  *) echo "stage must be 1, 2, or 3" >&2; exit 2 ;;
esac

[[ -f "$ROOT/$REQUIRED" ]] || { echo "missing prerequisite: $REQUIRED" >&2; exit 2; }
[[ -f "$ROOT/$DATA_REQUIRED" ]] || { echo "missing training data: $DATA_REQUIRED" >&2; exit 2; }
[[ -f "$ROOT/external/mmdetection/tools/train.py" ]] || { echo "run scripts/setup_env.sh first" >&2; exit 2; }
[[ -d "$ROOT/data/extracted/BuzzSet_challenge/train" ]] || { echo "missing training images" >&2; exit 2; }
[[ -d "$ROOT/data/extracted/BuzzSet_challenge/valid" ]] || { echo "missing validation images" >&2; exit 2; }
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NPROC="${#GPU_LIST[@]}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$ROOT/external/mmdetection/projects/CO-DETR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
python -m torch.distributed.run \
  --nproc_per_node="$NPROC" --master_port="$PORT" \
  external/mmdetection/tools/train.py "$CONFIG" --launcher pytorch
