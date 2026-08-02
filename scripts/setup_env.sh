#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV="${VENV:-$ROOT/.venv}"
MMDET_COMMIT="cfd5d3a985b0249de009b67d04f37263e11cdf3d"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
"$VENV/bin/python" -m pip install 'openmim==0.3.9'
"$VENV/bin/mim" install 'mmcv==2.0.1'
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"

if [[ ! -d "$ROOT/external/mmdetection/.git" ]]; then
  mkdir -p "$ROOT/external"
  git clone https://github.com/open-mmlab/mmdetection.git \
    "$ROOT/external/mmdetection"
fi

current="$(git -C "$ROOT/external/mmdetection" rev-parse HEAD)"
if [[ "$current" != "$MMDET_COMMIT" ]]; then
  if [[ -n "$(git -C "$ROOT/external/mmdetection" status --porcelain)" ]]; then
    echo "Refusing to change a modified MMDetection checkout." >&2
    exit 2
  fi
  git -C "$ROOT/external/mmdetection" fetch origin "$MMDET_COMMIT"
  git -C "$ROOT/external/mmdetection" checkout --detach "$MMDET_COMMIT"
fi

"$VENV/bin/python" - <<'PY'
import mmcv, mmdet, mmengine, torch, torchvision
print(f"torch={torch.__version__} torchvision={torchvision.__version__}")
print(f"mmcv={mmcv.__version__} mmdet={mmdet.__version__} mmengine={mmengine.__version__}")
PY
