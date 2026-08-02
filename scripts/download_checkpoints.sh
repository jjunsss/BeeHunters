#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${1:-pretrained}"
mkdir -p "$ROOT/checkpoints"

case "$KIND" in
  pretrained)
    URL='https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'
    OUT="$ROOT/checkpoints/co_dino_5scale_swin_large_16e_o365tococo.pth"
    SHA='614254c94b57acbff6a4448f7aa5d6315f8483b115438796e2606ce7a62712fe'
    ;;
  final)
    URL="${FINAL_CHECKPOINT_URL:-}"
    OUT="$ROOT/checkpoints/buzzspot_codino_etf_final_inference.pth"
    SHA="${FINAL_CHECKPOINT_SHA256:-12705c0395abc7dd1d6937555e344e63d5f203e65b22d0b4e0d9f96590dfe31f}"
    ;;
  *)
    echo "usage: $0 {pretrained|final}" >&2
    exit 2
    ;;
esac

if [[ -f "$OUT" ]]; then
  echo "$SHA  $OUT" | sha256sum --check --status || {
    echo "existing checkpoint has the wrong checksum: $OUT" >&2
    exit 1
  }
  echo "ok: $OUT"
  exit 0
fi

if [[ -z "$URL" ]]; then
  echo "Set FINAL_CHECKPOINT_URL to the released inference checkpoint." >&2
  exit 2
fi

PART="$OUT.part"
if command -v wget >/dev/null 2>&1; then
  wget --continue -O "$PART" "$URL"
else
  curl --fail --location --continue-at - --output "$PART" "$URL"
fi

echo "$SHA  $PART" | sha256sum --check --status || {
  echo "checksum mismatch: $PART" >&2
  exit 1
}
mv "$PART" "$OUT"
echo "ok: $OUT"
