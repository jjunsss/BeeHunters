#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP="${1:-$ROOT/data/raw/BuzzSet_challenge.zip}"
TAR="${2:-$ROOT/data/raw/BuzzSet_challenge_testphase.tar}"
EXTRACTED="$ROOT/data/extracted"
TRAIN_MARKER="$EXTRACTED/.buzzset_challenge.complete"
TEST_MARKER="$EXTRACTED/testphase/.buzzspot_testphase.complete"

[[ -f "$ZIP" ]] || { echo "missing: $ZIP" >&2; exit 2; }
[[ -f "$TAR" ]] || { echo "missing: $TAR" >&2; exit 2; }
[[ "$(stat -c '%s' "$ZIP")" == 35261367058 ]] || { echo "bad ZIP size" >&2; exit 2; }
[[ "$(stat -c '%s' "$TAR")" == 19557529600 ]] || { echo "bad TAR size" >&2; exit 2; }
if [[ ! -f "$TRAIN_MARKER" ]]; then
  echo '964ad8ef28c52cf9e43d77ff08ad8dbc45eb56301745801b4e641ee93d3a4453  '"$ZIP" | sha256sum --check --status || { echo "bad ZIP checksum" >&2; exit 2; }
fi
if [[ ! -f "$TEST_MARKER" ]]; then
  echo '24cf58858a33b7f1d06af37cb06d155ac5a5becff1e452212b1b8740f031e846  '"$TAR" | sha256sum --check --status || { echo "bad TAR checksum" >&2; exit 2; }
fi

mkdir -p "$EXTRACTED"
if [[ ! -f "$TRAIN_MARKER" ]]; then
  unzip -q -o "$ZIP" -d "$EXTRACTED"
  [[ -d "$EXTRACTED/BuzzSet_challenge/train" ]] || { echo "train extraction incomplete" >&2; exit 2; }
  [[ -d "$EXTRACTED/BuzzSet_challenge/valid" ]] || { echo "valid extraction incomplete" >&2; exit 2; }
fi
[[ -d "$EXTRACTED/BuzzSet_challenge/train" ]] || { echo "train data missing despite marker" >&2; exit 2; }
[[ -d "$EXTRACTED/BuzzSet_challenge/valid" ]] || { echo "valid data missing despite marker" >&2; exit 2; }

if [[ ! -f "$TEST_MARKER" ]]; then
  mkdir -p "$EXTRACTED/testphase"
  tar -xf "$TAR" -C "$EXTRACTED/testphase"
  ANN="$EXTRACTED/testphase/BuzzSpot_testphase/annotations"
  [[ -f "$ANN/train.json" ]] || { echo "train annotation extraction incomplete" >&2; exit 2; }
  [[ -f "$ANN/valid.json" ]] || { echo "valid annotation extraction incomplete" >&2; exit 2; }
  [[ -f "$ANN/test_testphase.json" ]] || { echo "FinalTest annotation extraction incomplete" >&2; exit 2; }
fi
[[ -d "$EXTRACTED/testphase/BuzzSpot_testphase/test_testphase" ]] || { echo "FinalTest images missing despite marker" >&2; exit 2; }

python "$ROOT/scripts/prepare_annotations.py"
printf '%s\n' 'BuzzSet_challenge.zip verified and extracted' > "$TRAIN_MARKER"
printf '%s\n' 'BuzzSet_challenge_testphase.tar verified and extracted' > "$TEST_MARKER"
