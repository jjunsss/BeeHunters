#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/raw"
mkdir -p "$OUT"

download() {
  local url="$1" path="$2" bytes="$3" sha="$4"
  if [[ -f "$path" && "$(stat -c '%s' "$path")" == "$bytes" ]]; then
    if echo "$sha  $path" | sha256sum --check --status; then
      echo "ok: $path"
      return
    fi
    echo "existing archive has the wrong checksum: $path" >&2
    exit 1
  fi
  local part="$path.part"
  if command -v wget >/dev/null 2>&1; then
    wget --continue -O "$part" "$url"
  else
    curl --fail --location --continue-at - --output "$part" "$url"
  fi
  [[ "$(stat -c '%s' "$part")" == "$bytes" ]] || {
    echo "size mismatch: $part" >&2
    exit 1
  }
  echo "$sha  $part" | sha256sum --check --status || {
    echo "checksum mismatch: $part" >&2
    exit 1
  }
  mv "$part" "$path"
  echo "ok: $path"
}

download \
  'https://phenoroam.phenorob.de/file-uploader/download/public/35261367058-BuzzSet_challenge.zip' \
  "$OUT/BuzzSet_challenge.zip" 35261367058 \
  '964ad8ef28c52cf9e43d77ff08ad8dbc45eb56301745801b4e641ee93d3a4453'
download \
  'https://phenoroam.phenorob.de/file-uploader/download/public/19557529600-BuzzSet_challenge_testphase.tar' \
  "$OUT/BuzzSet_challenge_testphase.tar" 19557529600 \
  '24cf58858a33b7f1d06af37cb06d155ac5a5becff1e452212b1b8740f031e846'
