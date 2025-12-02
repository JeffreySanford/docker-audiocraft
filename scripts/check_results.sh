#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKSPACE="$ROOT/workspace"

echo "Checking for output files in $WORKSPACE"
declare -a files=(
  "$WORKSPACE/out_small.wav"
  "$WORKSPACE/out_medium.wav"
  "$WORKSPACE/out_large_offloaded3.wav"
  "$WORKSPACE/out_large_rock.wav"
)

missing=0
for f in "${files[@]}"; do
  if [ -f "$f" ] && [ -s "$f" ]; then
    echo "OK: $f exists and non-empty"
  else
    echo "MISSING or EMPTY: $f"
    missing=1
  fi
done

if [ "$missing" -eq 1 ]; then
  echo "Some outputs missing or empty"
  exit 1
else
  echo "All outputs present"
  exit 0
fi
