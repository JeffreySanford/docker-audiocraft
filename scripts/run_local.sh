#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_local.sh
# Expects HUGGINGFACE_HUB_TOKEN in environment (export HUGGINGFACE_HUB_TOKEN=...) and Docker Desktop with GPU access
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
echo "Starting docker run with workspace mounted from $ROOT_DIR/workspace"
docker run --gpus all --rm -it \
  -v "$ROOT_DIR/workspace:/workspace" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e AUDIOCRAFT_CACHE_DIR=/workspace/cache \
  --shm-size=2g audiocraft:large.community bash -lc "python /workspace/run_large_offload3.py"
