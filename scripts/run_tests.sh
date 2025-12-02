#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKSPACE="$ROOT/workspace"
LOGDIR="$WORKSPACE/tests/logs"
mkdir -p "$LOGDIR"

echo "Building docker image (if needed)..."
IMAGE_NAME="audiocraft:large.community"
if ! docker build -f docker/Dockerfile.large.community -t "$IMAGE_NAME" .; then
  echo "Docker build failed, falling back to community image 'ecchigoshujinsama/musicgen-audiocraft:latest'"
  IMAGE_NAME="ecchigoshujinsama/musicgen-audiocraft:latest"
fi

echo "Testing small model (6s)..."
docker run --gpus all --rm -it \
  -v "$WORKSPACE:/workspace" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/workspace/cache" \
  --shm-size=2g "$IMAGE_NAME" bash -lc "python /workspace/generate_fsdp.py --model small --prompt 'a short piano loop' --duration 6 --output /workspace/out_small.wav" | tee "$LOGDIR/small.log"

echo "Testing medium model (10s)..."
docker run --gpus all --rm -it \
  -v "$WORKSPACE:/workspace" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/workspace/cache" \
  --shm-size=2g "$IMAGE_NAME" bash -lc "python /workspace/generate_fsdp.py --model medium --prompt 'a soft ambient pad' --duration 10 --output /workspace/out_medium.wav" | tee "$LOGDIR/medium.log"

echo "Testing large model (15s) with offload..."
docker run --gpus all --rm -it \
  -v "$WORKSPACE:/workspace" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/workspace/cache" \
  -e "APPLY_PROPOSED=1" \
  -e "FORCE_CPU_COMP=1" \
  --shm-size=2g "$IMAGE_NAME" bash -lc "python /workspace/run_large_offload3.py" | tee "$LOGDIR/large.log"

echo "Testing large model (dry-run) to validate fuzzy matching proposals..."
docker run --gpus all --rm -it \
  -v "$WORKSPACE:/workspace" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/workspace/cache" \
  --shm-size=2g audiocraft:large.community bash -lc "python /workspace/run_large_offload3.py --dry-run --apply-proposed" | tee "$LOGDIR/large_dryrun.log"

if [ -f "$WORKSPACE/debug/comp_proposed_matches.json" ]; then
  echo "Proposed matches file created:"
  head -n 50 "$WORKSPACE/debug/comp_proposed_matches.json"
else
  echo "No proposed matches file found; dry-run failed to produce proposals"
fi

if [ -f "$WORKSPACE/debug/comp_proposed_matches_simple.json" ]; then
  echo "Applying simple mapping JSON to confirm apply-mapping works"
  docker run --gpus all --rm -it \
    -v "$WORKSPACE:/workspace" \
    -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
    -e "AUDIOCRAFT_CACHE_DIR=/workspace/cache" \
    --shm-size=2g audiocraft:large.community bash -lc "python /workspace/run_large_offload3.py --dry-run --apply-mapping /workspace/debug/comp_proposed_matches_simple.json" | tee "$LOGDIR/large_applymap.log"
else
  echo "No simple mapping JSON found to test apply-mapping"
fi

echo "Testing apply-proposed with save-applied to produce mapping-out"
docker run --gpus all --rm -it \
  -v "$WORKSPACE:/workspace" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/workspace/cache" \
  --shm-size=2g audiocraft:large.community bash -lc "python /workspace/run_large_offload3.py --dry-run --apply-proposed --save-applied --mapping-out /workspace/debug/comp_applied_mapping.json" | tee "$LOGDIR/large_applymap_save.log"

if [ -f "$WORKSPACE/debug/comp_applied_mapping.json" ]; then
  echo "Applied mapping saved at:"
  head -n 50 "$WORKSPACE/debug/comp_applied_mapping.json"
else
  echo "No applied mapping file created"
fi

echo "Test artifacts (saved audio)"
ls -lah "$WORKSPACE" | tee "$LOGDIR/files_after_tests.log"

echo "Running unit tests (fuzzy matching)"
docker run --rm -v "$WORKSPACE:/workspace" -w /workspace -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" audiocraft:large.community bash -lc "python -m pip install pytest && pytest -q workspace/tests/test_fuzzy.py" | tee "$LOGDIR/pytest.log"

echo "Running new mapping apply tests"
docker run --rm -v "$WORKSPACE:/workspace" -w /workspace -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" audiocraft:large.community bash -lc "python -m pip install pytest && pytest -q workspace/tests/test_mapping_apply.py" | tee "$LOGDIR/pytest_mapping.log"

echo "Running threshold tests"
docker run --rm -v "$WORKSPACE:/workspace" -w /workspace -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" audiocraft:large.community bash -lc "python -m pip install pytest && pytest -q workspace/tests/test_threshold.py" | tee "$LOGDIR/pytest_threshold.log"

echo "Run tests completed. Logs at: $LOGDIR"
echo "Use 'docker run -v $WORKSPACE:/workspace audiocraft:large.community bash -lc "python /workspace/debug/analyze_matches.py"' to inspect compression debug results"
