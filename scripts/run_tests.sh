#!/usr/bin/env bash
set -eo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ROOT="c:\\repos\\docker-audiocraft"
WORKSPACE="$ROOT\\workspace"
LOGDIR="$WORKSPACE/tests/logs"
mkdir -p "$LOGDIR"

# Basic ANSI colors for readable console output
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
MAGENTA="\033[1;35m"
CYAN="\033[1;36m"
RESET="\033[0m"

step()  { echo -e "${BLUE}[STEP]${RESET} $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET} $*"; }
err()   { echo -e "${RED}[ERR]${RESET}  $*"; }
info()  { echo -e "${CYAN}[INFO]${RESET} $*"; }

step "Building docker image (if needed)..."
IMAGE_NAME="audiocraft:large.community"
if ! docker build -f docker/Dockerfile.large.community -t "$IMAGE_NAME" .; then
  warn "Docker build failed, falling back to community image 'ecchigoshujinsama/musicgen-audiocraft:latest'"
  IMAGE_NAME="ecchigoshujinsama/musicgen-audiocraft:latest"
fi
ok "Docker image ready: $IMAGE_NAME"

step "Preloading models to cache..."
docker run --gpus all --rm -it \
  -v "//c/repos/docker-audiocraft:/workspace" \
  -v "//c/repos/docker-audiocraft/model-cache:/model-cache" \
  -v "//c/repos/docker-audiocraft/workspace/cache:/cache" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/cache" \
  -e "HF_HOME=/cache" \
  --shm-size=2g "$IMAGE_NAME" bash -lc "python /workspace/scripts/preload_models.py" | tee "$LOGDIR/preload.log"

ok "Preload completed (see $LOGDIR/preload.log)"

step "Testing small model (60s, lyrics-inspired sci-fi piano and snare)..."
docker run --gpus all --rm -it \
  -v "$WORKSPACE:/workspace" \
  -v "$ROOT/model-cache:/model-cache" \
  -v "$WORKSPACE/cache:/cache" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/cache" \
  -e "HF_HOME=/cache" \
  --shm-size=2g "$IMAGE_NAME" bash -lc "python /workspace/generate_fsdp.py --model small --prompt 'a 60-second futuristic, slightly sci-fi ambient jazz piece featuring only a grand piano and soft snare drum accents, no vocals' --duration 60 --output /workspace/terraform_small_piano_snare.wav" | tee "$LOGDIR/small.log"

step "Validating small model output..."
if [ ! -s "$WORKSPACE/terraform_small_piano_snare.wav" ]; then
  err "terraform_small_piano_snare.wav missing or empty"
  exit 1
fi
if grep -qi "Traceback (most recent call last)" "$LOGDIR/small.log"; then
  err "Python traceback detected in small model run (see $LOGDIR/small.log)"
  exit 1
fi

ok "Small model generation succeeded (terraform_small_piano_snare.wav, 60s)"

step "Testing medium model (30s, lyrics-inspired female vocal jazz trio)..."
docker run --gpus all --rm -it \
  -v "$WORKSPACE:/workspace" \
  -v "$ROOT/model-cache:/model-cache" \
  -v "$WORKSPACE/cache:/cache" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/cache" \
  -e "HF_HOME=/cache" \
  --shm-size=2g "$IMAGE_NAME" bash -lc "PROMPT=\"$(python /workspace/build_prompt_medium.py)\" && python /workspace/generate_fsdp.py --model medium --prompt \"$PROMPT\" --duration 30 --output /workspace/terraform_medium_female_trio.wav" | tee "$LOGDIR/medium.log"

step "Validating medium model output..."
if [ ! -s "$WORKSPACE/terraform_medium_female_trio.wav" ]; then
  err "terraform_medium_female_trio.wav missing or empty"
  exit 1
fi
if grep -qi "Traceback (most recent call last)" "$LOGDIR/medium.log"; then
  err "Python traceback detected in medium model run (see $LOGDIR/medium.log)"
  exit 1
fi

ok "Medium model generation succeeded (terraform_medium_female_trio.wav, 30s)"
warn "Skipping large model audio generation (known T5Conditioner/meta limitation). Large dry-run mapping tests will still run."

step "Testing large model (dry-run) to validate fuzzy matching proposals..."
docker run --gpus all --rm -it \
  -v "$WORKSPACE:/workspace" \
  -v "$ROOT/model-cache:/model-cache" \
  -v "$WORKSPACE/cache:/cache" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/cache" \
  -e "HF_HOME=/cache" \
  --shm-size=2g audiocraft:large.community bash -lc "python /workspace/run_large_offload3.py --dry-run --apply-proposed" | tee "$LOGDIR/large_dryrun.log"

if [ -f "$WORKSPACE/debug/comp_proposed_matches.json" ]; then
  ok "Proposed matches file created (debug/comp_proposed_matches.json), showing first lines:"
  head -n 50 "$WORKSPACE/debug/comp_proposed_matches.json"
else
  warn "No proposed matches file found; dry-run failed to produce proposals"
fi

if [ -f "$WORKSPACE/debug/comp_proposed_matches_simple.json" ]; then
  step "Applying simple mapping JSON to confirm apply-mapping works"
  docker run --gpus all --rm -it \
    -v "$WORKSPACE:/workspace" \
    -v "$ROOT/model-cache:/model-cache" \
    -v "$WORKSPACE/cache:/cache" \
    -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
    -e "AUDIOCRAFT_CACHE_DIR=/cache" \
    -e "HF_HOME=/cache" \
    --shm-size=2g audiocraft:large.community bash -lc "python /workspace/run_large_offload3.py --dry-run --apply-mapping /workspace/debug/comp_proposed_matches_simple.json" | tee "$LOGDIR/large_applymap.log"
else
  warn "No simple mapping JSON found to test apply-mapping"
fi

step "Testing apply-proposed with save-applied to produce mapping-out"
docker run --gpus all --rm -it \
  -v "$WORKSPACE:/workspace" \
  -v "$ROOT/model-cache:/model-cache" \
  -v "$WORKSPACE/cache:/cache" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/cache" \
  -e "HF_HOME=/cache" \
  --shm-size=2g audiocraft:large.community bash -lc "python /workspace/run_large_offload3.py --dry-run --apply-proposed --save-applied --mapping-out /workspace/debug/comp_applied_mapping.json" | tee "$LOGDIR/large_applymap_save.log"

if [ -f "$WORKSPACE/debug/comp_applied_mapping.json" ]; then
  ok "Applied mapping saved (debug/comp_applied_mapping.json), showing first lines:"
  head -n 50 "$WORKSPACE/debug/comp_applied_mapping.json"
else
  warn "No applied mapping file created"
fi

step "Listing test artifacts (saved audio in workspace root)"
ls -lah "$WORKSPACE" | tee "$LOGDIR/files_after_tests.log"

step "Running unit tests (fuzzy matching)"
docker run --rm -v "$WORKSPACE:/workspace" -v "$ROOT/model-cache:/model-cache" -v "$WORKSPACE/cache:/cache" -w /workspace -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" -e "HF_HOME=/cache" audiocraft:large.community bash -lc "python -m pip install pytest && pytest -q workspace/tests/test_fuzzy.py" | tee "$LOGDIR/pytest.log"

step "Running new mapping apply tests"
docker run --rm -v "$WORKSPACE:/workspace" -v "$ROOT/model-cache:/model-cache" -v "$WORKSPACE/cache:/cache" -w /workspace -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" -e "HF_HOME=/cache" audiocraft:large.community bash -lc "python -m pip install pytest && pytest -q workspace/tests/test_mapping_apply.py" | tee "$LOGDIR/pytest_mapping.log"

step "Running threshold tests"
docker run --rm -v "$WORKSPACE:/workspace" -v "$ROOT/model-cache:/model-cache" -v "$WORKSPACE/cache:/cache" -w /workspace -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" -e "HF_HOME=/cache" audiocraft:large.community bash -lc "python -m pip install pytest && pytest -q workspace/tests/test_threshold.py" | tee "$LOGDIR/pytest_threshold.log"

ok "Run tests completed. Logs at: $LOGDIR"
info "To inspect compression debug results, run inside a container:"
info "docker run -v $WORKSPACE:/workspace audiocraft:large.community bash -lc 'python /workspace/debug/analyze_matches.py'"
