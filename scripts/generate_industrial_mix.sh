#!/usr/bin/env bash
set -eo pipefail

# Root paths (host side, assuming Git Bash on Windows)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKSPACE="$ROOT/workspace"
LOGDIR="$WORKSPACE/tests/logs"
mkdir -p "$LOGDIR"

IMAGE_NAME="audiocraft:large.community"

RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
CYAN="\033[1;36m"
RESET="\033[0m"

step()  { echo -e "${BLUE}[STEP]${RESET} $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET} $*"; }
err()   { echo -e "${RED}[ERR]${RESET}  $*"; }
info()  { echo -e "${CYAN}[INFO]${RESET} $*"; }

step "Building docker image for industrial mix if needed..."
docker build -f "$ROOT/docker/Dockerfile.large.community" -t "$IMAGE_NAME" "$ROOT" >/dev/null
ok "Docker image ready: $IMAGE_NAME"

step "Generating 5 x 60s small-model industrial ambience tracks..."
SMALL_OUT_DIR="$WORKSPACE/output_industrial_small"
MED_OUT_DIR="$WORKSPACE/output_industrial_medium"
mkdir -p "$SMALL_OUT_DIR" "$MED_OUT_DIR"

# Build a small one-line Python helper to print the style strings
SMALL_STYLE_CMD="from industrial_styles import SMALL_INDUSTRIAL_STYLE; print(SMALL_INDUSTRIAL_STYLE)"
MEDIUM_STYLE_CMD="from industrial_styles import MEDIUM_INDUSTRIAL_STYLE; print(MEDIUM_INDUSTRIAL_STYLE)"

for i in 1 2 3 4 5; do
  out_file="/workspace/output_industrial_small/industrial_small_${i}.wav"
  info "Small track $i -> $(basename "$out_file")"
  docker run --gpus all --rm -it \
    -v "//$ROOT/workspace:/workspace" \
    -v "//$ROOT/model-cache:/model-cache" \
    -v "//$ROOT/workspace/cache:/cache" \
    -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
    -e "AUDIOCRAFT_CACHE_DIR=/cache" \
    -e "HF_HOME=/cache" \
    --shm-size=2g "$IMAGE_NAME" bash -lc "STYLE=\$(python -c \"$SMALL_STYLE_CMD\"); \
      python /workspace/generate_fsdp.py \
      --model small \
      --prompt \"\$STYLE\" \
      --duration 60 \
      --output $out_file" | tee "$LOGDIR/industrial_small_${i}.log"

done

step "Generating 3 x 30s medium-model heavy industrial dubstep transitions..."
for i in 1 2 3; do
  out_file="/workspace/output_industrial_medium/industrial_medium_${i}.wav"
  info "Medium track $i -> $(basename "$out_file")"
  if [ -n "$HF_MODEL_ID" ]; then
    info "Using HF model: $HF_MODEL_ID for medium track $i"
    docker run --gpus all --rm -it \
      -v "//$ROOT/workspace:/workspace" \
      -v "//$ROOT/model-cache:/model-cache" \
      -v "//$ROOT/workspace/cache:/cache" \
      -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
      -e "AUDIOCRAFT_CACHE_DIR=/cache" \
      -e "HF_HOME=/cache" \
      --shm-size=2g "$IMAGE_NAME" bash -lc "STYLE=\$(python -c \"$MEDIUM_STYLE_CMD\"); \
        python /workspace/hf_generate.py --model-id \"$HF_MODEL_ID\" --prompt \"\$STYLE\" --duration 30 --output $out_file" | tee "$LOGDIR/industrial_medium_${i}.log"
  else
    docker run --gpus all --rm -it \
      -v "//$ROOT/workspace:/workspace" \
      -v "//$ROOT/model-cache:/model-cache" \
      -v "//$ROOT/workspace/cache:/cache" \
      -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
      -e "AUDIOCRAFT_CACHE_DIR=/cache" \
      -e "HF_HOME=/cache" \
      --shm-size=2g "$IMAGE_NAME" bash -lc "STYLE=\$(python -c \"$MEDIUM_STYLE_CMD\"); \
        python /workspace/generate_fsdp.py \
        --model medium \
        --prompt \"\$STYLE\" \
        --duration 30 \
        --output $out_file" | tee "$LOGDIR/industrial_medium_${i}.log"
  fi

done

step "Stitching all clips into one continuous industrial mix with ffmpeg..."
FINAL_OUT="$WORKSPACE/industrial_mix_full.wav"
TMP_LIST="$WORKSPACE/output_industrial_concat_list.txt"

# Build concat list in desired order: S1, M1, S2, M2, S3, M3, S4, S5
cat > "$TMP_LIST" <<EOF
file 'output_industrial_small/industrial_small_1.wav'
file 'output_industrial_medium/industrial_medium_1.wav'
file 'output_industrial_small/industrial_small_2.wav'
file 'output_industrial_medium/industrial_medium_2.wav'
file 'output_industrial_small/industrial_small_3.wav'
file 'output_industrial_medium/industrial_medium_3.wav'
file 'output_industrial_small/industrial_small_4.wav'
file 'output_industrial_small/industrial_small_5.wav'
EOF

# Use ffmpeg inside a container to avoid host dependencies
step "Running ffmpeg concat..."
docker run --rm -v "//$ROOT/workspace:/workspace" "$IMAGE_NAME" bash -lc "ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i /workspace/$(basename "$TMP_LIST") -c copy /workspace/$(basename "$FINAL_OUT")" || {
  err "ffmpeg stitching failed"
  exit 1
}

ok "Industrial mix created at: $FINAL_OUT"
info "Small stems in: $SMALL_OUT_DIR"
info "Medium stems in: $MED_OUT_DIR" 
