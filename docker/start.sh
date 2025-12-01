#!/usr/bin/env bash
set -euo pipefail

CMD=${1:-handler}
# Allow overriding accelerate config via env var; fallback to accelerate_config_legacy if not present
ACCEL_CONFIG=${ACCEL_CONFIG:-/workspace/accelerate_config_legacy.yaml}
ACCEL_CONFIG_FSDP_LEGACY=/workspace/accelerate_config_fsdp_legacy.yaml

case "$CMD" in
  handler)
    python -u /workspace/handler.py
    ;;
  app)
    python -u /workspace/app.py
    ;;
  accelerate)
    # Launch the accelerate example; useful for large models with offload
    # Forward additional args to the underlying script
    shift || true
    accelerate launch --config_file ${ACCEL_CONFIG} /workspace/generate_accelerate.py "$@"
    ;;
  fsdp)
    # Run the FSDP + accelerate dispatch example (use ACCEL_CONFIG if set, otherwise use fsdp config)
    shift || true
    accelerate launch --config_file ${ACCEL_CONFIG:-$ACCEL_CONFIG_FSDP_LEGACY} /workspace/generate_fsdp.py "$@"
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Usage: start.sh [handler|app|accelerate]"
    exit 1
    ;;
esac
