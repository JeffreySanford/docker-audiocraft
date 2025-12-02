# Music Generation Pipeline

This project wraps Meta's MusicGen models (`facebook/musicgen-small`, `-medium`, `-large`) in a Docker-based workflow tuned for consumer GPUs.

## Models and Modes

- `small` / `medium`: loaded via `generate_fsdp.py` using Accelerate + FSDP.
  - Run on GPU with automatic sharding/dispatch.
  - Prompts are short natural-language descriptions.
- `large`: driven by `run_large_offload3.py`.

Model manifest and local caches

- We keep a small manifest at `workspace/models_manifest.json` mapping friendly keys to HF model ids and local cache folders. Use `scripts/download_model.py <hf_repo_id>` to populate `model-cache/` and then pass `--model-key <key>` to `workspace/hf_generate.py` to load the cached model without changing scripts.
  - Language model (LM) is sharded across GPU and CPU using `accelerate.dispatch_model` and a balanced `max_memory` map.
  - Compression/codec (Encodec) is kept primarily on CPU/disk with optional offload.
  - Text conditioning uses a T5 encoder model, pinned to CPU to avoid meta-device issues, with embeddings moved to the LM device.

## Test Prompts and Durations

The default test script (`scripts/run_tests.sh`) exercises three configurations:

- **Small (30s)**: `a 30-second grand piano jazz improvisation`
- **Medium (30s)**: `a 30-second grand piano jazz trio with bass and drums`
- **Large (≈30–60s)**: jazz-oriented prompt with optional vocals, taken from `lyrics_terraform_my_heart.txt` when present.

## Offload Strategy for `musicgen-large`

- LM weights are loaded from cached checkpoints into a meta skeleton, then fully materialized and dispatched across GPU 0 and CPU.
- Compression weights are matched from the cached state dict using several heuristics (exact name, suffix, shape, and fuzzy matching), with a final fallback to `load_checkpoint_and_dispatch`.
- Environment variables:
  - `AUDIOCRAFT_CACHE_DIR`: root for audiocraft/model caches.
  - `HF_HOME`: Hugging Face cache root (Transformers model files).
  - `FORCE_CPU_COMP=1`: keep compression on CPU only.
  - `APPLY_PROPOSED=1`: auto-apply high-confidence fuzzy key matches.

## Resource Utilization

- GPU memory is managed via Accelerate's `get_balanced_memory`, using all but a small safety margin of VRAM when dispatching the LM.
- CPU parallelism is controlled by `--cpu-util-target` and `--num-threads` in `run_large_offload3.py`:
  - By default, threads are chosen as `num_threads ≈ cpu_util_target * logical_cores`.
  - Set `--cpu-util-target 0.8` to aim for ~80% CPU utilization.

## Outputs

- Generated audio files are written under `/workspace` inside the container and mapped back to the host:
  - Small: `out_small.wav`
  - Medium: `out_medium.wav`
  - Large: `output/out_large_with_lyrics.wav` (or a slugified title-based name when lyrics are present).

Listen to these WAV files after a test run to subjectively evaluate audio quality and style adherence (grand piano jazz, trio feel, and vocals where possible).