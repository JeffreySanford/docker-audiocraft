# Audio Generation with MusicGen

This document describes how we use Audiocraft/MusicGen in Docker to generate industrial, jazz, ambient, and experimental tracks on a single NVIDIA RTX 3080.

## Goals

- Run `musicgen-small` and `musicgen-medium` reliably in Docker.
- Generate 30–180 second tracks with stable prompts and repeatable results.
- Use rich, verbose text prompts to steer style (house, industrial, jazz, ambience).
- Keep GPU memory usage under control on a 10 GB card.

## Environment Overview

- **Hardware:** NVIDIA RTX 3080 (10 GB VRAM), Windows host.
- **Containerization:** Docker Desktop with `--gpus all`.
- **Repository:** `docker-audiocraft` (this repo).
- **Image:** `audiocraft:large.community`, built from `docker/Dockerfile.large.community`.
- **Model caches:**
  - `model-cache/` on the host, mounted into the container.
  - Within containers, `HF_HOME=/cache` and `AUDIOCRAFT_CACHE_DIR=/cache`.

## Core Commands

The main script used for music generation is `workspace/generate_fsdp.py`. Within Docker, we typically run commands like:

```bash
python /workspace/generate_fsdp.py \
  --model small \
  --prompt "<long descriptive style text>" \
  --duration 60 \
  --output /workspace/output/track.wav
```

Key flags:

- `--model small|medium|large`: selects MusicGen checkpoint.
- `--prompt`: natural-language description of the desired audio.
- `--duration`: length in seconds.
- `--output`: absolute path to the output WAV in the container.

## Prompting Strategy

Prompt quality matters more than parameter tweaking for the small and medium models. Good prompts:

- Are **long and descriptive** (multiple sentences or a full paragraph).
- Mention **tempo or feel** (e.g. “around 128 BPM, 4-on-the-floor kick”).
- Specify **genre** and **sub-genre** (e.g. “industrial house / EBM with KMFDM-like aggression”).
- Describe **instrumentation** (drums, bass, synths, vocals, ambience).
- Clarify **what not to include** (e.g. “no vocals, instrumental only”).

We keep these prompts in code or in helper modules (see `workspace/industrial_styles.py`) so they can be reused consistently across runs.

## Duration vs. GPU Constraints

On a 3080:

- **Small and medium models** are comfortable up to ~60 seconds per generation.
- 90–120 seconds is possible but can approach the limits when decoding audio (Encodec stage).
- **Large model** is significantly more demanding and prone to OOMs and internal T5 issues; we treat it as experimental.

By default we:

- Use 60 seconds for most small-model stems.
- Use 30 seconds for medium-model transitions.

This balances quality with reliability on a 10 GB card.

## Testing Flows

We use `scripts/run_tests.sh` to:

- Build the Docker image.
- Preload caches.
- Generate test clips for small and medium models.
- Run basic Python tests.

This ensures changes to prompts or configs do not silently break the generation workflow.

## Takeaways

- Keep prompts long and explicit.
- Prefer small/medium models for reliability on consumer GPUs.
- Use Docker to freeze dependencies and CUDA stack.
- Treat large model experiments as optional and best suited for bigger GPUs or cloud.
