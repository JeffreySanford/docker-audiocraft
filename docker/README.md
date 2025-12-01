# AudioCraft MusicGen Docker Preset (GPU + CPU)

This directory contains example Dockerfiles to run Meta's MusicGen (AudioCraft) models inside containers, and a small sample `generate.py` that generates a short audio clip.

⚠️ Notes:
- AudioCraft recommends GPU for inference. The `Dockerfile.gpu` uses `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`. Adjust if you have a different CUDA version.
- The code and models themselves are from the public AudioCraft repo by Meta. You must accept model license terms (CC-BY-NC4.0 for model weights).

Contents:
- `Dockerfile.gpu` - base on PyTorch with CUDA for GPU inference.
- `Dockerfile.cpu` - base Python image for testing (CPU-only; inference will be slow).
- `examples/generate.py` - Minimal text-to-music example using the `audiocraft` API.

Note on PyAV/FFmpeg (why we use conda)
------------------------------------
PyAV (the `av` pip package) depends on FFmpeg/libav headers and libraries at build time. If the
system-installed FFmpeg dev headers differ from the ABI expected by the PyAV/av pip wheel, pip tries
to build the package from source and compilation can fail with errors like:

  error: 'AV_CODEC_CAP_OTHER_THREADS' undeclared (first use in this function)

To avoid that, the `Dockerfile.*` in this folder uses micromamba/conda-forge to install a compatible
`ffmpeg` and `pyav` binary packages. This reduces the risk of build-time failures and keeps the image
smaller and build reproducible.

Quick start (build & run):

1) Build GPU image (requires Docker Desktop with WSL2 + GPU support or Linux with nvidia-docker installed):

```bash
# from the repo root
cd docker
docker build -f Dockerfile.gpu -t audiocraft:gpu .
```

2) Run the container with GPU support:

```bash
# Run interactively with GPU access
docker run --gpus all -it --rm -v "$PWD:/workspace" -w /workspace audiocraft:gpu bash
# Inside container:
# python examples/generate.py "80s pop track with bassy drums and synth"
```

3) For Windows 11 users:
- Install Docker Desktop and enable WSL 2 integration and the GPU under Settings > Resources > WSL Integration
- Alternatively, follow the official NVIDIA Container Toolkit instructions.

For WSL2 + Docker Desktop GPU users:
- Ensure WSL2 GPU support is enabled in Docker Desktop and that your machine's NVIDIA drivers are up-to-date.
- If you use WSL2, run Docker from a WSL2 distro's shell (Ubuntu) and map Windows paths as required. Use `--gpus all` and verify `nvidia-smi` is visible inside containers.

Running from Windows PowerShell (no WSL shell required)
-------------------------------------------------
If you prefer PowerShell / Windows, Docker Desktop still uses WSL2 under the hood for Linux containers. To verify GPU passthrough quickly:

```powershell
# checks that the NVIDIA container runtime is available inside a container
docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

To run the Gradio UI from PowerShell (mapping local workspace):

```powershell
cd C:\repos\docker-audiocraft\docker
docker compose up --build musicgen_local_gradio
```

If you run into permission issues mapping paths, use WSL2 shell or adjust Windows folder security or Docker Desktop shared drives settings.

4) Pull a community Docker image (optional):
- There are many community images on Docker Hub tagged `audiocraft` / `musicgen`. Examples:
  - `ecchigoshujinsama/musicgen-audiocraft`
  - `sxk1633/musicgen`
- Be careful running community images: examine their Dockerfiles and trust levels, and ensure license compatibility for model weights.

Tips & Troubleshooting:
- If you get errors like "CUDA not found" or "Torch not built with CUDA", check that your Docker host has GPU access and your image matches the host CUDA/runtime.
- Install/enable Docker Desktop WSL2 GPU support, or use `--gpus all` on Linux with NVIDIA Container Toolkit.
- If you need a specific CUDA version, swap the `pytorch/pytorch` base image tag.

Performance hints for heavy models (RTX 3080, 10 GB VRAM):

- Use `facebook/musicgen-small` or `facebook/musicgen-medium` for best success on 10GB GPUs. `facebook/musicgen-large` (3.3B) is likely to exceed 10GB VRAM without CPU offload or FSDP/accelerate tricks.
- Use FP16 (`.half()` or torch.autocast) to reduce VRAM usage and speed up inference.
- Consider `accelerate` with CPU offload and sharded strategies if you want to try loading large models (slower but may reduce VRAM requirements).
- Use `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` environment variable and `torch.cuda.set_per_process_memory_fraction(0.9)` in Python to help control fragmentation.
- Expected generation times (rough guideline), 4s audio:
  - small: ~2-4s
  - medium: ~5-12s depending on cold start and whether caching is used
  - large: may not fit on 10GB, expect much longer times if using CPU offload.

If you see errors about building `av` or `AV_CODEC_CAP_OTHER_THREADS` during `pip install` or `docker build`, this generally means the version of FFmpeg installed system-wide doesn't match what PyAV expected at build-time. Recommended fixes:

- Use the community image (it includes working prebuilt binaries) `ecchigoshujinsama/musicgen-audiocraft`.
- Build using the provided `Dockerfile.gpu` which uses conda-forge binary packages for `ffmpeg` and `pyav` to avoid compile-time mismatches.
- As an alternative you can install FFmpeg + pyav via `conda`/`mamba` in your image or use an image like `conda/miniconda3` and then `pip install av`.

Example: Use the container and run the script with a custom prompt:

```bash
# Linux or WSL:
docker run --gpus all -it --rm -v "$PWD:/workspace" -w /workspace audiocraft:gpu bash -c "python examples/generate.py 'happy EDM with bass'"
```

For advanced users, consider implementing a FastAPI/Gradio server inside the container to expose endpoints for generation, or extend the image with Hugging Face credentials if you need to pull models from the Hugging Face Hub using `huggingface-cli login`.

License & model weights:
- The AudioCraft code is MIT-licensed, model weights are generally CC-BY-NC 4.0. This repository only provides Docker examples, not the weights themselves.

Using a community image (recommended for quick start)
-----------------------------------------------

If you don't want to build the image locally, you can use a community image that bundles AudioCraft and dependencies (we verified `ecchigoshujinsama/musicgen-audiocraft`). This is faster and reliable for testing, but double-check the Dockerfile and licensing for your use:

```bash
# Pull the image
docker pull ecchigoshujinsama/musicgen-audiocraft:latest

# Run a quick generation that saves out.wav in the repo docker folder (assuming Windows path to the repo)
docker run --rm --gpus all -it -v "C:/repos/docker-audiocraft/docker:/workspace" ecchigoshujinsama/musicgen-audiocraft:latest bash -lc "python3 -c 'from audiocraft.models import MusicGen; m=MusicGen.get_pretrained(\"facebook/musicgen-small\"); m.set_generation_params(duration=4); wav=m.generate([\"a calm piano loop\"]); import torchaudio; torchaudio.save(\"/workspace/out.wav\", wav[0].cpu(), m.sample_rate); print(\"Saved /workspace/out.wav\")'"
```

To run the prebuilt service (via `docker-compose`), set `HUGGINGFACE_TOKEN` if you need to pull private models, then run:

```bash
cd docker
docker compose up --build

If you prefer to run the local-built image rather than the community one, use:

```bash
# build and run the local GPU image (uses Dockerfile.gpu)
docker compose up --build --no-deps musicgen_local

Run the Gradio UI (local build):

```bash
docker compose up --build musicgen_local_gradio
```

Run the accelerate demo (local build):

```bash
docker compose up --build musicgen_local_accelerate

Accelerate FSDP/CPU offload example (large model)
------------------------------------------------
To attempt running `facebook/musicgen-large` (3.3B) you can use accelerate with FSDP/CPU offload using the
`accelerate_config.yaml` provided. This will cause the model to be sharded and offloaded to CPU/SSD when GPU memory is insufficient.

- Ensure you map an offload folder in compose or via `-v ./offload:/workspace/offload`.
- Start the container with the `musicgen_local_accelerate` compose service or run the start script:

```bash
docker compose up --build musicgen_local_accelerate
# Alternatively (more interactive), run:
docker run --gpus all -it --rm -v "$PWD:/workspace" -v "$PWD/offload:/workspace/offload" audiocraft:gpu /workspace/start.sh accelerate --model large --prompt "a moody synth loop" --duration 4
```

- Notes:
- This approach uses accelerate's FSDP/auto device mapping to potentially split the model across devices and offload parameters to CPU. It requires substantial CPU RAM and may be slower but helps run large models on limited GPU VRAM.

Best practices & troubleshooting for FSDP & offload:
- Confirm you have enough CPU RAM and disk space for offload; large models may use tens of GBs of RAM or disk while sharding.
- Use an `offload` folder (mapped as `./offload:/workspace/offload`) and make sure it's on an SSD if possible to avoid slow disk I/O.
- Choose your `accelerate_config_fsdp.yaml` for multi-GPU/`fsdp_config` if you have multiple GPUs; for single-GPU use `accelerate_config_offload.yaml`.
- If you get an error about missing FSDP features, ensure your PyTorch/accelerate/deepspeed versions are compatible and that the base image includes built PyTorch with FSDP support.
- You may need to adjust `accelerate_config.yaml` (increase num_processes, set `offload.device: cpu`, `offload.offload_folder`, etc).

Using the FSDP example (targeting `facebook/musicgen-large`):

To explicitly try to run the FSDP/dispatch flow use:

```bash
docker compose run --rm -v "$PWD:/workspace" -v "$PWD/offload:/workspace/offload" musicgen_local_accelerate /workspace/start.sh fsdp --model large --prompt "a moody synth loop" --duration 4

Run a one-off medium model generation (quick check)
------------------------------------------------
If you'd like to just verify the medium model generation (a quick smoke test) run:

```bash
cd docker
docker compose run --rm -v "$PWD:/workspace" -v "$PWD/cache:/workspace/cache" musicgen_local python /workspace/generate_test_medium.py
```


```bash
docker run --gpus all -it --rm -v "$PWD:/workspace" audiocraft:gpu python /workspace/generate_test_medium.py
```

Or with compose service (recommended to avoid re-building your main local image):

```bash
cd docker
docker compose up --build --no-deps musicgen_local_test_medium
```

This runs `generate_test_medium.py` via the `musicgen_local_test_medium` service and saves `out_medium.wav` to the repo `docker` folder.
```

This command will run the accelerate-dispatch-based `generate_fsdp.py` which attempts to load the large model via `init_empty_weights()` and `load_checkpoint_and_dispatch()`.
It is a best-effort approach; results vary by host memory and the availability of accelerate + FSDP features.
```
```
```

The service will run the `handler.py` script inside the container — modify the `command` in `docker-compose.yml` to run demos or a web UI instead.
