**Models**: Model management and manifest
- **Manifest file**: `workspace/models_manifest.json` — maps short keys to HF ids and local cache folder names.
- **Local cache**: put downloaded model checkpoints under the repo `model-cache/` directory (do NOT commit checkpoints to git).
- **Download helper**: `scripts/download_model.py <hf_repo_id>` — downloads model files into `model-cache/` using `huggingface_hub.snapshot_download`.

Quick actions
- Download a model locally:
```
python scripts/download_model.py facebook/musicgen-style
```
- Verify manifest and call `hf_generate.py` by key:
```
python workspace/hf_generate.py --model-key musicgen-style --prompt "..." --duration 10 --output /workspace/out.wav --style-file /workspace/style_clip.wav
```
Notes
- The repository mounts `model-cache` into Docker containers as `/model-cache` for inference.
- Large models require significant disk and possibly cloud GPUs; prefer `musicgen-small` for local 10GB GPUs.
- Always check each model's license before production use.
