# docker-audiocraft

This repo contains a Docker-based set up for running Meta MusicGen (`facebook/musicgen-large`) locally and offloading with `accelerate` to avoid OOM on a single 10GB GPU.

## Quick commands (Windows with Docker Desktop / WSL)

1) Build (if you need to build locally):

```bash
docker build -f docker/Dockerfile.large.community -t audiocraft:large.community .
```

2) Run using `docker run` with cache mount and HF token (recommended):

```bash
docker run --gpus all --rm -it \
  -v "C:/repos/docker-audiocraft/workspace:/workspace" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e "AUDIOCRAFT_CACHE_DIR=/workspace/cache" \
  --shm-size=2g audiocraft:large.community bash -lc "python /workspace/run_large_offload3.py"
```

3) Or use `docker-compose` (if your Compose supports GPU pass-through):

```bash
# run
docker compose up --build
# stop
docker compose down
```

## Debug Logs
  - `comp_param_names.txt` — list of expected comp module param names
  - `comp_state_keys.txt` — list of keys in `best_state`
  - `comp_unmatched_params.txt` — final unmatched param names after heuristics
  - `comp_match_summary.json` — JSON summary with counts
  - `analyze_matches.py` — small helper script that summarizes the JSON and prints unmatched keys and sample state keys
  - `scripts/run_tests.sh` — run small, medium and large generation sequentially (captures logs)
  - `scripts/run_tests.ps1` — same for PowerShell
    - `scripts/check_results.sh` — verify generated WAV files are present and non-empty after tests
  
Run the scripts with these helpful env flags:
 - `INTERACTIVE=1` : enable prompt to accept/reject proposed fuzzy matches in the running container (useful to accept automatic suggestions safely).
 - To force compression on CPU: set `FORCE_CPU_COMP=1` env var (script will not attempt to dispatch compression to GPU).
 - `APPLY_PROPOSED=1` : when present, automatically apply all proposed fuzzy matches non-interactively (use with caution). This will attempt to assign candidate checkpoint keys to module params before attempting accelerated dispatch.
 - `APPLY_PROPOSED=1` : when present, automatically apply all proposed fuzzy matches non-interactively (use with caution). This will attempt to assign candidate checkpoint keys to module params before attempting accelerated dispatch.
 - `--proposed-threshold` : CLI option to configure the threshold for auto-applying fuzzy proposals (default 0.85). Use lower values to be more permissive.
 - `--save-applied` : CLI flag to save the mapping applied (written to `--mapping-out`, default `/workspace/debug/comp_applied_mapping.json`).
 - `--force-cpu-lm` : CLI flag to keep LM on CPU (do not dispatch to GPU) when testing or when GPU memory is insufficient.

## Model Cache Integrity

To ensure model files are not corrupted and to avoid re-downloading, the repository includes checksum verification scripts.

### Initial Setup
After models are downloaded, update the checksums:

```bash
# Using Python directly (if installed)
python scripts/check_model_checksums.py --update --cache-dir /path/to/model-cache

# Or via Docker
docker run --rm -v "C:/repos/docker-audiocraft/model-cache:/cache" audiocraft:large.community python /workspace/scripts/check_model_checksums.py --update --cache-dir /cache
```

### Monthly Verification
Run monthly to check for changes or corruption:

```bash
# PowerShell script
.\scripts\check_model_checksums.ps1 -CacheDir "C:\repos\docker-audiocraft\model-cache"

# Or Python
python scripts/check_model_checksums.py --cache-dir /path/to/model-cache
```

If checksums don't match, the script will report failures. You may need to clear the cache and re-download models.

### Scheduling Monthly Checks
On Windows, use Task Scheduler to run the PowerShell script monthly:
1. Open Task Scheduler
2. Create a new task
3. Set trigger to monthly
4. Action: Start a program
5. Program: `powershell.exe`
6. Arguments: `-File "C:\repos\docker-audiocraft\scripts\check_model_checksums.ps1" -CacheDir "C:\repos\docker-audiocraft\model-cache"`

## Notes
- If compression keys are still unmatched, inspect the logs under `/workspace/debug` and adjust heuristics in `run_large_offload2.py` and `run_large_offload3.py`.
 - If compression keys are still unmatched, inspect the logs under `/workspace/debug` and adjust heuristics in `run_large_offload3.py` (the canonical offload script).
- `AUDIOCRAFT_CACHE_DIR` is highly recommended to avoid re-downloading large hf blobs; make sure you have enough disk space.

## Documentation

Additional details about how generation and testing work are available under the `docs/` folder:

- `docs/generation.md` – overview of the MusicGen models, offload strategy, prompts, and resource utilization.
- `docs/testing.md` – end-to-end description of `scripts/run_tests.sh`, what each test does, and where logs/artifacts are stored.
