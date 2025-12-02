# Testing Workflow

This repository includes a repeatable testing harness to validate small, medium, and large MusicGen configurations inside Docker.

## Main Test Script: `scripts/run_tests.sh`

The Bash script `scripts/run_tests.sh` performs the following steps on a Windows host (using Docker Desktop):

1. **Build the image**
   - Builds `audiocraft:large.community` from `docker/Dockerfile.large.community`.
   - Falls back to the upstream community image if the build fails.

2. **Preload models**
   - Runs `/workspace/scripts/preload_models.py` inside a GPU-enabled container.
   - Pre-downloads LM, compression, and T5 weights for small, medium, and large models.
   - Uses host-mounted caches:
     - `model-cache/`
     - `workspace/cache/`
  - Optionally, use `scripts/download_model.py <hf_repo_id>` on the host to populate `model-cache/` before running tests. The manifest `workspace/models_manifest.json` maps short keys to HF ids and cache paths used by `workspace/hf_generate.py`.

3. **Small model test (30s)**
   - Invokes `generate_fsdp.py` with:
     - `--model small`
     - `--prompt "a 30-second grand piano jazz improvisation"`
     - `--duration 30`
   - Writes `out_small.wav` to `/workspace`.
   - Validates:
     - WAV exists and is non-empty.
     - `small.log` does not contain a Python traceback.

4. **Medium model test (30s)**
   - Invokes `generate_fsdp.py` with:
     - `--model medium`
     - `--prompt "a 30-second grand piano jazz trio with bass and drums"`
     - `--duration 30`
   - Writes `out_medium.wav` to `/workspace`.
   - Performs the same file and log validation as the small test.

5. **Large model test (offloaded)**
   - Runs `run_large_offload3.py` with environment variables:
     - `AUDIOCRAFT_CACHE_DIR=/cache`
     - `HF_HOME=/cache`
     - `APPLY_PROPOSED=1` (auto-apply high-confidence fuzzy mapping proposals).
     - `FORCE_CPU_COMP=1` (keep compression on CPU even if offload is possible).
   - `run_large_offload3.py`:
     - Loads LM and compression from cached checkpoints.
     - Applies a series of mapping heuristics and optional fuzzy matching to align state dict keys.
     - Dispatches the LM across GPU and CPU using Accelerate.
     - Generates an audio clip (typically 30–60 seconds) using lyrics and style metadata when available.

6. **Dry-run and mapping tests**
   - Additional invocations of `run_large_offload3.py` in `--dry-run` mode to:
     - Generate and save proposed fuzzy matches (`comp_proposed_matches.json`).
     - Test `--apply-mapping` using `comp_proposed_matches_simple.json`.
     - Exercise `--apply-proposed` + `--save-applied` to produce `comp_applied_mapping.json`.

7. **Unit tests**
   - Runs `pytest` suites inside the container for:
     - `workspace/tests/test_fuzzy.py` (fuzzy name-matching logic).
     - `workspace/tests/test_mapping_apply.py` (mapping application semantics).
     - `workspace/tests/test_threshold.py` (auto-apply threshold behavior).

## Logs and Artifacts

All test logs are written under:

- `workspace/tests/logs/`
  - `preload.log` – model preload output.
  - `small.log`, `medium.log`, `large.log` – generation logs per size.
  - `large_dryrun.log`, `large_applymap.log`, `large_applymap_save.log` – mapping/dry-run logs.
  - `pytest*.log` – unit test runs.

Compression/Large-model debug files are written under:

- `workspace/debug/`
  - `comp_param_names.txt`
  - `comp_state_keys.txt`
  - `comp_unmatched_params.txt`
  - `comp_match_summary.json`
  - `comp_proposed_matches.json`
  - `comp_proposed_matches_simple.json`
  - `comp_applied_mapping.json` (when `--save-applied` is used)

## Running Tests Manually

From the repository root on the host (Windows, Bash):

```bash
./scripts/run_tests.sh
```

To run only the large-model portion:

```bash
./scripts/run_tests.sh large
```

(If a `large` sub-mode is not present in your local copy, you can still inspect `scripts/run_tests.sh` and extract the large-model `docker run` command for manual execution.)

## Interpreting Results

- Successful run:
  - All three WAV files are present and non-empty.
  - No `Traceback (most recent call last)` entries in small/medium logs.
  - Large-model log completes generation without meta/cuda device errors and writes a WAV under `output/`.
- On failure:
  - Inspect the corresponding log in `workspace/tests/logs/`.
  - For compression mismatches, open `workspace/debug/comp_unmatched_params.txt` and `comp_proposed_matches.json` to see which weights require manual mapping or heuristic tuning.
