param(
    [string]$Token = $env:HUGGINGFACE_HUB_TOKEN
)

if (-not $Token) {
    Write-Host "HUGGINGFACE_HUB_TOKEN is required; pass -Token or set env var"
    exit 1
}

docker run --gpus all --rm -it `
    -v "${PWD}\workspace:/workspace" `
    -e HUGGINGFACE_HUB_TOKEN="$Token" `
    -e AUDIOCRAFT_CACHE_DIR=/workspace/cache `
    --shm-size=2g audiocraft:large.community bash -lc "python /workspace/run_large_offload3.py"
