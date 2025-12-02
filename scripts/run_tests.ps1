param(
    [string]$Token = $env:HUGGINGFACE_HUB_TOKEN
)

if (-not $Token) {
    Write-Host "HUGGINGFACE_HUB_TOKEN not set; please provide -Token or set env var"
    exit 1
}

$root = (Split-Path -Path $MyInvocation.MyCommand.Definition -Parent) + "\.."
$workspace = Join-Path $root 'workspace'
$logdir = Join-Path $workspace 'tests\logs'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

Write-Host "Building image (if needed)..."
docker build -f docker/Dockerfile.large.community -t audiocraft:large.community .

Write-Host "Testing small model..."
docker run --gpus all --rm -it `
    -v "${workspace}:/workspace" `
    -e HUGGINGFACE_HUB_TOKEN="$Token" `
    -e AUDIOCRAFT_CACHE_DIR=/workspace/cache `
    --shm-size=2g audiocraft:large.community bash -lc "python /workspace/generate_fsdp.py --model small --prompt 'a short piano loop' --duration 6 --output /workspace/out_small.wav" | Tee-Object -FilePath (Join-Path $logdir 'small.log')

Write-Host "Testing medium model..."
docker run --gpus all --rm -it `
    -v "${workspace}:/workspace" `
    -e HUGGINGFACE_HUB_TOKEN="$Token" `
    -e AUDIOCRAFT_CACHE_DIR=/workspace/cache `
    --shm-size=2g audiocraft:large.community bash -lc "python /workspace/generate_fsdp.py --model medium --prompt 'an ambient pad' --duration 10 --output /workspace/out_medium.wav" | Tee-Object -FilePath (Join-Path $logdir 'medium.log')

Write-Host "Testing large model (offload + force CPU compression)..."
docker run --gpus all --rm -it `
    -v "${workspace}:/workspace" `
    -e HUGGINGFACE_HUB_TOKEN="$Token" `
    -e AUDIOCRAFT_CACHE_DIR=/workspace/cache `
    -e APPLY_PROPOSED=1 `
    -e FORCE_CPU_COMP=1 `
    --shm-size=2g audiocraft:large.community bash -lc "python /workspace/run_large_offload3.py" | Tee-Object -FilePath (Join-Path $logdir 'large.log')

Write-Host "Logs saved in $logdir"
