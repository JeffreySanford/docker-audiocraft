Param(
    [string]$ContainerName = "audiocraft_dev",
    [string]$Image = "audiocraft:large.community",
    [string]$HostCache = "C:\\repos\\docker-audiocraft\\model-cache",
    [string]$HostOutput = "C:\\repos\\docker-audiocraft\\output"
)

Write-Host "Recreate container '$ContainerName' from image '$Image'"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "docker CLI not found in PATH"
    exit 1
}

# Ensure host dirs exist
foreach ($d in @($HostCache, $HostOutput)) {
    if (-not (Test-Path -Path $d)) {
        Write-Host "Creating host directory: ${d}"
        try { New-Item -ItemType Directory -Path $d -Force | Out-Null }
        catch { Write-Error "Failed to create ${d}: $_"; exit 1 }
    }
}

# Stop and remove existing container if present
$existing = docker ps -a -q -f "name=^/$ContainerName$"
if ($existing) {
    Write-Host "Stopping and removing existing container: $ContainerName"
    docker rm -f $ContainerName | Out-Null
}

# Run the container with explicit Windows bind mounts so host files are visible
Write-Host "Starting container with mounts:`n  $HostCache -> /cache`n  $HostOutput -> /workspace/output"

# Compose docker run args carefully. Use ${...} around Windows paths so PowerShell doesn't misparse the ':' in paths like 'C:\'.
$args = @(
    'run', '-d', '--gpus', 'all', '--shm-size=2g', '--name', $ContainerName,
    '-e', 'AUDIOCRAFT_CACHE_DIR=/cache',
    '-v', "${HostCache}:/cache",
    '-v', "${HostOutput}:/workspace/output",
    '--entrypoint', 'tail',
    $Image,
    '-f', '/dev/null'
)

Write-Host "Running: docker $($args -join ' ')"
$proc = Start-Process -FilePath 'docker' -ArgumentList $args -NoNewWindow -Wait -PassThru
if ($proc.ExitCode -ne 0) {
    Write-Error "Failed to start container (exit $($proc.ExitCode)). Check docker logs or permissions."
    exit $proc.ExitCode
}

Write-Host "Container '$ContainerName' started. Verifying mount inside container..."
docker exec $ContainerName bash -lc "ls -la /workspace || true"

Write-Host "Done. Use 'docker logs -f $ContainerName' or 'docker exec -it $ContainerName bash' to inspect." 
