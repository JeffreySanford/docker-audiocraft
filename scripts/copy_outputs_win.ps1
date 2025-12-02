Param(
    [string]$Container = "audiocraft_dev",
    [string]$HostOutput = "C:\\repos\\docker-audiocraft\\output"
Param(
    [string]$Container = "audiocraft_dev",
    [string]$HostOutput = "C:\\repos\\docker-audiocraft\\output"
)

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "docker CLI not found in PATH"
    exit 1
}

# Ensure host output directory exists
if (-not (Test-Path -Path $HostOutput)) {
    try { New-Item -ItemType Directory -Path $HostOutput -Force | Out-Null }
    catch { Write-Error "Failed to create host output dir ${HostOutput}: $_"; exit 1 }
}

Write-Host "Listing files in container ${Container}:/workspace/output ..."
$ls = docker exec ${Container} bash -lc "ls -A /workspace/output || true" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to list /workspace/output in container '${Container}'. Is the container running?"
    exit 1
}

$files = @()
foreach ($line in $ls -split "`n") {
    $file = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($file)) { continue }
    $files += $file
}

if ($files.Count -eq 0) {
    Write-Host "No files found in container:/workspace/output"
    exit 0
}

Write-Host "Found $($files.Count) files; copying to ${HostOutput} and overwriting if present..."

foreach ($f in $files) {
    $src = "${Container}:/workspace/output/$f"
    $dest = Join-Path -Path $HostOutput -ChildPath $f
    Write-Host "Copying: $src -> $dest"
    docker cp $src $dest
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "docker cp failed for $f - falling back to stream-copy via docker exec"
        # Fallback: use cmd.exe to run docker exec cat and redirect stdout to host file (keeps binary fidelity)
        $cmd = "docker exec ${Container} cat /workspace/output/$f > \"$dest\""
        Start-Process -FilePath cmd.exe -ArgumentList "/c", $cmd -NoNewWindow -Wait
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Fallback stream-copy also failed for $f"
        }
    }
}

Write-Host "Copy complete."
