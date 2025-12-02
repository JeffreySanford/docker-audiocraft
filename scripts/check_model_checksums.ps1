# Model Cache Integrity Check Script
# Run this script monthly to verify model checksums

param(
    [string]$CacheDir = "C:\repos\docker-audiocraft\model-cache",
    [switch]$Update
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ScriptDir
$ChecksumScript = Join-Path $RepoDir "scripts\check_model_checksums.py"

# Check if Python is available
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $pythonCmd) {
    Write-Error "Python not found. Please install Python 3."
    exit 1
}

# Build the command
$scriptArgs = @($ChecksumScript, "--cache-dir", $CacheDir)
if ($Update) {
    $scriptArgs += "--update"
}

# Run the checksum script
Write-Host "Running model checksum check..."
Write-Host "Command: $($pythonCmd.Source) $($scriptArgs -join ' ')"

try {
    & $pythonCmd.Source @scriptArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Checksum check completed successfully."
    } else {
        Write-Warning "Checksum check found issues."
        exit $LASTEXITCODE
    }
} catch {
    Write-Error "Failed to run checksum script: $_"
    exit 1
}