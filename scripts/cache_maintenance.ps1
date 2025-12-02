Param(
    [int]$PurgeOlderThanDays = 365,
    [string]$CachePath = 'C:\\repos\\docker-audiocraft\\model-cache'
)

<#
This script prunes cache files older than a given number of days.
Default behaviour is conservative (365 days). To run monthly pruning, call with -PurgeOlderThanDays 30

Example:
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\cache_maintenance.ps1 -PurgeOlderThanDays 30

#>

if (-not (Test-Path -Path $CachePath)) {
    Write-Host "Cache path does not exist: $CachePath. Nothing to do."
    exit 0
}

$cutoff = (Get-Date).AddDays(-$PurgeOlderThanDays)
Write-Host "Pruning files under $CachePath older than $PurgeOlderThanDays days (cutoff: $cutoff)"

$items = Get-ChildItem -Path $CachePath -Recurse -Force -File -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -lt $cutoff }
$count = 0
foreach ($it in $items) {
    try {
        Remove-Item -LiteralPath $it.FullName -Force -ErrorAction Stop
        $count++
    } catch {
        Write-Warning "Failed to remove $($it.FullName): $_"
    }
}
Write-Host "Pruned $count files."
