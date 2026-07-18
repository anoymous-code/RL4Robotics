# Relay: wait for collector process to exit, then start ACT training.
# Retries when torch DLL load fails due to pagefile pressure.
param(
    [int]$CollectPid = 0,
    [int]$Steps = 20000,
    [int]$Batch = 16
)

if ($CollectPid -gt 0) {
    Write-Host "waiting for collector pid $CollectPid ..."
    while (Get-Process -Id $CollectPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
    Write-Host "collector finished; starting training in 10 s"
    Start-Sleep -Seconds 10
}

$py = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
for ($try = 1; $try -le 12; $try++) {
    Write-Host "training attempt $try : steps=$Steps batch=$Batch"
    & $py (Join-Path $PSScriptRoot "train_act.py") --steps $Steps --batch $Batch
    if ($LASTEXITCODE -eq 0) {
        Write-Host "TRAINING DONE"
        exit 0
    }
    Write-Host "start failed (exit $LASTEXITCODE, likely pagefile); retry in 5 min"
    Start-Sleep -Seconds 300
}
Write-Host "GIVING UP after retries"
exit 1
