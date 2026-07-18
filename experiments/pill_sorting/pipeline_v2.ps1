# Relay pipeline: noisy-expert collection -> retrain -> eval
param(
    [int]$CollectN = 85,
    [int]$CollectStart = 267,
    [int]$CollectSeed = 3
)
$py = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"

Write-Host "=== [1/3] collect $CollectN noisy-expert demos (start=$CollectStart) ==="
& $py (Join-Path $PSScriptRoot "collect_demos.py") --n $CollectN --seed $CollectSeed --start $CollectStart
if ($LASTEXITCODE -ne 0) { Write-Host "collect exited $LASTEXITCODE (continuing with existing data)" }

Write-Host "=== [2/3] retrain ACT (20k steps) ==="
for ($try = 1; $try -le 6; $try++) {
    & $py (Join-Path $PSScriptRoot "train_act.py") --steps 20000 --batch 16
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host "train attempt $try failed (exit $LASTEXITCODE); retry in 3 min"
    Start-Sleep -Seconds 180
}

Write-Host "=== [3/3] evaluate 20 rollouts ==="
& $py (Join-Path $PSScriptRoot "eval_act.py") --n 20 --video
Write-Host "PIPELINE DONE"
