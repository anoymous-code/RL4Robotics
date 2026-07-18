# Relay pipeline: noisy-expert collection -> retrain -> eval
$py = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"

Write-Host "=== [1/3] collect 150 noisy-expert demos ==="
& $py (Join-Path $PSScriptRoot "collect_demos.py") --n 150 --seed 2 --start 200
if ($LASTEXITCODE -ne 0) { Write-Host "collect exited $LASTEXITCODE (continuing with whatever data exists)" }

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
