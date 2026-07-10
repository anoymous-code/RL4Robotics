# 下载实验所需的第三方模型库（third_party/ 不入 git 库）
# 用法: powershell -File scripts\download_assets.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $root 'third_party\mujoco_menagerie'

if (Test-Path (Join-Path $dest 'aloha\aloha.xml')) {
    Write-Host 'mujoco_menagerie 已存在，跳过下载'
    exit 0
}

git clone --filter=blob:none --sparse --depth 1 `
    https://github.com/google-deepmind/mujoco_menagerie.git $dest
if ($LASTEXITCODE -ne 0) {
    Write-Host '直连失败，尝试本地代理 127.0.0.1:7890 ...'
    git -c http.proxy=http://127.0.0.1:7890 clone --filter=blob:none --sparse --depth 1 `
        https://github.com/google-deepmind/mujoco_menagerie.git $dest
}
git -C $dest sparse-checkout set aloha
Write-Host '完成：third_party\mujoco_menagerie\aloha'
