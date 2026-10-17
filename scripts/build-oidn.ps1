# Copyright (c) 2026 jesus luque.
#
# Open Image Denoise for athenea on Windows, GPU devices only: the counterpart
# of build-oidn.sh. CUDA is the device, as on Linux: the engine's own device on
# an NVIDIA card. Run from PowerShell:
#   scripts\build-oidn.ps1 [-Version 2.5.1]
# The prefix is $env:ATHENEA_OIDN_ROOT, or D:\tools\oidn-<version>; TBB comes
# from the OpenUSD install ($env:ATHENEA_USD_ROOT, or D:\tools\usd-26.08-mx), so
# the process holds one TBB.

param([string]$Version = "2.5.1")
# Native programs are judged by their exit code: under "Stop", Windows
# PowerShell 5.1 takes anything they write to stderr (git's progress) as a
# failure.
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "msvc-env.ps1")

$prefix = if ($env:ATHENEA_OIDN_ROOT) { $env:ATHENEA_OIDN_ROOT } else { "D:\tools\oidn-$Version" }
$srcRoot = if ($env:ATHENEA_OIDN_SRC) { $env:ATHENEA_OIDN_SRC } else { "D:\tools\src" }
$usdRoot = if ($env:ATHENEA_USD_ROOT) { $env:ATHENEA_USD_ROOT } else { "D:\tools\usd-26.08-mx" }
$jobs = if ($env:JOBS) { $env:JOBS } else { $env:NUMBER_OF_PROCESSORS }
$src = Join-Path $srcRoot "oidn-$Version"

if (-not (Test-Path $src)) {
    New-Item -ItemType Directory -Force -Path $srcRoot | Out-Null
    $archive = Join-Path $srcRoot "oidn-$Version.src.tar.gz"
    curl.exe -L --fail -o $archive "https://github.com/RenderKit/oidn/releases/download/v$Version/oidn-$Version.src.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw "download failed" }
    tar -xzf $archive -C $srcRoot
    if ($LASTEXITCODE -ne 0) { throw "unpacking failed" }
}
$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { throw "CUDA_PATH is not set: install the CUDA toolkit" }
Write-Host "OIDN: CUDA toolkit at $cuda"
$nvcc = (Join-Path $cuda "bin\nvcc.exe") -replace "\\", "/"

cmake -S $src -B "$src\build" -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_INSTALL_PREFIX=$prefix" `
    "-DTBB_ROOT=$usdRoot" `
    -DOIDN_APPS=OFF `
    -DOIDN_DEVICE_CPU=OFF `
    -DOIDN_DEVICE_CUDA=ON `
    "-DCMAKE_CUDA_COMPILER=$nvcc"
if ($LASTEXITCODE -ne 0) { throw "configure failed" }
cmake --build "$src\build" -j $jobs
if ($LASTEXITCODE -ne 0) { throw "build failed" }
cmake --install "$src\build"
if ($LASTEXITCODE -ne 0) { throw "install failed" }
Write-Host "OIDN $Version installed at $prefix"
