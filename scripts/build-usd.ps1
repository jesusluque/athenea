# Copyright (c) 2026 jesus luque.
#
# OpenUSD with MaterialX and OpenVDB for athenea on Windows: the counterpart of
# build-usd.sh, with the same build_usd.py options. Run from PowerShell:
#   scripts\build-usd.ps1 [-Version v26.08]
# The prefix is $env:ATHENEA_USD_ROOT, or D:\tools\usd-26.08-mx; the sources go
# to $env:ATHENEA_USD_SRC, or D:\tools\src\OpenUSD-26.08.

param([string]$Version = "v26.08")
# Native programs are judged by their exit code: under "Stop", Windows
# PowerShell 5.1 takes anything they write to stderr (git's progress) as a
# failure.
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "msvc-env.ps1")

$plain = $Version.TrimStart("v")
$prefix = if ($env:ATHENEA_USD_ROOT) { $env:ATHENEA_USD_ROOT } else { "D:\tools\usd-$plain-mx" }
$src = if ($env:ATHENEA_USD_SRC) { $env:ATHENEA_USD_SRC } else { "D:\tools\src\OpenUSD-$plain" }
$jobs = if ($env:JOBS) { $env:JOBS } else { $env:NUMBER_OF_PROCESSORS }

if (-not (Test-Path (Join-Path $src ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $src) | Out-Null
    git clone --depth 1 --branch $Version https://github.com/PixarAnimationStudios/OpenUSD.git $src
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}
Write-Host "OpenUSD $Version at $(git -C $src rev-parse --short HEAD), into $prefix"
$env:CMAKE_POLICY_VERSION_MINIMUM = "3.5"

python (Join-Path $src "build_scripts\build_usd.py") `
    --build-variant release `
    --no-python `
    --no-docs `
    --no-tests `
    --no-tutorials `
    --materialx `
    --no-embree `
    --no-ptex `
    --openvdb `
    --build-args "OpenVDB,-DOPENVDB_BUILD_NANOVDB=ON -DNANOVDB_USE_OPENVDB=ON" `
    --no-openimageio `
    --no-opencolorio `
    --onetbb `
    --usd-imaging `
    --tools `
    --examples `
    --generator Ninja `
    -j $jobs `
    $prefix
if ($LASTEXITCODE -ne 0) { throw "build_usd.py failed" }
Write-Host "OpenUSD $Version installed at $prefix"
Write-Host "Configure athenea with -DATHENEA_USD_ROOT=$prefix (the Windows presets default to it)."
