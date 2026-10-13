# Copyright (c) 2026 jesus luque.
#
# OpenColorIO for athenea on Windows, the library alone with its dependencies
# built in: the counterpart of build-ocio.sh. Run from PowerShell:
#   scripts\build-ocio.ps1 [-Version 2.5.2]
# The prefix is $env:ATHENEA_OCIO_ROOT, or D:\tools\ocio-<version>.

param([string]$Version = "2.5.2")
# Native programs are judged by their exit code: under "Stop", Windows
# PowerShell 5.1 takes anything they write to stderr (git's progress) as a
# failure.
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "msvc-env.ps1")

$prefix = if ($env:ATHENEA_OCIO_ROOT) { $env:ATHENEA_OCIO_ROOT } else { "D:\tools\ocio-$Version" }
$srcRoot = if ($env:ATHENEA_OCIO_SRC) { $env:ATHENEA_OCIO_SRC } else { "D:\tools\src" }
$jobs = if ($env:JOBS) { $env:JOBS } else { $env:NUMBER_OF_PROCESSORS }
$src = Join-Path $srcRoot "OpenColorIO-$Version"

if (-not (Test-Path $src)) {
    New-Item -ItemType Directory -Force -Path $srcRoot | Out-Null
    $archive = Join-Path $srcRoot "ocio-$Version.tar.gz"
    curl.exe -L --fail -o $archive "https://github.com/AcademySoftwareFoundation/OpenColorIO/archive/refs/tags/v$Version.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw "download failed" }
    tar -xzf $archive -C $srcRoot
    if ($LASTEXITCODE -ne 0) { throw "unpacking failed" }
}
cmake -S $src -B "$src\build" -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_INSTALL_PREFIX=$prefix" `
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON `
    -DOCIO_INSTALL_EXT_PACKAGES=ALL `
    -DOCIO_BUILD_APPS=OFF `
    -DOCIO_BUILD_PYTHON=OFF `
    -DOCIO_BUILD_OPENFX=OFF `
    -DOCIO_BUILD_TESTS=OFF `
    -DOCIO_BUILD_GPU_TESTS=OFF `
    -DOCIO_BUILD_DOCS=OFF `
    -DOCIO_USE_OIIO_FOR_APPS=OFF
if ($LASTEXITCODE -ne 0) { throw "configure failed" }
cmake --build "$src\build" -j $jobs
if ($LASTEXITCODE -ne 0) { throw "build failed" }
cmake --install "$src\build"
if ($LASTEXITCODE -ne 0) { throw "install failed" }
Write-Host "OpenColorIO $Version installed at $prefix"
