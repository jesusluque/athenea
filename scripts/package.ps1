# Copyright (c) 2026 jesus luque.
#
# A release of athenea for Windows, from a build tree: `cpack` over the
# install (cmake/Package.cmake, cmake/InstallRuntime.cmake) -- the programs,
# the Hydra plugin, the AOFX and OpenFX bundles and the shaders, with the DLLs
# they load in bin\ (where Windows looks first): OpenUSD with MaterialX and
# OpenVDB, OpenColorIO, Slang, OIDN, CUDA's NVRTC and nvJitLink, and the MSVC
# runtime -- and the per-user install.ps1 and uninstall.ps1 at its root. The
# counterpart of package.sh.
#
#   scripts\package.ps1 <build dir> [out dir]
#
# The tools are where the build found them (its CMakeCache.txt); the NVIDIA
# driver is the machine's. Writes athenea-<version>-windows-x64.zip and the
# same folder unpacked beside it, which installer-windows.ps1 takes.
param([Parameter(Mandatory = $true)][string]$Build, [string]$Out = (Get-Location).Path)
$ErrorActionPreference = "Continue"

$Build = (Resolve-Path $Build).Path
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$cache = Get-Content (Join-Path $Build "CMakeCache.txt")
$line = $cache | Where-Object { $_ -match "^CMAKE_PROJECT_VERSION(:[A-Z]+)?=" } | Select-Object -First 1
$version = ($line -split "=", 2)[1]
$name = "athenea-$version-windows-x64"

& cpack --config (Join-Path $Build "CPackConfig.cmake") -B $Out
if ($LASTEXITCODE -ne 0) { throw "cpack failed ($LASTEXITCODE)" }
Remove-Item -Recurse -Force (Join-Path $Out "_CPack_Packages") -ErrorAction SilentlyContinue
$zip = Join-Path $Out "$name.zip"
$pkg = Join-Path $Out $name
if (Test-Path $pkg) { Remove-Item -Recurse -Force $pkg }
Expand-Archive -Path $zip -DestinationPath $Out
Write-Host $zip
