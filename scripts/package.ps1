# Copyright (c) 2026 jesus luque.
#
# A release of athenea for Windows, from a build tree: the programs, the Hydra
# plugin, the AOFX bundles and the shaders, with the DLLs they load beside the
# programs in bin\ (where Windows looks first) -- OpenUSD with MaterialX and
# OpenVDB, OpenColorIO, Slang, OIDN, and the MSVC runtime -- and OpenUSD's
# plugInfo, its plugins and MaterialX's libraries where OpenUSD looks for them.
# The counterpart of package.sh.
#
#   scripts\package.ps1 <build dir> [out dir]
#
# The tools are where the build found them (its CMakeCache.txt), CUDA's
# redistributable runtime included; the NVIDIA driver is the machine's.
# Writes athenea-<version>-windows-x64.zip.
param([Parameter(Mandatory = $true)][string]$Build, [string]$Out = (Get-Location).Path)
$ErrorActionPreference = "Stop"

$Build = (Resolve-Path $Build).Path
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$src = Split-Path $PSScriptRoot -Parent
$cache = Get-Content (Join-Path $Build "CMakeCache.txt")
function Value($name) {
    $line = $cache | Where-Object { $_ -match "^$name(:[A-Z]+)?=" } | Select-Object -First 1
    if ($line) { return ($line -split "=", 2)[1] } else { return "" }
}
$version = Value "CMAKE_PROJECT_VERSION"
$slang = Value "SLANG_ROOT"; $usd = Value "ATHENEA_USD_ROOT"; $ocio = Value "ATHENEA_OCIO_ROOT"; $oidn = Value "ATHENEA_OIDN_ROOT"
$name = "athenea-$version-windows-x64"
$pkg = Join-Path $Out $name
if (Test-Path $pkg) { Remove-Item -Recurse -Force $pkg }
foreach ($d in "bin", "plugin\usd", "aofx", "share\doc\athenea\third-party") {
    New-Item -ItemType Directory -Force (Join-Path $pkg $d) | Out-Null
}

# The engine.
foreach ($program in "athenea.exe", "athenea-mcp.exe") {
    $p = Join-Path $Build "bin\$program"
    if (Test-Path $p) { Copy-Item $p "$pkg\bin\" }
}
Copy-Item -Recurse "$Build\shaders" "$pkg\shaders"
Copy-Item -Recurse "$Build\aofx\*.aofx.bundle" "$pkg\aofx\"
Copy-Item -Recurse "$Build\plugin\usd\*" "$pkg\plugin\usd\"

# OpenUSD: its DLLs beside the programs, and its plugInfo in usd\ beside them
# (OpenUSD finds it relative to its own DLLs' folder), its own plugins,
# MaterialX's libraries.
Copy-Item "$usd\bin\*.dll" "$pkg\bin\"
Copy-Item "$usd\lib\*.dll" "$pkg\bin\"
Copy-Item -Recurse "$usd\lib\usd" "$pkg\bin\usd"
if (Test-Path "$usd\plugin\usd") { Copy-Item -Recurse "$usd\plugin\usd\*" "$pkg\plugin\usd\" -ErrorAction SilentlyContinue }
Copy-Item -Recurse "$usd\libraries" "$pkg\libraries"
foreach ($root in $ocio, $slang, $oidn) {
    if ($root -and (Test-Path "$root\bin")) { Copy-Item "$root\bin\*.dll" "$pkg\bin\" }
}

# CUDA's redistributable runtime, which slang-rhi loads when it compiles a
# kernel (NVRTC and its builtins, nvJitLink, cudart) and the headers NVRTC
# compiles against (include\): from $env:ATHENEA_CUDA_RUNTIME, a folder with
# bin\ and include\ such as NVIDIA's redistributable archives unpacked
# together, or else from the toolkit the build used (its runtime DLLs only).
# With include\ in the package the engine points CUDA_PATH at the package
# (gpu/Device.cpp), so a machine needs no toolkit. The driver (nvcuda.dll,
# OptiX) is the machine's own.
$runtime = $env:ATHENEA_CUDA_RUNTIME
$nvcc = Value "CMAKE_CUDA_COMPILER"
if (-not $runtime -and $nvcc) { $runtime = Split-Path (Split-Path $nvcc -Parent) -Parent }
if ($runtime) {
    foreach ($pattern in "nvrtc64_*.dll", "nvrtc-builtins64_*.dll", "nvJitLink_*.dll", "cudart64_*.dll") {
        Copy-Item "$runtime\bin\$pattern" "$pkg\bin\" -ErrorAction SilentlyContinue
    }
    if ($env:ATHENEA_CUDA_RUNTIME -and (Test-Path "$runtime\include\cuda_runtime.h")) {
        Copy-Item -Recurse "$runtime\include" "$pkg\include"
    }
    $eula = @("$runtime\EULA.txt", "$runtime\LICENSE") | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($eula) { Copy-Item $eula "$pkg\share\doc\athenea\third-party\CUDA-EULA.txt" }
}

# DXC (dxcompiler.dll, dxil.dll), which slang-rhi compiles D3D12's kernels
# with: from $env:ATHENEA_DXC, a folder holding them (bin\x64 of a DirectX
# Shader Compiler release); without it the package runs on CUDA and Vulkan and
# D3D12 makes no pipeline.
if ($env:ATHENEA_DXC) {
    foreach ($dll in "dxcompiler.dll", "dxil.dll") {
        Copy-Item (Join-Path $env:ATHENEA_DXC $dll) "$pkg\bin\" -ErrorAction SilentlyContinue
    }
}

# The MSVC runtime the programs were linked against, from the Build Tools.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -property installationPath
$crt = Get-ChildItem "$vs\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT" -Directory | Sort-Object FullName | Select-Object -Last 1
if ($crt) { Copy-Item "$($crt.FullName)\*.dll" "$pkg\bin\" }

# Licences.
foreach ($f in "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md") {
    if (Test-Path "$src\$f") { Copy-Item "$src\$f" "$pkg\share\doc\athenea\" }
}
function Lic($as, $from) { if (Test-Path $from) { Copy-Item $from "$pkg\share\doc\athenea\third-party\$as" } }
Lic "OpenUSD.txt" "$usd\LICENSE.txt"
Get-ChildItem "$slang\LICENSES\*", "$slang\LICENSE*" -ErrorAction SilentlyContinue | ForEach-Object { Lic "Slang-$($_.Name)" $_.FullName }
Lic "OpenColorIO.txt" "$ocio\share\doc\OpenColorIO\LICENSE"
Lic "OIDN.txt" "$oidn\share\doc\OpenImageDenoise\LICENSE.txt"

$zip = Join-Path $Out "$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $pkg -DestinationPath $zip -CompressionLevel Optimal
Write-Host $zip
