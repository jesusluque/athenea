# Copyright (c) 2026 jesus luque.
#
# The Blender add-on for Windows as one directory and a zip Blender installs
# from disk, from a windows-x64-blender build (macOS and Linux have
# package-blender-addon.sh):
#
#   athenea_hydra\*.py, blender_manifest.toml (platforms windows-x64)
#   athenea_hydra\plugin\usd\hdAthenea\   hdAthenea.dll and every DLL it loads that is
#                                         neither Blender's nor Windows' (Slang's, zlib...),
#                                         and DXC's, which D3D12 compiles kernels with. The
#                                         add-on puts this folder on PATH before USD loads
#                                         the plugin.
#   athenea_hydra\plugin\usd\bin, include CUDA's runtime (NVRTC and its builtins, nvJitLink,
#                                         cudart) and the headers NVRTC compiles against:
#                                         the engine points CUDA_PATH at plugin\usd when it
#                                         finds them there, so CUDA opens with the driver alone.
#   athenea_hydra\plugin\materialx, plugin\aofx (Mesh2Splat, SplatBakeFilter,
#   SplatTransferZonal, Measure), shaders
#   athenea_hydra\LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, THIRD_PARTY_LICENSES.md,
#   licenses\<component>\
#
#   scripts\package-blender-addon.ps1 [build dir] [output dir]
#     (defaults build\windows-x64-blender and build\blender-addon)
#
# ATHENEA_CUDA_RUNTIME names the CUDA runtime folder (bin\ and include\), as
# package.ps1 takes it; ATHENEA_DXC the folder with dxcompiler.dll and dxil.dll.
# BLENDER_DIR is the Blender the build was made against: a DLL found in its
# blender.shared is Blender's and stays out.
param([string]$Build = "", [string]$Out = "")
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Build) { $Build = Join-Path $Repo "build\windows-x64-blender" }
if (-not $Out) { $Out = Join-Path $Repo "build\blender-addon" }
$Build = (Resolve-Path $Build).Path
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = (Resolve-Path $Out).Path
function Fail($why) { throw "package-blender-addon: $why" }
function CacheValue($name) {
    $line = Select-String -Path (Join-Path $Build "CMakeCache.txt") -Pattern "^${name}:[A-Z]*=(.*)$" | Select-Object -First 1
    if ($line) { return $line.Matches[0].Groups[1].Value } else { return "" }
}

$plugin = Join-Path $Build "plugin\usd\hdAthenea\hdAthenea.dll"
if (-not (Test-Path $plugin)) { Fail "no $plugin: cmake --build --preset windows-x64-blender first" }
$slang = CacheValue "SLANG_ROOT"
$blenderShared = CacheValue "ATHENEA_BLENDER_LIB"
if (-not $blenderShared -or -not (Test-Path $blenderShared)) { Fail "the build's ATHENEA_BLENDER_LIB ($blenderShared) is not there" }
$src = if ($env:ATHENEA_TOOLS_SRC) { $env:ATHENEA_TOOLS_SRC } else { "D:\tools\src" }
$cudaRuntime = $env:ATHENEA_CUDA_RUNTIME
$dxc = $env:ATHENEA_DXC
if (-not $cudaRuntime -or -not (Test-Path (Join-Path $cudaRuntime "bin"))) { Fail "ATHENEA_CUDA_RUNTIME must name a folder with bin\ and include\" }
if (-not $dxc -or -not (Test-Path (Join-Path $dxc "dxcompiler.dll"))) { Fail "ATHENEA_DXC must name the folder with dxcompiler.dll and dxil.dll" }

$addon = Join-Path $Out "athenea_hydra"
if (Test-Path $addon) { Remove-Item -Recurse -Force $addon }
New-Item -ItemType Directory -Force -Path (Join-Path $addon "plugin"), (Join-Path $addon "licenses") | Out-Null

# --- the add-on and what it loads -------------------------------------------------
Copy-Item (Join-Path $Repo "integrations\blender\athenea_hydra\*.py") $addon
$manifest = Join-Path $addon "blender_manifest.toml"
# One zip a system: Blender installs a zip only where its manifest says it
# runs. Written without a byte order mark, which TOML does not take.
$text = ((Get-Content (Join-Path $Repo "integrations\blender\athenea_hydra\blender_manifest.toml")) `
    -replace '^platforms = .*$', 'platforms = ["windows-x64"]') -join "`n"
[IO.File]::WriteAllText($manifest, $text + "`n", (New-Object Text.UTF8Encoding($false)))
Copy-Item -Recurse (Join-Path $Build "plugin\usd") (Join-Path $addon "plugin\usd")
Copy-Item -Recurse (Join-Path $Build "plugin\materialx") (Join-Path $addon "plugin\materialx")
Copy-Item -Recurse (Join-Path $Build "shaders") (Join-Path $addon "shaders")
$tests = Join-Path $addon "shaders\athenea\test"
if (Test-Path $tests) { Remove-Item -Recurse -Force $tests }
New-Item -ItemType Directory -Force -Path (Join-Path $addon "plugin\aofx") | Out-Null
foreach ($bundle in "Mesh2Splat", "SplatBakeFilter", "SplatTransferZonal", "Measure") {
    $from = Join-Path $Build "aofx\$bundle.aofx.bundle"
    if (-not (Test-Path $from)) { Fail "no $bundle bundle in $Build\aofx" }
    Copy-Item -Recurse $from (Join-Path $addon "plugin\aofx\")
}

# Beside hdAthenea: every DLL it names that is neither Blender's nor Windows',
# from the folders the build used, then what those name in turn.
$here = Join-Path $addon "plugin\usd\hdAthenea"
$dumpbin = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
if (-not $dumpbin) { Fail "dumpbin is needed (run scripts\msvc-env.ps1 first)" }
$searched = @("$slang\bin", (Split-Path $plugin -Parent), (CacheValue "ATHENEA_USD_ROOT") + "\bin",
              "D:\tools\usd-26.08-mx\bin", (CacheValue "ATHENEA_OCIO_ROOT") + "\bin") | Where-Object { $_ -and (Test-Path $_) }
$queue = New-Object System.Collections.Generic.Queue[string]
$queue.Enqueue($plugin)
$seen = @{}
while ($queue.Count -gt 0) {
    $file = $queue.Dequeue()
    $names = & dumpbin /nologo /dependents $file | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '\.dll$' }
    foreach ($name in $names) {
        $key = $name.ToLower()
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true
        if (Test-Path (Join-Path $blenderShared $name)) { continue }
        if (Test-Path (Join-Path $env:SystemRoot "System32\$name")) { continue }
        if ($key -match '^(api-ms-|ext-ms-|vcruntime|msvcp|ucrtbase|nvcuda)') { continue }
        $found = $searched | ForEach-Object { Join-Path $_ $name } | Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $found) { Fail "hdAthenea loads $name, which is neither Blender's nor Windows' and is not in $($searched -join ', ')" }
        $to = Join-Path $here $name
        if (-not (Test-Path $to)) { Copy-Item $found $to }
        $queue.Enqueue($found)
    }
}
# Loaded by name at run time rather than linked: Slang's GLSL module and
# glslang (Vulkan's SPIR-V), and DXC (D3D12).
foreach ($name in "slang-glsl-module.dll", "slang-glslang.dll") {
    $from = Join-Path "$slang\bin" $name
    if (Test-Path $from) { Copy-Item -Force $from $here }
}
Copy-Item -Force (Join-Path $dxc "dxcompiler.dll"), (Join-Path $dxc "dxil.dll") $here

# CUDA's runtime and headers where the engine looks for a package's own.
$usdDir = Join-Path $addon "plugin\usd"
New-Item -ItemType Directory -Force -Path (Join-Path $usdDir "bin"), (Join-Path $usdDir "include") | Out-Null
Copy-Item (Join-Path $cudaRuntime "bin\*.dll") (Join-Path $usdDir "bin")
Copy-Item -Recurse (Join-Path $cudaRuntime "include\*") (Join-Path $usdDir "include")
if (-not (Test-Path (Join-Path $usdDir "bin\nvrtc64_120_0.dll"))) { Fail "no nvrtc64_120_0.dll in $cudaRuntime\bin" }

# --- licences ---------------------------------------------------------------------
function Licence($component, [string[]]$files) {
    $dir = Join-Path $addon "licenses\$component"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    foreach ($f in $files) {
        if (-not (Test-Path $f)) { Fail "licence file missing for ${component}: $f" }
        Copy-Item -Recurse $f $dir
    }
}
Copy-Item (Join-Path $Repo "LICENSE"), (Join-Path $Repo "NOTICE"), (Join-Path $Repo "THIRD_PARTY_NOTICES.md"),
          (Join-Path $Repo "integrations\blender\THIRD_PARTY_LICENSES.md") $addon
# Carried by the package.
Licence slang @("$slang\LICENSE", "$slang\LICENSES")
Licence materialx @("$src\MaterialX-1.39.5\LICENSE", "$src\MaterialX-1.39.5\THIRD-PARTY.md")
$dxcLicence = @("$dxc\LICENSE.TXT", "$dxc\..\..\LICENSE.TXT", "$dxc\..\..\LICENSE-MIT.txt", "$dxc\..\..\LICENSE-MS.txt") | Where-Object { Test-Path $_ }
if (-not $dxcLicence) { Fail "no DXC licence beside $dxc" }
Licence dxc $dxcLicence
$cudaLicence = @("$cudaRuntime\LICENSE", "$cudaRuntime\EULA.txt") | Where-Object { Test-Path $_ }
if (-not $cudaLicence) { Fail "no CUDA licence (LICENSE or EULA.txt) in $cudaRuntime" }
Licence cuda $cudaLicence
if (Test-Path (Join-Path $addon "shaders\nanovdb")) {
    $vdb = Get-ChildItem "D:\tools\usd-26.08-mx\src" -Directory -Filter "openvdb-*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($vdb) { Licence openvdb @("$($vdb.FullName)\LICENSE") } else { Fail "no OpenVDB source for its licence" }
}
# Blender's own, compiled against.
Licence openusd @("$src\OpenUSD-26.08\LICENSE.txt", "$src\OpenUSD-26.08\NOTICE.txt")
Licence onetbb @("$src\oneTBB-2022.3.0\LICENSE.txt", "$src\oneTBB-2022.3.0\third-party-programs.txt")
Licence opencolorio @("$src\OpenColorIO-2.5.2\LICENSE", "$src\OpenColorIO-2.5.2\THIRD-PARTY.md")
$oidnDoc = Get-ChildItem -Recurse -Filter "third-party-programs.txt" "D:\tools\oidn-2.5.1" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $oidnDoc) { Fail "no OIDN third-party-programs.txt under D:\tools\oidn-2.5.1" }
Licence oidn @((Join-Path $oidnDoc.DirectoryName "LICENSE.txt"), $oidnDoc.FullName)

# --- the zip ----------------------------------------------------------------------
$version = (Select-String -Path $manifest -Pattern '^version = "(.*)"').Matches[0].Groups[1].Value
$zip = Join-Path $Out "athenea_hydra-$version-windows-x64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
# tar writes forward slashes, which Blender's extension installer expects.
Push-Location $addon
& tar.exe -a -c -f $zip *
Pop-Location
if ($LASTEXITCODE -ne 0) { Fail "tar could not write $zip" }
$addon
$zip
