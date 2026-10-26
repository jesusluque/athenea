# Copyright (c) 2026 jesus luque.
#
# The Blender add-on for Windows as one directory and a zip Blender installs
# from disk, from a windows-x64-blender build: the counterpart of
# package-blender-addon.sh.
#
#   athenea_hydra\*.py, blender_manifest.toml   (platforms: windows-x64)
#   athenea_hydra\plugin\usd\        hdAthenea with Slang's and zlib's DLLs beside it, and the schemas
#   athenea_hydra\plugin\materialx\  MaterialX 1.39.5's libraries\
#   athenea_hydra\plugin\aofx\       the Mesh2Splat, SplatBakeFilter, SplatTransferZonal and Measure bundles
#   athenea_hydra\shaders\           found by hdAthenea three levels up
#   athenea_hydra\LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, THIRD_PARTY_LICENSES.md
#   athenea_hydra\licenses\<component>\   every third party's own licence files
#
# Debug files (.pdb, .ilk) stay behind. A licence file that is missing from the
# toolchain's trees is taken from Blender's own SPDX texts (license\spdx) when
# it is that licence's text; with neither, the script stops, so a package
# never goes out without them.
#
#   scripts\package-blender-addon.ps1 [-Build <dir>] [-Out <dir>]
param(
    [string]$Build = "",
    [string]$Out = "",
    [string]$ToolsSrc = $(if ($env:ATHENEA_TOOLS_SRC) { $env:ATHENEA_TOOLS_SRC } else { "D:\tools\src" })
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not $Build) { $Build = Join-Path $repo "build\windows-x64-blender" }
if (-not $Out) { $Out = Join-Path $repo "build\blender-addon" }
$Build = (Resolve-Path $Build).Path
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path

$cache = Get-Content (Join-Path $Build "CMakeCache.txt")
function Value($name) {
    $line = $cache | Where-Object { $_ -match "^$name(:[A-Z]+)?=" } | Select-Object -First 1
    if ($line) { return ($line -split "=", 2)[1] } else { return "" }
}
$slang = Value "SLANG_ROOT"
$blenderLib = Value "ATHENEA_BLENDER_LIB"
$blenderSpdx = if ($blenderLib) { Join-Path (Split-Path $blenderLib -Parent) "license\spdx" } else { "" }
$materialxSrc = Value "ATHENEA_MATERIALX_SLANG_SRC"
if (-not $materialxSrc) { $materialxSrc = Join-Path $ToolsSrc "MaterialX-1.39.5" }

$plugin = Join-Path $Build "plugin\usd\hdAthenea\hdAthenea.dll"
if (-not (Test-Path $plugin)) { throw "no ${plugin}: cmake --build --preset windows-x64-blender first" }

$addon = Join-Path $Out "athenea_hydra"
if (Test-Path $addon) { Remove-Item -Recurse -Force $addon }
New-Item -ItemType Directory -Force "$addon\plugin\aofx", "$addon\licenses" | Out-Null

# --- the add-on and what it loads -------------------------------------------------
$sourceAddon = Join-Path $repo "integrations\blender\athenea_hydra"
Copy-Item "$sourceAddon\*.py" $addon
# The manifest names the platform its binaries are for: this package's.
# Written without a byte order mark, which TOML does not allow.
$manifest = (Get-Content "$sourceAddon\blender_manifest.toml") -replace '^platforms = .*$', 'platforms = ["windows-x64"]'
[System.IO.File]::WriteAllLines("$addon\blender_manifest.toml", [string[]]$manifest, (New-Object System.Text.UTF8Encoding $false))
$skip = @("*.pdb", "*.ilk", "*.lib", "*.exp")
function Copy-Tree($from, $to) {
    & robocopy $from $to /E /NFL /NDL /NJH /NJS /NP /XF @skip | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "copying $from failed (robocopy $LASTEXITCODE)" }
}
Copy-Tree (Join-Path $Build "plugin\usd") "$addon\plugin\usd"
Copy-Tree (Join-Path $Build "plugin\materialx") "$addon\plugin\materialx"
Copy-Tree (Join-Path $Build "shaders") "$addon\shaders"
if (Test-Path "$addon\shaders\athenea\test") { Remove-Item -Recurse -Force "$addon\shaders\athenea\test" }
foreach ($bundle in "Mesh2Splat", "SplatBakeFilter", "SplatTransferZonal", "Measure") {
    $from = Join-Path $Build "aofx\$bundle.aofx.bundle"
    if (-not (Test-Path $from)) { throw "no $bundle bundle in $Build\aofx" }
    Copy-Tree $from "$addon\plugin\aofx\$bundle.aofx.bundle"
}
# What hdAthenea links beyond Blender's DLLs: Slang's (the build copies them
# beside it) and zlib's. The add-on puts that folder on PATH before USD opens
# the plugin.
$beside = "$addon\plugin\usd\hdAthenea"
foreach ($dll in "slang.dll", "slang-compiler.dll", "z.dll") {
    if (-not (Test-Path (Join-Path $beside $dll))) { throw "no $dll beside hdAthenea in $Build" }
}

# --- licences ---------------------------------------------------------------------
foreach ($f in "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md") { Copy-Item (Join-Path $repo $f) $addon }
Copy-Item (Join-Path $repo "integrations\blender\THIRD_PARTY_LICENSES.md") $addon
# Licence <component> <spdx text to fall back on, or ""> <file>...
function Licence($component, $spdx, [string[]]$files) {
    $dir = Join-Path $addon "licenses\$component"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $copied = 0
    foreach ($file in $files) {
        if (Test-Path $file) { Copy-Item -Recurse $file $dir; $copied++ }
    }
    if ($copied -gt 0) { return }
    $fallback = if ($spdx -and $blenderSpdx) { Join-Path $blenderSpdx "$spdx.txt" } else { "" }
    if ($fallback -and (Test-Path $fallback)) { Copy-Item $fallback $dir; return }
    throw "licence files missing for $component`: $($files -join ', ')"
}
# Carried by the package.
Licence "slang" "" @("$slang\LICENSE", "$slang\LICENSES")
Licence "materialx" "Apache-2.0" @("$materialxSrc\LICENSE", "$materialxSrc\THIRD-PARTY.md")
Licence "zlib" "Zlib" @("$ToolsSrc\zlib-1.3.1\LICENSE", "$ToolsSrc\zlib\LICENSE")
if (Test-Path "$addon\shaders\nanovdb") {
    Licence "openvdb" "MPL-2.0" @("$ToolsSrc\openvdb-10.1.0\LICENSE")
}
# Blender's own, compiled against.
Licence "openusd" "" @("$ToolsSrc\OpenUSD-26.08\LICENSE.txt", "$ToolsSrc\OpenUSD-26.08\NOTICE.txt")
Licence "onetbb" "" @("$ToolsSrc\oneTBB-2022.3.0\LICENSE.txt", "$ToolsSrc\oneTBB-2022.3.0\third-party-programs.txt")
Licence "opencolorio" "" @("$ToolsSrc\OpenColorIO-2.5.2\LICENSE", "$ToolsSrc\OpenColorIO-2.5.2\THIRD-PARTY.md")
$oidnDoc = Join-Path (Value "ATHENEA_OIDN_ROOT") "share\doc\OpenImageDenoise"
Licence "oidn" "Apache-2.0" @("$oidnDoc\LICENSE.txt", "$oidnDoc\third-party-programs.txt")

# --- the zip ----------------------------------------------------------------------
$version = (Select-String -Path "$addon\blender_manifest.toml" -Pattern '^version = "(.*)"').Matches[0].Groups[1].Value
$zip = Join-Path $Out "athenea_hydra-$version-windows-x64.zip"
if (Test-Path $zip) { Remove-Item $zip }
# Blender reads the manifest at the zip's root, so the folder's contents go in,
# not the folder; entries are named with '/' (Windows PowerShell's
# Compress-Archive writes '\').
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::Open($zip, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in Get-ChildItem $addon -Recurse -File) {
        $name = $file.FullName.Substring($addon.Length + 1) -replace '\\', '/'
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $file.FullName, $name,
            [System.IO.Compression.CompressionLevel]::Optimal)
    }
} finally { $archive.Dispose() }
$addon
$zip
