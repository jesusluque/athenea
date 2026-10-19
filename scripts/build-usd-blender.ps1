# Copyright (c) 2026 jesus luque.
#
# The headers and import libraries hdAthenea is compiled against to run inside
# Blender on Windows: the counterpart of build-usd-blender.sh. Headers only, as
# there: at run time the plugin uses Blender's own usd_ms.dll, tbb12.dll,
# MaterialX*.dll, OpenColorIO and OIDN, so the process holds one of each.
#
# What differs from macOS and Linux: Blender ships its DLLs without import
# libraries, so each one's exports are read (dumpbin /exports) and turned into
# a .lib the linker takes (lib /def); and Python is a DLL here (python313.dll),
# linked through python313.lib rather than left to the executable.
#
#   scripts\build-usd-blender.ps1 -BlenderDir <folder holding blender.exe> [-Version v26.08]
#
# The prefix is $env:ATHENEA_USD_BLENDER_ROOT, or D:\tools\usd-26.08-blender;
# sources go to $env:ATHENEA_SRC, or D:\tools\src. Python 3.13 (for its
# headers and python313.lib) is $env:ATHENEA_PYTHON313, or `uv python find 3.13`,
# or `py -3.13`.

param(
    [Parameter(Mandatory = $true)][string]$BlenderDir,
    [string]$Version = "v26.08"
)
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "msvc-env.ps1")

$plain = $Version.TrimStart("v")
$prefix = if ($env:ATHENEA_USD_BLENDER_ROOT) { $env:ATHENEA_USD_BLENDER_ROOT } else { "D:\tools\usd-$plain-blender" }
$src = if ($env:ATHENEA_SRC) { $env:ATHENEA_SRC } else { "D:\tools\src" }
$work = "$prefix-build"
$jobs = if ($env:JOBS) { $env:JOBS } else { $env:NUMBER_OF_PROCESSORS }
$namespace = "pxrBlender_v" + $plain.Replace(".", "_")
$shared = Join-Path $BlenderDir "blender.shared"
if (-not (Test-Path (Join-Path $shared "usd_ms.dll"))) { throw "no usd_ms.dll in $shared" }
New-Item -ItemType Directory -Force -Path $src, $work, $prefix | Out-Null

function Clone($url, $tag, $dir) {
    if (-not (Test-Path (Join-Path $dir ".git"))) {
        git clone -q --depth 1 --branch $tag $url $dir
        if ($LASTEXITCODE -ne 0) { throw "git clone $url failed" }
    }
}
function Run($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed" }
}
Clone https://github.com/PixarAnimationStudios/OpenUSD.git $Version "$src\OpenUSD-$plain"
Clone https://github.com/uxlfoundation/oneTBB.git v2022.3.0 "$src\oneTBB-2022.3.0"
Clone https://github.com/AcademySoftwareFoundation/MaterialX.git v1.39.4 "$src\MaterialX-1.39.4"
Clone https://github.com/AcademySoftwareFoundation/MaterialX.git v1.39.5 "$src\MaterialX-1.39.5"
Clone https://github.com/PixarAnimationStudios/OpenSubdiv.git v3_7_0 "$src\OpenSubdiv-3.7.0"

# Python 3.13: its headers, and the import library of the python313.dll
# Blender ships.
$python = $env:ATHENEA_PYTHON313
if (-not $python -and (Get-Command uv -ErrorAction SilentlyContinue)) { $python = (uv python find 3.13 2>$null) }
if (-not $python -and (Get-Command py -ErrorAction SilentlyContinue)) { $python = (py -3.13 -c "import sys; print(sys.executable)" 2>$null) }
if (-not $python) { throw "Python 3.13 is needed for its headers: uv python install 3.13" }
$pyInclude = & $python -c "import sysconfig; print(sysconfig.get_paths()['include'])"
$pyLib = Join-Path (Split-Path $python) "libs\python313.lib"
if (-not (Test-Path $pyLib)) { throw "no python313.lib beside $python" }
$env:CMAKE_POLICY_VERSION_MINIMUM = "3.5"

# oneTBB: built so USD's configure finds a TBBConfig; never linked by the plugin.
cmake -S "$src\oneTBB-2022.3.0" -B "$work\tbb" -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_INSTALL_PREFIX=$prefix" -DTBB_TEST=OFF -DTBB_EXAMPLES=OFF -DTBBMALLOC_BUILD=OFF | Out-Null; Run "tbb configure"
cmake --build "$work\tbb" -j $jobs; Run "tbb build"
cmake --install "$work\tbb" | Out-Null

# MaterialX 1.39.4, shared, without Python, viewer or tests.
cmake -S "$src\MaterialX-1.39.4" -B "$work\materialx" -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_INSTALL_PREFIX=$prefix" -DMATERIALX_BUILD_SHARED_LIBS=ON -DMATERIALX_BUILD_PYTHON=OFF `
    -DMATERIALX_BUILD_VIEWER=OFF -DMATERIALX_BUILD_GRAPH_EDITOR=OFF -DMATERIALX_BUILD_TESTS=OFF `
    -DMATERIALX_BUILD_GEN_MDL=OFF -DMATERIALX_BUILD_GEN_OSL=OFF -DMATERIALX_BUILD_RENDER=ON | Out-Null; Run "materialx configure"
cmake --build "$work\materialx" -j $jobs; Run "materialx build"
cmake --install "$work\materialx" | Out-Null

# OpenSubdiv 3.7.0: its headers.
$osd = "$src\OpenSubdiv-3.7.0\opensubdiv"
Get-ChildItem $osd -Recurse -Filter *.h | ForEach-Object {
    $to = Join-Path "$prefix\include\opensubdiv" $_.FullName.Substring($osd.Length + 1)
    New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
    Copy-Item $_.FullName $to -Force
}

# OpenUSD, Blender's way: its constructor sections renamed (usd_ctor.diff), on
# a copy of the sources.
$usdSrc = "$work\OpenUSD-$plain-src"
robocopy "$src\OpenUSD-$plain" $usdSrc /MIR /XD .git /NFL /NDL /NJH /NJS /NP | Out-Null
foreach ($f in "pxr\base\arch\attributes.h", "pxr\base\arch\attributes.cpp") {
    $path = Join-Path $usdSrc $f
    (Get-Content $path -Raw).Replace("pxrctor", "pxbctor").Replace("pxrdtor", "pxbdtor") | Set-Content -NoNewline $path
}
cmake -S $usdSrc -B "$work\usd" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$prefix" `
    "-DCMAKE_PREFIX_PATH=$prefix" "-DPXR_SET_INTERNAL_NAMESPACE=$namespace" -DPXR_BUILD_MONOLITHIC=ON `
    -DPXR_ENABLE_PYTHON_SUPPORT=ON -DPXR_USE_PYTHON_3=ON `
    "-DPython3_EXECUTABLE=$python" "-DPython3_INCLUDE_DIR=$pyInclude" "-DPython3_LIBRARY=$pyLib" `
    -DPXR_BUILD_IMAGING=ON -DPXR_BUILD_USD_IMAGING=ON -DPXR_ENABLE_GL_SUPPORT=ON `
    -DPXR_ENABLE_MATERIALX_SUPPORT=ON "-DMaterialX_DIR=$prefix\lib\cmake\MaterialX" "-DOPENSUBDIV_ROOT_DIR=$prefix" `
    -DPXR_ENABLE_OPENVDB_SUPPORT=OFF -DPXR_BUILD_OPENIMAGEIO_PLUGIN=OFF -DPXR_BUILD_OPENCOLORIO_PLUGIN=OFF `
    -DPXR_ENABLE_PTEX_SUPPORT=OFF -DPXR_ENABLE_OSL_SUPPORT=OFF -DPXR_ENABLE_HDF5_SUPPORT=OFF `
    -DPXR_BUILD_EMBREE_PLUGIN=OFF -DPXR_BUILD_USDVIEW=OFF -DPXR_BUILD_USD_TOOLS=OFF `
    -DPXR_BUILD_TESTS=OFF -DPXR_BUILD_EXAMPLES=OFF -DPXR_BUILD_TUTORIALS=OFF -DPXR_BUILD_DOCUMENTATION=OFF | Out-Null
Run "usd configure"
# Only the headers: every library's header copy.
$targets = ninja -C "$work\usd" -t targets all | ForEach-Object { if ($_ -match '^([A-Za-z0-9_]+_headerfiles):') { $Matches[1] } }
ninja -C "$work\usd" -j $jobs @targets | Out-Null; Run "usd headers"
robocopy "$work\usd\include" "$prefix\include" /E /NFL /NDL /NJH /NJS /NP | Out-Null
if (-not (Select-String -Path "$prefix\include\pxr\pxr.h" -Pattern "${namespace}__pxrReserved__" -Quiet)) { throw "pxr.h: namespace is not $namespace" }
if (-not (Select-String -Path "$prefix\include\pxr\base\arch\attributes.h" -Pattern "pxbctor" -Quiet)) { throw "attributes.h is not Blender's" }

# Import libraries for Blender's DLLs: their exports, as a module definition,
# into a .lib under $prefix\lib\blender.
$implib = "$prefix\lib\blender"
New-Item -ItemType Directory -Force -Path $implib | Out-Null
$dlls = Get-ChildItem $shared -Filter *.dll | Where-Object { $_.BaseName -match '^(usd_ms|tbb12|MaterialX\w+|opencolorio_\w+|OpenImageDenoise(_core)?)$' }
foreach ($dll in $dlls) {
    $names = dumpbin /nologo /exports $dll.FullName | ForEach-Object {
        if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)') { $Matches[1] }
    }
    $def = Join-Path $implib "$($dll.BaseName).def"
    @("LIBRARY $($dll.Name)", "EXPORTS") + $names | Set-Content -Encoding ascii $def
    lib /nologo /machine:x64 "/def:$def" "/out:$implib\$($dll.BaseName).lib" | Out-Null; Run "lib $($dll.Name)"
    Write-Host "$($dll.Name): $($names.Count) exports"
}
Copy-Item $pyLib $implib -Force

Write-Host "$namespace headers at $prefix\include, import libraries at $implib (Python $pyInclude)"
Write-Host "Configure athenea with --preset windows-x64-blender and BLENDER_DIR=$BlenderDir."
