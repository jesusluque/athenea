#!/bin/bash
# Copyright (c) 2026 jesus luque.
#
# A release of athenea for macOS or Linux, from a build tree: the programs, the
# Hydra plugin, the AOFX bundles and the shaders, beside the libraries they
# load (OpenUSD with MaterialX, OpenVDB, TBB and Boost, OpenColorIO, Slang,
# OIDN, and on Linux CUDA's redistributable runtime), laid out
# as athenea looks for them -- bin/, lib/, libraries/, plugin/usd/, aofx/,
# shaders/ -- and the absolute library paths of the building machine replaced
# by ones relative to the program. Windows has scripts/package.ps1.
#
#   scripts/package.sh <build dir> [out dir]
#
# The tools are where the build found them (SLANG_ROOT, ATHENEA_USD_ROOT,
# ATHENEA_OCIO_ROOT, ATHENEA_OIDN_ROOT in its CMakeCache.txt). Writes
# athenea-<version>-<os>-<arch>.tar.gz (Linux) or .zip (macOS) in the out dir.
set -euo pipefail

build=$(cd "${1:?usage: package.sh <build dir> [out dir]}" && pwd)
out=$(mkdir -p "${2:-$PWD}" && cd "${2:-$PWD}" && pwd)
src=$(cd "$(dirname "$0")/.." && pwd)
cache="$build/CMakeCache.txt"
value() { sed -n "s/^$1:[A-Z]*=//p" "$cache" | head -1; }

version=$(value CMAKE_PROJECT_VERSION)
slang=$(value SLANG_ROOT)
usd=$(value ATHENEA_USD_ROOT)
ocio=$(value ATHENEA_OCIO_ROOT)
oidn=$(value ATHENEA_OIDN_ROOT)
case "$(uname -s)" in
    Darwin) os=macos; arch=arm64 ;;
    Linux)  os=linux; arch=$(uname -m) ;;
    *) echo "package.sh: macOS or Linux; Windows has package.ps1" >&2; exit 1 ;;
esac
name="athenea-$version-$os-$arch"
pkg="$out/$name"
rm -rf "$pkg"
mkdir -p "$pkg/bin" "$pkg/lib" "$pkg/plugin/usd" "$pkg/aofx" "$pkg/share/doc/athenea/third-party"

# The engine.
for program in athenea athenea-mcp; do
    [ -x "$build/bin/$program" ] && cp "$build/bin/$program" "$pkg/bin/"
done
cp -R "$build/shaders" "$pkg/shaders"
cp -R "$build/aofx/"*.aofx.bundle "$pkg/aofx/"
cp -R "$build/plugin/usd/"* "$pkg/plugin/usd/"

# OpenUSD: its libraries with their plugInfo (lib/usd), its own plugins, and
# MaterialX's standard libraries, where hdMtlx looks for them.
cp -R "$usd/lib/usd" "$pkg/lib/usd"
cp -R "$usd/plugin/usd/"* "$pkg/plugin/usd/" 2>/dev/null || true
cp -R "$usd/libraries" "$pkg/libraries"
copy_libs() {   # every shared library in $1, symlinks kept
    find "$1" -maxdepth 1 \( -name '*.dylib' -o -name '*.so' -o -name '*.so.*' \) -exec cp -P {} "$pkg/lib/" \;
}
copy_libs "$usd/lib"
[ -n "$ocio" ] && [ -d "$ocio/lib" ] && copy_libs "$ocio/lib"
copy_libs "$slang/lib"
[ -n "$oidn" ] && [ -d "$oidn/lib" ] && copy_libs "$oidn/lib"

# CUDA's redistributable runtime on Linux, which slang-rhi loads by name when it
# compiles a kernel (NVRTC and its builtins, nvJitLink) and OIDN's CUDA device
# links (cudart): from the toolkit the build used. The driver is the machine's.
cuda=""
if [ "$os" = linux ]; then
    nvcc=$(value CMAKE_CUDA_COMPILER)
    if [ -n "$nvcc" ]; then
        cuda=$(dirname "$(dirname "$nvcc")")
        for pattern in 'libnvrtc.so*' 'libnvrtc-builtins.so*' 'libnvJitLink.so*' 'libcudart.so*'; do
            find "$cuda/lib64/" -maxdepth 1 -name "$pattern" -exec cp -P {} "$pkg/lib/" \;
        done
    fi
fi

# Licences: athenea's, then each library's beside its name.
for f in LICENSE NOTICE THIRD_PARTY_NOTICES.md; do [ -f "$src/$f" ] && cp "$src/$f" "$pkg/share/doc/athenea/"; done
cp "$src/CHANGELOG.md" "$pkg/share/doc/athenea/"
lic() { [ -f "$2" ] && cp "$2" "$pkg/share/doc/athenea/third-party/$1" || true; }
lic OpenUSD.txt "$usd/LICENSE.txt"
for f in "$slang/LICENSE" "$slang/LICENSE.txt" "$slang/LICENSES/"*; do lic "Slang-$(basename "$f")" "$f"; done
lic OpenColorIO.txt "$ocio/share/doc/OpenColorIO/LICENSE"
lic OIDN.txt "$oidn/share/doc/OpenImageDenoise/LICENSE.txt"
[ -n "$cuda" ] && lic CUDA-EULA.txt "$cuda/EULA.txt"

# Library paths relative to each binary.
if [ "$os" = macos ]; then
    relink() {   # $1: file, $2: its rpath to lib/
        local f=$1
        otool -l "$f" | awk '/LC_RPATH/ {getline; getline; print $2}' | while read -r old; do
            install_name_tool -delete_rpath "$old" "$f" 2>/dev/null || true
        done
        install_name_tool -add_rpath "$2" "$f"
        codesign --force --sign - "$f" 2>/dev/null
    }
    for f in "$pkg/bin/"*; do relink "$f" @executable_path/../lib; done
    find "$pkg/plugin/usd" -name '*.dylib' | while read -r f; do relink "$f" @loader_path/../../lib; done
    find "$pkg/aofx" -type f -name '*.aofx' | while read -r f; do
        # aofx/<Name>.aofx.bundle/Contents/MacOS/<Name>.aofx: five folders down from the root.
        relink "$f" @loader_path/../../../../lib
    done
    find "$pkg/lib" -maxdepth 1 -type f -name '*.dylib' | while read -r f; do relink "$f" @loader_path; done
    (cd "$out" && rm -f "$name.zip" && ditto -c -k --keepParent "$name" "$name.zip")
    echo "$out/$name.zip"
else
    command -v patchelf >/dev/null || { echo "package.sh: patchelf is needed on Linux" >&2; exit 1; }
    for f in "$pkg/bin/"*; do patchelf --set-rpath '$ORIGIN/../lib' "$f"; done
    find "$pkg/plugin/usd" -name '*.so' | while read -r f; do patchelf --set-rpath '$ORIGIN/../../lib' "$f"; done
    find "$pkg/aofx" -type f -name '*.aofx' | while read -r f; do patchelf --set-rpath '$ORIGIN/../../../../lib' "$f"; done
    find "$pkg/lib" -maxdepth 1 -type f -name '*.so*' | while read -r f; do patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null || true; done
    (cd "$out" && tar -czf "$name.tar.gz" "$name")
    echo "$out/$name.tar.gz"
fi
