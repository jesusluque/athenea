#!/usr/bin/env bash
# Copyright (c) 2026 jesus luque.
#
# The Blender add-on as one directory and a zip Blender installs from disk,
# from a macos-arm64-blender or linux-x86_64-blender build (Windows has
# package-blender-addon.ps1):
#
#   athenea_hydra/*.py, blender_manifest.toml (its `platforms` this system's)
#   athenea_hydra/plugin/usd/        hdAthenea (with libslang, libzstd and libwebp beside it,
#                                    named @loader_path and signed ad hoc on macOS, found
#                                    by an $ORIGIN runpath on Linux) and the schemas
#   athenea_hydra/plugin/usd/lib, include   on Linux, CUDA's NVRTC and the headers it compiles
#                                    against (ATHENEA_CUDA_RUNTIME), where the engine points
#                                    CUDA_PATH
#   athenea_hydra/plugin/materialx/  MaterialX 1.39.5's libraries/
#   athenea_hydra/plugin/aofx/       the Mesh2Splat, SplatBakeFilter, SplatTransferZonal and Measure bundles
#   athenea_hydra/shaders/           found by hdAthenea three levels up
#   athenea_hydra/LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, THIRD_PARTY_LICENSES.md
#   athenea_hydra/licenses/<component>/   every third party's own licence files
#
# THIRD_PARTY_LICENSES.md (integrations/blender) lists what the package carries
# and what it leaves to Blender; this script copies each licence file it names
# from the toolchain's trees and stops if one is missing, so a package never
# goes out without them.
#
# Usage: scripts/package-blender-addon.sh [build dir] [output dir]
#   (defaults build/<os>-blender and build/blender-addon)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
case "$(uname -s)" in
    Darwin) OS=macos; PLATFORM=macos-arm64; DEFAULT_BUILD=macos-arm64-blender ;;
    Linux)  OS=linux; PLATFORM=linux-x64;   DEFAULT_BUILD=linux-x86_64-blender ;;
    *) echo "package-blender-addon: $(uname -s) is neither macOS nor Linux" >&2; exit 1 ;;
esac
BUILD="${1:-$REPO/build/$DEFAULT_BUILD}"
OUT="${2:-$REPO/build/blender-addon}"
mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
SLANG_ROOT="${SLANG_ROOT:-$HOME/tools/slang}"
SRC_ROOT="${ATHENEA_TOOLS_SRC:-$HOME/tools/src}"
OIDN_ROOT="${ATHENEA_OIDN_ROOT:-$HOME/tools/oidn-2.5.1}"
OPENVDB_SRC="${ATHENEA_OPENVDB_SRC:-$HOME/tools/usd-26.08-mx/src/openvdb-10.1.0}"
MATERIALX_SRC="${ATHENEA_MATERIALX_SLANG_SRC:-$SRC_ROOT/MaterialX-1.39.5}"
USD_SRC="${ATHENEA_USD_SRC:-$SRC_ROOT/OpenUSD-26.08}"
TBB_SRC="${ATHENEA_TBB_SRC:-$SRC_ROOT/oneTBB-2022.3.0}"
OCIO_SRC="${ATHENEA_OCIO_SRC:-$SRC_ROOT/OpenColorIO-2.5.2}"

fail() { echo "package-blender-addon: $*" >&2; exit 1; }

PLUGIN="$BUILD/plugin/usd/hdAthenea/hdAthenea.so"
[[ -f "$PLUGIN" ]] || fail "no $PLUGIN: cmake --build --preset $DEFAULT_BUILD first"

ADDON="$OUT/athenea_hydra"
rm -rf "$ADDON"
mkdir -p "$ADDON/plugin" "$ADDON/licenses"

# --- the add-on and what it loads -------------------------------------------------
cp "$REPO"/integrations/blender/athenea_hydra/*.py "$REPO/integrations/blender/athenea_hydra/blender_manifest.toml" "$ADDON/"
# One zip a system: Blender installs a zip only where its manifest says it runs.
sed -i.bak -E "s/^platforms = .*/platforms = [\"$PLATFORM\"]/" "$ADDON/blender_manifest.toml" && rm "$ADDON/blender_manifest.toml.bak"
cp -R "$BUILD/plugin/usd" "$ADDON/plugin/usd"
cp -R "$BUILD/plugin/materialx" "$ADDON/plugin/materialx"
cp -R "$BUILD/shaders" "$ADDON/shaders"
rm -rf "$ADDON/shaders/athenea/test"
mkdir -p "$ADDON/plugin/aofx"
for bundle in Mesh2Splat SplatBakeFilter SplatTransferZonal Measure; do
    [[ -d "$BUILD/aofx/$bundle.aofx.bundle" ]] || fail "no $bundle bundle in $BUILD/aofx"
    cp -R "$BUILD/aofx/$bundle.aofx.bundle" "$ADDON/plugin/aofx/"
done
if [[ "$OS" == macos ]]; then
    # hdAthenea's rpath starts at @loader_path: Slang beside it is the one it loads.
    cp "$SLANG_ROOT"/lib/libslang-compiler.0.*.dylib "$SLANG_ROOT"/lib/libslang-glsl-module-*.dylib \
        "$ADDON/plugin/usd/hdAthenea/"

    # zstd and libwebp (SPZ and SOG) are Homebrew's on the build machine: copied
    # beside hdAthenea, renamed to @loader_path and signed ad hoc. Blender 5.3
    # carries com.apple.security.cs.disable-library-validation, so it loads
    # libraries signed by no team (proposal 084).
    HOMEBREW="${HOMEBREW_PREFIX:-/opt/homebrew}"
    HERE="$ADDON/plugin/usd/hdAthenea"
    carry() {   # carry <library path>: beside hdAthenea, its id @loader_path/<name>
        local name; name="$(basename "$1")"
        [[ -f "$1" ]] || fail "no $1 to carry (brew install zstd webp)"
        cp "$1" "$HERE/$name"
        chmod u+w "$HERE/$name"
        install_name_tool -id "@loader_path/$name" "$HERE/$name" 2>/dev/null
    }
    carry "$HOMEBREW/opt/zstd/lib/libzstd.1.dylib"
    carry "$HOMEBREW/opt/webp/lib/libwebp.7.dylib"
    carry "$HOMEBREW/opt/webp/lib/libsharpyuv.0.dylib"
    install_name_tool -change "@rpath/libsharpyuv.0.dylib" "@loader_path/libsharpyuv.0.dylib" "$HERE/libwebp.7.dylib" 2>/dev/null
    for name in libzstd.1.dylib libwebp.7.dylib; do
        old="$(otool -L "$HERE/hdAthenea.so" | awk -v n="$name" '$1 ~ n {print $1}')"
        [[ -n "$old" ]] && install_name_tool -change "$old" "@loader_path/$name" "$HERE/hdAthenea.so" 2>/dev/null
    done
    if otool -L "$HERE/hdAthenea.so" | grep -q "$HOMEBREW"; then
        fail "hdAthenea still names a library under $HOMEBREW"
    fi
    for library in "$HERE"/*.dylib "$HERE/hdAthenea.so"; do
        codesign --force -s - "$library" 2>/dev/null || fail "could not sign $library"
    done
else
    # Linux: Slang and the system's zstd and libwebp (SPZ and SOG), which a
    # machine may not have, beside hdAthenea, each under the name it is loaded
    # by and dereferenced: Blender unpacks an extension with Python's zipfile,
    # which keeps no symbolic link. Blender's own libraries (OpenUSD, TBB,
    # OIIO, MaterialX...) are already in the process by their sonames. Every
    # runpath is $ORIGIN, the build machine's dropped.
    HERE="$ADDON/plugin/usd/hdAthenea"
    command -v patchelf >/dev/null || fail "patchelf is needed on Linux"
    carry_so() {   # carry_so <file> [name]: copied whole under its soname, or the name given
        local name="${2:-$(patchelf --print-soname "$1" 2>/dev/null)}"
        [[ -n "$name" ]] || name="$(basename "$1")"
        cp -L "$1" "$3/$name"
    }
    for library in "$SLANG_ROOT"/lib/libslang-compiler.so.0.* "$SLANG_ROOT"/lib/libslang-glsl-module-*.so \
                   "$SLANG_ROOT"/lib/libslang-glslang-*.so; do
        [[ -f "$library" ]] && carry_so "$library" "" "$HERE"
    done
    for name in libzstd.so.1 libwebp.so.7 libsharpyuv.so.0; do
        found="$(ldd "$PLUGIN" | awk -v n="$name" '$1 == n {print $3}')"
        [[ -n "$found" && -f "$found" ]] || fail "hdAthenea does not load $name from a file here"
        carry_so "$found" "$name" "$HERE"
    done
    # CUDA's runtime where the engine looks for a package's own (plugin/usd/lib
    # and plugin/usd/include, beside hdAthenea's folder), as on Windows:
    # ATHENEA_CUDA_RUNTIME names a folder with lib/ and include/. NVRTC goes in
    # as libnvrtc.so, the one name slang-rhi asks for; its builtins and
    # nvJitLink under the sonames NVRTC loads them by.
    CUDA_RT="${ATHENEA_CUDA_RUNTIME:-}"
    [[ -e "$CUDA_RT/lib/libnvrtc.so" && -f "$CUDA_RT/include/cuda_runtime.h" ]] ||
        fail "ATHENEA_CUDA_RUNTIME must name a folder with lib/libnvrtc.so and include/cuda_runtime.h"
    CUDA_LIB="$ADDON/plugin/usd/lib"
    mkdir -p "$CUDA_LIB" "$ADDON/plugin/usd/include"
    carry_so "$CUDA_RT/lib/libnvrtc.so" libnvrtc.so "$CUDA_LIB"
    for library in "$CUDA_RT"/lib/libnvrtc-builtins.so.*.*.* "$CUDA_RT"/lib/libnvJitLink.so.*.*.*; do
        [[ -f "$library" ]] && carry_so "$library" "" "$CUDA_LIB"
    done
    cp -RL "$CUDA_RT/include/." "$ADDON/plugin/usd/include/"
    find "$HERE" "$CUDA_LIB" -maxdepth 1 -type f -name '*.so*' | while read -r library; do
        chmod u+w "$library"
        patchelf --set-rpath '$ORIGIN' "$library" 2>/dev/null || true
    done
    patchelf --set-rpath '$ORIGIN' "$HERE/hdAthenea.so"
    find "$ADDON/plugin/aofx" -type f -name '*.aofx' | while read -r bundle; do
        patchelf --set-rpath '$ORIGIN/../../../../usd/hdAthenea' "$bundle"
    done
    if find "$ADDON" -type l | grep -q .; then fail "a symbolic link in the add-on: $(find "$ADDON" -type l | head -1)"; fi
fi

# --- licences ---------------------------------------------------------------------
# licence <component> <file>...: each file into licenses/<component>/, keeping
# its name (a directory is copied whole).
licence() {
    local component="$1"; shift
    mkdir -p "$ADDON/licenses/$component"
    for file in "$@"; do
        [[ -e "$file" ]] || fail "licence file missing for $component: $file"
        cp -R "$file" "$ADDON/licenses/$component/"
    done
}

cp "$REPO/LICENSE" "$REPO/NOTICE" "$REPO/THIRD_PARTY_NOTICES.md" \
   "$REPO/integrations/blender/THIRD_PARTY_LICENSES.md" "$ADDON/"

# Carried by the package.
licence slang "$SLANG_ROOT/LICENSE" "$SLANG_ROOT/LICENSES"
licence materialx "$MATERIALX_SRC/LICENSE" "$MATERIALX_SRC/THIRD-PARTY.md"
if [[ "$OS" == macos ]]; then
    licence zstd "$HOMEBREW/opt/zstd/LICENSE" "$HOMEBREW/opt/zstd/COPYING"
    licence libwebp "$HOMEBREW/opt/webp/COPYING"
else
    licence zstd /usr/share/doc/libzstd1/copyright
    licence libwebp /usr/share/doc/libwebp7/copyright
    licence cuda "$(ls "$CUDA_RT"/LICENSE "$CUDA_RT"/EULA.txt 2>/dev/null | head -1)"
fi
if [[ -d "$ADDON/shaders/nanovdb" ]]; then
    licence openvdb "$OPENVDB_SRC/LICENSE"
fi

# Blender's own, compiled against: their headers' code is in hdAthenea.
licence openusd "$USD_SRC/LICENSE.txt" "$USD_SRC/NOTICE.txt"
licence onetbb "$TBB_SRC/LICENSE.txt" "$TBB_SRC/third-party-programs.txt"
licence opencolorio "$OCIO_SRC/LICENSE" "$OCIO_SRC/THIRD-PARTY.md"
OIDN_DOC="$OIDN_ROOT/share/doc/OpenImageDenoise"
licence oidn "$OIDN_DOC/LICENSE.txt" "$OIDN_DOC/third-party-programs.txt" \
    "$OIDN_DOC/third-party-programs-oneTBB.txt" "$OIDN_DOC/third-party-programs-DPCPP.txt"

# --- the zip ----------------------------------------------------------------------
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ADDON/blender_manifest.toml")"
ZIP="$OUT/athenea_hydra-$VERSION-$PLATFORM.zip"
rm -f "$ZIP"
if command -v zip >/dev/null; then
    (cd "$ADDON" && zip -qry "$ZIP" .)
else
    python3 - "$ADDON" "$ZIP" <<'PY'
import os, sys, zipfile
root, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for here, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(here, name)
            info = zipfile.ZipInfo.from_file(path, os.path.relpath(path, root))
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(path, "rb") as f:
                z.writestr(info, f.read())
PY
fi
echo "$ADDON"
echo "$ZIP"
