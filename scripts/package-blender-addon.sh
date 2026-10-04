#!/usr/bin/env bash
# Copyright (c) 2026 jesus luque.
#
# The Blender add-on as one directory and a zip Blender installs from disk,
# from a macos-arm64-blender build:
#
#   athenea_hydra/*.py, blender_manifest.toml
#   athenea_hydra/plugin/usd/        hdAthenea (with libslang, libzstd and libwebp beside it,
#                                    named @loader_path and signed ad hoc) and the schemas
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
#   (defaults build/macos-arm64-blender and build/blender-addon)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$REPO/build/macos-arm64-blender}"
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
[[ -f "$PLUGIN" ]] || fail "no $PLUGIN: cmake --build --preset macos-arm64-blender first"

ADDON="$OUT/athenea_hydra"
rm -rf "$ADDON"
mkdir -p "$ADDON/plugin" "$ADDON/licenses"

# --- the add-on and what it loads -------------------------------------------------
cp "$REPO"/integrations/blender/athenea_hydra/*.py "$REPO/integrations/blender/athenea_hydra/blender_manifest.toml" "$ADDON/"
cp -R "$BUILD/plugin/usd" "$ADDON/plugin/usd"
cp -R "$BUILD/plugin/materialx" "$ADDON/plugin/materialx"
cp -R "$BUILD/shaders" "$ADDON/shaders"
rm -rf "$ADDON/shaders/athenea/test"
mkdir -p "$ADDON/plugin/aofx"
for bundle in Mesh2Splat SplatBakeFilter SplatTransferZonal Measure; do
    [[ -d "$BUILD/aofx/$bundle.aofx.bundle" ]] || fail "no $bundle bundle in $BUILD/aofx"
    cp -R "$BUILD/aofx/$bundle.aofx.bundle" "$ADDON/plugin/aofx/"
done
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
licence zstd "$HOMEBREW/opt/zstd/LICENSE" "$HOMEBREW/opt/zstd/COPYING"
licence libwebp "$HOMEBREW/opt/webp/COPYING"
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
ZIP="$OUT/athenea_hydra-$VERSION.zip"
rm -f "$ZIP"
(cd "$ADDON" && zip -qry "$ZIP" .)
echo "$ADDON"
echo "$ZIP"
