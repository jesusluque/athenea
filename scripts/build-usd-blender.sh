#!/usr/bin/env bash
# Copyright (c) 2026 jesus luque.
#
# The headers hdAthenea is compiled against to run inside Blender, into
# ~/tools/usd-<version>-blender. Headers only: at link time and at run time
# the plugin uses Blender's own libusd_ms, libtbb, libMaterialX*, OCIO and
# OIDN (cmake/BlenderUsd.cmake), so the process holds one of each.
#
# What has to match Blender's build, from its
# build_files/build_environment/cmake/usd.cmake and versions.cmake:
#   - the OpenUSD tag (Blender 5.3 alpha ships 26.03: `nm libusd_ms.dylib`
#     shows pxrBlender_v26_03__pxrReserved__);
#   - PXR_SET_INTERNAL_NAMESPACE=pxrBlender_v<version>, every symbol's name;
#   - Python support ON: it changes VtValue's type-info table and
#     TfAnyWeakPtr's vtable, so a plugin compiled without it builds a VtValue
#     Blender's USD reads wrongly. Python 3.13's headers only (uv's
#     standalone 3.13; Blender ships pyconfig.h alone);
#   - Blender's usd_ctor.diff: ARCH_CONSTRUCTOR entries go in a section
#     named "pxbctor" (not "pxrctor"), and Blender's libusd_ms runs only
#     those. Unpatched, a plugin's TF_REGISTRY_FUNCTIONs -- its TfType, its
#     renderer plugin -- never run. Applied here as the rename it is;
#   - oneTBB 2022.3 and MaterialX 1.39.4 headers (namespace MaterialX_v1_39_4);
#   - OpenSubdiv 3.7.0 headers, against Blender's libosdCPU/GPU.
# The PXR_*_SUPPORT_ENABLED definitions (GL, Metal, OpenVDB, OIIO) are USD's
# own compile flags, not in any installed header, so OpenVDB and the OIIO
# plugin are left off here without changing what a consumer compiles.
#
# Usage: scripts/build-usd-blender.sh [usd-tag]   (default v26.03)
set -euo pipefail

VERSION="${1:-v26.03}"
V="${VERSION#v}"
PREFIX="${ATHENEA_USD_BLENDER_ROOT:-$HOME/tools/usd-${V}-blender}"
SRC="${ATHENEA_SRC:-$HOME/tools/src}"
WORK="${PREFIX}-build"
JOBS="${JOBS:-6}"
NAMESPACE="pxrBlender_v${V//./_}"
TBB_TAG=v2022.3.0
MATERIALX_TAG=v1.39.4
OPENSUBDIV_TAG=v3_7_0
BLENDER_LIB="${BLENDER_LIB:-/Applications/Blender.app/Contents/Resources/lib}"

clone() {   # url tag dir
    [[ -d "$3/.git" ]] || git clone -q --depth 1 --branch "$2" "$1" "$3"
}
mkdir -p "$SRC" "$WORK" "$PREFIX"
clone https://github.com/PixarAnimationStudios/OpenUSD.git "$VERSION" "$SRC/OpenUSD-$V"
clone https://github.com/uxlfoundation/oneTBB.git "$TBB_TAG" "$SRC/oneTBB-${TBB_TAG#v}"
clone https://github.com/AcademySoftwareFoundation/MaterialX.git "$MATERIALX_TAG" "$SRC/MaterialX-${MATERIALX_TAG#v}"
clone https://github.com/PixarAnimationStudios/OpenSubdiv.git "$OPENSUBDIV_TAG" "$SRC/OpenSubdiv-${OPENSUBDIV_TAG#v}"

PYTHON="${ATHENEA_PYTHON313:-$(uv python find 3.13 2>/dev/null || true)}"
if [[ -z "$PYTHON" ]]; then
    echo "Python 3.13 is needed for its headers: uv python install 3.13" >&2
    exit 1
fi
PY_INCLUDE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PY_LIB="$("$PYTHON" -c 'import sysconfig,os; print(os.path.join(sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY")))')"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# oneTBB: built (it is small) so USD's configure finds a TBBConfig; the
# library is never linked by the plugin.
cmake -S "$SRC/oneTBB-${TBB_TAG#v}" -B "$WORK/tbb" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DTBB_TEST=OFF -DTBB_EXAMPLES=OFF -DTBBMALLOC_BUILD=OFF > /dev/null
cmake --build "$WORK/tbb" -j "$JOBS" && cmake --install "$WORK/tbb" > /dev/null

# MaterialX 1.39.4, shared, without Python, viewer or tests.
cmake -S "$SRC/MaterialX-${MATERIALX_TAG#v}" -B "$WORK/materialx" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DMATERIALX_BUILD_SHARED_LIBS=ON -DMATERIALX_BUILD_PYTHON=OFF \
    -DMATERIALX_BUILD_VIEWER=OFF -DMATERIALX_BUILD_GRAPH_EDITOR=OFF -DMATERIALX_BUILD_TESTS=OFF \
    -DMATERIALX_BUILD_GEN_MDL=OFF -DMATERIALX_BUILD_GEN_OSL=OFF -DMATERIALX_BUILD_RENDER=ON > /dev/null
cmake --build "$WORK/materialx" -j "$JOBS" && cmake --install "$WORK/materialx" > /dev/null

# OpenSubdiv 3.7.0: its headers, installed beside Blender's libraries.
mkdir -p "$PREFIX/include/opensubdiv" "$PREFIX/lib"
( cd "$SRC/OpenSubdiv-${OPENSUBDIV_TAG#v}/opensubdiv" && find . -name '*.h' -exec rsync -R {} "$PREFIX/include/opensubdiv/" \; )
ln -sfn "$BLENDER_LIB/libosdCPU.dylib" "$PREFIX/lib/libosdCPU.dylib"
ln -sfn "$BLENDER_LIB/libosdGPU.dylib" "$PREFIX/lib/libosdGPU.dylib"

# OpenUSD, Blender's way. Blender's patch, as the rename it is.
USD_SRC="$SRC/OpenUSD-$V"
for f in pxr/base/arch/attributes.h pxr/base/arch/attributes.cpp; do
    sed -i '' -e 's/pxrctor/pxbctor/g' -e 's/pxrdtor/pxbdtor/g' "$USD_SRC/$f"
done
cmake -S "$USD_SRC" -B "$WORK/usd" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DPXR_SET_INTERNAL_NAMESPACE="$NAMESPACE" \
    -DPXR_BUILD_MONOLITHIC=ON \
    -DPXR_ENABLE_PYTHON_SUPPORT=ON -DPXR_USE_PYTHON_3=ON \
    -DPython3_EXECUTABLE="$PYTHON" -DPython3_INCLUDE_DIR="$PY_INCLUDE" -DPython3_LIBRARY="$PY_LIB" \
    -DPXR_BUILD_IMAGING=ON -DPXR_BUILD_USD_IMAGING=ON -DPXR_ENABLE_GL_SUPPORT=ON \
    -DPXR_ENABLE_MATERIALX_SUPPORT=ON -DMaterialX_DIR="$PREFIX/lib/cmake/MaterialX" \
    -DOPENSUBDIV_ROOT_DIR="$PREFIX" \
    -DPXR_ENABLE_OPENVDB_SUPPORT=OFF -DPXR_BUILD_OPENIMAGEIO_PLUGIN=OFF -DPXR_BUILD_OPENCOLORIO_PLUGIN=OFF \
    -DPXR_ENABLE_PTEX_SUPPORT=OFF -DPXR_ENABLE_OSL_SUPPORT=OFF -DPXR_ENABLE_HDF5_SUPPORT=OFF \
    -DPXR_BUILD_EMBREE_PLUGIN=OFF -DPXR_BUILD_USDVIEW=OFF -DPXR_BUILD_USD_TOOLS=OFF \
    -DPXR_BUILD_TESTS=OFF -DPXR_BUILD_EXAMPLES=OFF -DPXR_BUILD_TUTORIALS=OFF -DPXR_BUILD_DOCUMENTATION=OFF
# Only the headers: every library's header copy, then the tree they land in.
HEADER_TARGETS=$(ninja -C "$WORK/usd" -t targets all | sed -n 's/^\([A-Za-z0-9_]*_headerfiles\):.*/\1/p')
ninja -C "$WORK/usd" -j "$JOBS" $HEADER_TARGETS > /dev/null
rsync -a "$WORK/usd/include/" "$PREFIX/include/"
# The plugInfo/schemas are Blender's (lib/usd); nothing to install.
grep -q "${NAMESPACE}__pxrReserved__" "$PREFIX/include/pxr/pxr.h" || { echo "pxr.h: namespace is not ${NAMESPACE}" >&2; exit 1; }
grep -q pxbctor "$PREFIX/include/pxr/base/arch/attributes.h" || { echo "attributes.h is not Blender's" >&2; exit 1; }
echo "${NAMESPACE} headers at ${PREFIX}/include (Python ${PY_INCLUDE})"
echo "Configure athenea with --preset macos-arm64-blender."
