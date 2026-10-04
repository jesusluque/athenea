#!/usr/bin/env bash
# Copyright (c) 2026 jesus luque.
#
# Puts Dawn, the WebGPU implementation Chrome uses, into ~/tools/dawn-<version>
# for the WebGPU backend (ATHENEA_WEBGPU=ON, the macos-arm64-webgpu and
# linux-x86_64-webgpu presets).
#
# The prebuilt binaries slang-rhi is written against
# (github.com/shader-slang/webgpu-dawn-binaries), not a build from source:
# slang-rhi's WGPU backend loads exactly that libdawn by name and calls the
# webgpu.h of that release, and a Dawn of another revision is another C API.
# Building Dawn itself is an hour of a machine for the same library. The
# version and the hashes are slang-rhi's own (its CMakeLists.txt, at the pin in
# cmake/Dependencies.cmake); they move together.
#
# The configure step points slang-rhi's FetchPackage(dawn) at this directory
# (FETCHCONTENT_SOURCE_DIR_DAWN), so nothing is downloaded at build time.
#
# Usage: scripts/build-dawn.sh   (ATHENEA_DAWN_ROOT overrides where it goes)
set -euo pipefail

VERSION="138.0.7204.168"
PREFIX="${ATHENEA_DAWN_ROOT:-$HOME/tools/dawn-${VERSION}}"

case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)  PLATFORM=macos-aarch64
                   SHA256=8a17471681a5158a2c6cf607c2201384030d5a02404aeb87fddedd4d087c6285 ;;
    Darwin-x86_64) PLATFORM=macos-x86_64
                   SHA256=1727b4242c65d8be37b1225cb94eaf6d45dd2e715d6c07f83e380441fea8b936 ;;
    Linux-x86_64)  PLATFORM=linux-x86_64
                   SHA256=527216981b1e486ec03470ae5af1b7f8020fd2de042b4ace12b10281d08fe669 ;;
    Linux-aarch64) PLATFORM=linux-aarch64
                   SHA256=128a58d6ae967966af39003b05848d4ae59a7048a88a3b5374c94c73fbdba4b1 ;;
    *) echo "build-dawn.sh: no Dawn binaries for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

if [[ -f "$PREFIX/include/dawn/webgpu.h" ]]; then
    echo "Dawn ${VERSION} already at ${PREFIX}"
    exit 0
fi

ARCHIVE="webgpu-dawn-${VERSION}-${PLATFORM}.zip"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
curl -L --fail -o "$WORK/$ARCHIVE" \
    "https://github.com/shader-slang/webgpu-dawn-binaries/releases/download/v${VERSION}/${ARCHIVE}"
if command -v sha256sum >/dev/null; then
    GOT="$(sha256sum "$WORK/$ARCHIVE" | cut -d' ' -f1)"
else
    GOT="$(shasum -a 256 "$WORK/$ARCHIVE" | cut -d' ' -f1)"
fi
if [[ "$GOT" != "$SHA256" ]]; then
    echo "build-dawn.sh: $ARCHIVE has SHA-256 $GOT, expected $SHA256" >&2
    exit 1
fi
mkdir -p "$WORK/unpacked"
unzip -q "$WORK/$ARCHIVE" -d "$WORK/unpacked"
# The archive holds include/ and lib/ (lib64/ on Linux) at its root, or under
# one directory named after it.
ROOT="$WORK/unpacked"
if [[ ! -d "$ROOT/include" ]]; then
    ROOT="$(dirname "$(find "$WORK/unpacked" -type d -name include -maxdepth 2 | head -1)")"
fi
mkdir -p "$PREFIX"
cp -R "$ROOT"/. "$PREFIX"/
echo "Dawn ${VERSION} installed at ${PREFIX}"
