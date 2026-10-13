#!/bin/bash
# Copyright (c) 2026 jesus luque.
#
# The Linux installer: package.sh's folder as a .deb that installs to
# /opt/athenea and links its programs into /usr/bin. Built on Ubuntu 24.04; the
# NVIDIA driver (CUDA, OptiX) is the system's, not part of the package.
#
#   scripts/installer-linux.sh <package folder> [out dir]
set -euo pipefail
pkgdir=$(cd "${1:?usage: installer-linux.sh <package folder> [out dir]}" && pwd)
out=$(mkdir -p "${2:-$(dirname "$pkgdir")}" && cd "${2:-$(dirname "$pkgdir")}" && pwd)
name=$(basename "$pkgdir")
version=$(echo "$name" | sed -E 's/^athenea-([0-9.]+)-.*/\1/')
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/deb/opt" "$work/deb/usr/bin" "$work/deb/DEBIAN"
cp -a "$pkgdir" "$work/deb/opt/athenea"
for program in athenea athenea-mcp; do
    [ -x "$work/deb/opt/athenea/bin/$program" ] && ln -s "/opt/athenea/bin/$program" "$work/deb/usr/bin/$program"
done
size=$(du -sk "$work/deb/opt" | cut -f1)
cat > "$work/deb/DEBIAN/control" <<CONTROL
Package: athenea
Version: $version
Section: graphics
Priority: optional
Architecture: amd64
Installed-Size: $size
Depends: libc6 (>= 2.39), libstdc++6, libgl1, libx11-6
Recommends: nvidia-driver-570 | nvidia-driver
Maintainer: jesus luque (https://github.com/jesusluque)
Homepage: https://github.com/jesusluque/athenea
Description: Gaussian splats on the GPU, and a relight baker for them
 athenea renders OpenUSD stages and Gaussian splat clouds on the GPU (CUDA on
 Linux), converts meshes into Gaussians and bakes their light (mesh2splat).
CONTROL
dpkg-deb --root-owner-group --build "$work/deb" "$out/athenea_${version}_amd64.deb" >/dev/null
echo "$out/athenea_${version}_amd64.deb"
