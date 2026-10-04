#!/bin/bash
# Copyright (c) 2026 jesus luque.
# Phase 2 of matx (proposal 086 §2): the OpenPBR examples on the shader ball, fetched from the
# OpenPBR repository at a pinned commit (Apache-2.0) into <MATX_WORK>/cache/openpbr. CPU and
# network only. Re-running skips what is there; every file is checked against openpbr.sha256
# (in git beside this script), so a changed upstream file fails here, not in a measure.
#   bash fetch_phase2.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
COMMIT=$(python3 -c "import matx_common as m; print(m.OPENPBR_COMMIT)")
CACHE=$(python3 -c "import matx_common as m; print(m.OPENPBR_CACHE)")
NAMES=$(python3 -c "import matx_common as m; print(' '.join(m.PHASE2_OPENPBR))")
URL="https://raw.githubusercontent.com/AcademySoftwareFoundation/OpenPBR/$COMMIT"
mkdir -p "$CACHE"
[ -f "$CACHE/LICENSE" ] || curl -fsSL "$URL/LICENSE" -o "$CACHE/LICENSE"
for name in $NAMES; do
  f="open_pbr_$name.mtlx"
  [ -s "$CACHE/$f" ] || { curl -fsSL "$URL/examples/$f" -o "$CACHE/$f.part" && mv "$CACHE/$f.part" "$CACHE/$f"; echo "fetched $f"; }
done
cd "$CACHE"
shasum -a 256 -c "$HERE/openpbr.sha256" --quiet && echo "$CACHE: all match openpbr.sha256 ($COMMIT)"
