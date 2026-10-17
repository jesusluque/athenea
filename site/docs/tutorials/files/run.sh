#!/bin/bash
# Copyright (c) 2026 jesus luque.
# The four steps of the athenea tutorials on the Open Chess Set pawn.
# Expects ../aswf-assets (git sparse checkout of usd-wg/assets, full_assets/OpenChessSet) beside this folder,
# and ATHENEA pointing at the binary (default: athenea on PATH).
set -e
cd "$(dirname "$0")"
A=${ATHENEA:-athenea}
$A mesh2splat mesh.usda --prim /World/Pawn --cell-from-camera /World/Camera --camera-pixels 1024 --transfer --bake-samples 128 -o pawn_gs.usdc
$A stage gs.usda --camera /World/Camera -o gs.exr
$A stage mesh.usda --camera /World/Camera --technique rt --path-total 1024 --denoise -o pt.exr
$A compare gs.exr pt.exr
