#!/usr/bin/env bash
# Copyright (c) 2026 jesus luque.
# sparrow-shadow-frame.sh FRAME WIDTHxHEIGHT PATHS OUTDIR
# The four renders a shadow-catcher frame is composited from (docs/decisions.md,
# "A shadow catcher by composition"): A the bird over a white ground, B the
# ground alone, C the bird alone over the sky, M the bird alone with no dome,
# whose alpha is its coverage.
#   SPARROW   the folder with the Film*.usda stages (required)
#   ATHENEA   the athenea binary (default: athenea on PATH)
# On Linux the splat shadows trace inline rays, which CUDA has not: set
# ATHENEA_BACKEND=vulkan there.
set -euo pipefail
f=$1; size=$2; paths=$3; out=$4
here=$(cd "$(dirname "$0")" && pwd)
A=${SPARROW:?set SPARROW to the folder with the sparrow film stages}
R=${ATHENEA:-athenea}
read -r -a row <<< "$(sed -n "${f}p" "$here/sparrow-film-camera.txt")"
cam=(--eye "${row[1]}" "${row[2]}" "${row[3]}" --target "${row[4]}" "${row[5]}" "${row[6]}" --up 0 0 1
     --focal "${row[7]}" --size "$size" --time "$f")
"$R" stage "$A/FilmGsWhite.usda"  --technique rt --path-total "$paths" --splat-shadows "${cam[@]}" -o "$out/A_$f.exr" >/dev/null 2>&1
"$R" stage "$A/FilmGsGround.usda" --technique rt --path-total "$paths" "${cam[@]}" -o "$out/B_$f.exr" >/dev/null 2>&1
"$R" stage "$A/FilmGsBird.usda"   --technique rt --path-total "$paths" "${cam[@]}" -o "$out/C_$f.exr" >/dev/null 2>&1
"$R" stage "$A/FilmGsMask.usda"   --technique rt --path-total 16 "${cam[@]}" -o "$out/M_$f.exr" >/dev/null 2>&1
