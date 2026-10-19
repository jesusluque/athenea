#!/usr/bin/env bash
# Copyright (c) 2026 jesus luque.
# sparrow-shadow-film.sh [FIRST] [LAST]: every frame of the film as a shadow-catcher
# composite (sky x A/B where the bird is not, the bird where it is), then the mp4
# when the range is the whole. Frames already composited are skipped, so a range
# can be split across machines.
#   SPARROW   the folder with the Film*.usda stages (required)
#   OUT       where the frames go (default: $SPARROW/film_shadow)
#   ATHENEA   the athenea binary (default: athenea on PATH)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${SPARROW:?set SPARROW to the folder with the sparrow film stages}"
out=${OUT:-$SPARROW/film_shadow}
first=${1:-1}; last=${2:-410}
mkdir -p "$out"
for f in $(seq "$first" "$last"); do
    [ -f "$out/final_$(printf %03d "$f").png" ] && continue
    t0=$(date +%s)
    "$here/sparrow-shadow-frame.sh" "$f" 1920x1080 64 "$out"
    oiiotool "$out/A_$f.exr" --ch R,G,B "$out/B_$f.exr" --ch R,G,B --div --clamp:min=0:max=1 --subc 1 \
        --mulc 0.589,0.703,0.9025 "$out/M_$f.exr" --ch A,A,A --mulc -1 --addc 1 --mul "$out/C_$f.exr" --ch R,G,B \
        --add --colorconvert linear sRGB -o "$out/final_$(printf %03d "$f").png" 2>/dev/null
    rm -f "$out/A_$f.exr" "$out/B_$f.exr" "$out/C_$f.exr" "$out/M_$f.exr"
    echo "frame $f: $(( $(date +%s) - t0 )) s"
done
if [ "$first" = 1 ] && [ "$last" = 410 ]; then
    ffmpeg -y -framerate 30 -i "$out/final_%03d.png" -c:v libx264 -pix_fmt yuv420p -crf 16 "$out/Film_shadow.mp4" >/dev/null 2>&1
fi
echo FILM_DONE
