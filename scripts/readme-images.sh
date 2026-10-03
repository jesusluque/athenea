#!/bin/zsh
# The README's pictures, made again from the assets rather than kept by hand.
#
# The sparrow lives in ~/tools/assets/Sparrow, outside the repo (models do not
# go in it); only these PNGs do. Everything here is one `athenea stage` call and an
# oiiotool conversion, so what the README shows is what the engine draws today.
#
#   scripts/readme-images.sh [outdir]      default: docs/images
set -e

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
out=${1:-$repo/docs/images}
athenea=$repo/build/macos-arm64-release/bin/athenea
assets=$HOME/tools/assets/Sparrow
work=$(mktemp -d)
trap 'rm -rf $work' EXIT
mkdir -p $out

[ -x $athenea ] || { echo "no release build at $athenea" >&2; exit 1; }
[ -f $assets/Sparrow_vis.usdc ] || { echo "no sparrow at $assets" >&2; exit 1; }

# A sky and a sun over whichever sparrow is asked for: the cloud with its baked
# visibility field, or the mesh it was converted from. The same two lights, so
# the two pictures are comparable.
lit() {   # lit <layer> <file>
    cat > $work/$2 <<EOF
#usda 1.0
(
    defaultPrim = "World"
    startTimeCode = 1
    endTimeCode = 61
    timeCodesPerSecond = 30
    upAxis = "Z"
    subLayers = [ @$1@ ]
)
def Scope "ReadmeLights"
{
    def DomeLight "Sky"
    {
        float inputs:intensity = 0.7
        color3f inputs:color = (0.55, 0.66, 0.85)
    }
    def DistantLight "Sun"
    {
        float inputs:intensity = 3.2
        float inputs:angle = 1.2
        bool inputs:normalize = 1
        color3f inputs:color = (1, 0.94, 0.82)
        double3 xformOp:rotateXYZ = (-52, 0, 36)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
}
EOF
}

# The pose the wings are widest in, and a camera close enough to see a barb.
pose=(--time 24 --eye 0.16 -0.16 0.06 --target 0 0 0.035 --up 0 0 1 --focal 35)

lit $assets/Sparrow_vis.usdc cloud.usda
lit $assets/SparrowBird.usda mesh.usda

echo "1/3  the cloud, rasterised"
$athenea stage $work/cloud.usda --technique raster --light-samples 16 \
    $pose --size 1280x800 -o $work/cloud.exr
oiiotool $work/cloud.exr --ch R,G,B --colorconvert linear sRGB -o $out/sparrow-cloud.png

# The same camera for both, so the two halves are the same picture of the same
# bird and only the representation differs.
echo "2/3  the mesh it was converted from, beside it"
$athenea stage $work/mesh.usda --technique rt --path-total 64 \
    $pose --size 1280x800 -o $work/mesh_full.exr
oiiotool $work/mesh_full.exr --ch R,G,B --colorconvert linear sRGB --resize 640x400 \
    --text:x=10:y=392:size=18:color=1,1,0.2 "geometry, path traced" -o $work/mesh_half.png
oiiotool $work/cloud.exr --ch R,G,B --colorconvert linear sRGB --resize 640x400 \
    --text:x=10:y=392:size=18:color=1,1,0.2 "gaussians, rasterised" -o $work/cloud_half.png
oiiotool $work/mesh_half.png $work/cloud_half.png --mosaic 2x1 -o $out/sparrow-mesh-cloud.png

echo "3/3  a frame of the film, its shadow caught by an invisible ground"
$here/film/sparrow-shadow-frame.sh 1 1280x720 64 $work
oiiotool $work/A_1.exr --ch R,G,B $work/B_1.exr --ch R,G,B --div \
    --clamp:min=0:max=1 --subc 1 --mulc 0.589,0.703,0.9025 \
    $work/M_1.exr --ch A,A,A --mulc -1 --addc 1 --mul \
    $work/C_1.exr --ch R,G,B --add --colorconvert linear sRGB \
    -o $out/sparrow-film.png

ls -la $out
