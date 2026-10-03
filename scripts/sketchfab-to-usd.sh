#!/usr/bin/env bash
# A Sketchfab download turned into a USD stage this engine can open.
#
# Sketchfab hands out a zip: usually glTF (scene.gltf + a .bin + textures),
# sometimes the author's own FBX, OBJ or .blend. Whatever is in it, Blender
# reads it and writes USD with its materials and textures, which is the same
# road the Fox took (scripts/fetch-fox.sh) and for the same reason -- nothing
# else on this machine reads glTF materials into USD.
#
# The download itself is not here: it needs the account that accepted the
# model's licence, so it is yours to make. The models do not go in this
# repository either; this writes beside the others in ~/tools/assets.
#
#   scripts/sketchfab-to-usd.sh <archive.zip> [name] [assets]
#
# `name` names the asset and its stage (default: the archive's own name), and
# `assets` is where it lands (default ~/tools/assets/<name>).
set -euo pipefail

ARCHIVE="${1:?usage: sketchfab-to-usd.sh <archive.zip> [name] [assets]}"
NAME="${2:-$(basename "${ARCHIVE%.*}")}"
DEST="${3:-$HOME/tools/assets/$NAME}"
BLENDER="${ATHENEA_BLENDER:-$(command -v blender || echo /Applications/Blender.app/Contents/MacOS/Blender)}"

[[ -f "$ARCHIVE" ]] || { echo "no archive at $ARCHIVE" >&2; exit 1; }
[[ -x "$BLENDER" ]] || { echo "no Blender at $BLENDER: set ATHENEA_BLENDER" >&2; exit 1; }

mkdir -p "$DEST/source"
echo "athenea: unpacking into $DEST/source"
unzip -o -q "$ARCHIVE" -d "$DEST/source"

# What to import: glTF first (Sketchfab's own conversion, materials included),
# then the author's formats. The largest candidate wins where there are
# several, since a Sketchfab zip often carries a low-poly preview beside the
# real thing.
pick() {
    find "$DEST/source" -type f -iname "$1" -not -path '*/__MACOSX/*' -print0 |
        xargs -0 ls -S 2>/dev/null | head -1
}
SOURCE=""
for pattern in '*.gltf' '*.glb' '*.fbx' '*.blend' '*.obj' '*.dae' '*.usdz' '*.usdc' '*.usda'; do
    SOURCE="$(pick "$pattern" || true)"
    [[ -n "$SOURCE" ]] && break
done
[[ -n "$SOURCE" ]] || { echo "nothing importable in $ARCHIVE" >&2; exit 1; }
echo "athenea: importing $(basename "$SOURCE")"

EXT="${SOURCE##*.}"
IMPORT="bpy.ops.import_scene.gltf(filepath=r'$SOURCE')"
case "$(echo "$EXT" | tr "[:upper:]" "[:lower:]")" in
    fbx)   IMPORT="bpy.ops.import_scene.fbx(filepath=r'$SOURCE')" ;;
    obj)   IMPORT="bpy.ops.wm.obj_import(filepath=r'$SOURCE')" ;;
    dae)   IMPORT="bpy.ops.wm.collada_import(filepath=r'$SOURCE')" ;;
    blend) IMPORT="bpy.ops.wm.open_mainfile(filepath=r'$SOURCE')" ;;
    usd*)  IMPORT="bpy.ops.wm.usd_import(filepath=r'$SOURCE')" ;;
esac

# The orientation is left exactly as the file had it. Blender is already Z up
# and -Y forward, and asking the exporter to "convert" to that turned the car
# around: a camera placed by hand saw its back where Blender saw its front,
# which is the kind of difference nobody notices until two renderers are put
# side by side.
"$BLENDER" --background --factory-startup --python-expr "
import bpy, math
if '$EXT' != 'blend':
    bpy.ops.wm.read_factory_settings(use_empty=True)
$IMPORT
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not meshes:
    raise SystemExit('athenea: the import produced no mesh')
print('athenea: %d meshes, %d materials' % (len(meshes), len(bpy.data.materials)))

# A WORLD WHOSE IMAGE IS NOT HERE IS NOT A WORLD.
#
# A Sketchfab .blend keeps the author's environment map by absolute path --
# on the author's machine, inside their Windows temp folder. Blender exports
# the world as a DomeLight all the same, pointing at a file that does not
# exist; a renderer then lights the scene with what a dome with no image is,
# which is a white one at full strength. Measured on the Mustang: the stage
# came out lit by an extra dome of radiance 1, three times the sky anybody
# added afterwards, and every material looked washed out and wrong.
# THE TEXTURES ARE IN THE ZIP; THE PATHS ARE NOT, AND NEITHER ARE THE NAMES.
#
# A .blend keeps its images by the path the author had them at -- a Windows
# temp folder, in the Mustang's case -- so out of the box Blender loads none
# of them and every textured material renders as its fallback. The zip does
# carry them, beside the .blend, but Sketchfab renames them on the way in:
# 'carpet_Base Color.jpg' in the blend is 'carpet_BaseColor.jpeg' in the zip.
# So find_missing_files, which matches by name, finds nothing, and the
# match has to ignore what the renaming changes -- spaces, punctuation, case
# and the extension. Measured on the Mustang: 22 images missing before, 0
# after, and half the car's materials stopped rendering as flat colour.
import os, re

def key(name):
    return re.sub(r'[^a-z0-9]', '', os.path.splitext(name)[0].lower())

beside = {}
for root, _dirs, files in os.walk(r'$DEST/source'):
    for name in files:
        beside.setdefault(key(name), os.path.join(root, name))

def here(image):
    # Whether the file is where the image says, which has_data does not
    # answer: an image Blender has not been asked to draw yet has no data
    # however good its path is.
    return bool(image.filepath) and os.path.exists(bpy.path.abspath(image.filepath))

missing = [image for image in bpy.data.images if image.source == 'FILE' and image.filepath and not here(image)]
found = 0
for image in missing:
    at = beside.get(key(os.path.basename(image.filepath.replace(chr(92), '/'))))
    if at:
        try:
            image.filepath = at
            image.reload()   # a packed image refuses, and says so; the path still points home
        except Exception as e:
            print('athenea: %s: %s' % (image.name, e))
        found += here(image)
if missing:
    still = [image.name for image in bpy.data.images
             if image.source == 'FILE' and image.filepath and not here(image)]
    print('athenea: %d image(s) the paths did not reach, %d found beside the file%s'
          % (len(missing), found, (', %d still missing: %s' % (len(still), ', '.join(still[:4]))) if still else ''))
# The exporter copies a texture under the image datablock's name and writes
# the path under the source file's; where a .blend was made on one machine and
# packaged on another those two differ (carpet_Base Color.jpg against
# carpet_BaseColor.jpeg) and the network points at a file nobody wrote. Naming
# the datablock after its file makes the two agree.
for image in bpy.data.images:
    if image.source == 'FILE' and image.filepath:
        stem = os.path.basename(bpy.path.abspath(image.filepath))
        if stem and image.name != stem:
            image.name = stem

world = bpy.context.scene.world
if world is not None and world.use_nodes:
    for node in world.node_tree.nodes:
        if node.type == 'TEX_ENVIRONMENT' and (node.image is None or not node.image.has_data):
            print('athenea: the world image is missing; the stage is exported without it')
            bpy.context.scene.world = None
            break
# MATERIALX AS WELL AS UsdPreviewSurface.
#
# UsdPreviewSurface has no transmission, so the exporter says what it can:
# a Principled BSDF with Transmission 1 -- glass -- becomes an opacity of 0,
# which means: this surface is not there. Measured on the Mustang, every
# window was an open hole here while Cycles showed glass reflecting the sky.
# MaterialX keeps the transmission, and this engine's material language IS
# MaterialX, so the network is written beside the preview surface and a
# renderer takes whichever it understands.
bpy.ops.wm.usd_export(filepath=r'$DEST/$NAME.usdc',
                      export_materials=True,
                      export_textures_mode='NEW',
                      generate_preview_surface=True,
                      generate_materialx_network=True,
                      relative_paths=True)
" 2>&1 | grep -vE '^(Blender|Read prefs|found bundled|Warning)' || true

[[ -f "$DEST/$NAME.usdc" ]] || { echo "the export produced no $NAME.usdc" >&2; exit 1; }
echo "wrote $DEST/$NAME.usdc"
echo "  look at it:  build/macos-arm64-release/bin/athenea view $DEST/$NAME.usdc --frame-all"
