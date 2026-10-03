# Copyright (c) 2026 jesus luque.
"""The add-on's mesh2splat operator, headless, end to end.

Blender's default cube (its material made a colour, so a wrong colour space
shows) is rendered through hdAthenea as a mesh; then converted three ways --
referenced USD, imported Gaussian-splat points, a referenced .athc -- each
rendered through hdAthenea with the cube hidden and compared with the mesh's
render by `athenea compare` (the Measure effect). What is checked:

  - the operator finishes, writes its file next to the .blend, brings in an
    Empty (reference) or a POINTCLOUD of type GAUSSIAN_SPLAT (points), and
    hides the cube;
  - each splat render covers what the mesh covers (alpha means within 0.03)
    and is near it in colour (relMSE under 0.25, an 8-bit p99 under 40: a
    baked cloud at 128 cells and 64 paths is not the mesh, and these bound
    a gross failure -- wrong colour space, nothing drawn, the cloud elsewhere).

  blender -b --factory-startup --python-exit-code 1 \
      --python tests/blender/convert_and_render.py -- --out <dir>

Writes <dir>/scene.blend, <dir>/splats/*, and <dir>/{mesh,reference,points,
athc}.{exr,png}. Uses the GPU: run on a GPU turn.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import bpy  # noqa: E402


def main():
    args = common.arguments()
    out = os.path.abspath(args.get("out", "/Users/muriel/luc/athenea-renders/blender-phase2"))
    os.makedirs(out, exist_ok=True)
    checks = common.Checks()

    common.enable_addon()
    scene = common.scene_for_athenea()
    cube = bpy.data.objects["Cube"]
    material = cube.active_material
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.8, 0.25, 0.1, 1.0)
    light = bpy.data.objects["Light"]
    light.data.energy = 1000.0
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "scene.blend"))

    mesh = common.render(os.path.join(out, "mesh"))

    settings = scene.athenea_convert
    settings.budget = 400_000
    settings.resolution = 128
    settings.bake = True
    settings.bake_degree = 2
    settings.bake_samples = 64
    settings.lights = 'SCENE'
    settings.directory = "//splats/"

    def convert(label, output_format, bring_in):
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        cube.hide_set(False)
        cube.hide_render = False
        cube.select_set(True)
        bpy.context.view_layer.objects.active = cube
        settings.output_format = output_format
        settings.bring_in = bring_in
        before = set(bpy.data.objects)
        result = bpy.ops.athenea.mesh_to_splats()
        checks.check('FINISHED' in result, f"{label}: the operator finished ({settings.status})")
        added = [o for o in bpy.data.objects if o not in before]
        extension = ".athc" if output_format == 'ATHC' else ".usdc"
        written = os.path.join(out, "splats", "scene_Cube" + extension)
        checks.check(os.path.isfile(written), f"{label}: {written} written")
        if bring_in == 'REFERENCE':
            checks.check(len(added) == 1 and added[0].type == 'EMPTY' and "athenea_cloud" in added[0],
                         f"{label}: one Empty naming the cloud")
        else:
            clouds = [o for o in added if o.type == 'POINTCLOUD']
            checks.check(len(clouds) == 1 and getattr(clouds[0].data, "type", None) == 'GAUSSIAN_SPLAT',
                         f"{label}: one Gaussian-splat point cloud imported")
        checks.check(cube.hide_render, f"{label}: the cube is hidden")
        checks.check(scene.hydra.export_method == 'USD', f"{label}: the export method is USD")
        image = common.render(os.path.join(out, label))
        numbers = common.compare(image, mesh)
        alpha = abs(numbers["image_mean"][3] - numbers["reference_mean"][3])
        checks.check(numbers["image_mean"][3] > 0.01, f"{label}: the cloud is drawn (alpha {numbers['image_mean'][3]:.4f})")
        checks.check(alpha < 0.03, f"{label}: covers what the mesh covers (alpha difference {alpha:.4f})")
        checks.check(numbers["relMSE"] < 0.25, f"{label}: relMSE {numbers['relMSE']:.4g} under 0.25")
        checks.check(numbers["p99"] < 40, f"{label}: 8-bit p99 {numbers['p99']} under 40")
        for o in added:
            bpy.data.objects.remove(o, do_unlink=True)

    convert("reference", 'USDC', 'REFERENCE')
    convert("points", 'USDC', 'POINTS')
    convert("athc", 'ATHC', 'REFERENCE')

    # Refused before anything runs: a .athc cannot be imported as points.
    settings.output_format = 'ATHC'
    settings.bring_in = 'POINTS'
    cube.select_set(True)
    try:
        refused = bpy.ops.athenea.mesh_to_splats()
    except RuntimeError as error:   # an operator's ERROR report raises in a script
        refused = {'CANCELLED'}
        print(f"refused as expected: {error}")
    checks.check('CANCELLED' in refused, "a .athc brought in as points is refused")
    checks.finish()


main()
