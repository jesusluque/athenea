# Copyright (c) 2026 jesus luque.
"""The module end to end, headless, on the GPU: convert, render, export.

A red cube on a ground plane under one of Blender's own HDRIs (forest.exr)
as the world. What is checked:

  - the ground truth: the cube alone, path traced through hdAthenea
    (technique PATH, 256 paths, denoised), the ground hidden;
  - *Convert to Gaussian Splats* at the preview quality with the shadow
    catcher: a TX cloud and a catcher, two Empties, the cube and the ground
    hidden; the cloud alone rendered as the raster and compared with the
    ground truth (drawn; alpha within 0.03; relMSE under 0.25, 8-bit p99
    under 40 -- bounds for a gross failure: the world not reaching the
    cloud, the colour space, the place); then with its catcher, which must
    add coverage (the shadow, drawn black, covers the ground's place);
  - *Compare with Cycles*: measured, its numbers printed (not bounded: the
    look's gap is what it reports), the scene put back;
  - *Export Gaussian Splats*: SPZ and PLY written, not empty;
  - the cloud brought in as Blender splats (flattened under the world and
    imported): one GAUSSIAN_SPLAT point cloud, drawn;
  - Esc: a job asked to stop at once ends with code 4 or, if the cube was
    converted before it looked, 0 (said, not failed).

Renders: <out>/{gt,tx,tx_catcher,points}.{exr,png}, 1024 x 768, and
<out>/splats/compare/{cycles,athenea,difference}.exr.

  blender -b --factory-startup --python-exit-code 1 \\
      --python tests/blender/tx_module.py -- --out <dir>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import bpy  # noqa: E402


def world_hdri(name="forest.exr"):
    path = os.path.join(os.path.dirname(bpy.app.binary_path), "..", "Resources",
                        f"{bpy.app.version[0]}.{bpy.app.version[1]}", "datafiles", "studiolights", "world", name)
    path = os.path.abspath(path)
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    texture = nodes.new("ShaderNodeTexEnvironment")
    texture.image = bpy.data.images.load(path)
    background = nodes.get("Background")
    world.node_tree.links.new(texture.outputs["Color"], background.inputs["Color"])
    return path


def main():
    args = common.arguments()
    out = os.path.abspath(args.get("out", "/Users/muriel/luc/athenea-renders/blender"))
    os.makedirs(out, exist_ok=True)
    checks = common.Checks()

    addon = common.enable_addon()
    from athenea_hydra import commands, convert, export  # noqa: E402, F401

    scene = common.scene_for_athenea(1024, 768)
    cube = bpy.data.objects["Cube"]
    principled = cube.active_material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.8, 0.25, 0.1, 1.0)
    bpy.data.objects["Light"].hide_render = True    # the world lights it
    bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, -1.0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    print("world:", world_hdri())
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "tx_module.blend"))

    # -- ground truth: the cube alone, path traced -----------------------------
    ground.hide_render = True
    scene.athenea.technique = 'PATH'
    scene.athenea.paths = 256
    scene.athenea.denoise = True
    gt = common.render(os.path.join(out, "gt"))
    ground.hide_render = False
    scene.athenea.technique = 'RASTER'

    # -- convert -------------------------------------------------------------
    settings = scene.athenea_convert
    settings.quality = 'PREVIEW'
    settings.shadow_catcher = True
    settings.bring_in = 'REFERENCE'
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    before = set(bpy.data.objects)
    result = bpy.ops.athenea.mesh_to_splats()
    checks.check('FINISHED' in result, f"convert: finished ({settings.status})")
    added = [o for o in bpy.data.objects if o not in before]
    clouds = [o for o in added if o.type == 'EMPTY' and convert.CLOUD_PROPERTY in o]
    catchers = [o for o in clouds if o.get(convert.CATCHER_PROPERTY)]
    checks.check(len(clouds) == 2 and len(catchers) == 1, f"convert: a cloud and its catcher ({len(clouds)})")
    checks.check(cube.hide_render and ground.hide_render, "convert: the cube and its ground hidden")

    for catcher in catchers:
        catcher.hide_render = True
        catcher.hide_set(True)
    tx = common.render(os.path.join(out, "tx"))
    numbers = common.compare(tx, gt)
    alpha = abs(numbers["image_mean"][3] - numbers["reference_mean"][3])
    checks.check(numbers["image_mean"][3] > 0.01, f"tx: the cloud is drawn (alpha {numbers['image_mean'][3]:.4f})")
    checks.check(alpha < 0.03, f"tx: covers what the cube covers (alpha difference {alpha:.4f})")
    checks.check(numbers["relMSE"] < 0.25, f"tx: relMSE {numbers['relMSE']:.4g} under 0.25 against the GT")
    checks.check(numbers["p99"] < 40, f"tx: 8-bit p99 {numbers['p99']} under 40")
    alone = numbers["image_mean"][3]
    for catcher in catchers:
        catcher.hide_render = False
        catcher.hide_set(False)
    with_catcher = common.compare(common.render(os.path.join(out, "tx_catcher")), gt)
    checks.check(with_catcher["image_mean"][3] > alone + 0.005,
                 f"tx: the catcher adds its shadow ({with_catcher['image_mean'][3]:.4f} against {alone:.4f})")

    # -- the look against Cycles: the cube and its ground as Cycles' shadow
    # catcher, against the cloud and its catcher ----------------------------
    result = bpy.ops.athenea.compare_cycles()
    look = scene.athenea_look
    checks.check('FINISHED' in result and look.rel_mse >= 0.0,
                 f"compare: Cycles against Athenea measured (relMSE {look.rel_mse:.4g}, p99 {look.p99})")
    checks.check(scene.render.engine == 'ATHENEA_HYDRA' and cube.hide_render and ground.hide_render,
                 "compare: the scene put back")

    # -- export --------------------------------------------------------------
    for file_format in ('SPZ', 'PLY'):
        target = os.path.join(out, "splats", "tx_module" + export.EXTENSIONS[file_format])
        result = bpy.ops.athenea.flatten(filepath=target, file_format=file_format)
        size = os.path.getsize(target) if os.path.isfile(target) else 0
        checks.check('FINISHED' in result and size > 1000, f"export: {target} ({size} bytes)")

    # -- Blender splats ------------------------------------------------------
    for o in clouds:
        bpy.data.objects.remove(o, do_unlink=True)
    cube.hide_set(False)
    cube.hide_render = False
    ground.hide_set(False)
    ground.hide_render = False
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    settings.bring_in = 'POINTS'
    settings.shadow_catcher = False
    before = set(bpy.data.objects)
    result = bpy.ops.athenea.mesh_to_splats()
    checks.check('FINISHED' in result, f"points: finished ({settings.status})")
    points = [o for o in bpy.data.objects if o not in before and o.type == 'POINTCLOUD']
    checks.check(len(points) == 1 and getattr(points[0].data, "type", None) == 'GAUSSIAN_SPLAT',
                 "points: one Gaussian-splat point cloud")
    ground.hide_render = True
    flat = common.compare(common.render(os.path.join(out, "points")), gt)
    checks.check(flat["image_mean"][3] > 0.01, f"points: drawn (alpha {flat['image_mean'][3]:.4f})")

    # -- Esc -----------------------------------------------------------------
    for o in points:
        bpy.data.objects.remove(o, do_unlink=True)
    cube.select_set(True)
    settings.bring_in = 'REFERENCE'
    conversion = convert.Conversion(bpy.context, settings)
    conversion.export()
    conversion.start()
    conversion.job.cancel()
    conversion.job.wait()
    code = conversion.job.steps[0].code
    checks.check(code in (commands.CANCELLED, 0), f"cancel: the job ends ({code})")
    print(f"cancel: {'stopped' if code == commands.CANCELLED else 'finished before it looked'}")
    checks.finish()


main()
