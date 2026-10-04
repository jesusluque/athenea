# Copyright (c) 2026 jesus luque.
"""The add-on as a module, headless and without the GPU.

What is checked, none of it opening a device (hdAthenea is registered with
USD's plugin registry, never loaded):

  - registration: the render engine, the operators (convert, export, the
    comparison with Cycles), the panels, the scene
    settings and the File > Export entry; this Blender's USD is the one
    hdAthenea was built against;
  - the render settings the engine hands the delegate, for the viewport and
    for F12 (raster and path traced), and the passes it registers;
  - a conversion's export: the default cube selected over a ground plane
    becomes `/object` and `/surroundings` in one stage with the cube's up
    axis, and the arguments are mesh2splat's TX ones (with the shadow
    catcher as a second, optional step), with a quality and with overrides;
  - an export's stage: a cloud's Empty references its file under it, beside
    the scene's lights and its world as a DomeLight.

  blender -b --factory-startup --python-exit-code 1 \\
      --python tests/blender/test_module.py -- --out <dir>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import bpy  # noqa: E402


def main():
    args = common.arguments()
    out = os.path.abspath(args.get("out", "/private/tmp/athenea-blender-module"))
    os.makedirs(out, exist_ok=True)
    checks = common.Checks()

    addon = common.enable_addon()
    from athenea_hydra import commands, convert, export  # noqa: E402

    # -- registration --------------------------------------------------------
    checks.check(addon.usd_mismatch() is None, f"this Blender's USD is hdAthenea's ({addon.usd_mismatch()})")
    checks.check(addon.AtheneaHydraRenderEngine.is_registered, "the render engine is registered")
    checks.check(hasattr(bpy.ops.athenea, "mesh_to_splats"), "the convert operator is registered")
    checks.check(hasattr(bpy.ops.athenea, "flatten"), "the export operator is registered")
    checks.check(hasattr(bpy.ops.athenea, "compare_cycles") and hasattr(bpy.types.Scene, "athenea_look"),
                 "the comparison with Cycles is registered")
    checks.check(hasattr(bpy.types.Scene, "athenea") and hasattr(bpy.types.Scene, "athenea_convert"),
                 "the scene carries the render and conversion settings")
    checks.check(export.menu_export in bpy.types.TOPBAR_MT_file_export._dyn_ui_initialize(),
                 "File > Export has Gaussian Splats")
    checks.check(addon.register_plugin(), "hdAthenea is known to USD's plugin registry")

    scene = common.scene_for_athenea()
    settings = scene.athenea_convert
    checks.check(settings.quality == 'FINAL' and settings.shadow_catcher and settings.bring_in == 'REFERENCE',
                 "defaults: final quality, a shadow catcher, a relit cloud")

    # -- render settings -----------------------------------------------------
    engine = addon.AtheneaHydraRenderEngine
    viewport = engine.get_render_settings(None, 'VIEWPORT')
    checks.check(viewport.get('athenea:technique') == "raster" and viewport.get('athenea:colourHalf'),
                 "the viewport draws the raster, its colour as half floats")
    final = engine.get_render_settings(None, 'FINAL')
    checks.check(final.get('athenea:technique') == "raster" and final.get('aovToken:Combined') == "color"
                 and final.get('aovToken:Normal') == "normal" and final.get('aovToken:DiffCol') == "albedo",
                 "F12 draws the raster with colour, depth, normal and albedo")
    scene.athenea.technique = 'PATH'
    scene.athenea.paths = 64
    traced = engine.get_render_settings(None, 'FINAL')
    checks.check(traced.get('athenea:technique') == "rt" and traced.get('athenea:pathTotal') == 64,
                 "a path traced F12 asks for its paths")
    scene.athenea.technique = 'RASTER'

    # -- a conversion's export ----------------------------------------------
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, -1.0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    cube = bpy.data.objects["Cube"]
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "module.blend"))
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    conversion = convert.Conversion(bpy.context, settings)
    checks.check(conversion.validate() is None, f"the cube converts ({conversion.validate()})")
    conversion.export()
    checks.check(os.path.isfile(conversion.mesh_stage), f"{conversion.mesh_stage} written")
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(conversion.mesh_stage)
    meshes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    checks.check(any(m.startswith("/object/Cube") for m in meshes), f"the cube is under /object ({meshes})")
    checks.check(any(m.startswith("/surroundings/Ground") for m in meshes), "the ground is under /surroundings")
    checks.check(UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z, "the stage keeps Blender's Z up")
    steps = conversion.steps()
    first = steps[0].args
    checks.check(steps[0].command == "mesh2splat" and "--transfer" in first and
                 first[first.index("--prim") + 1] == "/object", "mesh2splat converts /object as a TX transfer")
    checks.check("--density" in first and first[first.index("--bake-samples") + 1] == "256",
                 "final quality: per-mesh density, 256 paths")
    checks.check(len(steps) == 2 and steps[1].optional and "--shadow-catcher" in steps[1].args,
                 "the shadow catcher is a second, optional conversion")
    settings.quality = 'PREVIEW'
    settings.max_splats = 12345
    settings.thin_glass = "Windscreen, /object/_materials/Visor"
    preview = conversion.arguments()
    checks.check(preview[preview.index("--resolution") + 1] == "256", "preview quality: 256 cells")
    checks.check(preview[len(preview) - 1 - preview[::-1].index("--max-splats") + 1] == "12345",
                 "a budget override comes after the quality's")
    checks.check(preview.count("--thin-glass") == 2 and "Windscreen" in preview,
                 "thin glass overrides are passed one material each")
    del stage   # a layer held open is one the next export cannot write
    settings.surroundings = False
    alone = convert.Conversion(bpy.context, settings)
    alone.export()
    checks.check(len(alone.steps()) == 1 and alone.notes, "no surroundings: no catcher, and it says so")
    settings.surroundings = True

    # -- an export's stage ---------------------------------------------------
    cloud = os.path.join(out, "cloud.usda")
    with open(cloud, "w") as f:
        f.write('#usda 1.0\n(\n    defaultPrim = "Cloud"\n)\n\ndef Xform "Cloud"\n{\n}\n')
    empty = bpy.data.objects.new("Cloud_splats", None)
    empty[convert.CLOUD_PROPERTY] = cloud
    scene.collection.objects.link(empty)
    checks.check(export.scene_clouds(scene) == [empty], "the scene's clouds are its Empties that name one")
    stage_path = os.path.join(out, "flatten_stage.usdc")
    export.export_scene_stage(scene, bpy.context.view_layer, [empty], stage_path)
    flat = Usd.Stage.Open(stage_path)
    references = [p for p in flat.Traverse() if p.GetName() == "cloud" and p.HasAuthoredReferences()]
    checks.check(len(references) == 1, "the Empty references its cloud")
    domes = [p for p in flat.Traverse() if p.GetTypeName() == "DomeLight"]
    lights = [p for p in flat.Traverse() if p.GetTypeName().endswith("Light") and p.GetTypeName() != "DomeLight"]
    checks.check(len(domes) == 1, f"the world is a DomeLight ({[str(p.GetPath()) for p in domes]})")
    checks.check(len(lights) >= 1, "the scene's light is in it")
    meshes = [p for p in flat.Traverse() if p.IsA(UsdGeom.Mesh)]
    checks.check(not meshes, "and no mesh")
    del flat

    checks.check(commands.ABI == 2, "the add-on speaks the entry points' ABI 2")
    checks.finish()


main()
