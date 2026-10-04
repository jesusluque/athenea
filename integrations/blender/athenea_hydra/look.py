# Copyright (c) 2026 jesus luque.
"""The look against Cycles: the same camera, the meshes in Cycles and the
clouds in Athenea, measured by `athenea compare`.

What the user converts is judged by what it replaces. The button renders the
frame twice from the scene's camera:
  - Cycles, with the meshes a cloud was made from shown again and the ground
    a shadow catcher stands for made Cycles' own shadow catcher -- the same
    idea, so both frames hold the object and its shadow over a transparent
    film;
  - Athenea, as the scene is (the clouds, relit, their catchers drawn);
and `athenea compare` (the Measure effect, on the GPU) measures the second
against the first: relMSE, the 8-bit p99 and a heatmap. The three images
are loaded into Blender (`Athenea Cycles`, `Athenea Render`, `Athenea
Difference`) and the numbers are shown in the render panel. Nothing is
computed here; the scene is put back as it was.
"""

import os
import re

import bpy

from . import commands, convert

_NUMBER = r"([-+0-9.eE]+|nan|inf)"


class AtheneaLookResult(bpy.types.PropertyGroup):
    rel_mse: bpy.props.FloatProperty(name="relMSE", default=-1.0)
    p99: bpy.props.IntProperty(name="8-bit p99", default=-1)
    directory: bpy.props.StringProperty(name="Where", default="")


def _save(scene, path):
    result = bpy.data.images["Render Result"]
    image = scene.render.image_settings
    kept = (image.file_format, image.color_depth, image.color_mode)
    image.file_format = 'OPEN_EXR'
    image.color_depth = '32'
    image.color_mode = 'RGBA'
    try:
        result.save_render(path, scene=scene)
    finally:
        image.file_format, image.color_depth, image.color_mode = kept


def _load(path, name):
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    image = bpy.data.images.load(path, check_existing=False)
    image.name = name
    return image


class ATHENEA_OT_compare_cycles(bpy.types.Operator):
    """Render the camera's frame with Cycles (the meshes) and with Athenea (the clouds), and measure the difference on the GPU"""
    bl_idname = "athenea.compare_cycles"
    bl_label = "Compare with Cycles"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.scene.camera is not None and bool(bpy.data.filepath)

    def execute(self, context):
        scene = context.scene
        directory = bpy.path.abspath("//splats/compare")
        os.makedirs(directory, exist_ok=True)
        cycles_exr = os.path.join(directory, "cycles.exr")
        athenea_exr = os.path.join(directory, "athenea.exr")
        heat_exr = os.path.join(directory, "difference.exr")

        clouds = [o for o in scene.objects if convert.SOURCES_PROPERTY in o]
        sources = {bpy.data.objects.get(n) for o in clouds for n in o[convert.SOURCES_PROPERTY].split(",")}
        sources.discard(None)
        grounds = {bpy.data.objects.get(o.get(convert.GROUND_PROPERTY, "")) for o in clouds}
        grounds.discard(None)
        engine = scene.render.engine
        transparent = scene.render.film_transparent
        kept = {o: (o.hide_render, getattr(o, "is_shadow_catcher", False)) for o in sources | grounds | set(clouds)}
        try:
            scene.render.film_transparent = True
            # Cycles: what the clouds replace.
            scene.render.engine = 'CYCLES'
            for o in sources | grounds:
                o.hide_render = False
            for o in grounds:
                o.is_shadow_catcher = True
            for o in clouds:
                o.hide_render = True
            bpy.ops.render.render()
            _save(scene, cycles_exr)
            # Athenea: the scene as it is.
            for o, (hidden, catcher) in kept.items():
                o.hide_render = hidden
                if hasattr(o, "is_shadow_catcher"):
                    o.is_shadow_catcher = catcher
            scene.render.engine = 'ATHENEA_HYDRA'
            bpy.ops.render.render()
            _save(scene, athenea_exr)
        finally:
            scene.render.engine = engine
            scene.render.film_transparent = transparent
            for o, (hidden, catcher) in kept.items():
                o.hide_render = hidden
                if hasattr(o, "is_shadow_catcher"):
                    o.is_shadow_catcher = catcher

        args = [athenea_exr, cycles_exr, "--heatmap", heat_exr]
        for path in commands.bundle_dirs():
            args += ["--path", path]
        job = commands.Job([commands.Step("compare", args, "comparing")])
        try:
            job.start()
        except RuntimeError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        job.wait()
        failure = job.failure()
        if failure:
            self.report({'ERROR'}, failure)
            return {'CANCELLED'}
        text = "\n".join(job.log)
        look = scene.athenea_look
        hdr = re.search(r"relMSE " + _NUMBER, text)
        codes = re.search(r"8-bit\s+p99 (\d+)", text)
        look.rel_mse = float(hdr.group(1)) if hdr else -1.0
        look.p99 = int(codes.group(1)) if codes else -1
        look.directory = directory
        for path, name in ((cycles_exr, "Athenea Cycles"), (athenea_exr, "Athenea Render"),
                           (heat_exr, "Athenea Difference")):
            if os.path.isfile(path):
                _load(path, name)
        self.report({'INFO'}, f"Athenea against Cycles: relMSE {look.rel_mse:.4g}, 8-bit p99 {look.p99}")
        return {'FINISHED'}


classes = (AtheneaLookResult, ATHENEA_OT_compare_cycles)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.athenea_look = bpy.props.PointerProperty(type=AtheneaLookResult)


def unregister():
    del bpy.types.Scene.athenea_look
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
