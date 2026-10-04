# Copyright (c) 2026 jesus luque.
"""The scene's clouds as standard Gaussian Splatting files: `athenea flatten`.

A TX cloud is relit every frame; a file for another viewer (SuperSplat,
Spark, a glTF reader) holds one lighting. `athenea flatten` makes it: each
cloud under the scene's sky and lights, fitted to the degree-3 harmonics
every 3DGS viewer reads, on the GPU (docs/decisions.md, task TXF).

The stage it reads is Blender's own USD export of the clouds' Empties (the
add-on's hook references each cloud under its Empty), the scene's visible
lights and its world as a DomeLight. Nothing else is computed here.
"""

import os

import bpy
from bpy_extras.io_utils import ExportHelper

from . import commands, convert

FORMATS = (('SPZ', "SPZ", "Niantic's compressed splats, version 3 (what Spark and most web viewers read)"),
           ('PLY', "PLY", "The 3DGS paper's layout (INRIA): every viewer reads it, the largest"),
           ('GLB', "glTF (.glb)", "glTF with KHR_gaussian_splatting"))
EXTENSIONS = {'SPZ': ".spz", 'PLY': ".ply", 'GLB': ".glb"}


def scene_clouds(scene, selected_only=False):
    """The Empties that name a cloud, visible in the scene."""
    return [o for o in scene.objects
            if o.type == 'EMPTY' and convert.CLOUD_PROPERTY in o and o.visible_get()
            and (not selected_only or o.select_get())]


def export_scene_stage(scene, view_layer, clouds, path):
    """`clouds` (Empties), the visible lights and the world into `path`."""
    lights = [o for o in scene.objects if o.type == 'LIGHT' and o.visible_get()]
    convert.export_usd(path, list(clouds) + lights, "/root", view_layer,
                       export_meshes=False, export_materials=False, export_cameras=False,
                       export_lights=True, convert_world_material=True, export_points=False,
                       export_volumes=False, export_hair=False, export_curves=False)


class ATHENEA_OT_flatten(bpy.types.Operator, ExportHelper):
    """Write the scene's Athenea clouds, lit by its world and lights, as a standard Gaussian Splatting file (on the GPU)"""
    bl_idname = "athenea.flatten"
    bl_label = "Export Gaussian Splats"
    bl_options = {'REGISTER'}

    filename_ext = ".spz"
    filter_glob: bpy.props.StringProperty(default="*.spz;*.ply;*.glb", options={'HIDDEN'})

    file_format: bpy.props.EnumProperty(name="Format", items=FORMATS, default='SPZ')
    selected_only: bpy.props.BoolProperty(
        name="Selected only", description="Only the selected clouds' Empties", default=False)
    exposure: bpy.props.FloatProperty(
        name="Exposure", description="Stops on the light before it is encoded (--exposure)", default=0.0,
        soft_min=-10.0, soft_max=10.0)

    running = False

    @classmethod
    def poll(cls, context):
        return not ATHENEA_OT_flatten.running and bool(scene_clouds(context.scene))

    def check(self, context):
        # The file name follows the format.
        wanted = EXTENSIONS[self.file_format]
        stem, extension = os.path.splitext(self.filepath)
        if extension.lower() != wanted:
            self.filepath = stem + wanted
            return True
        return False

    def _job(self, context):
        clouds = scene_clouds(context.scene, self.selected_only)
        if not clouds:
            raise RuntimeError("no Athenea cloud to export (convert one first)")
        target = bpy.path.abspath(self.filepath)
        stem, _ = os.path.splitext(target)
        stage = stem + "_flatten_stage.usdc"
        export_scene_stage(context.scene, context.view_layer, clouds, stage)
        args = [stage, "-o", target, "--format", self.file_format.lower()]
        if self.exposure != 0.0:
            args += ["--exposure", f"{self.exposure:g}"]
        for directory in commands.bundle_dirs():
            args += ["--path", directory]
        job = commands.Job([commands.Step("flatten", args, "flattening")])
        job.start()
        return job, target

    def _end(self, job, target):
        ATHENEA_OT_flatten.running = False
        for line in job.log:
            print("athenea_hydra:", line)
        failure = job.failure()
        if failure or not os.path.isfile(target):
            self.report({'ERROR'}, failure or f"flatten wrote no {target}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Gaussian splats: wrote {target} in {job.seconds:.1f} s")
        return {'FINISHED'}

    def execute(self, context):
        self.check(context)
        try:
            job, target = self._job(context)
        except RuntimeError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        ATHENEA_OT_flatten.running = True
        if context.window is None:
            job.wait()
            return self._end(job, target)
        self._state = (job, target)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.progress_begin(0.0, 1.0)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        job, target = self._state
        alive = job.drain()
        context.window_manager.progress_update(job.progress)
        if context.workspace is not None:
            context.workspace.status_text_set(f"Gaussian splats: {job.status}")
        if alive:
            return {'RUNNING_MODAL'}
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        if context.workspace is not None:
            context.workspace.status_text_set(None)
        return self._end(job, target)


def menu_export(self, context):
    self.layout.operator(ATHENEA_OT_flatten.bl_idname, text="Gaussian Splats (Athenea: .spz, .ply, .glb)")


classes = (ATHENEA_OT_flatten,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
