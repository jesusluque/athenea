# Copyright (c) 2026 jesus luque.
"""Selected objects into gaussians: `athenea mesh2splat --transfer`, in Blender.

The conversion is the engine's TX transfer -- how much of any sky reaches
each gaussian, from every direction, with its glass told apart (sheet, slab
or solid), its metals and its lenses -- so the cloud is relit by whatever
world the scene has, every frame. Every per-asset choice is the engine's
own, detected on the GPU; the panel picks a quality and whether the ground
gets a shadow catcher, and keeps the engine's overrides in a closed
subpanel.

What happens, in order:
  1. the selected meshes are exported with Blender's USD exporter under
     `/object`, and the scene's other visible meshes under `/surroundings`
     (what the bake's rays meet: the ground, a wall), into one stage
     `<dir>/<blend>_<name>_mesh.usda` that sublayers the two;
  2. mesh2splat converts `/object` with `--transfer`; then, with the shadow
     catcher on, again with `--shadow-catcher`: a layer of gaussians on the
     ground it finds by itself (the largest flat mesh at the object's
     bottom), baked the same way and drawn black where the object takes the
     light. No ground found is said, and the object's cloud stays;
  3. the clouds come in as Empties whose files hdAthenea draws (relit live,
     the TX raster), or flattened under the scene's current world into a
     standard Gaussian Splatting file and imported as Blender's own splat
     point cloud (what any engine draws, lit once).

Esc stops it: mesh2splat looks between the transfer's slices.
"""

import os
import time

import bpy

from . import commands

# The Empty's custom property naming the file its cloud comes from; the
# export hook in __init__ references it.
CLOUD_PROPERTY = "athenea_cloud"
# Set on an Empty whose cloud is a shadow catcher.
CATCHER_PROPERTY = "athenea_catcher"
# On a cloud: the meshes it was converted from (comma list), and on a
# catcher the ground it stands for.
SOURCES_PROPERTY = "athenea_sources"
GROUND_PROPERTY = "athenea_ground"
# A Points object whose colours are linear light: the hook writes
# primvars:athenea:splat:linear on it.
LINEAR_PROPERTY = "athenea_linear"

# What a quality asks of mesh2splat. Preview: a quick look at the asset.
# Final: what the engine's TX measurements are made at (docs/decisions.md,
# task TX): each mesh its own density, 256 paths a gaussian.
QUALITY = {
    'PREVIEW': ["--resolution", "256", "--max-splats", "1000000", "--bake-samples", "64",
                "--bake-extra", "0"],
    'FINAL': ["--density", "per-mesh", "--resolution", "512", "--max-splats", "8000000",
              "--bake-samples", "256"],
}


class AtheneaConvertSettings(bpy.types.PropertyGroup):
    quality: bpy.props.EnumProperty(
        name="Quality",
        items=(('PREVIEW', "Preview", "A quick conversion to look at: 256 cells across, a million gaussians "
                "at most, 64 paths a gaussian"),
               ('FINAL', "Final", "Each mesh at its own density, 512 cells across it, 8 million gaussians at "
                "most, 256 paths a gaussian")),
        default='FINAL')
    shadow_catcher: bpy.props.BoolProperty(
        name="Shadow catcher", description="Also convert the shadow the selection casts on the ground under "
        "it (found by itself: the largest flat mesh at its bottom) as gaussians drawn black", default=True)
    bring_in: bpy.props.EnumProperty(
        name="Result",
        items=(('REFERENCE', "Relit cloud", "The cloud as converted, relit live by Athenea under the scene's "
                "world and lights"),
               ('POINTS', "Blender splats", "The cloud flattened under the scene's current world into a "
                "standard Gaussian Splatting file and imported as Blender's own splat point cloud: any "
                "engine draws it, lit once")),
        default='REFERENCE')
    directory: bpy.props.StringProperty(
        name="Directory", description="Where the exported meshes and the clouds are written",
        default="//splats/", subtype='DIR_PATH')
    hide_source: bpy.props.BoolProperty(
        name="Hide the meshes", description="Hide the converted meshes, and the ground under a shadow "
        "catcher, once the clouds are in", default=True)
    # Overrides: the engine decides these itself.
    max_splats: bpy.props.IntProperty(
        name="Gaussians", description="Most gaussians over the selection; 0 is the quality's own "
        "(--max-splats)", default=0, min=0)
    surroundings: bpy.props.BoolProperty(
        name="Surroundings", description="The scene's other visible meshes are what the bake's rays meet "
        "(the ground darkens the underside); off, the selection is baked alone and no catcher is made",
        default=True)
    thin_glass: bpy.props.StringProperty(
        name="Thin glass", description="Materials (comma list) whose glass is one sheet whatever its mesh "
        "looks like (--thin-glass); the conversion tells sheet, slab and solid apart by itself")
    solid_glass: bpy.props.StringProperty(
        name="Solid glass", description="Materials (comma list) whose glass is solid whatever its mesh "
        "looks like (--solid-glass)")
    skinned: bpy.props.BoolProperty(
        name="Skinned", description="Carry the armature: the cloud deforms with the rig, its transfer kept "
        "in each gaussian's own frame (--skinned)", default=False)
    # What the last conversion said, for the panel.
    status: bpy.props.StringProperty(name="Status", default="")
    progress: bpy.props.FloatProperty(name="Progress", default=0.0, min=0.0, max=1.0, subtype='FACTOR')


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "selection"


def _materials(text):
    return [m.strip() for m in text.split(",") if m.strip()]


def export_usd(path, objects, root, view_layer, **options):
    """Blender's USD export of exactly `objects` under `root`; the
    selection is put back as it was."""
    selected = [o for o in view_layer.objects if o.select_get()]
    active = view_layer.objects.active
    try:
        for o in selected:
            o.select_set(False)
        for o in objects:
            o.select_set(True)
        result = bpy.ops.wm.usd_export(
            filepath=path, check_existing=False, selected_objects_only=True, root_prim_path=root,
            export_textures_mode='NEW', overwrite_textures=True, evaluation_mode='RENDER', **options)
    finally:
        for o in objects:
            o.select_set(False)
        for o in selected:
            o.select_set(True)
        view_layer.objects.active = active
    if 'FINISHED' not in result or not os.path.isfile(path):
        raise RuntimeError(f"Blender's USD export did not write {path}")


def compose_stage(path, layers):
    """A stage that sublayers `layers` (beside it), with the first one's up
    axis and unit: a root layer's metadata is the stage's, a sublayer's not.
    Only Sdf, and nothing kept open: a layer left in USD's registry is one
    the next export cannot write again."""
    from pxr import Sdf
    first = Sdf.Layer.FindOrOpen(layers[0])
    if first is None:
        raise RuntimeError(f"cannot read {layers[0]}")
    up = first.pseudoRoot.GetInfo("upAxis") if first.pseudoRoot.HasInfo("upAxis") else "Z"
    unit = first.pseudoRoot.GetInfo("metersPerUnit") if first.pseudoRoot.HasInfo("metersPerUnit") else 1.0
    del first
    root = Sdf.Layer.FindOrOpen(path) if os.path.exists(path) else Sdf.Layer.CreateNew(path)
    if root is None:
        raise RuntimeError(f"cannot write {path}")
    root.Clear()
    root.subLayerPaths = [os.path.basename(layer) for layer in layers]
    root.pseudoRoot.SetInfo("upAxis", up)
    root.pseudoRoot.SetInfo("metersPerUnit", unit)
    if not root.Save():
        raise RuntimeError(f"cannot write {path}")


def _free_stem(stem):
    """`stem`, or `stem_2`, `stem_3`...: the first whose files no stage in
    this process holds. Hydra keeps a drawn cloud's layer open, and a USD
    layer that is open cannot be written again under its name."""
    from pxr import Sdf
    candidate, n = stem, 1
    while any(Sdf.Layer.Find(candidate + suffix) is not None
              for suffix in (".usdc", "_catcher.usdc", "_object.usdc", "_surroundings.usdc", "_mesh.usda",
                             "_flat.usdc", "_flatten_stage.usdc")):
        n += 1
        candidate = f"{stem}_{n}"
    return candidate


class Conversion:
    """One conversion: export, mesh2splat (and its catcher), bring in."""

    def __init__(self, context, settings):
        self.settings = settings
        self.scene = context.scene
        self.view_layer = context.view_layer
        self.meshes = [o for o in context.selected_objects if o.type == 'MESH']
        self.job = None
        self.flatten = None          # the second job, for Blender splats
        self.mesh_stage = ""
        self.output = ""
        self.catcher = ""
        self.stem = ""
        self.has_ground_candidates = False
        self.brought_in = []
        self.notes = []
        self.started = time.monotonic()

    @property
    def name(self):
        return self.meshes[0].name if len(self.meshes) == 1 else "selection"

    def validate(self):
        """What would make the conversion wrong, said before anything runs."""
        if not bpy.data.filepath:
            return "Save the .blend first: the clouds are written next to it"
        if not self.meshes:
            return "Select at least one mesh object"
        if self.settings.bring_in == 'POINTS' and self.settings.skinned:
            return "Blender splats keep no rig: bring a skinned cloud in as a relit cloud"
        return None

    def surroundings(self):
        chosen = set(self.meshes)
        return [o for o in self.view_layer.objects
                if o.type == 'MESH' and o not in chosen and o.visible_get() and not o.hide_render]

    # -- 1. export ----------------------------------------------------------

    def export(self):
        s = self.settings
        directory = bpy.path.abspath(s.directory)
        os.makedirs(directory, exist_ok=True)
        blend = _safe(os.path.splitext(os.path.basename(bpy.data.filepath))[0])
        self.stem = _free_stem(os.path.join(directory, f"{blend}_{_safe(self.name)}"))
        self.mesh_stage = self.stem + "_mesh.usda"
        self.output = self.stem + ".usdc"
        self.catcher = self.stem + "_catcher.usdc"

        objects = set(self.meshes)
        if s.skinned:
            for mesh in self.meshes:
                for modifier in mesh.modifiers:
                    if modifier.type == 'ARMATURE' and modifier.object is not None:
                        objects.add(modifier.object)
                if mesh.parent is not None and mesh.parent.type == 'ARMATURE':
                    objects.add(mesh.parent)
        geometry = dict(export_materials=True, generate_preview_surface=True, export_cameras=False,
                        export_lights=False, convert_world_material=False, export_points=False,
                        export_volumes=False, export_hair=False, export_curves=False)
        layers = [self.stem + "_object.usdc"]
        export_usd(layers[0], objects, "/object", self.view_layer, export_animation=s.skinned,
                   export_armatures=s.skinned, only_deform_bones=s.skinned, **geometry)
        others = self.surroundings() if s.surroundings else []
        if others:
            layers.append(self.stem + "_surroundings.usdc")
            export_usd(layers[1], others, "/surroundings", self.view_layer, export_animation=False,
                       export_armatures=False, **geometry)
        compose_stage(self.mesh_stage, layers)
        self.has_ground_candidates = bool(others)

    # -- 2. mesh2splat ------------------------------------------------------

    def arguments(self, catcher=False):
        s = self.settings
        args = [self.mesh_stage, "-o", self.catcher if catcher else self.output, "--no-camera",
                "--prim", "/object", "--transfer"]
        args += QUALITY[s.quality]
        if s.max_splats > 0:
            args += ["--max-splats", str(s.max_splats)]
        if catcher:
            args.append("--shadow-catcher")
        else:
            for material in _materials(s.thin_glass):
                args += ["--thin-glass", material]
            for material in _materials(s.solid_glass):
                args += ["--solid-glass", material]
            if s.skinned:
                args.append("--skinned")
        for directory in commands.bundle_dirs():
            args += ["--path", directory]
        return args

    def steps(self):
        steps = [commands.Step("mesh2splat", self.arguments(), "converting", weight=3.0)]
        if self.settings.shadow_catcher and self.settings.surroundings and self.has_ground_candidates:
            steps.append(commands.Step("mesh2splat", self.arguments(catcher=True), "shadow catcher",
                                       weight=1.0, optional=True))
        elif self.settings.shadow_catcher:
            self.notes.append("no shadow catcher: no other visible mesh to be the ground")
        return steps

    def start(self):
        self.job = commands.Job(self.steps())
        self.job.start()

    # -- 3. bring in --------------------------------------------------------

    def clouds(self):
        """The files written, and whether each is the catcher."""
        found = [(self.output, False)]
        if len(self.job.steps) > 1 and self.job.steps[1].code == 0 and os.path.isfile(self.catcher):
            found.append((self.catcher, True))
        return found

    def link_references(self):
        """An Empty per cloud, in a collection of its own."""
        collection = bpy.data.collections.new(f"{self.name}_splats")
        self.scene.collection.children.link(collection)
        made = []
        for path, catcher in self.clouds():
            empty = bpy.data.objects.new(f"{self.name}_{'catcher' if catcher else 'splats'}", None)
            empty.empty_display_type = 'CUBE'
            empty.empty_display_size = 0.25
            empty[CLOUD_PROPERTY] = bpy.path.relpath(path)
            if catcher:
                empty[CATCHER_PROPERTY] = True
            collection.objects.link(empty)
            made.append(empty)
        return made

    def start_flatten(self):
        """Blender splats: the clouds flattened under the current world."""
        from . import export
        self.brought_in = self.link_references()
        stage = self.stem + "_flatten_stage.usdc"
        export.export_scene_stage(self.scene, self.view_layer, self.brought_in, stage)
        args = [stage, "-o", self.stem + "_flat", "--format", "usdc"]
        for directory in commands.bundle_dirs():
            args += ["--path", directory]
        self.flatten = commands.Job([commands.Step("flatten", args, "flattening under the world")])
        self.flatten.start()

    def import_flattened(self):
        """The flattened file as Blender's own splat point cloud, in place of
        the Empties that named the clouds."""
        flat = self.stem + "_flat.usdc"
        collection = self.brought_in[0].users_collection[0] if self.brought_in else None
        for empty in self.brought_in:
            bpy.data.objects.remove(empty, do_unlink=True)
        self.brought_in = []
        before = set(bpy.data.objects)
        result = bpy.ops.wm.usd_import(
            filepath=flat, import_cameras=False, import_lights=False, import_materials=False,
            import_meshes=False, import_points=True, import_volumes=False, import_curves=False,
            set_frame_range=False, create_world_material=False)
        if 'FINISHED' not in result:
            raise RuntimeError(f"Blender's USD import did not read {flat}")
        for o in bpy.data.objects:
            if o in before:
                continue
            self.brought_in.append(o)
            if collection is not None and o.type == 'POINTCLOUD':
                for owner in list(o.users_collection):
                    owner.objects.unlink(o)
                collection.objects.link(o)
                o.name = f"{self.name}_splats"

    def finish(self):
        s = self.settings
        if s.bring_in == 'REFERENCE':
            self.brought_in = self.link_references()
        grounds = self.catcher_grounds() if any(catcher for _, catcher in self.clouds()) else []
        # What each cloud stands for, for the comparison with Cycles (look.py):
        # the meshes it was made from, and the ground a catcher replaces.
        for o in self.brought_in:
            o[SOURCES_PROPERTY] = ",".join(m.name for m in self.meshes)
            if o.get(CATCHER_PROPERTY) and grounds:
                o[GROUND_PROPERTY] = grounds[0].name
        if s.hide_source:
            hidden = list(self.meshes) + grounds
            for o in hidden:
                o.hide_render = True
                o.hide_set(True)
        # Blender's Hydra export method hands a delegate no point cloud and
        # runs no USD export hook: the clouds are drawn only with USD's.
        hydra = getattr(self.scene, "hydra", None)
        if hydra is not None and hydra.export_method != 'USD':
            hydra.export_method = 'USD'
            self.notes.append("the scene's Hydra export method is now USD, which the clouds need")

    def catcher_grounds(self):
        """The ground the catcher stands on, by the path mesh2splat printed
        ("the shadow catcher of /object on /surroundings/Plane/Plane, ...")."""
        found = []
        for line in self.job.log:
            if "the shadow catcher of" in line and " on " in line:
                path = line.split(" on ", 1)[1].split(",", 1)[0].strip()
                parts = [p for p in path.split("/") if p]
                if len(parts) >= 2 and parts[0] == "surroundings":
                    o = bpy.data.objects.get(parts[1])
                    if o is not None:
                        found.append(o)
        return found


class ATHENEA_OT_mesh_to_splats(bpy.types.Operator):
    """Convert the selected meshes into gaussians relit by any sky (mesh2splat --transfer, on the GPU). Esc stops it"""
    bl_idname = "athenea.mesh_to_splats"
    bl_label = "Convert to Gaussian Splats"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _conversion = None
    running = False

    @classmethod
    def poll(cls, context):
        return (not ATHENEA_OT_mesh_to_splats.running and
                any(o.type == 'MESH' for o in context.selected_objects))

    def _begin(self, context):
        settings = context.scene.athenea_convert
        conversion = Conversion(context, settings)
        problem = conversion.validate()
        if problem:
            self.report({'ERROR'}, problem)
            return None
        settings.progress = 0.0
        settings.status = "exporting the meshes"
        try:
            conversion.export()
            conversion.start()
        except RuntimeError as error:
            settings.status = str(error)
            self.report({'ERROR'}, str(error))
            return None
        ATHENEA_OT_mesh_to_splats.running = True
        return conversion

    def _fail(self, settings, message):
        ATHENEA_OT_mesh_to_splats.running = False
        settings.status = message
        self.report({'WARNING'} if message == "cancelled" else {'ERROR'}, message)
        return {'CANCELLED'}

    def _converted(self, context, conversion):
        """After mesh2splat: bring in, or flatten first (None while it runs)."""
        settings = context.scene.athenea_convert
        for line in conversion.job.log:
            print("athenea_hydra:", line)
        failure = conversion.job.failure()
        if failure or not os.path.exists(conversion.output):
            return self._fail(settings, failure or f"mesh2splat wrote no {conversion.output}")
        if settings.bring_in == 'POINTS':
            try:
                conversion.start_flatten()
            except RuntimeError as error:
                return self._fail(settings, str(error))
            return None
        return self._end(context, conversion)

    def _end(self, context, conversion):
        settings = context.scene.athenea_convert
        if conversion.flatten is not None:
            for line in conversion.flatten.log:
                print("athenea_hydra:", line)
            failure = conversion.flatten.failure()
            if failure:
                return self._fail(settings, failure)
            try:
                conversion.import_flattened()
            except RuntimeError as error:
                return self._fail(settings, str(error))
        try:
            conversion.finish()
        except RuntimeError as error:
            return self._fail(settings, str(error))
        ATHENEA_OT_mesh_to_splats.running = False
        settings.progress = 1.0
        seconds = time.monotonic() - conversion.started
        catcher = " and its shadow catcher" if any(c for _, c in conversion.clouds()) else ""
        settings.status = f"wrote {bpy.path.relpath(conversion.output)}{catcher} in {seconds:.1f} s"
        for warning in conversion.job.warnings + conversion.notes:
            self.report({'WARNING'}, warning)
        self.report({'INFO'}, f"Gaussian splats: {settings.status}")
        return {'FINISHED'}

    def execute(self, context):
        # Without a window (blender -b, a script) there is no event loop to
        # be modal in: the conversion runs to its end here.
        conversion = self._begin(context)
        if conversion is None:
            return {'CANCELLED'}
        conversion.job.wait()
        result = self._converted(context, conversion)
        if result is not None:
            return result
        conversion.flatten.wait()
        return self._end(context, conversion)

    def invoke(self, context, event):
        if context.window_manager is None or context.window is None:
            return self.execute(context)
        conversion = self._begin(context)
        if conversion is None:
            return {'CANCELLED'}
        self._conversion = conversion
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.progress_begin(0.0, 1.0)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _stop_modal(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        if context.workspace is not None:
            context.workspace.status_text_set(None)

    def modal(self, context, event):
        conversion = self._conversion
        job = conversion.flatten or conversion.job
        if event.type == 'ESC' and event.value == 'PRESS':
            job.cancel()
            context.scene.athenea_convert.status = "stopping"
            return {'RUNNING_MODAL'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        settings = context.scene.athenea_convert
        alive = job.drain()
        # The conversion is 90% of the bar where a flattening follows.
        if conversion.flatten is None:
            settings.progress = job.progress * (0.9 if settings.bring_in == 'POINTS' else 1.0)
        else:
            settings.progress = 0.9 + 0.1 * job.progress
        if not job.cancelled:
            settings.status = job.status or settings.status
        context.window_manager.progress_update(settings.progress)
        if context.workspace is not None:
            context.workspace.status_text_set(f"Gaussian splats: {settings.status} (Esc stops)")
        for area in context.screen.areas if context.screen else []:
            if area.type in {'VIEW_3D', 'PROPERTIES'}:
                area.tag_redraw()
        if alive:
            return {'RUNNING_MODAL'}
        if conversion.flatten is None:
            result = self._converted(context, conversion)
            if result is None:
                return {'RUNNING_MODAL'}
        else:
            result = self._end(context, conversion)
        self._stop_modal(context)
        return result


class VIEW3D_PT_athenea_splats(bpy.types.Panel):
    bl_label = "Gaussian Splats"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Athenea"

    def draw(self, context):
        settings = context.scene.athenea_convert
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(settings, "quality")
        row = layout.row()
        row.active = settings.surroundings
        row.prop(settings, "shadow_catcher")
        layout.prop(settings, "bring_in")
        layout.prop(settings, "directory")
        layout.prop(settings, "hide_source")
        layout.operator(ATHENEA_OT_mesh_to_splats.bl_idname, icon='POINTCLOUD_DATA')
        if ATHENEA_OT_mesh_to_splats.running:
            layout.progress(factor=settings.progress, text=settings.status or "converting")
        elif settings.status:
            layout.label(text=settings.status)
        from .export import ATHENEA_OT_flatten
        layout.operator(ATHENEA_OT_flatten.bl_idname, icon='EXPORT')


class VIEW3D_PT_athenea_splats_overrides(bpy.types.Panel):
    bl_label = "Overrides"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Athenea"
    bl_parent_id = "VIEW3D_PT_athenea_splats"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        settings = context.scene.athenea_convert
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(settings, "max_splats")
        layout.prop(settings, "surroundings")
        layout.prop(settings, "thin_glass")
        layout.prop(settings, "solid_glass")
        layout.prop(settings, "skinned")


classes = (AtheneaConvertSettings, ATHENEA_OT_mesh_to_splats, VIEW3D_PT_athenea_splats,
           VIEW3D_PT_athenea_splats_overrides)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.athenea_convert = bpy.props.PointerProperty(type=AtheneaConvertSettings)


def unregister():
    del bpy.types.Scene.athenea_convert
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
