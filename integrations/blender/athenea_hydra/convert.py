# Copyright (c) 2026 jesus luque.
"""Selected meshes into a gaussian cloud, by mesh2splat, inside Blender.

The conversion is `athenea mesh2splat`, run in this process through the
entry point hdAthenea carries (`athenea_mesh2splat`, apps/athenea/src/
Embedded.cpp): the plugin is built against Blender's own USD, so the stage
the conversion reads and the one it writes are that USD's, and nothing loads
a second one. Its work is the Mesh2Splat effect's, on the GPU; this module
only exports the selection, hands over the arguments, relays the lines and
brings the result in.

What happens, in order:
  1. the selected meshes (and, for a skinned conversion, their armatures;
     for a bake under the scene's lights, the lights and the world) are
     exported with Blender's USD exporter to `<dir>/<blend>_<name>_mesh.usdc`;
  2. mesh2splat converts that stage to `<dir>/<blend>_<name>.usdc` (or
     `.athc`) on a worker thread -- ctypes lets go of the GIL for the call --
     and its lines come back through a queue the UI drains;
  3. the cloud is brought in: as an Empty the add-on's USD export hook
     references the file under (everything the conversion wrote, drawn by
     hdAthenea), or imported as Blender's own Gaussian-splat point cloud
     (what Blender keeps of it: positions, sizes, rotations, harmonics).
"""

import ctypes
import os
import queue
import threading
import time

import bpy

# The Empty's custom property naming the file its cloud comes from; the
# export hook in __init__ references it.
CLOUD_PROPERTY = "athenea_cloud"
# A Points object whose colours are linear light (a conversion's): the hook
# writes primvars:athenea:splat:linear on it.
LINEAR_PROPERTY = "athenea_linear"

_SinkType = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_library = None


def plugin_library():
    """hdAthenea, loaded through USD's registry and then opened by ctypes
    (the same image: dlopen counts a second open of a loaded file)."""
    global _library
    if _library is not None:
        return _library
    from . import plugin_dir, register_plugin
    if not register_plugin():
        raise RuntimeError("hdAthenea is not registered (see the add-on preferences)")
    from pxr import Plug
    plugin = Plug.Registry().GetPluginWithName("hdAthenea")
    if plugin is None:
        raise RuntimeError("USD knows no plugin named hdAthenea")
    if not plugin.isLoaded:
        plugin.Load()
    path = plugin.path or os.path.join(plugin_dir(), "hdAthenea", "hdAthenea.so")
    library = ctypes.CDLL(path)
    try:
        abi = library.athenea_embedded_abi
        entry = library.athenea_mesh2splat
    except AttributeError:
        raise RuntimeError(f"{path} carries no mesh2splat (built without ATHENEA_HYDRA_COMMANDS)")
    abi.restype = ctypes.c_int
    abi.argtypes = []
    if abi() != 1:
        raise RuntimeError(f"{path}: entry point ABI {abi()}, this add-on speaks 1")
    entry.restype = ctypes.c_int
    entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), _SinkType, ctypes.c_void_p]
    _library = library
    return library


def bundle_dirs():
    """Where the Mesh2Splat and SplatBakeFilter bundles are: beside the
    plugin in a package (`plugin/aofx`), or the build tree's `aofx`. The
    plugin also knows its own build tree's."""
    from . import plugin_dir
    directory = plugin_dir()
    if not directory:
        return []
    found = []
    for candidate in (os.path.join(os.path.dirname(directory), "aofx"),
                      os.path.join(os.path.dirname(os.path.dirname(directory)), "aofx")):
        if os.path.isdir(candidate) and candidate not in found:
            found.append(candidate)
    return found


class AtheneaConvertSettings(bpy.types.PropertyGroup):
    """The panel's options: the main ones of `athenea mesh2splat`."""
    budget: bpy.props.IntProperty(
        name="Budget", description="Most gaussians over the whole selection (--max-splats), shared "
        "between the meshes in proportion to what each wants", default=2_000_000, min=1)
    resolution: bpy.props.IntProperty(
        name="Resolution", description="Cells across the longest side of the selection (--resolution)",
        default=512, min=8, max=65536)
    bake: bpy.props.BoolProperty(
        name="Bake light", description="Bake the light the meshes receive into the gaussians; off, the "
        "cloud carries the material and is relit every frame (--no-bake)", default=True)
    bake_degree: bpy.props.IntProperty(
        name="Degree", description="Spherical harmonics the bake fits; 0 is one colour a gaussian "
        "(--bake-degree)", default=2, min=0, max=3)
    bake_samples: bpy.props.IntProperty(
        name="Samples", description="Paths every gaussian takes first (--bake-samples)",
        default=128, min=1, max=65536)
    lights: bpy.props.EnumProperty(
        name="Lights", description="What the bake is lit by",
        items=(('SCENE', "Scene", "The scene's visible lights and its world, exported with the meshes"),
               ('DEFAULT', "Default", "A dome and a sun (--default-lights), whatever the scene has")),
        default='SCENE')
    lod_levels: bpy.props.IntProperty(
        name="LOD levels", description="Levels of detail, each converted again at half the resolution "
        "(--lod-levels); a .athc builds its own instead", default=1, min=1, max=8)
    skinned: bpy.props.BoolProperty(
        name="Skinned", description="Carry the armature: the cloud deforms with the rig; forces no bake "
        "(--skinned)", default=False)
    output_format: bpy.props.EnumProperty(
        name="Format", items=(('USDC', "USD (.usdc)", "A ParticleField stage: everything the conversion keeps"),
                              ('ATHC', ".athc", "Levels of detail and streaming; no rig, no relit material")),
        default='USDC')
    bring_in: bpy.props.EnumProperty(
        name="Bring in as",
        items=(('REFERENCE', "Referenced USD", "An Empty whose cloud hdAthenea draws from the file as "
                "written (rig, levels, relit material, normals)"),
               ('POINTS', "Gaussian-splat points", "Blender's own Gaussian-splat point cloud, imported: "
                "positions, sizes, rotations and harmonics only")),
        default='REFERENCE')
    directory: bpy.props.StringProperty(
        name="Directory", description="Where the exported meshes and the cloud are written",
        default="//splats/", subtype='DIR_PATH')
    hide_source: bpy.props.BoolProperty(
        name="Hide the meshes", description="Hide the converted meshes once the cloud is in", default=True)
    # What the last conversion said, for the panel.
    status: bpy.props.StringProperty(name="Status", default="")
    progress: bpy.props.FloatProperty(name="Progress", default=0.0, min=0.0, max=1.0, subtype='FACTOR')


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "selection"


class Conversion:
    """One conversion: export, mesh2splat on a thread, bring in."""

    def __init__(self, context, settings):
        self.settings = settings
        self.scene = context.scene
        self.view_layer = context.view_layer
        self.meshes = [o for o in context.selected_objects if o.type == 'MESH']
        self.lines = queue.Queue()
        self.log = []
        self.errors = []
        self.code = None
        self.thread = None
        self.started = 0.0
        self.mesh_stage = ""
        self.output = ""
        self.levels_written = 0
        self.levels = 1
        self.brought_in = []
        self._sink = _SinkType(self._on_line)

    # -- checks -----------------------------------------------------------

    def validate(self):
        """What would make the conversion wrong, said before anything runs."""
        s = self.settings
        if not bpy.data.filepath:
            return "Save the .blend first: the cloud is written next to it"
        if not self.meshes:
            return "Select at least one mesh object"
        if s.output_format == 'ATHC':
            if s.skinned:
                return "A .athc cannot carry a rig: choose USD for a skinned cloud"
            if s.lod_levels > 1:
                return "A .athc builds its own levels of detail: set LOD levels to 1"
            if s.bring_in == 'POINTS':
                return "Blender cannot import a .athc: bring it in as Referenced USD"
        if s.bring_in == 'POINTS' and (s.skinned or s.lod_levels > 1):
            return ("Gaussian-splat points keep neither a rig nor levels of detail: "
                    "bring the cloud in as Referenced USD")
        return None

    # -- 1. export ----------------------------------------------------------

    def export(self):
        s = self.settings
        directory = bpy.path.abspath(s.directory)
        os.makedirs(directory, exist_ok=True)
        stem = _safe(os.path.splitext(os.path.basename(bpy.data.filepath))[0])
        name = _safe(self.meshes[0].name if len(self.meshes) == 1 else "selection")
        self.mesh_stage = os.path.join(directory, f"{stem}_{name}_mesh.usdc")
        extension = ".athc" if s.output_format == 'ATHC' else ".usdc"
        self.output = os.path.join(directory, f"{stem}_{name}{extension}")
        self.levels = max(1, s.lod_levels) if s.output_format == 'USDC' else 1

        export = set(self.meshes)
        if s.skinned:
            for mesh in self.meshes:
                for modifier in mesh.modifiers:
                    if modifier.type == 'ARMATURE' and modifier.object is not None:
                        export.add(modifier.object)
                if mesh.parent is not None and mesh.parent.type == 'ARMATURE':
                    export.add(mesh.parent)
        bake = s.bake and not s.skinned
        if bake and s.lights == 'SCENE':
            export.update(o for o in self.scene.objects if o.type == 'LIGHT' and o.visible_get())

        selected = [o for o in self.view_layer.objects if o.select_get()]
        active = self.view_layer.objects.active
        try:
            for o in selected:
                o.select_set(False)
            for o in export:
                o.select_set(True)
            result = bpy.ops.wm.usd_export(
                filepath=self.mesh_stage, check_existing=False, selected_objects_only=True,
                export_animation=s.skinned, export_armatures=s.skinned, only_deform_bones=s.skinned,
                export_materials=True, generate_preview_surface=True, export_textures_mode='NEW',
                overwrite_textures=True, export_cameras=False, export_points=False, export_volumes=False,
                export_hair=False, export_curves=False, export_lights=bake and s.lights == 'SCENE',
                convert_world_material=bake and s.lights == 'SCENE', evaluation_mode='RENDER')
        finally:
            for o in export:
                o.select_set(False)
            for o in selected:
                o.select_set(True)
            self.view_layer.objects.active = active
        if 'FINISHED' not in result or not os.path.isfile(self.mesh_stage):
            raise RuntimeError(f"Blender's USD export did not write {self.mesh_stage}")

    # -- 2. mesh2splat ------------------------------------------------------

    def arguments(self):
        s = self.settings
        bake = s.bake and not s.skinned
        args = [self.mesh_stage, "-o", self.output, "--no-camera",
                "--max-splats", str(s.budget), "--resolution", str(s.resolution)]
        if bake:
            args += ["--bake-degree", str(s.bake_degree), "--bake-samples", str(s.bake_samples)]
            if s.lights == 'DEFAULT':
                args.append("--default-lights")
        else:
            args.append("--no-bake")
        if s.skinned:
            args.append("--skinned")
        if self.levels > 1:
            args += ["--lod-levels", str(self.levels)]
        for directory in bundle_dirs():
            args += ["--path", directory]
        return args

    def _on_line(self, error, text, _user):
        self.lines.put((int(error), text.decode("utf-8", "replace") if text else ""))

    def start(self):
        library = plugin_library()
        args = self.arguments()
        encoded = [a.encode("utf-8") for a in args]
        argv = (ctypes.c_char_p * len(encoded))(*encoded)
        self.log.append("mesh2splat " + " ".join(args))

        def work():
            try:
                self.code = library.athenea_mesh2splat(len(encoded), argv, self._sink, None)
            except Exception as error:   # a ctypes failure, not the command's
                self.lines.put((1, f"mesh2splat: {error}\n"))
                self.code = 1

        self.started = time.monotonic()
        self.thread = threading.Thread(target=work, name="athenea-mesh2splat", daemon=True)
        self.thread.start()

    def drain(self):
        """The lines so far, into the log and the progress. True while running."""
        while True:
            try:
                error, text = self.lines.get_nowait()
            except queue.Empty:
                break
            for line in text.splitlines():
                line = line.rstrip()
                if not line:
                    continue
                self.log.append(line)
                if error:
                    self.errors.append(line)
                self._advance(line)
        return self.thread is not None and self.thread.is_alive()

    def _advance(self, line):
        """A fraction from what mesh2splat says it has done: a level is read
        (textures), converted (a mesh's `-> N splats`), baked, then written."""
        s = self.settings
        within = 0.05
        if "textures decoded" in line:
            within = 0.15
        elif " splats of " in line:
            within = 0.35
        elif line.startswith("mesh2splat: baked") or line.startswith("mesh2splat: traced"):
            within = 0.85
        elif line.startswith("mesh2splat: wrote"):
            self.levels_written += 1
            within = 0.0
        fraction = min((self.levels_written + within) / max(self.levels, 1), 1.0)
        s.progress = max(s.progress, fraction)
        s.status = line.replace("mesh2splat: ", "")[:160]

    def wait(self):
        """For a script (Blender in the background has no event loop)."""
        while self.drain():
            time.sleep(0.1)
        self.drain()

    # -- 3. bring in --------------------------------------------------------

    def bring_in(self):
        s = self.settings
        name = (self.meshes[0].name if len(self.meshes) == 1 else "selection") + "_splats"
        collection = self.view_layer.active_layer_collection.collection
        if s.bring_in == 'REFERENCE':
            empty = bpy.data.objects.new(name, None)
            empty.empty_display_type = 'CUBE'
            empty.empty_display_size = 0.25
            empty[CLOUD_PROPERTY] = bpy.path.relpath(self.output)
            collection.objects.link(empty)
            self.brought_in = [empty]
        else:
            before = set(bpy.data.objects)
            result = bpy.ops.wm.usd_import(
                filepath=self.output, import_cameras=False, import_lights=False, import_materials=False,
                import_meshes=False, import_points=True, import_volumes=False, import_curves=False,
                set_frame_range=False, create_world_material=False)
            if 'FINISHED' not in result:
                raise RuntimeError(f"Blender's USD import did not read {self.output}")
            self.brought_in = [o for o in bpy.data.objects if o not in before]
            for o in self.brought_in:
                if o.type == 'POINTCLOUD':
                    o.name = name
                    # The conversion writes light, not a capture's sRGB; the
                    # import keeps no flag for it, so the object does.
                    o[LINEAR_PROPERTY] = True
        if s.hide_source:
            for mesh in self.meshes:
                mesh.hide_render = True
                mesh.hide_set(True)
        # Blender's Hydra export method hands a delegate no point cloud and
        # runs no USD export hook: the cloud is drawn only with USD's.
        hydra = getattr(self.scene, "hydra", None)
        if hydra is not None and hydra.export_method != 'USD':
            hydra.export_method = 'USD'
            self.log.append("athenea: the scene's Hydra export method is now USD, which the cloud needs")


class ATHENEA_OT_mesh_to_splats(bpy.types.Operator):
    """Convert the selected meshes into a gaussian cloud with mesh2splat (on the GPU)"""
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

    def _end(self, context, conversion):
        ATHENEA_OT_mesh_to_splats.running = False
        settings = context.scene.athenea_convert
        seconds = time.monotonic() - conversion.started
        for line in conversion.log:
            print("athenea_hydra:", line)
        if conversion.code != 0 or not os.path.exists(conversion.output):
            message = conversion.errors[-1] if conversion.errors else f"mesh2splat exited with {conversion.code}"
            if conversion.code == 3:
                message += " (the GPU ran out of memory: a smaller budget or resolution)"
            settings.status = message
            self.report({'ERROR'}, message)
            return {'CANCELLED'}
        try:
            conversion.bring_in()
        except RuntimeError as error:
            settings.status = str(error)
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        settings.progress = 1.0
        settings.status = f"wrote {bpy.path.relpath(conversion.output)} in {seconds:.1f} s"
        for warning in conversion.errors:
            self.report({'WARNING'}, warning)
        self.report({'INFO'}, f"Gaussian splats: {settings.status}")
        return {'FINISHED'}

    def execute(self, context):
        # Without a window (blender -b, a script) there is no event loop to
        # be modal in: the conversion runs to its end here.
        conversion = self._begin(context)
        if conversion is None:
            return {'CANCELLED'}
        conversion.wait()
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

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        settings = context.scene.athenea_convert
        alive = self._conversion.drain()
        context.window_manager.progress_update(settings.progress)
        if context.workspace is not None:
            context.workspace.status_text_set(f"Gaussian splats: {settings.status}")
        for area in context.screen.areas if context.screen else []:
            if area.type in {'VIEW_3D', 'PROPERTIES'}:
                area.tag_redraw()
        if alive:
            return {'RUNNING_MODAL'}
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        if context.workspace is not None:
            context.workspace.status_text_set(None)
        return self._end(context, self._conversion)


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
        layout.prop(settings, "budget")
        layout.prop(settings, "resolution")
        column = layout.column(heading="Bake")
        column.active = not settings.skinned
        column.prop(settings, "bake", text="Light")
        sub = column.column()
        sub.active = settings.bake and not settings.skinned
        sub.prop(settings, "bake_degree")
        sub.prop(settings, "bake_samples")
        sub.prop(settings, "lights")
        layout.prop(settings, "lod_levels")
        layout.prop(settings, "skinned")
        layout.prop(settings, "output_format")
        layout.prop(settings, "bring_in")
        layout.prop(settings, "directory")
        layout.prop(settings, "hide_source")
        layout.operator(ATHENEA_OT_mesh_to_splats.bl_idname, icon='POINTCLOUD_DATA')
        if ATHENEA_OT_mesh_to_splats.running:
            layout.progress(factor=settings.progress, text=settings.status or "converting")
        elif settings.status:
            layout.label(text=settings.status)


classes = (AtheneaConvertSettings, ATHENEA_OT_mesh_to_splats, VIEW3D_PT_athenea_splats)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.athenea_convert = bpy.props.PointerProperty(type=AtheneaConvertSettings)


def unregister():
    del bpy.types.Scene.athenea_convert
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
