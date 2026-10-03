# Copyright (c) 2026 jesus luque.
"""Athenea as a Blender render engine, through Blender's own Hydra.

Blender's HydraRenderEngine creates the render delegate named by
`bl_delegate_id` through its own USD (libusd_ms). hdAthenea is a Hydra
plugin built against that USD (the macos-arm64-blender preset); it has to be
known to USD's plugin registry before the first engine is made, which is
what register() does with pxr.Plug -- Blender 5.x has no
_bpy_hydra.register_plugins.

Where the plugin is: $ATHENEA_HYDRA_PLUGIN_DIR, else `plugin/usd` inside
this add-on (a packaged build), else the add-on preference.
"""

import os

import bpy

_registered_plugin_dir = None


def _addon_dir():
    return os.path.dirname(os.path.abspath(__file__))


def plugin_dir():
    """The directory holding hdAthenea/ and atheneaSchemas/ (their plugInfo)."""
    candidates = [os.environ.get("ATHENEA_HYDRA_PLUGIN_DIR", ""),
                  os.path.join(_addon_dir(), "plugin", "usd")]
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        candidates.append(prefs.plugin_dir)
    except (KeyError, AttributeError):
        pass
    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "hdAthenea", "resources", "plugInfo.json")):
            return candidate
    return None


def register_plugin():
    """Hands hdAthenea to USD's plugin registry, once per process."""
    global _registered_plugin_dir
    if _registered_plugin_dir:
        return True
    directory = plugin_dir()
    if not directory:
        print("athenea_hydra: hdAthenea not found; set ATHENEA_HYDRA_PLUGIN_DIR or the add-on preference")
        return False
    # MaterialX 1.39.5's libraries for hdAthenea's material compiler
    # (Blender's are 1.39.4's, without the Slang generator's).
    materialx = os.path.join(os.path.dirname(directory), "materialx")
    if os.path.isdir(os.path.join(materialx, "libraries")) and not os.environ.get("ATHENEA_MATERIALX_ROOT"):
        os.environ["ATHENEA_MATERIALX_ROOT"] = materialx
    from pxr import Plug
    registered = Plug.Registry().RegisterPlugins(directory)
    names = [p.name for p in registered]
    print(f"athenea_hydra: registered {names} from {directory}")
    _registered_plugin_dir = directory
    return "hdAthenea" in names or Plug.Registry().GetPluginWithName("hdAthenea") is not None


class AtheneaPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    plugin_dir: bpy.props.StringProperty(
        name="hdAthenea plugin directory",
        description="The directory holding hdAthenea/ (a build's plugin/usd)",
        subtype='DIR_PATH',
    )

    def draw(self, context):
        self.layout.prop(self, "plugin_dir")


class AtheneaHydraRenderEngine(bpy.types.HydraRenderEngine):
    bl_idname = 'ATHENEA_HYDRA'
    bl_label = "Athenea"
    bl_info = "Athenea's Hydra render delegate"

    bl_use_preview = False
    bl_use_gpu_context = False
    bl_use_materialx = True

    bl_delegate_id = 'HdAtheneaRendererPlugin'

    def get_render_settings(self, engine_type):
        if engine_type == 'VIEWPORT':
            return {}
        return {
            'aovToken:Combined': "color",
            'aovToken:Depth': "depth",
        }

    def update_render_passes(self, scene, render_layer):
        if render_layer.use_pass_combined:
            self.register_pass(scene, render_layer, 'Combined', 4, 'RGBA', 'COLOR')
        if render_layer.use_pass_z:
            self.register_pass(scene, render_layer, 'Depth', 1, 'Z', 'VALUE')


class AtheneaSplatExportHook(bpy.types.USDHook):
    """Writes what Blender's USD export leaves out of a Gaussian-splat cloud.

    A PointCloud of type GAUSSIAN_SPLAT reaches USD -- and through it Hydra,
    with the USD export method -- as a Points prim carrying its attributes as
    primvars, all but `radiance:base` (a FLOAT4: the DC coefficient and the
    opacity), which the writer has no USD type for. This hook adds it, as
    `primvars:radiance:base` (float4[], vertex), copied from the evaluated
    attribute as it is: no value is computed here. hdAthenea draws such a
    Points prim as a splat cloud and lays the attributes out on the GPU.

    It runs inside every USD export (Hydra's own included), and acts only
    where the scene renders with Athenea.
    """
    bl_idname = "athenea_splat_export"
    bl_label = "Athenea Gaussian splats"
    bl_description = "Exports a Gaussian-splat point cloud's radiance:base, which Blender's USD writer drops"

    @staticmethod
    def on_export(export_context):
        depsgraph = export_context.get_depsgraph()
        if depsgraph is None or depsgraph.scene.render.engine != AtheneaHydraRenderEngine.bl_idname:
            return True
        stage = export_context.get_stage()
        for path, ids in export_context.get_prim_map().items():
            for owner in ids:
                if isinstance(owner, bpy.types.Object) and owner.type == 'POINTCLOUD':
                    _write_splat_base(stage, stage.GetPrimAtPath(path), owner.evaluated_get(depsgraph))
        return True


def _write_splat_base(stage, xform, evaluated):
    """`radiance:base` onto the Points prim the writer made under `xform`."""
    import numpy
    from pxr import Sdf, UsdGeom, Vt
    cloud = evaluated.data
    base = cloud.attributes.get("radiance:base") if cloud is not None else None
    if not xform or base is None or base.data_type != 'FLOAT4' or base.domain != 'POINT':
        return
    for prim in xform.GetChildren():
        if not prim.IsA(UsdGeom.Points):
            continue
        points = UsdGeom.Points(prim).GetPointsAttr().Get()
        if points is None or len(points) != len(base.data):
            print(f"athenea_hydra: {prim.GetPath()}: {len(base.data)} radiance:base values for "
                  f"{0 if points is None else len(points)} points; not written")
            continue
        values = numpy.empty((len(base.data), 4), dtype=numpy.float32)
        base.data.foreach_get("vector", values.ravel())
        primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "radiance:base", Sdf.ValueTypeNames.Float4Array, UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec4fArray.FromNumpy(values))


def _panels():
    """Blender's panels that say they work with this engine."""
    exclude = {'VIEWLAYER_PT_filter', 'VIEWLAYER_PT_layer_passes'}
    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        engines = getattr(panel, 'COMPAT_ENGINES', None)
        if engines is not None and 'BLENDER_RENDER' in engines and panel.__name__ not in exclude:
            panels.append(panel)
    return panels


_classes = (AtheneaPreferences, AtheneaHydraRenderEngine, AtheneaSplatExportHook)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    register_plugin()
    for panel in _panels():
        panel.COMPAT_ENGINES.add(AtheneaHydraRenderEngine.bl_idname)


def unregister():
    for panel in _panels():
        panel.COMPAT_ENGINES.discard(AtheneaHydraRenderEngine.bl_idname)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
