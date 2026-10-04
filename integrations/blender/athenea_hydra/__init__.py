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

The module, in Blender's terms:
  - the render engine *Athenea*: the viewport and F12 draw the scene through
    hdAthenea, a converted cloud relit under the scene's world (a DomeLight)
    and lights, its shadow catcher composited (render settings below);
  - *Convert to Gaussian Splats* (convert.py): selected meshes into a TX
    cloud and its shadow catcher, by mesh2splat in this process;
  - *Export Gaussian Splats* (export.py): the clouds as SPZ / PLY / glTF
    under the current world, by `athenea flatten`;
  - *Compare with Cycles* (look.py): the meshes in Cycles against the clouds
    in Athenea from the same camera, measured by `athenea compare`.
"""

import os

import bpy

from . import convert, export, look

# The OpenUSD hdAthenea is compiled against: Blender's (its libusd_ms, in its
# own namespace). The target is Blender's development build, whose USD can
# move under it; one whose USD is another would not load the plugin, and
# says so less clearly than this.
USD_VERSION = (0, 26, 8)

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
            return os.path.abspath(candidate)
    return None


def usd_mismatch():
    """Why this Blender's USD cannot load hdAthenea, or None."""
    try:
        from pxr import Usd
        version = tuple(Usd.GetVersion())
    except ImportError:
        return "this Blender has no USD module"
    if version != USD_VERSION:
        built = ".".join(str(v) for v in USD_VERSION[1:])
        found = ".".join(str(v) for v in version[1:])
        return (f"hdAthenea was built against OpenUSD {built} and this Blender has {found}: "
                "rebuild it against this Blender (scripts/build-usd-blender.sh, preset macos-arm64-blender)")
    return None


def register_plugin():
    """Hands hdAthenea to USD's plugin registry, once per process."""
    global _registered_plugin_dir
    if _registered_plugin_dir:
        return True
    problem = usd_mismatch()
    if problem:
        print(f"athenea_hydra: {problem}")
        return False
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


class AtheneaRenderSettings(bpy.types.PropertyGroup):
    """What the render panel offers: little, since the frame's look is the
    cloud's and the world's. The viewport always draws the raster."""
    technique: bpy.props.EnumProperty(
        name="Final render",
        items=(('RASTER', "Raster", "The gaussians rasterised, relit by the world and the lights: what the "
                "viewport shows and a player draws"),
               ('PATH', "Path traced", "The ground truth: the scene path traced (slow), what the raster is "
                "measured against")),
        default='RASTER')
    paths: bpy.props.IntProperty(
        name="Paths", description="Paths a pixel a path traced render gathers (athenea:pathTotal)",
        default=256, min=1, max=65536)
    denoise: bpy.props.BoolProperty(
        name="Denoise", description="Open Image Denoise over the path traced frame", default=True)
    indirect: bpy.props.BoolProperty(
        name="Bounced light", description="A converted cloud adds the light it bounced "
        "(athenea:splatTransferIndirect)", default=True)


class AtheneaHydraRenderEngine(bpy.types.HydraRenderEngine):
    bl_idname = 'ATHENEA_HYDRA'
    bl_label = "Athenea"
    bl_info = "Athenea's Hydra render delegate"

    bl_use_preview = False
    bl_use_gpu_context = False
    bl_use_materialx = True

    bl_delegate_id = 'HdAtheneaRendererPlugin'

    def get_render_settings(self, engine_type):
        _warn_hidden_splats()
        settings = getattr(bpy.context.scene, "athenea", None)
        common = {'athenea:splatTransferIndirect': bool(settings.indirect) if settings else True}
        if engine_type == 'VIEWPORT':
            # Blender draws a non-Storm delegate's viewport by mapping its
            # colour buffer and uploading it to a texture (DrawTexture::
            # create_from_buffer, every frame): half floats are half the
            # bytes read back and uploaded, and Blender takes them as such.
            return dict(common, **{'athenea:colourHalf': True, 'athenea:technique': "raster"})
        final = dict(common, **{
            'aovToken:Combined': "color",
            'aovToken:Depth': "depth",
            'aovToken:Normal': "normal",
            'aovToken:DiffCol': "albedo",
        })
        if settings is not None and settings.technique == 'PATH':
            final.update({'athenea:technique': "rt", 'athenea:pathTotal': int(settings.paths),
                          'athenea:pathSamples': 16, 'athenea:pathBounces': 6,
                          'athenea:denoise': bool(settings.denoise)})
        else:
            final['athenea:technique'] = "raster"
        return final

    def update_render_passes(self, scene, render_layer):
        if render_layer.use_pass_combined:
            self.register_pass(scene, render_layer, 'Combined', 4, 'RGBA', 'COLOR')
        if render_layer.use_pass_z:
            self.register_pass(scene, render_layer, 'Depth', 1, 'Z', 'VALUE')
        if render_layer.use_pass_normal:
            self.register_pass(scene, render_layer, 'Normal', 3, 'XYZ', 'VECTOR')
        if render_layer.use_pass_diffuse_color:
            self.register_pass(scene, render_layer, 'DiffCol', 3, 'RGB', 'COLOR')


class RENDER_PT_athenea(bpy.types.Panel):
    bl_label = "Athenea"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'ATHENEA_HYDRA'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        settings = context.scene.athenea
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(settings, "technique")
        column = layout.column()
        column.active = settings.technique == 'PATH'
        column.prop(settings, "paths")
        column.prop(settings, "denoise")
        layout.prop(settings, "indirect")
        hydra = getattr(context.scene, "hydra", None)
        if hydra is not None and hydra.export_method != 'USD':
            box = layout.box()
            box.label(text="Gaussian splats need the USD export method", icon='ERROR')
            box.prop(hydra, "export_method")
        problem = usd_mismatch()
        if problem:
            layout.label(text=problem, icon='ERROR')
        box = layout.box()
        box.operator(look.ATHENEA_OT_compare_cycles.bl_idname, icon='IMAGE_REFERENCE')
        result = context.scene.athenea_look
        if result.rel_mse >= 0.0:
            box.label(text=f"against Cycles: relMSE {result.rel_mse:.4g}, 8-bit p99 {result.p99}")


class AtheneaSplatExportHook(bpy.types.USDHook):
    """Writes what Blender's USD export leaves out of a Gaussian-splat cloud.

    A PointCloud of type GAUSSIAN_SPLAT reaches USD -- and through it Hydra,
    with the USD export method -- as a Points prim carrying its attributes as
    primvars, all but `radiance:base` (a FLOAT4: the DC coefficient and the
    opacity), which the writer has no USD type for. This hook adds it, as
    `primvars:radiance:base` (float4[], vertex), copied from the evaluated
    attribute as it is: no value is computed here. hdAthenea draws such a
    Points prim as a splat cloud and lays the attributes out on the GPU.

    It runs inside every USD export (Hydra's own included). A point cloud is
    completed only where the scene renders with Athenea; a converted cloud's
    Empty is given its reference in every export.
    """
    bl_idname = "athenea_splat_export"
    bl_label = "Athenea Gaussian splats"
    bl_description = "Exports a Gaussian-splat point cloud's radiance:base, which Blender's USD writer drops"

    @staticmethod
    def on_export(export_context):
        depsgraph = export_context.get_depsgraph()
        if depsgraph is None:
            return True
        # A cloud's Empty references its file in every export (a stage for
        # `athenea flatten` is one, whatever engine the scene renders with);
        # a point cloud's radiance only where Athenea draws it.
        athenea = depsgraph.scene.render.engine == AtheneaHydraRenderEngine.bl_idname
        stage = export_context.get_stage()
        for path, ids in export_context.get_prim_map().items():
            for owner in ids:
                if not isinstance(owner, bpy.types.Object):
                    continue
                if owner.type == 'POINTCLOUD' and athenea:
                    _write_splat_base(stage, stage.GetPrimAtPath(path), owner.evaluated_get(depsgraph),
                                      bool(owner.get(convert.LINEAR_PROPERTY, False)))
                elif owner.type == 'EMPTY' and convert.CLOUD_PROPERTY in owner:
                    _reference_cloud(stage, stage.GetPrimAtPath(path), owner[convert.CLOUD_PROPERTY])
        return True


def _reference_cloud(stage, xform, file):
    """The cloud a conversion wrote, under the Empty that stands for it.

    A USD file is referenced whole (its default prim: the ParticleField, its
    rig, its levels of detail); a `.athc` is named by a ParticleField with
    AtheneaStreamedAssetAPI, which hdAthenea streams. Nothing is read here.
    """
    from pxr import Sdf, UsdGeom
    if not xform:
        return
    path = bpy.path.abspath(file)
    if not os.path.isfile(path):
        print(f"athenea_hydra: {xform.GetPath()}: the cloud {path} is not there; not referenced")
        return
    at = xform.GetPath().AppendChild("cloud")
    if path.endswith(".athc"):
        prim = stage.DefinePrim(at, "ParticleField3DGaussianSplat")
        prim.AddAppliedSchema("AtheneaStreamedAssetAPI")
        primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "athenea:asset", Sdf.ValueTypeNames.Asset, UsdGeom.Tokens.constant)
        primvar.Set(Sdf.AssetPath(path))
    else:
        stage.DefinePrim(at).GetReferences().AddReference(path)


def _write_splat_base(stage, xform, evaluated, linear=False):
    """`radiance:base` onto the Points prim the writer made under `xform`,
    and, for a converted cloud, that its colours are linear light."""
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
        if linear:
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "athenea:splat:linear", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)


def _warn_hidden_splats():
    """Blender's Hydra export method hands no point cloud to a delegate:
    a scene with Gaussian splats renders them only with the USD one."""
    scene = bpy.context.scene
    hydra = getattr(scene, "hydra", None)
    if hydra is None or hydra.export_method != 'HYDRA':
        return
    if any(o.type == 'POINTCLOUD' and getattr(o.data, "type", None) == 'GAUSSIAN_SPLAT'
           for o in scene.objects):
        print("athenea_hydra: Gaussian splats are drawn only with the USD export method "
              "(Render Properties > Hydra > Export Method, scene.hydra.export_method = 'USD')")


def _panels():
    """Blender's panels that say they work with this engine."""
    exclude = {'VIEWLAYER_PT_filter', 'VIEWLAYER_PT_layer_passes'}
    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        engines = getattr(panel, 'COMPAT_ENGINES', None)
        if engines is not None and 'BLENDER_RENDER' in engines and panel.__name__ not in exclude:
            panels.append(panel)
    return panels


_classes = (AtheneaPreferences, AtheneaRenderSettings, AtheneaHydraRenderEngine, RENDER_PT_athenea,
            AtheneaSplatExportHook)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.athenea = bpy.props.PointerProperty(type=AtheneaRenderSettings)
    convert.register()
    export.register()
    look.register()
    problem = usd_mismatch()
    if problem:
        # Registered all the same, so the panel can say it; nothing loads.
        print(f"athenea_hydra: {problem}")
    else:
        register_plugin()
    for panel in _panels():
        panel.COMPAT_ENGINES.add(AtheneaHydraRenderEngine.bl_idname)


def unregister():
    for panel in _panels():
        panel.COMPAT_ENGINES.discard(AtheneaHydraRenderEngine.bl_idname)
    look.unregister()
    export.unregister()
    convert.unregister()
    del bpy.types.Scene.athenea
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
