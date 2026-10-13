# Preparing a USD stage for athenea

This page covers what a USD stage says to athenea and how a host application
finds the engine: the environment a host such as usdview or Blender runs with,
the materials the conversion and the renderer read, the lights and their
units, the `athenea:*` render settings with their products and vars, the
codeless API schemas a cloud carries, and what the engine ignores and what it
refuses. Every name below is the one the engine reads; the full tables are in
the operations document linked at the end.

## Pointing an application at the plugin

`<build>/plugin/usd` holds the Hydra delegate `hdAthenea` and the codeless
schemas together, so one variable, `PXR_PLUGINPATH_NAME`, finds both. A host
then offers the renderer under its own name, and the declared render
settings appear in its renderer-settings panel.

| Variable | What it does |
|---|---|
| `PXR_PLUGINPATH_NAME` | points a USD application at `<build>/plugin/usd`. Needed by any host that is not `athenea` itself, which registers `<its binary>/../plugin/usd` on its own at start-up. |
| `ATHENEA_SHADER_DIR` | where the Slang shaders are read from. Without it the engine looks for a `shaders` directory holding `athenea/` beside the image its code was loaded from, or up to three directories above it — `<build>/shaders` for the Hydra plugin loaded by usdview or Blender — then beside the executable, then at the directory it was built with. `athenea info` prints the one in use. |
| `ATHENEA_MATERIALX_ROOT` | a directory holding MaterialX's `libraries/`, read by hdAthenea's material compiler in place of the libraries the host's USD loaded. Unset by default. For a host whose MaterialX predates the Slang generator (Blender 5.3 ships 1.39.4: no `genslang` implementations, older node definitions), pointed at 1.39.5's. Read once, when the first material compiles; the host's own renderers keep theirs. |

```sh
export PXR_PLUGINPATH_NAME=<build>/plugin/usd
export ATHENEA_SHADER_DIR=<build>/shaders
export ATHENEA_MATERIALX_ROOT=<MaterialX 1.39.5>   # the directory holding libraries/
```

A host that does not offer the renderer has not found the plugin. A shader
compile error naming a path means the shaders on disk do not match the binary;
`ATHENEA_SHADER_DIR` set to this build's `shaders` resolves it.

## Materials

The renderer reads `UsdPreviewSurface` 2.6 with both `opacityMode`s, and
MaterialX 1.39 graphs, OpenPBR among them with its weights held to the
specification's range. Graphs are compiled into Slang modules, named by a
hash of their source, so materials that differ only in values share one.

`athenea mesh2splat` reads the same materials when it turns meshes into
gaussians. What it takes from each model:

| What | UsdPreviewSurface | standard_surface | OpenPBR | glTF |
|---|---|---|---|---|
| opacity map with no threshold of its own (cut at `--opacity-cut`, `0.5`) | `opacity` | `opacity` | `geometry_opacity` | `alpha` in BLEND |
| the material's own threshold, used instead | `opacityThreshold` | | | `alpha_cutoff` in MASK |
| emission | `emissiveColor` | `emission` times `emission_color` | `emission_luminance` times `emission_color` (nits) | `emissive` times `emissive_strength` |

The height that displaces gaussians is read from UsdPreviewSurface's
`displacement` (through UsdUVTexture's `scale` and `bias`, on the channel
connected), or from a MaterialX `displacement` node (`ND_displacement_float`,
times its `scale`) that the material's displacement terminal names, in the
mesh's own units. A thin-walled glass is OpenPBR's `geometry_thin_walled`.
The glass material's IOR is written as `primvars:athenea:splat:ior`; a cloud keeps one
index, and with two glasses of different IOR the first is kept and the
conversion says so. A vector displacement is not read, and an OpenPBR coat
over the emission is not carried. A mesh whose GeomSubsets (`materialBind`
family) bind materials of their own is converted a subset at a time, each with
its material.

A texture is read in the colour space its material names — MaterialX's
`colorspace`, a UsdUVTexture's `sourceColorSpace`, USD's `colorSpace` on the
file input — and brought into the working space (linear Rec.709) on the
device. `raw` and `data` read the file as data; no name, or `auto`, reads an
8-bit image as sRGB and anything else as linear.

```usda
def Material "Screen"
{
    token outputs:surface.connect = </World/Looks/Screen/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:emissiveColor.connect = </World/Looks/Screen/Glow.outputs:rgb>
        float inputs:opacityThreshold = 0.5
        token outputs:surface
    }

    def Shader "Glow"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./emission_gradient.png@
        token inputs:sourceColorSpace = "sRGB"
        float2 inputs:st.connect = </World/Looks/Screen/UV.outputs:result>
        float3 outputs:rgb
    }

    def Shader "UV"
    {
        uniform token info:id = "UsdPrimvarReader_float2"
        string inputs:varname = "st"
        float2 outputs:result
    }
}
```

## Lights and their units

UsdLux lights are read with `ShadowAPI` (`shadow:color`, `shadow:distance`,
`shadow:falloff`), `DomeLight_1.poleAxis`, light linking and IES profiles.

A `DistantLight` is in UsdLux's units. `intensity` (times `2^exposure` and
`color`) is the radiance of the sun's disc, in nits. With `normalize = 1` it
is divided by the disc's projected solid angle, `pi sin^2(angle / 2)`, and
becomes the illuminance on a surface facing the light, in lux. An `angle` of 0
is a parallel light whose irradiance is the intensity either way. A sun
authored without `normalize` therefore lays `intensity * pi sin^2(angle / 2)`:
UsdLux's default sun (50000 at 0.53 degrees) lays about 3.4, and one of
intensity 3 at the default angle is all but black.

```usda
def DistantLight "Sun"
{
    float inputs:angle = 0.53
    float inputs:intensity = 3
    bool inputs:normalize = 1
    string athenea:lightGroup = "key"
}
```

With `normalize = 1`, the sun above lays an illuminance of 3 lux on a surface
facing it. A `DomeLight` ignores `normalize`, as the schema says. A cloud that
asked to be relit takes its body from the dome image's irradiance and its
reflection from the sky convolved to its roughness; four domes are prepared,
and a fifth is drawn as its colour. `--default-lights` on `athenea stage` puts a
dome and a sun in the session layer for a stage that authors none; the
default sun is normalized.

`athenea:lightGroup` (a string on a light prim) names the group a light's
contribution is gathered under; USD has no light groups of its own.

## Render settings

Settings are authored in the `athenea:` namespace on a `UsdRenderSettings`
prim, or set by a host through the delegate. The settings the delegate
declares appear in a host's settings panel; among them:

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `athenea:technique` | token | `raster` | `raster` or `rt` |
| `athenea:visibility` | token | `automatic` | how meshes are seen: `automatic`, `raster`, `rays`, `bvh` |
| `athenea:settleStreams` | bool | `false` | wait for streamed assets before drawing; a still sets it |
| `athenea:lightSamples` | int | `1` | samples per light |
| `athenea:pathSamples` | int | `1` | rt: paths a pixel each pass |
| `athenea:pathBounces` | int | `1` | rt: bounces after the first hit |
| `athenea:pathTotal` | int | `1` | rt: paths a pixel to converge to |
| `athenea:denoise` | bool | `false` | denoise once gathered |
| `athenea:motionBuckets` | int | `4` | rt: shutter slices, 1 to 8 |
| `athenea:splatShadows` | bool | `false` | rt: a relit cloud shadows itself |
| `athenea:cloudShadows` | bool | `true` | a cloud's transmittance map at each light |

The rest of the declared group (`athenea:pathAdaptive`, `athenea:pathMis`,
`athenea:cloudShadowTerms` and others) is in §4.2 of the operations document.
Read where they are found, but not declared: `athenea:shutter`,
`athenea:lens` (on a camera), `athenea:exrHalf`, `athenea:disableMotionBlur`,
`athenea:disableDepthOfField` and `athenea:lightGroup` (on a light).

Each `UsdRenderProduct` becomes one EXR whose channels are its render vars. A
var's `sourceName` decides what it reads: `Ci` is the colour, `z` the depth, a
`primvar` source reads a primvar, and a light-path expression of the form
`C.*<L.'NAME'>` reads that light group. `CryptoObject00`, `01` and `02` are
the Cryptomatte layers; a product that holds them also carries the four
standard attributes that name them, including the manifest. A Cryptomatte
layer is always float, whatever `athenea:exrHalf` says.

```usda
def Scope "Render"
{
    def RenderSettings "Settings"
    {
        rel camera = </World/Camera>
        rel products = </Render/Product>
        int2 resolution = (1920, 1080)
        token athenea:technique = "rt"
        int athenea:pathSamples = 8
        int athenea:pathTotal = 512
        bool athenea:denoise = 1
    }

    def RenderProduct "Product"
    {
        token productName = "shot.exr"
        rel orderedVars = [</Render/Vars/beauty>, </Render/Vars/key>, </Render/Vars/CryptoObject00>]
        int2 resolution = (1920, 1080)
    }

    def Scope "Vars"
    {
        def RenderVar "beauty"
        {
            string sourceName = "Ci"
            token dataType = "color3f"
        }
        def RenderVar "key"
        {
            string sourceName = "C.*<L.'key'>"
            token sourceType = "lpe"
        }
        def RenderVar "CryptoObject00"
        {
            string sourceName = "CryptoObject00"
            token dataType = "color4f"
        }
    }
}
```

`athenea stage shot.usda --render-settings /Render/Settings` renders that
prim's products, one EXR per product, and stops. A product with no resolution
or no vars is refused.

## The API schemas a cloud carries

Eight codeless API schemas, applied to a prim like any other. Every attribute
is a primvar, so it inherits down the hierarchy.

| Schema | What it says | Written by |
|---|---|---|
| `AtheneaSplatLightingAPI` | whether a cloud is relit by the scene (`relight`), already holds the light on its body (`litBody`), is in linear light (`linear`); its metallic, roughness, transmission, `ior`, transfer arrays (`transferDirect`, `transferIndirect`, `shadowBits`), `thinWalled`, shading `normal` and `emission` | `athenea mesh2splat`; `ior` by hand for a cloud from elsewhere |
| `AtheneaSplatSkinningAPI` | the joints that carry a cloud | `athenea mesh2splat --skinned` |
| `AtheneaSplatVisibilityAPI` | what a skinned cloud casts, baked by part | `athenea visibility` |
| `AtheneaSplatCryptomatteAPI` | which prim each gaussian came from (`cryptoObject`, `cryptoManifest`) | the conversion |
| `AtheneaSplatEditAPI` | keep, remove or grade the clouds at and below a prim, inside a box or sphere | by hand |
| `AtheneaStreamedAssetAPI` | a cloud drawn from a `.athc` | by hand |
| `AtheneaPointStyleAPI` | how a `UsdGeomPoints` is drawn | by hand |
| `AtheneaVolumeAPI` | how a Volume scatters where no Material is bound | by hand |

`relight` says the colours are an albedo the scene's lights must light;
`litBody` says they are already the light on the material's body, which is
what a baked conversion writes; a cloud carrying the two transfer arrays needs
no `litBody`. `linear` says the colours are linear Rec.709; without it they are
taken for a capture's sRGB.

For a cloud from elsewhere, `primvars:athenea:splat:ior` is what makes a glass
bend the sky (1.5 is glass; at 0 the transmitted half is translucency). An
edit is authored on the cloud's prim or on any prim above it:

```usda
over "Capture" (
    prepend apiSchemas = ["AtheneaSplatLightingAPI", "AtheneaSplatEditAPI"]
)
{
    float primvars:athenea:splat:ior = 1.5 ( interpolation = "constant" )
    bool primvars:athenea:edit:active = 1 ( interpolation = "constant" )
    token primvars:athenea:edit:shape = "box" ( interpolation = "constant" )
    token primvars:athenea:edit:mode = "remove" ( interpolation = "constant" )
    float3 primvars:athenea:edit:centre = (0, 0, 0) ( interpolation = "constant" )
    float3 primvars:athenea:edit:size = (1, 1, 1) ( interpolation = "constant" )
}
```

`centre` is in the cloud's own space and `size` holds the half extents of the
box, or the sphere's radius in x. A grade (`tint`, `saturation`,
`brightness`) works on each splat's colour in linear light.

## What the engine ignores, and what it refuses

The engine **ignores** what it has no meaning for: a primvar it does not read,
a render setting outside the `athenea:` namespace, a schema it does not know.
Nothing is reported, because a stage carries what other renderers need.

It **refuses**, with a message, what it is asked for and cannot do:

| Refused | Message or cause |
|---|---|
| a `.spz` in a build without zstd | `this build reads no .spz` |
| a `.sog` without libwebp, a `.vdb` without OpenVDB | the file is refused when read |
| mesh visibility by rays on a device with no ray tracing | `mesh visibility by rays: the device has no ray tracing` |
| a `.athc` drawn without `--lod` | `a .athc is drawn through its levels of detail: give --lod` |
| a render product with no resolution or no vars | `render product '<path>' has no resolution` / `no vars` |

## Checking a stage in the viewer

`athenea view` draws through the same delegate. `--aov cryptomatte` starts the
window on the Cryptomatte, and `--isolate <prim>` shows that prim's matte
alone, white on black:

```sh
athenea view car_gs.usdc --aov cryptomatte
athenea view car_gs.usdc --isolate /root/Kapoot/Object_57 --frames 10 --snapshot matte.exr
```

**Reference:**
[§1.2 The runtime environment](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#12-the-runtime-environment) ·
[§3.2 Rendering a stage](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#32-rendering-a-stage) ·
[§3.4 Colour](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#34-colour) ·
[§4.1 Pointing an application at the plugin](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#41-pointing-an-application-at-the-plugin) ·
[§4.2 Render settings](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#42-render-settings) ·
[§4.3 The API schemas](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#43-the-api-schemas) ·
[§4.4 What the engine ignores, and what it refuses](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#44-what-the-engine-ignores-and-what-it-refuses) ·
[§5 The viewer](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#5-the-viewer)
