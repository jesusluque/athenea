// Copyright (c) 2026 jesus luque.
//
// A USD stage through the engine's Hydra delegate: UsdImaging's scene indices
// feed a render index whose delegate is this engine, and a task controller
// asks for colour, depth and whatever outputs are requested. Images come back
// to the host (render), or stay on the device for a viewport (draw).
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/render/Camera.h"
#include "athenea/render/SplatQuery.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/usd/PrimData.h"
#include "athenea/usd/RenderSettings.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/technique/DisplayTransform.h"
#include "athenea/technique/SplatVisibility.h"

namespace athenea::gpu {
class Device;
class ShaderLibrary;
}

namespace athenea::usd {

struct StageImage {
    uint32_t           width = 0;
    uint32_t           height = 0;
    std::vector<float> rgba;    ///< bottom row first, linear, premultiplied
    std::vector<float> depth;   ///< view z (distance along the view axis; 0 where nothing was drawn), bottom row first
};

/// A prim of the stage, for a tree view.
struct StagePrim {
    std::string path;
    std::string name;
    std::string type;
    bool        hasChildren = false;
    bool        instance = false;   ///< a native instance: its children are its prototype's
};

/// The children of `prim` in the stage at `stage`, WITHOUT a renderer: no
/// Hydra, no engine, no device -- what a host's tree view asks for as its rows
/// unfold, on whatever thread draws the tree. The same rows
/// `StageRenderer::children` answers. The stage is kept open between calls,
/// keyed on its path and modification time, so unfolding a big stage one row
/// at a time does not open it once a row.
[[nodiscard]] Result<std::vector<StagePrim>> outline(const std::filesystem::path& stage,
                                                     const std::string& prim);

/// Tells USD where this engine's own plugins are -- the codeless schemas
/// (AtheneaSplatEditAPI and the rest) and hdAthenea -- before any stage is opened.
/// What `PXR_PLUGINPATH_NAME` does for a program that sets it; an embedder
/// calls this once at start-up with the directory its build recorded, so a
/// stage that applies one of the schemas reads it as that schema rather than
/// as an unknown token. Safe to call twice.
void registerPlugins(const std::filesystem::path& directory);

/// The stage's cameras by path, in traversal order, the same way: no renderer.
/// What a host offers when it asks which camera to look through.
[[nodiscard]] Result<std::vector<std::string>> stageCameras(const std::filesystem::path& stage);

/// A variant set of the stage, and what it is set to.
///
/// Variant sets are USD's own answer to "pick one of these": a prim carries a
/// named set, the set carries named variants, and a selection says which one
/// composes. The engine has no idea what any of them mean -- an animation, a
/// level of detail, a shirt -- so a host can offer every set a stage brings
/// without knowing the asset.
struct StageVariantSet {
    std::string              prim;       ///< the prim that carries the set
    std::string              name;       ///< the set's name
    std::vector<std::string> variants;   ///< what it may be set to
    std::string              selected;   ///< what it is set to, empty for none
};

/// A DOME LIGHT AND THE SKY IT CARRIES, so that a host can offer to change it.
///
/// A dome is the one light whose whole character is a file: swap the image and
/// the frame is lit by a different room, with nothing else on the stage
/// touched. A host that lists these can let someone try one sky after another
/// against the same asset, which is the only way to see what a relightable
/// cloud actually buys.
struct StageDome {
    std::string prim;       ///< its path, which is how it is set
    std::string name;       ///< its prim name, for a panel
    std::string texture;    ///< the image it carries, resolved; empty for a dome that is only a colour
    float       rotation = 0.0F;   ///< degrees about the up axis, as `setDomeRotation` writes them
};

/// A LIGHT ON THE STAGE, AND WHETHER IT IS LIGHTING THE FRAME.
///
/// A stage brings its own lights, and a host that swaps the sky needs to be
/// able to take the rest away: a fixed sun left on under every image is a
/// highlight that no sky explains.
struct StageLight {
    std::string prim;       ///< its path, which is how it is switched
    std::string name;       ///< its prim name, for a panel
    std::string type;       ///< its schema, `DistantLight`, `DomeLight_1`, ...
    bool        on = true;  ///< false where the prim is inactive
};

/// What a pixel saw.
struct StagePick {
    std::string rprim;          ///< Hydra's path (a prototype's, under instancing)
    std::string prim;           ///< the USD prim it came from
    int32_t     instance = -1;  ///< which instance of it
    /// WHAT THE MATTE CALLS WHAT IS THERE, where the frame kept one: the id
    /// of the nearest rank at that pixel and, from the frame's manifest, its
    /// name. This is how a cloud is picked at all -- a gaussian writes no
    /// `primId`, so a pixel of splats has nothing else to say who it is.
    uint32_t    cryptoId = 0;
    std::string cryptoName;
    float       cryptoCoverage = 0.0F;
};

/// What `bakeVisibility` hands back (Engine::BakedVisibility, without the
/// delegate's types).
struct BakedVisibilityArrays {
    std::vector<float>   parts;    ///< 12 floats a part
    std::vector<int32_t> texels;   ///< two f16 a word
    std::vector<int32_t> partOf;   ///< the part of each gaussian
    std::vector<int32_t> ambient;  ///< a probe's mean, for domes
    uint32_t             partCount = 0;
};

class StageRenderer {
public:
    /// `device`: a device the host already drives, shared rather than a
    /// second one opened beside it. Null opens one, as before. Every call on
    /// the renderer then belongs to the one thread that host drives it from.
    [[nodiscard]] static Result<std::unique_ptr<StageRenderer>> open(
        const std::filesystem::path& stage, std::shared_ptr<gpu::Device> device = nullptr);

    /// Hide `prim`, and so everything under it, or give it back the
    /// visibility the stage authored. A session opinion, like a variant
    /// selection or a dome's sky: nothing on disk changes, and "shown" is not
    /// a forced `inherited` but no opinion at all, so a prim the stage itself
    /// made invisible stays so when a host stops hiding it.
    [[nodiscard]] Result<void> setPrimVisible(const std::string& prim, bool visible);

    /// Point a shader's asset input -- a UsdUVTexture's or a MaterialX
    /// image's `file` -- at another texture, as a session opinion. Empty
    /// clears the opinion: the texture the stage authored comes back. The
    /// input's other metadata (its colour space) stays the stage's.
    [[nodiscard]] Result<void> setShaderTexture(const std::string& shader, const std::string& input,
                                                const std::string& texture);

    /// Fill the texture a material names `aofx://<name>` from the host's
    /// device buffer: float4 texels, `rowPixels` a row, bottom row first.
    /// Point a shader at it with `setShaderTexture(shader, input,
    /// "aofx://slot1")`; call this every frame the picture changes, before
    /// drawing. The texels are the host's as they are (raw), and a copy on
    /// the device: nothing crosses to the CPU.
    [[nodiscard]] Result<void> updateExternalTexture(const std::string& name, const gpu::Buffer& rgba,
                                                     uint32_t width, uint32_t height, uint32_t rowPixels);

    /// Plugin directories USD is to read, beside the ones it found itself:
    /// its own plugins and the engine's schemas, where a process keeps them
    /// somewhere USD would not look. Returns how many plugins that added.
    ///
    /// PXR_PLUGINPATH_NAME is no substitute inside a process: USD reads it in
    /// a static constructor (Plug_InitConfig, base/plug/initConfig.cpp), when
    /// the library is loaded -- before main, and so before any code of an app
    /// can set it. An iOS app, whose plugins are in its bundle and whose
    /// environment nobody sets before launch, finds none, and the first stage
    /// it opens stops in a fatal error asking for the asset resolver.
    static size_t registerPlugins(const std::vector<std::filesystem::path>& directories);
    ~StageRenderer();

    /// `camera` is a UsdGeomCamera prim path; empty takes the first camera.
    /// `technique` is the delegate's `athenea:technique` setting: "raster" or "rt".
    /// A path traced image is drawn until it holds `setPathTotal` paths.
    [[nodiscard]] Result<StageImage> render(const std::string& camera, double time,
                                            uint32_t width, uint32_t height,
                                            const std::string& technique = "raster");

    /// The same, from a camera that is not on the stage (the engine's
    /// convention: looking down its own -Z, as USD's cameras do).
    [[nodiscard]] Result<StageImage> render(const render::Camera& camera, double time, uint32_t width,
                                            uint32_t height, const std::string& technique = "raster");

    /// The same frames for a viewport: drawn and left on the device, nothing
    /// read back, streamed assets filling in over the frames that follow.
    [[nodiscard]] Result<void> draw(const std::string& camera, double time, uint32_t width, uint32_t height,
                                    const std::string& technique = "raster");
    [[nodiscard]] Result<void> draw(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                    const std::string& technique = "raster");

    /// An image left on the device: what `render` draws -- the streams
    /// settled, a path traced frame gathered to its total -- with no
    /// readback. For a host that carries the frame on from `displaySource`
    /// itself, a compositor whose every frame must be whole.
    [[nodiscard]] Result<void> drawImage(const std::string& camera, double time, uint32_t width,
                                         uint32_t height, const std::string& technique = "raster");
    [[nodiscard]] Result<void> drawImage(const render::Camera& camera, double time, uint32_t width,
                                         uint32_t height, const std::string& technique = "raster");

    /// The device the engine draws on: a window's surface is made on it.
    [[nodiscard]] gpu::Device& device();
    /// The engine's shader library on that device, for kernels drawn beside it.
    [[nodiscard]] gpu::ShaderLibrary& library();

    /// The last frame's `aov` ("color", "depth", "primId", "instanceId",
    /// "elementId", "Neye", "normal") as DisplayTransform reads it. An AOV no
    /// mesh drew, or one not requested, has no buffer: it shows as the background.
    [[nodiscard]] Result<technique::DisplaySource> displaySource(const std::string& aov);

    /// A camera of the engine's own framing what the stage draws, as athenea view
    /// opens on a stage: a small frame commits the scene (raster, whatever
    /// `technique` is), the bounds of what it drew place an orbit camera of
    /// `focal` mm. For a stage without cameras.
    [[nodiscard]] Result<render::Camera> framingCamera(double time, double focal = 35.0,
                                                       const std::string& technique = "raster");

    /// Where what the last frame drew is, in world space.
    [[nodiscard]] Result<std::optional<scene::Bounds>> bounds();

    /// The prim under pixel (x, y) of the last frame, from the top left; nothing
    /// where no mesh drew. "primId" must be among the requested outputs.
    [[nodiscard]] Result<std::optional<StagePick>> pick(uint32_t x, uint32_t y);

    /// The children of the prim at `path` ("/" for the stage's root).
    [[nodiscard]] std::vector<StagePrim> children(const std::string& path) const;

    /// 'Y' or 'Z'.
    [[nodiscard]] char upAxis() const;
    [[nodiscard]] double endTimeCode() const;

    /// How renders find what meshes a pixel sees: "automatic" (the default:
    /// rays where the device has ray queries, else raster, else compute BVHs),
    /// "raster", "rays" or "bvh". The delegate's `athenea:visibility` setting.
    [[nodiscard]] Result<void> setMeshVisibility(const std::string& route);

    /// Whether the stage authors a UsdLux light of its own.
    [[nodiscard]] bool hasLights() const;
    /// A sky dome and a sun in the stage's session layer, or not. For a stage
    /// that authors no lights: without them the path tracer lights the first
    /// hit from the eye, as the raster does, and has nothing to bounce. The
    /// file is not touched; the lights live at `/atheneaDefaultLights` and reach
    /// the renderer through Hydra as authored ones do.
    [[nodiscard]] Result<void> setDefaultLights(bool on);

    /// The light this stage's meshes carry, at points somebody names.
    ///
    /// `rays` holds two `float4` a point -- where its ray starts and how near
    /// it may hit, then which way it goes -- and what comes back, one `float4`
    /// a point, is the radiance leaving the surface that ray finds, path
    /// traced with this stage's own lights, shadows and bounces. The stage is
    /// synced first, by drawing one pixel of it, because the meshes and their
    /// materials have to be on the device before anything can be asked of
    /// them.
    ///
    /// What it is for: `athenea mesh2splat`, which turns a mesh into gaussians
    /// that carry the light the mesh had.
    /// `degree` asks for spherical harmonics rather than a colour: 0 gives one
    /// `float4` a point, and 1, 2 or 3 give (degree + 1)^2 of them, the
    /// constant term first and the rest in `evaluateRest`'s order. What a
    /// single colour cannot hold is the direction the light leaves in, which
    /// is what makes a converted mesh look like polished plastic; the
    /// harmonics are what a trained cloud carries for the same reason.
    /// `transfer` measures how much of an environment reaches each point --
    /// visibility times the cosine, the surface's albedo taken as one --
    /// rather than the light leaving it. What comes back is then one entry a
    /// coefficient whose rgb is the indirect half and whose alpha is the
    /// direct one, and one entry more carrying the coverage.
    /// `facing`, four floats a point where given, is the way a point faces
    /// when that is not the surface's, w 1 where it says so and 0 where the
    /// surface's stands: a displaced gaussian stands off the flat mesh and is
    /// lit, and projected, as the relief faces.
    [[nodiscard]] Result<std::vector<float>> bakePoints(const std::vector<float>& rays, uint32_t count,
                                                        double time, uint32_t samples = 64,
                                                        uint32_t bounces = 3, uint32_t degree = 0,
                                                        bool transfer = false,
                                                        const std::vector<float>* facing = nullptr);

    /// Samples per light per pixel: one for an interactive frame, more where
    /// an area light's noise would be read as error.
    void setLightSamples(uint32_t samples);

    /// One light per sample, chosen by power, rather than every light at every
    /// pixel.
    void setChooseLights(bool choose);
    /// `athenea:splatShadows`: a relit cloud shadows itself, one ray a splat.
    void setSplatShadows(bool shadows);
    /// Whether a transferred cloud adds the indirect half it carries.
    void setSplatTransferIndirect(bool indirect);
    /// Whether a gaussian reflects the cloud it belongs to.
    void setSplatReflections(bool reflect);
    /// `athenea:cloudShadows`: the frame's clouds shadow its meshes, measured
    /// from each light into a map and read with no ray. On by default.
    void setCloudShadows(bool shadows);
    /// `athenea:cloudShadowResolution`: texels a side of that map, per light.
    void setCloudShadowResolution(uint32_t texels);
    /// `athenea:cloudShadowTerms`: 1 is the total optical depth (exact for a
    /// receiver behind the cloud, a floor); 3, 5, 7 add the Fourier pairs a
    /// receiver inside the cloud needs -- a cloud shadowing itself. 0 is
    /// automatic: 1 for meshes alone, 5 where a relit cloud receives.
    void setCloudShadowTerms(uint32_t terms);
    /// `athenea:cloudShadowDensity`: what the map's optical depth is multiplied
    /// by. 1 is what the cloud's opacity says; less lets light through it.
    void setCloudShadowDensity(float density);
    /// `athenea:antialias`: a path traced frame moves the camera inside the pixel
    /// between the passes it gathers, so its edges are averaged over the
    /// pixel rather than sampled at its middle. On by default; off is what
    /// two frames compared pixel by pixel need.
    void setAntialias(bool on);

    /// Bakes the per-part visibility of the ParticleField at `prim` (a cloud
    /// a skeleton carries), commits the stage first so it is on the device,
    /// and hands back the two arrays the file will carry.
    [[nodiscard]] Result<BakedVisibilityArrays> bakeVisibility(const std::string& prim,
                                                               const technique::VisibilityParts& parts,
                                                               const technique::VisibilityBakeOptions& options,
                                                               double time = 0.0);

    /// The path traced technique ("rt" over meshes): paths a pixel each pass
    /// gathers, bounces after the first hit, and the paths a pixel at which
    /// the frame is finished. A total of one -- the default -- never
    /// accumulates, which is what a moving camera wants.
    void setPathSamples(uint32_t samples);
    void setPathBounces(uint32_t bounces);
    /// Motion blur's shutter slices for `rt`, 1 to 8; the shutter itself is
    /// the camera's.
    void setMotionBuckets(uint32_t buckets);
    /// The shutter for a camera that authors none: `athenea stage --eye` and the
    /// viewer's free camera. A stage camera's own is taken when this is not
    /// set. In the raster route it is a cloud's motion blur; in the traced
    /// one it is that and the shutter slices both.
    void setShutter(double open, double close);
    /// Subdivision surfaces refined this many levels (0: the control mesh),
    /// as usdview's complexity sets it.
    void setRefineLevel(uint32_t level);
    void setPathTotal(uint32_t total);
    /// Denoise a path traced frame once it has gathered its total.
    void setDenoise(bool denoise);
    /// Adaptive sampling, and the relative error a pixel stops at.
    void setPathAdaptive(bool adaptive);
    /// Weigh light sampling and material sampling by the power heuristic
    /// (the default), or light a surface by light sampling alone.
    void setPathMis(bool mis);
    void setPathError(float error);

    /// How many paths a pixel the frame on the device holds, and whether it
    /// holds all it is going to. A frame that is not path traced has nothing
    /// to gather and reads as finished.
    [[nodiscard]] uint32_t pathAccumulated() const;
    [[nodiscard]] bool pathConverged() const;
    /// The mesh pools' generation and positions revision (Engine's): a
    /// deformation raises the second and not the first.
    [[nodiscard]] uint64_t meshGeneration() const;
    /// The coordinate systems bound to a mesh prim (UsdShadeCoordSysAPI):
    /// names and transforms, as the delegate resolved them.
    [[nodiscard]] std::vector<CoordSysBinding> coordSysBindings(const std::string& prim) const;
    [[nodiscard]] uint64_t meshPositionsRevision() const;

    /// The Hydra outputs renders produce, colour and depth always among them
    /// ("primId", "instanceId", "elementId", "Neye", "normal", "primvars:st"...).
    void requestOutputs(const std::vector<std::string>& aovs);

    /// The last render's Hydra render buffer for `aov` ("color", "depth"), as
    /// a host mapping it reads it: the buffer's own format, bottom row first
    /// (Hydra's layout, Storm's and hdEmbree's).
    [[nodiscard]] Result<std::vector<uint8_t>> mappedOutput(const std::string& aov);

    /// What the ids of the last frame's Cryptomatte are called: path -> id, in
    /// the order a manifest lists them. Empty until a frame kept a matte.
    [[nodiscard]] std::map<std::string, uint32_t> cryptoManifest() const;
    /// How many times a splat cloud's arrays have been uploaded: once a cloud
    /// unless what it holds changes. What a time step is held to.
    [[nodiscard]] uint64_t cloudUploads() const;

    /// WHAT THE LAST FRAME HELD: splats handed over, splats a tile kept, the
    /// pairs the blend walked, the scene's instances and the frame's lights.
    /// Counts the frame already had; nothing is measured to answer this.
    struct Counters {
        uint32_t splats = 0;
        uint32_t visibleSplats = 0;
        uint32_t splatPairs = 0;
        uint32_t meshInstances = 0;
        uint32_t lights = 0;
        bool     cryptomatte = false;
    };
    [[nodiscard]] Counters counters() const;

    /// Every camera prim on the stage.
    [[nodiscard]] std::vector<std::string> cameras() const;

    /// Every variant set on the stage, in prim order, with its selection.
    [[nodiscard]] std::vector<StageVariantSet> variantSets() const;
    /// Set one of them, so that the variant's opinions compose from now on.
    ///
    /// The selection is written to the stage's session layer, so the file on
    /// disk is not touched; what is drawn changes on the next frame, because
    /// a selection is a composition change and Hydra is told to resync the
    /// prims under it.
    [[nodiscard]] Result<void> setVariantSelection(const std::string& prim, const std::string& set,
                                                   const std::string& variant);
    /// The same, spelled as USD spells a variant selection inside a path:
    /// `/World{clip=air_fly_A0}`.
    [[nodiscard]] Result<void> setVariantSelection(const std::string& variantPath);

    /// WHAT A PICKED PRIM'S GAUSSIANS ARE MADE OF, counted on the device over
    /// every cloud the scene holds. A `count` of 0 means no gaussian carries
    /// that id.
    [[nodiscard]] Result<render::SplatIdReading> measureSplatId(uint32_t cryptoId);
    /// What the frame says that prim is made of, over what its gaussians
    /// carry. A negative metallic, roughness or transmission leaves the
    /// gaussian's own, so a tint does not flatten a material that came out of
    /// a map. Read by the shading, never baked: nothing on disk is touched.
    void setSplatOverride(const render::SplatOverride& said);
    void clearSplatOverride(uint32_t cryptoId);
    void clearSplatOverrides();
    [[nodiscard]] std::vector<render::SplatOverride> splatOverrides() const;

    /// Every dome light on the stage, in prim order, with the image it carries.
    [[nodiscard]] std::vector<StageDome> domes() const;
    /// Put `texture` on that dome. Empty puts back whatever the stage
    /// authored, since the opinion is written to the session layer and
    /// clearing it is how a session opinion goes away.
    ///
    /// The prepared environment is keyed on the dome's record, so the sky's
    /// harmonics and its prefiltered chain are rebuilt on the next frame and
    /// not before: changing this costs one frame, not every frame.
    [[nodiscard]] Result<void> setDomeTexture(const std::string& prim, const std::string& texture);
    /// Turn that dome `degrees` about the stage's up axis. An environment is
    /// only right once it is turned, and turning it is also the quickest way
    /// to see that a reflection is reading the sky and not a constant.
    [[nodiscard]] Result<void> setDomeRotation(const std::string& prim, float degrees);

    /// Every light the stage authored, in prim order, those switched off
    /// included; the default lights a host adds are not the stage's.
    [[nodiscard]] std::vector<StageLight> lights() const;
    /// Switch that light on or off. Off deactivates the prim in the session
    /// layer, so Hydra removes the light as if it had never been authored;
    /// on clears that opinion, and asserts it only where the stage itself
    /// authored the light inactive. A dome switched off takes its sky with
    /// it, background included.
    [[nodiscard]] Result<void> setLightOn(const std::string& prim, bool on);

    /// The interval the stage's animation occupies: the first and last time
    /// code any authored attribute has a sample at, narrowed to the stage's
    /// declared start and end where it declares them. A stage with nothing
    /// animated gets the declared interval, and one that declares nothing
    /// either gets {0, 0}.
    ///
    /// It is what a timeline should show: the declared interval belongs to the
    /// root layer, so a stage that composes one animation of thirty frames out
    /// of seventy in a variant set still declares the longest of them.
    [[nodiscard]] std::pair<double, double> animationRange() const;

    /// A UsdRender settings prim, as Hydra's renderSettings bprim holds it
    /// once it is made the scene's active one and synced: its products and
    /// vars, purposes, colour space and `athenea:` settings.
    [[nodiscard]] Result<RenderSettingsInfo> renderSettings(const std::string& path);
    /// Renders every product of that settings prim -- each at its own
    /// resolution from its own camera, its vars as the layers of one OpenEXR
    /// written where `productName` says (relative to `directory`), 32-bit
    /// floats unless `athenea:exrHalf` is set -- with `includedPurposes` as the
    /// render tags and its `athenea:` settings applied. Returns the files written.
    [[nodiscard]] Result<std::vector<std::filesystem::path>> renderProducts(const std::string& path, double time,
                                                                            const std::filesystem::path& directory = {});
    /// The purposes the next renders draw ("default", "render", "proxy",
    /// "guide"): Hydra's render tags. Empty: default and render.
    void setIncludedPurposes(const std::vector<std::string>& purposes);
    /// The material binding purposes the next renders resolve, in order, as
    /// a settings prim's `materialBindingPurposes` lists them ("full",
    /// "preview", "" for the all-purpose binding). Empty: "full", then "".
    /// The first named purpose is the one looked for, the all-purpose
    /// binding the fallback: a second named purpose is not consulted.
    void setMaterialBindingPurposes(const std::vector<std::string>& purposes);

    /// The stage's timeCodesPerSecond and startTimeCode: how a frame on a
    /// clock maps to a USD time.
    [[nodiscard]] double timeCodesPerSecond() const;
    [[nodiscard]] double startTimeCode() const;

private:
    [[nodiscard]] Result<void> executeUntilGathered(uint32_t width, uint32_t height);
    StageRenderer();
    [[nodiscard]] Result<void> aim(const std::string& camera, double time, const std::string& technique);
    [[nodiscard]] Result<void> aim(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                   const std::string& technique);
    [[nodiscard]] Result<void> execute(uint32_t width, uint32_t height);
    [[nodiscard]] Result<StageImage> readImage(uint32_t width, uint32_t height);
    [[nodiscard]] const render::RenderTargets* lastTargets() const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace athenea::usd
