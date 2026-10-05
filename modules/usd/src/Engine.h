// Copyright (c) 2026 jesus luque.
//
// What a Hydra render delegate holds: the device, the renderers, and the
// flat scene the prims sync into.
//
// THE SCENE, AND WHO MAY TOUCH IT
//
// Hydra syncs prims on worker threads. A prim's Sync only arranges its arrays
// into CPU records and hands them here under the lock; nothing touches the
// device. The render pass then, on the thread that executes it, uploads
// whatever changed (the GPU decode) and renders. So the device has one caller,
// which is the rule openFXplayer's gpu_host lives by, and the scene the render
// reads is a committed one.
//
// Entries are keyed by prim path and live in slots with generations, so a
// handle to a destroyed prim is a stale handle and never somebody else's cloud.
#pragma once

#include <array>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <atomic>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/path.h>

#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/technique/Environment.h"
#include "athenea/usd/GaussianStats.h"
#include "athenea/usd/PrimData.h"
#include "athenea/geom/Curves.h"
#include "athenea/geom/Mesh.h"
#include "athenea/geom/Skinner.h"
#include "athenea/geom/Subdivision.h"
#include "athenea/lod/Lod.h"
#include "athenea/technique/Visibility.h"
#include "athenea/technique/SplatVisibility.h"
#include "athenea/io/Vdb.h"
#include "athenea/world/VolumeSet.h"
#include "athenea/scene/SplatSkinner.h"
#include "athenea/world/GpuScene.h"
#include "athenea/world/Instancing.h"
#include "athenea/lod/Athc.h"
#include "athenea/material/MaterialCompiler.h"
#include "athenea/material/TextureStore.h"
#include "athenea/io/Ies.h"
#include "athenea/light/LightTable.h"
#include "athenea/technique/MaterialShading.h"
#include "athenea/technique/Denoiser.h"
#include "athenea/technique/PathTracer.h"
#include "athenea/technique/SplatShadowMap.h"
#include "athenea/technique/SplatShadows.h"
#include "athenea/technique/EmissiveTable.h"
#include "athenea/render/GaussianRayTracer.h"
#include "athenea/render/PointRasterizer.h"
#include "athenea/render/SplatQuery.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::usd {

/// AtheneaStreamedAssetAPI: a .athc drawn with its levels of detail.
struct StreamedAsset {
    std::string path;              ///< resolved; empty when the prim has none
    float       threshold = 1.0F;  ///< px a merged cell may span
    uint64_t    budget = 0;        ///< splats on the device; 0 reads the file whole

    bool operator==(const StreamedAsset&) const = default;
};

/// WHAT A CLOUD WAS UPLOADED FROM, by the identity of the arrays USD handed
/// over.
///
/// `VtArray` is copy-on-write, so a `Get` at a new time of an attribute that
/// has no time samples returns the same buffer: where the data is and how much
/// of it there is, together, say whether anything the decode depends on
/// actually changed. `skinningXforms` is deliberately not among them -- it is
/// the one array that does change every frame, and no decode depends on it.
struct CloudIdentity {
    std::array<const void*, 17> data{};
    std::array<size_t, 17>      bytes{};
    int                         shDegree = -1;

    [[nodiscard]] bool operator==(const CloudIdentity& other) const noexcept {
        return data == other.data && bytes == other.bytes && shDegree == other.shDegree;
    }
};

struct SplatEntry {
    std::optional<ParticleFieldArrays>  pending;   ///< synced, not yet uploaded
    CloudIdentity                       uploaded;  ///< what `gpu` was decoded from
    std::unique_ptr<scene::GpuSplats>   gpu;
    render::Mat4                        objectToWorld = render::Mat4::identity();
    /// WHAT THE SHUTTER MOVES IT BY: the transform at the shutter's close
    /// less the one at its open, element by element, already scaled to the
    /// shutter (Hydra brackets it). All zero where nothing moves.
    std::array<double, 16>              transformStep{};
    bool                                visible = true;
    render::SplatEdit                   edit;
    /// AtheneaSplatLightingAPI: relit by the scene's lights rather than shown as
    /// it was baked.
    bool                                relight = false;
    /// `primvars:athenea:splat:litBody`: its colours are light already, so what
    /// relighting adds is the polish alone (`athenea mesh2splat --bake`).
    bool                                litBody = false;
    /// `primvars:athenea:splat:catcher`: a shadow catcher, drawn black as
    /// opaque as what its object took (`athenea mesh2splat --shadow-catcher`).
    bool                                catcher = false;
    /// `primvars:athenea:splat:ior`: the index its transmitting gaussians bend
    /// the sky by. 0 bends nothing, which is every cloud that does not say.
    float                               ior = 0.0F;
    std::vector<pxr::TfToken>           categories;   ///< what a light's link is tested against
    /// What this cloud's per-gaussian Cryptomatte ids are called: the manifest
    /// `primvars:athenea:splat:cryptoManifest` carries, path -> id. Empty for a
    /// capture, whose gaussians carry no ids.
    std::map<std::string, uint32_t>     cryptoManifest;
    /// Its level of detail, where it is one of several (ParticleFieldArrays).
    std::string                         lodGroup;
    float                               lodCell = 0.0F;
    float                               lodThreshold = 1.0F;
    /// A level's pose, kept rather than applied: only the level a view draws
    /// is posed (Engine::render), since posing the others costs the frame
    /// their whole cloud for nothing it shows.
    std::optional<ParticleFieldArrays>  deferredPose;
    bool                                deferredReupload = false;
    std::optional<StreamedAsset>        assetPending;
    StreamedAsset                       asset;
    std::unique_ptr<lod::LodCloud>      lodCloud;   ///< the asset read whole
    std::unique_ptr<lod::StreamingPool> pool;       ///< or streamed
    /// AtheneaSplatSkinningAPI. The cloud on the device is the bind pose; `posed`
    /// is the same cloud with the skeleton's transforms in it, and is what a
    /// frame draws. Its buffers outlive the frame, so the cloud a renderer
    /// holds never changes identity between them.
    std::unique_ptr<scene::GpuSplats>   posed;
    gpu::Buffer                         influences;   ///< float2 (joint, weight), `perSplat` a gaussian
    uint32_t                            perSplat = 4; ///< influences a gaussian (SkelBindingAPI's elementSize)
    /// The weights' gradients across each gaussian, `perSplat - 1` words of
    /// two halves a kept gaussian; empty for a cloud converted without them.
    gpu::Buffer                         weightGradients;
    gpu::Buffer                         xforms;       ///< four float4 a joint, this frame's
    /// Under a shutter: the same joints at its other end, and what the
    /// skinner made of the two -- a displacement a gaussian, in the cloud's
    /// own space. `motionScale` is the shutter over the span the two poses
    /// were sampled across, since Hydra brackets rather than clips.
    gpu::Buffer                         xformsEnd;
    gpu::Buffer                         motion;
    float                               motionScale = 0.0F;
    uint32_t                            joints = 0;
    std::array<float, 16>               geomBind{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                                 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
};

struct InstancerEntry {
    InstancerArrays  arrays;
    pxr::SdfPath     parent;
    uint64_t         version = 0;
};

struct MeshEntry {
    std::optional<MeshArrays>              pending;
    std::optional<CurveArrays>             pendingCurves;   ///< a BasisCurves prim: built as a tube mesh
    std::vector<InstancerLink>             instancing;      ///< innermost first; empty: not instanced
    world::InstanceChain                   chain;           ///< composed from `instancing`
    /// Under a shutter, when an instancer in the chain moves: the chain at the
    /// samples bracketing it, and when those are (the first moving level's).
    world::InstanceChain                   chainStart;
    world::InstanceChain                   chainEnd;
    double                                 chainTimeStart = 0.0;
    double                                 chainTimeEnd = 0.0;
    std::vector<uint64_t>                  chainVersions;   ///< the instancer versions `chain` was made from
    bool                                   chainDirty = false;
    std::shared_ptr<const geom::GpuMesh>   gpu;
    uint64_t                               topologyKey = 0;   ///< the key its GpuMesh was built with
    render::Mat4                           objectToWorld = render::Mat4::identity();
    /// Motion blur: the prim at the shutter's open and close, where they
    /// differ from the frame -- transforms, and meshes built from the points
    /// there under the same topology key.
    MeshTransforms                         shutter;
    pxr::VtValue                           pendingStart, pendingEnd;   ///< points to build gpuStart/gpuEnd from
    double                                 pointsTimeStart = 0.0, pointsTimeEnd = 0.0;
    std::shared_ptr<const geom::GpuMesh>   gpuStart, gpuEnd;
    MeshLook                               look;
    std::vector<pxr::SdfPath>              subsetMaterials;   ///< per GeomSubset the mesh was built with
    uint32_t                               primId = 0;
    pxr::TfToken                           renderTag;
    bool                                   visible = true;
};

struct MaterialEntry {
    std::shared_ptr<void>                   document;   ///< MaterialX::DocumentPtr; null: nothing MaterialX reads
    bool                                    pending = true;
    bool                                    cutout = false;        ///< its opacity cuts samples away: visibility evaluates it
    bool                                    transparent = false;   ///< opacityMode transparent: the tracer keeps the specular
    std::optional<material::Transmission>   transmission;        ///< lets light through, and how much: the bake's open-or-not rays look
    std::optional<material::CompiledMaterial> compiled;
    /// Where it cuts: its opacity alone, what a shadow ray asks at a candidate.
    std::optional<material::CompiledMaterial> opacity;
    /// Its `volume` terminal, for a Volume that binds it.
    std::optional<material::VolumeCoefficients> volume;
};

struct PointsEntry {
    std::optional<PointsArrays>         pending;
    std::unique_ptr<scene::GpuPoints>   gpu;
    render::Mat4                        objectToWorld = render::Mat4::identity();
    render::PointStyle                  style;
    bool                                visible = true;
};

/// The layout of a Hydra render buffer's pixels, for `Engine::writeAov`.
struct AovLayout {
    uint32_t channels = 4;
    uint32_t componentBytes = 4;
    uint32_t componentKind = 2;   ///< 0 unorm8, 1 float16, 2 float32
};

/// Which image an AOV reads.
enum class AovKind {
    Colour, Depth, PrimId, InstanceId, ElementId, EyeNormal, WorldNormal, Primvar, Albedo, ShadingNormal, LightGroup,
    /// A Cryptomatte layer: `AovSource::primvar` is which of them (0, 1, 2),
    /// each a float4 of two (id, coverage) ranks.
    Crypto
};

struct AovSource {
    AovKind  kind = AovKind::Colour;
    uint32_t primvar = 0;   ///< AovKind::Primvar: its index in AovRequest::primvars; LightGroup: in lightGroups
};

/// Where an AOV lives on the device, as shaders/athenea/usd/aov_convert.slang
/// reads it.
struct AovView {
    const gpu::Buffer* buffer = nullptr;   ///< null: nothing drew it, so its clear value
    uint32_t           source = 0;         ///< 0 float4 (colour, normals, primvars), 1 view z, 2 uint ids
    uint32_t           stride = 1;         ///< entries per pixel
    uint32_t           offset = 0;         ///< the AOV's entry within them
    bool               ids = false;        ///< cleared to -1, not 0
};

/// What a frame should compute beyond colour and depth.
/// Points to bake the frame's light at (`Engine::bakePoints`).
struct BakeRequest {
    /// Two `float4` a point: where its ray starts and how near it may hit,
    /// then which way it goes. The caller writes these; nothing here invents
    /// a direction for a point.
    const gpu::Buffer* rays = nullptr;
    uint32_t           count = 0;
    uint32_t           samples = 64;
    uint32_t           bounces = 3;
    /// How many spherical harmonics to fit: 1 is a colour alone, 4, 9 and 16
    /// are degrees 1 to 3 (technique::BakePoints).
    uint32_t           coefficients = 1;
    /// MEASURE TRANSFER RATHER THAN RADIANCE (technique::BakePoints): what
    /// reaches each point from an environment, with its own albedo taken as
    /// one, so the answer does not depend on the light it was baked under.
    bool               transfer = false;
    /// And its open directions on a finer grid over the whole sphere, 16 or
    /// 32 cells a side (technique::BakePoints::cellSide); 0 the first 8 x 8.
    uint32_t           cellSide = 0;
    /// KEEP THE SUMS, DIRECT APART FROM INDIRECT (technique::BakePoints::split).
    bool               split = false;
    /// AND READ THE MATERIAL BACK at each point (technique::BakePoints::material).
    bool               material = false;
    /// Which paths these are: a pass that adds to an earlier one draws
    /// others. The same seed draws the same paths.
    uint32_t           seed = 0;
    /// Where the answer lands: one `float4` a point, the radiance in rgb.
    render::RenderTargets* out = nullptr;
};

struct AovRequest {
    bool                     ids = false;       ///< primId, instanceId, elementId
    bool                     normals = false;   ///< Neye, normal
    /// albedo, shadingNormal: the path tracer's guides, which cost it a
    /// buffer -- and on Metal a buffer is what decides whether a cloud can
    /// shadow a mesh in the same frame (technique::PathTracer).
    bool                     aux = false;
    std::vector<std::string> primvars;          ///< "primvars:NAME" outputs, by NAME
    /// "lightGroup:NAME" outputs, by NAME: each light's direct contribution
    /// under its group (`athenea:lightGroup`). At most technique::kMaxLightGroups.
    std::vector<std::string> lightGroups;
    /// CryptoObject00..02: which prims covered each pixel and by how much.
    /// Meshes are named by their path; a cloud is named per gaussian where it
    /// was converted from prims that have paths (`athenea mesh2splat`), and left
    /// out where it was captured.
    bool                     cryptomatte = false;
};

/// How the engine draws a frame.
enum class Technique {
    Raster,     ///< tile rasteriser, points composited
    /// Rays. A frame of nothing but splats is GaussianRayTracer's, whole.
    /// With meshes in it the surfaces are path traced and the splats
    /// composed over them by the rasteriser, since the tracer takes no
    /// `under` layer: splats inside the rays is still to be written.
    RayTraced,
};

/// Which route finds what meshes a pixel sees. All three fill the same
/// visibility targets with the same ids (tests/technique/test_visibility.cpp).
enum class MeshVisibility {
    Automatic,  ///< rays where the device has ray queries, else raster, else the compute BVH
    Raster,     ///< VisibilityRaster
    Rays,       ///< VisibilityTrace: the device's acceleration structures
    Bvh,        ///< VisibilityBvh: compute BVHs, for devices with neither
};

/// WHAT THE LAST FRAME HELD, as the frame itself counted it: no readback and
/// nothing measured for this, only the numbers the passes already had.
struct FrameCounters {
    uint32_t splats = 0;          ///< gaussians handed to the rasteriser
    uint32_t visibleSplats = 0;   ///< of those, the ones a tile kept
    uint32_t splatPairs = 0;      ///< (tile, splat) pairs the blend walked
    uint32_t meshInstances = 0;   ///< instance records in the scene
    uint32_t lights = 0;          ///< lights in the frame's table
    bool     cryptomatte = false; ///< the frame kept a matte
};

class Engine {
public:
    /// Null with a reason when there is no device; Hydra then gets nothing drawn.
    static std::unique_ptr<Engine> create(std::string& why);
    /// The same on a device somebody else opened -- a host that already
    /// drives the GPU and shares it with another runtime (the compositor this
    /// engine is built into: its gpe adopts this very device). Null opens one,
    /// which is what `create(why)` does. Every call still comes from one
    /// thread; the host's thread is that thread.
    static std::unique_ptr<Engine> create(std::shared_ptr<gpu::Device> device, std::string& why);

    // --- from Sync (any thread) ---
    void setSplats(const pxr::SdfPath& id, std::optional<ParticleFieldArrays> raw,
                   const render::Mat4* transform, std::optional<bool> visible,
                   std::optional<render::SplatEdit> edit = std::nullopt,
                   std::optional<StreamedAsset> asset = std::nullopt,
                   std::optional<bool> relight = std::nullopt,
                   std::optional<std::vector<pxr::TfToken>> categories = std::nullopt,
                   std::optional<bool> litBody = std::nullopt,
                   std::optional<float> ior = std::nullopt,
                   std::optional<render::Mat4> transformStep = std::nullopt,
                   std::optional<bool> catcher = std::nullopt);
    void setPoints(const pxr::SdfPath& id, std::optional<PointsArrays> raw,
                   const render::Mat4* transform, std::optional<bool> visible,
                   std::optional<render::PointStyle> style);
    /// A BasisCurves prim: a mesh entry whose GpuMesh is a tube over its spans.
    void setCurves(const pxr::SdfPath& id, int32_t primId, const pxr::TfToken& renderTag,
                   std::optional<CurveArrays> arrays, const render::Mat4* transform, std::optional<bool> visible,
                   std::optional<MeshLook> look);
    void setMesh(const pxr::SdfPath& id, int32_t primId, const pxr::TfToken& renderTag,
                 std::optional<MeshArrays> arrays, const render::Mat4* transform, std::optional<bool> visible,
                 std::optional<MeshLook> look,
                 std::optional<std::vector<InstancerLink>> instancing = std::nullopt,
                 std::optional<MeshTransforms> shutter = std::nullopt);
    /// A material's network as a MaterialX document (null where hdMtlx could not
    /// read it: its meshes show displayColor). Compiled at the next commit.
    void setMaterial(const pxr::SdfPath& id, std::shared_ptr<void> mtlxDocument);
    /// An `aofx://` texture filled from the host's device buffer (see
    /// TextureStore::updateExternal). On the rendering thread, before the
    /// frame that samples it.
    [[nodiscard]] Result<void> updateExternalTexture(const std::string& name, const gpu::Buffer& rgba,
                                                     uint32_t width, uint32_t height, uint32_t rowPixels);
    void removeMaterial(const pxr::SdfPath& id);
    /// A UsdLux light, as the delegate read it. Lights light the meshes; the
    /// splats carry their own radiance until AtheneaSplatLightingAPI (M5).
    void setLight(const pxr::SdfPath& id, const light::Light& lamp,
                  std::vector<InstancerLink> instancing = {});
    void removeLight(const pxr::SdfPath& id);
    /// Samples per light per pixel. One is what an interactive frame takes;
    /// a render that wants an area light without noise asks for more.
    void setLightSamples(uint32_t samples);
    /// One light per sample, chosen by power, instead of every light at every
    /// pixel: exact either way, and which is cheaper is a measurement.
    void setChooseLights(bool choose);
    /// Whether a relit splat casts a shadow ray against the cloud's own
    /// proxies (`athenea:splatShadows`). Off: it takes each light whole, which
    /// is what relighting did before there was a ray to ask with.
    void setSplatShadows(bool shadows);
    /// Whether a cloud that carries a transfer adds its indirect half.
    void setSplatTransferIndirect(bool indirect);
    /// Whether a gaussian reflects the cloud it belongs to, at a ray each.
    void setSplatReflections(bool reflect);

    /// WHAT A FRAME SAYS A PRIM IS MADE OF, keyed on the Cryptomatte id its
    /// gaussians carry: what a host edits after picking one.
    ///
    /// Read by the shading, never baked. The rows live until they are
    /// cleared; setting one that is already there replaces it. Nothing on
    /// disk is touched, and a cloud with no ids is unaffected.
    void setSplatOverride(const render::SplatOverride& said);
    void clearSplatOverride(uint32_t id);
    void clearSplatOverrides();
    [[nodiscard]] std::vector<render::SplatOverride> splatOverrides() const;
    /// What the gaussians carrying `id` are made of now, counted on the
    /// device over every cloud in the scene (render::measureSplatId).
    [[nodiscard]] Result<render::SplatIdReading> measureSplatId(uint32_t id);
    /// Whether the clouds of a frame cast a shadow on the meshes of it
    /// (`athenea:cloudShadows`), measured from each light into a map and read
    /// with no ray. On by default: a bird that casts nothing is wrong, and
    /// the map costs one pass over the gaussians a light. Off, a mesh under a
    /// cloud is lit as if the cloud were not there, which is what every
    /// raster frame did before.
    void setCloudShadows(bool shadows);
    /// Whether the raster route reads a dome prefiltered on a mesh
    /// (`athenea:domePrefiltered`): the harmonics' irradiance and the sky's
    /// mip chain, shadowed by the cloud map's dome directions, with no sample
    /// and no ray. On by default; off, a dome is sampled and traced as it
    /// was, one sample a pixel and its grain.
    void setDomePrefiltered(bool prefiltered);
    /// Texels a side of that map, per light (`athenea:cloudShadowResolution`).
    void setCloudShadowResolution(uint32_t texels);
    /// How many terms of the map each texel keeps (`athenea:cloudShadowTerms`):
    /// 1 is the total optical depth, which is exact for a receiver behind the
    /// whole cloud -- a floor -- and wrong for one inside it; 3, 5 or 7 add
    /// Fourier pairs that resolve the depth a receiver stands at, which is
    /// what a cloud shadowing itself needs. 0: 1 where nothing but meshes
    /// receive, 5 where a relit cloud does.
    void setCloudShadowTerms(uint32_t terms);
    /// What the map's optical depth is multiplied by (`athenea:cloudShadowDensity`):
    /// the shadow's density, as a compositor means it. 1 is what the cloud's
    /// own opacity says, and the only value that is a measurement.
    void setCloudShadowDensity(float density);
    /// Whether a path traced frame moves the camera inside the pixel between
    /// the passes it is gathered from (`athenea:antialias`). On: edges are
    /// averaged over the pixel's area, at no cost in rays. Off: every ray
    /// goes through the pixel's middle, which is what a comparison of two
    /// frames pixel by pixel needs.
    void setAntialias(bool on);
    /// Where a material's relative texture path is relative to: the folder
    /// the stage was opened from. A Hydra material network does not carry it,
    /// and a MaterialX network's `file` arrives exactly as the asset wrote it.
    void setAssetSearchPath(const std::filesystem::path& directory);
    /// Paths a pixel a path traced frame gathers, and how many bounces each
    /// one takes after its first hit. One of each is what an interactive
    /// frame affords.
    void setPathSamples(uint32_t samples);
    void setPathBounces(uint32_t bounces);
    /// Motion blur's shutter slices (1 to 8) for a path traced frame whose
    /// prims move over the camera's shutter; one is no blur.
    void setMotionBuckets(uint32_t buckets);
    /// The camera's shutter, in frames about the frame: what the buckets
    /// span, and what the prims' samples are placed against.
    void setShutter(double open, double close);
    /// Paths a pixel at which a path traced frame is finished. One -- the
    /// default -- is a frame that never accumulates, which is what a viewport
    /// showing a moving camera wants.
    void setPathTotal(uint32_t total);
    /// Denoise a path traced frame once it has gathered `pathTotal` paths
    /// (every frame, when the total is one). Off by default.
    void setDenoise(bool denoise);
    /// Adaptive: a pixel stops taking paths once its relative standard error
    /// falls below `error`; the frame is gathered when every covered pixel
    /// has stopped or `pathTotal` is reached, whichever first.
    void setPathAdaptive(bool adaptive);
    void setPathMis(bool mis);
    void setPathError(float error);

    /// How many paths a pixel the path traced frame on screen has gathered,
    /// and whether that is all it is going to gather. A frame that is not a
    /// path traced one has nothing to gather and is always finished.
    [[nodiscard]] uint32_t pathAccumulated() const noexcept;
    [[nodiscard]] bool pathConverged() const noexcept;
    /// The mesh pools' generation (a repack each) and positions revision (a
    /// deformation in place each), so a host can tell which one a change was.
    [[nodiscard]] uint64_t meshGeneration() const noexcept;
    /// The coordinate systems bound to a mesh prim, as its last Sync read them.
    [[nodiscard]] std::vector<CoordSysBinding> coordSysOf(const pxr::SdfPath& id) const;
    [[nodiscard]] uint64_t meshPositionsRevision() const noexcept;
    /// Volumes and the field assets they read. The medium is path traced;
    /// the raster technique draws no volume, and says so once.
    void setVolume(const pxr::SdfPath& id, VolumeArrays arrays);
    void removeVolume(const pxr::SdfPath& id);
    void setVolumeField(const pxr::SdfPath& id, VolumeFieldAsset asset);
    void removeVolumeField(const pxr::SdfPath& id);
    void setInstancer(const pxr::SdfPath& id, const pxr::SdfPath& parent, InstancerArrays arrays);
    void removeInstancer(const pxr::SdfPath& id);
    void remove(const pxr::SdfPath& id);

    // --- from the render pass (one thread) ---
    /// Uploads what changed. Returns how many entries were uploaded.
    Result<size_t> commit();
    /// `settleStreams`: before drawing, cut and load until the streamed
    /// assets hold what this view wants (as much as their budgets allow) --
    /// for an image that must be complete. Otherwise streams fill in over
    /// the frames that follow.
    Result<void> render(const render::Projection& projection, const render::RenderSettings& settings,
                        render::RenderTargets& targets, Technique technique = Technique::Raster,
                        bool settleStreams = false, const pxr::TfTokenVector* renderTags = nullptr,
                        const AovRequest& aovs = {}, MeshVisibility visibility = MeshVisibility::Automatic,
                        const BakeRequest* bake = nullptr);

    /// The light this stage's meshes carry, at points somebody names: for
    /// every point, a ray from just off the surface back down onto it, path
    /// traced with the scene's own lights, its shadows and its bounces. What
    /// comes back is the radiance leaving that point along its normal.
    ///
    /// It is a frame in every way but the camera -- the same materials, the
    /// same lights, the same integrator -- so it is a render with a bake
    /// request in it rather than a pipeline of its own. What it is for:
    /// turning a mesh into gaussians that carry the light the mesh had.
    /// WHAT A BAKE OF A CLOUD'S VISIBILITY HANDS BACK: the two arrays the
    /// file will carry, as the device wrote them.
    struct BakedVisibility {
        std::vector<float>   parts;    ///< 12 floats a part
        std::vector<int32_t> texels;   ///< two f16 a word
        std::vector<int32_t> partOf;   ///< the part of each gaussian
        std::vector<int32_t> ambient;  ///< a probe's mean, for domes
        uint32_t             partCount = 0;
    };
    /// Bakes the per-part visibility of the committed cloud at `id`, which
    /// must be one a skeleton carries (its influences say which part each
    /// gaussian is). The cloud keeps the fields afterwards, so the frames
    /// that follow already read them.
    [[nodiscard]] Result<BakedVisibility> bakeVisibility(const pxr::SdfPath& id, const technique::VisibilityParts& parts,
                                                         const technique::VisibilityBakeOptions& options);

    [[nodiscard]] Result<void> bakePoints(const BakeRequest& bake, const render::Projection& projection,
                                          const render::RenderSettings& settings);

    [[nodiscard]] gpu::Device& device() noexcept { return *device_; }
    [[nodiscard]] gpu::ShaderLibrary& library() noexcept { return *library_; }

    /// WHEN THE DEVICE RUNS OUT OF MEMORY.
    ///
    /// What a frame that failed with OutOfMemory asks before it is tried
    /// again (the render pass does, once). Every call gives back what is
    /// made again on demand -- the rasteriser's grown buffers, the ray
    /// tracers' structures and proxies, the denoiser -- and gives up one
    /// thing more, in this order: splat shadows (`athenea:splatShadows`), then
    /// a level of detail at a time, each one level coarser for a LOD group
    /// (athenea:lod:group), a cut of twice the pixels for a streamed asset,
    /// and its streaming budget halved. Returns what it did, for the
    /// warning and the viewer's panel; empty when nothing is left to give.
    [[nodiscard]] std::string relieveMemory();
    /// How many levels coarser than asked the engine draws now (0: as asked).
    [[nodiscard]] uint32_t lodBias() const noexcept { return lodBias_; }
    /// How many times memory was given up -- relieveMemory, or splat shadows
    /// skipped for a budget they would not fit -- and what the last one did.
    [[nodiscard]] uint32_t reliefs() const noexcept { return reliefs_; }
    [[nodiscard]] const std::string& lastRelief() const noexcept { return lastRelief_; }
    /// Whether splat shadows were given up for memory, by relieveMemory or
    /// because the proxies would not fit the device's budget.
    [[nodiscard]] bool splatShadowsGivenUp() const noexcept { return memoryNoSplatShadows_.load(); }
    /// The last frame's failure, kept by the render pass, which Hydra gives
    /// no way to return (StageRenderer::execute does), once.
    void noteFrameError(Error error) { frameError_ = std::move(error); }
    [[nodiscard]] std::optional<Error> takeFrameError() {
        std::optional<Error> out = std::move(frameError_);
        frameError_.reset();
        return out;
    }

    /// The targets the last render drew into (owned by the render pass).
    [[nodiscard]] const render::RenderTargets* lastTargets() const noexcept { return lastTargets_; }

    /// Where everything drawn in the last frame is, in world space: mesh boxes
    /// folded on the device, cloud boxes through their prims' transforms.
    /// Nothing before a frame has drawn anything.
    [[nodiscard]] Result<std::optional<scene::Bounds>> bounds();

    /// The last frame's `aov` on the device, for a caller that shows it
    /// there (athenea view) rather than reading it back.
    [[nodiscard]] AovView aovView(const render::RenderTargets& targets, AovSource aov) const;

    /// What the last frame held (`FrameCounters`).
    [[nodiscard]] const FrameCounters& frameCounters() const noexcept { return counters_; }

    /// THE GAUSSIANS ON SCREEN, for a panel (GaussianStats): gathered only
    /// while a panel asks for them, since the device counts a few things more
    /// for it -- four small dispatches and a copy that nothing waits for.
    void setCountSplats(bool on) { countSplats_.store(on); }
    /// Time each stage of the rasteriser: every stage then waits for the
    /// device, so the frame is slower by what the waits cost.
    void setTimeSplatStages(bool on) { timeSplatStages_.store(on); }
    [[nodiscard]] const GaussianStats& gaussianStats() const noexcept { return gaussianStats_; }

    /// What the last frame's Cryptomatte ids are called: path -> id, for the
    /// manifest an EXR carries and for anything that has to name an id.
    /// How many times a splat cloud's arrays have been uploaded, in all.
    /// The level of each LOD group a view draws (athenea:lod:group).
    [[nodiscard]] std::set<const SplatEntry*> lodLevelsFor(const render::Projection& projection) const;
    [[nodiscard]] uint64_t cloudUploads() const noexcept { return cloudUploads_.load(); }
    [[nodiscard]] const std::map<std::string, uint32_t>& cryptoManifest() const noexcept {
        return cryptoManifest_;
    }

    /// A render target as a Hydra render buffer's bytes, converted on the
    /// device -- format, row order, and for depth the host projection's [0, 1]
    /// from view z (`projection` is the host's row-vector matrix, 16 values) --
    /// and read into `into`.
    [[nodiscard]] Result<void> writeAov(const render::RenderTargets& targets, AovSource source,
                                        const AovLayout& layout, const double* projection, std::span<uint8_t> into);

private:
    Engine() = default;

    /// relieveMemory's state: how many levels coarser, and whether the
    /// splat shadows are given up.
    uint32_t                                  lodBias_ = 0;
    std::atomic<bool>                         memoryNoSplatShadows_{false};
    std::optional<Error>                      frameError_;
    uint32_t                                  reliefs_ = 0;
    std::string                               lastRelief_;
    /// The streaming budget, in splats, a streamed asset is opened with: what
    /// it asks for (or the whole file where it asks for none), held to half
    /// of what the device's budget has left, and halved once for every
    /// level relieveMemory has given up. 0: read the file whole.
    [[nodiscard]] uint64_t streamBudget(const pxr::SdfPath& id, const StreamedAsset& asset) const;
    /// Whether the splat shadow proxies of `gaussians` fit what the device's
    /// budget has left; says once why not.
    [[nodiscard]] bool splatShadowsFit(uint64_t gaussians);
    uint64_t                                  shadowGaussians_ = 0;   ///< what the shadow tracer was last prepared for

    std::shared_ptr<gpu::Device>              device_;
    std::unique_ptr<gpu::ShaderLibrary>       library_;
    std::optional<scene::CloudLoader>         loader_;
    std::optional<render::TileRasterizer>     rasterizer_;
    std::optional<render::PointRasterizer>    pointRasterizer_;
    std::optional<render::GaussianRayTracer>  rayTracer_;   ///< made on first use
    std::optional<lod::CutSelector>           cutter_;      ///< made on first use
    std::optional<gpu::ComputeKernel>         aovConvert_;  ///< made on first use
    const render::RenderTargets*              lastTargets_ = nullptr;
    render::RenderTargets                     pointLayer_;

    mutable std::mutex                        guard_;
    std::map<pxr::SdfPath, SplatEntry>        splats_;
    std::map<pxr::SdfPath, PointsEntry>       points_;
    std::map<pxr::SdfPath, MeshEntry>         meshes_;
    std::map<pxr::SdfPath, InstancerEntry>    instancers_;
    uint64_t                                  instancerVersion_ = 0;
    uint64_t nextTopologyKey_ = 0;   ///< one per topology a mesh was given
    std::optional<world::Instancing>          instancing_;
    std::optional<geom::MeshBuilder>          meshBuilder_;
    std::optional<geom::Skinner>              skinner_;
    std::optional<geom::CurveBuilder>         curveBuilder_;
    std::optional<geom::Subdivider>           subdivider_;   ///< made on first use
    std::optional<world::GpuScene>            scene_;
    /// A cloud that carries a baked visibility has its factors measured
    /// before the frame -- one float a splat a light -- and both routes read
    /// them instead of casting a ray. Grown to the frame's clouds.
    std::optional<technique::SplatVisibility> splatVisibility_;
    gpu::Buffer                               visibilityFactors_;
    /// This frame's clouds that carry one, noted as the instance list is
    /// built so their factors sit at the slots the renderers will read.
    struct MeasuredCloud {
        const scene::GpuSplats* drawn = nullptr;   ///< what the frame draws (posed, or the cloud)
        const scene::GpuSplats* fields = nullptr;  ///< what carries the baked fields (the bind pose)
        const gpu::Buffer*      xforms = nullptr;  ///< the skeleton's, or null for a cloud nothing moves
        uint32_t                slot = 0;
        std::array<float, 12>   rows{};
        uint64_t                categories = 0;
        std::array<float, 12>   geomBind{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    };
    std::vector<MeasuredCloud>                frameMeasured_;
    /// Every visible cloud of the frame with its slot, whether or not it
    /// carries a baked field: what the shadow map fills factors for, since a
    /// cloud without a field would otherwise take every light whole.
    struct SlottedCloud {
        const scene::GpuSplats* drawn = nullptr;
        uint32_t                slot = 0;
        bool                    relight = false;
        bool                    hasField = false;
        std::array<float, 12>   rows{};
    };
    std::vector<SlottedCloud>                 frameClouds_;
    uint32_t                                  frameSlots_ = 0;
    /// Measures the frame's factors into `into`, once, for either route.
    [[nodiscard]] Result<void> measureVisibility(render::SplatLights& into);
    std::optional<scene::SplatSkinner>        splatSkinner_;
    std::optional<technique::VisibilityRaster> visibilityRaster_;   ///< each made on first use
    std::optional<world::RayTracingScene>      rayTracingScene_;
    std::optional<technique::VisibilityTrace>  visibilityTrace_;
    std::optional<world::BvhScene>             bvhScene_;
    std::optional<technique::VisibilityBvh>    visibilityBvh_;
    std::optional<technique::MaterialPrograms> materialPrograms_;
    std::optional<technique::MaterialShading> materialShading_;
    /// What this frame's clouds stop, measured from each light into a map and
    /// read with no ray: how a mesh is shadowed by a cloud in the raster
    /// route, and on a device that cannot trace at all.
    std::optional<technique::SplatShadowMap>  cloudShadowMap_;
    std::optional<technique::PathTracer>      pathTracer_;   ///< made on first use
    technique::PathAux                        pathAux_;      ///< the last path traced frame's albedo and normal
    bool                                      pathAuxValid_ = false;
    /// The last frame's light groups: `lightGroupCount_` planes of float4,
    /// a pixel each, the means in `colour` and the path tracer's sums in
    /// `sum`; sized for `lightGroupPixels_`.
    gpu::Buffer                               lightGroupColour_;
    uint32_t                                  lightGroupCount_ = 0;
    uint64_t                                  lightGroupPixels_ = 0;
    std::optional<technique::Denoiser>        denoiser_;     ///< made on first use
    std::atomic<bool>                         denoise_{false};
    bool                                      denoiserFailed_ = false;   ///< said once
    std::map<pxr::SdfPath, MaterialEntry>     materials_;
    /// A light as the delegate read it, and the instancers above it, whose
    /// chain is composed on the device as a mesh's is.
    struct LightEntry {
        light::Light               lamp;
        std::vector<InstancerLink> instancing;
        world::InstanceChain       chain;
        world::InstanceChain       chainStart;       ///< as a mesh's: where an instancer in the chain moves
        world::InstanceChain       chainEnd;
        double                     chainTimeStart = 0.0;
        double                     chainTimeEnd = 0.0;
        std::vector<uint64_t>      chainVersions;
        bool                       chainDirty = false;
    };
    std::map<pxr::SdfPath, LightEntry>        lights_;
    float                                     lightSceneRadius_ = 1.0F;   ///< the scene's reach, for domes' and suns' power
    uint64_t                                  lightRadiusGeneration_ = ~uint64_t{0};
    uint64_t                                  lightRadiusRevision_ = ~uint64_t{0};
    std::map<pxr::SdfPath, VolumeArrays>      volumes_;
    std::map<pxr::SdfPath, VolumeFieldAsset>  volumeFields_;
    /// Grids read, by file and grid name; null for one that failed to read
    /// (reported once).
    std::map<std::pair<std::string, std::string>, std::shared_ptr<const io::NanoGrid>> nanoGrids_;
    std::optional<world::VolumeSet>           volumeSet_;
    uint64_t                                  volumesVersion_ = 1;   ///< raised by any volume or field change
    uint64_t                                  volumesBuilt_ = 0;     ///< the version volumeSet_ holds
    uint32_t                                  volumesDrawn_ = 0;     ///< how many volumes volumeSet_ holds
    bool                                      volumesUndrawnSaid_ = false;
    /// IES profiles by path, read once; a file that cannot be read is said
    /// once and the light goes unshaped.
    std::map<std::string, std::shared_ptr<const io::IesProfile>> iesProfiles_;
    std::set<std::string>                     iesFailed_;
    std::optional<light::LightTable>          lightTable_;
    std::atomic<uint32_t>                     lightSamples_{1};
    std::atomic<bool>                         chooseLights_{false};
    std::atomic<bool>                         splatShadows_{false};
    std::atomic<bool>                         cloudShadows_{true};
    std::atomic<bool>                         domePrefiltered_{true};
    std::atomic<uint32_t>                     cloudShadowTexels_{1024};
    std::atomic<uint32_t>                     cloudShadowTerms_{0};
    std::atomic<float>                        cloudShadowDensity_{1.0F};
    std::atomic<bool>                         antialias_{true};
    std::filesystem::path                     assetSearchPath_;
    /// Built only for relit splats to shadow against: the Hardware route, so
    /// there is a structure an inline ray can trace (the frame's own tracer
    /// may be on the compute route, which has none).
    std::optional<render::GaussianRayTracer>  shadowTracer_;
    /// The same proxies packed into one buffer, for the path tracer's shadow
    /// rays (technique::SplatShadows), and whether this frame built them.
    std::optional<technique::SplatShadows> splatShadowScene_;
    bool                                   shadowTracerReady_ = false;
    std::atomic<uint32_t>                     pathSamples_{1};
    std::atomic<uint32_t>                     pathBounces_{1};
    std::atomic<uint32_t>                     motionBuckets_{4};
    std::atomic<double>                       shutterOpen_{0.0};
    std::atomic<double>                       shutterClose_{0.0};
    std::atomic<uint32_t>                     pathTotal_{1};
    std::atomic<bool>                         pathAdaptive_{false};
    std::atomic<bool>                         pathMis_{true};
    std::atomic<bool>                         transferIndirect_{true};
    /// How many times a cloud's arrays were uploaded: once a cloud, unless
    /// what it holds changed (tests hold a time step to it).
    double                                    lastCommitMs_ = 0.0;   ///< ATHENEA_STAGES: the last commit's time
    double                                    meshShadowMapMs_ = 0.0;   ///< ATHENEA_STAGES: the cloud map's build
    /// What the cloud shadow map was last built from (Engine::render): a
    /// frame with the same key reads it again.
    std::vector<uint64_t>                     cloudShadowKey_;
    std::atomic<uint64_t>                     cloudUploads_{0};
    std::atomic<bool>                         splatReflections_{false};
    /// The per-prim material table and the buffer it is uploaded into. The
    /// rows are the host's; `splatOverrideRows_` is what the frame bound.
    std::vector<render::SplatOverride>        splatOverrides_;
    gpu::Buffer                               splatOverrideBuffer_;
    uint32_t                                  splatOverrideRows_ = 0;
    bool                                      splatOverridesDirty_ = false;
    [[nodiscard]] Result<void> commitSplatOverrides();
    std::atomic<float>                        pathError_{0.02F};
    technique::PathProgress                   pathProgress_;   ///< after the last adaptive pass
    uint32_t                                  pathSeed_ = 0;   ///< which samples a path traced frame takes
    /// What the last path traced frame was of. A frame that matches it in
    /// every particular is the same frame continued, and its paths are added
    /// to the mean; anything else starts the mean again. The revision is what
    /// says the scene itself moved: `commit` raises it whenever it uploads,
    /// and so does every setting that changes what a path would find.
    struct PathState {
        render::Mat4 worldToView{};
        double       focalX = 0.0;
        double       focalY = 0.0;
        double       centreX = 0.0;
        double       centreY = 0.0;
        double       nearZ = 0.0;
        double       farZ = 0.0;
        bool         orthographic = false;
        double       lensRadius = 0.0;
        double       focusDistance = 0.0;
        double       distortionK1 = 0.0;
        double       distortionK2 = 0.0;
        uint32_t     width = 0;
        uint32_t     height = 0;
        uint32_t     samples = 0;
        uint32_t     bounces = 0;
        bool         adaptive = false;
        bool         mis = true;
        bool         cameraMoves = false;
        render::Mat4 cameraStart = render::Mat4::identity();
        render::Mat4 cameraEnd = render::Mat4::identity();
        float        error = 0.0F;
        uint64_t     revision = 0;
        uint64_t     tags = 0;         ///< a hash of the render tags drawn: purposes that change start the mean again
        bool         traced = false;   ///< the last frame was path traced at all
        /// Mat4 has no comparison of its own, so the camera is compared
        /// element by element: identical bits are what "has not moved" means.
        [[nodiscard]] bool operator==(const PathState& o) const {
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    if (worldToView.at(r, c) != o.worldToView.at(r, c) || cameraStart.at(r, c) != o.cameraStart.at(r, c) ||
                        cameraEnd.at(r, c) != o.cameraEnd.at(r, c)) {
                        return false;
                    }
                }
            }
            if (cameraMoves != o.cameraMoves) {
                return false;
            }
            return focalX == o.focalX && focalY == o.focalY && centreX == o.centreX && centreY == o.centreY &&
                   nearZ == o.nearZ && farZ == o.farZ && orthographic == o.orthographic &&
                   lensRadius == o.lensRadius && focusDistance == o.focusDistance &&
                   distortionK1 == o.distortionK1 && distortionK2 == o.distortionK2 && width == o.width &&
                   height == o.height && samples == o.samples && bounces == o.bounces && adaptive == o.adaptive && mis == o.mis &&
                   error == o.error && revision == o.revision && tags == o.tags && traced == o.traced;
        }
    };
    PathState                                 pathState_;
    std::atomic<uint64_t>                     revision_{1};   ///< raised by anything a path would see
    /// Frames still to draw before what is gathered is the shutter's: a
    /// shutter changed after the prims synced is answered by dirtying them
    /// (RenderPass), and they carry their new samples only from the next
    /// Sync -- so this frame and the one that resamples are not converged,
    /// however many paths they hold.
    std::atomic<int>                          shutterSettle_{0};
    /// A bit per category name, as they are first seen: a prim's mask and a
    /// light's link have to agree on the numbering, and this is the only
    /// place that sees both. Past 64 names a category cannot be represented
    /// and its link reaches nothing, which is said once.
    std::map<std::string, uint32_t>           categoryBits_;
    [[nodiscard]] uint32_t categoryBit(const std::string& name);
    [[nodiscard]] uint64_t categoryMask(const std::vector<pxr::TfToken>& names);
    bool                                      materialsChanged_ = true;
    bool                                      materialCutouts_ = false;   ///< a material in the frame cuts samples away
    std::unique_ptr<material::MaterialCompiler> compiler_;
    bool                                      compilerFailed_ = false;
    std::unique_ptr<material::TextureStore>   textures_;
    std::map<pxr::SdfPath, uint32_t>          materialRows_;     ///< into materialRecords_; absent: row 0, the fallback
    std::vector<std::string>                  materialSlotNames_;   ///< the extra primvar slots the blob was written for
    gpu::Buffer                               materialRecords_;
    gpu::Buffer                               materialBlob_;
    /// Row and blob for this frame's materials, primvar slots set on the scene.
    [[nodiscard]] Result<void> prepareMaterials(const std::vector<std::string>& aovPrimvars);
    /// The light table a frame of relit splats needs, for both routes that
    /// draw one (AtheneaSplatLightingAPI).
    /// Each lamp's own pictures on the device: a dome's image in the texture
    /// table the materials sample, and an IES profile read off disk. Both
    /// routes that build a light table come through here, since a cloud
    /// standing alone under a dome has one to show as much as a mesh does.
    [[nodiscard]] Result<void> prepareLightImages(std::vector<light::Light>& lamps);
    /// The domes' skies, prepared for the kernels that cannot sample them
    /// (`technique::Environment`). Rebuilt only when a dome's record changes,
    /// which is what keeps it off the frame's path.
    [[nodiscard]] Result<void> prepareEnvironment(const std::vector<light::Light>& lamps);
    /// The prepared sky handed to a renderer, where there is one.
    void bindEnvironment(render::SplatLights& lights) const;
    /// The `cryptoOfPrim_` table and the manifest beside it, rebuilt when the
    /// scene's prims have changed since the last matte.
    [[nodiscard]] Result<void> prepareCryptoTable();
    [[nodiscard]] Result<void> prepareSplatLights(std::vector<light::Light>& lamps,
                                                  std::span<const render::SplatInstance> splats);
    std::optional<gpu::ComputeKernel>          nearest_;
    std::optional<gpu::ComputeKernel>          domeBackground_;
    std::optional<gpu::ComputeKernel>          exposure_;   ///< made on first use
    /// The emitting triangles as a light, and what they were weighed from.
    struct EmissiveKey {
        uint64_t          generation = ~uint64_t{0};
        uint64_t          positions = 0;
        uint64_t          revision = 0;
        rhi::IBuffer*     records = nullptr;
        uint32_t          instances = 0;
        [[nodiscard]] bool operator==(const EmissiveKey& o) const {
            return generation == o.generation && positions == o.positions && revision == o.revision &&
                   records == o.records && instances == o.instances;
        }
    };
    std::optional<technique::EmissiveTable>    emissiveTable_;
    EmissiveKey                                emissiveKey_;
    std::optional<gpu::ComputeKernel>          domeGroups_;     ///< the domes' background in their groups' planes
    std::optional<gpu::ComputeKernel>          groupsScaled_;   ///< the path tracer's group means, copied out
    /// The camera's exposure over the composed frame, once, after everything.
    [[nodiscard]] Result<void> applyExposure(double stops, uint32_t width, uint32_t height,
                                             render::RenderTargets& targets);
    /// The frame's domes over what it drew nothing on, after everything else.
    /// The frame's light group planes into lightGroupColour_: the path
    /// tracer's means copied out of its accumulation (the raster writes
    /// there itself), so the domes and the exposure change a copy.
    [[nodiscard]] Result<void> gatherLightGroups(bool traced, uint32_t width, uint32_t height);
    /// Puts a cloud a skeleton carries where the skeleton is. The uploaded
    /// cloud stays the bind pose and `entry.posed` is what a frame draws, so
    /// what a renderer holds never changes identity between frames.
    [[nodiscard]] Result<void> carryCloud(const pxr::SdfPath& id, SplatEntry& entry,
                                          bool reuploaded);
    [[nodiscard]] static CloudIdentity identityOf(const ParticleFieldArrays& arrays);

    [[nodiscard]] Result<void> paintDomes(const render::Projection& projection, uint32_t width, uint32_t height,
                                          render::RenderTargets& targets);
    technique::VisibilityTargets              visibility_;
    FrameCounters                             counters_;
    /// The Gaussians panel's numbers (`gaussianStats`), and what gathers them.
    std::atomic<bool>                         countSplats_{false};
    std::atomic<bool>                         timeSplatStages_{false};
    uint64_t                                  frameSerial_ = 0;
    GaussianStats                             gaussianStats_;
    /// The prims of the frames whose counts are still on their way, by tag:
    /// a count arrives by instance, and only the frame it was drawn in says
    /// which prim each instance was.
    struct CountedFrame {
        uint64_t                 tag = 0;
        std::vector<std::string> prims;
    };
    std::vector<CountedFrame>                 countedFrames_;
    std::optional<render::SplatCounters>      lastCounted_;
    std::vector<std::string>                  lastCountedPrims_;
    /// This frame's clouds, levels and submissions into `gaussianStats_`.
    void noteGaussians(const std::string& route, std::span<const render::SplatInstance> splats,
                       std::span<const std::string> prims, std::span<const lod::CutStats> cutStats,
                       std::span<const std::string> cutPrims, const std::set<const SplatEntry*>& levels);
    /// Whatever counts the device has finished since, into `gaussianStats_`.
    void takeSplatCounters();
    std::optional<technique::Environment>     environment_;
    /// The dome records the environment was prepared from: a frame whose
    /// domes read the same builds nothing.
    std::vector<light::LightRecord>           environmentKeys_;
    std::optional<technique::CryptoShading>   cryptoShading_;
    /// The Cryptomatte id of each prim id, as the frame's meshes hash: a table
    /// the id plane is read through, so an instanced prim needs no record of
    /// its own. Rebuilt when the scene's prims change.
    gpu::Buffer                               cryptoOfPrim_;
    uint32_t                                  cryptoPrimCount_ = 0;
    uint64_t                                  cryptoTableRevision_ = ~uint64_t{0};
    size_t                                    cryptoTableMeshes_ = 0;
    /// What each id in the matte is called: the manifest a reader needs, the
    /// drawn meshes' paths and whatever the frame's clouds brought with them.
    std::map<std::string, uint32_t>           cryptoManifest_;
    std::optional<technique::AovShading>      aovShading_;
    technique::AovBuffers                     aovs_;
    bool                                      aovsValid_ = false;
    render::RenderTargets                     meshLayer_;
    render::RenderTargets                     opaqueLayer_;
};

}   // namespace athenea::usd
