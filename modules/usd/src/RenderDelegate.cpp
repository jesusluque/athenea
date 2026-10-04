// Copyright (c) 2026 jesus luque.
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/coordSys.h>
#include <pxr/imaging/hd/extComputation.h>
#include <pxr/imaging/hd/sceneIndexPluginRegistry.h>
#include <pxr/imaging/hdsi/coordSysPrimSceneIndex.h>
#include <pxr/imaging/hdsi/implicitSurfaceSceneIndex.h>
#include <pxr/imaging/hdsi/nurbsApproximatingSceneIndex.h>
#include <pxr/imaging/hdsi/pinnedCurveExpandingSceneIndex.h>
#include <pxr/imaging/hdsi/tetMeshConversionSceneIndex.h>
#include <pxr/imaging/hdsi/lightLinkingSceneIndex.h>
#include <pxr/imaging/hdsi/materialPrimvarTransferSceneIndex.h>
#include <pxr/imaging/hdsi/renderSettingsFilteringSceneIndex.h>
#include <pxr/imaging/hdsi/velocityMotionResolvingSceneIndex.h>

#include "RenderDelegate.h"
#include "Resample.h"
#include "SkelSplat.h"

#include <pxr/imaging/hd/primvarsSchema.h>

#include <pxr/imaging/hd/xformSchema.h>
#include "Camera.h"

#include "Light.h"
#include "Material.h"

#include <pxr/base/tf/staticTokens.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/imaging/hd/tokens.h>

#include "Instancer.h"
#include "Curves.h"
#include "Mesh.h"
#include "ParticleField.h"
#include "Points.h"
#include "RenderBuffer.h"
#include "RenderPass.h"
#include "RenderSettings.h"
#include "Volume.h"
#include "athenea/core/Log.h"

PXR_NAMESPACE_OPEN_SCOPE

/// Which prims the light linking scene index is about. Its defaults are not
/// ours to assume: without saying so it sees the collections and marks
/// nobody, which is what the probe showed -- a good membership expression and
/// no categories at all.
static HdContainerDataSourceHandle _lightLinkingArgs() {
    VtArray<TfToken> lights;
    for (const TfToken& type : HdLightPrimTypeTokens()) {
        lights.push_back(type);
    }
    const VtArray<TfToken> geometry{HdPrimTypeTokens->mesh, HdPrimTypeTokens->basisCurves,
                                    HdPrimTypeTokens->points, HdPrimTypeTokens->volume,
                                    HdPrimTypeTokens->instancer};
    return HdRetainedContainerDataSource::New(
        HdsiLightLinkingSceneIndexTokens->lightPrimTypes,
        HdRetainedTypedSampledDataSource<VtArray<TfToken>>::New(lights),
        HdsiLightLinkingSceneIndexTokens->geometryPrimTypes,
        HdRetainedTypedSampledDataSource<VtArray<TfToken>>::New(geometry));
}

static HdContainerDataSourceHandle _renderSettingsArgs() {
    static const VtArray<TfToken> prefixes{TfToken("athenea")};
    return HdRetainedContainerDataSource::New(
        HdsiRenderSettingsFilteringSceneIndexTokens->namespacePrefixes,
        HdRetainedTypedSampledDataSource<VtArray<TfToken>>::New(prefixes));
}

static HdContainerDataSourceHandle _implicitSurfaceArgs() {
    // Every implicit type to a mesh: the delegate has no primitive of its
    // own for any of them.
    const TfToken toMesh = HdsiImplicitSurfaceSceneIndexTokens->toMesh;
    return HdRetainedContainerDataSource::New(
        HdPrimTypeTokens->sphere, HdRetainedTypedSampledDataSource<TfToken>::New(toMesh),
        HdPrimTypeTokens->cube, HdRetainedTypedSampledDataSource<TfToken>::New(toMesh),
        HdPrimTypeTokens->cone, HdRetainedTypedSampledDataSource<TfToken>::New(toMesh),
        HdPrimTypeTokens->cylinder, HdRetainedTypedSampledDataSource<TfToken>::New(toMesh),
        HdPrimTypeTokens->capsule, HdRetainedTypedSampledDataSource<TfToken>::New(toMesh),
        HdPrimTypeTokens->plane, HdRetainedTypedSampledDataSource<TfToken>::New(toMesh));
}

void HdAtheneaRegisterSceneIndices() {
    // Once, and from anywhere: a host that makes the delegate itself never
    // goes through plug's discovery, so a registry function alone would not
    // run at all -- measured, by tracing it and seeing nothing.
    static const bool once = [] {
        // Velocities first, at the start of phase 0: a prim that authors
        // `velocities` (and `accelerations`) gets its points and instance
        // positions sampled at any shutter time from them, as UsdGeom's
        // velocity interpolation rules say -- so the delegate's shutter
        // samples read the same whether a stage authored samples or
        // velocities. It sits just downstream of the stage and upstream of
        // instancing's aggregation, which is where hdsi expects it.
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle& inputArgs) -> HdSceneIndexBaseRefPtr {
                return HdsiVelocityMotionResolvingSceneIndex::New(inputScene, inputArgs);
            },
            nullptr, 0, HdSceneIndexPluginRegistry::InsertionOrderAtStart);
        // Geometry the delegate does not draw natively, turned into meshes
        // and curves it does, in phase 1: implicit surfaces (sphere, cube,
        // cone, cylinder, capsule, plane) tessellated by hdsi, tetrahedral
        // meshes as their surface triangles, NURBS patches approximated, and
        // pinned curves expanded to the basis the delegate takes.
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle& inputArgs) -> HdSceneIndexBaseRefPtr {
                return HdsiImplicitSurfaceSceneIndex::New(inputScene, inputArgs);
            },
            _implicitSurfaceArgs(), 1, HdSceneIndexPluginRegistry::InsertionOrderAtStart);
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle&) -> HdSceneIndexBaseRefPtr {
                return HdsiTetMeshConversionSceneIndex::New(inputScene);
            },
            nullptr, 1, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle&) -> HdSceneIndexBaseRefPtr {
                return HdsiNurbsApproximatingSceneIndex::New(inputScene);
            },
            nullptr, 1, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle&) -> HdSceneIndexBaseRefPtr {
                return HdsiPinnedCurveExpandingSceneIndex::New(inputScene);
            },
            nullptr, 1, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
        // A splat cloud a Skeleton carries (SkelBindingAPI): the Skeleton's
        // resolved joints, which UsdSkel's imaging worked out upstream for
        // its meshes, put on the cloud as the primvar the engine skins by.
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle&) -> HdSceneIndexBaseRefPtr {
                return HdAtheneaSkelSplatSceneIndex::New(inputScene);
            },
            nullptr, 1, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
        // Coordinate systems bound to any xformable become coordSys prims
        // under it, in phase 2, so a material's binding names a prim of
        // that type with that prim's transform.
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle&) -> HdSceneIndexBaseRefPtr {
                return HdsiCoordSysPrimSceneIndex::New(inputScene);
            },
            nullptr, 2, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle& inputArgs) -> HdSceneIndexBaseRefPtr {
                return HdsiLightLinkingSceneIndex::New(inputScene, inputArgs);
            },
            _lightLinkingArgs(), 3, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
        // Primvars authored on a material reach the geometry bound to it
        // (a mesh's own win), in the phase Storm transfers them: a material
        // reading "displayColor" from its own prim otherwise read nothing.
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle&) -> HdSceneIndexBaseRefPtr {
                return HdsiMaterialPrimvarTransferSceneIndex::New(inputScene);
            },
            nullptr, 3, HdSceneIndexPluginRegistry::InsertionOrderAtStart);
        // Last: the delegate's own pass-through, which sends the notices the
        // scene does not carry (a shutter changed) to everything downstream.
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle&) -> HdSceneIndexBaseRefPtr {
                return HdAtheneaResampleSceneIndex::New(inputScene);
            },
            nullptr, 4, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
        // Render settings prims keep the `athenea:` namespaced settings and
        // their products reach the bprim; the other renderers' are dropped.
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "athenea",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle& inputArgs) -> HdSceneIndexBaseRefPtr {
                return HdsiRenderSettingsFilteringSceneIndex::New(inputScene, inputArgs);
            },
            _renderSettingsArgs(), 3, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
        return true;
    }();
    (void)once;
}


void HdAtheneaRenderDelegate::SetTerminalSceneIndex(const HdSceneIndexBaseRefPtr& terminalSceneIndex) {
    HdRenderDelegate::SetTerminalSceneIndex(terminalSceneIndex);
    _terminal = terminalSceneIndex;
}

bool HdAtheneaRenderDelegate::ResampleAllPrims() const {
    const HdSceneIndexBaseRefPtr terminal(_terminal);
    return terminal && HdAtheneaResampleUpstream(terminal, HdDataSourceLocatorSet{HdXformSchema::GetDefaultLocator(),
                                                                              HdPrimvarsSchema::GetDefaultLocator()}) > 0;
}

bool HdAtheneaRenderDelegate::DirtyVisibilityBelow(const SdfPath& root) const {
    const HdSceneIndexBaseRefPtr terminal(_terminal);
    return terminal &&
           HdAtheneaResampleUpstream(terminal, HdDataSourceLocatorSet{HdVisibilitySchema::GetDefaultLocator()}, root) > 0;
}

HdAtheneaRenderDelegate::HdAtheneaRenderDelegate() { _Setup(); }

HdAtheneaRenderDelegate::HdAtheneaRenderDelegate(std::shared_ptr<athenea::gpu::Device> device)
    : _device(std::move(device)) {
    _Setup();
}

HdAtheneaRenderDelegate::HdAtheneaRenderDelegate(HdRenderSettingsMap const& settings)
    : HdRenderDelegate(settings) {
    _Setup();
}

HdAtheneaRenderDelegate::~HdAtheneaRenderDelegate() {
    _param.reset();
    _engine.reset();
}

void HdAtheneaRenderDelegate::_Setup() {
    std::string why;
    _engine = athenea::usd::Engine::create(_device, why);
    if (_engine == nullptr) {
        athenea::log::error("hdAthenea: no engine: {}", why);
        // Kept, so a host with no console to read -- an app on a phone -- can
        // say why instead of only that.
        _engineError = why;
    }
    _param = std::make_unique<HdAtheneaRenderParam>(_engine.get());
    _registry = std::make_shared<HdResourceRegistry>();
}

TfTokenVector const& HdAtheneaRenderDelegate::GetSupportedRprimTypes() const {
    static const TfTokenVector types{HdPrimTypeTokens->mesh, HdPrimTypeTokens->particleField,
                                     HdPrimTypeTokens->points, HdPrimTypeTokens->basisCurves,
                                     HdPrimTypeTokens->volume};
    return types;
}

TfTokenVector const& HdAtheneaRenderDelegate::GetSupportedSprimTypes() const {
    static const TfTokenVector types{HdPrimTypeTokens->camera,      HdPrimTypeTokens->material,
                                     HdPrimTypeTokens->sphereLight, HdPrimTypeTokens->diskLight,
                                     HdPrimTypeTokens->rectLight,   HdPrimTypeTokens->distantLight,
                                     HdPrimTypeTokens->domeLight,   HdPrimTypeTokens->cylinderLight,
                                     HdPrimTypeTokens->extComputation, HdPrimTypeTokens->coordSys};
    return types;
}

TfTokenVector HdAtheneaRenderDelegate::GetMaterialRenderContexts() const {
    // MaterialX networks first, then UsdPreviewSurface ones, both through hdMtlx.
    return {TfToken("mtlx"), TfToken()};
}

TfTokenVector const& HdAtheneaRenderDelegate::GetSupportedBprimTypes() const {
    static const TfTokenVector types{HdPrimTypeTokens->renderBuffer, HdPrimTypeTokens->renderSettings,
                                     TfToken("openvdbAsset")};
    return types;
}

TfTokenVector HdAtheneaRenderDelegate::GetRenderSettingsNamespaces() const {
    // The namespaced settings a render settings prim keeps for this
    // renderer: `athenea:pathTotal` and the rest, as the delegate's own settings.
    static const TfTokenVector namespaces{TfToken("athenea")};
    return namespaces;
}

HdRenderPassSharedPtr HdAtheneaRenderDelegate::CreateRenderPass(HdRenderIndex* index,
                                                            HdRprimCollection const& collection) {
    return std::make_shared<HdAtheneaRenderPass>(index, collection, _engine.get(), this);
}

TF_DEFINE_PRIVATE_TOKENS(_atheneaSettings, ((technique, "athenea:technique"))((settleStreams, "athenea:settleStreams"))
                                           ((visibility, "athenea:visibility"))((lightSamples, "athenea:lightSamples"))((chooseLights, "athenea:chooseLights"))
                                           ((pathSamples, "athenea:pathSamples"))((pathBounces, "athenea:pathBounces"))((pathTotal, "athenea:pathTotal"))((denoise, "athenea:denoise"))((pathAdaptive, "athenea:pathAdaptive"))((pathMis, "athenea:pathMis"))((pathError, "athenea:pathError"))((motionBuckets, "athenea:motionBuckets"))((shutter, "athenea:shutter"))((disableMotionBlur, "athenea:disableMotionBlur"))((disableDepthOfField, "athenea:disableDepthOfField"))
                                           (raster)(rt)(automatic)(rays)(bvh));

HdRenderSettingDescriptorList HdAtheneaRenderDelegate::GetRenderSettingDescriptors() const {
    HdRenderSettingDescriptor technique;
    technique.name = "Technique (raster | rt)";
    technique.key = _atheneaSettings->technique;
    technique.defaultValue = VtValue(_atheneaSettings->raster);
    HdRenderSettingDescriptor settle;
    settle.name = "Wait for streamed assets before drawing";
    settle.key = _atheneaSettings->settleStreams;
    settle.defaultValue = VtValue(false);
    HdRenderSettingDescriptor visibility;
    visibility.name = "Mesh visibility (automatic | raster | rays | bvh)";
    visibility.key = _atheneaSettings->visibility;
    visibility.defaultValue = VtValue(_atheneaSettings->automatic);
    HdRenderSettingDescriptor samples;
    samples.name = "Samples per light";
    samples.key = _atheneaSettings->lightSamples;
    samples.defaultValue = VtValue(1);
    HdRenderSettingDescriptor choose;
    choose.name = "One light per sample, chosen by power";
    choose.key = _atheneaSettings->chooseLights;
    choose.defaultValue = VtValue(false);
    HdRenderSettingDescriptor paths;
    paths.name = "Paths per pixel (rt)";
    paths.key = _atheneaSettings->pathSamples;
    paths.defaultValue = VtValue(1);
    HdRenderSettingDescriptor bounces;
    bounces.name = "Bounces after the first hit (rt)";
    bounces.key = _atheneaSettings->pathBounces;
    bounces.defaultValue = VtValue(1);
    HdRenderSettingDescriptor total;
    total.name = "Paths per pixel to converge to (rt)";
    total.key = _atheneaSettings->pathTotal;
    total.defaultValue = VtValue(1);
    HdRenderSettingDescriptor denoise;
    denoise.name = "Denoise the path traced frame once gathered (rt)";
    denoise.key = _atheneaSettings->denoise;
    denoise.defaultValue = VtValue(false);
    HdRenderSettingDescriptor adaptive;
    adaptive.name = "Adaptive: a pixel stops once its error is below the target (rt)";
    adaptive.key = _atheneaSettings->pathAdaptive;
    adaptive.defaultValue = VtValue(false);
    HdRenderSettingDescriptor error;
    error.name = "Adaptive: relative standard error a pixel stops at (rt)";
    error.key = _atheneaSettings->pathError;
    error.defaultValue = VtValue(0.02f);
    HdRenderSettingDescriptor mis;
    mis.name = "Weigh light sampling and material sampling (MIS) (rt)";
    mis.key = _atheneaSettings->pathMis;
    mis.defaultValue = VtValue(true);
    HdRenderSettingDescriptor motion;
    motion.name = "Motion blur: shutter slices, 1 to 8 (rt)";
    motion.key = _atheneaSettings->motionBuckets;
    motion.defaultValue = VtValue(4);
    // What a cloud casts: a transmittance map at each light, read with no ray,
    // so a host can turn it off, sharpen it, resolve its depth or thin it out
    // from the same panel every other setting is in.
    HdRenderSettingDescriptor reflectCloud;
    reflectCloud.name = "rt: a cloud reflects itself, one ray a gaussian";
    reflectCloud.key = TfToken("athenea:splatReflections");
    reflectCloud.defaultValue = VtValue(false);
    HdRenderSettingDescriptor transferIndirect;
    transferIndirect.name = "A cloud's transfer adds its indirect half (the light it bounced)";
    transferIndirect.key = TfToken("athenea:splatTransferIndirect");
    transferIndirect.defaultValue = VtValue(true);
    HdRenderSettingDescriptor displayBlend;
    displayBlend.name = "Splats blended as a standard viewer blends them (sRGB values summed)";
    displayBlend.key = TfToken("athenea:splatDisplayBlend");
    displayBlend.defaultValue = VtValue(false);
    HdRenderSettingDescriptor splatShadows;
    splatShadows.name = "A relit cloud shadows itself, one ray a splat (rt)";
    splatShadows.key = TfToken("athenea:splatShadows");
    splatShadows.defaultValue = VtValue(false);
    HdRenderSettingDescriptor cloudShadows;
    cloudShadows.name = "Cloud shadows: a transmittance map at each light, no ray";
    cloudShadows.key = TfToken("athenea:cloudShadows");
    cloudShadows.defaultValue = VtValue(true);
    HdRenderSettingDescriptor domePrefiltered;
    domePrefiltered.name = "Raster: a dome lights a mesh prefiltered, no sample and no ray";
    domePrefiltered.key = TfToken("athenea:domePrefiltered");
    domePrefiltered.defaultValue = VtValue(true);
    HdRenderSettingDescriptor cloudTexels;
    cloudTexels.name = "Cloud shadow map: texels a side, per light";
    cloudTexels.key = TfToken("athenea:cloudShadowResolution");
    cloudTexels.defaultValue = VtValue(1024);
    HdRenderSettingDescriptor cloudTerms;
    cloudTerms.name = "Cloud shadow terms: 1 the total, 3/5/7 with Fourier pairs (0: by what receives)";
    cloudTerms.key = TfToken("athenea:cloudShadowTerms");
    cloudTerms.defaultValue = VtValue(0);
    HdRenderSettingDescriptor antialias;
    antialias.name = "Antialias a gathered frame (a sub-pixel offset a pass)";
    antialias.key = TfToken("athenea:antialias");
    antialias.defaultValue = VtValue(true);
    HdRenderSettingDescriptor cloudDensity;
    cloudDensity.name = "Cloud shadow density (1 is what the cloud's opacity says)";
    cloudDensity.key = TfToken("athenea:cloudShadowDensity");
    cloudDensity.defaultValue = VtValue(1.0f);
    return {technique, settle,      visibility,  samples,     choose,     paths,       bounces,
            total,     denoise,     adaptive,    error,       mis,        motion,      transferIndirect, displayBlend, reflectCloud, splatShadows,
            cloudShadows, cloudTexels, cloudTerms, cloudDensity, antialias, domePrefiltered};
}

athenea::usd::MeshVisibility HdAtheneaRenderDelegate::GetMeshVisibility() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->visibility);
    std::string name;
    if (value.IsHolding<TfToken>()) {
        name = value.UncheckedGet<TfToken>().GetString();
    } else if (value.IsHolding<std::string>()) {
        name = value.UncheckedGet<std::string>();
    }
    if (name == _atheneaSettings->raster.GetString()) return athenea::usd::MeshVisibility::Raster;
    if (name == _atheneaSettings->rays.GetString()) return athenea::usd::MeshVisibility::Rays;
    if (name == _atheneaSettings->bvh.GetString()) return athenea::usd::MeshVisibility::Bvh;
    return athenea::usd::MeshVisibility::Automatic;
}

bool HdAtheneaRenderDelegate::GetSplatShadows() const {
    const VtValue value = GetRenderSetting(TfToken("athenea:splatShadows"));
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

bool HdAtheneaRenderDelegate::GetCloudShadows() const {
    // On unless the stage says otherwise: a cloud that casts nothing on the
    // floor under it is wrong, and the map costs one pass a light.
    const VtValue value = GetRenderSetting(TfToken("athenea:cloudShadows"));
    return !value.IsHolding<bool>() || value.UncheckedGet<bool>();
}

bool HdAtheneaRenderDelegate::GetDomePrefiltered() const {
    // On unless the stage says otherwise: a sampled dome is grain and a ray a
    // pixel on every mesh under it.
    const VtValue value = GetRenderSetting(TfToken("athenea:domePrefiltered"));
    return !value.IsHolding<bool>() || value.UncheckedGet<bool>();
}

uint32_t HdAtheneaRenderDelegate::GetCloudShadowResolution() const {
    const VtValue value = GetRenderSetting(TfToken("athenea:cloudShadowResolution"));
    if (value.IsHolding<int>()) return static_cast<uint32_t>(std::max(value.UncheckedGet<int>(), 0));
    if (value.IsHolding<unsigned int>()) return value.UncheckedGet<unsigned int>();
    return 1024;
}

uint32_t HdAtheneaRenderDelegate::GetCloudShadowTerms() const {
    const VtValue value = GetRenderSetting(TfToken("athenea:cloudShadowTerms"));
    if (value.IsHolding<int>()) return static_cast<uint32_t>(std::max(value.UncheckedGet<int>(), 0));
    if (value.IsHolding<unsigned int>()) return value.UncheckedGet<unsigned int>();
    return 0;   // automatic
}

float HdAtheneaRenderDelegate::GetCloudShadowDensity() const {
    const VtValue value = GetRenderSetting(TfToken("athenea:cloudShadowDensity"));
    if (value.IsHolding<float>()) return value.UncheckedGet<float>();
    if (value.IsHolding<double>()) return static_cast<float>(value.UncheckedGet<double>());
    return 1.0F;
}

bool HdAtheneaRenderDelegate::GetAntialias() const {
    // On unless the stage says otherwise: a gathered frame costs nothing to
    // antialias, and a caller that needs two frames to agree pixel by pixel
    // is the one who has to say so.
    const VtValue value = GetRenderSetting(TfToken("athenea:antialias"));
    return !value.IsHolding<bool>() || value.UncheckedGet<bool>();
}

bool HdAtheneaRenderDelegate::GetChooseLights() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->chooseLights);
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

uint32_t HdAtheneaRenderDelegate::GetLightSamples() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->lightSamples);
    if (value.IsHolding<int>()) {
        return static_cast<uint32_t>(std::max(value.UncheckedGet<int>(), 1));
    }
    if (value.IsHolding<unsigned int>()) {
        return std::max(value.UncheckedGet<unsigned int>(), 1u);
    }
    return 1;
}

namespace {
/// An int render setting, however the host spelled its type.
uint32_t _UintSetting(const VtValue& value, uint32_t fallback, uint32_t least) {
    if (value.IsHolding<int>()) {
        return std::max(static_cast<uint32_t>(std::max(value.UncheckedGet<int>(), 0)), least);
    }
    if (value.IsHolding<unsigned int>()) {
        return std::max(value.UncheckedGet<unsigned int>(), least);
    }
    return fallback;
}
}   // namespace

uint32_t HdAtheneaRenderDelegate::GetPathSamples() const {
    return _UintSetting(GetRenderSetting(_atheneaSettings->pathSamples), 1, 1);
}

uint32_t HdAtheneaRenderDelegate::GetPathBounces() const {
    return _UintSetting(GetRenderSetting(_atheneaSettings->pathBounces), 1, 0);
}

uint32_t HdAtheneaRenderDelegate::GetMotionBuckets() const {
    return std::min(_UintSetting(GetRenderSetting(_atheneaSettings->motionBuckets), 4, 1), 8u);
}

bool HdAtheneaRenderDelegate::GetSplatReflections() const {
    const VtValue value = GetRenderSetting(TfToken("athenea:splatReflections"));
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

bool HdAtheneaRenderDelegate::GetSplatTransferIndirect() const {
    const VtValue value = GetRenderSetting(TfToken("athenea:splatTransferIndirect"));
    return !value.IsHolding<bool>() || value.UncheckedGet<bool>();
}

bool HdAtheneaRenderDelegate::GetSplatDisplayBlend() const {
    const VtValue value = GetRenderSetting(TfToken("athenea:splatDisplayBlend"));
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

bool HdAtheneaRenderDelegate::GetPathMis() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->pathMis);
    return !value.IsHolding<bool>() || value.UncheckedGet<bool>();
}

bool HdAtheneaRenderDelegate::GetPathAdaptive() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->pathAdaptive);
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

float HdAtheneaRenderDelegate::GetPathError() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->pathError);
    if (value.IsHolding<float>()) return value.UncheckedGet<float>();
    if (value.IsHolding<double>()) return static_cast<float>(value.UncheckedGet<double>());
    return 0.02F;
}

std::optional<std::pair<double, double>> HdAtheneaRenderDelegate::GetLensOverride() const {
    // The same reason as the shutter below: a camera the engine made for
    // itself carries a diaphragm nowhere Hydra can see, since a free camera is
    // two matrices. The pair is (lens radius in scene units, focus distance),
    // and it beats a stage camera's own, so the viewer's slider is not
    // silently overruled by the stage.
    const VtValue value = GetRenderSetting(TfToken("athenea:lens"));
    if (value.IsHolding<GfVec2d>()) {
        const GfVec2d v = value.UncheckedGet<GfVec2d>();
        if (v[0] > 0.0) {
            return std::make_pair(v[0], v[1]);
        }
    }
    return std::nullopt;
}

std::optional<std::pair<double, double>> HdAtheneaRenderDelegate::GetShutterOverride() const {
    // A camera the engine made for itself (`athenea stage --eye`, `athenea view`'s
    // free camera) authors no shutter, so a caller that wants motion blur out
    // of one has to say the shutter here. A stage camera's own is used when
    // this is not set.
    const VtValue value = GetRenderSetting(_atheneaSettings->shutter);
    if (value.IsHolding<GfVec2d>()) {
        const GfVec2d v = value.UncheckedGet<GfVec2d>();
        return std::make_pair(v[0], v[1]);
    }
    return std::nullopt;
}

bool HdAtheneaRenderDelegate::GetDisableMotionBlur() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->disableMotionBlur);
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

bool HdAtheneaRenderDelegate::GetDisableDepthOfField() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->disableDepthOfField);
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

bool HdAtheneaRenderDelegate::GetDenoise() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->denoise);
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

uint32_t HdAtheneaRenderDelegate::GetPathTotal() const {
    return _UintSetting(GetRenderSetting(_atheneaSettings->pathTotal), 1, 1);
}

bool HdAtheneaRenderDelegate::GetSettleStreams() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->settleStreams);
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

athenea::usd::Technique HdAtheneaRenderDelegate::GetTechnique() const {
    const VtValue value = GetRenderSetting(_atheneaSettings->technique);
    std::string name;
    if (value.IsHolding<TfToken>()) {
        name = value.UncheckedGet<TfToken>().GetString();
    } else if (value.IsHolding<std::string>()) {
        name = value.UncheckedGet<std::string>();
    }
    return name == _atheneaSettings->rt.GetString() ? athenea::usd::Technique::RayTraced
                                                : athenea::usd::Technique::Raster;
}

HdRprim* HdAtheneaRenderDelegate::CreateRprim(TfToken const& typeId, SdfPath const& id) {
    if (typeId == HdPrimTypeTokens->particleField) {
        return new HdAtheneaParticleField(id);
    }
    if (typeId == HdPrimTypeTokens->points) {
        return new HdAtheneaPoints(id);
    }
    if (typeId == HdPrimTypeTokens->basisCurves) {
        return new HdAtheneaBasisCurves(id);
    }
    if (typeId == HdPrimTypeTokens->mesh) {
        return new HdAtheneaMesh(id);
    }
    if (typeId == HdPrimTypeTokens->volume) {
        return new HdAtheneaVolume(id);
    }
    return nullptr;
}

void HdAtheneaRenderDelegate::DestroyRprim(HdRprim* rprim) { delete rprim; }

HdInstancer* HdAtheneaRenderDelegate::CreateInstancer(HdSceneDelegate* delegate, SdfPath const& id) {
    return new HdAtheneaInstancer(delegate, id);
}

void HdAtheneaRenderDelegate::DestroyInstancer(HdInstancer* instancer) { delete instancer; }

HdSprim* HdAtheneaRenderDelegate::CreateSprim(TfToken const& typeId, SdfPath const& id) {
    if (typeId == HdPrimTypeTokens->material) {
        return new HdAtheneaMaterial(id);
    }
    if (HdPrimTypeIsLight(typeId)) {
        return new HdAtheneaLight(typeId, id);
    }
    if (typeId == HdPrimTypeTokens->extComputation) {
        // Skinning's inputs travel as ext computation prims (usdSkelImaging);
        // the mesh reads them at Sync and the engine runs the computation
        // itself, on the device.
        return new HdExtComputation(id);
    }
    if (typeId == HdPrimTypeTokens->coordSys) {
        // A coordinate system a material may name: the prim hdsi made for
        // it carries the name and the transform; a mesh reads its bindings.
        return new HdCoordSys(id);
    }
    return typeId == HdPrimTypeTokens->camera ? new HdAtheneaCamera(id) : nullptr;
}

HdSprim* HdAtheneaRenderDelegate::CreateFallbackSprim(TfToken const& typeId) {
    if (typeId == HdPrimTypeTokens->material) {
        return new HdAtheneaMaterial(SdfPath::EmptyPath());
    }
    if (HdPrimTypeIsLight(typeId)) {
        return new HdAtheneaLight(typeId, SdfPath::EmptyPath());
    }
    if (typeId == HdPrimTypeTokens->extComputation) {
        return new HdExtComputation(SdfPath::EmptyPath());
    }
    if (typeId == HdPrimTypeTokens->coordSys) {
        return new HdCoordSys(SdfPath::EmptyPath());
    }
    return typeId == HdPrimTypeTokens->camera ? new HdAtheneaCamera(SdfPath::EmptyPath()) : nullptr;
}

void HdAtheneaRenderDelegate::DestroySprim(HdSprim* sprim) { delete sprim; }

HdBprim* HdAtheneaRenderDelegate::CreateBprim(TfToken const& typeId, SdfPath const& id) {
    if (typeId == HdPrimTypeTokens->renderSettings) {
        return new HdAtheneaRenderSettings(id);
    }
    if (typeId == TfToken("openvdbAsset")) {
        return new HdAtheneaVolumeField(id);
    }
    return typeId == HdPrimTypeTokens->renderBuffer ? new HdAtheneaRenderBuffer(id, _engine.get()) : nullptr;
}

HdBprim* HdAtheneaRenderDelegate::CreateFallbackBprim(TfToken const& typeId) {
    if (typeId == HdPrimTypeTokens->renderSettings) {
        return new HdAtheneaRenderSettings(SdfPath::EmptyPath());
    }
    if (typeId == TfToken("openvdbAsset")) {
        return new HdAtheneaVolumeField(SdfPath::EmptyPath());
    }
    return typeId == HdPrimTypeTokens->renderBuffer ? new HdAtheneaRenderBuffer(SdfPath::EmptyPath(), _engine.get())
                                                    : nullptr;
}

void HdAtheneaRenderDelegate::DestroyBprim(HdBprim* bprim) { delete bprim; }

HdAovDescriptor HdAtheneaRenderDelegate::GetDefaultAovDescriptor(TfToken const& name) const {
    // A Cryptomatte layer is (id, coverage, id, coverage) and has to stay
    // float32: half would round an id into another id's name.
    if (name.GetString().rfind("CryptoObject", 0) == 0) {
        return HdAovDescriptor(HdFormatFloat32Vec4, false, VtValue(GfVec4f(0.0F)));
    }
    if (name == HdAovTokens->color || name.GetString().rfind("lightGroup:", 0) == 0) {
        return HdAovDescriptor(HdFormatFloat32Vec4, false, VtValue(GfVec4f(0.0F)));
    }
    if (name == HdAovTokens->depth) {
        return HdAovDescriptor(HdFormatFloat32, false, VtValue(1.0F));
    }
    if (name == HdAovTokens->primId || name == HdAovTokens->instanceId || name == HdAovTokens->elementId) {
        return HdAovDescriptor(HdFormatInt32, false, VtValue(-1));
    }
    if (name == HdAovTokens->Neye || name == HdAovTokens->normal || name == TfToken("albedo") ||
        name == TfToken("shadingNormal") || name.GetString().rfind("primvars:", 0) == 0) {
        return HdAovDescriptor(HdFormatFloat32Vec3, false, VtValue(GfVec3f(0.0F)));
    }
    return HdAovDescriptor();
}

PXR_NAMESPACE_CLOSE_SCOPE
