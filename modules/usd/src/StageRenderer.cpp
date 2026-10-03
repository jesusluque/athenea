// Copyright (c) 2026 jesus luque.
#include "athenea/usd/StageRenderer.h"

#include "athenea/core/Log.h"

#include <map>
#include <mutex>
#include <algorithm>
#include <bit>
#include <optional>
#include <cmath>
#include <array>
#include <cstring>

#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdx/taskController.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/frustum.h>
#include <pxr/base/gf/range1f.h>
#include <pxr/imaging/cameraUtil/framing.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/variantSets.h>
#include <pxr/imaging/hd/primOriginSchema.h>
#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usd/editContext.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdHydra/renderPassAPI.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdRender/pass.h>
#include <pxr/usd/usdRender/settings.h>
#include <pxr/usd/usdVol/particleField3DGaussianSplat.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>

#include <pxr/imaging/hd/sceneIndexPluginRegistry.h>
#include <pxr/imaging/hd/renderSettings.h>
#include <pxr/imaging/hd/dependencyForwardingSceneIndex.h>
#include <pxr/imaging/hdsi/legacyDisplayStyleOverrideSceneIndex.h>
#include <pxr/imaging/hdsi/sceneGlobalsSceneIndex.h>
#include <pxr/base/tf/stringUtils.h>

#include "athenea/core/Hash.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/io/Exr.h"
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>

#include "BindingPurposes.h"
#include "RenderDelegate.h"
#include "RenderParam.h"
#include "RenderSettings.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace athenea::usd {

struct StageRenderer::Impl {
    UsdStageRefPtr                         stage;
    std::unique_ptr<HdAtheneaRenderDelegate>   delegate;
    HdsiLegacyDisplayStyleOverrideSceneIndexRefPtr displayStyle;
    HdsiSceneGlobalsSceneIndexRefPtr       globals;   ///< the active render settings prim, the frame
    HdAtheneaBindingPurposesSceneIndexRefPtr   bindingPurposes;   ///< render settings' materialBindingPurposes
    HdRenderIndex*                         index = nullptr;
    UsdImagingSceneIndices                 sceneIndices;
    std::unique_ptr<HdxTaskController>     controller;
    HdEngine                               engine;

    ~Impl() {
        controller.reset();
        delete index;
        delegate.reset();
    }
};

StageRenderer::StageRenderer() : impl_(std::make_unique<Impl>()) {}
StageRenderer::~StageRenderer() = default;

size_t StageRenderer::registerPlugins(const std::vector<std::filesystem::path>& directories) {
    std::vector<std::string> paths;
    for (const std::filesystem::path& directory : directories) {
        paths.push_back(directory.string());
    }
    return pxr::PlugRegistry::GetInstance().RegisterPlugins(paths).size();
}

Result<std::unique_ptr<StageRenderer>> StageRenderer::open(const std::filesystem::path& path,
                                                          std::shared_ptr<gpu::Device> device) {
    auto renderer = std::unique_ptr<StageRenderer>(new StageRenderer());
    Impl& impl = *renderer->impl_;
    impl.stage = UsdStage::Open(path.string());
    if (!impl.stage) {
        return Error::make(ErrorCode::IoFailure, "cannot open USD stage '{}'", path.string());
    }
    impl.delegate = std::make_unique<HdAtheneaRenderDelegate>(std::move(device));
    if (!impl.delegate->HasEngine()) {
        return Error::make(ErrorCode::DeviceFailure, "the render delegate has no GPU: {}",
                           impl.delegate->EngineError());
    }
    // Where this stage's relative texture paths are relative to.
    if (auto* param = static_cast<HdAtheneaRenderParam*>(impl.delegate->GetRenderParam()); param != nullptr) {
        if (auto* engine = param->GetEngine(); engine != nullptr) {
            engine->setAssetSearchPath(std::filesystem::absolute(path).parent_path());
        }
    }
    impl.index = HdRenderIndex::New(impl.delegate.get(), HdDriverVector());
    if (impl.index == nullptr) {
        return Error(ErrorCode::InternalError, "cannot make a Hydra render index");
    }
    // The chain is made empty and the stage given to it once the render
    // index observes it: a filtering scene index learns of a prim only from
    // the added notice that passes through it, and one made after the stage
    // had populated its input never hears of the prims already there. The
    // light linking scene index builds its collection cache from exactly
    // those notices, so with the stage given first every category came out
    // empty (docs/decisions.md, M5 and M9).
    // A BINDING NOBODY DECLARED IS STILL A BINDING.
    //
    // UsdShade has required `MaterialBindingAPI` to be applied since 21.11,
    // and the scene index path enforces it: a prim that authors
    // `material:binding` without the schema applied arrives at Hydra with no
    // binding at all, and every mesh of it draws its displayColor. Plenty of
    // assets in the wild do exactly that -- Sketchfab's usdz exporter among
    // them, so a car downloaded this morning came out uniformly grey while
    // usdview, which still has the legacy delegate's forgiving path, showed
    // it painted.
    //
    // So the schema is applied here, to the prims that already bind, IN THE
    // SESSION LAYER: the file on disk is not touched, and what changes is
    // only that Hydra now sees what the asset plainly says.
    {
        const UsdEditContext session(impl.stage, impl.stage->GetSessionLayer());
        size_t applied = 0;
        for (const UsdPrim& prim : impl.stage->TraverseAll()) {
            if (prim.HasAPI<UsdShadeMaterialBindingAPI>()) {
                continue;
            }
            // The vector is held: `GetAuthoredProperties` returns a new one
            // each call, and iterators from two of them do not compare.
            const std::vector<UsdProperty> properties = prim.GetAuthoredProperties();
            const bool binds = std::any_of(properties.begin(), properties.end(), [](const UsdProperty& property) {
                static const std::string kPrefix = "material:binding";
                const std::string& name = property.GetName().GetString();
                return name.rfind(kPrefix, 0) == 0 && (name.size() == kPrefix.size() || name[kPrefix.size()] == ':');
            });
            if (binds && UsdShadeMaterialBindingAPI::Apply(prim)) {
                ++applied;
            }
        }
        if (applied > 0) {
            athenea::log::info("hdAthenea: {} prim(s) bind a material without MaterialBindingAPI applied; applied in the "
                      "session layer so Hydra sees them",
                      applied);
        }
    }
    // THE HINTS A CLOUD CARRIES, SAID OUT LOUD. ParticleField3DGaussianSplat's
    // `projectionModeHint` and `sortingModeHint` are, the schema says, hints
    // a renderer is free to ignore, and Hydra does not carry them (usdVolImaging
    // 26.08 leaves them out of the prim's data source). They are read off the
    // stage here so a cloud trained for a projection or an order this engine
    // does not draw is at least named: the rasteriser projects in perspective
    // and sorts by view depth, the tracer meets particles along the ray.
    {
        size_t tangential = 0;
        size_t byDistance = 0;
        size_t byRay = 0;
        for (const UsdPrim& prim : impl.stage->Traverse()) {
            if (!prim.IsA<UsdVolParticleField3DGaussianSplat>()) {
                continue;
            }
            const UsdVolParticleField3DGaussianSplat cloud(prim);
            TfToken projection;
            TfToken sorting;
            cloud.GetProjectionModeHintAttr().Get(&projection);
            cloud.GetSortingModeHintAttr().Get(&sorting);
            tangential += projection == UsdVolTokens->tangential ? 1 : 0;
            byDistance += sorting == UsdVolTokens->cameraDistance ? 1 : 0;
            byRay += sorting == UsdVolTokens->rayHitDistance ? 1 : 0;
        }
        if (tangential > 0) {
            athenea::log::info("hdAthenea: {} cloud(s) hint a tangential projection; the rasteriser projects in perspective",
                           tangential);
        }
        if (byDistance > 0) {
            athenea::log::info("hdAthenea: {} cloud(s) hint sorting by camera distance; the rasteriser sorts by view depth",
                           byDistance);
        }
        if (byRay > 0) {
            athenea::log::info("hdAthenea: {} cloud(s) hint sorting by ray hit distance, which is the traced route's order; "
                           "the rasteriser sorts by view depth",
                           byRay);
        }
    }
    UsdImagingCreateSceneIndicesInfo info;
    impl.sceneIndices = UsdImagingCreateSceneIndices(info);
    // Through the filters registered for this renderer before the index sees
    // it: that is what resolves a light's collections into the categories
    // GetCategories then reports. Inserting the stage's own chain directly
    // skips every one of them.
    HdAtheneaRegisterSceneIndices();
    // The display style's fallback refine level, for setRefineLevel: what
    // usdview's complexity sets, the same way.
    impl.displayStyle = HdsiLegacyDisplayStyleOverrideSceneIndex::New(impl.sceneIndices.finalSceneIndex);
    // The scene's globals -- which render settings prim is active, which
    // frame -- are what a settings prim's `IsActive` and the products read.
    impl.globals = HdsiSceneGlobalsSceneIndex::New(impl.displayStyle);
    // The dependencies the scene indices declare (a settings prim's
    // `active` on the globals) become dirty notices only through a
    // forwarding scene index at the end of the chain.
    // Material bindings resolved by the purposes render settings name,
    // UsdRender's default until one does: "full", then the all-purpose one.
    impl.bindingPurposes = HdAtheneaBindingPurposesSceneIndex::New(
        impl.globals, {HdTokens->full, HdMaterialBindingsSchemaTokens->allPurpose});
    const HdSceneIndexBaseRefPtr scene = HdDependencyForwardingSceneIndex::New(
        HdSceneIndexPluginRegistry::GetInstance().AppendSceneIndicesForRenderer("athenea", impl.bindingPurposes));
    impl.index->InsertSceneIndex(scene, SdfPath::AbsoluteRootPath());
    impl.sceneIndices.stageSceneIndex->SetStage(impl.stage);
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();

    impl.controller = std::make_unique<HdxTaskController>(
        impl.index, SdfPath("/__atheneaTaskController"), /*gpuEnabled=*/false);
    impl.controller->SetRenderOutputs({HdAovTokens->color, HdAovTokens->depth});
    HdRprimCollection collection(HdTokens->geometry, HdReprSelector(HdReprTokens->smoothHull));
    collection.SetRootPath(SdfPath::AbsoluteRootPath());
    impl.controller->SetCollection(collection);
    return renderer;
}

/// A camera framing what the stage draws, as athenea view opens on a stage: a
/// small raster frame commits the scene, the engine's bounds of what it
/// drew (a kernel's) place the camera, from the same yaw and pitch.
Result<render::Camera> StageRenderer::framingCamera(double time, double focal, const std::string& technique) {
    const char axis = upAxis();
    const std::array<double, 3> up = axis == 'Z' ? std::array<double, 3>{0.0, 0.0, 1.0}
                                                 : std::array<double, 3>{0.0, 1.0, 0.0};
    const auto orbit = [&](const std::array<double, 3>& target, double distance, double radius) {
        const double yaw = 0.6;
        const double pitch = 0.35;
        const double cp = std::cos(pitch);
        const std::array<double, 3> away =
            axis == 'Z' ? std::array<double, 3>{cp * std::sin(yaw), -cp * std::cos(yaw), std::sin(pitch)}
                        : std::array<double, 3>{cp * std::sin(yaw), std::sin(pitch), cp * std::cos(yaw)};
        render::Camera camera = render::Camera::lookingAt(
            {target[0] + away[0] * distance, target[1] + away[1] * distance, target[2] + away[2] * distance},
            {target[0], target[1], target[2]}, {up[0], up[1], up[2]});
        camera.lens.focal = focal;
        camera.lens.nearZ = std::max(distance * 1e-3, 1e-4);
        camera.lens.farZ = distance + radius * 8.0 + 1.0;
        return camera;
    };
    ATHENEA_TRY(draw(orbit({0.0, 0.0, 0.0}, 10.0, 5.0), time, 64, 64, technique == "rt" ? "raster" : technique));
    auto found = bounds();
    if (!found) return std::move(found).error();
    const std::optional<scene::Bounds>& bounds = *found;
    if (!bounds) {
        return Error(ErrorCode::NotFound, "--frame-all: the stage draws nothing to frame");
    }
    std::array<double, 3> target{};
    double diagonal = 0.0;
    for (size_t k = 0; k < 3; ++k) {
        target[k] = 0.5 * (double(bounds->min[k]) + double(bounds->max[k]));
        const double d = double(bounds->max[k]) - double(bounds->min[k]);
        diagonal += d * d;
    }
    const double radius = std::max(0.5 * std::sqrt(diagonal), 1e-3);
    const double halfFov = std::atan(0.5 * 18.672 / focal);
    return orbit(target, radius / std::sin(halfFov) * 1.05, radius);
}

std::vector<std::string> StageRenderer::cameras() const {
    std::vector<std::string> out;
    for (const UsdPrim& prim : impl_->stage->Traverse()) {
        if (prim.IsA<UsdGeomCamera>()) {
            out.push_back(prim.GetPath().GetString());
        }
    }
    return out;
}

std::vector<StageVariantSet> StageRenderer::variantSets() const {
    std::vector<StageVariantSet> out;
    // Traverse() stops at instance boundaries and unloaded payloads, which is
    // right here: a set nobody can reach is a set nobody can choose from.
    for (const UsdPrim& prim : impl_->stage->Traverse()) {
        if (!prim.HasVariantSets()) {
            continue;
        }
        UsdVariantSets sets = prim.GetVariantSets();
        for (const std::string& name : sets.GetNames()) {
            StageVariantSet found;
            found.prim = prim.GetPath().GetString();
            found.name = name;
            found.variants = sets.GetVariantSet(name).GetVariantNames();
            found.selected = sets.GetVariantSelection(name);
            out.push_back(std::move(found));
        }
    }
    return out;
}

Result<void> StageRenderer::setVariantSelection(const std::string& prim, const std::string& set,
                                                const std::string& variant) {
    Impl& impl = *impl_;
    const UsdPrim carrier = impl.stage->GetPrimAtPath(SdfPath(prim));
    if (!carrier) {
        return Error(ErrorCode::NotFound, "no prim at " + prim);
    }
    UsdVariantSet chosen = carrier.GetVariantSets().GetVariantSet(set);
    if (!chosen) {
        return Error(ErrorCode::NotFound, prim + " has no variant set " + set);
    }
    const std::vector<std::string> names = chosen.GetVariantNames();
    if (std::find(names.begin(), names.end(), variant) == names.end()) {
        return Error(ErrorCode::InvalidArgument, set + " has no variant " + variant);
    }
    {
        // The session layer, as the default lights are: the selection is this
        // session's choice, not an edit of the asset.
        const UsdEditContext session(impl.stage, impl.stage->GetSessionLayer());
        if (!chosen.SetVariantSelection(variant)) {
            return Error(ErrorCode::InternalError, "cannot select " + variant + " of " + prim + "{" + set + "}");
        }
    }
    // A selection recomposes the prim and everything under it; the scene index
    // turns USD's notices into resyncs when it is told to catch up.
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    return ok();
}

Result<void> StageRenderer::setVariantSelection(const std::string& variantPath) {
    const SdfPath path(variantPath);
    if (!path.IsPrimVariantSelectionPath()) {
        return Error(ErrorCode::InvalidArgument,
                     variantPath + ": a variant selection is written /Prim/Path{set=variant}");
    }
    const std::pair<std::string, std::string> selection = path.GetVariantSelection();
    return setVariantSelection(path.GetPrimPath().GetString(), selection.first, selection.second);
}

namespace {

/// A dome, whichever of its schemas the stage used.
///
/// UsdLux has two: `DomeLight` and, since 24.11, `DomeLight_1`, which is the
/// one that carries `poleAxis`. They are different prim types and
/// `UsdLuxDomeLight` matches only the first, so a stage written the new way
/// would have shown no sky to change. Hydra folds both into one token and so
/// does this, by the name rather than by the class.
bool isDome(const UsdPrim& prim) {
    if (!prim.HasAPI<UsdLuxLightAPI>()) {
        return false;
    }
    const std::string type = prim.GetTypeName().GetString();
    return type == "DomeLight" || type.rfind("DomeLight_", 0) == 0;
}

/// How far a dome's image has to be tipped for its pole to be the stage's up.
///
/// A plain `DomeLight` keeps its image's pole on +Y whatever the stage says --
/// only `DomeLight_1` carries `poleAxis` -- so on a Z-up stage its sky lies on
/// its side: looking level, the frame shows the zenith. That is USD's reading
/// and Storm's, and a stage that authors the dome is left to it. A host that
/// swaps skies in wants the horizon level, so the swap tips the dome +90
/// degrees about X, the pole onto +Z; none for `DomeLight_1`, whose poleAxis
/// says, or a Y-up stage.
float domeTilt(const UsdStageRefPtr& stage, const UsdPrim& dome) {
    return UsdGeomGetStageUpAxis(stage) == UsdGeomTokens->z && dome.GetTypeName() == TfToken("DomeLight") ? 90.0F
                                                                                                           : 0.0F;
}

/// Where a dome's image is named, in both of them.
const TfToken& domeTextureToken() {
    static const TfToken token("inputs:texture:file");
    return token;
}

}   // namespace

namespace {

/// The engine behind a delegate, or nothing before the first frame made one.
Engine* engineBehind(HdAtheneaRenderDelegate& delegate) {
    auto* param = static_cast<HdAtheneaRenderParam*>(delegate.GetRenderParam());
    return param != nullptr ? param->GetEngine() : nullptr;
}

}   // namespace

Result<render::SplatIdReading> StageRenderer::measureSplatId(uint32_t cryptoId) {
    Engine* engine = engineBehind(*impl_->delegate);
    if (engine == nullptr) {
        return Error(ErrorCode::InternalError, "no engine yet: draw a frame first");
    }
    return engine->measureSplatId(cryptoId);
}

void StageRenderer::setSplatOverride(const render::SplatOverride& said) {
    if (Engine* engine = engineBehind(*impl_->delegate); engine != nullptr) {
        engine->setSplatOverride(said);
    }
}

void StageRenderer::clearSplatOverride(uint32_t cryptoId) {
    if (Engine* engine = engineBehind(*impl_->delegate); engine != nullptr) {
        engine->clearSplatOverride(cryptoId);
    }
}

void StageRenderer::clearSplatOverrides() {
    if (Engine* engine = engineBehind(*impl_->delegate); engine != nullptr) {
        engine->clearSplatOverrides();
    }
}

std::vector<render::SplatOverride> StageRenderer::splatOverrides() const {
    Engine* engine = engineBehind(*impl_->delegate);
    return engine != nullptr ? engine->splatOverrides() : std::vector<render::SplatOverride>{};
}

std::vector<StageDome> StageRenderer::domes() const {
    std::vector<StageDome> out;
    for (const UsdPrim& prim : impl_->stage->Traverse()) {
        if (!isDome(prim)) {
            continue;
        }
        StageDome found;
        found.prim = prim.GetPath().GetString();
        found.name = prim.GetName().GetString();
        SdfAssetPath asset;
        if (UsdAttribute file = prim.GetAttribute(domeTextureToken()); file && file.Get(&asset)) {
            // Resolved, because that is the path the texture store opens and
            // the one a host should show; a dome with no image resolves to
            // nothing and is a colour.
            found.texture = asset.GetResolvedPath().empty() ? asset.GetAssetPath() : asset.GetResolvedPath();
        }
        GfVec3f turn(0.0F);
        bool resets = false;
        for (const UsdGeomXformOp& op : UsdGeomXformable(prim).GetOrderedXformOps(&resets)) {
            if (op.GetOpType() == UsdGeomXformOp::TypeRotateXYZ && op.Get(&turn)) {
                break;
            }
        }
        found.rotation = UsdGeomGetStageUpAxis(impl_->stage) == UsdGeomTokens->z ? turn[2] : turn[1];
        out.push_back(std::move(found));
    }
    return out;
}

Result<void> StageRenderer::setDomeTexture(const std::string& prim, const std::string& texture) {
    Impl& impl = *impl_;
    const UsdPrim carrier = impl.stage->GetPrimAtPath(SdfPath(prim));
    if (!carrier || !isDome(carrier)) {
        return Error(ErrorCode::NotFound, "no dome light at " + prim);
    }
    {
        const UsdEditContext session(impl.stage, impl.stage->GetSessionLayer());
        UsdAttribute file = carrier.CreateAttribute(domeTextureToken(), SdfValueTypeNames->Asset);
        if (texture.empty()) {
            // Not a blank path but no opinion at all: what the stage authored
            // comes back, which is what "the stage's own sky" has to mean.
            if (!file.Clear()) {
                return Error(ErrorCode::InternalError, "cannot clear the sky on " + prim);
            }
        } else if (!file.Set(SdfAssetPath(texture))) {
            return Error(ErrorCode::InternalError, "cannot set the sky on " + prim);
        }
    }
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    // An image on a plain dome of a Z-up stage lies on its side (domeTilt):
    // a dome with no rotation of its own is given the tip with the image,
    // unturned; one that has a rotation keeps it, and a turn keeps the tip.
    if (!texture.empty() && domeTilt(impl.stage, carrier) != 0.0F) {
        bool resets = false;
        bool rotated = false;
        for (const UsdGeomXformOp& op : UsdGeomXformable(carrier).GetOrderedXformOps(&resets)) {
            rotated = rotated || op.GetOpType() == UsdGeomXformOp::TypeRotateXYZ;
        }
        if (!rotated) {
            return setDomeRotation(prim, 0.0F);
        }
    }
    return ok();
}

Result<void> StageRenderer::setShaderTexture(const std::string& shader, const std::string& input,
                                             const std::string& texture) {
    Impl& impl = *impl_;
    const UsdShadeShader node(impl.stage->GetPrimAtPath(SdfPath(shader)));
    if (!node) {
        return Error(ErrorCode::NotFound, "no shader at " + shader);
    }
    UsdShadeInput in = node.GetInput(TfToken(input));
    if (!in) {
        if (texture.empty()) {
            return ok();   // nothing authored, nothing to clear
        }
        return Error::make(ErrorCode::NotFound, "{} has no input '{}'", shader, input);
    }
    if (in.GetTypeName() != SdfValueTypeNames->Asset) {
        return Error::make(ErrorCode::InvalidArgument, "{}.inputs:{} is not a file", shader, input);
    }
    {
        const UsdEditContext session(impl.stage, impl.stage->GetSessionLayer());
        UsdAttribute attribute = in.GetAttr();
        if (texture.empty()) {
            if (!attribute.Clear()) {
                return Error(ErrorCode::InternalError, "cannot clear the texture on " + shader);
            }
        } else if (!attribute.Set(SdfAssetPath(texture))) {
            return Error(ErrorCode::InternalError, "cannot set the texture on " + shader);
        }
        // USD dirties a descendant shader's node as `nodes/<its name>`, but a
        // node inside a NodeGraph is keyed by its path under the material
        // (`NG_Wood/image`), so the retained network of the output's render
        // context -- MaterialX's `mtlx` -- survives the change. Re-authoring
        // the enclosing material's own output connections, with the same
        // targets, dirties the whole material instead.
        for (UsdPrim up = node.GetPrim().GetParent(); up; up = up.GetParent()) {
            if (const UsdShadeMaterial material{up}) {
                for (UsdShadeOutput& output : material.GetOutputs()) {
                    SdfPathVector targets;
                    UsdAttribute terminal = output.GetAttr();
                    if (terminal.GetConnections(&targets) && !targets.empty()) {
                        terminal.SetConnections(targets);
                    }
                }
                break;
            }
        }
    }

    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    return ok();
}

Result<void> StageRenderer::updateExternalTexture(const std::string& name, const gpu::Buffer& rgba,
                                                  uint32_t width, uint32_t height, uint32_t rowPixels) {
    if (!impl_->delegate->HasEngine()) {
        return Error(ErrorCode::DeviceFailure, "the render delegate has no GPU");
    }
    return impl_->delegate->GetEngine().updateExternalTexture(name, rgba, width, height, rowPixels);
}

Result<void> StageRenderer::setDomeRotation(const std::string& prim, float degrees) {
    Impl& impl = *impl_;
    const UsdPrim carrier = impl.stage->GetPrimAtPath(SdfPath(prim));
    if (!carrier || !isDome(carrier)) {
        return Error(ErrorCode::NotFound, "no dome light at " + prim);
    }
    const bool zUp = UsdGeomGetStageUpAxis(impl.stage) == UsdGeomTokens->z;
    // A turn is about the up axis only: what the op already says about the
    // other two -- a tilt the stage authored, or the one a swapped sky needs
    // (domeTilt) -- is kept, where writing the whole vector used to lay a
    // tipped sky back on its side.
    const auto turned = [&](GfVec3f was) {
        if (zUp) {
            was[2] = degrees;
        } else {
            was[1] = degrees;
        }
        return was;
    };
    {
        const UsdEditContext session(impl.stage, impl.stage->GetSessionLayer());
        UsdGeomXformable xform(carrier);
        // The op the stage already has, where it has one: adding a second
        // rotate to the order would turn the sky twice.
        bool resets = false;
        for (const UsdGeomXformOp& op : xform.GetOrderedXformOps(&resets)) {
            if (op.GetOpType() == UsdGeomXformOp::TypeRotateXYZ) {
                GfVec3f was(0.0F);
                op.Get(&was);
                if (!UsdGeomXformOp(op).Set(turned(was))) {
                    return Error(ErrorCode::InternalError, "cannot turn " + prim);
                }
                impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
                return ok();
            }
        }
        if (!xform.AddRotateXYZOp().Set(turned(GfVec3f(domeTilt(impl.stage, carrier), 0.0F, 0.0F)))) {
            return Error(ErrorCode::InternalError, "cannot turn " + prim);
        }
    }
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    return ok();
}

namespace {
const SdfPath& defaultLightsPath() {
    static const SdfPath path("/atheneaDefaultLights");
    return path;
}
}   // namespace

std::vector<StageLight> StageRenderer::lights() const {
    std::vector<StageLight> out;
    // Every prim, the inactive ones too: a light switched off is still one
    // a panel has to offer to switch back on.
    for (const UsdPrim& prim : impl_->stage->Traverse(UsdPrimAllPrimsPredicate)) {
        if (prim.GetPath().HasPrefix(defaultLightsPath()) || !prim.HasAPI<UsdLuxLightAPI>()) {
            continue;
        }
        StageLight found;
        found.prim = prim.GetPath().GetString();
        found.name = prim.GetName().GetString();
        found.type = prim.GetTypeName().GetString();
        found.on = prim.IsActive();
        out.push_back(std::move(found));
    }
    return out;
}

Result<void> StageRenderer::setLightOn(const std::string& prim, bool on) {
    Impl& impl = *impl_;
    const UsdPrim carrier = impl.stage->GetPrimAtPath(SdfPath(prim));
    if (!carrier || !carrier.HasAPI<UsdLuxLightAPI>()) {
        return Error(ErrorCode::NotFound, "no light at " + prim);
    }
    {
        const UsdEditContext session(impl.stage, impl.stage->GetSessionLayer());
        if (on) {
            // The session's opinion goes, and what the stage authored comes
            // back; only a light the file itself left off needs saying so.
            if (!carrier.ClearActive()) {
                return Error(ErrorCode::InternalError, "cannot switch on " + prim);
            }
            const UsdPrim again = impl.stage->GetPrimAtPath(SdfPath(prim));
            if (again && !again.IsActive() && !again.SetActive(true)) {
                return Error(ErrorCode::InternalError, "cannot switch on " + prim);
            }
        } else if (!carrier.SetActive(false)) {
            return Error(ErrorCode::InternalError, "cannot switch off " + prim);
        }
    }
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    return ok();
}

std::pair<double, double> StageRenderer::animationRange() const {
    const UsdStage& stage = *impl_->stage;
    const bool declared = stage.HasAuthoredTimeCodeRange();
    double first = 0.0;
    double last = 0.0;
    bool any = false;
    for (const UsdPrim& prim : impl_->stage->Traverse()) {
        for (const UsdAttribute& attr : prim.GetAuthoredAttributes()) {
            // Only the sample times, never a value: this walks a stage, and a
            // skinned cloud's samples are hundreds of matrices each.
            if (attr.GetNumTimeSamples() == 0) {
                continue;
            }
            std::vector<double> times;
            if (!attr.GetTimeSamples(&times) || times.empty()) {
                continue;
            }
            const double primFirst = times.front();
            const double primLast = times.back();
            first = any ? std::min(first, primFirst) : primFirst;
            last = any ? std::max(last, primLast) : primLast;
            any = true;
        }
    }
    if (!any) {
        return declared ? std::pair{stage.GetStartTimeCode(), stage.GetEndTimeCode()} : std::pair{0.0, 0.0};
    }
    if (declared) {
        first = std::max(first, stage.GetStartTimeCode());
        last = std::min(last, stage.GetEndTimeCode());
    }
    return {first, std::max(first, last)};
}

namespace {

/// A UsdRender purpose as Hydra's render tag.
TfToken tagOfPurpose(const std::string& purpose) {
    if (purpose == "default" || purpose.empty()) return HdRenderTagTokens->geometry;
    if (purpose == "render") return HdRenderTagTokens->render;
    if (purpose == "proxy") return HdRenderTagTokens->proxy;
    if (purpose == "guide") return HdRenderTagTokens->guide;
    return TfToken(purpose);
}

/// A var's source as the delegate's AOV: hd's names as they are, RenderMan's
/// "Ci" and "z" as colour and depth, and of light path expressions the one
/// that names a light group -- C.*<L.'NAME'> -- as "lightGroup:NAME".
Result<std::string> aovOfVar(const RenderVarInfo& var) {
    if (var.sourceType == "lpe") {
        const size_t open = var.sourceName.find("<L.'");
        const size_t close = open != std::string::npos ? var.sourceName.find("'>", open + 4) : std::string::npos;
        if (open != std::string::npos && close != std::string::npos && close > open + 4) {
            return "lightGroup:" + var.sourceName.substr(open + 4, close - open - 4);
        }
        return Error::make(ErrorCode::Unsupported,
                           "render var '{}': of light path expressions only a light group's (C.*<L.'NAME'>) is read",
                           var.name);
    }
    if (var.sourceType == "primvar") {
        return "primvars:" + var.sourceName;
    }
    if (var.sourceName == "Ci") return std::string("color");
    if (var.sourceName == "z") return std::string("depth");
    return var.sourceName;
}

std::string textOf(const VtValue& value) {
    if (value.IsHolding<std::string>()) return value.UncheckedGet<std::string>();
    if (value.IsHolding<TfToken>()) return value.UncheckedGet<TfToken>().GetString();
    if (value.IsHolding<bool>()) return value.UncheckedGet<bool>() ? "true" : "false";
    if (value.IsHolding<int>()) return std::to_string(value.UncheckedGet<int>());
    if (value.IsHolding<float>()) return TfStringify(value.UncheckedGet<float>());
    if (value.IsHolding<double>()) return TfStringify(value.UncheckedGet<double>());
    return TfStringify(value);
}

}   // namespace

void StageRenderer::setIncludedPurposes(const std::vector<std::string>& purposes) {
    TfTokenVector tags;
    if (purposes.empty()) {
        tags = {HdRenderTagTokens->geometry, HdRenderTagTokens->render};
    }
    for (const std::string& purpose : purposes) {
        const TfToken tag = tagOfPurpose(purpose);
        if (std::find(tags.begin(), tags.end(), tag) == tags.end()) {
            tags.push_back(tag);
        }
    }
    impl_->controller->SetRenderTags(tags);
}

void StageRenderer::setMaterialBindingPurposes(const std::vector<std::string>& purposes) {
    TfTokenVector tokens;
    for (const std::string& purpose : purposes) {
        tokens.emplace_back(purpose);   // "" is the all-purpose binding's name
    }
    if (tokens.empty()) {
        tokens = {HdTokens->full, HdMaterialBindingsSchemaTokens->allPurpose};
    }
    impl_->bindingPurposes->SetPurposes(tokens);
    impl_->sceneIndices.stageSceneIndex->ApplyPendingUpdates();
}

Result<RenderSettingsInfo> StageRenderer::renderSettings(const std::string& path) {
    Impl& impl = *impl_;
    const SdfPath id(path);
    if (!impl.stage->GetPrimAtPath(id).IsValid()) {
        return Error::make(ErrorCode::NotFound, "no prim at '{}'", path);
    }
    // Made the scene's active settings prim, then synced -- with no tasks,
    // so only the prims themselves, not a frame -- and read from the bprim.
    impl.globals->SetActiveRenderSettingsPrimPath(id);
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    // Made active, the prim's `active` is dirtied through its dependency on
    // the globals (the filtering scene index declares it, the dependency
    // forwarding one turns it into a notice, the emulation into a dirty
    // bit) and read again at Sync: with no tasks, only the prims sync.
    {
        HdTaskSharedPtrVector none;
        HdTaskContext context;
        impl.index->SyncAll(&none, &context);
    }
    const auto* prim = dynamic_cast<const HdAtheneaRenderSettings*>(impl.index->GetBprim(HdPrimTypeTokens->renderSettings, id));
    if (prim == nullptr) {
        return Error::make(ErrorCode::NotFound, "'{}' is not a render settings prim Hydra delivered", path);
    }
    RenderSettingsInfo info;
    info.path = path;
    info.active = prim->IsActive();
    info.syncs = prim->GetSyncCount();
    for (const TfToken& purpose : prim->GetIncludedPurposes()) info.includedPurposes.push_back(purpose.GetString());
    for (const TfToken& purpose : prim->GetMaterialBindingPurposes()) {
        info.materialBindingPurposes.push_back(purpose.GetString());
    }
    info.renderingColorSpace = prim->GetRenderingColorSpace().GetString();
    if (prim->GetCamera().IsHolding<SdfPath>()) {
        info.camera = prim->GetCamera().UncheckedGet<SdfPath>().GetString();
    }
    info.disableMotionBlur = prim->GetDisableMotionBlur();
    info.disableDepthOfField = prim->GetDisableDepthOfField();
    for (const auto& [key, value] : prim->GetNamespacedSettings()) {
        info.settings[key] = textOf(value);
    }
    // THE PASSES THAT RENDER FROM THESE SETTINGS, AND WHO THEY ARE FOR.
    // A RenderPass names its settings by `renderSource`, and since 26.08
    // `HydraRenderPassAPI` lets it name the renderer too (`hydra:rendererName`,
    // what usdrecord consults when nobody says --renderer). Read off the
    // stage, since the pass is a pipeline prim Hydra draws nothing for; a
    // pass meant for another renderer is reported, not silently drawn here.
    for (const UsdPrim& candidate : impl.stage->Traverse()) {
        if (!candidate.IsA<UsdRenderPass>()) {
            continue;
        }
        const UsdRenderPass pass(candidate);
        SdfPathVector sources;
        pass.GetRenderSourceRel().GetTargets(&sources);
        if (std::find(sources.begin(), sources.end(), id) == sources.end()) {
            continue;
        }
        RenderPassInfo one;
        one.path = candidate.GetPath().GetString();
        TfToken passType;
        pass.GetPassTypeAttr().Get(&passType);
        one.passType = passType.GetString();
        if (candidate.HasAPI<UsdHydraRenderPassAPI>()) {
            TfToken renderer;
            UsdHydraRenderPassAPI(candidate).GetHydraRendererNameAttr().Get(&renderer);
            one.rendererName = renderer.GetString();
        }
        one.forThisRenderer = one.rendererName.empty() || one.rendererName == "athenea" ||
                              one.rendererName == "HdAtheneaRendererPlugin";
        if (!one.forThisRenderer) {
            athenea::log::info("hdAthenea: render pass {} names the renderer '{}' (hydra:rendererName); this is athenea",
                           one.path, one.rendererName);
        }
        info.passes.push_back(std::move(one));
    }
    for (const HdRenderSettings::RenderProduct& product : prim->GetRenderProducts()) {
        RenderProductInfo p;
        p.path = product.productPath.GetString();
        p.name = product.name.GetString();
        p.type = product.type.GetString();
        p.width = static_cast<uint32_t>(std::max(product.resolution[0], 0));
        p.height = static_cast<uint32_t>(std::max(product.resolution[1], 0));
        p.camera = product.cameraPath.GetString();
        p.disableMotionBlur = product.disableMotionBlur;
        p.disableDepthOfField = product.disableDepthOfField;
        for (const HdRenderSettings::RenderProduct::RenderVar& var : product.renderVars) {
            p.vars.push_back({var.varPath.GetName(), var.sourceName, var.sourceType.GetString(), var.dataType.GetString()});
        }
        info.products.push_back(std::move(p));
    }
    return info;
}

Result<std::vector<std::filesystem::path>> StageRenderer::renderProducts(const std::string& path, double time,
                                                                         const std::filesystem::path& directory) {
    Impl& impl = *impl_;
    auto info = renderSettings(path);
    if (!info) return std::move(info).error();
    if (info->products.empty()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}' names no render products", path);
    }
    // The prim's own settings for this renderer, as the delegate's.
    const auto* prim =
        static_cast<const HdRenderSettings*>(impl.index->GetBprim(HdPrimTypeTokens->renderSettings, SdfPath(path)));
    std::string technique = "raster";
    bool half = false;
    for (const auto& [key, value] : prim->GetNamespacedSettings()) {
        if (key == "athenea:technique" && !textOf(value).empty()) {
            technique = textOf(value);
        } else if (key == "athenea:exrHalf") {
            half = value.IsHolding<bool>() && value.UncheckedGet<bool>();
        } else if (key.rfind("athenea:", 0) == 0) {
            impl.delegate->SetRenderSetting(TfToken(key), value);
        }
    }
    setIncludedPurposes(info->includedPurposes);
    setMaterialBindingPurposes(info->materialBindingPurposes);
    std::vector<std::filesystem::path> written;
    // The per-product switches below hold for the products alone, however
    // this returns.
    struct Switches {
        HdAtheneaRenderDelegate* delegate;
        ~Switches() {
            delegate->SetRenderSetting(TfToken("athenea:disableMotionBlur"), VtValue(false));
            delegate->SetRenderSetting(TfToken("athenea:disableDepthOfField"), VtValue(false));
        }
    } switches{impl.delegate.get()};
    for (const RenderProductInfo& product : info->products) {
        if (product.width == 0 || product.height == 0) {
            return Error::make(ErrorCode::InvalidArgument, "render product '{}' has no resolution", product.path);
        }
        if (product.vars.empty()) {
            return Error::make(ErrorCode::InvalidArgument, "render product '{}' has no vars", product.path);
        }
        std::vector<std::string> aovs;
        for (const RenderVarInfo& var : product.vars) {
            auto aov = aovOfVar(var);
            if (!aov) return std::move(aov).error();
            aovs.push_back(*aov);
        }
        requestOutputs(aovs);
        // The product's own switches, or its settings prim's, for this product.
        impl.delegate->SetRenderSetting(TfToken("athenea:disableMotionBlur"),
                                        VtValue(info->disableMotionBlur || product.disableMotionBlur));
        impl.delegate->SetRenderSetting(TfToken("athenea:disableDepthOfField"),
                                        VtValue(info->disableDepthOfField || product.disableDepthOfField));
        const std::string camera = !product.camera.empty() ? product.camera : info->camera;
        auto image = render(camera, time, product.width, product.height, technique);
        if (!image) return std::move(image).error();
        // Each var's layer, from the same Hydra buffer a host maps -- depth
        // as the view z the engine's own image carries, as `athenea stage -o`.
        std::vector<std::vector<uint32_t>> planes;
        std::vector<io::ExrChannel> channels;
        bool cryptoLayers = false;
        const size_t pixels = size_t{product.width} * product.height;
        for (size_t k = 0; k < product.vars.size(); ++k) {
            const RenderVarInfo& var = product.vars[k];
            const std::string& aov = aovs[k];
            if (aov == "depth") {
                planes.emplace_back(pixels);
                std::memcpy(planes.back().data(), image->depth.data(), pixels * 4);
                channels.push_back({var.name == "depth" ? "Z" : var.name, io::ExrChannelType::Float, {}});
                continue;
            }
            HdRenderBuffer* buffer = impl.controller->GetRenderOutput(TfToken(aov));
            if (buffer == nullptr) {
                return Error::make(ErrorCode::NotFound, "render var '{}': no output '{}'", var.name, aov);
            }
            auto bytes = mappedOutput(aov);
            if (!bytes) return std::move(bytes).error();
            const HdFormat format = buffer->GetFormat();
            const size_t components = HdGetComponentCount(format);
            const bool integer = HdGetComponentFormat(format) == HdFormatInt32;
            if (HdDataSizeOfFormat(format) != components * 4 || bytes->size() < pixels * components * 4) {
                return Error::make(ErrorCode::Unsupported, "render var '{}': output '{}' is not 32-bit", var.name, aov);
            }
            static constexpr const char* kColour[4] = {"R", "G", "B", "A"};
            static constexpr const char* kVector[4] = {"x", "y", "z", "w"};
            for (size_t c = 0; c < components; ++c) {
                planes.emplace_back(pixels);
                for (size_t p = 0; p < pixels; ++p) {
                    std::memcpy(&planes.back()[p], bytes->data() + (p * components + c) * 4, 4);
                }
                std::string name = var.name;
                if (components == 4) {
                    name = var.name == "color" ? std::string(kColour[c]) : var.name + "." + kColour[c];
                } else if (components > 1) {
                    name = var.name + "." + kVector[c];
                }
                // A Cryptomatte layer stays float whatever the product asked
                // for: an id rounded to half is another id's name.
                const bool crypto = aov.rfind("CryptoObject", 0) == 0;
                channels.push_back({name, integer   ? io::ExrChannelType::Uint
                                          : (half && !crypto) ? io::ExrChannelType::Half
                                                              : io::ExrChannelType::Float,
                                    {}});
                if (crypto) {
                    cryptoLayers = true;
                }
            }
        }
        for (size_t k = 0; k < channels.size(); ++k) {
            channels[k].words = planes[k];
        }
        std::filesystem::path file(product.name.empty() ? product.path.substr(1) + ".exr" : product.name);
        if (file.is_relative() && !directory.empty()) {
            file = directory / file;
        }
        // THE MANIFEST, where the product carries a matte: what a reader needs
        // to turn a pixel's numbers back into prim paths, as the Cryptomatte
        // specification's four string attributes per layer.
        std::vector<io::ExrAttribute> attributes;
        if (cryptoLayers) {
            const std::string key = core::cryptomatteLayerKey("CryptoObject");
            std::string manifest = "{";
            bool first = true;
            for (const auto& [name, id] : cryptoManifest()) {
                manifest += first ? "\"" : ",\"";
                manifest += name + "\":\"" + core::hex8(id) + "\"";
                first = false;
            }
            manifest += "}";
            const std::string prefix = "cryptomatte/" + key + "/";
            attributes.push_back(io::ExrAttribute::text(prefix + "name", "CryptoObject"));
            attributes.push_back(io::ExrAttribute::text(prefix + "hash", "MurmurHash3_32"));
            attributes.push_back(io::ExrAttribute::text(prefix + "conversion", "uint32_to_float32"));
            attributes.push_back(io::ExrAttribute::text(prefix + "manifest", manifest));
        }
        ATHENEA_TRY(io::writeExrChannels(file, product.width, product.height, channels, attributes));
        written.push_back(file);
    }
    return written;
}

uint64_t StageRenderer::cloudUploads() const {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->cloudUploads() : 0;
}

std::map<std::string, uint32_t> StageRenderer::cryptoManifest() const {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->cryptoManifest() : std::map<std::string, uint32_t>{};
}

StageRenderer::Counters StageRenderer::counters() const {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    if (engine == nullptr) {
        return {};
    }
    const FrameCounters& c = engine->frameCounters();
    return {c.splats, c.visibleSplats, c.splatPairs, c.meshInstances, c.lights, c.cryptomatte};
}

void StageRenderer::setGaussianStats(bool on) {
    if (Engine* engine = engineBehind(*impl_->delegate); engine != nullptr) {
        engine->setCountSplats(on);
    }
}

void StageRenderer::setTimeSplatStages(bool on) {
    if (Engine* engine = engineBehind(*impl_->delegate); engine != nullptr) {
        engine->setTimeSplatStages(on);
    }
}

GaussianStats StageRenderer::gaussianStats() const {
    const Engine* engine = engineBehind(*impl_->delegate);
    return engine != nullptr ? engine->gaussianStats() : GaussianStats{};
}

void StageRenderer::requestOutputs(const std::vector<std::string>& aovs) {
    TfTokenVector outputs{HdAovTokens->color, HdAovTokens->depth};
    for (const std::string& aov : aovs) {
        const TfToken token(aov);
        if (std::find(outputs.begin(), outputs.end(), token) == outputs.end()) {
            outputs.push_back(token);
        }
    }
    impl_->controller->SetRenderOutputs(outputs);
}

Result<std::vector<uint8_t>> StageRenderer::mappedOutput(const std::string& aov) {
    HdRenderBuffer* buffer = impl_->controller->GetRenderOutput(TfToken(aov));
    if (buffer == nullptr) {
        return Error::make(ErrorCode::NotFound, "no render output '{}'", aov);
    }
    buffer->Resolve();
    const size_t bytes = size_t{buffer->GetWidth()} * buffer->GetHeight() * HdDataSizeOfFormat(buffer->GetFormat());
    std::vector<uint8_t> out(bytes);
    const auto* mapped = static_cast<const uint8_t*>(buffer->Map());
    std::memcpy(out.data(), mapped, bytes);
    buffer->Unmap();
    return out;
}

Result<void> StageRenderer::setMeshVisibility(const std::string& route) {
    if (route != "automatic" && route != "raster" && route != "rays" && route != "bvh") {
        return Error::make(ErrorCode::InvalidArgument, "mesh visibility '{}': automatic, raster, rays or bvh", route);
    }
    impl_->delegate->SetRenderSetting(TfToken("athenea:visibility"), VtValue(TfToken(route)));
    return ok();
}

bool StageRenderer::hasLights() const {
    for (const UsdPrim& prim : impl_->stage->Traverse()) {
        if (prim.GetPath().HasPrefix(defaultLightsPath())) {
            continue;
        }
        if (prim.HasAPI<UsdLuxLightAPI>()) {
            return true;
        }
    }
    return false;
}

Result<void> StageRenderer::setDefaultLights(bool on) {
    Impl& impl = *impl_;
    const UsdEditContext session(impl.stage, impl.stage->GetSessionLayer());
    if (!on) {
        if (impl.stage->GetPrimAtPath(defaultLightsPath())) {
            impl.stage->RemovePrim(defaultLightsPath());
        }
        return ok();
    }
    if (impl.stage->GetPrimAtPath(defaultLightsPath())) {
        return ok();
    }
    impl.stage->DefinePrim(defaultLightsPath(), TfToken("Scope"));
    UsdLuxDomeLight sky = UsdLuxDomeLight::Define(impl.stage, defaultLightsPath().AppendChild(TfToken("Sky")));
    if (!sky) {
        return Error(ErrorCode::InternalError, "cannot define the default sky in the session layer");
    }
    sky.CreateIntensityAttr(VtValue(0.6F));
    UsdLuxDistantLight sun = UsdLuxDistantLight::Define(impl.stage, defaultLightsPath().AppendChild(TfToken("Sun")));
    if (!sun) {
        return Error(ErrorCode::InternalError, "cannot define the default sun in the session layer");
    }
    // Normalized, so 2.5 is the illuminance it lays rather than the radiance
    // of its two degree disc (UsdLux's normalize on a DistantLight).
    sun.CreateIntensityAttr(VtValue(2.5F));
    sun.CreateAngleAttr(VtValue(2.0F));
    sun.CreateNormalizeAttr(VtValue(true));
    // A distant light shines down its own -Z: tilted from overhead toward
    // the viewer's side, about whichever axis is up.
    const bool zUp = UsdGeomGetStageUpAxis(impl.stage) == UsdGeomTokens->z;
    UsdGeomXformable(sun.GetPrim()).AddRotateXYZOp().Set(zUp ? GfVec3f(35.0F, 0.0F, 30.0F) : GfVec3f(-55.0F, 30.0F, 0.0F));
    return ok();
}

void StageRenderer::setLightSamples(uint32_t samples) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:lightSamples"),
                                      VtValue(static_cast<int>(std::max(samples, 1u))));
}

void StageRenderer::setChooseLights(bool choose) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:chooseLights"), VtValue(choose));
}
void StageRenderer::setCloudShadows(bool shadows) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:cloudShadows"), VtValue(shadows));
}

void StageRenderer::setCloudShadowResolution(uint32_t texels) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:cloudShadowResolution"), VtValue(int(texels)));
}

void StageRenderer::setCloudShadowTerms(uint32_t terms) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:cloudShadowTerms"), VtValue(int(terms)));
}

void StageRenderer::setAntialias(bool on) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:antialias"), VtValue(on));
}

void StageRenderer::setCloudShadowDensity(float density) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:cloudShadowDensity"), VtValue(density));
}

void StageRenderer::setSplatTransferIndirect(bool indirect) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:splatTransferIndirect"), VtValue(indirect));
}

void StageRenderer::setSplatReflections(bool reflect) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:splatReflections"), VtValue(reflect));
}

void StageRenderer::setSplatShadows(bool shadows) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:splatShadows"), VtValue(shadows));
}
void StageRenderer::setPathSamples(uint32_t samples) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:pathSamples"),
                                      VtValue(static_cast<int>(std::max(samples, 1u))));
}
void StageRenderer::setPathBounces(uint32_t bounces) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:pathBounces"), VtValue(static_cast<int>(bounces)));
}
void StageRenderer::setRefineLevel(uint32_t level) {
    if (impl_->displayStyle) {
        impl_->displayStyle->SetRefineLevelFallback(level > 0 ? std::optional<int>(static_cast<int>(level))
                                                               : std::nullopt);
    }
}
void StageRenderer::setMotionBuckets(uint32_t buckets) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:motionBuckets"), VtValue(static_cast<int>(buckets)));
}
void StageRenderer::setShutter(double open, double close) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:shutter"), VtValue(GfVec2d(open, close)));
}

void StageRenderer::setPathMis(bool mis) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:pathMis"), VtValue(mis));
}

void StageRenderer::setPathAdaptive(bool adaptive) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:pathAdaptive"), VtValue(adaptive));
}
void StageRenderer::setPathError(float error) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:pathError"), VtValue(error));
}
void StageRenderer::setDenoise(bool denoise) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:denoise"), VtValue(denoise));
}
void StageRenderer::setPathTotal(uint32_t total) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:pathTotal"), VtValue(static_cast<int>(std::max(total, 1u))));
}
uint32_t StageRenderer::pathAccumulated() const {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->pathAccumulated() : 0;
}

uint64_t StageRenderer::meshGeneration() const {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->meshGeneration() : 0;
}

std::vector<CoordSysBinding> StageRenderer::coordSysBindings(const std::string& prim) const {
    const auto* param = static_cast<const HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->coordSysOf(SdfPath(prim)) : std::vector<CoordSysBinding>{};
}

uint64_t StageRenderer::meshPositionsRevision() const {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->meshPositionsRevision() : 0;
}
bool StageRenderer::pathConverged() const {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine == nullptr || engine->pathConverged();
}

double StageRenderer::timeCodesPerSecond() const {
    return impl_->stage->GetTimeCodesPerSecond();
}

double StageRenderer::startTimeCode() const {
    return impl_->stage->GetStartTimeCode();
}

Result<void> StageRenderer::aim(const std::string& camera, double time, const std::string& technique) {
    Impl& impl = *impl_;
    if (technique != "raster" && technique != "rt") {
        return Error::make(ErrorCode::InvalidArgument, "technique '{}': raster or rt", technique);
    }
    impl.delegate->SetRenderSetting(TfToken("athenea:technique"), VtValue(TfToken(technique)));
    std::string cameraPath = camera;
    if (cameraPath.empty()) {
        const auto all = cameras();
        if (all.empty()) {
            return Error(ErrorCode::NotFound, "the stage has no camera");
        }
        cameraPath = all.front();
    }
    if (!impl.stage->GetPrimAtPath(SdfPath(cameraPath)).IsA<UsdGeomCamera>()) {
        return Error::make(ErrorCode::NotFound, "no camera at '{}'", cameraPath);
    }
    // The camera's shutter, ahead of the first Sync, so the first frame
    // samples at it rather than the second.
    {
        const UsdGeomCamera usdCamera(impl.stage->GetPrimAtPath(SdfPath(cameraPath)));
        double open = 0.0;
        double close = 0.0;
        usdCamera.GetShutterOpenAttr().Get(&open, UsdTimeCode(time));
        usdCamera.GetShutterCloseAttr().Get(&close, UsdTimeCode(time));
        // An asked-for shutter wins over the camera's own: it is a control
        // somebody is holding (`athenea view`'s slider, `--shutter`), and a
        // control that silently loses to the stage is a broken control.
        if (const auto asked = impl.delegate->GetShutterOverride(); asked) {
            open = asked->first;
            close = asked->second;
        }
        if (auto* param = static_cast<HdAtheneaRenderParam*>(impl.delegate->GetRenderParam()); param != nullptr) {
            const bool changed = param->GetShutterOpen() != open || param->GetShutterClose() != close;
            param->SetShutter(open, close);
            // Everything synced about the old shutter samples again: every
            // prim's transform and primvars dirtied, which reaches meshes,
            // instancers, lights and the camera alike -- through the
            // delegate's own scene index, as a host's pass does it.
            if (changed && !impl.delegate->ResampleAllPrims()) {
                return Error(ErrorCode::InternalError, "the renderer's resampling scene index is not in the chain");
            }
        }
    }
    impl.sceneIndices.stageSceneIndex->SetTime(UsdTimeCode(time));
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    impl.controller->SetCameraPath(SdfPath(cameraPath));
    return ok();
}

Result<void> StageRenderer::aim(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                const std::string& technique) {
    Impl& impl = *impl_;
    if (technique != "raster" && technique != "rt") {
        return Error::make(ErrorCode::InvalidArgument, "technique '{}': raster or rt", technique);
    }
    impl.delegate->SetRenderSetting(TfToken("athenea:technique"), VtValue(TfToken(technique)));
    // THE STAGE'S ACTIVE RENDER SETTINGS SWITCH BLUR OFF FOR THIS CAMERA TOO.
    // `disableMotionBlur` and `disableDepthOfField` on the settings prim the
    // stage names (`renderSettingsPrimPath`) held for its products and for
    // nothing else; a camera of our own, with a shutter and a lens asked for
    // by `athenea:shutter` and `athenea:lens`, blurred whatever the stage said. The
    // products set the same two settings for themselves and put them back.
    {
        bool noBlur = false;
        bool noLens = false;
        if (const UsdRenderSettings active = UsdRenderSettings::GetStageRenderSettings(impl.stage); active) {
            active.GetDisableMotionBlurAttr().Get(&noBlur);
            active.GetDisableDepthOfFieldAttr().Get(&noLens);
        }
        impl.delegate->SetRenderSetting(TfToken("athenea:disableMotionBlur"), VtValue(noBlur));
        impl.delegate->SetRenderSetting(TfToken("athenea:disableDepthOfField"), VtValue(noLens));
    }
    // A camera of our own has no shutter to read, so an asked-for one is put
    // on the render param here, ahead of the first Sync, for the same reason
    // the stage camera's is above: otherwise the first frame samples at the
    // frame and only the second blurs.
    if (const auto asked = impl.delegate->GetShutterOverride(); asked && !impl.delegate->GetDisableMotionBlur()) {
        if (auto* param = static_cast<HdAtheneaRenderParam*>(impl.delegate->GetRenderParam()); param != nullptr) {
            const bool changed = param->GetShutterOpen() != asked->first ||
                                 param->GetShutterClose() != asked->second;
            param->SetShutter(asked->first, asked->second);
            if (changed && !impl.delegate->ResampleAllPrims()) {
                return Error(ErrorCode::InternalError, "the renderer's resampling scene index is not in the chain");
            }
        }
    }
    impl.sceneIndices.stageSceneIndex->SetTime(UsdTimeCode(time));
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    // A camera's matrices, as a free camera: world to camera, and the lens's
    // frustum (apertures in the image's aspect, so nothing is conformed).
    const render::Mat4 toCamera = aofx::xform::inverseAffine(camera.cameraToWorld);
    GfMatrix4d view;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            view[c][r] = toCamera.at(r, c);   // row-vector matrices are the transpose
        }
    }
    // The diaphragm, which a free camera's two matrices cannot carry: as a
    // render setting, in the units the projection uses (focalLength is in
    // tenths of a scene unit, as UsdGeomCamera has it).
    // In the stage's own units, which is what the caller cannot know: a
    // camera of our own is given an f-number, and what that is in scene units
    // depends on what a scene unit is.
    const double perUnit = std::max(UsdGeomGetStageMetersPerUnit(impl.stage), 1e-9);
    const double radius =
        camera.lens.fStop > 0.0 ? camera.lens.focal * 0.001 / perUnit / (2.0 * camera.lens.fStop) : 0.0;
    impl.delegate->SetRenderSetting(TfToken("athenea:lens"),
                                    VtValue(GfVec2d(radius, camera.lens.focusDistance)));
    GfCamera lens;
    lens.SetFocalLength(static_cast<float>(camera.lens.focal));
    lens.SetHorizontalAperture(static_cast<float>(camera.lens.haperture));
    lens.SetVerticalAperture(static_cast<float>(camera.lens.haperture * height / width));
    lens.SetClippingRange(GfRange1f(static_cast<float>(camera.lens.nearZ), static_cast<float>(camera.lens.farZ)));
    if (camera.lens.projection == render::Lens::Projection::Orthographic) {
        lens.SetProjection(GfCamera::Orthographic);
    }
    impl.controller->SetCameraPath(SdfPath());
    impl.controller->SetFreeCameraMatrices(view, lens.GetFrustum().ComputeProjectionMatrix());
    return ok();
}

Result<StageImage> StageRenderer::render(const std::string& camera, double time, uint32_t width,
                                         uint32_t height, const std::string& technique) {
    // An image, not a viewport: streamed assets are loaded before it is drawn.
    impl_->delegate->SetRenderSetting(TfToken("athenea:settleStreams"), VtValue(true));
    ATHENEA_TRY(aim(camera, time, technique));
    ATHENEA_TRY(executeUntilGathered(width, height));
    return readImage(width, height);
}

Result<BakedVisibilityArrays> StageRenderer::bakeVisibility(const std::string& prim,
                                                            const technique::VisibilityParts& parts,
                                                            const technique::VisibilityBakeOptions& options,
                                                            double time) {
    Impl& impl = *impl_;
    if (!impl.delegate->HasEngine()) {
        return Error(ErrorCode::DeviceFailure, "visibility: the render delegate has no GPU");
    }
    // On the device first, as a bake of its light is: one traced pixel.
    impl.delegate->SetRenderSetting(TfToken("athenea:settleStreams"), VtValue(true));
    auto framing = framingCamera(time, 35.0, "rt");
    if (!framing) return std::move(framing).error();
    ATHENEA_TRY(aim(*framing, time, 1, 1, "rt"));
    ATHENEA_TRY(execute(1, 1));
    athenea::usd::Engine& engine = impl.delegate->GetEngine();
    auto baked = engine.bakeVisibility(SdfPath(prim), parts, options);
    if (!baked) return std::move(baked).error();
    BakedVisibilityArrays out;
    out.parts = std::move(baked->parts);
    out.texels = std::move(baked->texels);
    out.partOf = std::move(baked->partOf);
    out.ambient = std::move(baked->ambient);
    out.partCount = baked->partCount;
    return out;
}

Result<std::vector<float>> StageRenderer::bakePoints(const std::vector<float>& rays, uint32_t count,
                                                     double time, uint32_t samples, uint32_t bounces,
                                                     uint32_t degree, bool transfer,
                                                     const std::vector<float>* facing) {
    Impl& impl = *impl_;
    if (count == 0 || rays.size() < size_t{count} * 8) {
        return Error(ErrorCode::InvalidArgument, "bake: two float4 a point, and at least one point");
    }
    if (facing != nullptr && facing->size() < size_t{count} * 4) {
        return Error(ErrorCode::InvalidArgument, "bake: a facing is four floats a point");
    }
    if (!impl.delegate->HasEngine()) {
        return Error(ErrorCode::DeviceFailure, "bake: the render delegate has no GPU");
    }
    gpu::Device& device = impl.delegate->GetEngine().device();
    // The kernel's layout, three float4 a point: the caller's two, and the
    // way the gaussian faces where it gave one (its w 1) -- a copy, nothing
    // computed.
    std::vector<float> laid(size_t{count} * 12, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        std::copy_n(rays.data() + size_t{k} * 8, 8, laid.data() + size_t{k} * 12);
        if (facing != nullptr) {
            std::copy_n(facing->data() + size_t{k} * 4, 4, laid.data() + size_t{k} * 12 + 8);
        }
    }
    gpu::BufferDesc desc;
    desc.bytes = laid.size() * sizeof(float);
    desc.elementBytes = 16;
    desc.label = "bake.rays";
    auto buffer = gpu::Buffer::create(device, desc, laid.data());
    if (!buffer) return std::move(buffer).error();
    auto answer = bakePointsOnDevice(*buffer, count, time, samples, bounces, degree, transfer);
    if (!answer) return std::move(answer).error();
    // Already a point's entries together: read back as it is.
    return answer->readAll<float>(device);
}

Result<gpu::Buffer> StageRenderer::bakePointsOnDevice(const gpu::Buffer& rays, uint32_t count, double time,
                                                      uint32_t samples, uint32_t bounces, uint32_t degree,
                                                      bool transfer, uint32_t batch) {
    Impl& impl = *impl_;
    if (count == 0 || !rays.valid() || rays.bytes() < uint64_t{count} * 48) {
        return Error(ErrorCode::InvalidArgument, "bake: three float4 a point, and at least one point");
    }
    if (!impl.delegate->HasEngine()) {
        return Error(ErrorCode::DeviceFailure, "bake: the render delegate has no GPU");
    }
    // The stage has to be on the device before its light can be asked for,
    // and what puts it there is a frame: one pixel of it, path traced, with
    // the streams settled so nothing is still on its way in.
    impl.delegate->SetRenderSetting(TfToken("athenea:settleStreams"), VtValue(true));
    auto framing = framingCamera(time, 35.0, "rt");
    if (!framing) return std::move(framing).error();
    ATHENEA_TRY(aim(*framing, time, 1, 1, "rt"));
    ATHENEA_TRY(execute(1, 1));

    athenea::usd::Engine& engine = impl.delegate->GetEngine();
    gpu::Device& device = engine.device();
    auto gather = gpu::ComputeKernel::create(engine.library(), "athenea/usd/bake_gather", "bakeGather");
    if (!gather) return std::move(gather).error();

    const uint32_t coefficients = (std::min(degree, 3u) + 1) * (std::min(degree, 3u) + 1);
    // A transfer writes two planes more: the others' alpha is the direct half
    // of the transfer, so the coverage travels on its own, and after it the
    // sixty-four visibility bits, carried as the floats they are the bits of.
    const uint32_t entries = coefficients + (transfer ? 2u : 0u);
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{count} * entries * 16;
    desc.elementBytes = 16;
    desc.label = "bake.answer";
    auto answer = gpu::Buffer::create(device, desc);
    if (!answer) return std::move(answer).error();

    render::RenderSettings settings;
    settings.width = 1;
    settings.height = 1;
    const render::Projection projection = render::projectionFor(*framing, 1, 1);
    render::RenderTargets out;
    // IN PASSES: a pass's rays are the caller's buffer where one pass holds
    // them all, and otherwise copied out of it on the device into one buffer
    // a pass long; the pass's planes are gathered into the answer at its
    // place. What the tracer allocates is sized by the pass.
    const uint32_t perPass = std::min(batch > 0 ? batch : kBakeBatch, count);
    gpu::Buffer passRays;
    if (perPass < count) {
        desc.bytes = uint64_t{perPass} * 48;
        desc.elementBytes = 16;
        desc.label = "bake.passRays";
        auto made = gpu::Buffer::create(device, desc);
        if (!made) return std::move(made).error();
        passRays = std::move(*made);
    }
    for (uint32_t first = 0; first < count; first += perPass) {
        const uint32_t n = std::min(perPass, count - first);
        if (passRays.valid()) {
            gpu::CommandBatch copy(device);
            copy.encoder()->copyBuffer(passRays.rhi(), 0, rays.rhi(), uint64_t{first} * 48, uint64_t{n} * 48);
            copy.markDirty();
            ATHENEA_TRY(copy.submit(true));
        }
        athenea::usd::BakeRequest bake;
        bake.rays = passRays.valid() ? &passRays : &rays;
        bake.count = n;
        bake.samples = samples;
        bake.bounces = bounces;
        bake.coefficients = coefficients;
        bake.transfer = transfer;
        bake.out = &out;
        ATHENEA_TRY(engine.bakePoints(bake, projection, settings));
        if (!out.colour.valid()) {
            return Error(ErrorCode::InternalError, "bake: the frame wrote nothing");
        }
        // The kernel writes a plane an entry over the pass's grid; what the
        // caller wants is a point's entries together, laid out on the device.
        gpu::CommandBatch laid(device);
        gather->dispatch(laid, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["planes"].setBinding(out.colour.rhi());
            cursor["answer"].setBinding(answer->rhi());
            cursor["params"]["count"].setData(n);
            cursor["params"]["first"].setData(first);
            cursor["params"]["entries"].setData(entries);
            cursor["params"]["plane"].setData(out.width * out.height);
        });
        ATHENEA_TRY(laid.submit(true));
    }
    return std::move(*answer);
}

Result<void> allotBakePasses(gpu::ShaderLibrary& library, const gpu::Buffer& sums, uint32_t count,
                             uint32_t coefficients, const BakeOptions& options, gpu::Buffer& allot) {
    if (count == 0 || !sums.valid() || !allot.valid() || allot.bytes() < uint64_t{count} * 4) {
        return Error(ErrorCode::InvalidArgument, "bake: no sums to weigh, or no room for the allotment");
    }
    if (options.passSamples == 0) {
        return Error(ErrorCode::InvalidArgument, "bake: an extra pass of no paths");
    }
    gpu::Device& device = library.device();
    const auto kernel = [&](const char* entry) {
        return gpu::ComputeKernel::create(library, "athenea/usd/bake_resolve", entry);
    };
    auto weigh = kernel("bakeWeigh");
    if (!weigh) return std::move(weigh).error();
    auto total = kernel("bakeWeighTotal");
    if (!total) return std::move(total).error();
    auto share = kernel("bakeAllot");
    if (!share) return std::move(share).error();
    const uint32_t groups = (count + 255) / 256;
    const auto make = [&](uint64_t elements, uint32_t elementBytes, const char* label) -> Result<gpu::Buffer> {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(elements, 4) * elementBytes;
        desc.elementBytes = elementBytes;
        desc.label = label;
        return gpu::Buffer::create(device, desc);
    };
    auto weights = make(count, 4, "bake.weights");
    if (!weights) return std::move(weights).error();
    auto partials = make(uint64_t{groups} + 1, 4, "bake.partials");
    if (!partials) return std::move(partials).error();
    // The budget in whole passes: a count, not a measurement.
    const float budgetPasses =
        static_cast<float>(static_cast<double>(options.extraSamples) * count / options.passSamples);
    const auto bind = [&](rhi::ShaderCursor cursor) {
        for (const char* name : {"planes", "rays", "passRays", "slots", "counter", "direct", "indirect", "guides",
                                 "answer"}) {
            cursor[name].setBinding(weights->rhi());
        }
        cursor["sums"].setBinding(sums.rhi());
        cursor["weights"].setBinding(weights->rhi());
        cursor["partials"].setBinding(partials->rhi());
        cursor["allot"].setBinding(allot.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["coefficients"].setData(coefficients);
        cursor["params"]["maxPasses"].setData(options.maxPasses);
        cursor["params"]["budgetPasses"].setData(budgetPasses);
        cursor["params"]["meanFloor"].setData(options.meanFloor);
    };
    gpu::CommandBatch batch(device);
    weigh->dispatch(batch, {groups * 256, 1, 1}, bind);
    total->dispatch(batch, {256, 1, 1}, bind);
    share->dispatch(batch, {count, 1, 1}, bind);
    return batch.submit(true);
}

Result<BakeSplit> StageRenderer::bakeSplitOnDevice(const gpu::Buffer& rays, uint32_t count, double time,
                                                   const BakeOptions& options) {
    Impl& impl = *impl_;
    if (count == 0 || !rays.valid() || rays.bytes() < uint64_t{count} * 48) {
        return Error(ErrorCode::InvalidArgument, "bake: three float4 a point, and at least one point");
    }
    if (!impl.delegate->HasEngine()) {
        return Error(ErrorCode::DeviceFailure, "bake: the render delegate has no GPU");
    }
    // The stage on the device first, as bakePointsOnDevice puts it there.
    impl.delegate->SetRenderSetting(TfToken("athenea:settleStreams"), VtValue(true));
    auto framing = framingCamera(time, 35.0, "rt");
    if (!framing) return std::move(framing).error();
    ATHENEA_TRY(aim(*framing, time, 1, 1, "rt"));
    ATHENEA_TRY(execute(1, 1));

    athenea::usd::Engine& engine = impl.delegate->GetEngine();
    gpu::Device& device = engine.device();
    gpu::ShaderLibrary& library = engine.library();
    const auto kernel = [&](const char* entry) {
        return gpu::ComputeKernel::create(library, "athenea/usd/bake_resolve", entry);
    };
    auto sumsAdd = kernel("bakeSumsAdd");
    if (!sumsAdd) return std::move(sumsAdd).error();
    auto select = kernel("bakeSelect");
    if (!select) return std::move(select).error();
    auto splitFit = kernel("bakeSplitFit");
    if (!splitFit) return std::move(splitFit).error();

    const uint32_t degree = std::min(options.degree, 3u);
    const uint32_t coefficients = (degree + 1) * (degree + 1);
    const uint32_t entries = 2 * coefficients + 3;
    const auto make = [&](uint64_t elements, uint32_t elementBytes, const char* label) -> Result<gpu::Buffer> {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(elements, 1) * elementBytes;
        desc.elementBytes = elementBytes;
        desc.label = label;
        return gpu::Buffer::create(device, desc);
    };
    BakeSplit split;
    split.count = count;
    split.coefficients = coefficients;
    auto sums = make(uint64_t{count} * entries, 16, "bake.sums");
    if (!sums) return std::move(sums).error();
    split.sums = std::move(*sums);
    auto allot = make(count, 4, "bake.allot");
    if (!allot) return std::move(allot).error();
    split.allot = std::move(*allot);
    {
        gpu::CommandBatch clear(device);
        clear.encoder()->clearBuffer(split.allot.rhi(), 0, uint64_t{count} * 4);
        clear.markDirty();
        ATHENEA_TRY(clear.submit(true));
    }
    // What every kernel here declares is bound whatever it uses: a stand-in
    // for the buffers a kernel has no use for.
    auto none = make(4, 16, "bake.none");
    if (!none) return std::move(none).error();
    const uint32_t perPass = std::min(options.batch > 0 ? options.batch : kBakeBatch, count);
    auto passRays = make(uint64_t{perPass} * 3, 16, "bake.passRays");
    if (!passRays) return std::move(passRays).error();

    render::RenderSettings settings;
    settings.width = 1;
    settings.height = 1;
    const render::Projection projection = render::projectionFor(*framing, 1, 1);
    render::RenderTargets out;
    // Every buffer the module declares is bound for every kernel, the ones a
    // kernel does not touch to a stand-in; the callers bind over them.
    const auto bindAll = [&](rhi::ShaderCursor cursor, const gpu::Buffer& planes, const gpu::Buffer& slots,
                             const gpu::Buffer& packed) {
        cursor["planes"].setBinding(planes.rhi());
        cursor["sums"].setBinding(split.sums.rhi());
        cursor["rays"].setBinding(rays.rhi());
        cursor["passRays"].setBinding(packed.rhi());
        cursor["slots"].setBinding(slots.rhi());
        cursor["allot"].setBinding(split.allot.rhi());
        for (const char* name : {"weights", "partials", "counter", "direct", "indirect", "guides", "answer"}) {
            cursor[name].setBinding(none->rhi());
        }
        cursor["params"]["coefficients"].setData(coefficients);
    };
    // ONE PASS of `samples` paths at the points `from` holds (the cloud's own
    // rays, or a packed selection of them), in batches, its sums written at
    // their gaussians or added to them.
    const auto tracePass = [&](const gpu::Buffer& from, uint32_t points, uint32_t samples, uint32_t seed,
                               bool accumulate, const gpu::Buffer* slots) -> Result<void> {
        for (uint32_t first = 0; first < points; first += perPass) {
            const uint32_t n = std::min(perPass, points - first);
            const gpu::Buffer* batchRays = &from;
            if (n < points || slots != nullptr) {
                gpu::CommandBatch copy(device);
                copy.encoder()->copyBuffer(passRays->rhi(), 0, from.rhi(), uint64_t{first} * 48, uint64_t{n} * 48);
                copy.markDirty();
                ATHENEA_TRY(copy.submit(true));
                batchRays = &*passRays;
            }
            athenea::usd::BakeRequest bake;
            bake.rays = batchRays;
            bake.count = n;
            bake.samples = samples;
            bake.bounces = options.bounces;
            bake.coefficients = coefficients;
            bake.split = true;
            // A batch draws its own paths too: the batch's grid restarts at
            // zero, and the seed is what keeps it from repeating the first.
            bake.seed = seed + first * 31u;
            bake.out = &out;
            ATHENEA_TRY(engine.bakePoints(bake, projection, settings));
            if (!out.colour.valid()) {
                return Error(ErrorCode::InternalError, "bake: the frame wrote nothing");
            }
            gpu::CommandBatch add(device);
            sumsAdd->dispatch(add, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                bindAll(cursor, out.colour, slots != nullptr ? *slots : *none, *none);
                cursor["params"]["count"].setData(n);
                cursor["params"]["first"].setData(first);
                cursor["params"]["plane"].setData(out.width * out.height);
                cursor["params"]["indexed"].setData(slots != nullptr ? 1u : 0u);
                cursor["params"]["accumulate"].setData(accumulate ? 1u : 0u);
            });
            ATHENEA_TRY(add.submit(true));
        }
        return ok();
    };
    ATHENEA_TRY(tracePass(rays, count, std::max(options.samples, 1u), options.seed, false, nullptr));

    if (options.extraSamples > 0 && options.passSamples > 0 && options.maxPasses > 0) {
        // WHERE THE REST OF THE BUDGET GOES. Each gaussian weighed from its
        // first pass, the weights summed, and its share in whole passes.
        ATHENEA_TRY(allotBakePasses(library, split.sums, count, coefficients, options, split.allot));
        auto slots = make(count, 4, "bake.slots");
        if (!slots) return std::move(slots).error();
        auto packed = make(uint64_t{count} * 3, 16, "bake.selectedRays");
        if (!packed) return std::move(packed).error();
        auto counter = make(4, 4, "bake.selected");
        if (!counter) return std::move(counter).error();
        const auto bindSelect = [&](rhi::ShaderCursor cursor) {
            bindAll(cursor, *none, *slots, *packed);
            cursor["counter"].setBinding(counter->rhi());
            cursor["params"]["count"].setData(count);
        };
        for (uint32_t pass = 1; pass <= options.maxPasses; ++pass) {
            const uint32_t zero[4] = {0, 0, 0, 0};
            ATHENEA_TRY(counter->write(device, 0, sizeof(zero), zero));
            gpu::CommandBatch batch(device);
            select->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                bindSelect(cursor);
                cursor["params"]["pass"].setData(pass);
            });
            ATHENEA_TRY(batch.submit(true));
            uint32_t selected = 0;
            ATHENEA_TRY(counter->read(device, 0, sizeof(selected), &selected));
            if (selected == 0) {
                break;   // allotted no further: every later pass would be empty too
            }
            ATHENEA_TRY(tracePass(*packed, selected, options.passSamples, options.seed + pass * 7919u, true,
                                  &*slots));
            split.extraPoints += selected;
            split.extraPasses = pass;
        }
    }

    // Each half fitted over every path its gaussian took.
    auto direct = make(uint64_t{count} * coefficients, 16, "bake.direct");
    if (!direct) return std::move(direct).error();
    auto indirect = make(uint64_t{count} * coefficients, 16, "bake.indirect");
    if (!indirect) return std::move(indirect).error();
    auto guides = make(count, 16, "bake.guides");
    if (!guides) return std::move(guides).error();
    split.direct = std::move(*direct);
    split.indirect = std::move(*indirect);
    split.guides = std::move(*guides);
    gpu::CommandBatch fit(device);
    splitFit->dispatch(fit, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bindAll(cursor, *none, *none, *none);
        cursor["direct"].setBinding(split.direct.rhi());
        cursor["indirect"].setBinding(split.indirect.rhi());
        cursor["guides"].setBinding(split.guides.rhi());
        cursor["params"]["count"].setData(count);
    });
    ATHENEA_TRY(fit.submit(true));
    return split;
}

Result<gpu::Buffer> StageRenderer::combineBake(const BakeSplit& split, const gpu::Buffer& rays) {
    Impl& impl = *impl_;
    if (!impl.delegate->HasEngine()) {
        return Error(ErrorCode::DeviceFailure, "bake: the render delegate has no GPU");
    }
    if (split.count == 0 || !split.sums.valid() || !split.direct.valid() || !split.indirect.valid()) {
        return Error(ErrorCode::InvalidArgument, "bake: no split bake to combine");
    }
    athenea::usd::Engine& engine = impl.delegate->GetEngine();
    gpu::Device& device = engine.device();
    auto combine = gpu::ComputeKernel::create(engine.library(), "athenea/usd/bake_resolve", "bakeCombine");
    if (!combine) return std::move(combine).error();
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{split.count} * split.coefficients * 16;
    desc.elementBytes = 16;
    desc.label = "bake.answer";
    auto answer = gpu::Buffer::create(device, desc);
    if (!answer) return std::move(answer).error();
    gpu::CommandBatch batch(device);
    combine->dispatch(batch, {split.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        // Every buffer the module declares, the ones this kernel does not
        // touch to the allotment, which is there whatever the bake was.
        for (const char* name : {"planes", "passRays", "slots", "weights", "partials", "counter", "guides"}) {
            cursor[name].setBinding(split.allot.rhi());
        }
        cursor["allot"].setBinding(split.allot.rhi());
        cursor["sums"].setBinding(split.sums.rhi());
        cursor["rays"].setBinding(rays.rhi());
        cursor["direct"].setBinding(split.direct.rhi());
        cursor["indirect"].setBinding(split.indirect.rhi());
        cursor["answer"].setBinding(answer->rhi());
        cursor["params"]["count"].setData(split.count);
        cursor["params"]["coefficients"].setData(split.coefficients);
    });
    ATHENEA_TRY(batch.submit(true));
    return std::move(*answer);
}

Result<StageImage> StageRenderer::render(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                         const std::string& technique) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:settleStreams"), VtValue(true));
    ATHENEA_TRY(aim(camera, time, width, height, technique));
    ATHENEA_TRY(executeUntilGathered(width, height));
    return readImage(width, height);
}

Result<void> StageRenderer::executeUntilGathered(uint32_t width, uint32_t height) {
    // An image is drawn until the path traced frame holds what it was asked
    // for: the pass reports itself unconverged until then, and a frame that is
    // not path traced is whole at once. The cap is against a total no pass
    // count could reach.
    for (uint32_t pass = 0; pass < 65536; ++pass) {
        ATHENEA_TRY(execute(width, height));
        if (pathConverged()) {
            return ok();
        }
    }
    return Error(ErrorCode::InternalError, "the path traced frame did not gather its total in 65536 passes");
}

Result<void> StageRenderer::draw(const std::string& camera, double time, uint32_t width, uint32_t height,
                                 const std::string& technique) {
    // A viewport: streamed assets fill in over the frames that follow.
    impl_->delegate->SetRenderSetting(TfToken("athenea:settleStreams"), VtValue(false));
    ATHENEA_TRY(aim(camera, time, technique));
    return execute(width, height);
}

Result<void> StageRenderer::draw(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                 const std::string& technique) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:settleStreams"), VtValue(false));
    ATHENEA_TRY(aim(camera, time, width, height, technique));
    return execute(width, height);
}

Result<void> StageRenderer::drawImage(const std::string& camera, double time, uint32_t width,
                                      uint32_t height, const std::string& technique) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:settleStreams"), VtValue(true));
    ATHENEA_TRY(aim(camera, time, technique));
    return executeUntilGathered(width, height);
}

Result<void> StageRenderer::drawImage(const render::Camera& camera, double time, uint32_t width,
                                      uint32_t height, const std::string& technique) {
    impl_->delegate->SetRenderSetting(TfToken("athenea:settleStreams"), VtValue(true));
    ATHENEA_TRY(aim(camera, time, width, height, technique));
    return executeUntilGathered(width, height);
}

Result<void> StageRenderer::execute(uint32_t width, uint32_t height) {
    Impl& impl = *impl_;
    impl.controller->SetRenderBufferSize(GfVec2i(static_cast<int>(width), static_cast<int>(height)));
    impl.controller->SetFraming(CameraUtilFraming(
        GfRect2i(GfVec2i(0), static_cast<int>(width), static_cast<int>(height))));
    HdTaskSharedPtrVector tasks = impl.controller->GetRenderingTasks();
    impl.engine.Execute(impl.index, &tasks);
    // What the render pass met and Hydra has no way to return: the device out
    // of memory, after the pass had given back what it could and tried again.
    if (auto* param = static_cast<HdAtheneaRenderParam*>(impl.delegate->GetRenderParam()); param != nullptr) {
        if (Engine* engine = param->GetEngine(); engine != nullptr) {
            if (std::optional<Error> failed = engine->takeFrameError(); failed.has_value()) {
                return std::move(*failed);
            }
        }
    }
    const render::RenderTargets* targets = lastTargets();
    if (targets == nullptr || targets->width != width || targets->height != height) {
        return Error(ErrorCode::InternalError, "the render pass drew nothing");
    }
    return ok();
}

StageRenderer::MemoryRelief StageRenderer::memoryRelief() const {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    MemoryRelief relief;
    if (engine != nullptr) {
        relief.times = engine->reliefs();
        relief.last = engine->lastRelief();
        relief.lodBias = engine->lodBias();
    }
    return relief;
}

std::string StageRenderer::relieveMemory() {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->relieveMemory() : std::string();
}

const render::RenderTargets* StageRenderer::lastTargets() const {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->lastTargets() : nullptr;
}

Result<StageImage> StageRenderer::readImage(uint32_t width, uint32_t height) {
    // The engine's own targets, as the render pass left them: bottom row
    // first and view z already, so the only step is the readback.
    const render::RenderTargets* targets = lastTargets();
    if (targets == nullptr || targets->width != width || targets->height != height) {
        return Error(ErrorCode::InternalError, "the render pass drew nothing to read");
    }
    StageImage image;
    image.width = width;
    image.height = height;
    auto rgba = targets->colour.readAll<float>(impl_->delegate->GetEngineDevice());
    if (!rgba) return std::move(rgba).error();
    auto depth = targets->depth.readAll<float>(impl_->delegate->GetEngineDevice());
    if (!depth) return std::move(depth).error();
    image.rgba = std::move(*rgba);
    image.depth = std::move(*depth);
    image.rgba.resize(size_t{width} * height * 4);
    image.depth.resize(size_t{width} * height);
    return image;
}

gpu::Device& StageRenderer::device() {
    return impl_->delegate->GetEngineDevice();
}

gpu::ShaderLibrary& StageRenderer::library() {
    return impl_->delegate->GetEngineLibrary();
}

Result<technique::DisplaySource> StageRenderer::displaySource(const std::string& aov) {
    const render::RenderTargets* targets = lastTargets();
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    if (targets == nullptr || engine == nullptr) {
        return Error(ErrorCode::InvalidArgument, "nothing drawn yet");
    }
    AovSource source;
    technique::DisplaySource shown;
    if (aov == "color") {
        source.kind = AovKind::Colour;
        shown.kind = technique::DisplaySource::Kind::Colour;
    } else if (aov == "depth") {
        source.kind = AovKind::Depth;
        shown.kind = technique::DisplaySource::Kind::Depth;
    } else if (aov == "primId" || aov == "instanceId" || aov == "elementId") {
        source.kind = aov == "primId" ? AovKind::PrimId : aov == "instanceId" ? AovKind::InstanceId : AovKind::ElementId;
        shown.kind = technique::DisplaySource::Kind::Ids;
    } else if (aov == "cryptomatte") {
        // The matte itself, previewed: every id a colour of its own, mixed by
        // what it covers. The view reads all three layers, so it takes the
        // buffer whole and the layer offset stays zero.
        source.kind = AovKind::Crypto;
        shown.kind = technique::DisplaySource::Kind::Crypto;
    } else if (aov.rfind("CryptoObject", 0) == 0 && aov.size() == 14 && aov[12] == '0' &&
               aov[13] >= '0' && aov[13] <= '2') {
        source.kind = AovKind::Crypto;
        source.primvar = static_cast<uint32_t>(aov[13] - '0');
        shown.kind = technique::DisplaySource::Kind::Colour;
    } else if (aov == "Neye" || aov == "normal") {
        source.kind = aov == "Neye" ? AovKind::EyeNormal : AovKind::WorldNormal;
        shown.kind = technique::DisplaySource::Kind::Vector;
    } else {
        return Error::make(ErrorCode::InvalidArgument, "no display for AOV '{}'", aov);
    }
    const AovView view = engine->aovView(*targets, source);
    shown.buffer = view.buffer;
    shown.stride = view.stride;
    shown.offset = view.offset;
    shown.width = targets->width;
    shown.height = targets->height;
    shown.bottomRowFirst = true;
    return shown;
}

Result<std::optional<scene::Bounds>> StageRenderer::bounds() {
    auto* param = static_cast<HdAtheneaRenderParam*>(impl_->delegate->GetRenderParam());
    Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    if (engine == nullptr) {
        return std::optional<scene::Bounds>{};
    }
    return engine->bounds();
}

Result<std::optional<StagePick>> StageRenderer::pick(uint32_t x, uint32_t y) {
    Impl& impl = *impl_;
    const render::RenderTargets* targets = lastTargets();
    auto* param = static_cast<HdAtheneaRenderParam*>(impl.delegate->GetRenderParam());
    Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    if (targets == nullptr || engine == nullptr || x >= targets->width || y >= targets->height) {
        return std::optional<StagePick>{};
    }
    // Bookkeeping, not an image read back: one pixel's ids, and the nearest
    // rank of the matte where the frame kept one.
    const uint64_t pixel = uint64_t{targets->height - 1 - y} * targets->width + x;   // bottom row first
    StagePick picked;
    if (targets->crypto.valid()) {
        // The first layer's first rank: (id, coverage) in its first two floats.
        float rank[2] = {0.0F, 0.0F};
        ATHENEA_TRY(targets->crypto.read(impl.delegate->GetEngineDevice(), pixel * 16, sizeof(rank), rank));
        const uint32_t id = std::bit_cast<uint32_t>(rank[0]);
        if (rank[1] > 0.0F && id != 0) {
            picked.cryptoId = id;
            picked.cryptoCoverage = rank[1];
            for (const auto& [name, named] : engine->cryptoManifest()) {
                if (named == id) {
                    picked.cryptoName = name;
                    break;
                }
            }
        }
    }
    const AovView view = engine->aovView(*targets, {AovKind::PrimId, 0});
    uint32_t ids[2] = {0xFFFFFFFFu, 0xFFFFFFFFu};
    if (view.buffer != nullptr) {
        ATHENEA_TRY(view.buffer->read(impl.delegate->GetEngineDevice(), pixel * view.stride * 4, sizeof(ids), ids));
    }
    // A cloud writes no `primId`, so a pixel of gaussians is named by the
    // matte alone -- which is the whole of what it has to say.
    if (ids[0] == 0xFFFFFFFFu) {
        if (picked.cryptoId == 0) {
            return std::optional<StagePick>{};
        }
        picked.prim = picked.cryptoName;
        return std::optional<StagePick>(std::move(picked));
    }
    const SdfPath rprim = impl.index->GetRprimPathFromPrimId(static_cast<int>(ids[0]));
    if (rprim.IsEmpty()) {
        return picked.cryptoId != 0 ? std::optional<StagePick>(std::move(picked)) : std::optional<StagePick>{};
    }
    picked.rprim = rprim.GetString();
    picked.instance = static_cast<int32_t>(ids[1]);
    const HdSceneIndexPrim prim = impl.index->GetTerminalSceneIndex()->GetPrim(rprim);
    const SdfPath origin = HdPrimOriginSchema::GetFromParent(prim.dataSource).GetOriginPath(HdPrimOriginSchemaTokens->scenePath);
    picked.prim = origin.IsEmpty() ? picked.rprim : origin.GetString();
    return std::optional<StagePick>(std::move(picked));
}

namespace {

/// A prim's children as rows, for `children` and `outline` alike.
std::vector<StagePrim> childrenOf(const UsdStageRefPtr& stage, const std::string& path) {
    std::vector<StagePrim> out;
    const UsdPrim parent = path.empty() || path == "/" ? stage->GetPseudoRoot()
                                                       : stage->GetPrimAtPath(SdfPath(path));
    if (!parent) {
        return out;
    }
    for (const UsdPrim& child : parent.GetChildren()) {
        StagePrim row;
        row.path = child.GetPath().GetString();
        row.name = child.GetName().GetString();
        row.type = child.GetTypeName().GetString();
        row.hasChildren = !child.GetChildren().empty();
        row.instance = child.IsInstance();
        out.push_back(std::move(row));
    }
    return out;
}

/// Stages opened for `outline` and `stageCameras`, by path, until the file
/// changes. A mutex because a tree view and a render may ask at once.
struct Outlined {
    std::filesystem::file_time_type when;
    UsdStageRefPtr                  stage;
};

Result<UsdStageRefPtr> outlineStage(const std::filesystem::path& path) {
    static std::mutex                      guard;
    static std::map<std::string, Outlined> opened;
    std::error_code                        failed;
    const auto when = std::filesystem::last_write_time(path, failed);
    if (failed) {
        return Error::make(ErrorCode::IoFailure, "cannot read USD stage '{}'", path.string());
    }
    const std::lock_guard<std::mutex> held(guard);
    const std::string                 key = std::filesystem::absolute(path).string();
    if (const auto found = opened.find(key); found != opened.end() && found->second.when == when) {
        return found->second.stage;
    }
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    if (!stage) {
        return Error::make(ErrorCode::IoFailure, "cannot open USD stage '{}'", path.string());
    }
    opened[key] = Outlined{when, stage};
    return stage;
}

}   // namespace

void registerPlugins(const std::filesystem::path& directory) {
    StageRenderer::registerPlugins({directory});
}

Result<std::vector<StagePrim>> outline(const std::filesystem::path& stage, const std::string& prim) {
    auto opened = outlineStage(stage);
    if (!opened) {
        return std::move(opened).error();
    }
    return childrenOf(*opened, prim);
}

Result<std::vector<std::string>> stageCameras(const std::filesystem::path& stage) {
    auto opened = outlineStage(stage);
    if (!opened) {
        return std::move(opened).error();
    }
    std::vector<std::string> out;
    for (const UsdPrim& prim : (*opened)->Traverse()) {
        if (prim.IsA<UsdGeomCamera>()) {
            out.push_back(prim.GetPath().GetString());
        }
    }
    return out;
}

std::vector<StagePrim> StageRenderer::children(const std::string& path) const {
    return childrenOf(impl_->stage, path);
}

Result<void> StageRenderer::setPrimVisible(const std::string& prim, bool visible) {
    Impl& impl = *impl_;
    const UsdPrim target = impl.stage->GetPrimAtPath(SdfPath(prim));
    if (!target) {
        return Error(ErrorCode::NotFound, "no prim at " + prim);
    }
    const UsdGeomImageable imageable(target);
    if (!imageable) {
        return Error(ErrorCode::InvalidArgument, prim + " is not something that can be seen");
    }
    {
        const UsdEditContext session(impl.stage, impl.stage->GetSessionLayer());
        UsdAttribute visibility = imageable.GetVisibilityAttr();
        if (visible) {
            // No opinion rather than `inherited`: what the stage authored
            // comes back, including a prim it made invisible itself.
            if (visibility && !visibility.Clear()) {
                return Error(ErrorCode::InternalError, "cannot show " + prim);
            }
        } else if (!imageable.CreateVisibilityAttr().Set(UsdGeomTokens->invisible)) {
            return Error(ErrorCode::InternalError, "cannot hide " + prim);
        }
    }
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    // What the change does not dirty itself: an instancer's visibility is
    // the instancer's, and the prototypes it hides read it only when their
    // own visibility is synced (HdAtheneaInstancersVisible). Every prim beneath
    // is dirtied through the delegate's scene index -- the change tracker's
    // marks do not reach prims a scene index owns.
    impl.delegate->DirtyVisibilityBelow(SdfPath(prim));
    return ok();
}

char StageRenderer::upAxis() const {
    return UsdGeomGetStageUpAxis(impl_->stage) == UsdGeomTokens->z ? 'Z' : 'Y';
}

double StageRenderer::endTimeCode() const {
    return impl_->stage->GetEndTimeCode();
}

}   // namespace athenea::usd
