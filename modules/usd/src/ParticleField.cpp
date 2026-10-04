// Copyright (c) 2026 jesus luque.
#include "ParticleField.h"
#include "Instancer.h"

#include "athenea/core/Log.h"

#include <algorithm>
#include <array>
#include <limits>

#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/gf/quath.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdVol/tokens.h>

#include "RenderParam.h"
#include "athenea/usd/HydraCamera.h"
#include "athenea/usd/PrimData.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// AtheneaSplatEditAPI's constant primvars (modules/usd/schemas). Constant primvars
// are inherited down the namespace, so an edit authored on an Xform stands
// over every cloud below it -- openFXplayer's Edit node over its subtree.
TF_DEFINE_PRIVATE_TOKENS(_editTokens,
    ((active, "athenea:edit:active"))
    ((shape, "athenea:edit:shape"))
    ((mode, "athenea:edit:mode"))
    ((centre, "athenea:edit:centre"))
    ((size, "athenea:edit:size"))
    ((tint, "athenea:edit:tint"))
    ((saturation, "athenea:edit:saturation"))
    ((brightness, "athenea:edit:brightness"))
    ((opacity, "athenea:edit:opacity"))
    ((minOpacity, "athenea:edit:minOpacity"))
    ((maxScale, "athenea:edit:maxScale"))
    ((invert, "athenea:edit:invert"))
    (sphere)(keep)(remove)
);

// AtheneaStreamedAssetAPI's.
TF_DEFINE_PRIVATE_TOKENS(_assetTokens,
    ((asset, "athenea:asset"))
    ((threshold, "athenea:lod:threshold"))
    ((budget, "athenea:stream:budget"))
);

float floatOf(VtValue const& value, float fallback) {
    if (value.IsHolding<float>()) return value.UncheckedGet<float>();
    if (value.IsHolding<double>()) return static_cast<float>(value.UncheckedGet<double>());
    if (value.IsHolding<VtFloatArray>() && !value.UncheckedGet<VtFloatArray>().empty())
        return value.UncheckedGet<VtFloatArray>()[0];
    return fallback;
}

bool boolOf(VtValue const& value, bool fallback) {
    if (value.IsHolding<bool>()) return value.UncheckedGet<bool>();
    if (value.IsHolding<VtBoolArray>() && !value.UncheckedGet<VtBoolArray>().empty())
        return value.UncheckedGet<VtBoolArray>()[0];
    return fallback;
}

std::array<float, 3> vec3Of(VtValue const& value, std::array<float, 3> fallback) {
    if (value.IsHolding<GfVec3f>()) {
        const GfVec3f v = value.UncheckedGet<GfVec3f>();
        return {v[0], v[1], v[2]};
    }
    if (value.IsHolding<GfVec3d>()) {
        const GfVec3d v = value.UncheckedGet<GfVec3d>();
        return {static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])};
    }
    if (value.IsHolding<VtVec3fArray>() && !value.UncheckedGet<VtVec3fArray>().empty()) {
        const GfVec3f v = value.UncheckedGet<VtVec3fArray>()[0];
        return {v[0], v[1], v[2]};
    }
    return fallback;
}

TfToken tokenOf(VtValue const& value) {
    if (value.IsHolding<TfToken>()) return value.UncheckedGet<TfToken>();
    if (value.IsHolding<VtTokenArray>() && !value.UncheckedGet<VtTokenArray>().empty())
        return value.UncheckedGet<VtTokenArray>()[0];
    if (value.IsHolding<std::string>()) return TfToken(value.UncheckedGet<std::string>());
    return TfToken();
}

athenea::render::SplatEdit editOf(HdSceneDelegate* delegate, SdfPath const& id) {
    using Edit = athenea::render::SplatEdit;
    Edit edit;
    edit.active = boolOf(delegate->Get(id, _editTokens->active), false);
    if (!edit.active) {
        return edit;
    }
    edit.shape = tokenOf(delegate->Get(id, _editTokens->shape)) == _editTokens->sphere ? Edit::Shape::Sphere
                                                                                       : Edit::Shape::Box;
    const TfToken mode = tokenOf(delegate->Get(id, _editTokens->mode));
    edit.mode = mode == _editTokens->keep ? Edit::Mode::Keep
              : mode == _editTokens->remove ? Edit::Mode::Remove
                                            : Edit::Mode::Grade;
    edit.centre = vec3Of(delegate->Get(id, _editTokens->centre), edit.centre);
    edit.size = vec3Of(delegate->Get(id, _editTokens->size), edit.size);
    edit.tint = vec3Of(delegate->Get(id, _editTokens->tint), edit.tint);
    edit.saturation = floatOf(delegate->Get(id, _editTokens->saturation), edit.saturation);
    edit.brightness = floatOf(delegate->Get(id, _editTokens->brightness), edit.brightness);
    edit.opacity = floatOf(delegate->Get(id, _editTokens->opacity), edit.opacity);
    edit.minOpacity = floatOf(delegate->Get(id, _editTokens->minOpacity), edit.minOpacity);
    edit.maxScale = floatOf(delegate->Get(id, _editTokens->maxScale), edit.maxScale);
    edit.invert = boolOf(delegate->Get(id, _editTokens->invert), false);
    return edit;
}

/// AtheneaSplatLightingAPI: whether this cloud is relit rather than shown as it
/// was baked.
bool relightOf(HdSceneDelegate* delegate, SdfPath const& id) {
    static const TfToken kRelight("athenea:splat:relight");
    return boolOf(delegate->Get(id, kRelight), false);
}

/// Whether its colours are light already: what a conversion bakes into them
/// (`athenea mesh2splat`), so that relighting adds the polish and nothing else.
bool litBodyOf(HdSceneDelegate* delegate, SdfPath const& id) {
    static const TfToken kLit("athenea:splat:litBody");
    return boolOf(delegate->Get(id, kLit), false);
}

/// The index its transmitting gaussians bend by: 0, and nothing bends.
float iorOf(HdSceneDelegate* delegate, SdfPath const& id) {
    static const TfToken kIor("athenea:splat:ior");
    return floatOf(delegate->Get(id, kIor), 0.0F);
}

athenea::usd::StreamedAsset assetOf(HdSceneDelegate* delegate, SdfPath const& id) {
    athenea::usd::StreamedAsset asset;
    VtValue value = delegate->Get(id, _assetTokens->asset);
    SdfAssetPath path;
    if (value.IsHolding<SdfAssetPath>()) {
        path = value.UncheckedGet<SdfAssetPath>();
    } else if (value.IsHolding<VtArray<SdfAssetPath>>() && !value.UncheckedGet<VtArray<SdfAssetPath>>().empty()) {
        path = value.UncheckedGet<VtArray<SdfAssetPath>>()[0];
    }
    asset.path = !path.GetResolvedPath().empty() ? path.GetResolvedPath() : path.GetAssetPath();
    if (asset.path.empty()) {
        return asset;
    }
    asset.threshold = std::max(floatOf(delegate->Get(id, _assetTokens->threshold), asset.threshold), 0.0F);
    const VtValue budget = delegate->Get(id, _assetTokens->budget);
    if (budget.IsHolding<int64_t>()) {
        asset.budget = static_cast<uint64_t>(std::max<int64_t>(budget.UncheckedGet<int64_t>(), 0));
    } else if (budget.IsHolding<int>()) {
        asset.budget = static_cast<uint64_t>(std::max(budget.UncheckedGet<int>(), 0));
    } else if (budget.IsHolding<VtInt64Array>() && !budget.UncheckedGet<VtInt64Array>().empty()) {
        asset.budget = static_cast<uint64_t>(std::max<int64_t>(budget.UncheckedGet<VtInt64Array>()[0], 0));
    }
    return asset;
}


}   // namespace

HdDirtyBits HdAtheneaParticleField::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::Clean | HdChangeTracker::InitRepr | HdChangeTracker::DirtyPoints |
           HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyTransform |
           HdChangeTracker::DirtyVisibility;
}

void HdAtheneaParticleField::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam,
                              HdDirtyBits* dirtyBits, TfToken const&) {
    SdfPath const& id = GetId();
    auto* param = static_cast<HdAtheneaRenderParam*>(renderParam);
    auto* engine = param->GetEngine();
    if (engine == nullptr) {
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        return;
    }

    std::optional<athenea::usd::ParticleFieldArrays> raw;
    // AN ARRAY THAT DOES NOT CHANGE OVER TIME IS READ ONCE (`_held`): asked
    // of its data source whether it has samples, which reads no values. What
    // has samples, or cannot say, is read every time, as everything was.
    const auto varies = [&](TfToken const& key) {
        const HdSceneIndexBaseRefPtr index = delegate->GetRenderIndex().GetTerminalSceneIndex();
        if (!index) {
            return true;
        }
        const HdSampledDataSourceHandle value =
            HdPrimvarsSchema::GetFromParent(index->GetPrim(id).dataSource).GetPrimvar(key).GetPrimvarValue();
        if (!value) {
            return true;
        }
        std::vector<HdSampledDataSource::Time> times;
        return value->GetContributingSampleTimesForInterval(std::numeric_limits<float>::lowest(),
                                                            std::numeric_limits<float>::max(), &times);
    };
    const auto held = [&](TfToken const& key) -> VtValue {
        if (const auto found = _held.find(key); found != _held.end() && !varies(key)) {
            return found->second;
        }
        VtValue value = delegate->Get(id, key);
        if (!varies(key)) {
            _held[key] = value;
        } else {
            _held.erase(key);
        }
        return value;
    };
    if ((*dirtyBits & (HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyPrimvar)) != 0) {
        // The float attribute, or its half twin: kept as they come, halves
        // turned into floats on the device.
        const auto either = [&](TfToken const& full, TfToken const& half) {
            VtValue value = held(full);
            return athenea::usd::streamOf(value).empty() ? held(half) : value;
        };
        athenea::usd::ParticleFieldArrays arrays;
        arrays.positions = either(UsdVolTokens->positions, UsdVolTokens->positionsh);
        arrays.orientations = either(UsdVolTokens->orientations, UsdVolTokens->orientationsh);
        arrays.scales = either(UsdVolTokens->scales, UsdVolTokens->scalesh);
        arrays.opacities = either(UsdVolTokens->opacities, UsdVolTokens->opacitiesh);
        VtValue degree = held(UsdVolTokens->radianceSphericalHarmonicsDegree);
        arrays.shDegree = degree.IsHolding<int>() ? degree.UncheckedGet<int>() : 0;
        arrays.shCoefficients = either(UsdVolTokens->radianceSphericalHarmonicsCoefficients,
                                       UsdVolTokens->radianceSphericalHarmonicsCoefficientsh);
        // AtheneaSplatLightingAPI's other two: what a relit gaussian reflects
        // with, which a cloud converted from a mesh knows and a capture does
        // not. Primvars, so they arrive without the namespace.
        static const TfToken kMetallic("athenea:splat:metallic");
        static const TfToken kRoughness("athenea:splat:roughness");
        static const TfToken kTransmission("athenea:splat:transmission");
        arrays.metallic = held(kMetallic);
        arrays.roughness = held(kRoughness);
        arrays.transmission = held(kTransmission);
        // AtheneaSplatCryptomatteAPI: the prim each gaussian came from, and the
        // names of those ids. A capture has neither.
        static const TfToken kCryptoObject("athenea:splat:cryptoObject");
        static const TfToken kCryptoManifest("athenea:splat:cryptoManifest");
        arrays.cryptoObject = held(kCryptoObject);
        // AtheneaSplatLightingAPI's transfer: what an environment puts on each
        // gaussian, so a cloud can be lit by a sky it was never baked under.
        static const TfToken kTransferDirect("athenea:splat:transferDirect");
        static const TfToken kTransferIndirect("athenea:splat:transferIndirect");
        arrays.transferDirect = held(kTransferDirect);
        arrays.transferIndirect = held(kTransferIndirect);
        static const TfToken kTransferReflected("athenea:splat:transferReflected");
        arrays.transferReflected = held(kTransferReflected);
        // The direct half as zonal lobes in each gaussian's frame, which turn
        // with it: what a cloud a skeleton carries keeps.
        static const TfToken kTransferZonal("athenea:splat:transferZonal");
        arrays.transferZonal = held(kTransferZonal);
        static const TfToken kLodGroup("athenea:lod:group");
        static const TfToken kLodCell("athenea:lod:cell");
        static const TfToken kLodThreshold("athenea:lod:threshold");
        arrays.lodGroup = tokenOf(delegate->Get(id, kLodGroup)).GetString();
        arrays.lodCell = floatOf(delegate->Get(id, kLodCell), 0.0F);
        arrays.lodThreshold = floatOf(delegate->Get(id, kLodThreshold), 1.0F);
        static const TfToken kThinWalled("athenea:splat:thinWalled");
        arrays.thinWalled = held(kThinWalled);
        static const TfToken kSchlickMetal("athenea:splat:schlickMetal");
        arrays.schlickMetal = held(kSchlickMetal);
        static const TfToken kCurvature("athenea:splat:curvature");
        arrays.curvature = held(kCurvature);
        static const TfToken kShadowBits("athenea:splat:shadowBits");
        arrays.shadowBits = held(kShadowBits);
        // The shading normal a conversion keeps apart from the frame.
        static const TfToken kNormal("athenea:splat:normal");
        arrays.normals = held(kNormal);
        // The space its colours are in: light where it says so, a capture's
        // sRGB where it does not (common/color.slang, `cloudLight`).
        static const TfToken kLinear("athenea:splat:linear");
        arrays.linear = boolOf(held(kLinear), false);
        // And the light it gives off by itself, a lamp's shade or a screen.
        static const TfToken kEmission("athenea:splat:emission");
        arrays.emission = held(kEmission);
        // And what its material layered over the base: specular, coat, sheen.
        static const TfToken kSpecularWeight("athenea:splat:specularWeight");
        static const TfToken kSpecularColour("athenea:splat:specularColor");
        static const TfToken kSpecularIor("athenea:splat:specularIor");
        static const TfToken kCoatWeight("athenea:splat:coatWeight");
        static const TfToken kCoatRoughness("athenea:splat:coatRoughness");
        static const TfToken kCoatIor("athenea:splat:coatIor");
        static const TfToken kSheenColour("athenea:splat:sheenColor");
        static const TfToken kSheenRoughness("athenea:splat:sheenRoughness");
        arrays.specularWeight = held(kSpecularWeight);
        arrays.specularColour = held(kSpecularColour);
        arrays.specularIor = held(kSpecularIor);
        arrays.coatWeight = held(kCoatWeight);
        arrays.coatRoughness = held(kCoatRoughness);
        arrays.coatIor = held(kCoatIor);
        arrays.sheenColour = held(kSheenColour);
        arrays.sheenRoughness = held(kSheenRoughness);
        static const TfToken kCoatDarkening("athenea:splat:coatDarkening");
        arrays.coatDarkening = held(kCoatDarkening);
        const VtValue manifest = held(kCryptoManifest);
        if (manifest.IsHolding<std::string>()) {
            arrays.cryptoManifest = manifest.UncheckedGet<std::string>();
        } else if (manifest.IsHolding<TfToken>()) {
            arrays.cryptoManifest = manifest.UncheckedGet<TfToken>().GetString();
        }
        // AtheneaSplatSkinningAPI: the rig, if a skeleton carries this cloud. The
        // first three do not change over time and the fourth is the only
        // thing that does, which is what makes an animated cloud cost four
        // kilobytes a frame instead of tens of megabytes.
        static const TfToken kJointIndices("athenea:splat:jointIndices");
        static const TfToken kJointWeights("athenea:splat:jointWeights");
        static const TfToken kGeomBind("athenea:splat:geomBindTransform");
        static const TfToken kXforms("athenea:splat:skinningXforms");
        arrays.jointIndices = held(kJointIndices);
        arrays.jointWeights = held(kJointWeights);
        static const TfToken kJointWeightGradients("athenea:splat:jointWeightGradients");
        arrays.jointWeightGradients = held(kJointWeightGradients);
        arrays.geomBindTransform = held(kGeomBind);
        arrays.skinningXforms = held(kXforms);
        // SkelBindingAPI, the specification's own binding, where the cloud
        // carries that instead: the same influences and bind transform under
        // UsdSkel's names, any number of joints a gaussian. The joints'
        // transforms arrive under `athenea:splat:skinningXforms` either way --
        // authored, as a cloud's cache of them, or put there by
        // HdAtheneaSkelSplatSceneIndex out of the bound Skeleton's animation.
        static const TfToken kSkelIndices("skel:jointIndices");
        static const TfToken kSkelWeights("skel:jointWeights");
        static const TfToken kSkelBind("skel:geomBindTransform");
        static const TfToken kSkelMethod("skel:skinningMethod");
        if (!arrays.jointIndices.IsHolding<VtIntArray>()) {
            arrays.jointIndices = held(kSkelIndices);
            arrays.jointWeights = held(kSkelWeights);
            if (!arrays.geomBindTransform.IsHolding<GfMatrix4d>()) {
                arrays.geomBindTransform = held(kSkelBind);
            }
            if (arrays.jointIndices.IsHolding<VtIntArray>() &&
                tokenOf(held(kSkelMethod)) == TfToken("dualQuaternion")) {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    athenea::log::warn("hdAthenea: {} asks for dual quaternion skinning; a cloud is skinned by linear "
                                   "blend here",
                                   id.GetString());
                }
            }
        }
        // AND WHERE THE SHUTTER LEAVES THEM. Hydra hands back the authored
        // samples that bracket the shutter, at their own times, so the two
        // poses are usually a whole frame apart and the displacement between
        // them has to be scaled; `xformTime*` is what says by how much. Only
        // when a shutter is open: without one this is a `Get` and nothing
        // more, as it was.
        if (param->GetShutterClose() > param->GetShutterOpen()) {
            float times[2] = {0.0F, 0.0F};
            pxr::VtValue values[2];
            const size_t n = std::min(delegate->SamplePrimvar(id, kXforms, param->GetShutterOpen(),
                                                              param->GetShutterClose(), 2, times, values),
                                      size_t{2});
            if (n >= 2 && values[0] != values[1] && times[1] > times[0]) {
                arrays.skinningXforms = values[0];
                arrays.skinningXformsEnd = values[1];
                arrays.motionScale = static_cast<float>(
                    (param->GetShutterClose() - param->GetShutterOpen()) / (times[1] - times[0]));
            }
        }
        // AtheneaSplatVisibilityAPI: the baked fields, where `athenea visibility`
        // wrote them. Two arrays that never change over time.
        static const TfToken kVisibilityParts("athenea:splat:visibilityParts");
        static const TfToken kVisibilityTexels("athenea:splat:visibilityTexels");
        arrays.visibilityParts = held(kVisibilityParts);
        arrays.visibilityTexels = held(kVisibilityTexels);
        static const TfToken kVisibilityPartOf("athenea:splat:visibilityPartOf");
        arrays.visibilityPartOf = held(kVisibilityPartOf);
        static const TfToken kVisibilityAmbient("athenea:splat:visibilityAmbient");
        arrays.visibilityAmbient = held(kVisibilityAmbient);
        raw = std::move(arrays);
    }
    std::optional<athenea::render::SplatEdit> edit;
    std::optional<athenea::usd::StreamedAsset> asset;
    if ((*dirtyBits & HdChangeTracker::DirtyPrimvar) != 0) {
        edit = editOf(delegate, id);
        asset = assetOf(delegate, id);
    }

    athenea::render::Mat4 transform;
    std::optional<athenea::render::Mat4> transformStep;
    const bool transformDirty = HdChangeTracker::IsTransformDirty(*dirtyBits, id);
    if (transformDirty) {
        transform = athenea::usd::fromUsd(delegate->GetTransform(id));
        // WHERE THE SHUTTER TAKES IT, as a mesh's transform is sampled: the
        // two bracketing samples, their difference scaled to the shutter, as
        // the skinning's is. Zero where nothing moves or no shutter is open.
        athenea::render::Mat4 step;
        step.m.fill(0.0);
        if (param->GetShutterClose() > param->GetShutterOpen()) {
            float times[2] = {0.0F, 0.0F};
            GfMatrix4d values[2];
            const size_t n = delegate->SampleTransform(id, static_cast<float>(param->GetShutterOpen()),
                                                       static_cast<float>(param->GetShutterClose()), 2, times,
                                                       values);
            if (n >= 2 && values[0] != values[n - 1] && times[n - 1] > times[0]) {
                const athenea::render::Mat4 start = athenea::usd::fromUsd(values[0]);
                const athenea::render::Mat4 end = athenea::usd::fromUsd(values[n - 1]);
                const double scale = (param->GetShutterClose() - param->GetShutterOpen()) /
                                     static_cast<double>(times[n - 1] - times[0]);
                for (size_t k = 0; k < 16; ++k) {
                    step.m[k] = (end.m[k] - start.m[k]) * scale;
                }
            }
        }
        transformStep = step;
    }
    std::optional<bool> visible;
    if (HdChangeTracker::IsVisibilityDirty(*dirtyBits, id)) {
        _UpdateVisibility(delegate, dirtyBits);
        visible = IsVisible() && HdAtheneaInstancersVisible(delegate, GetInstancerId());
    }
    engine->setSplats(id, std::move(raw), transformDirty ? &transform : nullptr, visible, edit, std::move(asset),
                      relightOf(delegate, id),
                      [&] {
                          const VtArray<TfToken> cats = delegate->GetCategories(id);
                          return std::vector<TfToken>(cats.begin(), cats.end());
                      }(),
                      litBodyOf(delegate, id), iorOf(delegate, id), transformStep);
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdAtheneaParticleField::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdAtheneaRenderParam*>(renderParam)->GetEngine()) {
        engine->remove(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
