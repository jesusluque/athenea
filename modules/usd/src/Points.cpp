// Copyright (c) 2026 jesus luque.
#include "Points.h"
#include "Instancer.h"

#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/base/tf/staticTokens.h>

#include <optional>
#include <string>

#include "RenderParam.h"
#include "athenea/core/Log.h"
#include "athenea/usd/HydraCamera.h"
#include "athenea/usd/PrimData.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

TF_DEFINE_PRIVATE_TOKENS(_atheneaTokens,
    ((sizeInPixels, "athenea:sizeInPixels"))
    ((edl, "athenea:edl"))
    ((surfaceOffset, "athenea:surfaceOffset"))
);

// Blender's Gaussian-splat PointCloud, as its USD export writes one: a
// UsdGeomPoints whose primvars are the cloud's own attributes.
TF_DEFINE_PRIVATE_TOKENS(_blenderTokens,
    (rotation)(scale)
    ((radianceBase, "radiance:base"))
    ((sh0, "radiance:sh_0"))
);

bool holdsArray(VtValue const& value) {
    return !athenea::usd::streamOf(value).empty();
}

/// A Gaussian-splat PointCloud out of Blender: a rotation and a scale a
/// point, and its radiance (`radiance:base`, or harmonics past DC). Its
/// arrays as they are; their layout is the decode's (scene/streams.slang).
std::optional<athenea::usd::ParticleFieldArrays> blenderSplats(HdSceneDelegate* delegate, SdfPath const& id) {
    VtValue rotation = delegate->Get(id, _blenderTokens->rotation);
    if (!rotation.IsHolding<VtQuatfArray>() && !rotation.IsHolding<VtQuathArray>()) {
        return std::nullopt;
    }
    VtValue scale = delegate->Get(id, _blenderTokens->scale);
    VtValue base = delegate->Get(id, _blenderTokens->radianceBase);
    VtValue sh0 = delegate->Get(id, _blenderTokens->sh0);
    if (!holdsArray(scale) || (!holdsArray(base) && !holdsArray(sh0))) {
        return std::nullopt;
    }
    athenea::usd::ParticleFieldArrays arrays;
    arrays.positions = delegate->Get(id, HdTokens->points);
    arrays.orientations = std::move(rotation);
    arrays.scales = std::move(scale);
    arrays.radianceBase = std::move(base);
    if (holdsArray(sh0)) {
        arrays.shPlanes.push_back(std::move(sh0));
        for (int k = 1; k < 15; ++k) {
            VtValue plane = delegate->Get(id, TfToken("radiance:sh_" + std::to_string(k)));
            if (!holdsArray(plane)) {
                break;
            }
            arrays.shPlanes.push_back(std::move(plane));
        }
    }
    if (arrays.radianceBase.IsEmpty()) {
        athenea::log::warn("hdAthenea: {}: a Gaussian-splat point cloud without radiance:base, drawn opaque and "
                           "grey (Blender's USD export drops it; the athenea_hydra add-on writes it)",
                           id.GetString());
    }
    return arrays;
}

float firstFloat(VtValue const& value, float fallback) {
    if (value.IsHolding<float>()) return value.UncheckedGet<float>();
    if (value.IsHolding<double>()) return static_cast<float>(value.UncheckedGet<double>());
    if (value.IsHolding<VtFloatArray>() && !value.UncheckedGet<VtFloatArray>().empty())
        return value.UncheckedGet<VtFloatArray>()[0];
    return fallback;
}

}   // namespace

TfTokenVector const& HdAtheneaPoints::GetBuiltinPrimvarNames() const {
    static const TfTokenVector names{HdTokens->points, HdTokens->widths, HdTokens->displayColor};
    return names;
}

HdDirtyBits HdAtheneaPoints::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::Clean | HdChangeTracker::InitRepr | HdChangeTracker::DirtyPoints |
           HdChangeTracker::DirtyWidths | HdChangeTracker::DirtyPrimvar |
           HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility;
}

void HdAtheneaPoints::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam,
                       HdDirtyBits* dirtyBits, TfToken const&) {
    SdfPath const& id = GetId();
    auto* engine = static_cast<HdAtheneaRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr) {
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        return;
    }
    std::optional<athenea::usd::PointsArrays> raw;
    std::optional<athenea::usd::ParticleFieldArrays> splats;
    std::optional<athenea::render::PointStyle> style;
    const bool arraysDirty = (*dirtyBits & (HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyPrimvar |
                                            HdChangeTracker::DirtyWidths)) != 0;
    if (arraysDirty) {
        // BLENDER'S SPLATS ARE SPLATS. Its Gaussian-splat point cloud reaches
        // a delegate as Points; drawn as a cloud, under this prim's id. A
        // prim that turns from one into the other leaves the old entry.
        splats = blenderSplats(delegate, id);
        if (splats.has_value() != _splats) {
            engine->remove(id);
            _splats = splats.has_value();
        }
    }
    if (arraysDirty && !_splats) {
        athenea::usd::PointsArrays arrays;
        arrays.positions = delegate->Get(id, HdTokens->points);
        arrays.colours = delegate->Get(id, HdTokens->displayColor);
        raw = std::move(arrays);

        athenea::render::PointStyle s;
        s.size = firstFloat(delegate->Get(id, HdTokens->widths), 0.01F);
        const float pixels = firstFloat(delegate->Get(id, _atheneaTokens->sizeInPixels), 0.0F);
        if (pixels > 0.0F) {
            s.sizeMode = athenea::render::PointStyle::Size::Pixels;
            s.size = pixels;
        }
        s.edlStrength = firstFloat(delegate->Get(id, _atheneaTokens->edl), 0.0F);
        s.surfaceDepthOffset = firstFloat(delegate->Get(id, _atheneaTokens->surfaceOffset), 0.0F);
        style = s;
    }
    athenea::render::Mat4 transform;
    const bool transformDirty = HdChangeTracker::IsTransformDirty(*dirtyBits, id);
    if (transformDirty) {
        transform = athenea::usd::fromUsd(delegate->GetTransform(id));
    }
    std::optional<bool> visible;
    if (HdChangeTracker::IsVisibilityDirty(*dirtyBits, id)) {
        _UpdateVisibility(delegate, dirtyBits);
        visible = IsVisible() && HdAtheneaInstancersVisible(delegate, GetInstancerId());
    }
    if (_splats) {
        // A capture: the colours are the sRGB it was trained in, which is
        // what a cloud that does not say otherwise is taken to hold.
        engine->setSplats(id, std::move(splats), transformDirty ? &transform : nullptr, visible);
    } else {
        engine->setPoints(id, std::move(raw), transformDirty ? &transform : nullptr, visible, style);
    }
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdAtheneaPoints::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdAtheneaRenderParam*>(renderParam)->GetEngine()) {
        engine->remove(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
