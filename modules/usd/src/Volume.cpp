// Copyright (c) 2026 jesus luque.
#include "Volume.h"
#include "Instancer.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/usd/sdf/assetPath.h>

#include "RenderParam.h"
#include "athenea/usd/HydraCamera.h"
#include "athenea/usd/PrimData.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

TF_DEFINE_PRIVATE_TOKENS(_atheneaVolumeTokens,
    (density)
    ((densityScale, "athenea:densityScale"))
    ((albedo, "athenea:albedo"))
    ((anisotropy, "athenea:anisotropy"))
);

float floatOf(VtValue const& value, float fallback) {
    if (value.IsHolding<float>()) return value.UncheckedGet<float>();
    if (value.IsHolding<double>()) return static_cast<float>(value.UncheckedGet<double>());
    if (value.IsHolding<VtFloatArray>() && !value.UncheckedGet<VtFloatArray>().empty())
        return value.UncheckedGet<VtFloatArray>()[0];
    return fallback;
}

}   // namespace

HdDirtyBits HdAtheneaVolume::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::Clean | HdChangeTracker::InitRepr | HdChangeTracker::DirtyPrimvar |
           HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility | HdChangeTracker::DirtyVolumeField |
           HdChangeTracker::DirtyMaterialId;
}

void HdAtheneaVolume::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
                       TfToken const&) {
    SdfPath const& id = GetId();
    auto* engine = static_cast<HdAtheneaRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr) {
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        return;
    }
    athenea::usd::VolumeArrays arrays;
    // The medium is the field named density, else the first field.
    for (const HdVolumeFieldDescriptor& field : delegate->GetVolumeFieldDescriptors(id)) {
        if (arrays.field.IsEmpty() || field.fieldName == _atheneaVolumeTokens->density) {
            arrays.field = field.fieldId;
            arrays.fieldName = field.fieldName.GetString();
        }
    }
    arrays.objectToWorld = athenea::usd::fromUsd(delegate->GetTransform(id));
    // How the medium is shaded is the bound Material's to say, as UsdVol
    // has it (a Material with a `volume` terminal); the primvars below are
    // this engine's own shorthand, and what a Volume without one gets.
    if ((*dirtyBits & HdChangeTracker::DirtyMaterialId) != 0) {
        SetMaterialId(delegate->GetMaterialId(id));
    }
    arrays.material = GetMaterialId();
    arrays.densityScale = floatOf(delegate->Get(id, _atheneaVolumeTokens->densityScale), 1.0F);
    const VtValue albedo = delegate->Get(id, _atheneaVolumeTokens->albedo);
    if (albedo.IsHolding<GfVec3f>()) {
        const GfVec3f& a = albedo.UncheckedGet<GfVec3f>();
        arrays.albedo = {a[0], a[1], a[2]};
    } else if (albedo.IsHolding<VtVec3fArray>() && !albedo.UncheckedGet<VtVec3fArray>().empty()) {
        const GfVec3f& a = albedo.UncheckedGet<VtVec3fArray>()[0];
        arrays.albedo = {a[0], a[1], a[2]};
    }
    arrays.g = floatOf(delegate->Get(id, _atheneaVolumeTokens->anisotropy), 0.0F);
    _UpdateVisibility(delegate, dirtyBits);
    arrays.visible = IsVisible() && HdAtheneaInstancersVisible(delegate, GetInstancerId());
    engine->setVolume(id, std::move(arrays));
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdAtheneaVolume::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdAtheneaRenderParam*>(renderParam)->GetEngine()) {
        engine->removeVolume(GetId());
    }
}

void HdAtheneaVolumeField::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
    auto* engine = static_cast<HdAtheneaRenderParam*>(renderParam)->GetEngine();
    if (engine != nullptr && (*dirtyBits & DirtyParams) != 0) {
        const SdfPath& id = GetId();
        athenea::usd::VolumeFieldAsset asset;
        const VtValue file = delegate->Get(id, HdFieldTokens->filePath);
        if (file.IsHolding<SdfAssetPath>()) {
            const SdfAssetPath& path = file.UncheckedGet<SdfAssetPath>();
            asset.path = !path.GetResolvedPath().empty() ? path.GetResolvedPath() : path.GetAssetPath();
        }
        const VtValue name = delegate->Get(id, HdFieldTokens->fieldName);
        if (name.IsHolding<TfToken>()) {
            asset.gridName = name.UncheckedGet<TfToken>().GetString();
        } else if (name.IsHolding<std::string>()) {
            asset.gridName = name.UncheckedGet<std::string>();
        }
        engine->setVolumeField(id, std::move(asset));
    }
    *dirtyBits = Clean;
}

void HdAtheneaVolumeField::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdAtheneaRenderParam*>(renderParam)->GetEngine()) {
        engine->removeVolumeField(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
