// Copyright (c) 2026 jesus luque.
#pragma once

#include <pxr/imaging/hd/field.h>
#include <pxr/imaging/hd/volume.h>

PXR_NAMESPACE_OPEN_SCOPE

/// A UsdVolVolume. Its fields name OpenVDB assets; the one named `density`
/// (else the first) is the medium. Constant primvars choose how it scatters:
/// `athenea:densityScale` (density to extinction per world unit, default 1),
/// `athenea:albedo` (color3f, default 0.8) and `athenea:anisotropy` (the
/// Henyey-Greenstein g, default 0). Drawn by the `rt` technique only.
class HdAtheneaVolume final : public HdVolume {
public:
    explicit HdAtheneaVolume(SdfPath const& id) : HdVolume(id) {}
    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
              TfToken const& reprToken) override;
    void Finalize(HdRenderParam* renderParam) override;

protected:
    void _InitRepr(TfToken const&, HdDirtyBits*) override {}
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override { return bits; }
};

/// A UsdVolOpenVDBAsset: the file and the grid in it a volume's field reads.
class HdAtheneaVolumeField final : public HdField {
public:
    explicit HdAtheneaVolumeField(SdfPath const& id) : HdField(id) {}
    HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }
    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;
    void Finalize(HdRenderParam* renderParam) override;
};

PXR_NAMESPACE_CLOSE_SCOPE
