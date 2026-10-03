// Copyright (c) 2026 jesus luque.
#pragma once

#include <pxr/imaging/hd/rprim.h>

PXR_NAMESPACE_OPEN_SCOPE

/// A UsdGeomPoints. Width is the diameter in world units (USD's meaning);
/// primvars:athenea:sizeInPixels, athenea:edl and athenea:surfaceOffset (constant) choose
/// the engine's point styles.
class HdAtheneaPoints final : public HdRprim {
public:
    explicit HdAtheneaPoints(SdfPath const& id) : HdRprim(id) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
              TfToken const& reprToken) override;
    void Finalize(HdRenderParam* renderParam) override;
    TfTokenVector const& GetBuiltinPrimvarNames() const override;

protected:
    void _InitRepr(TfToken const&, HdDirtyBits*) override {}
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override { return bits; }
};

PXR_NAMESPACE_CLOSE_SCOPE
