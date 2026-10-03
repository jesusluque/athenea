// Copyright (c) 2026 jesus luque.
#pragma once

#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/rprim.h>

#include <unordered_map>

PXR_NAMESPACE_OPEN_SCOPE

/// A UsdVolParticleField3DGaussianSplat, as Hydra delivers it.
class HdAtheneaParticleField final : public HdRprim {
public:
    explicit HdAtheneaParticleField(SdfPath const& id) : HdRprim(id) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
              TfToken const& reprToken) override;
    void Finalize(HdRenderParam* renderParam) override;
    TfTokenVector const& GetBuiltinPrimvarNames() const override {
        static TfTokenVector builtins;
        return builtins;
    }

protected:
    void _InitRepr(TfToken const&, HdDirtyBits*) override {}

private:
    /// WHAT THIS CLOUD HAS ALREADY READ, of the arrays that do not change
    /// over time. A step of the timeline dirties every primvar of a skinned
    /// cloud, since its joints have samples; reading again an array that
    /// has none gave back the same values in a new buffer -- the file keeps
    /// integer arrays compressed -- so the engine took the cloud for a new
    /// one and uploaded all of it, every frame (`Engine::commit`'s identity).
    /// Kept here, it is the same buffer, and nothing is read or uploaded.
    std::unordered_map<TfToken, VtValue, TfToken::HashFunctor> _held;
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override { return bits; }
};

PXR_NAMESPACE_CLOSE_SCOPE
