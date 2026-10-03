// Copyright (c) 2026 jesus luque.
#pragma once

#include <pxr/imaging/hd/instancer.h>

PXR_NAMESPACE_OPEN_SCOPE

/// A Hydra instancer (PointInstancer, native instancing): its primvars handed
/// to the engine as they are, composed into instance transforms on the device.
class HdAtheneaInstancer final : public HdInstancer {
public:
    HdAtheneaInstancer(HdSceneDelegate* delegate, SdfPath const& id) : HdInstancer(delegate, id) {}

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;
    void Finalize(HdRenderParam* renderParam) override;
};

/// Whether every instancer above a prototype is visible: an invisible
/// PointInstancer hides all its instances. Hydra 2.0 carries the instancer's
/// own visibility on the instancer (the scene delegate's `GetVisible` of it)
/// and leaves honouring it to the render delegate; the prototype's own
/// visibility does not include it. True for a prim nothing instances.
bool HdAtheneaInstancersVisible(HdSceneDelegate* delegate, SdfPath instancer);

PXR_NAMESPACE_CLOSE_SCOPE
