// Copyright (c) 2026 jesus luque.
//
// The renderSettings bprim: hd's HdRenderSettings holds what the scene
// index delivered (products, purposes, namespaced settings); this one
// counts its syncs, so a host can tell what arrived, and hands the active
// prim's renderingColorSpace to the engine.
#pragma once

#include <atomic>

#include <pxr/imaging/hd/renderSettings.h>

#include "RenderParam.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdAtheneaRenderSettings final : public HdRenderSettings {
public:
    explicit HdAtheneaRenderSettings(SdfPath const& id) : HdRenderSettings(id) {}

    [[nodiscard]] uint32_t GetSyncCount() const { return _syncs.load(); }

protected:
    void _Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, const HdDirtyBits* dirtyBits) override {
        HdRenderSettings::_Sync(sceneDelegate, renderParam, dirtyBits);
        _syncs.fetch_add(1);
        // The active settings' renderingColorSpace is the stage's working
        // space: handed to the engine, which resolves it at its next commit.
        if (IsActive() && renderParam != nullptr) {
            if (auto* engine = static_cast<HdAtheneaRenderParam*>(renderParam)->GetEngine()) {
                engine->setRenderingColorSpace(GetRenderingColorSpace().GetString());
            }
        }
    }

private:
    std::atomic<uint32_t> _syncs{0};
};

PXR_NAMESPACE_CLOSE_SCOPE
