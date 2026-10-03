// Copyright (c) 2026 jesus luque.
#pragma once

#include <atomic>

#include <pxr/imaging/hd/renderDelegate.h>

#include "Engine.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdAtheneaRenderParam final : public HdRenderParam {
public:
    explicit HdAtheneaRenderParam(athenea::usd::Engine* engine) : _engine(engine) {}
    [[nodiscard]] athenea::usd::Engine* GetEngine() const { return _engine; }

    /// The camera's shutter, in frames about the current time, for Sync to
    /// sample transforms and points at: set by the pass from the camera it
    /// draws (and by StageRenderer::aim ahead of the first Sync). Equal open
    /// and close is no shutter.
    void SetShutter(double open, double close) {
        _shutterOpen.store(open);
        _shutterClose.store(close);
    }
    [[nodiscard]] double GetShutterOpen() const { return _shutterOpen.load(); }
    [[nodiscard]] double GetShutterClose() const { return _shutterClose.load(); }

private:
    athenea::usd::Engine*   _engine = nullptr;
    std::atomic<double> _shutterOpen{0.0};
    std::atomic<double> _shutterClose{0.0};
};

PXR_NAMESPACE_CLOSE_SCOPE
