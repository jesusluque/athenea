// Copyright (c) 2026 jesus luque.
#pragma once

#include <memory>

#include <pxr/imaging/hd/renderPass.h>

#include "Engine.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdAtheneaRenderDelegate;

class HdAtheneaRenderPass final : public HdRenderPass {
public:
    HdAtheneaRenderPass(HdRenderIndex* index, HdRprimCollection const& collection,
                    athenea::usd::Engine* engine, const HdAtheneaRenderDelegate* delegate)
        : HdRenderPass(index, collection), _engine(engine), _delegate(delegate) {}

    /// A path traced frame is gathered over as many passes as it takes; every
    /// other frame is whole when it is drawn.
    bool IsConverged() const override { return _engine == nullptr || _engine->pathConverged(); }

protected:
    void _Execute(HdRenderPassStateSharedPtr const& renderPassState,
                  TfTokenVector const& renderTags) override;

private:
    athenea::usd::Engine*             _engine = nullptr;
    const HdAtheneaRenderDelegate*    _delegate = nullptr;
    /// Shared with the fills the render buffers hold until they are mapped.
    std::shared_ptr<athenea::render::RenderTargets> _targets = std::make_shared<athenea::render::RenderTargets>();
};

PXR_NAMESPACE_CLOSE_SCOPE
