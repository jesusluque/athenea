// Copyright (c) 2026 jesus luque.
//
// A visibility buffer shaded by its instances' MaterialX materials: one
// kernel over MaterialPrograms' generated dispatch that reconstructs each
// pixel's surface (material_surface.slang), evaluates its material into a
// lobe stack, and lights it -- by the frame's lights, with shadows where the
// device traces rays, linking, and either every light at every pixel or one
// chosen a sample by its power. A frame with no lights at all falls back to
// the headlight, a unit light from the eye, as HeadlightShading draws
// unshaded meshes.
#pragma once

#include <optional>
#include <string>

#include "athenea/core/Result.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/render/Camera.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/technique/MaterialPrograms.h"
#include "athenea/technique/Visibility.h"

namespace athenea::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace athenea::technique {

class MaterialShading {
public:
    [[nodiscard]] static Result<MaterialShading> create(gpu::ShaderLibrary& library);

    /// The kernel is rebuilt when `programs` dispatches to a different set.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs);
    /// The same, for a frame with light groups (their buffers declared) or
    /// without; `shade` switches as the frame asks.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs, bool groups);
    /// And with the cloud shadow map, which a frame has only where a cloud
    /// casts one: a frame without compiles the kernel it always did.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs, bool groups, bool clouds);

    [[nodiscard]] Result<void> shade(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                                     const render::Projection& projection, const MaterialFrame& frame,
                                     render::RenderTargets& out);

private:
    gpu::ShaderLibrary*              library_ = nullptr;
    gpu::Device*                     device_ = nullptr;
    std::optional<gpu::ComputeKernel> kernel_;
    std::string                      module_;
    bool                             groups_ = false;
    bool                             clouds_ = false;
    /// Whether the kernel in use traces shadow rays.
    bool                             shadowed_ = false;
    /// The device would not make the shadowed kernel: every kernel after is
    /// made without shadow rays (setPrograms).
    bool                             shadowsRefused_ = false;
};

}   // namespace athenea::technique
