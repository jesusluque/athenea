// Copyright (c) 2026 jesus luque.
//
// The solid layer's field on the device (SSF, phase 1): F and its analytic
// gradient at points, through a neighbour grid of Morton cells built from the
// layer alone. The method is in shaders/athenea/lod/ssf_field.slang; nothing
// of the grid is read back, not even its bounds.
#pragma once

#include <cstdint>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/algo/RadixSort.h"
#include "athenea/lod/Lod.h"

namespace athenea::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace athenea::lod {

class SolidField {
public:
    [[nodiscard]] static Result<SolidField> create(gpu::ShaderLibrary& library);

    /// The grid of `layer`, whose blobs' normals are `normals` (one packNormal
    /// word a blob: derived, never stored in the layer). Queued into `batch`;
    /// the field holds the buffers, so the layer may go out of scope.
    [[nodiscard]] Result<void> build(gpu::CommandBatch& batch, const SolidLayer& layer, const gpu::Buffer& normals);

    /// (grad F, F) as a float4 at each of `count` points (float4, xyz), into
    /// `out`. Queued into `batch`.
    void evaluate(gpu::CommandBatch& batch, const gpu::Buffer& points, uint32_t count, float tau,
                  const gpu::Buffer& out) const;

    /// Binds what ssf_field.slang declares -- the layer, the normals, the
    /// grid -- for a kernel that imports it.
    void bind(rhi::ShaderCursor cursor) const;

    [[nodiscard]] uint32_t blobs() const noexcept { return layer_.blobs; }
    /// gridInfo (ssf_field.slang's kGrid*), for a kernel that reads the grid.
    [[nodiscard]] const gpu::Buffer& info() const noexcept { return info_; }

private:
    gpu::Device*       device_ = nullptr;
    gpu::RadixSort     sort_;
    gpu::ComputeKernel clear_, bounds_, setup_, keys_, eval_;
    SolidLayer         layer_;
    gpu::Buffer        normals_;
    gpu::SortBuffers   sorting_;
    gpu::Buffer        bounds_words_;
    gpu::Buffer        info_;
};

}   // namespace athenea::lod
