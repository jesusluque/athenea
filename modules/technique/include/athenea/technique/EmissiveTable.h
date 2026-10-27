// Copyright (c) 2026 jesus luque.
//
// The frame's emitting triangles, as something next event estimation can
// choose: each material row's emission probed once on the device, and every
// triangle of the frame's records weighed by its world area times its row's
// emission luminance, accumulated into a distribution to sample. The probe
// stands in for a material that varies across a surface (a textured
// emitter); it shapes where samples go, not what they carry -- the emission
// a sample brings back is the material's at the point it lands on.
//
// One float buffer, since the path tracer's kernel is at its buffer limit:
//   [0] triangles, [1] records, [2] total power, [3] rows   (counts as uint bits)
//   [4 .. 7] the emitting triangles' box: its centre and half its diagonal
//   [8] the lights' reach: the scene radius a dome's and a sun's power carry squared
//   [9 .. 9 + rows)                          each row's emission luminance
//   [9 + rows .. + records + 1)              each record's first triangle (uint bits)
//   [.. + triangles + 1)                     the triangles' accumulated power, from 0
#pragma once

#include <optional>
#include <string>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/technique/MaterialPrograms.h"

namespace athenea::gpu {
class ShaderLibrary;
class Device;
}

namespace athenea::technique {

class EmissiveTable {
public:
    [[nodiscard]] static Result<EmissiveTable> create(gpu::ShaderLibrary& library);

    /// The words before the rows' luminances.
    static constexpr uint32_t kHeaderWords = 9;

    /// Probes the frame's material rows and weighs the scene's triangles.
    /// `rows` is the number of material records; `lightReach` the scene
    /// radius the light table weighed domes and suns by (LightTable::set).
    [[nodiscard]] Result<void> build(const MaterialFrame& frame, uint32_t rows, float lightReach);

    [[nodiscard]] const gpu::Buffer& table() const noexcept { return table_; }
    /// The total emitted power the table accumulated; 0: nothing emits.
    [[nodiscard]] float totalPower() const noexcept { return total_; }
    [[nodiscard]] uint32_t triangles() const noexcept { return triangles_; }

private:
    gpu::ShaderLibrary*               library_ = nullptr;
    gpu::Device*                      device_ = nullptr;
    std::optional<gpu::ComputeKernel> probe_, power_, accumulate_;
    std::string                       module_;
    gpu::Buffer                       table_;
    float                             total_ = 0.0F;
    uint32_t                          triangles_ = 0;
};

}   // namespace athenea::technique
