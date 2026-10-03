// Copyright (c) 2026 jesus luque.
#include "athenea/scene/SplatSkinner.h"

#include <array>

#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::scene {
namespace {

// GfMatrix4f is row-major with the translation in the last row and vectors on
// the left (v * M); the kernel multiplies M * v with rows, so the matrix goes
// over transposed -- the same turn `geom::Skinner` makes for a mesh, and for
// the same reason.
std::array<float, 16> transposed(const std::array<float, 16>& m) {
    std::array<float, 16> t{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            t[static_cast<size_t>(r * 4 + c)] = m[static_cast<size_t>(c * 4 + r)];
        }
    }
    return t;
}

void setRows(rhi::ShaderCursor cursor, const char* name, const std::array<float, 16>& m) {
    const std::array<float, 16> t = transposed(m);
    for (int r = 0; r < 4; ++r) {
        const std::string field = std::string(name) + std::to_string(r);
        cursor[field.c_str()].setData(t.data() + r * 4, 16);
    }
}

}   // namespace

Result<SplatSkinner> SplatSkinner::create(gpu::ShaderLibrary& library) {
    auto kernel = gpu::ComputeKernel::create(library, "athenea/scene/splat_skin", "splatSkin");
    if (!kernel) return std::move(kernel).error();
    SplatSkinner made;
    made.kernel_ = std::move(*kernel);
    return made;
}

Result<void> SplatSkinner::skin(gpu::CommandBatch& batch, const SplatSkinInput& input,
                                gpu::Buffer& positions, gpu::Buffer& shape, gpu::Buffer* motion,
                                gpu::Buffer* normals) {
    if (input.rest == nullptr || input.influences == nullptr || input.skinningXforms == nullptr) {
        return Error(ErrorCode::InvalidArgument, "a skinned cloud wants a rest cloud, its joints and their transforms");
    }
    const uint32_t count = input.rest->count;
    if (count == 0) {
        return ok();
    }
    const bool moves = motion != nullptr && input.skinningXformsEnd != nullptr;
    const bool turnsNormals = normals != nullptr && input.rest->hasNormals();
    kernel_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["restPositions"].setBinding(input.rest->positions.rhi());
        cursor["restShape"].setBinding(input.rest->shape.rhi());
        cursor["influences"].setBinding(input.influences->rhi());
        cursor["skinningXforms"].setBinding(input.skinningXforms->rhi());
        // Bound whether or not it is read, as every declared name must be;
        // the second pose falls back to the first, and the displacement
        // buffer to the positions it is the same length as.
        cursor["skinningXformsEnd"].setBinding(moves ? input.skinningXformsEnd->rhi()
                                                     : input.skinningXforms->rhi());
        cursor["motion"].setBinding(moves ? motion->rhi() : positions.rhi());
        cursor["positions"].setBinding(positions.rhi());
        cursor["shape"].setBinding(shape.rhi());
        cursor["restNormals"].setBinding(turnsNormals ? input.rest->normals.rhi() : input.rest->shape.rhi());
        cursor["normals"].setBinding(turnsNormals ? normals->rhi() : shape.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["count"].setData(count);
        p["perSplat"].setData(std::max(input.perSplat, 1U));
        p["motionOut"].setData(uint32_t{moves ? 1u : 0u});
        p["normalsOut"].setData(uint32_t{turnsNormals ? 1u : 0u});
        setRows(p, "geomBind", input.geomBindTransform);
        setRows(p, "skelToWorld", input.skelLocalToWorld);
        setRows(p, "worldToPrim", input.primWorldToLocal);
    });
    return ok();
}

}   // namespace athenea::scene
