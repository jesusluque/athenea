// Copyright (c) 2026 jesus luque.
#include "athenea/render/SceneGlass.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <vector>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"

namespace athenea::render {
namespace {

Result<gpu::Buffer> zeroed(gpu::Device& device, uint64_t elements, uint32_t elementBytes, const char* label) {
    const uint64_t bytes = std::max<uint64_t>(elements, 1) * elementBytes;
    const std::vector<uint8_t> zeros(bytes, 0);
    gpu::BufferDesc desc;
    desc.bytes = bytes;
    desc.elementBytes = elementBytes;
    desc.label = label;
    return gpu::Buffer::create(device, desc, zeros.data());
}

double msSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}   // namespace

Result<SceneGlass> SceneGlass::create(gpu::ShaderLibrary& library) {
    SceneGlass g;
    g.library_ = &library;
    auto gather = gpu::ComputeKernel::create(library, "athenea/neural/scene_glass", "glassGather");
    if (!gather) return std::move(gather).error();
    g.gather_ = std::move(*gather);
    auto seen = gpu::ComputeKernel::create(library, "athenea/neural/scene_glass", "glassSeen");
    if (!seen) return std::move(seen).error();
    g.seen_ = std::move(*seen);
    auto counters = zeroed(library.device(), 4, 4, "glass.counters");
    if (!counters) return std::move(counters).error();
    g.counters_ = *counters;
    // The compute route's tree is the one a kernel of ours can walk, on every device.
    RayTracerSettings settings;
    settings.route = RayTracingRoute::ComputeBvh;
    auto tracer = GaussianRayTracer::create(library, settings);
    if (!tracer) return std::move(tracer).error();
    g.tracer_.emplace(std::move(*tracer));
    return g;
}

Result<gpu::Buffer> SceneGlass::room(scene::GpuSplats& cloud) {
    gpu::Device& device = library_->device();
    auto pbr = zeroed(device, cloud.count, 4, "glass.cut.pbr");
    if (!pbr) return std::move(pbr).error();
    auto normals = zeroed(device, cloud.count, 4, "glass.cut.normals");
    if (!normals) return std::move(normals).error();
    cloud.pbr = *pbr;
    cloud.normals = *normals;
    return zeroed(device, cloud.count, 16, "glass.cut.matter");
}

Result<void> SceneGlass::gather(const scene::GpuSplats& piece, const gpu::Buffer& pieceMatter,
                                const Discovery::CutPlan& plan, scene::GpuSplats& into, gpu::Buffer& matter,
                                uint32_t base) {
    if (plan.drawn == 0) {
        return ok();
    }
    if (!piece.hasPbr() || !piece.hasNormals() || !into.hasPbr() || !into.hasNormals()) {
        return Error::make(ErrorCode::InvalidArgument, "a piece's glass is gathered from its pbr and its normals");
    }
    gpu::Device& device = library_->device();
    gpu::CommandBatch batch(device);
    gather_.dispatch(batch, {piece.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(piece.positions.rhi());
        cursor["shape"].setBinding(into.shape.rhi());
        cursor["pbr"].setBinding(piece.pbr.rhi());
        cursor["normals"].setBinding(piece.normals.rhi());
        cursor["matter"].setBinding(pieceMatter.rhi());
        cursor["frames"].setBinding(piece.positions.rhi());
        cursor["bvhBoxes"].setBinding(piece.positions.rhi());
        cursor["bvhChildren"].setBinding(plan.flags.rhi());
        cursor["bvhLeaves"].setBinding(plan.flags.rhi());
        cursor["counters"].setBinding(counters_.rhi());
        cursor["flags"].setBinding(plan.flags.rhi());
        cursor["places"].setBinding(plan.places.rhi());
        cursor["pbrOut"].setBinding(into.pbr.rhi());
        cursor["normalsOut"].setBinding(into.normals.rhi());
        cursor["matterOut"].setBinding(matter.rhi());
        cursor["params"]["count"].setData(piece.count);
        cursor["params"]["base"].setData(base);
    });
    return batch.submit(true);
}

Result<GlassSeen> SceneGlass::see(const Projection& projection, scene::GpuSplats& cloud, const gpu::Buffer& matter) {
    GlassSeen result;
    if (cloud.count == 0) {
        return result;
    }
    if (!cloud.hasPbr() || !cloud.hasNormals()) {
        return Error::make(ErrorCode::InvalidArgument, "a cloud's glass is seen through its pbr and its normals");
    }
    gpu::Device& device = library_->device();
    const auto start = std::chrono::steady_clock::now();
    const SplatInstance instance{&cloud, Mat4::identity(), {}};
    auto prepared = tracer_->prepare(projection, std::span<const SplatInstance>(&instance, 1), 0);
    if (!prepared) return std::move(prepared).error();
    const ParticleTree tree = tracer_->treeOf(cloud);
    if (!tree.valid()) {
        return Error::make(ErrorCode::InternalError, "no tree over the scene's {} gaussians", cloud.count);
    }
    result.treeMs = msSince(start);
    const auto traced = std::chrono::steady_clock::now();
    const std::array<uint32_t, 4> none{0, 0, 0, 0};
    ATHENEA_TRY(counters_.write(device, 0, sizeof(none), none.data()));
    gpu::CommandBatch batch(device);
    seen_.dispatch(batch, {cloud.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(cloud.positions.rhi());
        cursor["shape"].setBinding(cloud.shape.rhi());
        cursor["pbr"].setBinding(cloud.pbr.rhi());
        cursor["normals"].setBinding(cloud.normals.rhi());
        cursor["matter"].setBinding(matter.rhi());
        cursor["frames"].setBinding(tree.frames->rhi());
        cursor["bvhBoxes"].setBinding(tree.boxes->rhi());
        cursor["bvhChildren"].setBinding(tree.children->rhi());
        cursor["bvhLeaves"].setBinding(tree.leaves->rhi());
        cursor["counters"].setBinding(counters_.rhi());
        cursor["flags"].setBinding(cloud.pbr.rhi());
        cursor["places"].setBinding(cloud.pbr.rhi());
        cursor["pbrOut"].setBinding(counters_.rhi());
        cursor["normalsOut"].setBinding(counters_.rhi());
        cursor["matterOut"].setBinding(counters_.rhi());
        cursor["params"]["count"].setData(cloud.count);
        cursor["params"]["nodeBase"].setData(tree.nodeBase);
        cursor["params"]["leafBase"].setData(tree.base);
        cursor["params"]["eyeX"].setData(static_cast<float>(projection.eyeWorld.x));
        cursor["params"]["eyeY"].setData(static_cast<float>(projection.eyeWorld.y));
        cursor["params"]["eyeZ"].setData(static_cast<float>(projection.eyeWorld.z));
        cursor["params"]["linearCloud"].setData(uint32_t{cloud.linear ? 1u : 0u});
    });
    ATHENEA_TRY(batch.submit(true));
    std::array<uint32_t, 4> read{};
    ATHENEA_TRY(counters_.read(device, 0, sizeof(read), read.data()));
    result.glass = read[0];
    result.behind = read[1];
    result.mirrored = read[2];
    result.mirrors = read[3];
    result.seenMs = msSince(traced);
    // The colours changed where the buffers did not: whoever keeps something made of them makes it again.
    ++cloud.revision;
    return result;
}

}   // namespace athenea::render
