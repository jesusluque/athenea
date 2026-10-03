// Copyright (c) 2026 jesus luque.
#include "athenea/render/GaussianRayTracer.h"

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>

#include <slang-rhi/acceleration-structure-utils.h>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "FrameParams.h"

namespace athenea::render {
namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Result<gpu::Buffer> buffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                           rhi::BufferUsage extra = rhi::BufferUsage::None) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    desc.extraUsage = extra;
    return gpu::Buffer::create(device, desc);
}

template <typename T>
Result<gpu::Buffer> upload(gpu::Device& device, std::vector<T>& values, uint32_t element,
                           const char* label, rhi::BufferUsage extra = rhi::BufferUsage::None) {
    if (values.empty()) {
        values.resize(std::max<size_t>(element / sizeof(T), 1));
    }
    gpu::BufferDesc desc;
    desc.bytes = values.size() * sizeof(T);
    desc.elementBytes = element;
    desc.label = label;
    desc.extraUsage = extra;
    return gpu::Buffer::create(device, desc, values.data());
}

Result<rhi::ComPtr<rhi::IAccelerationStructure>> buildStructure(
    gpu::Device& device, const rhi::AccelerationStructureBuildDesc& build,
    rhi::AccelerationStructureKind kind, const char* label, uint64_t* updateScratch = nullptr) {
    rhi::AccelerationStructureSizes sizes;
    if (SLANG_FAILED(device.rhi()->getAccelerationStructureSizes(build, &sizes))) {
        return Error::make(ErrorCode::DeviceFailure, "no sizes for acceleration structure '{}'", label);
    }
    if (updateScratch != nullptr) {
        *updateScratch = std::max(*updateScratch, sizes.updateScratchSize);
    }
    auto scratch = buffer(device, sizes.scratchSize, 1, "rt.scratch");
    if (!scratch) return std::move(scratch).error();
    rhi::AccelerationStructureDesc desc;
    desc.kind = kind;
    desc.size = sizes.accelerationStructureSize;
    desc.label = label;
    rhi::ComPtr<rhi::IAccelerationStructure> structure;
    if (SLANG_FAILED(device.rhi()->createAccelerationStructure(desc, structure.writeRef()))) {
        return Error::make(ErrorCode::OutOfMemory, "cannot allocate acceleration structure '{}' ({} bytes)",
                           label, sizes.accelerationStructureSize);
    }
    gpu::CommandBatch batch(device);
    batch.encoder()->buildAccelerationStructure(build, structure, nullptr,
                                                rhi::BufferOffsetPair(scratch->rhi(), 0), 0, nullptr);
    batch.markDirty();
    // Waited for: the scratch buffer goes when this returns.
    ATHENEA_TRY(batch.submit(true));
    return structure;
}

void setProjection(rhi::ShaderCursor p, const Projection& projection, const RenderSettings& settings) {
    p["width"].setData(settings.width);
    p["height"].setData(settings.height);
    p["focalX"].setData(static_cast<float>(projection.focalX));
    p["focalY"].setData(static_cast<float>(projection.focalY));
    p["centreX"].setData(static_cast<float>(projection.centreX));
    p["centreY"].setData(static_cast<float>(projection.centreY));
    p["nearZ"].setData(static_cast<float>(projection.nearZ));
    p["farZ"].setData(static_cast<float>(projection.farZ));
    p["orthographic"].setData(uint32_t{projection.orthographic ? 1u : 0u});
    const std::array<float, 12> rows = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    static constexpr const char* kNames[12] = {"v00", "v01", "v02", "v03", "v10", "v11",
                                              "v12", "v13", "v20", "v21", "v22", "v23"};
    for (size_t k = 0; k < 12; ++k) {
        p[kNames[k]].setData(rows[k]);
    }
    p["linearCloud"].setData(uint32_t{0});   // per cloud, where a cloud is shaded
    p["bgR"].setData(settings.background[0]);
    p["bgG"].setData(settings.background[1]);
    p["bgB"].setData(settings.background[2]);
    p["bgA"].setData(settings.background[3]);
    p["depthMode"].setData(uint32_t{settings.depth == RenderSettings::Depth::Mean ? 0u : 1u});
    p["depthThreshold"].setData(settings.depthThreshold);
}

}   // namespace

bool GaussianRayTracer::hardwareSupported(const gpu::Device& device) noexcept {
    return device.caps().rayQuery && device.caps().accelerationStructure;
}

Result<GaussianRayTracer> GaussianRayTracer::create(gpu::ShaderLibrary& library,
                                                    RayTracerSettings settings) {
    const gpu::Caps& caps = library.device().caps();
    const bool hardware = hardwareSupported(library.device());
    // The proxies and their structures are one thing, drawing with them
    // another: a device that traces only in a pipeline (CUDA, through OptiX)
    // can build them for somebody else's rays -- a shadow query's -- and
    // cannot run this route's inline kernel. `render` says so where it is
    // asked to draw; `prepare` and `shadowScene` work either way.
    const bool structuresOnly = !hardware && caps.accelerationStructure && caps.rayTracing;
    if (settings.route == RayTracingRoute::Hardware && !hardware && !structuresOnly) {
        return Error::make(ErrorCode::Unsupported,
                           "the Hardware ray tracing route needs acceleration structures; "
                           "{} on '{}' has none", caps.apiName, caps.adapterName);
    }
    GaussianRayTracer r;
    r.device_ = &library.device();
    r.settings_ = settings;
    if (settings.route == RayTracingRoute::Auto) {
        // Metal, measured (M5 Pro, train_7k, 1080p): compute 343 ms a frame
        // and 168 ms to build, hardware 435 ms and 731 ms -- every non-opaque
        // candidate is a trip out of Metal's traversal. Vulkan: hardware,
        // not measured here. CUDA: compute (no RayQuery).
        const bool preferHardware = hardware && library.device().backend() != gpu::Backend::Metal;
        r.settings_.route = preferHardware ? RayTracingRoute::Hardware : RayTracingRoute::ComputeBvh;
    }
    r.settings_.maxSegments = std::max<uint32_t>(settings.maxSegments, 1);
    r.settings_.chunkSplats = std::clamp<uint32_t>(settings.chunkSplats, 1, uint32_t{1} << 24);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    ATHENEA_TRY(make(r.frames_kernel_, "athenea/rt/rt_frames", "rtFrames"));
    ATHENEA_TRY(make(r.shade_, "athenea/rt/rt_shade", "rtShade"));
    {
        // The two the shade kernel declares for relighting and does not read
        // when nothing is relit. A LightRecord is 96 bytes (light/lights.slang).
        gpu::BufferDesc desc;
        desc.bytes = 96;
        desc.elementBytes = 96;
        desc.label = "rt.lights.empty";
        auto lights = gpu::Buffer::create(*r.device_, desc);
        if (!lights) return std::move(lights).error();
        r.emptyLights_ = std::move(*lights);
        desc.bytes = 4;
        desc.elementBytes = 4;
        desc.label = "rt.shadow.empty";
        auto shadow = gpu::Buffer::create(*r.device_, desc);
        if (!shadow) return std::move(shadow).error();
        r.emptyShadow_ = std::move(*shadow);
        // Two words, because a lookup into a sky that was never prepared
        // still reads a texel's pair before `envLights` of 0 throws it away.
        desc.bytes = 16;
        desc.label = "rt.env.empty";
        auto envWords = gpu::Buffer::create(*r.device_, desc);
        if (!envWords) return std::move(envWords).error();
        r.emptyEnvWords_ = std::move(*envWords);
        desc.bytes = 16;
        desc.elementBytes = 16;
        desc.label = "rt.env.sh.empty";
        auto envSh = gpu::Buffer::create(*r.device_, desc);
        if (!envSh) return std::move(envSh).error();
        r.emptyEnvSh_ = std::move(*envSh);
    }
    if (r.settings_.route == RayTracingRoute::Hardware) {
        ATHENEA_TRY(make(r.proxy_, "athenea/rt/rt_proxy", "rtProxy"));
        if (!structuresOnly) {
            ATHENEA_TRY(make(r.renderSplit_, "athenea/rt/rt_render", "rtRenderSplit"));
            ATHENEA_TRY(make(r.render_, "athenea/rt/rt_render", "rtRender"));
        }
    } else {
        auto sort = gpu::RadixSort::create(library);
        if (!sort) return std::move(sort).error();
        r.sort_ = std::move(*sort);
        ATHENEA_TRY(make(r.bvhLeaves_, "athenea/rt/bvh_leaves", "bvhLeaves"));
        ATHENEA_TRY(make(r.bvhHierarchy_, "athenea/rt/bvh_hierarchy", "bvhHierarchy"));
        ATHENEA_TRY(make(r.bvhRefit_, "athenea/rt/bvh_refit", "bvhRefit"));
        ATHENEA_TRY(make(r.count_, "athenea/reference/count_nonzero", "countNonzero"));
        ATHENEA_TRY(make(r.bvhRender_, "athenea/rt/rt_bvh_render", "rtBvhRender"));
        ATHENEA_TRY(make(r.bvhHeights_, "athenea/rt/bvh_heights", "bvhHeights"));
        ATHENEA_TRY(make(r.bvhLevelKeys_, "athenea/rt/bvh_level_keys", "bvhLevelKeys"));
        ATHENEA_TRY(make(r.bvhRefitLevel_, "athenea/rt/bvh_refit_level", "bvhRefitLevel"));
    }
    return r;
}

const GaussianRayTracer::Cloud* GaussianRayTracer::find(const scene::GpuSplats& splats) const {
    const CloudKey key{&splats, splats.positions.rhi(), splats.count, splats.restPerColour};
    const auto found = std::find_if(clouds_.begin(), clouds_.end(),
                                    [&](const Cloud& c) { return c.key == key; });
    return found == clouds_.end() ? nullptr : &*found;
}

Result<void> GaussianRayTracer::rebuild(std::span<const SplatInstance> instances) {
    gpu::Device& device = *device_;
    const bool hardware = settings_.route == RayTracingRoute::Hardware;
    std::vector<Cloud> clouds;
    uint64_t splats = 0;
    uint64_t nodes = 0;
    uint32_t chunks = 0;
    for (const SplatInstance& instance : instances) {
        const scene::GpuSplats* cloud = instance.splats;
        if (cloud == nullptr || cloud->count == 0) {
            continue;
        }
        const CloudKey key{cloud, cloud->positions.rhi(), cloud->count, cloud->restPerColour};
        if (std::any_of(clouds.begin(), clouds.end(), [&](const Cloud& c) { return c.key == key; })) {
            continue;
        }
        Cloud entry;
        entry.key = key;
        entry.revision = cloud->revision;
        entry.base = static_cast<uint32_t>(splats);
        entry.firstChunk = chunks;
        entry.chunks = hardware ? (cloud->count + settings_.chunkSplats - 1) / settings_.chunkSplats : 0;
        entry.nodeBase = static_cast<uint32_t>(nodes);
        splats += cloud->count;
        nodes += cloud->count - 1;
        chunks += entry.chunks;
        clouds.push_back(entry);
    }
    if (splats > 0xFFFFFFFFull / 4) {
        return Error::make(ErrorCode::InvalidArgument,
                           "{} splats is more than the ray tracer indexes", splats);
    }
    auto frames = buffer(device, splats * 4, 16, "rt.frames");
    if (!frames) return std::move(frames).error();
    frames_ = std::move(*frames);
    // Until every structure stands, nothing here is a cloud built: a build
    // that fails leaves the next frame to build again.
    clouds_.clear();
    {
        gpu::CommandBatch batch(device);
        for (const Cloud& cloud : clouds) {
            ATHENEA_TRY(writeFrames(batch, cloud));
        }
        ATHENEA_TRY(batch.submit(true));
    }

    std::vector<rhi::ComPtr<rhi::IAccelerationStructure>> blas;
    gpu::Buffer boxes, children, leaves;
    // The refit's order, a word a node, only where a cloud is posed.
    const bool posed = std::any_of(clouds.begin(), clouds.end(), [](const Cloud& c) { return c.revision != 0; });
    levelNodes_ = gpu::Buffer{};
    if (!hardware && posed && settings_.refitsPerRebuild > 0) {
        auto made = buffer(device, std::max<uint64_t>(nodes, 1), 4, "bvh.levelNodes");
        if (!made) return std::move(made).error();
        levelNodes_ = std::move(*made);
    }
    if (hardware) {
        blas.reserve(chunks);
        updateScratchBytes_ = 0;
        for (Cloud& cloud : clouds) {
            ATHENEA_TRY(buildHardware(cloud, blas));
        }
        updateScratch_ = gpu::Buffer{};
        updateScratchCapacity_ = 0;
        ATHENEA_TRY(reserveUpdateScratch());
    } else {
        auto madeBoxes = buffer(device, std::max<uint64_t>(nodes, 1) * 2, 16, "bvh.boxes");
        if (!madeBoxes) return std::move(madeBoxes).error();
        auto madeChildren = buffer(device, std::max<uint64_t>(nodes, 1) * 2, 4, "bvh.children");
        if (!madeChildren) return std::move(madeChildren).error();
        auto madeLeaves = buffer(device, splats, 4, "bvh.leaves");
        if (!madeLeaves) return std::move(madeLeaves).error();
        boxes = std::move(*madeBoxes);
        children = std::move(*madeChildren);
        leaves = std::move(*madeLeaves);
        for (Cloud& cloud : clouds) {
            ATHENEA_TRY(buildBvh(cloud, boxes, children, leaves));
        }
    }

    clouds_ = std::move(clouds);
    blas_ = std::move(blas);
    bvhBoxes_ = std::move(boxes);
    bvhChildren_ = std::move(children);
    bvhLeaves_buffer_ = std::move(leaves);
    splats_ = static_cast<uint32_t>(splats);
    tlas_ = nullptr;
    return ok();
}

Result<void> GaussianRayTracer::reserveUpdateScratch() {
    // Always a buffer, even of nothing: an update is handed one whatever
    // size the backend asked for (Metal dereferences it).
    if (updateScratch_.valid() && updateScratchBytes_ <= updateScratchCapacity_) {
        return ok();
    }
    auto scratch = buffer(*device_, updateScratchBytes_, 1, "rt.updateScratch");
    if (!scratch) return std::move(scratch).error();
    updateScratch_ = std::move(*scratch);
    updateScratchCapacity_ = updateScratchBytes_;
    return ok();
}

Result<void> GaussianRayTracer::writeFrames(gpu::CommandBatch& batch, const Cloud& cloud) {
    const scene::GpuSplats& source = *cloud.key.cloud;
    frames_kernel_.dispatch(batch, {source.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(source.positions.rhi());
        cursor["shape"].setBinding(source.shape.rhi());
        cursor["frames"].setBinding(frames_.rhi());
        // What each particle transmits travels in the frame's spare
        // lane, so a query that walks the frames knows glass from
        // wall with no table of its own.
        cursor["pbr"].setBinding(source.hasPbr() ? source.pbr.rhi() : source.shape.rhi());
        cursor["params"]["hasPbr"].setData(uint32_t{source.hasPbr() ? 1u : 0u});
        cursor["params"]["count"].setData(source.count);
        cursor["params"]["base"].setData(cloud.base);
    });
    return ok();
}

Result<void> GaussianRayTracer::sync(std::span<const SplatInstance> instances, RayTracerStats& stats) {
    std::vector<const scene::GpuSplats*> wanted;
    for (const SplatInstance& instance : instances) {
        const scene::GpuSplats* cloud = instance.splats;
        if (cloud == nullptr || cloud->count == 0) {
            continue;
        }
        if (std::find(wanted.begin(), wanted.end(), cloud) == wanted.end()) {
            wanted.push_back(cloud);
        }
        stats.instances += 1;
    }
    bool same = wanted.size() == clouds_.size() && frames_.valid();
    for (size_t k = 0; same && k < wanted.size(); ++k) {
        const scene::GpuSplats& cloud = *wanted[k];
        same = clouds_[k].key == CloudKey{&cloud, cloud.positions.rhi(), cloud.count, cloud.restPerColour};
    }
    if (!same) {
        ATHENEA_TRY(rebuild(instances));
        stats.rebuilt = true;
        return ok();
    }
    // The same clouds: what moved is a pose. Its particles are the same ones
    // in the same order, so the structure's shape still holds them and only
    // its bounds are stale -- a refit, unless the cloud cannot take one or
    // has taken enough that the tree it was shaped as is due again.
    const bool hardware = settings_.route == RayTracingRoute::Hardware;
    for (Cloud& cloud : clouds_) {
        const uint32_t revision = cloud.key.cloud->revision;
        if (revision == cloud.revision) {
            continue;
        }
        const bool refit = cloud.refittable && settings_.refitsPerRebuild > 0 &&
                           cloud.refits < settings_.refitsPerRebuild;
        if (refit) {
            ATHENEA_TRY(hardware ? refitHardware(cloud) : refitBvh(cloud));
            cloud.refits += 1;
            stats.refitted += 1;
        } else if (!cloud.refittable && settings_.refitsPerRebuild > 0) {
            // Built before it was posed -- a BLAS that cannot be updated, no
            // room for a refit's order -- or with a tree too tall to refit
            // by heights: everything again, as before refits.
            ATHENEA_TRY(rebuild(instances));
            stats.rebuilt = true;
            return ok();
        } else {
            // In place: the cloud keeps its slots in every combined buffer.
            cloud.revision = revision;
            {
                gpu::CommandBatch batch(*device_);
                ATHENEA_TRY(writeFrames(batch, cloud));
                ATHENEA_TRY(batch.submit(true));
            }
            if (hardware) {
                std::vector<rhi::ComPtr<rhi::IAccelerationStructure>> blas;
                ATHENEA_TRY(buildHardware(cloud, blas));
                for (uint32_t c = 0; c < cloud.chunks; ++c) {
                    blas_[cloud.firstChunk + c] = std::move(blas[c]);
                }
                ATHENEA_TRY(reserveUpdateScratch());
            } else {
                ATHENEA_TRY(buildBvh(cloud, bvhBoxes_, bvhChildren_, bvhLeaves_buffer_));
            }
            cloud.refits = 0;
            stats.rebuilt = true;
        }
        cloud.revision = revision;
    }
    return ok();
}

Result<void> GaussianRayTracer::refitBvh(const Cloud& cloud) {
    const scene::GpuSplats& source = *cloud.key.cloud;
    gpu::CommandBatch batch(*device_);
    ATHENEA_TRY(writeFrames(batch, cloud));
    // Bottom up, a height at a time: each pass reads only boxes the passes
    // before it wrote. A fixed number of dispatches, nothing read back.
    for (size_t h = 1; h + 1 < cloud.levelStarts.size(); ++h) {
        const uint32_t start = cloud.levelStarts[h];
        const uint32_t count = cloud.levelStarts[h + 1] - start;
        if (count == 0) {
            continue;
        }
        bvhRefitLevel_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(source.positions.rhi());
            cursor["shape"].setBinding(source.shape.rhi());
            cursor["children"].setBinding(bvhChildren_.rhi());
            cursor["leaves"].setBinding(bvhLeaves_buffer_.rhi());
            cursor["levelNodes"].setBinding(levelNodes_.rhi());
            cursor["boxes"].setBinding(bvhBoxes_.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(source.count);
            p["nodeBase"].setData(cloud.nodeBase);
            p["leafBase"].setData(cloud.base);
            p["levelStart"].setData(start);
            p["levelCount"].setData(count);
        });
    }
    // Not waited for: the frame's own passes queue behind it.
    return batch.submit(false);
}

Result<void> GaussianRayTracer::refitHardware(const Cloud& cloud) {
    gpu::Device& device = *device_;
    const scene::GpuSplats& source = *cloud.key.cloud;
    const rhi::BufferUsage asInput = rhi::BufferUsage::AccelerationStructureBuildInput;
    // The proxies again, over the new pose: the same vertices and triangles
    // in the same places, which is what an update asks of its input.
    auto vertices = buffer(device, uint64_t{source.count} * 12, 16, "rt.proxyVertices", asInput);
    if (!vertices) return std::move(vertices).error();
    auto indices = buffer(device, uint64_t{source.count} * 60, 4, "rt.proxyIndices", asInput);
    if (!indices) return std::move(indices).error();
    gpu::CommandBatch batch(device);
    ATHENEA_TRY(writeFrames(batch, cloud));
    proxy_.dispatch(batch, {source.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(source.positions.rhi());
        cursor["shape"].setBinding(source.shape.rhi());
        cursor["vertices"].setBinding(vertices->rhi());
        cursor["indices"].setBinding(indices->rhi());
        cursor["params"]["count"].setData(source.count);
        cursor["params"]["base"].setData(settings_.chunkSplats);
    });
    // One scratch for every chunk: each update waits for the one before.
    for (uint32_t c = 0; c < cloud.chunks; ++c) {
        const uint32_t first = c * settings_.chunkSplats;
        const uint32_t count = std::min(settings_.chunkSplats, source.count - first);
        rhi::AccelerationStructureBuildInput input = {};
        input.type = rhi::AccelerationStructureBuildInputType::Triangles;
        input.triangles.vertexBuffers[0] = rhi::BufferOffsetPair(vertices->rhi(), uint64_t{first} * 12 * 16);
        input.triangles.vertexBufferCount = 1;
        input.triangles.vertexFormat = rhi::Format::RGB32Float;
        input.triangles.vertexCount = count * 12;
        input.triangles.vertexStride = 16;
        input.triangles.indexBuffer = rhi::BufferOffsetPair(indices->rhi(), uint64_t{first} * 60 * 4);
        input.triangles.indexFormat = rhi::IndexFormat::Uint32;
        input.triangles.indexCount = count * 60;
        input.triangles.flags = rhi::AccelerationStructureGeometryFlags::NoDuplicateAnyHitInvocation;
        rhi::AccelerationStructureBuildDesc build;
        build.inputs = &input;
        build.inputCount = 1;
        build.mode = rhi::AccelerationStructureBuildMode::Update;
        build.flags = rhi::AccelerationStructureBuildFlags::PreferFastTrace |
                      rhi::AccelerationStructureBuildFlags::AllowUpdate;
        rhi::IAccelerationStructure* structure = blas_[cloud.firstChunk + c].get();
        batch.encoder()->buildAccelerationStructure(build, structure, structure,
                                                    rhi::BufferOffsetPair(updateScratch_.rhi(), 0), 0, nullptr);
        batch.markDirty();
    }
    // Waited for: the proxies go when this returns.
    return batch.submit(true);
}

Result<void> GaussianRayTracer::buildHardware(Cloud& cloud,
                                              std::vector<rhi::ComPtr<rhi::IAccelerationStructure>>& blas) {
    gpu::Device& device = *device_;
    const scene::GpuSplats& source = *cloud.key.cloud;
    cloud.refittable = cloud.revision != 0 && settings_.refitsPerRebuild > 0;
    const rhi::BufferUsage asInput = rhi::BufferUsage::AccelerationStructureBuildInput;
    // The proxies only live until their structures are built.
    auto vertices = buffer(device, uint64_t{source.count} * 12, 16, "rt.proxyVertices", asInput);
    if (!vertices) return std::move(vertices).error();
    auto indices = buffer(device, uint64_t{source.count} * 60, 4, "rt.proxyIndices", asInput);
    if (!indices) return std::move(indices).error();
    {
        gpu::CommandBatch batch(device);
        proxy_.dispatch(batch, {source.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(source.positions.rhi());
            cursor["shape"].setBinding(source.shape.rhi());
            cursor["vertices"].setBinding(vertices->rhi());
            cursor["indices"].setBinding(indices->rhi());
            cursor["params"]["count"].setData(source.count);
            cursor["params"]["base"].setData(settings_.chunkSplats);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    for (uint32_t c = 0; c < cloud.chunks; ++c) {
        const uint32_t first = c * settings_.chunkSplats;
        const uint32_t count = std::min(settings_.chunkSplats, source.count - first);
        // A chunk is a window of both buffers, and the proxy kernel wrote its
        // indices relative to the chunk: vertex and primitive indices are
        // local, and no backend is asked to trust a vertex count larger than
        // the chunk (Metal took max(vertices, indices) / 3 as the triangle
        // count and read past the chunk's indices).
        rhi::AccelerationStructureBuildInput input = {};
        input.type = rhi::AccelerationStructureBuildInputType::Triangles;
        input.triangles.vertexBuffers[0] = rhi::BufferOffsetPair(vertices->rhi(), uint64_t{first} * 12 * 16);
        input.triangles.vertexBufferCount = 1;
        input.triangles.vertexFormat = rhi::Format::RGB32Float;
        input.triangles.vertexCount = count * 12;
        input.triangles.vertexStride = 16;
        input.triangles.indexBuffer = rhi::BufferOffsetPair(indices->rhi(), uint64_t{first} * 60 * 4);
        input.triangles.indexFormat = rhi::IndexFormat::Uint32;
        input.triangles.indexCount = count * 60;
        // Not opaque: every proxy entered is offered to the kernel.
        input.triangles.flags = rhi::AccelerationStructureGeometryFlags::NoDuplicateAnyHitInvocation;
        rhi::AccelerationStructureBuildDesc build;
        build.inputs = &input;
        build.inputCount = 1;
        build.flags = rhi::AccelerationStructureBuildFlags::PreferFastTrace;
        // A posed cloud's chunks are updated in place by the next pose
        // (refitHardware) rather than built again.
        if (cloud.refittable) {
            build.flags = build.flags | rhi::AccelerationStructureBuildFlags::AllowUpdate;
        }
        auto structure = buildStructure(device, build, rhi::AccelerationStructureKind::BottomLevel, "rt.blas",
                                        cloud.refittable ? &updateScratchBytes_ : nullptr);
        if (!structure) return std::move(structure).error();
        blas.push_back(std::move(*structure));
    }
    return ok();
}

Result<void> GaussianRayTracer::buildBvh(Cloud& cloud, gpu::Buffer& boxes, gpu::Buffer& children,
                                         gpu::Buffer& leaves) {
    gpu::Device& device = *device_;
    const scene::GpuSplats& source = *cloud.key.cloud;
    const uint32_t n = source.count;
    auto leafBoxes = buffer(device, uint64_t{n} * 2, 16, "bvh.leafBoxes");
    if (!leafBoxes) return std::move(leafBoxes).error();
    gpu::SortBuffers sorting;
    const auto assign = [&](gpu::Buffer& into, const char* label) -> Result<void> {
        auto made = buffer(device, n, 4, label);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(assign(sorting.keysLo, "bvh.keys"));
    ATHENEA_TRY(assign(sorting.values, "bvh.order"));
    ATHENEA_TRY(assign(sorting.scratchKeysLo, "bvh.keys2"));
    ATHENEA_TRY(assign(sorting.scratchValues, "bvh.order2"));
    auto changed = buffer(device, std::max<uint32_t>(n, 2) - 1, 4, "bvh.changed");
    if (!changed) return std::move(changed).error();
    auto changedCount = buffer(device, 1, 4, "bvh.changedCount");
    if (!changedCount) return std::move(changedCount).error();
    // A posed cloud's tree is refitted, not rebuilt, by the poses that
    // follow (refitBvh), a height at a time; the heights settle with the
    // boxes, in the same passes.
    cloud.refittable = false;
    cloud.levelStarts.clear();
    const bool levels = cloud.revision != 0 && levelNodes_.valid() && n >= 2;
    gpu::Buffer heights;
    if (levels) {
        auto made = buffer(device, n - 1, 4, "bvh.heights");
        if (!made) return std::move(made).error();
        heights = std::move(*made);
    }
    const auto setLevels = [&](rhi::ShaderCursor p) {
        p["count"].setData(n);
        p["nodeBase"].setData(cloud.nodeBase);
        p["leafBase"].setData(cloud.base);
    };

    const auto setBvh = [&](rhi::ShaderCursor p) {
        p["count"].setData(n);
        p["nodeBase"].setData(cloud.nodeBase);
        p["leafBase"].setData(cloud.base);
        p["boundsLoX"].setData(source.bounds.min[0]);
        p["boundsLoY"].setData(source.bounds.min[1]);
        p["boundsLoZ"].setData(source.bounds.min[2]);
        p["boundsHiX"].setData(source.bounds.max[0]);
        p["boundsHiY"].setData(source.bounds.max[1]);
        p["boundsHiZ"].setData(source.bounds.max[2]);
    };
    {
        gpu::CommandBatch batch(device);
        bvhLeaves_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(source.positions.rhi());
            cursor["shape"].setBinding(source.shape.rhi());
            cursor["leafBoxes"].setBinding(leafBoxes->rhi());
            cursor["mortonKeys"].setBinding(sorting.keysLo.rhi());
            cursor["mortonValues"].setBinding(sorting.values.rhi());
            setBvh(cursor["params"]);
        });
        ATHENEA_TRY(sort_.sort(batch, sorting, n, 30));
        bvhHierarchy_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["sortedKeys"].setBinding(sorting.keysLo.rhi());
            cursor["sortedValues"].setBinding(sorting.values.rhi());
            cursor["children"].setBinding(children.rhi());
            cursor["leaves"].setBinding(leaves.rhi());
            setBvh(cursor["params"]);
        });
        if (levels) {
            batch.encoder()->clearBuffer(heights.rhi(), 0, uint64_t{n - 1} * 4);
            batch.markDirty();
        }
        ATHENEA_TRY(batch.submit(true));
    }
    if (n < 2) {
        return ok();
    }
    // Refit until a pass changes nothing: the tree's depth in passes, read
    // back as one counter every kPassesPerCheck.
    static constexpr uint32_t kPassesPerCheck = 8;
    static constexpr uint32_t kMaxPasses = 256;
    for (uint32_t passes = 0; passes < kMaxPasses; passes += kPassesPerCheck) {
        gpu::CommandBatch batch(device);
        for (uint32_t k = 0; k < kPassesPerCheck; ++k) {
            bvhRefit_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["leafBoxes"].setBinding(leafBoxes->rhi());
                cursor["children"].setBinding(children.rhi());
                cursor["leaves"].setBinding(leaves.rhi());
                cursor["boxes"].setBinding(boxes.rhi());
                cursor["changed"].setBinding(changed->rhi());
                setBvh(cursor["params"]);
            });
            if (levels) {
                // After the boxes' pass, so a height that moved keeps the
                // flag set that the boxes' pass wrote.
                bvhHeights_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["children"].setBinding(children.rhi());
                    cursor["heights"].setBinding(heights.rhi());
                    cursor["changed"].setBinding(changed->rhi());
                    setLevels(cursor["params"]);
                });
            }
        }
        count_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["values"].setBinding(changed->rhi());
            cursor["total"].setBinding(changedCount->rhi());
            cursor["params"]["count"].setData(n - 1);
        });
        ATHENEA_TRY(batch.submit(true));
        uint32_t still = 0;
        ATHENEA_TRY(changedCount->read(device, 0, sizeof(still), &still));
        if (still == 0) {
            return levels ? buildLevels(cloud, heights) : ok();
        }
    }
    return Error::make(ErrorCode::InternalError, "BVH refit of {} particles did not settle in {} passes", n,
                       kMaxPasses);
}

Result<void> GaussianRayTracer::buildLevels(Cloud& cloud, const gpu::Buffer& heights) {
    // The internal nodes ordered by height -- a radix sort on an 8-bit key
    // -- and how many there are of each, which is the one thing the host
    // keeps: where each refit pass starts and how many threads it takes.
    gpu::Device& device = *device_;
    const uint32_t nodes = cloud.key.count - 1;
    static constexpr uint32_t kMaxLevels = 128;   // bvh_levels.slang
    gpu::SortBuffers sorting;
    for (auto [into, label] : {std::pair{&sorting.keysLo, "bvh.levelKeys"}, std::pair{&sorting.values, "bvh.levelOrder"},
                               std::pair{&sorting.scratchKeysLo, "bvh.levelKeys2"},
                               std::pair{&sorting.scratchValues, "bvh.levelOrder2"}}) {
        auto made = buffer(device, nodes, 4, label);
        if (!made) return std::move(made).error();
        *into = std::move(*made);
    }
    auto counts = buffer(device, kMaxLevels, 4, "bvh.levelCounts");
    if (!counts) return std::move(counts).error();
    {
        gpu::CommandBatch batch(device);
        batch.encoder()->clearBuffer(counts->rhi(), 0, uint64_t{kMaxLevels} * 4);
        batch.markDirty();
        bvhLevelKeys_.dispatch(batch, {nodes, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["heights"].setBinding(heights.rhi());
            cursor["levelKeys"].setBinding(sorting.keysLo.rhi());
            cursor["levelValues"].setBinding(sorting.values.rhi());
            cursor["levelCounts"].setBinding(counts->rhi());
            cursor["params"]["count"].setData(cloud.key.count);
            cursor["params"]["nodeBase"].setData(cloud.nodeBase);
            cursor["params"]["leafBase"].setData(cloud.base);
        });
        ATHENEA_TRY(sort_.sort(batch, sorting, nodes, 8));
        batch.encoder()->copyBuffer(levelNodes_.rhi(), uint64_t{cloud.nodeBase} * 4, sorting.values.rhi(), 0,
                                    uint64_t{nodes} * 4);
        batch.markDirty();
        ATHENEA_TRY(batch.submit(true));
    }
    std::array<uint32_t, kMaxLevels> perHeight{};
    ATHENEA_TRY(counts->read(device, 0, sizeof(perHeight), perHeight.data()));
    if (perHeight[kMaxLevels - 1] != 0) {
        // Taller than the key holds: heights from here on share a key and
        // their order within it is not bottom up. Such a cloud rebuilds.
        return ok();
    }
    size_t top = kMaxLevels - 1;
    while (top > 0 && perHeight[top] == 0) {
        --top;
    }
    cloud.levelStarts.assign(top + 2, 0);
    for (size_t h = 1; h <= top; ++h) {
        cloud.levelStarts[h + 1] = cloud.levelStarts[h] + perHeight[h];
    }
    cloud.refittable = true;
    return ok();
}

Result<void> GaussianRayTracer::prepareFrame(std::span<const SplatInstance> instances,
                                             const Vec3& eyeWorld, uint32_t shLimit,
                                             const SplatLights* lights) {
    gpu::Device& device = *device_;
    const bool hardware = settings_.route == RayTracingRoute::Hardware;
    std::vector<rhi::AccelerationStructureInstanceDescGeneric> generic;
    std::vector<float> data;          // per instance: world->cloud rows
    std::vector<uint32_t> indices;    // per instance: first particle, colour offset
    std::vector<uint32_t> bvh;        // ComputeBvh, per instance: node base, leaf base, particles
    struct Shade {
        const scene::GpuSplats* cloud;
        uint32_t                colourStart;
        Vec3                    eye;
        SplatEdit               edit;
        // What relighting needs: whether this instance asked for it, where it
        // stands in the world and which lights reach it.
        bool                    relight = false;
        bool                    litBody = false;
        bool                    transferIndirect = true;
        bool                    reflectCloud = false;
        float                   ior = 0.0F;
        std::array<float, 12>   toWorld{};
        uint64_t                categories = 0;
        const gpu::Buffer*      overrides = nullptr;
        uint32_t                overrideCount = 0;
    };
    std::vector<Shade> shades;
    uint64_t colours = 0;
    for (const SplatInstance& instance : instances) {
        if (instance.splats == nullptr || instance.splats->count == 0) {
            continue;
        }
        const Cloud* cloud = find(*instance.splats);
        if (cloud == nullptr) {
            return Error(ErrorCode::InternalError, "instance of a cloud the ray tracer did not build");
        }
        const Mat4 toCloud = aofx::xform::inverseAffine(instance.objectToWorld);
        const std::array<float, 12> rows = instance.objectToWorld.rows3x4();
        const std::array<float, 12> toCloudRows = toCloud.rows3x4();
        const auto colourStart = static_cast<uint32_t>(colours);
        shades.push_back({instance.splats, colourStart, toCloud.point(eyeWorld), instance.edit,
                          instance.relight && lights != nullptr && lights->any(), instance.litBody,
                          instance.transferIndirect, instance.reflectCloud, instance.ior, rows,
                          instance.categories, instance.overrides, instance.overrideCount});
        colours += instance.splats->count;
        if (!hardware) {
            data.insert(data.end(), toCloudRows.begin(), toCloudRows.end());
            // Colour index = particle id + offset; unsigned wrap-around makes
            // a negative offset work.
            indices.push_back(cloud->base);
            indices.push_back(colourStart - cloud->base);
            bvh.insert(bvh.end(), {cloud->nodeBase, cloud->base, instance.splats->count});
            continue;
        }
        for (uint32_t c = 0; c < cloud->chunks; ++c) {
            rhi::AccelerationStructureInstanceDescGeneric desc{};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 4; ++column) {
                    desc.transform[row][column] = rows[static_cast<size_t>(row * 4 + column)];
                }
            }
            desc.instanceID = cloud->firstChunk + c;
            desc.instanceMask = 0xFF;
            desc.instanceContributionToHitGroupIndex = 0;
            // Facing is decided in object space, so a mirroring transform
            // does not turn a proxy inside out (Metal measured; the Vulkan and
            // DXR specifications say the same). The proxies are wound
            // counter-clockwise seen from outside with a right-handed cross
            // product, which is the APIs' default front face.
            desc.flags = rhi::AccelerationStructureInstanceFlags::None;
            desc.accelerationStructure = blas_[cloud->firstChunk + c]->getHandle();
            generic.push_back(desc);
            data.insert(data.end(), toCloudRows.begin(), toCloudRows.end());
            indices.push_back(cloud->base + c * settings_.chunkSplats);
            indices.push_back(colourStart - cloud->base);
        }
    }
    if (colours > 0xFFFFFFFFull) {
        return Error::make(ErrorCode::InvalidArgument, "{} instanced splats is more than the ray tracer indexes",
                           colours);
    }
    // What a hit's instance index counts: on the hardware route the top level
    // holds one instance a chunk, and the tables above are written a chunk
    // as well, so that is the count the tables carry. Counting the clouds
    // instead told the packed shadow query (technique::SplatShadows) that a
    // hit in any chunk but the first was out of the tables, and a bird of
    // 5.9 million particles in six chunks cast the shadow of its first
    // million -- a spatial prefix, since the particles are in Morton order.
    instanceCount_ = static_cast<uint32_t>(hardware ? generic.size() : shades.size());
    auto dataBuffer = upload(device, data, 16, "rt.instanceData");
    if (!dataBuffer) return std::move(dataBuffer).error();
    auto indexBuffer = upload(device, indices, 4, "rt.instanceIndices");
    if (!indexBuffer) return std::move(indexBuffer).error();
    instanceData_ = std::move(*dataBuffer);
    instanceIndices_ = std::move(*indexBuffer);

    if (hardware) {
        const rhi::AccelerationStructureInstanceDescType type =
            rhi::getAccelerationStructureInstanceDescType(device.rhi());
        const size_t stride = rhi::getAccelerationStructureInstanceDescSize(type);
        std::vector<uint8_t> native(generic.size() * stride, 0);
        rhi::convertAccelerationStructureInstanceDescs(generic.size(), type, native.data(), stride,
                                                       generic.data(), sizeof(generic[0]));
        auto descs = upload(device, native, 0, "rt.instances", rhi::BufferUsage::AccelerationStructureBuildInput);
        if (!descs) return std::move(descs).error();
        instanceDescs_ = std::move(*descs);
        rhi::AccelerationStructureBuildInput input = {};
        input.type = rhi::AccelerationStructureBuildInputType::Instances;
        input.instances.instanceBuffer = rhi::BufferOffsetPair(instanceDescs_.rhi(), 0);
        input.instances.instanceStride = static_cast<uint32_t>(stride);
        input.instances.instanceCount = static_cast<uint32_t>(generic.size());
        rhi::AccelerationStructureBuildDesc build;
        build.inputs = &input;
        build.inputCount = 1;
        build.flags = rhi::AccelerationStructureBuildFlags::PreferFastBuild;
        auto structure = buildStructure(device, build, rhi::AccelerationStructureKind::TopLevel, "rt.tlas");
        if (!structure) return std::move(structure).error();
        tlas_ = std::move(*structure);
    } else {
        auto bvhBuffer = upload(device, bvh, 4, "bvh.instances");
        if (!bvhBuffer) return std::move(bvhBuffer).error();
        bvhInstances_ = std::move(*bvhBuffer);
    }

    if (colours > colourCapacity_ || !colours_.valid()) {
        auto made = buffer(device, colours, 16, "rt.colours");
        if (!made) return std::move(made).error();
        colours_ = std::move(*made);
        colourCapacity_ = std::max<uint64_t>(colours, 1);
    }
    // GLASS IS SHADED TWICE.
    //
    // What a glass ball shows is mostly not the sky but whatever stands
    // behind it, bent -- a chess pawn's collar under its own ball. The ray
    // that leaves the glass can be walked on until it meets a particle that
    // is not glass, and what that particle was shaded with is what the glass
    // shows; so everything is shaded first, and then the glass again, with
    // the rest of the cloud already written. A cloud with no index takes one
    // pass, as it always did.
    // A second pass is worth it where a cloud has glass to see through or
    // asks to reflect itself; both need everything else shaded first.
    bool anyGlass = false;
    for (const Shade& shade : shades) {
        anyGlass = anyGlass || shade.ior > 1.0F || shade.reflectCloud;
    }
    for (uint32_t pass = 0; pass < (anyGlass ? 2u : 1u); ++pass) {
    gpu::CommandBatch batch(device);
    for (const Shade& shade : shades) {
        if (pass != 0 && !(shade.ior > 1.0F) && !shade.reflectCloud) {
            continue;
        }
        shade_.dispatch(batch, {shade.cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(shade.cloud->positions.rhi());
            cursor["shape"].setBinding(shade.cloud->shape.rhi());
            cursor["sh"].setBinding(shade.cloud->sh.rhi());
            cursor["colours"].setBinding(colours_.rhi());
            // Every name a shader declares must be bound whether it is read
            // or not; the flags are what say which of them are.
            cursor["pbr"].setBinding(shade.cloud->hasPbr() ? shade.cloud->pbr.rhi() : shade.cloud->shape.rhi());
            cursor["normals"].setBinding(shade.cloud->hasNormals() ? shade.cloud->normals.rhi()
                                                                   : shade.cloud->shape.rhi());
            cursor["emission"].setBinding(shade.cloud->hasEmission() ? shade.cloud->emission.rhi()
                                                                     : shade.cloud->shape.rhi());
            cursor["lobes"].setBinding(shade.cloud->hasLobes() ? shade.cloud->lobes.rhi()
                                                               : shade.cloud->shape.rhi());
            // Which prim each gaussian came from, and what a pick said that
            // prim is made of. Both bound either way; `overrideCount` of 0
            // says the table is not read.
            const bool said = shade.overrides != nullptr && shade.overrideCount > 0 &&
                              shade.cloud->hasCrypto();
            cursor["cloudCrypto"].setBinding(shade.cloud->hasCrypto() ? shade.cloud->crypto.rhi()
                                                                      : shade.cloud->shape.rhi());
            cursor["splatOverrides"].setBinding(said ? shade.overrides->rhi() : emptyEnvSh_.rhi());
            cursor["params"]["overrideCount"].setData(said ? shade.overrideCount : 0u);
            cursor["lights"].setBinding(shade.relight ? lights->records->rhi() : emptyLights_.rhi());
            // Factors measured before the frame, where a cloud carries a
            // baked visibility: laid out one slot a splat in the order the
            // instances came, which is `colourStart`'s order too.
            const bool measured = shade.relight && lights != nullptr && lights->visibilityFactors != nullptr &&
                                  lights->visibilityLights > 0;
            cursor["shadowFactors"].setBinding(measured ? lights->visibilityFactors->rhi() : emptyShadow_.rhi());
            // The sky, where the frame prepared one: the same buffers the
            // rasteriser reads, so the two routes light a dome alike.
            const bool sky = shade.relight && lights != nullptr && lights->environment();
            cursor["envTexels"].setBinding(sky ? lights->envTexels->rhi() : emptyEnvWords_.rhi());
            cursor["envSh"].setBinding(sky ? lights->envSh->rhi() : emptyEnvSh_.rhi());
            cursor["envSun"].setBinding(sky ? lights->envSun->rhi() : emptyEnvSh_.rhi());
            cursor["envOfLight"].setBinding(sky ? lights->envOfLight->rhi() : emptyEnvWords_.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(shade.cloud->count);
            p["base"].setData(shade.colourStart);
            p["shadowBase"].setData(measured ? shade.colourStart : 0u);
            p["shadowLights"].setData(measured ? lights->visibilityLights : 0u);
            p["shWords"].setData(shade.cloud->shWords);
            p["restPerColour"].setData(shade.cloud->restPerColour);
            p["shLimit"].setData(shLimit);
            p["v03"].setData(static_cast<float>(shade.eye.x));
            p["v13"].setData(static_cast<float>(shade.eye.y));
            p["v23"].setData(static_cast<float>(shade.eye.z));
            setEdit(p["edit"], shade.edit);
            p["relight"].setData(uint32_t{shade.relight ? 1u : 0u});
            p["lightCount"].setData(shade.relight ? lights->count : 0u);
            p["categoriesLo"].setData(static_cast<uint32_t>(shade.categories & 0xFFFFFFFFu));
            p["categoriesHi"].setData(static_cast<uint32_t>(shade.categories >> 32));
            p["hasPbr"].setData(uint32_t{shade.cloud->hasPbr() ? 1u : 0u});
            p["hasNormals"].setData(uint32_t{shade.cloud->hasNormals() ? 1u : 0u});
            p["hasEmission"].setData(uint32_t{shade.cloud->hasEmission() ? 1u : 0u});
            p["hasLobes"].setData(uint32_t{shade.cloud->hasLobes() ? 1u : 0u});
            p["litBody"].setData(uint32_t{shade.litBody ? 1u : 0u});
            p["envLights"].setData(sky ? lights->envLights : 0u);
            p["envBaseSide"].setData(sky ? lights->envBaseSide : 1u);
            const bool carried = shade.cloud->hasTransfer();
            cursor["transfer"].setBinding(carried ? shade.cloud->transfer.rhi() : shade.cloud->shape.rhi());
            p["transferWords"].setData(carried ? shade.cloud->transferWords : 0u);
            p["transferCount"].setData(carried ? shade.cloud->transferCount : 0u);
            const bool open = carried && shade.cloud->hasShadowBits();
            cursor["shadowBits"].setBinding(open ? shade.cloud->shadowBits.rhi() : shade.cloud->shape.rhi());
            p["shadowBits"].setData(open ? 1u : 0u);
            p["transferIndirect"].setData(
                uint32_t{carried && shade.transferIndirect && shade.cloud->hasIndirect() ? 1u : 0u});
            p["ior"].setData(shade.ior);
            // The cloud's own tree, for the ray a glass gaussian casts. Only
            // the compute route builds one; on the hardware route the buffers
            // are empty and a count of zero says so.
            const Cloud* built = find(*shade.cloud);
            const bool tree = settings_.route != RayTracingRoute::Hardware && built != nullptr && bvhBoxes_.valid();
            cursor["frames"].setBinding(frames_.rhi());
            cursor["bvhBoxes"].setBinding(tree ? bvhBoxes_.rhi() : frames_.rhi());
            cursor["bvhChildren"].setBinding(tree ? bvhChildren_.rhi() : instanceIndices_.rhi());
            cursor["bvhLeaves"].setBinding(tree ? bvhLeaves_buffer_.rhi() : instanceIndices_.rhi());
            p["glassNodeBase"].setData(tree ? built->nodeBase : 0u);
            p["glassBase"].setData(tree ? built->base : 0u);
            p["glassParticles"].setData(tree ? shade.cloud->count : 0u);
            p["reflectCloud"].setData(uint32_t{shade.reflectCloud ? 1u : 0u});
            static const char* kWorldRow[12] = {"w00", "w01", "w02", "w03", "w10", "w11",
                                                "w12", "w13", "w20", "w21", "w22", "w23"};
            for (int k = 0; k < 12; ++k) {
                p[kWorldRow[k]].setData(shade.toWorld[static_cast<size_t>(k)]);
            }
            p["eyeWorldX"].setData(static_cast<float>(eyeWorld.x));
            p["eyeWorldY"].setData(static_cast<float>(eyeWorld.y));
            p["eyeWorldZ"].setData(static_cast<float>(eyeWorld.z));
            // Which space the cloud's colours are in: a capture's sRGB is
            // made light here, a splat at a time, and the blend is linear.
            p["linearCloud"].setData(uint32_t{shade.cloud->linear ? 1u : 0u});
            p["glassPass"].setData(pass);
        });
    }
    ATHENEA_TRY(batch.submit(pass + 1 < (anyGlass ? 2u : 1u)));
    }
    return ok();
}

Result<RayTracerStats> GaussianRayTracer::render(const Camera& camera,
                                                 std::span<const SplatInstance> instances,
                                                 const RenderSettings& settings,
                                                 RenderTargets& targets, const SplatLights* lights) {
    return render(projectionFor(camera, settings.width, settings.height), instances, settings, targets, lights);
}

Result<RayTracerStats> GaussianRayTracer::prepare(const Projection& projection,
                                                  std::span<const SplatInstance> instances,
                                                  uint32_t maxShDegree, const SplatLights* lights) {
    const auto start = Clock::now();
    RayTracerStats stats;
    stats.route = settings_.route;
    ATHENEA_TRY(sync(instances, stats));
    stats.splats = splats_;
    for (const Cloud& cloud : clouds_) {
        stats.chunks += cloud.chunks;
    }
    ATHENEA_TRY(prepareFrame(instances, projection.eyeWorld, maxShDegree, lights));
    stats.buildMs = msSince(start);
    stats.totalMs = stats.buildMs;
    return stats;
}

ShadowScene GaussianRayTracer::shadowScene() const noexcept {
    ShadowScene scene;
    scene.tlas = tlas_.get();
    scene.frames = &frames_;
    scene.colours = &colours_;
    scene.instanceData = &instanceData_;
    scene.instanceIndices = &instanceIndices_;
    scene.instances = instanceCount_;
    return scene;
}

Result<RayTracerStats> GaussianRayTracer::render(const Projection& projection,
                                                 std::span<const SplatInstance> instances,
                                                 const RenderSettings& settings,
                                                 RenderTargets& targets, const SplatLights* lights) {
    const auto start = Clock::now();
    RayTracerStats stats;
    stats.route = settings_.route;

    ATHENEA_TRY(sync(instances, stats));
    stats.splats = splats_;
    for (const Cloud& cloud : clouds_) {
        stats.chunks += cloud.chunks;
    }
    if (targets.width != settings.width || targets.height != settings.height || !targets.colour.valid()) {
        const uint64_t pixels = uint64_t{settings.width} * settings.height;
        auto colour = buffer(*device_, pixels, 16, "rt.target.colour");
        if (!colour) return std::move(colour).error();
        auto depth = buffer(*device_, pixels, 4, "rt.target.depth");
        if (!depth) return std::move(depth).error();
        targets.colour = std::move(*colour);
        targets.depth = std::move(*depth);
        targets.width = settings.width;
        targets.height = settings.height;
    }
    ATHENEA_TRY(prepareFrame(instances, projection.eyeWorld, settings.maxShDegree, lights));
    stats.buildMs = msSince(start);

    const auto renderStart = Clock::now();
    gpu::CommandBatch batch(*device_);
    const auto common = [&](rhi::ShaderCursor cursor) {
        cursor["frames"].setBinding(frames_.rhi());
        cursor["colours"].setBinding(colours_.rhi());
        cursor["instanceData"].setBinding(instanceData_.rhi());
        cursor["instanceIndices"].setBinding(instanceIndices_.rhi());
        cursor["colour"].setBinding(targets.colour.rhi());
        cursor["depth"].setBinding(targets.depth.rhi());
        rhi::ShaderCursor p = cursor["params"];
        setProjection(p, projection, settings);
        p["maxSegments"].setData(settings_.maxSegments);
        p["instances"].setData(instanceCount_);
        p["splitAt"].setData(settings_.splitAt);
    };
    if (settings_.route == RayTracingRoute::Hardware) {
        if (!device_->caps().rayQuery) {
            return Error(ErrorCode::Unsupported,
                         "the Hardware route draws with an inline ray, which this device has not: its "
                         "structures are still built, for a query that traces them in a pipeline");
        }
        const gpu::ComputeKernel& kernel = settings_.splitAt > 0.0F ? renderSplit_ : render_;
        kernel.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
            common(cursor);
            cursor["scene"].setBinding(tlas_.get());
        });
    } else {
        bvhRender_.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
            common(cursor);
            cursor["bvhBoxes"].setBinding(bvhBoxes_.rhi());
            cursor["bvhChildren"].setBinding(bvhChildren_.rhi());
            cursor["bvhLeaves"].setBinding(bvhLeaves_buffer_.rhi());
            cursor["bvhInstances"].setBinding(bvhInstances_.rhi());
        });
    }
    ATHENEA_TRY(batch.submit(true));
    stats.renderMs = msSince(renderStart);
    stats.totalMs = msSince(start);
    return stats;
}

}   // namespace athenea::render
