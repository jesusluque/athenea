// Copyright (c) 2026 jesus luque.
#include "athenea/render/Discovery.h"

#include <algorithm>
#include <array>

#include "aofx/Transform.h"
#include "athenea/gpu/Device.h"

namespace athenea::render {
namespace {

Result<gpu::Buffer> words(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                          const uint32_t* from = nullptr) {
    const std::vector<uint32_t> zeros(from == nullptr ? std::max<uint64_t>(count, 4) : 0, 0u);
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 4) * sizeof(uint32_t);
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, from != nullptr ? from : zeros.data());
}

}   // namespace

Result<Discovery> Discovery::create(gpu::ShaderLibrary& library) {
    Discovery d;
    d.library_ = &library;
    const auto make = [&](gpu::ComputeKernel& kernel, const char* entry) -> Result<void> {
        auto made = gpu::ComputeKernel::create(library, "athenea/neural/scene_discover", entry);
        if (!made) return std::move(made).error();
        kernel = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(make(d.clear_, "discoverClear"));
    ATHENEA_TRY(make(d.mark_, "discoverMark"));
    ATHENEA_TRY(make(d.count_, "discoverCount"));
    ATHENEA_TRY(make(d.mask_, "discoverMask"));
    ATHENEA_TRY(make(d.gone_, "discoverGone"));
    ATHENEA_TRY(make(d.sizes_, "discoverSizes"));
    {
        auto valid = gpu::ComputeKernel::create(library, "athenea/neural/scene_frame", "frameValid");
        if (!valid) return std::move(valid).error();
        d.frameValid_ = std::move(*valid);
        auto pack = gpu::ComputeKernel::create(library, "athenea/neural/scene_frame", "framePack");
        if (!pack) return std::move(pack).error();
        d.framePack_ = std::move(*pack);
        auto through = gpu::ComputeKernel::create(library, "athenea/neural/scene_frame", "frameSeenThrough");
        if (!through) return std::move(through).error();
        d.frameThrough_ = std::move(*through);
        auto renderValid = gpu::ComputeKernel::create(library, "athenea/neural/scene_render", "renderValid");
        if (!renderValid) return std::move(renderValid).error();
        d.renderValid_ = std::move(*renderValid);
        auto renderPack = gpu::ComputeKernel::create(library, "athenea/neural/scene_render", "renderPack");
        if (!renderPack) return std::move(renderPack).error();
        d.renderPack_ = std::move(*renderPack);
        auto renderSpare = gpu::ComputeKernel::create(library, "athenea/neural/scene_render", "renderSparePanes");
        if (!renderSpare) return std::move(renderSpare).error();
        d.renderSpare_ = std::move(*renderSpare);
        auto prefix = gpu::PrefixSum::create(library);
        if (!prefix) return std::move(prefix).error();
        d.prefix_.emplace(std::move(*prefix));
    }
    auto counters = words(library.device(), 4, 4, "discover.counters");
    if (!counters) return std::move(counters).error();
    d.counters_ = *counters;
    auto spare = words(library.device(), 16, 4, "discover.spare");
    if (!spare) return std::move(spare).error();
    d.spareWords_ = *spare;
    auto floats = words(library.device(), 16, 16, "discover.spareFloats");
    if (!floats) return std::move(floats).error();
    d.spareFloats_ = *floats;
    return d;
}

/// Every name the module declares, bound; a dispatch names its own after.
void Discovery::bind(rhi::ShaderCursor cursor) const {
    cursor["contribution"].setBinding(spareWords_.rhi());
    cursor["seen"].setBinding(spareWords_.rhi());
    cursor["counters"].setBinding(counters_.rhi());
    cursor["seenRead"].setBinding(spareWords_.rhi());
    cursor["lifted"].setBinding(spareWords_.rhi());
    cursor["positions"].setBinding(spareFloats_.rhi());
    cursor["shape"].setBinding(spareWords_.rhi());
    cursor["sizes"].setBinding(spareFloats_.rhi());
    cursor["positionsOut"].setBinding(spareFloats_.rhi());
    cursor["cleared"].setBinding(spareWords_.rhi());
}

Result<gpu::Buffer> Discovery::record(uint32_t count, std::span<const uint32_t> held) {
    if (!held.empty() && held.size() != count) {
        return Error(ErrorCode::InvalidArgument, "a record of what was seen is not of this piece's gaussians");
    }
    return words(library_->device(), count, 4, "discover.seen", held.empty() ? nullptr : held.data());
}

Result<gpu::Buffer> Discovery::zeros(uint32_t count) {
    return words(library_->device(), count, 4, "discover.contribution");
}

Result<LookCount> Discovery::mark(const gpu::Buffer& contribution, uint32_t base, gpu::Buffer& seen, uint32_t count,
                                  uint32_t stamp, float threshold) {
    gpu::Device& device = library_->device();
    gpu::CommandBatch batch(device);
    clear_.dispatch(batch, {4, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["cleared"].setBinding(counters_.rhi());
        cursor["counters"].setBinding(spareWords_.rhi());
        cursor["params"]["count"].setData(uint32_t{4});
    });
    mark_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["contribution"].setBinding(contribution.rhi());
        cursor["seen"].setBinding(seen.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["base"].setData(base);
        cursor["params"]["stamp"].setData(stamp);
        cursor["params"]["threshold"].setData(static_cast<uint32_t>(std::max(threshold, 0.0F) * 4096.0F + 0.5F));
    });
    ATHENEA_TRY(batch.submit(true));
    std::array<uint32_t, 4> read{};
    ATHENEA_TRY(counters_.read(device, 0, sizeof(read), read.data()));
    return LookCount{read[0], read[1]};
}

Result<uint32_t> Discovery::known(const gpu::Buffer& seen, uint32_t count, uint32_t upTo) {
    gpu::Device& device = library_->device();
    gpu::CommandBatch batch(device);
    clear_.dispatch(batch, {4, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["cleared"].setBinding(counters_.rhi());
        cursor["counters"].setBinding(spareWords_.rhi());
        cursor["params"]["count"].setData(uint32_t{4});
    });
    count_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["seenRead"].setBinding(seen.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["upTo"].setData(upTo);
    });
    ATHENEA_TRY(batch.submit(true));
    std::array<uint32_t, 4> read{};
    ATHENEA_TRY(counters_.read(device, 0, sizeof(read), read.data()));
    return read[2];
}

Result<FrameLook> Discovery::fromFrame(const gpu::Buffer& colour, const gpu::Buffer& depth, uint32_t width,
                                       uint32_t height, const Projection& projection, const gpu::Buffer* knownColour,
                                       const gpu::Buffer* knownDepth, float tolerance, float spread, float repaint,
                                       const gpu::Buffer* knownSize, float refine) {
    gpu::Device& device = library_->device();
    const uint32_t pixels = width * height;
    auto valid = words(device, pixels, 4, "frame.valid");
    if (!valid) return std::move(valid).error();
    auto dest = words(device, pixels, 4, "frame.dest");
    if (!dest) return std::move(dest).error();
    auto total = words(device, 4, 4, "frame.total");
    if (!total) return std::move(total).error();
    gpu::Buffer records = spareFloats_;
    const bool hasKnown = knownColour != nullptr && knownDepth != nullptr;
    const Mat4 viewToWorld = aofx::xform::inverseAffine(projection.worldToView);
    // Every name the module declares; a dispatch reads the few it uses.
    const auto bindFrame = [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(colour.rhi());
        cursor["depth"].setBinding(depth.rhi());
        cursor["knownColour"].setBinding(hasKnown ? knownColour->rhi() : colour.rhi());
        cursor["knownDepth"].setBinding(hasKnown ? knownDepth->rhi() : depth.rhi());
        cursor["knownSize"].setBinding(hasKnown && knownSize != nullptr ? knownSize->rhi() : colour.rhi());
        cursor["valid"].setBinding(spareWords_.rhi());
        cursor["validRead"].setBinding(valid->rhi());
        cursor["dest"].setBinding(dest->rhi());
        cursor["records"].setBinding(records.rhi());
        cursor["counters"].setBinding(counters_.rhi());
        cursor["seenThrough"].setBinding(spareFloats_.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["width"].setData(width);
        p["height"].setData(height);
        p["focalX"].setData(static_cast<float>(projection.focalX));
        p["focalY"].setData(static_cast<float>(projection.focalY));
        p["centreX"].setData(static_cast<float>(projection.centreX));
        p["centreY"].setData(static_cast<float>(projection.centreY));
        p["tolerance"].setData(tolerance);
        p["spread"].setData(spread);
        static const char* const kRows[12] = {"r0x", "r0y", "r0z", "r0w", "r1x", "r1y",
                                              "r1z", "r1w", "r2x", "r2y", "r2z", "r2w"};
        for (int k = 0; k < 12; ++k) {
            p[kRows[k]].setData(static_cast<float>(viewToWorld.at(k / 4, k % 4)));
        }
        p["hasKnown"].setData(uint32_t{hasKnown ? 1u : 0u});
        p["repaint"].setData(repaint);
        p["refine"].setData(hasKnown && knownSize != nullptr ? refine : 0.0F);
    };
    {
        gpu::CommandBatch batch(device);
        clear_.dispatch(batch, {4, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor);
            cursor["cleared"].setBinding(counters_.rhi());
            cursor["counters"].setBinding(spareWords_.rhi());
            cursor["params"]["count"].setData(uint32_t{4});
        });
        frameValid_.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
            bindFrame(cursor);
            cursor["valid"].setBinding(valid->rhi());
            cursor["validRead"].setBinding(spareWords_.rhi());
        });
        ATHENEA_TRY(prefix_->apply(batch, *valid, *dest, *total, pixels));
        ATHENEA_TRY(batch.submit(true));
    }
    FrameLook look;
    std::array<uint32_t, 4> read{};
    ATHENEA_TRY(counters_.read(device, 0, sizeof(read), read.data()));
    look.surfaces = read[0];
    look.held = read[1];
    ATHENEA_TRY(total->read(device, 0, sizeof(look.made), &look.made));
    if (look.made == 0) {
        return look;
    }
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{look.made} * kFrameRecord * sizeof(float);
    desc.elementBytes = sizeof(float);
    desc.label = "frame.records";
    auto held = gpu::Buffer::create(device, desc);
    if (!held) return std::move(held).error();
    records = *held;
    {
        gpu::CommandBatch batch(device);
        framePack_.dispatch(batch, {width, height, 1}, bindFrame);
        ATHENEA_TRY(batch.submit(true));
    }
    auto floats = records.readAll<float>(device);
    if (!floats) return std::move(floats).error();
    floats->resize(size_t{look.made} * kFrameRecord);
    look.records = std::move(*floats);
    return look;
}

Result<void> Discovery::sparePanes(gpu::Buffer& mask, const gpu::Buffer& aux, uint32_t width, uint32_t height) {
    gpu::Device& device = library_->device();
    gpu::CommandBatch batch(device);
    renderSpare_.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(spareFloats_.rhi());
        cursor["depth"].setBinding(spareFloats_.rhi());
        cursor["aux"].setBinding(aux.rhi());
        cursor["knownColour"].setBinding(spareFloats_.rhi());
        cursor["knownDepth"].setBinding(spareFloats_.rhi());
        cursor["knownFirst"].setBinding(spareFloats_.rhi());
        cursor["knownSize"].setBinding(spareFloats_.rhi());
        cursor["made"].setBinding(spareWords_.rhi());
        cursor["madeRead"].setBinding(spareWords_.rhi());
        cursor["which"].setBinding(spareWords_.rhi());
        cursor["whichRead"].setBinding(spareWords_.rhi());
        cursor["dest"].setBinding(spareWords_.rhi());
        cursor["records"].setBinding(spareFloats_.rhi());
        cursor["counters"].setBinding(counters_.rhi());
        cursor["seenThrough"].setBinding(mask.rhi());
        cursor["params"]["width"].setData(width);
        cursor["params"]["height"].setData(height);
    });
    return batch.submit(true);
}

io::SplatEncoding Discovery::renderEncoding() {
    io::SplatEncoding e;
    e.floatsPerRecord = kRenderRecord;
    e.x = 0; e.y = 1; e.z = 2;
    e.opacity = 3;
    e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
    e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.normal = 14;
    e.metallic = 17;
    e.roughness = 18;
    e.transmission = 19;
    e.restBase = 0;
    e.restPerColour = 0;
    return e;
}

Result<RenderLook> Discovery::fromRender(const gpu::Buffer& colour, const gpu::Buffer& depth, const gpu::Buffer& aux,
                                         uint32_t width, uint32_t height, const Projection& projection,
                                         const KnownScene& known, const RenderLookSettings& settings) {
    gpu::Device& device = library_->device();
    const uint32_t footprint = std::max<uint32_t>(settings.footprint, 1);
    const uint32_t cellsX = (width + footprint - 1) / footprint;
    const uint32_t cellsY = (height + footprint - 1) / footprint;
    const uint32_t cells = cellsX * cellsY;
    auto made = words(device, cells, 4, "render.made");
    if (!made) return std::move(made).error();
    auto which = words(device, cells, 4, "render.which");
    if (!which) return std::move(which).error();
    auto dest = words(device, cells, 4, "render.dest");
    if (!dest) return std::move(dest).error();
    auto total = words(device, 4, 4, "render.total");
    if (!total) return std::move(total).error();
    gpu::Buffer records = spareFloats_;
    const bool hasKnown = known.colour != nullptr && known.depth != nullptr;
    const Mat4 viewToWorld = aofx::xform::inverseAffine(projection.worldToView);
    // Every name the module declares; a dispatch names what it writes after.
    const auto bindLook = [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(colour.rhi());
        cursor["depth"].setBinding(depth.rhi());
        cursor["aux"].setBinding(aux.rhi());
        cursor["knownColour"].setBinding(hasKnown ? known.colour->rhi() : colour.rhi());
        cursor["knownDepth"].setBinding(hasKnown ? known.depth->rhi() : depth.rhi());
        cursor["knownFirst"].setBinding(hasKnown && known.first != nullptr ? known.first->rhi() : depth.rhi());
        cursor["knownSize"].setBinding(hasKnown && known.size != nullptr ? known.size->rhi() : colour.rhi());
        cursor["made"].setBinding(spareWords_.rhi());
        cursor["madeRead"].setBinding(made->rhi());
        cursor["which"].setBinding(spareWords_.rhi());
        cursor["whichRead"].setBinding(which->rhi());
        cursor["dest"].setBinding(dest->rhi());
        cursor["records"].setBinding(records.rhi());
        cursor["counters"].setBinding(counters_.rhi());
        cursor["seenThrough"].setBinding(spareFloats_.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["width"].setData(width);
        p["height"].setData(height);
        p["focalX"].setData(static_cast<float>(projection.focalX));
        p["focalY"].setData(static_cast<float>(projection.focalY));
        p["centreX"].setData(static_cast<float>(projection.centreX));
        p["centreY"].setData(static_cast<float>(projection.centreY));
        p["tolerance"].setData(settings.tolerance);
        p["spread"].setData(settings.spread);
        static const char* const kRows[12] = {"r0x", "r0y", "r0z", "r0w", "r1x", "r1y",
                                              "r1z", "r1w", "r2x", "r2y", "r2z", "r2w"};
        static const char* const kBack[12] = {"w0x", "w0y", "w0z", "w0w", "w1x", "w1y",
                                              "w1z", "w1w", "w2x", "w2y", "w2z", "w2w"};
        for (int k = 0; k < 12; ++k) {
            p[kRows[k]].setData(static_cast<float>(viewToWorld.at(k / 4, k % 4)));
            p[kBack[k]].setData(static_cast<float>(projection.worldToView.at(k / 4, k % 4)));
        }
        p["hasKnown"].setData(uint32_t{hasKnown ? 1u : 0u});
        p["repaint"].setData(settings.repaint);
        p["refine"].setData(hasKnown && known.size != nullptr ? settings.refine : 0.0F);
        p["footprint"].setData(footprint);
        p["glassOpacity"].setData(settings.glassOpacity);
        p["turned"].setData(settings.turned);
        p["panes"].setData(uint32_t{settings.panes ? 1u : 0u});
        p["hasFirst"].setData(uint32_t{hasKnown && known.first != nullptr ? 1u : 0u});
    };
    {
        gpu::CommandBatch batch(device);
        clear_.dispatch(batch, {4, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor);
            cursor["cleared"].setBinding(counters_.rhi());
            cursor["counters"].setBinding(spareWords_.rhi());
            cursor["params"]["count"].setData(uint32_t{4});
        });
        renderValid_.dispatch(batch, {cellsX, cellsY, 1}, [&](rhi::ShaderCursor cursor) {
            bindLook(cursor);
            cursor["made"].setBinding(made->rhi());
            cursor["madeRead"].setBinding(spareWords_.rhi());
            cursor["which"].setBinding(which->rhi());
            cursor["whichRead"].setBinding(spareWords_.rhi());
        });
        ATHENEA_TRY(prefix_->apply(batch, *made, *dest, *total, cells));
        ATHENEA_TRY(batch.submit(true));
    }
    RenderLook look;
    std::array<uint32_t, 4> read{};
    ATHENEA_TRY(counters_.read(device, 0, sizeof(read), read.data()));
    look.surfaces = read[0];
    look.held = read[1];
    look.panes = read[2];
    look.behind = read[3];
    ATHENEA_TRY(total->read(device, 0, sizeof(look.made), &look.made));
    if (look.made == 0) {
        return look;
    }
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{look.made} * kRenderRecord * sizeof(float);
    desc.elementBytes = sizeof(float);
    desc.label = "render.records";
    auto held = gpu::Buffer::create(device, desc);
    if (!held) return std::move(held).error();
    records = *held;
    {
        gpu::CommandBatch batch(device);
        renderPack_.dispatch(batch, {cellsX, cellsY, 1}, bindLook);
        ATHENEA_TRY(batch.submit(true));
    }
    look.records = std::move(records);
    return look;
}

Result<gpu::Buffer> Discovery::seenThrough(const gpu::Buffer& colour, const gpu::Buffer& depth, uint32_t width,
                                           uint32_t height, const gpu::Buffer& knownColour,
                                           const gpu::Buffer& knownDepth, float tolerance, uint32_t& pixels,
                                           float repaint, uint32_t* repainted, const gpu::Buffer* knownSize,
                                           float refine, const Projection* projection, float spread) {
    gpu::Device& device = library_->device();
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(uint64_t{width} * height, 4) * sizeof(float);
    desc.elementBytes = sizeof(float);
    desc.label = "frame.seenThrough";
    auto mask = gpu::Buffer::create(device, desc);
    if (!mask) return std::move(mask).error();
    gpu::CommandBatch batch(device);
    clear_.dispatch(batch, {4, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["cleared"].setBinding(counters_.rhi());
        cursor["counters"].setBinding(spareWords_.rhi());
        cursor["params"]["count"].setData(uint32_t{4});
    });
    frameThrough_.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(colour.rhi());
        cursor["depth"].setBinding(depth.rhi());
        cursor["knownColour"].setBinding(knownColour.rhi());
        cursor["knownDepth"].setBinding(knownDepth.rhi());
        cursor["knownSize"].setBinding(knownSize != nullptr ? knownSize->rhi() : knownColour.rhi());
        cursor["valid"].setBinding(spareWords_.rhi());
        cursor["validRead"].setBinding(spareWords_.rhi());
        cursor["dest"].setBinding(spareWords_.rhi());
        cursor["records"].setBinding(spareFloats_.rhi());
        cursor["counters"].setBinding(counters_.rhi());
        cursor["seenThrough"].setBinding(mask->rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["width"].setData(width);
        p["height"].setData(height);
        p["tolerance"].setData(tolerance);
        p["hasKnown"].setData(uint32_t{1});
        p["repaint"].setData(repaint);
        const bool sized = knownSize != nullptr && projection != nullptr;
        p["refine"].setData(sized ? refine : 0.0F);
        p["spread"].setData(spread);
        p["focalX"].setData(sized ? static_cast<float>(projection->focalX) : 1.0F);
    });
    ATHENEA_TRY(batch.submit(true));
    std::array<uint32_t, 4> read{};
    ATHENEA_TRY(counters_.read(device, 0, sizeof(read), read.data()));
    pixels = read[2];
    if (repainted != nullptr) {
        *repainted = read[3];
    }
    return *mask;
}

Result<uint32_t> Discovery::markGone(const gpu::Buffer& lifted, uint32_t base, gpu::Buffer& gone, uint32_t count,
                                     uint32_t stamp, float threshold, const scene::GpuSplats* cloud,
                                     const Projection* projection, float coarse) {
    gpu::Device& device = library_->device();
    gpu::CommandBatch batch(device);
    clear_.dispatch(batch, {4, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["cleared"].setBinding(counters_.rhi());
        cursor["counters"].setBinding(spareWords_.rhi());
        cursor["params"]["count"].setData(uint32_t{4});
    });
    gone_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["lifted"].setBinding(lifted.rhi());
        cursor["seen"].setBinding(gone.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["base"].setData(base);
        cursor["params"]["stamp"].setData(stamp);
        if (cloud != nullptr && projection != nullptr && coarse > 0.0F) {
            // The view's depth of a point of the cloud's own space, and the look's disc a unit of depth.
            cursor["positions"].setBinding(cloud->positions.rhi());
            cursor["shape"].setBinding(cloud->shape.rhi());
            const std::array<float, 4> viewZ{static_cast<float>(projection->worldToView.at(2, 0)),
                                             static_cast<float>(projection->worldToView.at(2, 1)),
                                             static_cast<float>(projection->worldToView.at(2, 2)),
                                             static_cast<float>(projection->worldToView.at(2, 3))};
            cursor["params"]["viewZ"].setData(viewZ);
            cursor["params"]["coarse"].setData(coarse / static_cast<float>(projection->focalX));
        }
        // The lift's sums are in 1/1024 of a pixel.
        cursor["params"]["threshold"].setData(static_cast<uint32_t>(std::max(threshold, 0.0F) * 1024.0F + 0.5F));
    });
    ATHENEA_TRY(batch.submit(true));
    std::array<uint32_t, 4> read{};
    ATHENEA_TRY(counters_.read(device, 0, sizeof(read), read.data()));
    return read[0];
}

Result<scene::GpuSplats> Discovery::without(const scene::GpuSplats& cloud, const gpu::Buffer& gone, uint32_t upTo) {
    gpu::Device& device = library_->device();
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(cloud.count, 1) * 16;
    desc.elementBytes = 16;
    desc.label = "discover.positions";
    auto positions = gpu::Buffer::create(device, desc);
    if (!positions) return std::move(positions).error();
    gpu::CommandBatch batch(device);
    mask_.dispatch(batch, {cloud.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["positions"].setBinding(cloud.positions.rhi());
        cursor["positionsOut"].setBinding(positions->rhi());
        cursor["seenRead"].setBinding(gone.rhi());
        cursor["params"]["count"].setData(cloud.count);
        cursor["params"]["drop"].setData(uint32_t{1});
        cursor["params"]["upTo"].setData(upTo);
    });
    ATHENEA_TRY(batch.submit(true));
    scene::GpuSplats kept = cloud;
    kept.positions = *positions;
    return kept;
}

Result<NeuralCloud> Discovery::sizes(const scene::GpuSplats& cloud) {
    gpu::Device& device = library_->device();
    NeuralCloud sized;
    sized.count = cloud.count;
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(uint64_t{cloud.count} * kNeuralStride, 4) * sizeof(float);
    desc.elementBytes = sizeof(float);
    desc.label = "discover.sizes";
    auto latent = gpu::Buffer::create(device, desc);
    if (!latent) return std::move(latent).error();
    // A network of zeros adds nothing: the colour is the base, which is the size.
    const std::vector<float> none(kNeuralWeights, 0.0F);
    auto weights = gpu::Buffer::fromSpan<float>(device, std::span<const float>(none), "discover.noWeights");
    if (!weights) return std::move(weights).error();
    sized.latent = *latent;
    sized.weights = *weights;
    gpu::CommandBatch batch(device);
    sizes_.dispatch(batch, {cloud.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["shape"].setBinding(cloud.shape.rhi());
        cursor["sizes"].setBinding(latent->rhi());
        cursor["params"]["count"].setData(cloud.count);
        cursor["params"]["upTo"].setData(kNeuralStride);
        cursor["params"]["stamp"].setData(kNeuralFeatures);
    });
    ATHENEA_TRY(batch.submit(true));
    return sized;
}

Result<scene::GpuSplats> Discovery::masked(const scene::GpuSplats& cloud, const gpu::Buffer& seen, uint32_t upTo) {
    gpu::Device& device = library_->device();
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(cloud.count, 1) * 16;
    desc.elementBytes = 16;
    desc.label = "discover.positions";
    auto positions = gpu::Buffer::create(device, desc);
    if (!positions) return std::move(positions).error();
    gpu::CommandBatch batch(device);
    mask_.dispatch(batch, {cloud.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["positions"].setBinding(cloud.positions.rhi());
        cursor["positionsOut"].setBinding(positions->rhi());
        cursor["seenRead"].setBinding(seen.rhi());
        cursor["params"]["count"].setData(cloud.count);
        cursor["params"]["upTo"].setData(upTo);
    });
    ATHENEA_TRY(batch.submit(true));
    scene::GpuSplats known = cloud;
    known.positions = *positions;
    return known;
}

}   // namespace athenea::render
