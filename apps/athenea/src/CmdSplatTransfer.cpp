// Copyright (c) 2026 jesus luque.
//
// `athenea splat-transfer`: a TX transfer baked onto a cloud that has no mesh.
//
// mesh2splat bakes a transfer by tracing a mesh; a cloud that never had one --
// surfels on the level set of a capture's opacity, a cloud made from renders --
// cannot be baked that way, and the path tracer meets a cloud only as a
// shadow. This bakes the cloud against itself and the stage's other clouds,
// with the splat ray tracer's own structures (rt/rt_splat_transfer.slang): the
// direct half and the open directions by quadrature over the octahedral cells,
// as the path tracer's transfer mode computes them; the indirect half and the
// reflected field from what the rays meet, one bounce off the clouds. Written
// as mesh2splat writes a transfer, so a cloud baked here relights as one
// baked there (docs/decisions.md, "A transfer baked on a cloud").
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/render/Camera.h"
#include "athenea/render/GaussianRayTracer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"

namespace athenea::cli {
namespace {

struct Options {
    std::string stage;
    std::string prim;
    std::string output;
    uint32_t    cells = 16;
    uint32_t    grid = 32;
    float       offset = 1.0F;
    float       cut = 1.0e-3F;
};

int fail(const char* what, const Error& error) {
    std::fprintf(stderr, "splat-transfer: %s: ", what);
    return cli::report(error);
}

int run(const Options& o) {
    if (o.cells != 16 && o.cells != 32) {
        std::fprintf(stderr, "splat-transfer: --cells is 16 or 32\n");
        return kExitFailure;
    }
    if (o.grid < o.cells || o.grid % o.cells != 0 || o.grid > 128) {
        std::fprintf(stderr, "splat-transfer: --grid is a multiple of --cells, at most 128\n");
        return kExitFailure;
    }

    // 1. Every cloud of the stage, on the device: the one baked and what it
    //    meets.
    auto prims = usd::particleFieldPrims(o.stage);
    if (!prims) return fail("the stage", prims.error());
    if (prims->empty()) {
        std::fprintf(stderr, "splat-transfer: '%s' holds no ParticleField\n", o.stage.c_str());
        return kExitFailure;
    }
    int baked = -1;
    for (size_t i = 0; i < prims->size(); ++i) {
        if ((*prims)[i].path == o.prim) {
            baked = static_cast<int>(i);
        }
        if ((*prims)[i].path == o.prim && (*prims)[i].ior > 1.0F) {
            std::printf("splat-transfer: %s is a solid glass (index %.3g): what is behind its surface is baked "
                        "through it\n", o.prim.c_str(), static_cast<double>((*prims)[i].ior));
        }
        if ((*prims)[i].moved) {
            std::printf("splat-transfer: %s stands under a transform; it is baked in its own space\n",
                        (*prims)[i].path.c_str());
        }
    }
    if (baked < 0) {
        std::fprintf(stderr, "splat-transfer: '%s' is not a ParticleField of '%s'\n", o.prim.c_str(), o.stage.c_str());
        return kExitFailure;
    }
    auto device = gpu::Device::create();
    if (!device) return fail("the device", device.error());
    gpu::ShaderLibrary library(*device);
    gpu::Device& dev = **device;
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return fail("the loader", loader.error());
    std::vector<std::unique_ptr<scene::GpuSplats>> clouds;
    for (const usd::ParticleFieldPrim& p : *prims) {
        auto splats = usd::uploadParticleField(library, *loader, o.stage, p.path);
        if (!splats) return fail(p.path.c_str(), splats.error());
        std::printf("splat-transfer: %s, %u gaussians%s\n", p.path.c_str(), splats->count,
                    splats->hasNormals() ? ", with normals" : ", no normals (each gaussian's thinnest axis)");
        clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*splats)));
    }

    // 2. Their structures, as a frame's shadow rays trace them.
    if (!render::GaussianRayTracer::hardwareSupported(dev)) {
        std::fprintf(stderr, "splat-transfer: this device has no hardware ray tracing, which the bake's queries need\n");
        return kExitFailure;
    }
    render::RayTracerSettings settings;
    settings.route = render::RayTracingRoute::Hardware;
    auto tracer = render::GaussianRayTracer::create(library, settings);
    if (!tracer) return fail("the ray tracer", tracer.error());
    std::vector<render::SplatInstance> instances;
    for (const auto& c : clouds) {
        render::SplatInstance instance;
        instance.splats = c.get();
        instances.push_back(instance);
    }
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 10.0}, {0.0, 0.0, 0.0});
    auto prepared = tracer->prepare(render::projectionFor(camera, 64, 64), instances, 0);
    if (!prepared) return fail("the structures", prepared.error());
    const render::ShadowScene scene = tracer->shadowScene();
    if (scene.tlas == nullptr || scene.instances == 0) {
        std::fprintf(stderr, "splat-transfer: the ray tracer built no structure\n");
        return kExitFailure;
    }
    // Each cloud's first particle in the tables: its first chunk's id base.
    std::vector<uint32_t> indices(size_t{scene.instances} * 2);
    if (auto read = scene.instanceIndices->read(dev, 0, indices.size() * 4, indices.data()); !read) {
        return fail("the tables", read.error());
    }
    std::vector<uint32_t> bases;
    uint32_t chunk = 0;
    uint32_t total = 0;
    for (const auto& c : clouds) {
        bases.push_back(indices[size_t{chunk} * 2]);
        chunk += (c->count + settings.chunkSplats - 1) / settings.chunkSplats;
        total = std::max(total, bases.back() + c->count);
    }

    // 3. The bake.
    const scene::GpuSplats& target = *clouds[static_cast<size_t>(baked)];
    const uint32_t n = target.count;
    // What the file declares: the arrays are written a record each, and a
    // record dropped on loading keeps zeros.
    const uint32_t records = std::max(target.declared, n);
    const bool hasOrigin = target.origin.valid();
    const uint32_t words = o.cells * o.cells / 32;
    const auto made = [&](uint64_t elements, uint32_t bytes, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(elements, 1) * bytes;
        desc.elementBytes = bytes;
        desc.label = label;
        return gpu::Buffer::create(dev, desc);
    };
    auto direct = made(uint64_t{total} * 16, 4, "splatTransfer.direct");
    auto albedo = made(total, 16, "splatTransfer.albedo");
    auto directOut = made(uint64_t{records} * 16, 4, "splatTransfer.directOut");
    auto bits = made(uint64_t{records} * words, 4, "splatTransfer.bits");
    auto bounced = made(uint64_t{records} * 48, 4, "splatTransfer.bounced");
    auto reflected = made(uint64_t{records} * 48, 4, "splatTransfer.reflected");
    const uint32_t zero[4] = {0, 0, 0, 0};
    gpu::BufferDesc counterDesc;
    counterDesc.bytes = sizeof(zero);
    counterDesc.elementBytes = 4;
    counterDesc.label = "splatTransfer.counters";
    auto counters = gpu::Buffer::create(dev, counterDesc, zero);
    if (!direct || !albedo || !directOut || !bits || !bounced || !reflected || !counters) {
        return fail("the arrays", Error(ErrorCode::OutOfMemory, "cannot allocate the bake's arrays"));
    }
    auto first = gpu::ComputeKernel::create(library, "athenea/rt/rt_splat_transfer", "splatTransferDirect");
    if (!first) return fail("the kernel", first.error());
    auto second = gpu::ComputeKernel::create(library, "athenea/rt/rt_splat_transfer", "splatTransferBounced");
    if (!second) return fail("the kernel", second.error());
    const auto bind = [&](rhi::ShaderCursor cursor, size_t cloud, bool writeBits) {
        const scene::GpuSplats& c = *clouds[cloud];
        cursor["frames"].setBinding(scene.frames->rhi());
        cursor["colours"].setBinding(scene.colours->rhi());
        cursor["instanceData"].setBinding(scene.instanceData->rhi());
        cursor["instanceIndices"].setBinding(scene.instanceIndices->rhi());
        cursor["scene"].setBinding(scene.tlas);
        cursor["positions"].setBinding(c.positions.rhi());
        cursor["shape"].setBinding(c.shape.rhi());
        cursor["normals"].setBinding(c.hasNormals() ? c.normals.rhi() : c.shape.rhi());
        cursor["direct"].setBinding(direct->rhi());
        cursor["directOut"].setBinding(directOut->rhi());
        cursor["origin"].setBinding(c.origin.valid() ? c.origin.rhi() : c.shape.rhi());
        cursor["albedo"].setBinding(albedo->rhi());
        cursor["bits"].setBinding(bits->rhi());
        cursor["bounced"].setBinding(bounced->rhi());
        cursor["reflected"].setBinding(reflected->rhi());
        cursor["counters"].setBinding(counters->rhi());
        cursor["params"]["count"].setData(c.count);
        cursor["params"]["base"].setData(bases[cloud]);
        cursor["params"]["grid"].setData(o.grid);
        cursor["params"]["cellSide"].setData(o.cells);
        cursor["params"]["hasNormals"].setData(c.hasNormals() ? 1u : 0u);
        cursor["params"]["linear"].setData(c.linear ? 1u : 0u);
        cursor["params"]["writeBits"].setData(writeBits ? 1u : 0u);
        cursor["params"]["hasOrigin"].setData(c.origin.valid() ? 1u : 0u);
        cursor["params"]["offset"].setData(o.offset);
        cursor["params"]["cut"].setData(o.cut);
        cursor["params"]["ior"].setData(writeBits || static_cast<int>(cloud) == baked ? (*prims)[cloud].ior : 0.0F);
    };
    // The direct half of every cloud (what the second pass meets), the one
    // baked last so its bits are its own.
    for (size_t c = 0; c < clouds.size(); ++c) {
        gpu::CommandBatch batch(dev);
        first->dispatch(batch, {clouds[c]->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor, c, static_cast<int>(c) == baked);
        });
        if (auto done = batch.submit(true); !done) return fail("the direct half", done.error());
    }
    {
        gpu::CommandBatch batch(dev);
        second->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor, static_cast<size_t>(baked), false);
        });
        if (auto done = batch.submit(true); !done) return fail("the indirect half", done.error());
    }
    uint32_t said[4] = {0, 0, 0, 0};
    if (auto read = counters->read(dev, 0, sizeof(said), said); !read) return fail("the counters", read.error());
    const double cellsOpen = n > 0 ? static_cast<double>(said[2]) / (static_cast<double>(n) * o.cells * o.cells) : 0.0;
    const double facing = said[2] > 0 ? static_cast<double>(said[3]) / static_cast<double>(said[2]) : 0.0;
    std::printf("splat-transfer: %u gaussians baked, %u rays in the first pass, %.1f%% of their cells open, %.0f%% "
                "of those on the side the normals face\n",
                said[0], said[1], 100.0 * cellsOpen, 100.0 * facing);
    // A sheet is open on both sides (half and half); a surface whose normals
    // point into it has its open directions behind them.
    if (said[2] > 0 && facing < 0.35 && !((*prims)[static_cast<size_t>(baked)].ior > 1.0F)) {
        std::printf("splat-transfer: most open directions are behind the normals: are they pointing into the "
                    "surface?\n");
    }

    // 4. Into the file: the stage flattened, the transfer on the prim.
    usd::SplatTransferArrays arrays;
    arrays.count = records;
    arrays.shadowWords = words;
    arrays.direct.resize(size_t{records} * 16);
    arrays.indirect.resize(size_t{records} * 48);
    arrays.reflected.resize(size_t{records} * 48);
    arrays.shadowBits.resize(size_t{records} * words);
    if (hasOrigin && records > n) {
        std::printf("splat-transfer: %u of the file's %u gaussians were dropped on loading; theirs are zeros\n",
                    records - n, records);
    }
    if (auto r = directOut->read(dev, 0, arrays.direct.size() * 4, arrays.direct.data()); !r) {
        return fail("the direct half", r.error());
    }
    if (auto r = bounced->read(dev, 0, arrays.indirect.size() * 4, arrays.indirect.data()); !r) {
        return fail("the indirect half", r.error());
    }
    if (auto r = reflected->read(dev, 0, arrays.reflected.size() * 4, arrays.reflected.data()); !r) {
        return fail("the field", r.error());
    }
    if (auto r = bits->read(dev, 0, arrays.shadowBits.size() * 4, arrays.shadowBits.data()); !r) {
        return fail("the cells", r.error());
    }
    if (auto written = usd::writeSplatTransfer(o.stage, o.output, o.prim, arrays); !written) {
        return fail("the file", written.error());
    }
    std::printf("splat-transfer: wrote %s\n", o.output.c_str());
    return 0;
}

}   // namespace

void addSplatTransfer(CLI::App& app) {
    auto o = std::make_shared<Options>();
    CLI::App* cmd = app.add_subcommand(
        "splat-transfer", "bake a TX transfer onto a cloud that has no mesh, tracing the stage's clouds themselves");
    cmd->add_option("stage", o->stage, "the stage: .usd / .usda / .usdc with the cloud and what surrounds it")
        ->required();
    cmd->add_option("--prim", o->prim, "the ParticleField baked")->required();
    cmd->add_option("-o,--output", o->output, "the stage written, flattened, the transfer on --prim")->required();
    cmd->add_option("--cells", o->cells, "cells a side of the open directions over the whole sphere: 16 or 32");
    cmd->add_option("--grid", o->grid, "rays a side of the octahedral grid the halves are summed over (a multiple of --cells)");
    cmd->add_option("--offset", o->offset, "a ray starts this many of its gaussian's widest sigma off the surface");
    cmd->add_option("--cut", o->cut, "transmittance under which a ray stops");
    cmd->callback([o] { std::exit(run(*o)); });
}

}   // namespace athenea::cli
