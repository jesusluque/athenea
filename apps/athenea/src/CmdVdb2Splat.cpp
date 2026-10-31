// Copyright (c) 2026 jesus luque.
//
// `athenea vdb2splat`: a .vdb volume into volumetric gaussians (proposal
// 103), and back. The grid is read and laid out as NanoVDB on the host, its
// leaves laid out as a picture by a kernel (world::VolumeSplats), the fit is
// the Vdb2Splat effect (plugins/vdb2splat) run through the AOFX host as
// mesh2splat's is, and what comes out is written as a ParticleField whose
// gaussians carry the extinction's peak, Cd and temperature. `--back` writes
// the mixture evaluated on the input's own topology as a .vdb the path
// tracer renders unchanged; `--measure` prints how far that is from the
// input and how far the mixture's closed-form transmittance is from ratio
// tracking through the input. What crosses to the processor is counts, the
// metrics' sums and the values a file holds.
#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "Commands.h"
#include "aofx/Effect.h"
#include "athenea/aofx/EffectRegistry.h"
#include "athenea/aofx/EffectRender.h"
#include "athenea/core/Platform.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu_host/Context.h"
#include "athenea/gpu_host/ImageStorage.h"
#include "athenea/image/Image.h"
#include "athenea/io/Vdb.h"
#include "athenea/usd/Export.h"
#include "athenea/world/VolumeSet.h"
#include "athenea/world/VolumeSplats.h"

namespace athenea::cli {
namespace {

struct Options {
    std::string              input;
    std::string              output;
    std::string              back;
    std::string              density = "density";
    std::string              temperature = "temperature";
    std::string              colour = "Cd";
    uint32_t                 target = 20000;
    uint32_t                 iterations = 100;
    uint32_t                 emIterations = 40;
    uint32_t                 attributeIterations = 20;
    uint32_t                 maxBlock = 64;
    double                   emissionWeight = 1.0;
    double                   initFraction = 0.6;
    uint32_t                 densifyEvery = 20;
    double                   learningRate = 0.1;
    double                   densityScale = 1.0;
    bool                     measure = false;
    uint32_t                 rays = 4096;
    uint32_t                 raySamples = 64;
    std::string              fixture;
    uint32_t                 fixtureSize = 96;
    uint32_t                 fixtureGaussians = 64;
    double                   voxelSize = 0.05;
    uint32_t                 seed = 1;
    std::vector<std::string> paths;
};

constexpr uint32_t kRowEntries = 4096;

image::PixelRect pictureFor(uint64_t entries) {
    const uint64_t rows = (entries + kRowEntries - 1) / kRowEntries;
    return {0, 0, static_cast<int32_t>(kRowEntries), static_cast<int32_t>(std::max<uint64_t>(rows, 1))};
}

Result<gpu::Buffer> viewOf(gpu_host::Context& context, const image::ImagePtr& image, const char* label) {
    gpu_host::ImageStorage* storage = context.sharedStorage();
    if (storage == nullptr) {
        return Error(ErrorCode::DeviceFailure, "the AOFX host has no device image storage");
    }
    const uint64_t buffer = storage->bufferFor(image->address());
    if (buffer == 0) {
        return Error::make(ErrorCode::DeviceFailure, "the {} picture is on the heap, not the device ({:.1f} MB)",
                           label, static_cast<double>(image->sizeBytes()) / (1024.0 * 1024.0));
    }
    return context.renderView(buffer, image->sizeBytes(), 16, label);
}

Result<gpu::Buffer> wordsOn(gpu::Device& device, const std::vector<uint32_t>& words, const char* label) {
    return gpu::Buffer::fromSpan<uint32_t>(device, words, label);
}

/// A fixture: a box of `size` voxels a side, every voxel active, filled on
/// the device -- `mixture` the sum of known random gaussians, `blob` a smooth
/// noise ball -- with a temperature grid beside its density.
Result<void> makeFixture(gpu::ShaderLibrary& library, const Options& o) {
    gpu::Device& device = library.device();
    const std::filesystem::path dir = std::filesystem::path(o.input).parent_path();
    const std::filesystem::path topology =
        (dir.empty() ? std::filesystem::path(".") : dir) / (std::filesystem::path(o.input).stem().string() + ".topology.vdb");
    const int32_t s = static_cast<int32_t>(o.fixtureSize);
    const io::VdbBox box{{0, 0, 0}, {s, s, s}, 0.0F};
    ATHENEA_TRY(io::writeVdbBoxes(topology, "density", o.voxelSize, std::span<const io::VdbBox>(&box, 1)));
    auto density = io::readVdbGrid(topology, "density");
    if (!density) return std::move(density).error();
    std::filesystem::remove(topology);
    io::NanoGrid temperature = *density;
    temperature.name = "temperature";
    world::PackedGrids packed = world::PackedGrids::of(*density, &temperature, nullptr);
    auto words = wordsOn(device, packed.words, "vdb2splat.fixture");
    if (!words) return std::move(words).error();
    auto splats = world::VolumeSplats::create(library);
    if (!splats) return std::move(splats).error();
    const float centre = static_cast<float>(s) * 0.5F;
    if (o.fixture == "mixture") {
        gpu::BufferDesc desc;
        desc.bytes = uint64_t{o.fixtureGaussians} * 64;
        desc.elementBytes = 16;
        desc.label = "vdb2splat.mixture";
        auto mixture = gpu::Buffer::create(device, desc);
        if (!mixture) return std::move(mixture).error();
        {
            gpu::CommandBatch batch(device);
            splats->randomMixture(batch, *mixture, o.fixtureGaussians,
                                  {static_cast<float>(s), static_cast<float>(s), static_cast<float>(s)},
                                  static_cast<float>(s) * 0.2F, 2.0F, static_cast<float>(s) * 0.08F, o.seed);
            ATHENEA_TRY(batch.submit(true));
        }
        ATHENEA_TRY(splats->splatToGrid(*mixture, o.fixtureGaussians, *words, packed.density, packed.leafCount,
                                        packed.temperature));
    } else {
        gpu::CommandBatch batch(device);
        splats->fixture(batch, *words, packed, 1, {centre, centre, centre}, static_cast<float>(s) * 0.4F, o.seed);
        ATHENEA_TRY(batch.submit(true));
    }
    ATHENEA_TRY(words->read(device, 0, packed.words.size() * 4, packed.words.data()));
    const std::array<io::NanoGrid, 2> grids{packed.unpack(packed.density, *density),
                                            packed.unpack(packed.temperature, temperature)};
    ATHENEA_TRY(io::writeNanoGrids(o.input, grids));
    cli::out("vdb2splat: wrote the %s fixture %s (%d^3 voxels, %u leaves; density and temperature)\n",
             o.fixture.c_str(), o.input.c_str(), s, density->leafCount);
    return ok();
}

void number(aofx_host::EffectJob& job, const char* name, double value) {
    job.params.push_back(aofx::ParamValue{name, {value}, {}});
}

Result<void> convert(gpu_host::Context& context, gpu::ShaderLibrary& library, aofx::Effect& effect,
                     const Options& o) {
    gpu::Device& device = library.device();
    auto names = io::vdbGridNames(o.input);
    if (!names) return std::move(names).error();
    const auto has = [&](const std::string& name) {
        return !name.empty() && std::find(names->begin(), names->end(), name) != names->end();
    };
    if (!has(o.density)) {
        return Error::make(ErrorCode::NotFound, "'{}' has no grid '{}'", o.input, o.density);
    }
    auto density = io::readVdbGrid(o.input, o.density);
    if (!density) return std::move(density).error();
    std::optional<io::NanoGrid> temperature;
    std::optional<io::NanoGrid> colour;
    if (has(o.temperature)) {
        auto read = io::readVdbGrid(o.input, o.temperature);
        if (!read) return std::move(read).error();
        temperature = std::move(*read);
    }
    if (has(o.colour)) {
        auto read = io::readVdbVec3Grid(o.input, o.colour);
        if (!read) return std::move(read).error();
        colour = std::move(*read);
    }
    cli::out("vdb2splat: %s: '%s' %u leaves, voxel %.4g%s%s\n", o.input.c_str(), density->name.c_str(),
             density->leafCount, density->voxelSize, temperature ? ", temperature" : "", colour ? ", Cd" : "");
    if (density->leafCount == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': grid '{}' holds no leaves", o.input, o.density);
    }
    world::PackedGrids packed = world::PackedGrids::of(*density, temperature ? &*temperature : nullptr,
                                                       colour ? &*colour : nullptr);
    auto words = wordsOn(device, packed.words, "vdb2splat.grids");
    if (!words) return std::move(words).error();
    auto splats = world::VolumeSplats::create(library);
    if (!splats) return std::move(splats).error();

    // The leaves, as the effect reads them.
    auto leaves = image::Image::create(pictureFor(uint64_t{packed.leafCount} * world::VolumeSplats::kLeafEntries));
    if (!leaves) return std::move(leaves).error();
    {
        auto view = viewOf(context, *leaves, "vdb2splat.leaves");
        if (!view) return std::move(view).error();
        gpu::CommandBatch batch(device);
        splats->packLeaves(batch, *words, packed, *view);
        ATHENEA_TRY(batch.submit(true));
        (*leaves)->deviceWrote();
    }
    auto records = image::Image::create(pictureFor(uint64_t{o.target} * 4));
    if (!records) return std::move(records).error();

    aofx_host::EffectJob job;
    job.bounds = (*records)->bounds();
    job.inputs.push_back({"Grid", *leaves});
    number(job, "leaves", packed.leafCount);
    number(job, "target", o.target);
    number(job, "iterations", o.iterations);
    number(job, "emIterations", o.emIterations);
    number(job, "attributeIterations", o.attributeIterations);
    number(job, "maxBlock", o.maxBlock);
    number(job, "emissionWeight", o.emissionWeight);
    number(job, "initFraction", o.initFraction);
    number(job, "densifyEvery", o.densifyEvery);
    number(job, "learningRate", o.learningRate);
    number(job, "hasTemperature", temperature ? 1.0 : 0.0);
    number(job, "hasColour", colour ? 1.0 : 0.0);
    auto rendered = aofx_host::renderEffect(context, effect, job);
    if (!rendered) return std::move(rendered).error();
    const std::vector<float>* told = (*rendered)->attached("gaussians");
    if (told == nullptr || told->size() < 8) {
        return Error(ErrorCode::DeviceFailure, "the Vdb2Splat effect did not say how many gaussians it wrote");
    }
    const auto count = static_cast<uint32_t>((*told)[0]);
    cli::out("vdb2splat: %u gaussians (budget %u; the cut began with %u, the coarsest blocks allow %u), blocks to "
             "%u voxels; the fit's relative L2 error %.5f over the active voxels\n",
             count, static_cast<uint32_t>((*told)[1]), static_cast<uint32_t>((*told)[4]),
             static_cast<uint32_t>((*told)[7]), 1u << static_cast<uint32_t>((*told)[6]),
             static_cast<double>((*told)[3]));
    if (count == 0) return Error(ErrorCode::InvalidArgument, "vdb2splat: no gaussian came out");
    auto gaussians = viewOf(context, *rendered, "vdb2splat.records");
    if (!gaussians) return std::move(gaussians).error();

    if (!o.output.empty()) {
        gpu::BufferDesc desc;
        desc.bytes = uint64_t{count} * 19 * 4;
        desc.elementBytes = 4;
        desc.label = "vdb2splat.exported";
        auto exported = gpu::Buffer::create(device, desc);
        if (!exported) return std::move(exported).error();
        {
            gpu::CommandBatch batch(device);
            splats->exportRecords(batch, *gaussians, count, density->indexToWorld, static_cast<float>(o.densityScale),
                                  *exported);
            ATHENEA_TRY(batch.submit(true));
        }
        std::vector<uint32_t> halves(size_t{count} * 5);
        ATHENEA_TRY(exported->read(device, uint64_t{count} * 14 * 4, halves.size() * 4, halves.data()));
        usd::DeviceSplatRecords raw;
        raw.source = o.input;
        raw.count = count;
        raw.records = *exported;
        raw.linear = true;
        io::SplatEncoding& e = raw.encoding;
        e.floatsPerRecord = 14;
        e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
        e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
        e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
        e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
        e.opacity_ = io::SplatEncoding::Opacity::Linear;
        e.scale_ = io::SplatEncoding::Scale::Linear;
        e.colour = io::SplatEncoding::Colour::LinearLight;
        e.rotation = io::SplatEncoding::Rotation::Float;
        usd::ExportOptions options;
        options.maxDegree = 0;
        options.linear = true;
        options.volumeHalves = halves;
        options.volumeTemperature = temperature.has_value();
        ATHENEA_TRY(platform::writeAtomically(o.output, [&](const std::filesystem::path& partial) {
            return usd::writeParticleFieldStage(library, raw, partial, options);
        }));
        cli::out("vdb2splat: wrote %s\n", o.output.c_str());
    }

    if (!o.back.empty() || o.measure) {
        // The way back, on a copy of the input's own words.
        auto returned = wordsOn(device, packed.words, "vdb2splat.back");
        if (!returned) return std::move(returned).error();
        ATHENEA_TRY(splats->splatToGrid(*gaussians, count, *returned, packed.density, packed.leafCount,
                                        packed.temperature));
        if (o.measure) {
            auto error = splats->compare(*returned, *words, packed.density, packed.leafCount);
            if (!error) return std::move(error).error();
            cli::out("vdb2splat: back on the input's topology: relative L2 %.5f, largest difference %.5g\n",
                     error->relativeL2, error->largest);
            auto set = world::VolumeSet::create(library);
            if (!set) return std::move(set).error();
            world::VolumeInput input;
            input.grid = &*density;
            input.densityScale = static_cast<float>(o.densityScale);
            {
                gpu::CommandBatch batch(device);
                ATHENEA_TRY(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
                ATHENEA_TRY(batch.submit(true));
            }
            auto rays = splats->rays(set->words(), *gaussians, count, density->indexToWorld, o.rays, o.raySamples,
                                     o.seed, 0.01F);
            if (!rays) return std::move(rays).error();
            cli::out("vdb2splat: %u random rays: mean |T_mix - T_ratio| %.5f, %u beyond 5 standard errors + 0.01, "
                     "largest optical depth %.3f\n",
                     rays->rays, rays->meanDifference, rays->beyond, rays->largestDepth);
        }
        if (!o.back.empty()) {
            std::vector<uint32_t> back(packed.words.size());
            ATHENEA_TRY(returned->read(device, 0, back.size() * 4, back.data()));
            world::PackedGrids out = packed;
            out.words = std::move(back);
            std::vector<io::NanoGrid> grids{out.unpack(out.density, *density)};
            if (temperature) grids.push_back(out.unpack(out.temperature, *temperature));
            ATHENEA_TRY(platform::writeAtomically(o.back, [&](const std::filesystem::path& partial) {
                return io::writeNanoGrids(partial, grids);
            }));
            cli::out("vdb2splat: wrote %s (the mixture on the input's topology)\n", o.back.c_str());
        }
    }
    return ok();
}

}   // namespace

void addVdb2Splat(CLI::App& app) {
    auto* cmd = app.add_subcommand("vdb2splat", "a .vdb volume into volumetric gaussians, and back onto its grid");
    auto o = std::make_shared<Options>();
    cmd->add_option("input", o->input, "the volume (.vdb); with --make-fixture, the file written")->required();
    cmd->add_option("-o,--output", o->output, "the gaussians, as a ParticleField stage (.usda, .usdc, .usd)");
    cmd->add_option("--back", o->back, "also write the mixture evaluated on the input's topology (.vdb)");
    cmd->add_option("--density", o->density, "the density grid's name");
    cmd->add_option("--temperature", o->temperature, "the temperature grid's name, read where there is one");
    cmd->add_option("--colour", o->colour, "the Cd grid's name (vec3), read where there is one");
    cmd->add_option("-n,--target", o->target, "how many gaussians, at most");
    cmd->add_option("--em-iterations", o->emIterations, "EM steps on the geometry, first");
    cmd->add_option("--iterations", o->iterations, "Adam steps on the geometry, after the EM steps");
    cmd->add_option("--attribute-iterations", o->attributeIterations, "Jacobi steps on temperature and Cd");
    cmd->add_option("--max-block", o->maxBlock, "the largest block a gaussian starts from, voxels a side")
        ->check(CLI::IsMember({8u, 16u, 32u, 64u, 128u}));
    cmd->add_option("--emission-weight", o->emissionWeight, "the temperature's spread beside the density's error");
    cmd->add_option("--init-fraction", o->initFraction, "the share of the budget the cut starts with")
        ->check(CLI::Range(0.01, 1.0));
    cmd->add_option("--densify-every", o->densifyEvery, "iterations between densify-and-prune rounds");
    cmd->add_option("--learning-rate", o->learningRate, "Adam's step");
    cmd->add_option("--density-scale", o->densityScale, "density to extinction a world unit, for --measure's rays "
                                                         "and the exported opacity");
    cmd->add_flag("--measure", o->measure, "print the way back's error and the rays' transmittance error");
    cmd->add_option("--rays", o->rays, "--measure's random rays");
    cmd->add_option("--ray-samples", o->raySamples, "ratio-tracking estimates a ray");
    cmd->add_option("--make-fixture", o->fixture, "write a generated volume to `input` and stop")
        ->check(CLI::IsMember({"mixture", "blob"}));
    cmd->add_option("--fixture-size", o->fixtureSize, "the fixture's voxels a side");
    cmd->add_option("--fixture-gaussians", o->fixtureGaussians, "a mixture fixture's gaussians");
    cmd->add_option("--voxel-size", o->voxelSize, "the fixture's voxel, world units");
    cmd->add_option("--seed", o->seed, "the fixture's and the rays' seed");
    cmd->add_option("--path", o->paths, "extra bundle directories (after $AOFX_PLUGIN_PATH)");
    cmd->callback([o] {
        const auto fail = [](const Error& error) {
            cli::err("vdb2splat: ");
            cli::fail(error);
        };
        if (!io::haveOpenVdb()) fail(Error(ErrorCode::Unsupported, "this build reads no .vdb (no OpenVDB)"));
        gpu_host::Context* context = gpu_host::installProcessContext();
        if (context == nullptr || context->compute() == nullptr) {
            fail(Error(ErrorCode::Unsupported, "no GPU compute device for AOFX kernels (gpe has no backend here)"));
        }
        gpu::ShaderLibrary library(context->deviceShared());
        Result<void> inside = ok();
        if (!o->fixture.empty()) {
            auto ran = context->run([&] { inside = makeFixture(library, *o); });
            if (!ran) fail(ran.error());
            if (!inside) fail(inside.error());
            return;
        }
        aofx_host::EffectRegistry registry;
        for (const std::string& path : o->paths) {
            registry.addSearchPath(path);
        }
        registry.addSearchPath(athenea::platform::executableDir() / ".." / "aofx");
#ifdef ATHENEA_AOFX_BUNDLE_DIR
        registry.addSearchPath(ATHENEA_AOFX_BUNDLE_DIR);
#endif
        registry.scan(context);
        aofx::Effect* effect = registry.find("rt.sparrow.aofx.vdb2splat");
        if (effect == nullptr) {
            fail(Error(ErrorCode::NotFound, "no Vdb2Splat bundle on the AOFX search path (try `athenea aofx list`)"));
        }
        auto ran = context->run([&] { inside = convert(*context, library, *effect, *o); });
        if (!ran) fail(ran.error());
        if (!inside) fail(inside.error());
    });
}

}   // namespace athenea::cli
