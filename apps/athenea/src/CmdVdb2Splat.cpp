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
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "AthvFrames.h"
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
#include "athenea/io/SplatWriters.h"
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
    bool                     noBins = false;
    bool                     profile = false;
    uint32_t                 deviceMb = 2048;   // the device pool's ceiling (unified memory)
    std::string              quantise;   // "half" or "byte": the compact record, then measured from it
    std::string              compactOut; // the compact records as a file
    std::string              against;    // a .vdb of the same topology: its density measured against input's
    double                   binSlack = 0.0;
    bool                     checkerboard = false;
    double                   splitShare = 0.5;
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
    // A sequence: `input` (and `--back`) with a run of '#' standing for the
    // frame number, frames [first, last].
    bool                     sequence = false;
    // A sequence's frames as 3DGS splats (PLY, '#' the frame), every slot;
    // or a .athv of such splats decoded in their place instead of a fit.
    std::string              cloud;
    std::string              fromAthv;
    std::vector<int>         frames{0, 0};
    bool                     independent = false;
    uint32_t                 warmEm = 0;
    uint32_t                 warmIterations = 30;
    double                   dt = 1.0;
    bool                     noAdvect = false;
    double                   warmLearningRate = 0.03;
};

/// `pattern` with its run of '#' replaced by `frame`, zero-padded to the run.
std::string framePath(const std::string& pattern, int frame) {
    const size_t at = pattern.find('#');
    if (at == std::string::npos) return pattern;
    size_t run = 0;
    while (at + run < pattern.size() && pattern[at + run] == '#') ++run;
    std::string number = std::to_string(frame);
    while (number.size() < run) number.insert(number.begin(), '0');
    return pattern.substr(0, at) + number + pattern.substr(at + run);
}

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
/// A sequence fixture: the noise blob carried by a known flow -- a turn
/// about the vertical through it of 2 pi / 120 a frame and a rise of 0.3
/// voxels a frame -- frame by frame, each with its density and its `vel`
/// (world units a frame) on one box topology.
Result<void> makeSequenceFixture(gpu::ShaderLibrary& library, const Options& o) {
    gpu::Device& device = library.device();
    const std::filesystem::path first(framePath(o.input, o.frames[0]));
    const std::filesystem::path topology =
        (first.parent_path().empty() ? std::filesystem::path(".") : first.parent_path()) / "sequence.topology.vdb";
    const int32_t s = static_cast<int32_t>(o.fixtureSize);
    io::VdbGridSpec densitySpec;
    densitySpec.name = "density";
    densitySpec.voxelSize = o.voxelSize;
    densitySpec.boxes = {io::VdbBox{{0, 0, 0}, {s, s, s}, 0.0F}};
    io::VdbGridSpec velSpec = densitySpec;
    velSpec.name = "vel";
    velSpec.type = io::VdbType::Vec3f;
    const std::array<io::VdbGridSpec, 2> specs{densitySpec, velSpec};
    ATHENEA_TRY(io::writeVdbGrids(topology, specs));
    auto density = io::readVdbGrid(topology, "density");
    if (!density) return std::move(density).error();
    auto vel = io::readVdbVec3Grid(topology, "vel");
    if (!vel) return std::move(vel).error();
    std::filesystem::remove(topology);
    auto splats = world::VolumeSplats::create(library);
    if (!splats) return std::move(splats).error();
    const float half = static_cast<float>(s) * 0.5F;
    for (int frame = o.frames[0]; frame <= o.frames[1]; ++frame) {
        world::PackedGrids packed = world::PackedGrids::of(*density, nullptr, &*vel);
        auto words = wordsOn(device, packed.words, "vdb2splat.sequence");
        if (!words) return std::move(words).error();
        {
            gpu::CommandBatch batch(device);
            splats->sequenceFixture(batch, *words, packed, {half, static_cast<float>(s) * 0.4F, half},
                                    static_cast<float>(s) * 0.3F, o.seed, 6.28318531F / 120.0F, 0.3F,
                                    static_cast<float>(frame - o.frames[0]), static_cast<float>(o.voxelSize));
            ATHENEA_TRY(batch.submit(true));
        }
        ATHENEA_TRY(words->read(device, 0, packed.words.size() * 4, packed.words.data()));
        const std::array<io::NanoGrid, 2> grids{packed.unpack(packed.density, *density),
                                                packed.unpack(packed.colour, *vel)};
        ATHENEA_TRY(io::writeNanoGrids(framePath(o.input, frame), grids));
    }
    cli::out("vdb2splat: wrote the blob-sequence fixture %s, frames %d to %d (%d^3 voxels, density and vel)\n",
             o.input.c_str(), o.frames[0], o.frames[1], s);
    return ok();
}

Result<void> makeFixture(gpu::ShaderLibrary& library, const Options& o) {
    if (o.fixture == "blob-sequence") return makeSequenceFixture(library, o);
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

/// The gaussians as a ParticleField stage at `path` (vsExport's records, the
/// medium's halves beside them).
Result<void> writeStage(gpu::ShaderLibrary& library, const world::VolumeSplats& splats, const gpu::Buffer& gaussians,
                        uint32_t count, const io::NanoGrid& density, bool hasTemperature, const Options& o,
                        const std::string& path) {
    gpu::Device& device = library.device();
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{count} * 19 * 4;
    desc.elementBytes = 4;
    desc.label = "vdb2splat.exported";
    auto exported = gpu::Buffer::create(device, desc);
    if (!exported) return std::move(exported).error();
    {
        gpu::CommandBatch batch(device);
        splats.exportRecords(batch, gaussians, count, density.indexToWorld, static_cast<float>(o.densityScale),
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
    options.volumeTemperature = hasTemperature;
    ATHENEA_TRY(platform::writeAtomically(path, [&](const std::filesystem::path& partial) {
        return usd::writeParticleFieldStage(library, raw, partial, options);
    }));
    cli::out("vdb2splat: wrote %s\n", path.c_str());

    return ok();
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
    if (!o.against.empty()) {
        // Another grid of the same topology (a NanoVDB quantisation of the
        // input brought back to floats, say) measured against the input.
        auto other = io::readVdbGrid(o.against, o.density);
        if (!other) return std::move(other).error();
        if (other->leafCount != density->leafCount || other->words.size() != density->words.size()) {
            return Error::make(ErrorCode::InvalidArgument, "'{}' and '{}' do not share a topology ({} and {} leaves)",
                               o.input, o.against, density->leafCount, other->leafCount);
        }
        world::PackedGrids a = world::PackedGrids::of(*density, nullptr, nullptr);
        world::PackedGrids b = world::PackedGrids::of(*other, nullptr, nullptr);
        auto wa = wordsOn(library.device(), a.words, "vdb2splat.input");
        if (!wa) return std::move(wa).error();
        auto wb = wordsOn(library.device(), b.words, "vdb2splat.against");
        if (!wb) return std::move(wb).error();
        auto splats = world::VolumeSplats::create(library);
        if (!splats) return std::move(splats).error();
        auto error = splats->compare(*wb, *wa, a.density, a.leafCount);
        if (!error) return std::move(error).error();
        cli::out("vdb2splat: %s against %s: density relative L2 %.5f, largest difference %.5g\n", o.against.c_str(),
                 o.input.c_str(), error->relativeL2, error->largest);
        return ok();
    }
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
    cli::out("vdb2splat: %s: '%s' %u leaves, voxel %.4g, %.2f MB as NanoVDB%s%s\n", o.input.c_str(),
             density->name.c_str(), density->leafCount, density->voxelSize,
             static_cast<double>(density->words.size()) * 4.0 / 1.0e6, temperature ? ", temperature" : "",
             colour ? ", Cd" : "");
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
    number(job, "binned", o.noBins ? 0.0 : 1.0);
    number(job, "profile", o.profile ? 1.0 : 0.0);
    number(job, "binSlack", o.binSlack);
    number(job, "checkerboard", o.checkerboard ? 1.0 : 0.0);
    number(job, "splitShare", o.splitShare);
    number(job, "attributeIterations", o.attributeIterations);
    number(job, "maxBlock", o.maxBlock);
    number(job, "emissionWeight", o.emissionWeight);
    number(job, "initFraction", o.initFraction);
    number(job, "densifyEvery", o.densifyEvery);
    number(job, "learningRate", o.learningRate);
    number(job, "hasTemperature", temperature ? 1.0 : 0.0);
    number(job, "hasColour", colour ? 1.0 : 0.0);
    const auto started = std::chrono::steady_clock::now();
    auto rendered = aofx_host::renderEffect(context, effect, job);
    if (!rendered) return std::move(rendered).error();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
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
    cli::out("vdb2splat: the fit took %.2f s (the effect, end to end)\n", seconds);
    if (const std::vector<float>* syncs = (*rendered)->attached("syncs"); syncs != nullptr && syncs->size() >= 2) {
        cli::out("vdb2splat: %u readbacks, the kernels loaded in %.1f ms, the leaves' lists built %u times\n",
                 static_cast<uint32_t>((*syncs)[0]), static_cast<double>((*syncs)[1]),
                 syncs->size() >= 3 ? static_cast<uint32_t>((*syncs)[2]) : 0u);
    }
    if (o.profile) {
        // Each kernel's time, waited for one by one: the largest first.
        std::vector<std::pair<double, std::string>> spent;
        double total = 0.0;
        for (const auto& [name, values] : (*rendered)->attachments()) {
            if (name.rfind("profile:", 0) != 0 || values.empty()) continue;
            spent.emplace_back(values[0], name.substr(8));
            total += values[0];
        }
        std::sort(spent.begin(), spent.end(), std::greater<>());
        for (const auto& [ms, name] : spent) {
            cli::out("vdb2splat: profile %-22s %9.1f ms %5.1f %%\n", name.c_str(), ms, 100.0 * ms / std::max(total, 1e-9));
        }
        cli::out("vdb2splat: profile %-22s %9.1f ms\n", "(kernels, waited)", total);
    }
    if (count == 0) return Error(ErrorCode::InvalidArgument, "vdb2splat: no gaussian came out");
    auto gaussians = viewOf(context, *rendered, "vdb2splat.records");
    if (!gaussians) return std::move(gaussians).error();
    if (!o.quantise.empty()) {
        // THE COMPACT RECORD: what follows -- the way back, --measure, -o --
        // reads the records the compact ones decode to.
        const uint32_t mode = o.quantise == "byte" ? 1u : 0u;
        const uint32_t words = mode == 0 ? 6u : 5u;
        gpu::BufferDesc cd;
        cd.bytes = uint64_t{count} * words * 4;
        cd.elementBytes = 4;
        cd.label = "vdb2splat.compact";
        auto compact = gpu::Buffer::create(device, cd);
        if (!compact) return std::move(compact).error();
        gpu::BufferDesc dd;
        dd.bytes = uint64_t{count} * 64;
        dd.elementBytes = 16;
        dd.label = "vdb2splat.dequantised";
        auto dequantised = gpu::Buffer::create(device, dd);
        if (!dequantised) return std::move(dequantised).error();
        auto ranges = splats->quantise(*gaussians, count, mode, *compact, *dequantised);
        if (!ranges) return std::move(ranges).error();
        cli::out("vdb2splat: the compact record, the factor in %s: %u bytes a gaussian, %.2f MB for %u\n",
                 mode == 0 ? "halves" : "bytes", words * 4, count * words * 4.0 / 1.0e6, count);
        if (!o.compactOut.empty()) {
            // The file: a 64-byte header (magic, version, count, mode, the
            // twelve ranges), the index-to-world map as 16 doubles, the words.
            std::vector<uint32_t> body(size_t{count} * words);
            ATHENEA_TRY(compact->read(device, 0, body.size() * 4, body.data()));
            std::vector<std::byte> bytes(64 + 128 + body.size() * 4);
            const uint32_t head[4] = {0x47565441u, 1u, count, mode};   // "ATVG"
            std::memcpy(bytes.data(), head, sizeof head);
            std::memcpy(bytes.data() + 16, ranges->data(), 48);
            std::memcpy(bytes.data() + 64, density->indexToWorld.data(), 128);
            std::memcpy(bytes.data() + 192, body.data(), body.size() * 4);
            ATHENEA_TRY(platform::writeAtomically(o.compactOut, [&](const std::filesystem::path& partial) {
                std::ofstream out(partial, std::ios::binary);
                out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                return out ? ok() : Error(ErrorCode::IoFailure, "vdb2splat: cannot write " + partial.string());
            }));
            cli::out("vdb2splat: wrote %s (%zu bytes)\n", o.compactOut.c_str(), bytes.size());
        }
        gaussians = std::move(*dequantised);
    }

    if (!o.output.empty()) {
        ATHENEA_TRY(writeStage(library, *splats, *gaussians, count, *density, temperature.has_value(), o, o.output));
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
            if (temperature) {
                // Weighed by the density, as the fit weighs it: where there is
                // no density a gaussian's temperature says nothing.
                auto heat = splats->compareWeighted(*returned, *words, packed.density, packed.leafCount,
                                                    packed.temperature);
                if (!heat) return std::move(heat).error();
                auto plain = splats->compare(*returned, *words, packed.temperature, temperature->leafCount);
                if (!plain) return std::move(plain).error();
                cli::out("vdb2splat: the temperature back on its grid: density-weighted relative L2 %.5f (largest "
                         "difference where there is density %.5g); unweighted over its active voxels %.5f\n",
                         heat->relativeL2, heat->largest, plain->relativeL2);
            }
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

/// A SEQUENCE, frame by frame. Warm (the default): each frame starts from
/// the frame before's gaussians in their own slots, carried by the frame
/// before's `vel` where the file has one, and refines them for `warmEm` EM
/// and `warmIterations` Adam steps; the first frame is a full fit.
/// `--independent`: every frame a full fit of its own. Each frame's mixture
/// is given back on its grid and judged there: its density error, the
/// flicker (what its frame-to-frame change gets wrong of the original's),
/// and, slot for slot, how many of the frame before's gaussians are alive
/// in the same slot and how far they moved over their size.
Result<void> runSequence(gpu_host::Context& context, gpu::ShaderLibrary& library, aofx::Effect& effect,
                         const Options& o) {
    gpu::Device& device = library.device();
    auto splats = world::VolumeSplats::create(library);
    if (!splats) return std::move(splats).error();
    // --from-athv: the frames' gaussians from the video, decoded in order.
    std::unique_ptr<AthvFrames> video;
    if (!o.fromAthv.empty()) {
        auto opened = AthvFrames::open(device, library, o.fromAthv);
        if (!opened) return std::move(opened).error();
        video = std::move(*opened);
    }
    std::optional<gpu::Buffer> decodedBefore;
    image::ImagePtr before;          // the frame before's output, slot for slot
    image::ImagePtr flowBefore;      // its flow, as the effect reads it
    std::optional<gpu::Buffer> backBefore, originalBefore;
    world::PackedGrids packedBefore;
    double errorSum = 0.0, flickerSum = 0.0, keptSum = 0.0, secondsSum = 0.0;
    uint32_t frames = 0, judged = 0;
    for (int frame = o.frames[0]; frame <= o.frames[1]; ++frame) {
        const std::string path = framePath(o.input, frame);
        auto density = io::readVdbGrid(path, o.density);
        if (!density) return std::move(density).error();
        auto names = io::vdbGridNames(path);
        if (!names) return std::move(names).error();
        std::optional<io::NanoGrid> vel;
        if (std::find(names->begin(), names->end(), "vel") != names->end()) {
            auto read = io::readVdbVec3Grid(path, "vel");
            if (!read) return std::move(read).error();
            vel = std::move(*read);
        }
        world::PackedGrids packed = world::PackedGrids::of(*density, nullptr, vel ? &*vel : nullptr);
        auto words = wordsOn(device, packed.words, "vdb2splat.frame");
        if (!words) return std::move(words).error();
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
        gpu::Buffer gaussians;
        image::ImagePtr rendered;
        uint32_t slotsOut = 0, alive = 0;
        double seconds = 0.0;
        if (video) {
            // The frame's splats as the video gave them, into records.
            const auto started = std::chrono::steady_clock::now();
            auto count = video->next();
            if (!count) return std::move(count).error();
            const scene::GpuSplats& store = video->store();
            gpu::BufferDesc desc;
            desc.bytes = std::max<uint64_t>(uint64_t{*count} * 64, 64);
            desc.elementBytes = 16;
            desc.label = "vdb2splat.decodedRecords";
            auto made = gpu::Buffer::create(device, desc);
            if (!made) return std::move(made).error();
            gaussians = std::move(*made);
            gpu::CommandBatch batch(device);
            splats->importSplats(batch, store.positions, store.shape, store.sh, store.shWords, video->restPerColour(),
                                 *count, density->indexToWorld, gaussians);
            ATHENEA_TRY(batch.submit(true));
            seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            slotsOut = *count;
            alive = *count;
        } else {
            const bool warm = !o.independent && before != nullptr;
            auto records = image::Image::create(pictureFor(uint64_t{o.target} * 4));
            if (!records) return std::move(records).error();
            aofx_host::EffectJob job;
            job.bounds = (*records)->bounds();
            job.inputs.push_back({"Grid", *leaves});
            if (warm) {
                job.inputs.push_back({"Previous", before});
                if (flowBefore && !o.noAdvect) job.inputs.push_back({"Velocity", flowBefore});
            }
            number(job, "leaves", packed.leafCount);
            number(job, "target", o.target);
            number(job, "iterations", warm ? o.warmIterations : o.iterations);
            number(job, "emIterations", warm ? o.warmEm : o.emIterations);
            number(job, "maxBlock", o.maxBlock);
            number(job, "initFraction", o.initFraction);
            number(job, "densifyEvery", o.densifyEvery);
            number(job, "learningRate", warm ? o.warmLearningRate : o.learningRate);
            number(job, "splitShare", o.splitShare);
            number(job, "keepSlots", 1.0);
            number(job, "dt", o.dt);
            const auto started = std::chrono::steady_clock::now();
            auto renderedOr = aofx_host::renderEffect(context, effect, job);
            if (!renderedOr) return std::move(renderedOr).error();
            rendered = *renderedOr;
            seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            const std::vector<float>* told = rendered->attached("gaussians");
            if (told == nullptr || told->size() < 9) {
                return Error(ErrorCode::DeviceFailure, "the Vdb2Splat effect did not say how many gaussians it wrote");
            }
            slotsOut = static_cast<uint32_t>((*told)[0]);
            alive = static_cast<uint32_t>((*told)[8]);
            auto gaussiansView = viewOf(context, rendered, "vdb2splat.records");
            if (!gaussiansView) return std::move(gaussiansView).error();
            gaussians = std::move(*gaussiansView);
        }
        auto returned = wordsOn(device, packed.words, "vdb2splat.back");
        if (!returned) return std::move(returned).error();
        ATHENEA_TRY(splats->splatToGrid(gaussians, slotsOut, *returned, packed.density, packed.leafCount));
        auto error = splats->compare(*returned, *words, packed.density, packed.leafCount);
        if (!error) return std::move(error).error();
        const bool warm = !video && !o.independent && before != nullptr;
        std::string line = "vdb2splat: frame " + std::to_string(frame) + (video ? " (decoded)" : warm ? " (warm)" : "") + ": " +
                           std::to_string(alive) + " gaussians, density relative L2 " +
                           std::to_string(error->relativeL2) + (video ? ", decode " : ", fit ") +
                           std::to_string(seconds) + " s, NanoVDB " + std::to_string(packed.words.size() * 4) + " B";
        if (backBefore && packedBefore.leafCount == packed.leafCount && packedBefore.words.size() == packed.words.size()) {
            auto flick = splats->flicker(*returned, *words, *backBefore, *originalBefore, packed.density,
                                         packed.leafCount);
            if (!flick) return std::move(flick).error();
            gpu::Buffer before0;
            if (video) {
                before0 = *decodedBefore;
            } else {
                auto view = viewOf(context, before, "vdb2splat.before");
                if (!view) return std::move(view).error();
                before0 = std::move(*view);
            }
            auto kept = splats->slots(before0, gaussians, slotsOut);
            if (!kept) return std::move(kept).error();
            const double share = kept->aliveBefore > 0 ? double(kept->aliveBoth) / kept->aliveBefore : 0.0;
            line += ", flicker " + std::to_string(*flick) + ", slots kept " + std::to_string(kept->aliveBoth) + "/" +
                    std::to_string(kept->aliveBefore) + " (" + std::to_string(share) + "), mean move " +
                    std::to_string(kept->meanMove) + " deviations";
            flickerSum += *flick;
            keptSum += share;
            ++judged;
        }
        cli::out("%s\n", line.c_str());
        errorSum += error->relativeL2;
        secondsSum += seconds;
        ++frames;
        if (!o.back.empty()) {
            std::vector<uint32_t> back(packed.words.size());
            ATHENEA_TRY(returned->read(device, 0, back.size() * 4, back.data()));
            world::PackedGrids out = packed;
            out.words = std::move(back);
            const std::array<io::NanoGrid, 1> grids{out.unpack(out.density, *density)};
            ATHENEA_TRY(io::writeNanoGrids(framePath(o.back, frame), grids));
        }
        if (!o.output.empty()) {
            ATHENEA_TRY(writeStage(library, *splats, gaussians, slotsOut, *density, false, o, framePath(o.output, frame)));
        }
        if (!o.cloud.empty()) {
            // Every slot as a 3DGS splat (a dead one a speck), for the video.
            gpu::BufferDesc desc;
            desc.bytes = std::max<uint64_t>(uint64_t{slotsOut} * 62 * 4, 4);
            desc.elementBytes = 4;
            desc.label = "vdb2splat.ply";
            auto ply = gpu::Buffer::create(device, desc);
            if (!ply) return std::move(ply).error();
            gpu::CommandBatch batch(device);
            splats->exportPly(batch, gaussians, slotsOut, density->indexToWorld, static_cast<float>(o.densityScale),
                              *ply);
            ATHENEA_TRY(batch.submit(true));
            std::vector<std::vector<float>> rows(1, std::vector<float>(size_t{slotsOut} * 62));
            ATHENEA_TRY(ply->read(device, 0, rows[0].size() * 4, rows[0].data()));
            const std::vector<std::string> comments{
                "athenea vdb2splat: a medium of gaussians, f_rest_0 ln sigma, f_rest_1 temperature"};
            ATHENEA_TRY(io::writePly3dgs(framePath(o.cloud, frame), rows, comments));
        }
        // What the next frame starts from: this frame's slots, and its flow.
        before = rendered;
        if (video) decodedBefore = gaussians;
        flowBefore = nullptr;
        if (vel) {
            auto flow = image::Image::create(pictureFor(uint64_t{packed.leafCount} * world::VolumeSplats::kLeafEntries));
            if (!flow) return std::move(flow).error();
            auto view = viewOf(context, *flow, "vdb2splat.flow");
            if (!view) return std::move(view).error();
            gpu::CommandBatch batch(device);
            splats->packVelocity(batch, *words, packed, static_cast<float>(1.0 / density->voxelSize), *view);
            ATHENEA_TRY(batch.submit(true));
            (*flow)->deviceWrote();
            flowBefore = *flow;
        }
        backBefore = std::move(*returned);
        originalBefore = std::move(*words);
        packedBefore = std::move(packed);
    }
    cli::out("vdb2splat: %u frames (%s): mean density relative L2 %.5f, mean flicker %.5f, mean slots kept %.4f, "
             "mean fit %.2f s a frame\n",
             frames, video ? "decoded" : o.independent ? "independent" : "warm", frames ? errorSum / frames : 0.0,
             judged ? flickerSum / judged : 0.0, judged ? keptSum / judged : 0.0, frames ? secondsSum / frames : 0.0);
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
    cmd->add_option("--split-share", o->splitShare, "a densify round's share of new slots that split the worst "
                                                      "gaussians (the rest are born at the worst voxels)")
        ->check(CLI::Range(0.0, 1.0));
    cmd->add_flag("--no-bins", o->noBins, "the forward pass over every gaussian for every leaf (for a timing)");
    cmd->add_option("--bin-slack", o->binSlack, "voxels a box is widened by in the leaves' lists, which are then "
                    "rebuilt only when a box leaves them (0: every iteration)");
    cmd->add_flag("--checkerboard", o->checkerboard, "each Adam step's gradient over half the voxels, alternating");
    cmd->add_option("--quantise", o->quantise, "the compact record, measured from it: half (24 B a gaussian, the "
                    "factor in halves) or byte (20 B)")->check(CLI::IsMember({"half", "byte"}));
    cmd->add_option("--compact", o->compactOut, "with --quantise: the compact records as a file");
    cmd->add_option("--against", o->against, "measure input's density against this .vdb's (the same topology) and "
                    "stop");
    cmd->add_option("--device-mb", o->deviceMb, "the device pool's ceiling in MB on unified memory: the fit's working "
                    "buffers (about 60 bytes a voxel of the leaves, and the leaves' lists) come out of it");
    cmd->add_flag("--profile", o->profile, "wait for every kernel and print each one's time (slower; a breakdown)");
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
        ->check(CLI::IsMember({"mixture", "blob", "blob-sequence"}));
    cmd->add_option("--fixture-size", o->fixtureSize, "the fixture's voxels a side");
    cmd->add_option("--fixture-gaussians", o->fixtureGaussians, "a mixture fixture's gaussians");
    cmd->add_option("--voxel-size", o->voxelSize, "the fixture's voxel, world units");
    cmd->add_option("--seed", o->seed, "the fixture's and the rays' seed");
    cmd->add_option("--path", o->paths, "extra bundle directories (after $AOFX_PLUGIN_PATH)");
    cmd->add_option("--cloud", o->cloud, "a sequence: each frame's gaussians, every slot, as a 3DGS PLY (a run of "
                    "'#' the frame), ln sigma and the temperature in its first rest coefficients of red");
    cmd->add_option("--from-athv", o->fromAthv, "a sequence: each frame's gaussians decoded from this .athv (of "
                    "--cloud's splats) instead of fitted, then judged as a fit is");
    cmd->add_flag("--sequence", o->sequence, "input (and --back) name frames by a run of '#'; --frames first last");
    cmd->add_option("--frames", o->frames, "a sequence's first and last frame")->expected(2);
    cmd->add_flag("--independent", o->independent, "a sequence: every frame a full fit of its own, no warm start");
    cmd->add_option("--warm-em", o->warmEm, "a sequence: EM steps of a warm frame");
    cmd->add_option("--warm-iterations", o->warmIterations, "a sequence: Adam steps of a warm frame");
    cmd->add_flag("--no-advect", o->noAdvect, "a sequence: warm frames start where the frame before left them");
    cmd->add_option("--warm-learning-rate", o->warmLearningRate, "a sequence: Adam's step on a warm frame");
    cmd->add_option("--dt", o->dt, "a sequence: frames the flow carries the gaussians over, frame to frame");
    cmd->callback([o] {
        const auto fail = [](const Error& error) {
            cli::err("vdb2splat: ");
            cli::fail(error);
        };
        if (!io::haveOpenVdb()) fail(Error(ErrorCode::Unsupported, "this build reads no .vdb (no OpenVDB)"));
        gpu_host::ContextDesc desc;
        desc.budgetBytes = size_t{o->deviceMb} << 20;
        gpu_host::Context* context = gpu_host::installProcessContext(desc);
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
        auto ran = context->run([&] {
            inside = o->sequence ? runSequence(*context, library, *effect, *o) : convert(*context, library, *effect, *o);
        });
        if (!ran) fail(ran.error());
        if (!inside) fail(inside.error());
    });
}

}   // namespace athenea::cli
