// Copyright (c) 2026 jesus luque.
//
// The splat bake filter, as an AOFX effect: a baked cloud's indirect light in,
// the same light averaged between neighbouring gaussians on their surface out
// (splatbakefilter.slang says how and why). This file is the declaration and
// the plumbing.
//
// WHY AN EFFECT. It makes new data out of a cloud's records and nothing else
// -- positions, normals, ids, the light and its noise -- which is what an
// effect is for (mesh2splat is one for the same reason), and as one it runs
// unchanged wherever an AOFX host does: `athenea mesh2splat` hands it the bake
// it has just taken, and a compositor could hand it any cloud's.
//
// A cloud is not a picture, so the host packs it into two: three entries a
// gaussian of what the filter weighs by, and the harmonics of the indirect
// light, `coefficients` entries a gaussian. The answer is a picture of the
// second's shape. The host also says how many gaussians there are; what comes
// back attached is how many a full cell left out and the cell it settled on.
#include <aofx/Effect.h>
#include <aofx/Entry.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

#include "aofx_kernels_splatbakefilter.h"

namespace {

/// Must match `BfParams` in splatbakefilter.slang exactly.
struct FilterUniforms {
    uint32_t count = 0;
    uint32_t coefficients = 1;
    uint32_t pointsWidth = 0;
    uint32_t pointsStride = 0;

    uint32_t indirectWidth = 0;
    uint32_t indirectStride = 0;
    uint32_t dstWidth = 0;
    uint32_t dstStride = 0;

    uint32_t gridSize = 0;
    uint32_t slots = 16;
    uint32_t iteration = 0;
    uint32_t iterations = 3;

    float    sigmaNormal = 32.0F;
    float    sigmaLuminance = 4.0F;
    float    cellScale = 1.5F;
    uint32_t pad0 = 0;
};
static_assert(sizeof(FilterUniforms) == 64, "must match BfParams exactly");

/// Counters the kernels keep: the footprints by quarter octave, then how many
/// were left out of a full cell, then the cell's bits -- in float4 pixels.
constexpr int kCounterPixels = (514 + 3) / 4;

class SplatBakeFilter final : public aofx::Effect {
public:
    void describe(aofx::EffectDesc& into) override {
        into.identifier = "rt.sparrow.aofx.splatbakefilter";
        into.label = "Splat Bake Filter";
        into.grouping = "Convert";
        into.description =
            "Averages a baked cloud's indirect light between neighbouring gaussians on their surface "
            "(a-trous over a hash grid), weighed by tangent-plane distance, normal, Cryptomatte id and "
            "each gaussian's noise, so the grain goes and the edges stay.";

        const auto clip = [&into](const char* name, const char* label) {
            aofx::ClipDesc desc;
            desc.name = name;
            desc.label = label;
            desc.optional = false;
            into.inputs.push_back(desc);
        };
        // Numbers, not pictures: nothing upstream should be asked to show them.
        clip("Points", "Points");
        clip("Indirect", "Indirect light");

        aofx::ParamDesc count;
        count.name = "count";
        count.label = "Gaussians";
        count.hint = "How many gaussians the pictures hold. The host sets it.";
        count.type = aofx::ParamType::Integer;
        count.defaults = {0.0};
        count.hardMin = {0.0};
        into.params.push_back(count);

        aofx::ParamDesc coefficients;
        coefficients.name = "coefficients";
        coefficients.label = "Harmonics";
        coefficients.hint = "Entries a gaussian of the Indirect picture: 1, 4, 9 or 16.";
        coefficients.type = aofx::ParamType::Integer;
        coefficients.defaults = {1.0};
        coefficients.hardMin = {1.0};
        coefficients.hardMax = {16.0};
        into.params.push_back(coefficients);

        aofx::ParamDesc iterations;
        iterations.name = "iterations";
        iterations.label = "Iterations";
        iterations.hint = "A-trous passes; each doubles the reach, the first a cell of a gaussian and a half.";
        iterations.type = aofx::ParamType::Integer;
        iterations.defaults = {3.0};
        iterations.hardMin = {1.0};
        iterations.hardMax = {8.0};
        into.params.push_back(iterations);

        aofx::ParamDesc sigmaNormal;
        sigmaNormal.name = "sigmaNormal";
        sigmaNormal.label = "Normal exponent";
        sigmaNormal.hint = "A neighbour counts by the cosine between the two normals to this power.";
        sigmaNormal.type = aofx::ParamType::Double;
        sigmaNormal.defaults = {32.0};
        sigmaNormal.hardMin = {0.0};
        into.params.push_back(sigmaNormal);

        aofx::ParamDesc sigmaLuminance;
        sigmaLuminance.name = "sigmaLuminance";
        sigmaLuminance.label = "Luminance edge";
        sigmaLuminance.hint =
            "A neighbour whose indirect light differs by this many of this gaussian's standard "
            "deviations counts e^-1 as much.";
        sigmaLuminance.type = aofx::ParamType::Double;
        sigmaLuminance.defaults = {4.0};
        sigmaLuminance.hardMin = {0.0};
        into.params.push_back(sigmaLuminance);

        aofx::ParamDesc cellScale;
        cellScale.name = "cellScale";
        cellScale.label = "Cell";
        cellScale.hint = "The grid's cell, in median gaussian footprints.";
        cellScale.type = aofx::ParamType::Double;
        cellScale.defaults = {1.5};
        cellScale.hardMin = {0.25};
        into.params.push_back(cellScale);
    }

    [[nodiscard]] std::vector<aofx::KernelDesc> kernels() const override {
        return {aofx::KernelDesc{"bfClear", "bfClear", k_splatbakefilter, k_splatbakefilterBytes},
                aofx::KernelDesc{"bfMeasure", "bfMeasure", k_splatbakefilter, k_splatbakefilterBytes},
                aofx::KernelDesc{"bfCell", "bfCell", k_splatbakefilter, k_splatbakefilterBytes},
                aofx::KernelDesc{"bfInsert", "bfInsert", k_splatbakefilter, k_splatbakefilterBytes},
                aofx::KernelDesc{"bfIterate", "bfIterate", k_splatbakefilter, k_splatbakefilterBytes}};
    }

    bool process(const aofx::RenderRequest& request) override {
        const aofx::InputPlane*  pointsPlane = request.input("Points");
        const aofx::InputPlane*  indirectPlane = request.input("Indirect");
        const aofx::OutputPlane* target = request.output("Color");
        if (pointsPlane == nullptr || indirectPlane == nullptr || target == nullptr ||
            !pointsPlane->buffer.isValid() || !indirectPlane->buffer.isValid() || !target->buffer.isValid() ||
            request.gpu == nullptr) {
            request.complaint = "the splat bake filter needs its Points, its Indirect light and somewhere to write";
            return false;
        }
        FilterUniforms uniforms;
        uniforms.count = static_cast<uint32_t>(std::max(request.number("count", 0.0), 0.0));
        if (uniforms.count == 0) {
            request.complaint = "the splat bake filter was given no gaussians: the host sets 'count'";
            return false;
        }
        uniforms.coefficients =
            static_cast<uint32_t>(std::clamp(request.number("coefficients", 1.0), 1.0, 16.0));
        uniforms.iterations = static_cast<uint32_t>(std::clamp(request.number("iterations", 3.0), 1.0, 8.0));
        uniforms.sigmaNormal = static_cast<float>(std::max(request.number("sigmaNormal", 32.0), 0.0));
        uniforms.sigmaLuminance = static_cast<float>(std::max(request.number("sigmaLuminance", 4.0), 0.0));
        uniforms.cellScale = static_cast<float>(std::max(request.number("cellScale", 1.5), 0.25));
        uniforms.pointsWidth = static_cast<uint32_t>(pointsPlane->buffer.width);
        uniforms.pointsStride = static_cast<uint32_t>(pointsPlane->buffer.stride);
        uniforms.indirectWidth = static_cast<uint32_t>(indirectPlane->buffer.width);
        uniforms.indirectStride = static_cast<uint32_t>(indirectPlane->buffer.stride);
        uniforms.dstWidth = static_cast<uint32_t>(target->buffer.width);
        uniforms.dstStride = static_cast<uint32_t>(target->buffer.stride);
        const uint64_t entries = uint64_t{uniforms.count} * uniforms.coefficients;
        const uint64_t room = static_cast<uint64_t>(target->buffer.width) * target->buffer.height;
        if (room < entries || static_cast<uint64_t>(pointsPlane->buffer.width) * pointsPlane->buffer.height <
                                 uint64_t{uniforms.count} * 3) {
            request.complaint = "the splat bake filter's pictures are too small for the gaussians they say they hold";
            return false;
        }
        // A cell a gaussian or so: the hash as many cells as there are
        // gaussians, rounded up to a power of two.
        uniforms.gridSize = std::bit_ceil(std::max<uint32_t>(uniforms.count, 1024));

        // Working space, in float4 pixels, in rows no wider than a picture
        // comfortably is.
        const auto scratchOf = [&](uint64_t quads) {
            const auto width = static_cast<int>(std::min<uint64_t>(std::max<uint64_t>(quads, 1), 4096));
            const auto height = static_cast<int>((std::max<uint64_t>(quads, 1) + width - 1) / width);
            return request.gpu->scratch(width, height);
        };
        const aofx::Buffer grid = scratchOf((uint64_t{uniforms.gridSize} * (uniforms.slots + 1) + 3) / 4);
        const aofx::Buffer ping = scratchOf(uniforms.iterations > 1 ? entries : 1);
        const aofx::Buffer pong = scratchOf(uniforms.iterations > 2 ? entries : 1);
        const aofx::Buffer variance = scratchOf((uint64_t{uniforms.count} * 4 + 3) / 4);
        const aofx::Buffer counters = request.gpu->scratch(kCounterPixels, 1);
        if (!grid.isValid() || !ping.isValid() || !pong.isValid() || !variance.isValid() || !counters.isValid()) {
            request.complaint = "the splat bake filter could not take the working space it needs";
            return false;
        }

        const aofx::KernelId clear = request.gpu->load("bfClear");
        const aofx::KernelId measure = request.gpu->load("bfMeasure");
        const aofx::KernelId cell = request.gpu->load("bfCell");
        const aofx::KernelId insert = request.gpu->load("bfInsert");
        const aofx::KernelId iterate = request.gpu->load("bfIterate");
        if (clear == aofx::kInvalidKernel || measure == aofx::kInvalidKernel || cell == aofx::kInvalidKernel ||
            insert == aofx::kInvalidKernel || iterate == aofx::kInvalidKernel) {
            return false;
        }
        const std::vector<aofx::Buffer> buffers{pointsPlane->buffer, indirectPlane->buffer, target->buffer, grid,
                                                ping, pong, variance, counters};
        const uint32_t clearThreads = std::max<uint32_t>(uniforms.gridSize, 514);
        if (!request.gpu->run(clear, aofx::Grid{clearThreads, 1, 1}, buffers, &uniforms, sizeof(uniforms)) ||
            !request.gpu->run(measure, aofx::Grid{uniforms.count, 1, 1}, buffers, &uniforms, sizeof(uniforms)) ||
            !request.gpu->run(cell, aofx::Grid{1, 1, 1}, buffers, &uniforms, sizeof(uniforms)) ||
            !request.gpu->run(insert, aofx::Grid{uniforms.count, 1, 1}, buffers, &uniforms, sizeof(uniforms))) {
            return false;
        }
        for (uint32_t pass = 0; pass < uniforms.iterations; ++pass) {
            uniforms.iteration = pass;
            if (!request.gpu->run(iterate, aofx::Grid{uniforms.count, 1, 1}, buffers, &uniforms,
                                  sizeof(uniforms))) {
                return false;
            }
        }
        uint32_t counted[kCounterPixels * 4] = {};
        if (!request.gpu->read(counters, counted, sizeof(counted))) {
            request.complaint = "the splat bake filter could not read back what it counted";
            return false;
        }
        float cellSize = 0.0F;
        cellSize = std::bit_cast<float>(counted[513]);
        // What went out: how many gaussians, how many a full cell left out
        // of the grid (each still filtered, by the neighbours that reach it),
        // and the cell.
        request.attach("filtered", {static_cast<float>(uniforms.count), static_cast<float>(counted[512]), cellSize});
        return true;
    }
};

}   // namespace

AOFX_EXPORT_EFFECTS(SplatBakeFilter)
