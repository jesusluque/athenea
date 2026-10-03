// Copyright (c) 2026 jesus luque.
//
// The splat transfer zonal fit, as an AOFX effect: a baked transfer's nine
// harmonics in the world in, one or two zonal lobes in each gaussian's own
// frame out (splattransferzonal.slang says how and why). This file is the
// declaration and the plumbing.
//
// WHY AN EFFECT. It makes new data out of a cloud's records and nothing else
// -- the bake's harmonics, its bits and the frame each gaussian had -- which is
// what an effect is for (the splat bake filter is one for the same reason),
// and as one it runs unchanged wherever an AOFX host does: `athenea mesh2splat
// --transfer` hands it the bake it has just taken.
//
// A cloud is not a picture, so the host packs it into one: four entries a
// gaussian. The answer is three entries a gaussian; what comes back attached
// is how many gaussians were fitted and the histogram of the fit's relative
// error against the nine harmonics.
#include <aofx/Effect.h>
#include <aofx/Entry.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "aofx_kernels_splattransferzonal.h"

namespace {

/// Must match `TzParams` in splattransferzonal.slang exactly.
struct ZonalUniforms {
    uint32_t count = 0;
    uint32_t lobes = 2;
    uint32_t inWidth = 0;
    uint32_t inStride = 0;

    uint32_t outWidth = 0;
    uint32_t outStride = 0;
    uint32_t rebin = 1;
    uint32_t pad0 = 0;
};
static_assert(sizeof(ZonalUniforms) == 32, "must match TzParams exactly");

/// The error's histogram by quarter octave, then the gaussians fitted.
constexpr uint32_t kBuckets = 48;
constexpr uint32_t kCounters = kBuckets + 1;
constexpr int      kCounterPixels = (kCounters + 3) / 4;

class SplatTransferZonal final : public aofx::Effect {
public:
    void describe(aofx::EffectDesc& into) override {
        into.identifier = "rt.sparrow.aofx.splattransferzonal";
        into.label = "Splat Transfer Zonal";
        into.grouping = "Convert";
        into.description =
            "Fits a baked transfer's nine harmonics with one or two zonal lobes whose axes are written in "
            "each gaussian's own frame, so the transfer turns with the gaussian (a cloud a skeleton carries); "
            "the shadow bits are laid out in that frame too.";

        aofx::ClipDesc transfer;
        transfer.name = "Transfer";
        transfer.label = "Transfer";
        transfer.optional = false;
        into.inputs.push_back(transfer);

        aofx::ParamDesc count;
        count.name = "count";
        count.label = "Gaussians";
        count.hint = "How many gaussians the picture holds. The host sets it.";
        count.type = aofx::ParamType::Integer;
        count.defaults = {0.0};
        count.hardMin = {0.0};
        into.params.push_back(count);

        aofx::ParamDesc lobes;
        lobes.name = "lobes";
        lobes.label = "Lobes";
        lobes.hint = "Zonal lobes a gaussian: 1 or 2.";
        lobes.type = aofx::ParamType::Integer;
        lobes.defaults = {2.0};
        lobes.hardMin = {1.0};
        lobes.hardMax = {2.0};
        into.params.push_back(lobes);

        aofx::ParamDesc rebin;
        rebin.name = "rebin";
        rebin.label = "Bits in the frame";
        rebin.hint = "1: the sixty-four shadow bits are laid out over the gaussian's own frame; 0: left as traced.";
        rebin.type = aofx::ParamType::Integer;
        rebin.defaults = {1.0};
        rebin.hardMin = {0.0};
        rebin.hardMax = {1.0};
        into.params.push_back(rebin);
    }

    [[nodiscard]] std::vector<aofx::KernelDesc> kernels() const override {
        return {aofx::KernelDesc{"tzClear", "tzClear", k_splattransferzonal, k_splattransferzonalBytes},
                aofx::KernelDesc{"tzFit", "tzFit", k_splattransferzonal, k_splattransferzonalBytes}};
    }

    bool process(const aofx::RenderRequest& request) override {
        const aofx::InputPlane*  transferPlane = request.input("Transfer");
        const aofx::OutputPlane* target = request.output("Color");
        if (transferPlane == nullptr || target == nullptr || !transferPlane->buffer.isValid() ||
            !target->buffer.isValid() || request.gpu == nullptr) {
            request.complaint = "the splat transfer zonal fit needs its Transfer and somewhere to write";
            return false;
        }
        ZonalUniforms uniforms;
        uniforms.count = static_cast<uint32_t>(std::max(request.number("count", 0.0), 0.0));
        if (uniforms.count == 0) {
            request.complaint = "the splat transfer zonal fit was given no gaussians: the host sets 'count'";
            return false;
        }
        uniforms.lobes = static_cast<uint32_t>(std::clamp(request.number("lobes", 2.0), 1.0, 2.0));
        uniforms.rebin = request.number("rebin", 1.0) > 0.5 ? 1u : 0u;
        uniforms.inWidth = static_cast<uint32_t>(transferPlane->buffer.width);
        uniforms.inStride = static_cast<uint32_t>(transferPlane->buffer.stride);
        uniforms.outWidth = static_cast<uint32_t>(target->buffer.width);
        uniforms.outStride = static_cast<uint32_t>(target->buffer.stride);
        const uint64_t in = static_cast<uint64_t>(transferPlane->buffer.width) * transferPlane->buffer.height;
        const uint64_t out = static_cast<uint64_t>(target->buffer.width) * target->buffer.height;
        if (in < uint64_t{uniforms.count} * 4 || out < uint64_t{uniforms.count} * 3) {
            request.complaint = "the splat transfer zonal fit's pictures are too small for the gaussians they say they hold";
            return false;
        }
        const aofx::Buffer counters = request.gpu->scratch(kCounterPixels, 1);
        if (!counters.isValid()) {
            request.complaint = "the splat transfer zonal fit could not take the working space it needs";
            return false;
        }
        const aofx::KernelId clear = request.gpu->load("tzClear");
        const aofx::KernelId fit = request.gpu->load("tzFit");
        if (clear == aofx::kInvalidKernel || fit == aofx::kInvalidKernel) {
            return false;
        }
        const std::vector<aofx::Buffer> buffers{transferPlane->buffer, target->buffer, counters};
        if (!request.gpu->run(clear, aofx::Grid{kCounters, 1, 1}, buffers, &uniforms, sizeof(uniforms)) ||
            !request.gpu->run(fit, aofx::Grid{uniforms.count, 1, 1}, buffers, &uniforms, sizeof(uniforms))) {
            return false;
        }
        uint32_t counted[kCounterPixels * 4] = {};
        if (!request.gpu->read(counters, counted, sizeof(counted))) {
            request.complaint = "the splat transfer zonal fit could not read back what it counted";
            return false;
        }
        // What went out: how many gaussians were fitted, then how many fell in
        // each quarter octave of relative error from 2^-10 up.
        std::vector<float> said;
        said.reserve(kBuckets + 1);
        said.push_back(static_cast<float>(counted[kBuckets]));
        for (uint32_t b = 0; b < kBuckets; ++b) {
            said.push_back(static_cast<float>(counted[b]));
        }
        request.attach("fitted", std::move(said));
        return true;
    }
};

}   // namespace

AOFX_EXPORT_EFFECTS(SplatTransferZonal)
