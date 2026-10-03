// Copyright (c) 2026 jesus luque.
//
// Two pictures measured against each other, on the device they are on.
//
// A QC node: Source against Reference -- a render against its golden, a
// delivery against what was approved -- and the numbers hung on the output for
// whatever reads them. The picture it writes is where they differ, as a
// heatmap; the numbers are the point.
//
// THE NUMBERS ARE ATHENEA'S
//
// The same quantities `athenea compare` has always printed, from the same
// arithmetic (`render::imageStats`, `compareHdr`, `compareImages`): each
// picture's mean and largest value per channel inside a window, the relative
// squared error, the 99th percentile and the largest of the per-pixel
// relative difference, and the same two of the largest 8-bit sRGB code-value
// difference with how many pixels are over a threshold. `athenea compare` is
// now this effect run through the AOFX host, so there is one implementation
// and the compositor's node and the engine's command cannot disagree.
//
// WHAT IS ATTACHED
//
// Four ids, floats all, a count written as two floats `high * 2^24 + low` so
// that it is exact past what one float holds (high is 0 below 16.7 million):
//
//   "source"     mean r g b a, max r g b a, sum r g b a, window pixels high, low
//   "reference"  the same for Reference (only when it is wired)
//   "hdr"        relMSE, p99 relative, max relative, the error's sum, pixels high, low
//   "codes"      p99, max, over high, low, threshold, pixels high, low
//
// The sums are there for a host that divides in double, as athenea's printer
// always has: a mean carried as a float is the same number rounded once more.
// The differences are over the whole picture -- they are distributions -- and
// the window is for the means.

#include <aofx/Effect.h>
#include <aofx/Entry.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "aofx_kernels_measure.h"

namespace {

/// Must match `MeasureParams` in measure.slang exactly.
struct MeasureUniforms {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t srcStride = 0;
    uint32_t refStride = 0;
    uint32_t hasReference = 0;
    uint32_t x0 = 0;
    uint32_t y0 = 0;
    uint32_t x1 = 0;
    uint32_t y1 = 0;
    uint32_t rowsStride = 0;
    uint32_t histStride = 0;
    uint32_t threshold = 2;
    uint32_t mode = 0;
    float    gain = 1.0F;
    uint32_t dstWidth = 0;
    uint32_t dstHeight = 0;
    uint32_t dstStride = 0;
    int32_t  dstOffsetX = 0;
    int32_t  dstOffsetY = 0;
    uint32_t pad0 = 0;
};
static_assert(sizeof(MeasureUniforms) == 80, "must match MeasureParams exactly");

/// Words of `results`, as measureFinish writes them.
constexpr size_t kResultWords = 33;

float asFloat(uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/// A count as two floats, each exact.
void pushCount(std::vector<float>& into, uint32_t count) {
    into.push_back(static_cast<float>(count >> 24U));
    into.push_back(static_cast<float>(count & 0xFFFFFFU));
}

class Measure final : public aofx::Effect {
public:
    void describe(aofx::EffectDesc& into) override {
        into.identifier = "rt.sparrow.aofx.measure";
        into.label = "Measure";
        into.grouping = "Analysis";
        into.description =
            "How far Source is from Reference, measured on the GPU.\n\n"
            "Attaches each picture's mean and largest value per channel (inside "
            "Window), the relative squared error, the 99th percentile and the "
            "largest relative difference, and the same two of the 8-bit sRGB "
            "code-value difference with the pixels over Threshold -- the numbers "
            "`athenea compare` prints, from the same kernels. The picture out is "
            "where the two differ.\n\n"
            "With no Reference wired it measures Source alone and passes it "
            "through.";

        aofx::ClipDesc source;
        source.name = "Source";
        source.label = "Source";
        into.inputs.push_back(source);

        aofx::ClipDesc reference;
        reference.name = "Reference";
        reference.label = "Reference";
        reference.optional = true;
        into.inputs.push_back(reference);

        into.outputs.push_back(aofx::PlaneDesc{"Color", "Colour", {"R", "G", "B", "A"}});

        aofx::ParamDesc mode;
        mode.name = "mode";
        mode.label = "Show";
        mode.type = aofx::ParamType::Choice;
        mode.choices = {{"source", "Source"},
                        {"difference", "Difference"},
                        {"relative", "Relative"},
                        {"codes", "Code values"}};
        mode.defaults = {3.0};
        mode.hint =
            "What the output picture is. Source passes it through. Difference "
            "is |Source - Reference| times Gain. Relative is a ramp of the "
            "largest channel's |S - R| / max(|R|, 0.001), red at 1 / Gain. "
            "Code values colours the pixels whose 8-bit difference is over "
            "Threshold, red at 32 / Gain, and leaves the rest black.";
        into.params.push_back(mode);

        aofx::ParamDesc gain;
        gain.name = "gain";
        gain.label = "Gain";
        gain.type = aofx::ParamType::Double;
        gain.defaults = {1.0};
        gain.displayMin = {0.0};
        gain.displayMax = {16.0};
        gain.hardMin = {0.0};
        gain.hint = "How much the heatmap is multiplied by. The numbers do not depend on it.";
        into.params.push_back(gain);

        aofx::ParamDesc threshold;
        threshold.name = "threshold";
        threshold.label = "Threshold";
        threshold.type = aofx::ParamType::Integer;
        threshold.defaults = {2.0};
        threshold.displayMin = {0.0};
        threshold.displayMax = {255.0};
        threshold.hardMin = {0.0};
        threshold.hardMax = {255.0};
        threshold.hint =
            "8-bit code values a pixel may differ by and not be counted as over. "
            "Two is athenea's: what dithering and rounding leave between two "
            "renders that agree.";
        into.params.push_back(threshold);

        aofx::ParamDesc window;
        window.name = "window";
        window.label = "Window";
        window.type = aofx::ParamType::Double;
        window.dimension = 4;
        window.defaults = {0.0, 0.0, 0.0, 0.0};
        window.hardMin = {0.0, 0.0, 0.0, 0.0};
        window.hint =
            "X0 Y0 X1 Y1 in Source's pixels, rows from the bottom: the means and "
            "the largest values are taken over [X0, X1) x [Y0, Y1). An X1 or Y1 "
            "of 0 is the picture's edge. The differences are always over the "
            "whole picture.";
        into.params.push_back(window);
    }

    [[nodiscard]] std::vector<aofx::KernelDesc> kernels() const override {
        return {aofx::KernelDesc{"measureRows", "measureRows", k_measure, k_measureBytes},
                aofx::KernelDesc{"measureReduce", "measureReduce", k_measure, k_measureBytes},
                aofx::KernelDesc{"measureFinish", "measureFinish", k_measure, k_measureBytes},
                aofx::KernelDesc{"measureHeatmap", "measureHeatmap", k_measure, k_measureBytes}};
    }

    bool process(const aofx::RenderRequest& request) override {
        const aofx::InputPlane*  source = request.input("Source");
        const aofx::OutputPlane* target = request.output("Color");
        if (source == nullptr || target == nullptr || !source->buffer.isValid() ||
            !target->buffer.isValid() || request.gpu == nullptr) {
            request.complaint = "measure needs its Source input and somewhere to write";
            return false;
        }
        const aofx::InputPlane* reference = request.input("Reference");
        const bool hasReference = reference != nullptr && reference->buffer.isValid();
        if (hasReference && (reference->buffer.width != source->buffer.width ||
                             reference->buffer.height != source->buffer.height)) {
            request.complaint = "measure compares pictures of one size: Source is " +
                                std::to_string(source->buffer.width) + "x" +
                                std::to_string(source->buffer.height) + ", Reference " +
                                std::to_string(reference->buffer.width) + "x" +
                                std::to_string(reference->buffer.height);
            return false;
        }

        MeasureUniforms uniforms;
        uniforms.width = static_cast<uint32_t>(source->buffer.width);
        uniforms.height = static_cast<uint32_t>(source->buffer.height);
        uniforms.srcStride = static_cast<uint32_t>(source->buffer.stride);
        uniforms.refStride = hasReference ? static_cast<uint32_t>(reference->buffer.stride) : 0U;
        uniforms.hasReference = hasReference ? 1U : 0U;

        // The window, as imageStats takes it: an X1 or Y1 of 0 is the edge,
        // and anything past the edge is the edge.
        const auto corner = [&](size_t k) {
            return static_cast<uint32_t>(std::clamp(std::floor(request.number("window", 0.0, k)), 0.0, 1.0e9));
        };
        uniforms.x0 = corner(0);
        uniforms.y0 = corner(1);
        uniforms.x1 = corner(2) == 0 ? uniforms.width : std::min(corner(2), uniforms.width);
        uniforms.y1 = corner(3) == 0 ? uniforms.height : std::min(corner(3), uniforms.height);
        if (uniforms.x0 >= uniforms.x1 || uniforms.y0 >= uniforms.y1) {
            request.complaint = "measure: an empty window";
            return false;
        }
        uniforms.threshold =
            static_cast<uint32_t>(std::clamp(std::lround(request.number("threshold", 2.0)), 0L, 255L));
        uniforms.mode = static_cast<uint32_t>(std::clamp(std::lround(request.number("mode", 3.0)), 0L, 3L));
        uniforms.gain = static_cast<float>(std::max(request.number("gain", 1.0), 0.0));
        uniforms.dstWidth = static_cast<uint32_t>(target->buffer.width);
        uniforms.dstHeight = static_cast<uint32_t>(target->buffer.height);
        uniforms.dstStride = static_cast<uint32_t>(target->buffer.stride);
        uniforms.dstOffsetX = target->buffer.rect.x1 - source->buffer.rect.x1;
        uniforms.dstOffsetY = target->buffer.rect.y1 - source->buffer.rect.y1;

        // Per row: two sums and two maxes a picture and the row's error, in
        // float4s; and 512 bins, four to a float4.
        const aofx::Buffer rows = request.gpu->scratch(5, source->buffer.height);
        const aofx::Buffer histogram = request.gpu->scratch(128, source->buffer.height);
        // 512 bins, then four float4s of sums and maxes and the error.
        const aofx::Buffer totals = request.gpu->scratch(133, 1);
        const aofx::Buffer results = request.gpu->scratch(9, 1);
        if (!rows.isValid() || !histogram.isValid() || !totals.isValid() || !results.isValid()) {
            request.complaint = "measure could not take the working space it counts in";
            return false;
        }
        uniforms.rowsStride = static_cast<uint32_t>(rows.stride);
        uniforms.histStride = static_cast<uint32_t>(histogram.stride) * 4U;

        const aofx::KernelId rowsKernel = request.gpu->load("measureRows");
        const aofx::KernelId reduce = request.gpu->load("measureReduce");
        const aofx::KernelId finish = request.gpu->load("measureFinish");
        const aofx::KernelId heatmap = request.gpu->load("measureHeatmap");
        if (rowsKernel == aofx::kInvalidKernel || reduce == aofx::kInvalidKernel ||
            finish == aofx::kInvalidKernel || heatmap == aofx::kInvalidKernel) {
            return false;
        }
        // Every buffer the kernels declare is bound; `hasReference` is what
        // stops the missing one being read.
        const std::vector<aofx::Buffer> buffers{source->buffer,
                                                hasReference ? reference->buffer : source->buffer,
                                                rows,
                                                histogram,
                                                totals,
                                                results,
                                                target->buffer};
        if (!request.gpu->run(rowsKernel, aofx::Grid{uniforms.height, 1, 1}, buffers, &uniforms,
                              sizeof(uniforms)) ||
            !request.gpu->run(reduce, aofx::Grid{513, 1, 1}, buffers, &uniforms, sizeof(uniforms)) ||
            !request.gpu->run(finish, aofx::Grid{1, 1, 1}, buffers, &uniforms, sizeof(uniforms)) ||
            !request.gpu->run(heatmap, aofx::Grid{uniforms.dstWidth, uniforms.dstHeight, 1}, buffers,
                              &uniforms, sizeof(uniforms))) {
            return false;
        }

        uint32_t words[36] = {};
        if (!request.gpu->read(results, words, sizeof(words))) {
            request.complaint = "measure could not read back what it measured";
            return false;
        }
        const auto image = [&](size_t first) {
            std::vector<float> values;
            for (size_t k = 0; k < 12; ++k) {
                values.push_back(asFloat(words[first + k]));
            }
            pushCount(values, words[32]);
            return values;
        };
        request.attach("source", image(0));
        if (!hasReference) {
            return true;
        }
        request.attach("reference", image(12));
        std::vector<float> hdr{asFloat(words[24]), asFloat(words[26]), asFloat(words[27]), asFloat(words[25])};
        pushCount(hdr, words[31]);
        request.attach("hdr", std::move(hdr));
        std::vector<float> codes{static_cast<float>(words[28]), static_cast<float>(words[29])};
        pushCount(codes, words[30]);
        codes.push_back(static_cast<float>(uniforms.threshold));
        pushCount(codes, words[31]);
        request.attach("codes", std::move(codes));
        static_assert(kResultWords <= 36, "results is nine float4s");
        return true;
    }
};

}   // namespace

AOFX_EXPORT_EFFECTS(Measure)
