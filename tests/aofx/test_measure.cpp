// Copyright (c) 2026 jesus luque.
//
// The Measure effect against the comparison it was ported from: two pictures
// written by a kernel (`shaders/athenea/test/measure_inputs.slang`), measured
// once through the AOFX host and once by render::imageStats, compareHdr and
// compareImages over the same memory, and every number the effect attaches
// held to theirs. What crosses to the processor is those numbers and nothing
// else.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "aofx/Effect.h"
#include "athenea/aofx/EffectRegistry.h"
#include "athenea/aofx/EffectRender.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu_host/Context.h"
#include "athenea/gpu_host/ImageStorage.h"
#include "athenea/image/Image.h"
#include "athenea/render/ReferenceRenderer.h"

using namespace athenea;

namespace {

// A width whose rows need no padding, so render's functions (which index
// y * width + x) and the effect (y * stride + x) read the same pixels; and a
// height no row group divides.
constexpr int32_t kWidth = 128;
constexpr int32_t kHeight = 77;

gpu::Buffer viewOf(gpu_host::Context& context, const image::ImagePtr& picture) {
    gpu_host::ImageStorage* storage = context.sharedStorage();
    REQUIRE(storage != nullptr);
    const uint64_t buffer = storage->bufferFor(picture->address());
    REQUIRE(buffer != 0);
    auto view = context.renderView(buffer, picture->sizeBytes(), 16, "test.image");
    REQUIRE(view);
    return std::move(*view);
}

image::ImagePtr pictureOf(int32_t width, int32_t height) {
    auto made = image::Image::create({0, 0, width, height});
    REQUIRE(made);
    return *made;
}

uint64_t countAt(const std::vector<float>& values, size_t at) {
    return (static_cast<uint64_t>(values[at]) << 24U) + static_cast<uint64_t>(values[at + 1]);
}

bool near(double a, double b) { return std::abs(a - b) <= 1.0e-6 * std::max(std::abs(b), 1.0e-30); }

/// "source" / "reference" against imageStats: the sum over the pixels is the
/// mean render divides out, to the bit; the float mean is that rounded once.
void checkStats(const std::vector<float>& attached, const render::ImageStats& expected) {
    REQUIRE(attached.size() == 14);
    CHECK(countAt(attached, 12) == expected.pixels);
    for (size_t c = 0; c < 4; ++c) {
        CHECK(static_cast<double>(attached[8 + c]) / static_cast<double>(expected.pixels) == expected.mean[c]);
        CHECK(near(attached[c], expected.mean[c]));
        CHECK(attached[4 + c] == expected.max[c]);
    }
}

struct Measured {
    image::ImagePtr source;
    image::ImagePtr reference;
};

Measured inputs(gpu::ShaderLibrary& library, uint32_t seed) {
    auto make = gpu::ComputeKernel::create(library, "athenea/test/measure_inputs", "measureInputs");
    REQUIRE(make);
    Measured m{pictureOf(kWidth, kHeight), pictureOf(kWidth, kHeight)};
    REQUIRE(m.source->stride() == kWidth);
    gpu_host::Context& gpu = *gpu_host::installProcessContext();
    const gpu::Buffer source = viewOf(gpu, m.source);
    const gpu::Buffer reference = viewOf(gpu, m.reference);
    gpu::CommandBatch batch(library.device());
    make->dispatch(batch, {kWidth, kHeight, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["source"].setBinding(source.rhi());
        cursor["reference"].setBinding(reference.rhi());
        cursor["params"]["width"].setData(uint32_t{kWidth});
        cursor["params"]["height"].setData(uint32_t{kHeight});
        cursor["params"]["stride"].setData(uint32_t{kWidth});
        cursor["params"]["seed"].setData(seed);
    });
    REQUIRE(batch.submit(true));
    m.source->deviceWrote();
    m.reference->deviceWrote();
    return m;
}

aofx::Effect* measure(aofx_host::EffectRegistry& registry, gpu_host::Context* gpu) {
    registry.addSearchPath(ATHENEA_AOFX_BUNDLES);
    registry.scan(gpu);
    return registry.find("rt.sparrow.aofx.measure");
}

}   // namespace

TEST_CASE("Measure attaches what render's comparison measures", "[aofx][measure]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = measure(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Measure bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    // The whole picture, and a window that starts off the origin and runs
    // past the right edge (clamped to it), as `athenea compare --window` gives.
    const std::vector<std::vector<uint32_t>> windows{{0, 0, 0, 0}, {9, 4, 1000, 60}};
    uint32_t seed = 1;
    for (const std::vector<uint32_t>& w : windows) {
        REQUIRE(gpu->run([&] {
            const Measured m = inputs(library, seed++);
            const gpu::Buffer source = viewOf(*gpu, m.source);
            const gpu::Buffer reference = viewOf(*gpu, m.reference);

            aofx_host::EffectJob job;
            job.inputs.push_back({"Source", m.source});
            job.inputs.push_back({"Reference", m.reference});
            job.params.push_back(aofx::ParamValue{
                "window", {double(w[0]), double(w[1]), double(w[2]), double(w[3])}, {}});
            auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
            REQUIRE(rendered);
            const image::Image& out = **rendered;

            auto sourceStats = render::imageStats(library, source, kWidth, kHeight, w[0], w[1], w[2], w[3]);
            auto referenceStats = render::imageStats(library, reference, kWidth, kHeight, w[0], w[1], w[2], w[3]);
            auto hdr = render::compareHdr(library, source, reference, kWidth, kHeight);
            auto codes = render::compareImages(library, source, reference, kWidth, kHeight);
            REQUIRE(sourceStats);
            REQUIRE(referenceStats);
            REQUIRE(hdr);
            REQUIRE(codes);

            const std::vector<float>* a = out.attached("source");
            const std::vector<float>* b = out.attached("reference");
            const std::vector<float>* h = out.attached("hdr");
            const std::vector<float>* c = out.attached("codes");
            REQUIRE(a != nullptr);
            REQUIRE(b != nullptr);
            REQUIRE(h != nullptr);
            REQUIRE(c != nullptr);
            checkStats(*a, *sourceStats);
            checkStats(*b, *referenceStats);

            // relMSE, its sum to the bit; the percentiles as the floats of
            // the same bin bounds.
            REQUIRE(h->size() == 6);
            CHECK(countAt(*h, 4) == hdr->pixels);
            CHECK(static_cast<double>((*h)[3]) / static_cast<double>(hdr->pixels) == hdr->relMse);
            CHECK(near((*h)[0], hdr->relMse));
            CHECK(near((*h)[1], hdr->p99Relative));
            CHECK(near((*h)[2], hdr->maxRelative));
            // A picture made to fill the bins, not one that agrees.
            CHECK(hdr->p99Relative > 0.0);
            CHECK(hdr->maxRelative > hdr->p99Relative);

            REQUIRE(c->size() == 7);
            CHECK(static_cast<uint32_t>((*c)[0]) == codes->p99);
            CHECK(static_cast<uint32_t>((*c)[1]) == codes->max);
            CHECK(countAt(*c, 2) == codes->over2);
            CHECK((*c)[4] == 2.0F);
            CHECK(countAt(*c, 5) == codes->pixels);
            CHECK(codes->over2 > 0);
            CHECK(codes->over2 < codes->pixels);
        }));
    }
}

TEST_CASE("Measure without a reference measures the source and passes it through", "[aofx][measure]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = measure(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Measure bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        const Measured m = inputs(library, 7);
        const gpu::Buffer source = viewOf(*gpu, m.source);
        aofx_host::EffectJob job;
        job.inputs.push_back({"Source", m.source});
        auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE(rendered);
        const image::Image& out = **rendered;
        CHECK(out.attached("reference") == nullptr);
        CHECK(out.attached("hdr") == nullptr);
        CHECK(out.attached("codes") == nullptr);
        const std::vector<float>* a = out.attached("source");
        REQUIRE(a != nullptr);
        auto stats = render::imageStats(library, source, kWidth, kHeight);
        REQUIRE(stats);
        checkStats(*a, *stats);

        // The picture out is the picture in: compared with it, nothing over
        // a code value of difference anywhere.
        REQUIRE(out.stride() == kWidth);
        const gpu::Buffer passed = viewOf(*gpu, *rendered);
        auto same = render::compareImages(library, passed, source, kWidth, kHeight);
        REQUIRE(same);
        CHECK(same->max == 0);
        CHECK(same->pixels == uint64_t{kWidth} * kHeight);
    }));
}

TEST_CASE("Measure refuses pictures of two sizes and an empty window", "[aofx][measure]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = measure(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Measure bundle is not built here");
    }
    REQUIRE(gpu->run([&] {
        aofx_host::EffectJob job;
        job.inputs.push_back({"Source", pictureOf(32, 16)});
        job.inputs.push_back({"Reference", pictureOf(16, 16)});
        auto refused = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE_FALSE(refused);
        CHECK(refused.error().toString().find("one size") != std::string::npos);

        aofx_host::EffectJob empty;
        empty.inputs.push_back({"Source", pictureOf(32, 16)});
        empty.params.push_back(aofx::ParamValue{"window", {4.0, 4.0, 4.0, 8.0}, {}});
        auto none = aofx_host::renderEffect(*gpu, *effect, empty);
        REQUIRE_FALSE(none);
        CHECK(none.error().toString().find("empty window") != std::string::npos);
    }));
}
