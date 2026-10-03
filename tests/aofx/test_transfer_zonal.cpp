// Copyright (c) 2026 jesus luque.
//
// The splat transfer zonal fit (plugins/splattransferzonal) and the frame's
// reading of what it wrote (`splatTransferFrame`, splat/splat_relight.slang):
// a transfer that is two lobes comes back as itself; one shaped as a bake's
// is approximated, and how well is stated; and whatever turns the gaussian --
// its own frame, as a skeleton's pose does, or its instance's rows -- turns
// the transfer and the sun's shadow bits with it. Everything the effect is fed
// is written by a kernel and everything it answers is checked by one
// (`shaders/athenea/test/transfer_zonal_check.slang`); what crosses to the
// processor is counters.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "aofx/Effect.h"
#include "athenea/aofx/EffectRegistry.h"
#include "athenea/aofx/EffectRender.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu_host/Context.h"
#include "athenea/gpu_host/ImageStorage.h"
#include "athenea/image/Image.h"

using namespace athenea;

namespace {

gpu::Buffer pictureView(gpu_host::Context& context, const image::ImagePtr& picture) {
    gpu_host::ImageStorage* storage = context.sharedStorage();
    REQUIRE(storage != nullptr);
    const uint64_t buffer = storage->bufferFor(picture->address());
    REQUIRE(buffer != 0);
    auto view = context.renderView(buffer, picture->sizeBytes(), 16, "test.image");
    REQUIRE(view);
    return std::move(*view);
}

/// The upper edge of the quarter octave the fraction `p` of the fitted
/// gaussians falls below, out of the effect's attached histogram.
double percentileOf(const std::vector<float>& said, double p) {
    const double fitted = said[0];
    double seen = 0.0;
    for (size_t b = 0; b + 1 < said.size(); ++b) {
        seen += said[b + 1];
        if (fitted > 0.0 && seen >= p * fitted) {
            return std::exp2((static_cast<double>(b) - 40.0 + 1.0) / 4.0);
        }
    }
    return std::exp2(2.0);
}

}   // namespace

TEST_CASE("a zonal transfer is fitted in the gaussian's frame and turns with it", "[aofx][transfer][zonal]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    registry.addSearchPath(ATHENEA_AOFX_BUNDLES);
    registry.scan(gpu);
    aofx::Effect* effect = registry.find("rt.sparrow.aofx.splattransferzonal");
    if (effect == nullptr) {
        SKIP("the SplatTransferZonal bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());
    REQUIRE(gpu->run([&] {
        const auto kernel = [&](const char* entry) {
            auto made = gpu::ComputeKernel::create(library, "athenea/test/transfer_zonal_check", entry);
            REQUIRE(made);
            return std::move(*made);
        };
        gpu::ComputeKernel synth = kernel("tzcSynth");
        gpu::ComputeKernel pack = kernel("tzcPack");
        gpu::ComputeKernel check = kernel("tzcCheck");
        constexpr uint32_t kCount = 4096;
        constexpr int32_t  kWidth = 1024;
        auto made = image::Image::create({0, 0, kWidth, static_cast<int32_t>(kCount * 4 / kWidth)});
        REQUIRE(made);
        const image::ImagePtr in = *made;
        const gpu::Buffer inView = pictureView(*gpu, in);
        const auto buffer = [&](uint64_t words, const char* label) {
            const std::vector<uint32_t> zeros(words, 0u);
            auto b = gpu::Buffer::fromSpan<uint32_t>(library.device(), zeros, label);
            REQUIRE(b);
            return std::move(*b);
        };
        gpu::Buffer packed = buffer(uint64_t{kCount} * 5, "test.zonal.packed");
        gpu::Buffer truth = buffer(uint64_t{kCount} * 12, "test.zonal.truth");

        for (uint32_t lobes : {2u, 1u}) {
            gpu::Buffer results = buffer(16, "test.zonal.results");
            const auto bind = [&](rhi::ShaderCursor cursor, const gpu::Buffer& fitted, const image::Image& out) {
                cursor["picture"].setBinding(inView.rhi());
                cursor["fitted"].setBinding(fitted.rhi());
                cursor["packed"].setBinding(packed.rhi());
                cursor["halves"].setBinding(packed.rhi());
                cursor["truth"].setBinding(truth.rhi());
                cursor["results"].setBinding(results.rhi());
                cursor["synth"]["count"].setData(kCount);
                cursor["synth"]["inWidth"].setData(static_cast<uint32_t>(in->bounds().width()));
                cursor["synth"]["inStride"].setData(static_cast<uint32_t>(in->stride()));
                cursor["synth"]["outWidth"].setData(static_cast<uint32_t>(out.bounds().width()));
                cursor["synth"]["outStride"].setData(static_cast<uint32_t>(out.stride()));
                cursor["synth"]["lobes"].setData(lobes);
            };
            {
                gpu::CommandBatch batch(library.device());
                synth.dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor c) { bind(c, inView, *in); });
                REQUIRE(batch.submit(true));
                in->deviceWrote();
            }
            aofx_host::EffectJob job;
            job.bounds = {0, 0, kWidth, static_cast<int32_t>(kCount * 3 / kWidth)};
            job.inputs.push_back({"Transfer", in});
            job.params.push_back(aofx::ParamValue{"count", {static_cast<double>(kCount)}, {}});
            job.params.push_back(aofx::ParamValue{"lobes", {static_cast<double>(lobes)}, {}});
            job.params.push_back(aofx::ParamValue{"rebin", {1.0}, {}});
            auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
            REQUIRE(rendered);
            const std::vector<float>* said = (*rendered)->attached("fitted");
            REQUIRE(said != nullptr);
            REQUIRE(said->size() == 49);
            CHECK((*said)[0] == static_cast<float>(kCount));
            const gpu::Buffer fitted = pictureView(*gpu, *rendered);
            {
                gpu::CommandBatch batch(library.device());
                pack.dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor c) { bind(c, fitted, **rendered); });
                check.dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor c) { bind(c, fitted, **rendered); });
                REQUIRE(batch.submit(true));
            }
            auto counted = results.readAll<uint32_t>(library.device());
            REQUIRE(counted);
            const std::vector<uint32_t>& r = *counted;
            const double bakeShaped = r[2] > 0 ? static_cast<double>(r[1]) * 1.0e-5 / r[2] : 0.0;
            std::printf("  zonal, %u lobe%s: exact lobes worst %.2e (%u off by > 0.5%%); a bake's shape against its "
                        "nine harmonics: mean %.3f, %u of %u under 10%%; all: median %.3g, p90 %.3g, p99 %.3g; "
                        "turned by the frame %u, by the rows %u, sun %u, bits %u off\n",
                        lobes, lobes == 1 ? "" : "s", static_cast<double>(r[7]) * 1.0e-6, r[0], bakeShaped, r[8],
                        r[2], percentileOf(*said, 0.5), percentileOf(*said, 0.9), percentileOf(*said, 0.99), r[3],
                        r[4], r[5], r[6]);
            // Whatever turns the gaussian turns what it reads, exactly: the
            // frame (a skeleton's pose), the instance's rows (a prim's
            // transform), and the sun's share through the bits.
            CHECK(r[3] == 0);
            CHECK(r[4] == 0);
            CHECK(r[5] == 0);
            // A frame that is the world's leaves the bits as they were.
            CHECK(r[6] == 0);
            if (lobes == 2) {
                // Two lobes hold two lobes, to the halves' rounding.
                CHECK(r[0] == 0);
                // A bake's shape -- the cosine with a cap of sky taken away
                // -- is held to within fifteen per cent on average.
                CHECK(bakeShaped < 0.15);
            }
        }
    }));
}
