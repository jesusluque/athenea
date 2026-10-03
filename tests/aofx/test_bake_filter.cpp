// Copyright (c) 2026 jesus luque.
//
// The splat bake filter (plugins/splatbakefilter) on a bake whose answer is
// known: a step of indirect light across a plane of gaussians, with noise
// added. The filter must take most of the noise away and leave the step where
// it was. Everything the effect is fed is written by a kernel and everything
// it answers is checked by one (`shaders/athenea/test/bake_filter_check.slang`);
// what crosses to the processor is five numbers.
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

image::ImagePtr picture(int32_t width, int32_t height) {
    auto made = image::Image::create({0, 0, width, height});
    REQUIRE(made);
    return *made;
}

}   // namespace

// A STEP OF LIGHT, NOISY, FILTERED. 64 x 64 gaussians a unit apart; the light
// is 0.2 left of the middle and 1.0 from it on, with a uniform noise of
// half-width 0.15 (a standard deviation of 0.087) and each gaussian's variance
// saying so. Away from the edge the error must fall to under a quarter; at the
// edge each column must keep its own side's light within 0.05 -- a step of
// 0.8 is nine standard deviations, which the luminance weight must see as an
// edge rather than as noise.
TEST_CASE("the splat bake filter takes a bake's noise and leaves its edges", "[aofx][bakefilter]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    registry.addSearchPath(ATHENEA_AOFX_BUNDLES);
    registry.scan(gpu);
    aofx::Effect* effect = registry.find("rt.sparrow.aofx.splatbakefilter");
    if (effect == nullptr) {
        SKIP("the SplatBakeFilter bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());
    REQUIRE(gpu->run([&] {
        auto synth = gpu::ComputeKernel::create(library, "athenea/test/bake_filter_check", "bfSynth");
        auto check = gpu::ComputeKernel::create(library, "athenea/test/bake_filter_check", "bfCheck");
        REQUIRE(synth);
        REQUIRE(check);
        constexpr uint32_t kAcross = 64;
        constexpr uint32_t kRows = 64;
        constexpr uint32_t kCount = kAcross * kRows;
        constexpr uint32_t kWidth = 1024;
        const image::ImagePtr points = picture(kWidth, kCount * 3 / kWidth);
        const image::ImagePtr light = picture(kWidth, kCount / kWidth);
        const gpu::Buffer pointsView = pictureView(*gpu, points);
        const gpu::Buffer lightView = pictureView(*gpu, light);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "test.results";
        auto results = gpu::Buffer::create(library.device(), desc);
        REQUIRE(results);
        const auto bind = [&](rhi::ShaderCursor cursor, const gpu::Buffer& filtered, const image::Image& out) {
            cursor["points"].setBinding(pointsView.rhi());
            cursor["light"].setBinding(lightView.rhi());
            cursor["filtered"].setBinding(filtered.rhi());
            cursor["results"].setBinding(results->rhi());
            cursor["synth"]["across"].setData(kAcross);
            cursor["synth"]["rows"].setData(kRows);
            cursor["synth"]["pointsWidth"].setData(static_cast<uint32_t>(points->bounds().width()));
            cursor["synth"]["pointsStride"].setData(static_cast<uint32_t>(points->stride()));
            cursor["synth"]["lightWidth"].setData(static_cast<uint32_t>(out.bounds().width()));
            cursor["synth"]["lightStride"].setData(static_cast<uint32_t>(out.stride()));
            cursor["synth"]["margin"].setData(uint32_t{8});
            cursor["synth"]["low"].setData(0.2F);
            cursor["synth"]["high"].setData(1.0F);
            cursor["synth"]["noise"].setData(0.15F);
        };
        {
            gpu::CommandBatch batch(library.device());
            synth->dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor c) { bind(c, lightView, *light); });
            REQUIRE(batch.submit(true));
            points->deviceWrote();
            light->deviceWrote();
        }
        aofx_host::EffectJob job;
        job.bounds = light->bounds();
        job.inputs.push_back({"Points", points});
        job.inputs.push_back({"Indirect", light});
        job.params.push_back(aofx::ParamValue{"count", {static_cast<double>(kCount)}, {}});
        job.params.push_back(aofx::ParamValue{"coefficients", {1.0}, {}});
        job.params.push_back(aofx::ParamValue{"iterations", {3.0}, {}});
        auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE(rendered);
        const std::vector<float>* said = (*rendered)->attached("filtered");
        REQUIRE(said != nullptr);
        REQUIRE(said->size() >= 3);
        CHECK((*said)[1] == 0.0F);   // no gaussian left out of a full cell
        const gpu::Buffer filtered = pictureView(*gpu, *rendered);
        {
            gpu::CommandBatch batch(library.device());
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) { bind(c, filtered, **rendered); });
            REQUIRE(batch.submit(true));
        }
        auto measured = results->readAll<float>(library.device());
        REQUIRE(measured);
        const std::vector<float>& m = *measured;
        std::printf("  bake filter: cell %.3g, error %.3g -> %.3g over %.0f gaussians; edge columns %.4f | %.4f\n",
                    static_cast<double>((*said)[2]), static_cast<double>(m[0]), static_cast<double>(m[1]),
                    static_cast<double>(m[4]), static_cast<double>(m[2]), static_cast<double>(m[3]));
        CHECK(m[4] > 1000.0F);
        CHECK(m[1] * 4.0F < m[0]);
        CHECK(std::abs(m[2] - 0.2F) < 0.05F);
        CHECK(std::abs(m[3] - 1.0F) < 0.05F);
    }));
}
