// Copyright (c) 2026 jesus luque.
//
// Electronic Arts' conversion, as this repository runs it: a quad of known
// size in, and every gaussian that comes back where it should be, as wide as
// one cell of the projection grid, flat against the quad, and the colour of
// the texel it stands on.
//
// Everything the effect is fed is written by a kernel and everything it
// answers is checked by one (`shaders/athenea/test/mesh2splat_check.slang`); what
// crosses to the processor is five counters, which must all be zero.
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
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

constexpr uint32_t kResolution = 16;
constexpr float    kSigma = 0.65F;
constexpr float    kFlatness = 0.1F;

/// A slang-rhi view of an image's own pixels, as the conversion's own host
/// takes one: the effect and the test write the same memory.
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

aofx::Effect* conversion(aofx_host::EffectRegistry& registry, gpu_host::Context* gpu) {
    registry.addSearchPath(ATHENEA_AOFX_BUNDLES);
    registry.scan(gpu);
    return registry.find("rt.sparrow.aofx.mesh2splat");
}

void number(aofx_host::EffectJob& job, const char* name, double value) {
    job.params.push_back(aofx::ParamValue{name, {value}, {}});
}

}   // namespace

TEST_CASE("a quad becomes a gaussian a cell, flat, one cell wide", "[aofx][mesh2splat]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuad");
        auto check = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sCheckQuad");
        REQUIRE(quad);
        REQUIRE(check);

        // The triangles: two of them, six entries each, in a picture 64
        // entries wide.
        constexpr uint32_t kMeshWidth = 64;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const auto meshStride = static_cast<uint32_t>(mesh->stride());

        // Room for every cell of the grid and then some: the quad covers
        // `resolution` squared of them, and the cells the diagonal passes
        // through belong to both triangles.
        const uint32_t budget = kResolution * kResolution * 2;
        const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 4), 1);

        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(meshStride);
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();

        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});
        aofx_host::EffectJob job;
        job.bounds = records->bounds();
        job.inputs.push_back({"Mesh", mesh});
        number(job, "triangles", 2);
        number(job, "resolution", kResolution);
        number(job, "maxSplats", budget);
        number(job, "flatness", kFlatness);
        number(job, "opacity", 1.0);
        number(job, "writePbr", 0.0);
        job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});
        job.params.push_back(aofx::ParamValue{"materialColour", {0.25, 0.5, 0.75}, {}});

        auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE(rendered);
        const std::vector<float>* counted = (*rendered)->attached("splats");
        REQUIRE(counted != nullptr);
        REQUIRE(counted->size() >= 4);
        const auto written = static_cast<uint32_t>((*counted)[0]);
        // Every cell whose centre the quad covers, and no more: the grid is
        // `resolution` cells across the model's longest side, the quad is
        // that side, so that is `resolution` squared -- plus, at most, the
        // cells the diagonal runs through, which both triangles claim.
        CHECK(written >= kResolution * kResolution);
        CHECK(written <= kResolution * kResolution + kResolution);
        CHECK((*counted)[2] == 0.0F);   // no triangle without a frame
        CHECK((*counted)[3] == 0.0F);   // no cell left unwalked

        const gpu::Buffer recordView = viewOf(*gpu, *rendered);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "test.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);

        gpu::CommandBatch second(library.device());
        check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(recordView.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["splats"].setData(written);
            cursor["params"]["recordPixels"].setData(uint32_t{4});
            cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
            cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
            cursor["params"]["resolution"].setData(kResolution);
            cursor["params"]["sigma"].setData(kSigma);
            cursor["params"]["flatness"].setData(kFlatness);
            cursor["params"]["tolerance"].setData(1.0e-3F);
            cursor["params"]["opacity"].setData(1.0F);
            const std::array<float, 4> colour{0.25F, 0.5F, 0.75F, 1.0F};
            const std::array<float, 4> axis{0.0F, 0.0F, 1.0F, 0.0F};
            cursor["params"]["colour"].setData(colour.data(), 16);
            cursor["params"]["axis"].setData(axis.data(), 16);
        });
        REQUIRE(second.submit(true));
        auto violations = counts->readAll<uint32_t>(library.device());
        REQUIRE(violations);
        CHECK((*violations)[0] == 0);   // every gaussian on the quad
        CHECK((*violations)[1] == 0);   // every one a cell wide, and flat
        CHECK((*violations)[2] == 0);   // every frame square to the quad
        CHECK((*violations)[3] == 0);   // every colour the material's
        CHECK((*violations)[4] == 0);   // every opacity the parameter's
    }));
}

// THE CELL, BOUNDED IN THE WORLD.
//
// The density is `resolution` cells across the box the host attaches, and
// that box used to be the whole model's: a 16-unit floor in a car's stage
// made the car's cell 8 times coarser, and a badge got the same cell as the
// hood. So the host may attach each mesh's own box, and the cell is then
// held between `cellMin` and `cellMax` in world units. Here: a quad whose
// cell is pulled coarser by the floor, a small quad measured over its own box
// and held either way, and the same small quad measured over the model's
// box -- the dilution, counted.
TEST_CASE("a cell bounded in the world holds whichever box the density is measured over",
          "[aofx][mesh2splat][cell]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    struct Run {
        const char* name;
        float       quadSize;
        float       boxSide;      ///< the attached box: (0,0,0)..(boxSide, boxSide, 0)
        double      cellMin;
        double      cellMax;
        double      byLongest;
        float       cell;         ///< the cell every gaussian must be sigma of (0: 1 / resolution)
        uint32_t    least;        ///< written, at least
        uint32_t    most;         ///< and at most
    };
    // A quad of `resolution` cells across has `resolution` squared of them,
    // plus at most the cells the diagonal runs through, which both triangles
    // claim; two by two is four to six.
    const Run runs[] = {
        {"unit quad, cell floored at a half", 1.0F, 1.0F, 0.5, 0.0, 0.0, 0.5F, 4, 6},
        {"small quad over its own box, ceiling not reached", 0.1F, 0.1F, 0.0, 1.0 / 16.0, 1.0, 0.00625F, 256, 272},
        {"small quad over its own box, floored", 0.1F, 0.1F, 0.05, 0.0, 1.0, 0.05F, 4, 6},
        {"small quad over the model's box: diluted", 0.1F, 1.0F, 0.0, 0.0, 0.0, 0.0F, 1, 8},
    };
    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuad");
        auto check = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sCheckQuad");
        REQUIRE(quad);
        REQUIRE(check);
        for (const Run& run : runs) {
            constexpr uint32_t kMeshWidth = 64;
            const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
            const gpu::Buffer meshView = viewOf(*gpu, mesh);
            const auto meshStride = static_cast<uint32_t>(mesh->stride());
            const uint32_t budget = kResolution * kResolution * 2;
            const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 4), 1);

            gpu::CommandBatch batch(library.device());
            quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["mesh"].setBinding(meshView.rhi());
                cursor["albedo"].setBinding(meshView.rhi());
                cursor["records"].setBinding(meshView.rhi());
                cursor["counts"].setBinding(meshView.rhi());
                cursor["params"]["meshWidth"].setData(kMeshWidth);
                cursor["params"]["meshStride"].setData(meshStride);
                cursor["params"]["quadSize"].setData(run.quadSize);
            });
            REQUIRE(batch.submit(true));
            mesh->deviceWrote();

            mesh->attach("bounds", {0.0F, 0.0F, 0.0F, run.boxSide, run.boxSide, 0.0F});
            aofx_host::EffectJob job;
            job.bounds = records->bounds();
            job.inputs.push_back({"Mesh", mesh});
            number(job, "triangles", 2);
            number(job, "resolution", kResolution);
            number(job, "maxSplats", budget);
            number(job, "flatness", kFlatness);
            number(job, "opacity", 1.0);
            number(job, "writePbr", 0.0);
            number(job, "cellMin", run.cellMin);
            number(job, "cellMax", run.cellMax);
            number(job, "cellByLongest", run.byLongest);
            job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});
            job.params.push_back(aofx::ParamValue{"materialColour", {0.25, 0.5, 0.75}, {}});

            auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
            REQUIRE(rendered);
            const std::vector<float>* counted = (*rendered)->attached("splats");
            REQUIRE(counted != nullptr);
            REQUIRE(counted->size() >= 4);
            const auto written = static_cast<uint32_t>((*counted)[0]);
            std::printf("  %-50s %u gaussians\n", run.name, written);
            CHECK(written >= run.least);
            CHECK(written <= run.most);
            CHECK((*counted)[3] == 0.0F);   // no cell left unwalked

            const gpu::Buffer recordView = viewOf(*gpu, *rendered);
            gpu::BufferDesc desc;
            desc.bytes = 8 * 4;
            desc.elementBytes = 4;
            desc.label = "test.counts";
            auto counts = gpu::Buffer::create(library.device(), desc);
            REQUIRE(counts);
            gpu::CommandBatch second(library.device());
            check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["mesh"].setBinding(meshView.rhi());
                cursor["albedo"].setBinding(meshView.rhi());
                cursor["records"].setBinding(recordView.rhi());
                cursor["counts"].setBinding(counts->rhi());
                cursor["params"]["splats"].setData(written);
                cursor["params"]["recordPixels"].setData(uint32_t{4});
                cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
                cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
                cursor["params"]["resolution"].setData(kResolution);
                cursor["params"]["sigma"].setData(kSigma);
                cursor["params"]["flatness"].setData(kFlatness);
                cursor["params"]["tolerance"].setData(1.0e-3F);
                cursor["params"]["opacity"].setData(1.0F);
                cursor["params"]["quadSize"].setData(run.quadSize);
                cursor["params"]["cell"].setData(run.cell);
                const std::array<float, 4> colour{0.25F, 0.5F, 0.75F, 1.0F};
                const std::array<float, 4> axis{0.0F, 0.0F, 1.0F, 0.0F};
                cursor["params"]["colour"].setData(colour.data(), 16);
                cursor["params"]["axis"].setData(axis.data(), 16);
            });
            REQUIRE(second.submit(true));
            auto violations = counts->readAll<uint32_t>(library.device());
            REQUIRE(violations);
            CHECK((*violations)[0] == 0);   // every gaussian on the quad
            CHECK((*violations)[1] == 0);   // every one the bounded cell wide, and flat
        }
    }));
}

TEST_CASE("a gaussian takes the colour of the texel it stands on", "[aofx][mesh2splat]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuad");
        auto checker = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sChecker");
        auto check = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sCheckColour");
        REQUIRE(quad);
        REQUIRE(checker);
        REQUIRE(check);

        constexpr uint32_t kMeshWidth = 64;
        constexpr uint32_t kChecker = 8;
        constexpr int32_t  kMapSize = 64;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const image::ImagePtr albedo = pictureOf(kMapSize, kMapSize);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const gpu::Buffer albedoView = viewOf(*gpu, albedo);
        const uint32_t budget = kResolution * kResolution * 2;
        const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 6), 1);

        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(albedoView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        checker->dispatch(batch, {kMapSize, kMapSize, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(albedoView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["albedoWidth"].setData(static_cast<uint32_t>(kMapSize));
            cursor["params"]["albedoHeight"].setData(static_cast<uint32_t>(kMapSize));
            cursor["params"]["albedoStride"].setData(static_cast<uint32_t>(albedo->stride()));
            cursor["params"]["checker"].setData(kChecker);
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();
        albedo->deviceWrote();

        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});
        aofx_host::EffectJob job;
        job.bounds = records->bounds();
        job.inputs.push_back({"Mesh", mesh});
        job.inputs.push_back({"Albedo", albedo});
        number(job, "triangles", 2);
        number(job, "resolution", kResolution);
        number(job, "maxSplats", budget);
        number(job, "writePbr", 1.0);
        job.params.push_back(aofx::ParamValue{"materialColour", {1.0, 1.0, 1.0}, {}});

        auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE(rendered);
        const std::vector<float>* counted = (*rendered)->attached("splats");
        REQUIRE(counted != nullptr);
        const auto written = static_cast<uint32_t>((*counted)[0]);
        REQUIRE(written >= kResolution * kResolution);
        // Six entries a record, since the texture coordinate the check reads
        // travels in the sixth.
        REQUIRE(counted->size() >= 5);
        REQUIRE((*counted)[4] == 6.0F);

        const gpu::Buffer recordView = viewOf(*gpu, *rendered);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "test.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);
        gpu::CommandBatch second(library.device());
        check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(albedoView.rhi());
            cursor["records"].setBinding(recordView.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["splats"].setData(written);
            cursor["params"]["recordPixels"].setData(uint32_t{6});
            cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
            cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
            cursor["params"]["tolerance"].setData(1.0e-2F);
            cursor["params"]["checker"].setData(kChecker);
        });
        REQUIRE(second.submit(true));
        auto violations = counts->readAll<uint32_t>(library.device());
        REQUIRE(violations);
        CHECK((*violations)[3] == 0);
    }));
}

TEST_CASE("a cut-out map leaves no gaussian where there is no surface",
          "[aofx][mesh2splat]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuad");
        auto quadUv2 = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuadUv2");
        auto checker = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sChecker");
        auto check = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sCheckCut");
        REQUIRE(quad);
        REQUIRE(quadUv2);
        REQUIRE(checker);
        REQUIRE(check);

        constexpr uint32_t kMeshWidth = 64;
        constexpr uint32_t kChecker = 8;
        constexpr int32_t  kMapSize = 64;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const image::ImagePtr uv2 = pictureOf(kMeshWidth, 1);
        const image::ImagePtr mask = pictureOf(kMapSize, kMapSize);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const gpu::Buffer uv2View = viewOf(*gpu, uv2);
        const gpu::Buffer maskView = viewOf(*gpu, mask);
        const uint32_t budget = kResolution * kResolution * 2;
        const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 6), 1);

        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(maskView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        quadUv2->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(uv2View.rhi());
            cursor["albedo"].setBinding(maskView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(uv2->stride()));
            cursor["params"]["checker"].setData(kChecker);
        });
        // Red on an even cell, green on an odd one: read on the red channel it
        // is a mask that keeps exactly half the quad.
        checker->dispatch(batch, {kMapSize, kMapSize, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(maskView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["albedoWidth"].setData(static_cast<uint32_t>(kMapSize));
            cursor["params"]["albedoHeight"].setData(static_cast<uint32_t>(kMapSize));
            cursor["params"]["albedoStride"].setData(static_cast<uint32_t>(mask->stride()));
            cursor["params"]["checker"].setData(kChecker);
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();
        uv2->deviceWrote();
        mask->deviceWrote();
        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});

        const auto convert = [&](bool withCut, bool bySecond = false) {
            aofx_host::EffectJob job;
            job.bounds = records->bounds();
            job.inputs.push_back({"Mesh", mesh});
            if (withCut) {
                job.inputs.push_back({"Opacity", mask});
            }
            if (bySecond) {
                // The cut read by the second set of coordinates.
                job.inputs.push_back({"Texcoord2", uv2});
                number(job, "opacityUv2", 1.0);
            }
            number(job, "triangles", 2);
            number(job, "resolution", kResolution);
            number(job, "maxSplats", budget);
            number(job, "writePbr", 1.0);
            if (withCut) {
                number(job, "opacityChannel", 1.0);   // red
                number(job, "opacityCut", 0.5);
            }
            return aofx_host::renderEffect(*gpu, *effect, job);
        };

        auto whole = convert(false);
        REQUIRE(whole);
        const std::vector<float>* wholeCount = (*whole)->attached("splats");
        REQUIRE(wholeCount != nullptr);
        const auto full = static_cast<uint32_t>((*wholeCount)[0]);
        REQUIRE(full >= kResolution * kResolution);

        auto cut = convert(true);
        REQUIRE(cut);
        const std::vector<float>* cutCount = (*cut)->attached("splats");
        REQUIRE(cutCount != nullptr);
        const auto kept = static_cast<uint32_t>((*cutCount)[0]);

        // HALF THE QUAD, WITHIN THE CELLS THE EDGE CUTS THROUGH. The mask is
        // a checkerboard of `kChecker` squares, so half of it is gone; what
        // the count cannot be exact about is the ring of texels an edge runs
        // through, which bilinear filtering carries either way.
        const double ratio = static_cast<double>(kept) / static_cast<double>(full);
        CHECK(ratio > 0.45);
        CHECK(ratio < 0.55);

        const auto violationsOf = [&](const image::ImagePtr& out, uint32_t written, uint32_t parity) {
            const gpu::Buffer recordView = viewOf(*gpu, out);
            gpu::BufferDesc desc;
            desc.bytes = 8 * 4;
            desc.elementBytes = 4;
            desc.label = "test.counts";
            auto counts = gpu::Buffer::create(library.device(), desc);
            REQUIRE(counts);
            gpu::CommandBatch second(library.device());
            check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["mesh"].setBinding(meshView.rhi());
                cursor["albedo"].setBinding(maskView.rhi());
                cursor["records"].setBinding(recordView.rhi());
                cursor["counts"].setBinding(counts->rhi());
                cursor["params"]["splats"].setData(written);
                cursor["params"]["recordPixels"].setData(uint32_t{6});
                cursor["params"]["dstWidth"].setData(static_cast<uint32_t>(out->bounds().width()));
                cursor["params"]["dstStride"].setData(static_cast<uint32_t>(out->stride()));
                cursor["params"]["checker"].setData(kChecker);
                cursor["params"]["parity"].setData(parity);
            });
            REQUIRE(second.submit(true));
            auto violations = counts->readAll<uint32_t>(library.device());
            REQUIRE(violations);
            return (*violations)[4];
        };
        // Not one gaussian stands well inside a square the mask cut away.
        CHECK(violationsOf(*cut, kept, 0) == 0);

        // THE CUT READ BY A SECOND SET OF COORDINATES, one cell down: the
        // same half is kept, and it is the other half -- the records still
        // carry the first set, so the check reads them with the parity
        // turned. The sparrow's feather cards read their shape off a map by
        // their second set, and read by the first they were cut to the
        // wrong texels.
        auto moved = convert(true, true);
        REQUIRE(moved);
        const std::vector<float>* movedCount = (*moved)->attached("splats");
        REQUIRE(movedCount != nullptr);
        const auto keptMoved = static_cast<uint32_t>((*movedCount)[0]);
        const double movedRatio = static_cast<double>(keptMoved) / static_cast<double>(full);
        std::printf("  cut by the first set keeps %u of %u, by the second %u\n", kept, full, keptMoved);
        CHECK(movedRatio > 0.45);
        CHECK(movedRatio < 0.55);
        CHECK(violationsOf(*moved, keptMoved, 1) == 0);
        CHECK(violationsOf(*moved, keptMoved, 0) > keptMoved / 4);

        // A SLICE: from the second triangle on, the quad's other half, and the
        // effect says every triangle fit. With a budget under the whole it
        // says which triangle the budget cut into instead -- the second, for
        // a budget the first fills -- which is where a host's next slice
        // starts.
        const auto slice = [&](uint32_t first, uint64_t maxSplats) {
            aofx_host::EffectJob job;
            job.bounds = records->bounds();
            job.inputs.push_back({"Mesh", mesh});
            number(job, "triangles", 2);
            number(job, "resolution", kResolution);
            number(job, "maxSplats", static_cast<double>(maxSplats));
            number(job, "writePbr", 1.0);
            number(job, "firstTriangle", first);
            auto out = aofx_host::renderEffect(*gpu, *effect, job);
            REQUIRE(out);
            const std::vector<float>* counts = (*out)->attached("splats");
            REQUIRE(counts != nullptr);
            REQUIRE(counts->size() >= 6);
            return std::array<uint32_t, 3>{static_cast<uint32_t>((*counts)[0]), static_cast<uint32_t>((*counts)[1]),
                                           static_cast<uint32_t>((*counts)[5])};
        };
        const auto second = slice(1, budget);
        std::printf("  from the second triangle: %u written of %u wanted, cut at triangle %u\n", second[0],
                    second[1], second[2]);
        CHECK(second[0] == second[1]);
        CHECK(second[0] > full / 3);
        CHECK(second[0] < full * 2 / 3);
        CHECK(second[2] == 2);
        const auto starved = slice(0, full / 2);
        std::printf("  the whole at half its budget: %u written of %u wanted, cut at triangle %u\n", starved[0],
                    starved[1], starved[2]);
        CHECK(starved[1] == full);
        CHECK(starved[2] <= 1);
        // And a triangle the budget cuts into is written by no run at all:
        // what the starved run wrote plus the run from where it was cut is
        // the whole, once each.
        const auto rest = slice(starved[2], budget);
        std::printf("  from where the budget cut: %u written; %u + %u == %u\n", rest[0], starved[0], rest[0],
                    full);
        CHECK(starved[0] + rest[0] == full);
    }));
}

// A NORMAL MAP IS READ IN THE FRAME ITS UVS GIVE, as the mesh reads it.
//
// The conversion took the tangent from the triangle's longest edge, which is
// the gaussian's own axis and nothing the map was painted against: on this
// quad that edge is the diagonal, while u runs along x. A constant map
// leaning towards +u must turn every gaussian towards +x, as the mesh's
// shading (dP/du, technique/material_surface.slang) turns its normal; read
// along the diagonal it leaned 20 degrees the wrong way.
TEST_CASE("a normal map turns a gaussian in the frame its UVs give", "[aofx][mesh2splat]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuad");
        auto fill = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sFill");
        auto check = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sCheckQuad");
        REQUIRE(quad);
        REQUIRE(fill);
        REQUIRE(check);

        constexpr uint32_t kMeshWidth = 64;
        constexpr int32_t  kMapSize = 16;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const image::ImagePtr normals = pictureOf(kMapSize, kMapSize);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const gpu::Buffer normalView = viewOf(*gpu, normals);
        const uint32_t budget = kResolution * kResolution * 2;
        const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 6), 1);
        // Tangent space (0.5, 0, 1): towards +u, away from the surface.
        const std::array<float, 4> texel{0.75F, 0.5F, 1.0F, 1.0F};

        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(normalView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        fill->dispatch(batch, {kMapSize, kMapSize, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(normalView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["albedoWidth"].setData(static_cast<uint32_t>(kMapSize));
            cursor["params"]["albedoHeight"].setData(static_cast<uint32_t>(kMapSize));
            cursor["params"]["albedoStride"].setData(static_cast<uint32_t>(normals->stride()));
            cursor["params"]["colour"].setData(texel.data(), 16);
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();
        normals->deviceWrote();

        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});
        aofx_host::EffectJob job;
        job.bounds = records->bounds();
        job.inputs.push_back({"Mesh", mesh});
        job.inputs.push_back({"Normal", normals});
        number(job, "triangles", 2);
        number(job, "resolution", kResolution);
        number(job, "maxSplats", budget);
        number(job, "useNormalMap", 1.0);
        auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE(rendered);
        const std::vector<float>* counted = (*rendered)->attached("splats");
        REQUIRE(counted != nullptr);
        const auto written = static_cast<uint32_t>((*counted)[0]);
        REQUIRE(written >= kResolution * kResolution);
        const auto recordPixels = static_cast<uint32_t>((*counted)[4]);

        const gpu::Buffer recordView = viewOf(*gpu, *rendered);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "test.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);
        const float length = std::sqrt(1.25F);
        const std::array<float, 4> axis{0.5F / length, 0.0F, 1.0F / length, 0.0F};
        gpu::CommandBatch second(library.device());
        check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(normalView.rhi());
            cursor["records"].setBinding(recordView.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["splats"].setData(written);
            cursor["params"]["recordPixels"].setData(recordPixels);
            cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
            cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
            cursor["params"]["resolution"].setData(kResolution);
            cursor["params"]["sigma"].setData(kSigma);
            cursor["params"]["flatness"].setData(kFlatness);
            cursor["params"]["tolerance"].setData(5.0e-3F);
            cursor["params"]["axis"].setData(axis.data(), 16);
        });
        REQUIRE(second.submit(true));
        auto violations = counts->readAll<uint32_t>(library.device());
        REQUIRE(violations);
        // Only the turn is this test's: the quad's other claims assume an
        // untilted frame.
        INFO("turned the wrong way: " << (*violations)[2] << " of " << written);
        CHECK((*violations)[2] == 0);
    }));
}

TEST_CASE("glass takes its transmission colour, and stops what the caller says", "[aofx][mesh2splat]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuad");
        auto check = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sCheckQuad");
        REQUIRE(quad);
        REQUIRE(check);
        constexpr uint32_t kMeshWidth = 64;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const uint32_t budget = kResolution * kResolution * 2;
        const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 4), 1);

        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();

        // Twice: with the whole opacity kept, which is what a gaussian did
        // before the parameter existed, and with a quarter of it, which is
        // the ball you can see the room through. A translucent material is
        // not a transparent one, so the default keeps everything and the
        // caller is the one who says otherwise.
        for (const double glass : {1.0, 0.25}) {
            mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});
            aofx_host::EffectJob job;
            job.bounds = records->bounds();
            job.inputs.push_back({"Mesh", mesh});
            number(job, "triangles", 2);
            number(job, "resolution", kResolution);
            number(job, "maxSplats", budget);
            number(job, "flatness", kFlatness);
            number(job, "opacity", 1.0);
            number(job, "writePbr", 0.0);
            // Glass: fully transmitting. What the conversion writes is what the
            // material says -- its opacity, its colour taken towards the
            // transmission colour, and the transmission in a channel of its own.
            // What a renderer makes of that is the renderer's.
            number(job, "transmission", 1.0);
        number(job, "glassOpacity", glass);
            job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});
            job.params.push_back(aofx::ParamValue{"materialColour", {1.0, 1.0, 1.0}, {}});
            job.params.push_back(aofx::ParamValue{"transmissionColour", {0.2, 0.5, 0.4}, {}});

            auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
            REQUIRE(rendered);
            const std::vector<float>* counted = (*rendered)->attached("splats");
            REQUIRE(counted != nullptr);
            const auto written = static_cast<uint32_t>((*counted)[0]);
            REQUIRE(written >= kResolution * kResolution);

            const gpu::Buffer recordView = viewOf(*gpu, *rendered);
            gpu::BufferDesc desc;
            desc.bytes = 8 * 4;
            desc.elementBytes = 4;
            desc.label = "test.counts";
            auto counts = gpu::Buffer::create(library.device(), desc);
            REQUIRE(counts);
            gpu::CommandBatch second(library.device());
            check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["mesh"].setBinding(meshView.rhi());
                cursor["albedo"].setBinding(meshView.rhi());
                cursor["records"].setBinding(recordView.rhi());
                cursor["counts"].setBinding(counts->rhi());
                cursor["params"]["splats"].setData(written);
                cursor["params"]["recordPixels"].setData(uint32_t{4});
                cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
                cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
                cursor["params"]["resolution"].setData(kResolution);
                cursor["params"]["sigma"].setData(kSigma);
                cursor["params"]["flatness"].setData(kFlatness);
                cursor["params"]["tolerance"].setData(1.0e-3F);
                cursor["params"]["opacity"].setData(static_cast<float>(glass));
                const std::array<float, 4> colour{0.2F, 0.5F, 0.4F, 1.0F};
                const std::array<float, 4> axis{0.0F, 0.0F, 1.0F, 0.0F};
                cursor["params"]["colour"].setData(colour.data(), 16);
                cursor["params"]["axis"].setData(axis.data(), 16);
            });
            REQUIRE(second.submit(true));
            auto violations = counts->readAll<uint32_t>(library.device());
            REQUIRE(violations);
            CHECK((*violations)[3] == 0);   // the transmission colour, multiplied in
            CHECK((*violations)[4] == 0);   // and the opacity the caller left it
        }
    }));
}

// THE SAME MESH MUST GIVE THE SAME ARRAY, not the same set in another order.
//
// The slot used to be taken with an atomic and the shader said so: "the order
// of the output is nobody's business". Two conversions of the chess pawn then
// wrote files that differed from the thousandth byte, which is fine for one
// still frame and impossible for a sequence -- a gaussian is followed from one
// pose to the next by being the same element. Counting each triangle's cells,
// settling where each triangle starts, and then writing at that offset makes
// the order the mesh's own.
TEST_CASE("two conversions of one mesh write the same array, bit for bit", "[aofx][mesh2splat]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuad");
        auto compare = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_compare", "m2sCompare");
        REQUIRE(quad);
        REQUIRE(compare);

        constexpr uint32_t kMeshWidth = 64;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const uint32_t budget = kResolution * kResolution * 2;
        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();
        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});

        const auto convert = [&]() -> image::ImagePtr {
            const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 4), 1);
            aofx_host::EffectJob job;
            job.bounds = records->bounds();
            job.inputs.push_back({"Mesh", mesh});
            number(job, "triangles", 2);
            number(job, "resolution", kResolution);
            number(job, "maxSplats", budget);
            number(job, "flatness", kFlatness);
            number(job, "opacity", 1.0);
            number(job, "writePbr", 0.0);
            job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});
            auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
            return rendered ? *rendered : image::ImagePtr{};
        };
        const image::ImagePtr one = convert();
        const image::ImagePtr two = convert();
        REQUIRE(one);
        REQUIRE(two);

        const gpu::Buffer viewOne = viewOf(*gpu, one);
        const gpu::Buffer viewTwo = viewOf(*gpu, two);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "compare.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);
        const auto entries = static_cast<uint32_t>(kResolution * kResolution * 4);
        gpu::CommandBatch second(library.device());
        compare->dispatch(second, {entries, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["first"].setBinding(viewOne.rhi());
            cursor["second"].setBinding(viewTwo.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["entries"].setData(entries);
            cursor["params"]["width"].setData(static_cast<uint32_t>(one->bounds().width()));
            cursor["params"]["stride"].setData(static_cast<uint32_t>(one->stride()));
        });
        REQUIRE(second.submit(true));
        uint32_t seen[2] = {0, 0};
        REQUIRE(counts->read(library.device(), 0, sizeof(seen), seen));
        std::printf("  %u entries compared, %u differ\n", seen[0], seen[1]);
        CHECK(seen[0] == entries);
        CHECK(seen[1] == 0);
    }));
}

// A GAUSSIAN IS CARRIED BY THE JOINTS ITS TRIANGLE IS CARRIED BY.
//
// The conversion blends the three corners' influences by the barycentric
// coordinates of the cell the gaussian stands in, which is what interpolating
// the skin means. Where every corner names one joint with all of its weight,
// every blend is that joint with all of its weight whatever the coordinates
// are -- and that is the case that catches the plumbing: a picture that did
// not arrive, an entry addressed wrong, a weight left unnormalised.
TEST_CASE("a gaussian keeps the joints of the triangle it stands on", "[aofx][mesh2splat][skinning]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuad");
        auto joints = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sInfluences");
        auto check = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sCheckInfluences");
        REQUIRE(quad);
        REQUIRE(joints);
        REQUIRE(check);

        constexpr uint32_t kMeshWidth = 64;
        constexpr float    kJoint = 3.0F;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const image::ImagePtr carried = pictureOf(kMeshWidth, 1);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const gpu::Buffer carriedView = viewOf(*gpu, carried);
        const uint32_t budget = kResolution * kResolution * 2;

        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        joints->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(carriedView.rhi());
            cursor["albedo"].setBinding(carriedView.rhi());
            cursor["records"].setBinding(carriedView.rhi());
            cursor["counts"].setBinding(carriedView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(carried->stride()));
            cursor["params"]["joint"].setData(kJoint);
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();
        carried->deviceWrote();
        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});

        const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 8), 1);
        aofx_host::EffectJob job;
        job.bounds = records->bounds();
        job.inputs.push_back({"Mesh", mesh});
        job.inputs.push_back({"Influences", carried});
        number(job, "triangles", 2);
        number(job, "resolution", kResolution);
        number(job, "maxSplats", budget);
        number(job, "flatness", kFlatness);
        number(job, "opacity", 1.0);
        number(job, "writePbr", 1.0);
        number(job, "writeInfluences", 1.0);
        job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});
        auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE(rendered);
        const std::vector<float>* counted = (*rendered)->attached("splats");
        REQUIRE(counted != nullptr);
        const auto written = static_cast<uint32_t>((*counted)[0]);
        REQUIRE(written >= kResolution * kResolution);
        // Eight entries a record, which is what the joints asked for.
        REQUIRE((*counted)[4] == 8.0F);

        const gpu::Buffer recordView = viewOf(*gpu, *rendered);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "joints.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);
        gpu::CommandBatch second(library.device());
        check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(recordView.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["splats"].setData(written);
            cursor["params"]["recordPixels"].setData(uint32_t{8});
            cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
            cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
            cursor["params"]["tolerance"].setData(1.0e-5F);
            cursor["params"]["joint"].setData(kJoint);
        });
        REQUIRE(second.submit(true));
        uint32_t violations[4] = {0, 0, 0, 0};
        REQUIRE(counts->read(library.device(), 0, sizeof(violations), violations));
        std::printf("  %u gaussians carried; %u wrong joint, %u spilled, %u unnormalised\n", written,
                    violations[0], violations[1], violations[2]);
        CHECK(violations[0] == 0);   // every one on the joint its corners named
        CHECK(violations[1] == 0);   // and nothing in the other three slots
        CHECK(violations[2] == 0);   // and the weights a whole
    }));
}

// A HEIGHT MAP RAISES EVERY GAUSSIAN AND SPLITS THE CELLS IT STRETCHES.
//
// The unit quad under a tent of height (`m2sTent`): 0 at both edges and at
// the middle half its scale, a slope of one on each side at a scale of a
// half. Every gaussian must stand on the quad raised by what the map reads
// there, face as the slope does, and be sized for a cell stretched by the
// slope's `sqrt(2)` and split in two along it -- so the quad takes twice the
// gaussians it takes flat. And a material with no map but a height of its
// own moves the whole surface and changes nothing else.
TEST_CASE("a displaced gaussian stands on the relief, faces it, and a steep cell is split",
          "[aofx][mesh2splat][displacement]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    struct Run {
        const char* name;
        bool        withMap;
        double      scale;
        double      bias;
        uint32_t    least;
        uint32_t    most;
    };
    // Flat, the quad is 256 cells and the diagonal's (256 to 272). Raised by
    // the tent, every cell off the ridge and the edges is split in two.
    const Run runs[] = {
        {"a tent of height, a slope of one", true, 0.5, 0.0, 440, 560},
        {"a constant height: the surface moved, nothing split", false, 0.0, 0.25, 256, 272},
    };
    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuad");
        auto tent = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sTent");
        auto check = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sCheckRelief");
        REQUIRE(quad);
        REQUIRE(tent);
        REQUIRE(check);
        for (const Run& run : runs) {
            constexpr uint32_t kMeshWidth = 64;
            constexpr int32_t  kMapSize = 128;
            const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
            const image::ImagePtr heights = pictureOf(kMapSize, kMapSize);
            const gpu::Buffer meshView = viewOf(*gpu, mesh);
            const gpu::Buffer heightView = viewOf(*gpu, heights);
            const uint32_t budget = kResolution * kResolution * 4;
            const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 9), 1);

            gpu::CommandBatch batch(library.device());
            quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["mesh"].setBinding(meshView.rhi());
                cursor["albedo"].setBinding(heightView.rhi());
                cursor["records"].setBinding(meshView.rhi());
                cursor["counts"].setBinding(meshView.rhi());
                cursor["params"]["meshWidth"].setData(kMeshWidth);
                cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
            });
            tent->dispatch(batch, {kMapSize, kMapSize, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["mesh"].setBinding(meshView.rhi());
                cursor["albedo"].setBinding(heightView.rhi());
                cursor["records"].setBinding(meshView.rhi());
                cursor["counts"].setBinding(meshView.rhi());
                cursor["params"]["albedoWidth"].setData(static_cast<uint32_t>(kMapSize));
                cursor["params"]["albedoHeight"].setData(static_cast<uint32_t>(kMapSize));
                cursor["params"]["albedoStride"].setData(static_cast<uint32_t>(heights->stride()));
            });
            REQUIRE(batch.submit(true));
            mesh->deviceWrote();
            heights->deviceWrote();

            mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});
            aofx_host::EffectJob job;
            job.bounds = records->bounds();
            job.inputs.push_back({"Mesh", mesh});
            if (run.withMap) {
                job.inputs.push_back({"Displacement", heights});
            }
            number(job, "triangles", 2);
            number(job, "resolution", kResolution);
            number(job, "maxSplats", budget);
            number(job, "flatness", kFlatness);
            number(job, "writePbr", 1.0);
            number(job, "displace", 1.0);
            number(job, "displaceChannel", 1.0);
            number(job, "displaceScale", run.scale);
            number(job, "displaceBias", run.bias);
            number(job, "displaceRefine", 8.0);
            job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});

            auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
            REQUIRE(rendered);
            const std::vector<float>* counted = (*rendered)->attached("splats");
            REQUIRE(counted != nullptr);
            REQUIRE(counted->size() >= 7);
            const auto written = static_cast<uint32_t>((*counted)[0]);
            std::printf("  %-55s %u gaussians\n", run.name, written);
            CHECK(written >= run.least);
            CHECK(written <= run.most);
            CHECK((*counted)[4] == 9.0F);   // six entries and the relief's three
            CHECK((*counted)[6] == 0.0F);   // no cell wanted more than the cap

            const gpu::Buffer recordView = viewOf(*gpu, *rendered);
            gpu::BufferDesc desc;
            desc.bytes = 8 * 4;
            desc.elementBytes = 4;
            desc.label = "test.counts";
            auto counts = gpu::Buffer::create(library.device(), desc);
            REQUIRE(counts);
            gpu::CommandBatch second(library.device());
            check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["mesh"].setBinding(meshView.rhi());
                cursor["albedo"].setBinding(heightView.rhi());
                cursor["records"].setBinding(recordView.rhi());
                cursor["counts"].setBinding(counts->rhi());
                cursor["params"]["splats"].setData(written);
                cursor["params"]["recordPixels"].setData(uint32_t{9});
                cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
                cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
                cursor["params"]["resolution"].setData(kResolution);
                cursor["params"]["sigma"].setData(kSigma);
                cursor["params"]["tolerance"].setData(2.0e-3F);
                cursor["params"]["albedoWidth"].setData(static_cast<uint32_t>(kMapSize));
                cursor["params"]["heightScale"].setData(static_cast<float>(run.scale));
                cursor["params"]["heightBias"].setData(static_cast<float>(run.bias));
            });
            REQUIRE(second.submit(true));
            auto violations = counts->readAll<uint32_t>(library.device());
            REQUIRE(violations);
            std::printf("    violations %u %u %u %u %u, clear %u\n", (*violations)[0], (*violations)[1],
                        (*violations)[2], (*violations)[3], (*violations)[4], (*violations)[5]);
            CHECK((*violations)[0] == 0);   // raised along the normal by what it says
            CHECK((*violations)[1] == 0);   // by what the map reads there
            CHECK((*violations)[2] == 0);   // facing as the slope does
            CHECK((*violations)[3] == 0);   // and turned to that
            CHECK((*violations)[4] == 0);   // sized for its share of the stretched cell
        }
    }));
}

// FEWER GAUSSIANS WHERE THE SURFACE IS THE SAME, AND NOT ONE FEWER ELSEWHERE.
//
// The unit quad at 64 cells across, one colour all over: with `simplify`, the
// blocks wholly inside a triangle with a block to spare on every side become
// one gaussian each, so the quad takes far fewer, all on it, each one, two,
// four or eight cells wide and all of the one colour. Under a checker whose
// squares are four cells wide -- narrower than the reach of any block that
// could merge -- nothing merges and the count is the unsimplified one, to
// the gaussian.
TEST_CASE("a surface that is the same across a block becomes one gaussian, and detail keeps its cells",
          "[aofx][mesh2splat][simplify]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());
    constexpr uint32_t kCells = 64;

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sQuad");
        auto checker = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sChecker");
        auto check = gpu::ComputeKernel::create(library, "athenea/test/mesh2splat_check", "m2sCheckBlocks");
        REQUIRE(quad);
        REQUIRE(checker);
        REQUIRE(check);
        constexpr uint32_t kMeshWidth = 64;
        constexpr int32_t  kMapSize = 128;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const image::ImagePtr albedo = pictureOf(kMapSize, kMapSize);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const gpu::Buffer albedoView = viewOf(*gpu, albedo);
        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(albedoView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        checker->dispatch(batch, {kMapSize, kMapSize, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(albedoView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["albedoWidth"].setData(static_cast<uint32_t>(kMapSize));
            cursor["params"]["albedoHeight"].setData(static_cast<uint32_t>(kMapSize));
            cursor["params"]["albedoStride"].setData(static_cast<uint32_t>(albedo->stride()));
            cursor["params"]["checker"].setData(kCells / 4);   // squares four cells wide
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();
        albedo->deviceWrote();
        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});

        const uint32_t budget = kCells * kCells * 2;
        const auto convert = [&](double simplify, bool withChecker) {
            const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 6), 1);
            aofx_host::EffectJob job;
            job.bounds = records->bounds();
            job.inputs.push_back({"Mesh", mesh});
            if (withChecker) {
                job.inputs.push_back({"Albedo", albedo});
            }
            number(job, "triangles", 2);
            number(job, "resolution", kCells);
            number(job, "maxSplats", budget);
            number(job, "flatness", kFlatness);
            number(job, "writePbr", 1.0);
            number(job, "simplify", simplify);
            number(job, "simplifyLevels", 3.0);
            job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});
            job.params.push_back(aofx::ParamValue{"materialColour", {0.25, 0.5, 0.75}, {}});
            auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
            REQUIRE(rendered);
            const std::vector<float>* counted = (*rendered)->attached("splats");
            REQUIRE(counted != nullptr);
            return std::pair{*rendered, static_cast<uint32_t>((*counted)[0])};
        };
        const auto [full, fullCount] = convert(0.0, false);
        const auto [merged, mergedCount] = convert(0.02, false);
        const auto [checked, checkedFull] = convert(0.0, true);
        const auto [checkedMerged, checkedMergedCount] = convert(0.02, true);
        std::printf("  one colour: %u gaussians a cell, %u simplified; under a checker: %u and %u\n", fullCount,
                    mergedCount, checkedFull, checkedMergedCount);
        // 4160 to 2144 measured: a quad is two triangles, and a block must
        // have a block to spare inside its own on every side, so the diagonal
        // keeps a band of cells.
        CHECK(mergedCount * 10 < fullCount * 6);
        CHECK(checkedMergedCount == checkedFull);

        const gpu::Buffer recordView = viewOf(*gpu, merged);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "test.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);
        gpu::CommandBatch second(library.device());
        check->dispatch(second, {mergedCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(albedoView.rhi());
            cursor["records"].setBinding(recordView.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["splats"].setData(mergedCount);
            cursor["params"]["recordPixels"].setData(uint32_t{6});
            cursor["params"]["dstWidth"].setData(static_cast<uint32_t>(merged->bounds().width()));
            cursor["params"]["dstStride"].setData(static_cast<uint32_t>(merged->stride()));
            cursor["params"]["resolution"].setData(kCells);
            cursor["params"]["sigma"].setData(kSigma);
            cursor["params"]["flatness"].setData(kFlatness);
            cursor["params"]["tolerance"].setData(1.0e-3F);
            const std::array<float, 4> colour{0.25F, 0.5F, 0.75F, 1.0F};
            cursor["params"]["colour"].setData(colour.data(), 16);
        });
        REQUIRE(second.submit(true));
        auto violations = counts->readAll<uint32_t>(library.device());
        REQUIRE(violations);
        std::printf("  simplified: %u off the quad, %u sized otherwise, %u of another colour, %u blocks\n",
                    (*violations)[0], (*violations)[1], (*violations)[2], (*violations)[3]);
        CHECK((*violations)[0] == 0);
        CHECK((*violations)[1] == 0);
        CHECK((*violations)[2] == 0);
        CHECK((*violations)[3] > 0);
    }));
}
