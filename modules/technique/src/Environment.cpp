// Copyright (c) 2026 jesus luque.
#include "athenea/technique/Environment.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "athenea/core/Log.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/light/LightTable.h"
#include "athenea/material/TextureStore.h"

namespace athenea::technique {
namespace {

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t elements, uint32_t elementBytes,
                                 const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(elements, 1) * elementBytes;
    desc.elementBytes = elementBytes;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

/// The side of a roughness level, as `environment.slang` computes it: level 0
/// is the sky's own width and the levels above it are capped, because a lobe
/// that wide cannot show more.
uint32_t sideOf(uint32_t level, uint32_t baseSide) {
    const uint32_t base = std::max(baseSide, kEnvironmentCoarseSide);
    return std::max(base >> std::min(level, kEnvironmentLevels - 1), kEnvironmentFloorSide);
}

/// Directions a level takes to be free of the source's own noise.
///
/// Level 0 is the sky as it is and needs one. Above it the lobe widens as the
/// level narrows, so the count doubles as the texels quarter: the work per
/// level halves going up the chain instead of staying flat, which is what
/// keeps a 2048 base to about thirty million samples in all.
uint32_t samplesOf(uint32_t level) {
    // A sun of tens of thousands in a lobe wants many more than the sky's
    // own noise does: 128 at level 4 drew the polish's highlight as a
    // scatter of blobs (s81). Filtered importance sampling takes most of
    // that; the count does the rest, and the coarse levels are small.
    return level == 0 ? 1U : std::min(64U << (level - 1), 1024U);
}

}   // namespace

uint32_t Environment::texelsPerDome(uint32_t baseSide) noexcept {
    uint32_t texels = 0;
    for (uint32_t level = 0; level < kEnvironmentLevels; ++level) {
        const uint32_t side = sideOf(level, baseSide);
        texels += side * side;
    }
    return texels;
}

uint32_t Environment::baseSideFor(uint32_t width, uint32_t height) noexcept {
    if (width == 0 || height == 0) {
        return kEnvironmentCoarseSide;
    }
    const double want = std::sqrt(2.0 * static_cast<double>(width) * static_cast<double>(height) /
                                  3.14159265358979);
    uint32_t side = kEnvironmentCoarseSide;
    while (side < kEnvironmentWidestSide && static_cast<double>(side * 2) <= want) {
        side *= 2;
    }
    return side;
}

Result<Environment> Environment::create(gpu::ShaderLibrary& library) {
    Environment env;
    env.device_ = &library.device();
    auto project = gpu::ComputeKernel::create(library, "athenea/technique/env_project", "envProject");
    if (!project) return std::move(project).error();
    auto prefilter = gpu::ComputeKernel::create(library, "athenea/technique/env_prefilter", "envPrefilter");
    if (!prefilter) return std::move(prefilter).error();
    auto sun = gpu::ComputeKernel::create(library, "athenea/technique/env_sun", "envSun");
    if (!sun) return std::move(sun).error();
    env.project_ = std::move(*project);
    env.prefilter_ = std::move(*prefilter);
    env.sunKernel_ = std::move(*sun);
    auto meshPack = gpu::ComputeKernel::create(library, "athenea/technique/env_mesh", "envMeshPack");
    if (!meshPack) return std::move(meshPack).error();
    env.meshPack_ = std::move(*meshPack);
    return env;
}

Result<void> Environment::build(const light::LightTable& table, const material::TextureStore& textures,
                                std::span<const uint32_t> domeLights,
                                std::span<const uint32_t> domeTextures, uint32_t lightCount) {
    const auto domes = static_cast<uint32_t>(std::min<size_t>(domeLights.size(), kEnvironmentDomes));
    domes_ = domes;
    lights_ = lightCount;
    if (domes == 0) {
        return ok();
    }
    // ONE WIDTH FOR EVERY SLICE, because they share a buffer with one stride.
    // The widest sky in the frame decides it: a small dome beside a 4k one
    // pays for texels it cannot fill, which is cheaper than a second layout.
    uint32_t baseSide = kEnvironmentCoarseSide;
    for (uint32_t slice = 0; slice < domes; ++slice) {
        const uint32_t id = slice < domeTextures.size() ? domeTextures[slice] : kEnvironmentNone;
        if (id >= textures.count()) {
            continue;
        }
        const auto info = textures.info(id);
        baseSide = std::max(baseSide, baseSideFor(info.width, info.height));
    }
    baseSide_ = baseSide;
    // Grown, never shrunk: a frame that loses a dome keeps the allocation for
    // the next one that has it.
    const uint64_t words = uint64_t{domes} * texelsPerDome(baseSide) * 2;
    if (!texels_.valid() || texels_.bytes() < words * 4) {
        auto made = deviceBuffer(*device_, words, 4, "environment.texels");
        if (!made) return std::move(made).error();
        texels_ = std::move(*made);
    }
    if (!sh_.valid() || sh_.bytes() < uint64_t{domes} * kEnvironmentCoefficients * 16) {
        auto made = deviceBuffer(*device_, uint64_t{domes} * kEnvironmentCoefficients, 16, "environment.sh");
        if (!made) return std::move(made).error();
        sh_ = std::move(*made);
    }
    if (!sun_.valid() || sun_.bytes() < uint64_t{domes} * 2 * 16) {
        auto made = deviceBuffer(*device_, uint64_t{domes} * 2, 16, "environment.sun");
        if (!made) return std::move(made).error();
        sun_ = std::move(*made);
    }

    // Which slice a light was given, for the frame to read beside its lights.
    std::vector<uint32_t> ofLight(std::max(lightCount, 1U), kEnvironmentNone);
    for (uint32_t slice = 0; slice < domes; ++slice) {
        if (domeLights[slice] < lightCount) {
            ofLight[domeLights[slice]] = slice;
        }
    }
    auto perLight = gpu::Buffer::fromSpan<uint32_t>(*device_, ofLight, "environment.domeOfLight");
    if (!perLight) return std::move(perLight).error();
    domeOfLight_ = std::move(*perLight);
    const std::vector<uint32_t> slices(domeLights.begin(), domeLights.begin() + domes);
    auto sliceBuffer = gpu::Buffer::fromSpan<uint32_t>(*device_, slices, "environment.domeLights");
    if (!sliceBuffer) return std::move(sliceBuffer).error();

    // WHICH LEVEL OF THE SKY THE PROJECTION READS. Nine coefficients over
    // every texel of a 4k sky is a million and a half reads in nine lanes,
    // and what it would buy is a decimal place of a number that a cosine lobe
    // is about to blur anyway. A level no wider than 256 is plenty, and the
    // chain is already there. It is measured on the domes' own images: the
    // widest texture in the store may be an albedo map that no sky reads.
    uint32_t projectLevel = 0;
    uint32_t widest = 0;
    for (uint32_t slice = 0; slice < domes; ++slice) {
        const uint32_t id = slice < domeTextures.size() ? domeTextures[slice] : kEnvironmentNone;
        if (id < textures.count()) {
            widest = std::max(widest, textures.info(id).width);
        }
    }
    while (widest > 256 && projectLevel < 12) {
        widest >>= 1;
        ++projectLevel;
    }

    gpu::CommandBatch batch(*device_);
    // THE SUN FIRST, because the projection has to know what not to project.
    // One thread a dome, walking 4096 directions twice: the sky's mean, its
    // brightest cell, and the disc around it. It reads a level of the source
    // fine enough to keep a disc from being averaged away -- two below the
    // one the harmonics read, which on a 4k sky is 1024 wide.
    const uint32_t sunLevel = projectLevel >= 2 ? projectLevel - 2 : 0;
    sunKernel_.dispatch(batch, {domes, 1, 1}, [&](rhi::ShaderCursor cursor) {
        table.bind(cursor);
        textures.bind(cursor["gTextures"]);
        cursor["domeLights"].setBinding(sliceBuffer->rhi());
        cursor["sunOut"].setBinding(sun_.rhi());
        cursor["params"]["domes"].setData(domes);
        cursor["params"]["level"].setData(sunLevel);
        cursor["params"]["projectLevel"].setData(projectLevel);
    });
    project_.dispatch(batch, {domes * kEnvironmentCoefficients, 1, 1}, [&](rhi::ShaderCursor cursor) {
        table.bind(cursor);
        textures.bind(cursor["gTextures"]);
        cursor["domeLights"].setBinding(sliceBuffer->rhi());
        cursor["sun"].setBinding(sun_.rhi());
        cursor["shOut"].setBinding(sh_.rhi());
        cursor["params"]["domes"].setData(domes);
        cursor["params"]["level"].setData(projectLevel);
    });
    for (uint32_t level = 0; level < kEnvironmentLevels; ++level) {
        const uint32_t side = sideOf(level, baseSide);
        // (k/7)^2, the curve `envRadiance` walks back with a square root.
        const float step = static_cast<float>(level) / static_cast<float>(kEnvironmentLevels - 1);
        const float roughness = step * step;
        prefilter_.dispatch(batch, {domes * side * side, 1, 1}, [&](rhi::ShaderCursor cursor) {
            table.bind(cursor);
            textures.bind(cursor["gTextures"]);
            cursor["domeLights"].setBinding(sliceBuffer->rhi());
            cursor["texelsOut"].setBinding(texels_.rhi());
            cursor["sun"].setBinding(sun_.rhi());   // the map is the sky without its sun
            cursor["params"]["domes"].setData(domes);
            cursor["params"]["level"].setData(level);
            cursor["params"]["samples"].setData(samplesOf(level));
            cursor["params"]["sourceLevel"].setData(projectLevel);
            cursor["params"]["baseSide"].setData(baseSide);
            cursor["params"]["roughness"].setData(roughness);
        });
    }
    // And the harmonics and the sun copied into the texture a mesh's shading
    // kernel reads, a row a slice (env_mesh.slang).
    if (!mesh_.valid()) {
        gpu::TextureDesc desc;
        desc.type = rhi::TextureType::Texture2D;
        desc.width = kEnvironmentMeshTexels;
        desc.height = kEnvironmentDomes;
        desc.format = rhi::Format::RGBA32Float;
        desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
        desc.label = "environment.mesh";
        auto made = gpu::Texture::create(*device_, desc);
        if (!made) return std::move(made).error();
        mesh_ = std::move(*made);
        auto view = mesh_.view(0);
        if (!view) return std::move(view).error();
        meshView_ = std::move(*view);
    }
    meshPack_.dispatch(batch, {kEnvironmentMeshTexels * kEnvironmentDomes, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["sh"].setBinding(sh_.rhi());
        cursor["sun"].setBinding(sun_.rhi());
        cursor["domeLights"].setBinding(sliceBuffer->rhi());
        cursor["meshOut"].setBinding(meshView_.get());
        cursor["params"]["domes"].setData(domes);
        cursor["params"]["rows"].setData(kEnvironmentDomes);
    });
    // Waits for the device: this is not a frame, and the frame that follows
    // reads what it wrote.
    ATHENEA_TRY(batch.submit(true));
    // WHAT WAS TAKEN OUT OF EACH SKY, said out loud. Thirty-two bytes a dome
    // off a buffer the device has already finished with: it costs nothing and
    // it is the one thing about a sky an operator cannot see by looking.
    std::vector<float> read(size_t{domes} * 8, 0.0F);
    if (sun_.read(*device_, 0, read.size() * sizeof(float), read.data())) {
        for (uint32_t slice = 0; slice < domes; ++slice) {
            const float* row = read.data() + size_t{slice} * 8;
            if (row[3] > 0.0F) {
                log::info("sky {}: a sun at ({:.3f}, {:.3f}, {:.3f}), {:.2f} degrees across, irradiance "
                          "{:.3f} {:.3f} {:.3f} (its texels from {:.4g} up)",
                          slice, row[0], row[1], row[2], 2.0F * row[3] * 57.2957795F, row[4], row[5], row[6],
                          row[7]);
            } else {
                log::info("sky {}: no sun (nothing {:.0f} times its median and more: the harmonics hold it all)",
                          slice, 32.0F);
            }
        }
    }
    return ok();
}

}   // namespace athenea::technique
