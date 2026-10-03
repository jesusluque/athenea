// Copyright (c) 2026 jesus luque.
//
// TextureStore: files read back texel for texel, sRGB colour decoded by its
// view and averaged as light down its mips, UDIM tiles found by uv.
#include "../gpu/GpuTest.h"

#include <catch2/catch_approx.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include <pxr/imaging/hio/image.h>

#include "athenea/material/TextureStore.h"

using namespace athenea;
namespace fs = std::filesystem;
PXR_NAMESPACE_USING_DIRECTIVE

namespace {

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "material";
    fs::create_directories(dir);
    return dir / name;
}

gpu::Buffer floatBuffer(gpu::Device& device, uint64_t count, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = count * 16;
    desc.elementBytes = 16;
    desc.label = label;
    auto made = gpu::Buffer::create(device, desc);
    if (!made) FAIL(made.error().toString());
    return *made;
}

/// An RGBA8 PNG of the check kernel's pattern (or one colour), generated on
/// the device and written by Hio.
void writePattern(test::Gpu& gpu, const fs::path& path, uint32_t w, uint32_t h, const float* colour = nullptr) {
    static gpu::ComputeKernel kPattern = [&] {
        auto made = gpu::ComputeKernel::create(*gpu.library, "athenea/test/material_textures", "materialPattern");
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    }();
    const uint32_t words = (w * h * 4 + 3) / 4;
    gpu::Buffer out = test::uintBuffer(*gpu.device, words, "pattern");
    {
        std::vector<uint32_t> zero(words, 0);
        REQUIRE(out.write(*gpu.device, 0, uint64_t{words} * 4, zero.data()));
    }
    gpu::CommandBatch batch(*gpu.device);
    kPattern.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["words"].setBinding(out.rhi());
        cursor["params"]["width"].setData(w);
        cursor["params"]["height"].setData(h);
        cursor["params"]["constant"].setData(uint32_t{colour != nullptr ? 1u : 0u});
        cursor["params"]["colourR"].setData(colour != nullptr ? colour[0] : 0.0F);
        cursor["params"]["colourG"].setData(colour != nullptr ? colour[1] : 0.0F);
        cursor["params"]["colourB"].setData(colour != nullptr ? colour[2] : 0.0F);
    });
    REQUIRE(batch.submit(true));
    std::vector<uint32_t> bytes(words);
    REQUIRE(out.read(*gpu.device, 0, uint64_t{words} * 4, bytes.data()));   // a file's contents: output
    fs::remove(path);
    HioImageSharedPtr image = HioImage::OpenForWriting(path.string());
    REQUIRE(image);
    HioImage::StorageSpec spec;
    spec.width = static_cast<int>(w);
    spec.height = static_cast<int>(h);
    spec.depth = 1;
    spec.format = HioFormatUNorm8Vec4;
    spec.data = bytes.data();
    REQUIRE(image->Write(spec));
}

}   // namespace

TEST_CASE("a PNG comes back texel for texel, its sRGB decoded by its view and averaged as light down its mips",
          "[material][texture]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const uint32_t w = 37;
    const uint32_t h = 23;
    const fs::path png = scratch("pattern.png");
    writePattern(*gpu, png, w, h);
    auto store = material::TextureStore::create(*gpu->library);
    if (!store) FAIL(store.error().toString());
    const uint32_t raw = (*store)->request(png.string(), "raw");
    const uint32_t srgb = (*store)->request(png.string(), "srgb_texture");
    const uint32_t missing = (*store)->request(scratch("absent.png").string());
    CHECK((*store)->request(png.string(), "raw") == raw);
    auto loaded = (*store)->commit();
    REQUIRE(loaded);
    CHECK(*loaded == 2);
    CHECK((*store)->info(raw).loaded);
    CHECK((*store)->info(srgb).loaded);
    CHECK(!(*store)->info(missing).loaded);

    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/material_textures", "materialLevelCheck");
    auto mean = gpu::ComputeKernel::create(*gpu->library, "athenea/test/material_textures", "materialLevelMean");
    auto probe = gpu::ComputeKernel::create(*gpu->library, "athenea/test/material_textures", "materialTableProbe");
    REQUIRE(check);
    REQUIRE(mean);
    REQUIRE(probe);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        const gpu::Texture& texture = (*store)->slotTexture(slot);
        auto view = texture.view(0);
        REQUIRE(view);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["level"].setBinding((*view).get());
            cursor["counts"].setBinding(counts.rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["height"].setData(h);
        });
        REQUIRE(batch.submit(true));
        uint32_t c[2] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        std::printf("  slot %u: %u of %u components differ from the file\n", slot, c[0], c[1]);
        CHECK(c[0] == 0);
        CHECK(c[1] == w * h * 4);
        CHECK(texture.mipCount() == gpu::mipChain(w, h));
    }
    // The means of level 0 and the last level, as light: the sRGB chain keeps
    // its light, the raw one its code values.
    for (uint32_t slot = 0; slot < 2; ++slot) {
        const gpu::Texture& texture = (*store)->slotTexture(slot);
        const float decode = slot == 1 ? 1.0F : 0.0F;
        float means[2][4] = {};
        for (uint32_t k = 0; k < 2; ++k) {
            const uint32_t mip = k == 0 ? 0 : texture.mipCount() - 1;
            auto view = texture.view(mip);
            REQUIRE(view);
            gpu::Buffer out = floatBuffer(*gpu->device, 1, "mean");
            gpu::CommandBatch batch(*gpu->device);
            mean->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["level"].setBinding((*view).get());
                cursor["sampled"].setBinding(out.rhi());
                cursor["params"]["width"].setData(texture.width(mip));
                cursor["params"]["height"].setData(texture.height(mip));
                cursor["params"]["colourR"].setData(decode);
            });
            REQUIRE(batch.submit(true));
            REQUIRE(out.read(*gpu->device, 0, sizeof(means[k]), means[k]));
        }
        std::printf("  slot %u (%s): mean r %.5f at level 0, %.5f at 1x1\n", slot, slot == 1 ? "as light" : "raw",
                    double(means[0][0]), double(means[1][0]));
        for (int c = 0; c < 3; ++c) {
            // The last level is one texel: 8 bits of it, whatever the mean.
            CHECK(std::abs(means[0][c] - means[1][c]) <= (slot == 1 ? 0.01F : 0.6F / 255.0F));
        }
    }
    // Samples through the table: texel centres of the sRGB texture decode as
    // sRGB; the missing file reports itself missing.
    gpu::Buffer probes = floatBuffer(*gpu->device, 3, "probes");
    const float probeData[12] = {(3.0F + 0.5F) / w, (5.0F + 0.5F) / h, float(srgb), 1.0F,
                                 (3.0F + 0.5F) / w, (5.0F + 0.5F) / h, float(raw), 1.0F,
                                 0.5F, 0.5F, float(missing), 0.0F};
    REQUIRE(probes.write(*gpu->device, 0, sizeof(probeData), probeData));
    gpu::Buffer sampled = floatBuffer(*gpu->device, 3, "sampled");
    {
        gpu::CommandBatch batch(*gpu->device);
        probe->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            (*store)->bind(cursor["table"]);
            cursor["probes"].setBinding(probes.rhi());
            cursor["sampled"].setBinding(sampled.rhi());
            cursor["params"]["count"].setData(uint32_t{3});
        });
        REQUIRE(batch.submit(true));
    }
    float s[12] = {};
    REQUIRE(sampled.read(*gpu->device, 0, sizeof(s), s));
    // Texel (3, 5) from the bottom is file row h - 1 - 5.
    const uint32_t fy = h - 1 - 5;
    const float code = float((3 * 31 + fy * 17 + 0 * 53 + (3 * fy) % 7) % 256) / 255.0F;
    const float light = code <= 0.04045F ? code / 12.92F : std::pow((code + 0.055F) / 1.055F, 2.4F);
    std::printf("  texel (3, 5) red: %.4f through sRGB (want %.4f), %.4f raw (want %.4f); missing found %.0f\n",
                double(s[0]), double(light), double(s[4]), double(code), double(s[11]));
    CHECK(s[0] == Catch::Approx(light).margin(2e-3F));
    CHECK(s[4] == Catch::Approx(code).margin(1e-5F));
    CHECK(s[3] == 1.0F);
    CHECK(s[11] == 0.0F);
}

TEST_CASE("a footprint filtered by its gradients and by the level they pick agree", "[material][texture]") {
    ATHENEA_REQUIRE_GPU(gpu);
    // Not every target lets a compute entry point sample with gradients:
    // CUDA has no SampleGrad there, so the table falls back to the level the
    // footprint lands on. Here both paths exist, so they can be compared.
    const uint32_t w = 64;
    const uint32_t h = 64;
    const fs::path png = scratch("footprint.png");
    writePattern(*gpu, png, w, h);
    auto store = material::TextureStore::create(*gpu->library);
    if (!store) FAIL(store.error().toString());
    const uint32_t slot = (*store)->request(png.string(), "raw");
    auto loaded = (*store)->commit();
    REQUIRE(loaded);
    auto footprint = gpu::ComputeKernel::create(*gpu->library, "athenea/test/material_textures", "materialFootprint");
    if (!footprint) FAIL(footprint.error().toString());
    const uint32_t count = 4;
    gpu::Buffer probes = floatBuffer(*gpu->device, count, "probes");
    const float probeData[count * 4] = {0.5F, 0.5F, float(slot), 1.0F / float(w),
                                        0.5F, 0.5F, float(slot), 2.0F / float(w),
                                        0.5F, 0.5F, float(slot), 4.0F / float(w),
                                        0.5F, 0.5F, float(slot), 8.0F / float(w)};
    REQUIRE(probes.write(*gpu->device, 0, sizeof(probeData), probeData));
    gpu::Buffer sampled = floatBuffer(*gpu->device, count, "sampled");
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 1, "counts");
    {
        gpu::CommandBatch batch(*gpu->device);
        footprint->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            (*store)->bind(cursor["table"]);
            cursor["probes"].setBinding(probes.rhi());
            cursor["sampled"].setBinding(sampled.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["params"]["count"].setData(count);
            cursor["params"]["tolerance"].setData(4.0e-3F);
        });
        REQUIRE(batch.submit(true));
    }
    float s[count * 4] = {};
    REQUIRE(sampled.read(*gpu->device, 0, sizeof(s), s));
    uint32_t differ = 0;
    REQUIRE(counts.read(*gpu->device, 0, sizeof(differ), &differ));
    std::printf("  footprints of 1, 2, 4 and 8 texels: levels %.2f %.2f %.2f %.2f, %u samples apart\n",
                double(s[3]), double(s[7]), double(s[11]), double(s[15]), differ);
    for (uint32_t k = 0; k < count; ++k) {
        CHECK(s[k * 4 + 3] == Catch::Approx(float(k)).margin(1e-5F));
    }
    CHECK(differ == 0);
}

TEST_CASE("a UDIM set samples the tile its uv falls in, and nothing where it has none", "[material][texture][udim]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const float red[3] = {0.8F, 0.2F, 0.1F};
    const float blue[3] = {0.1F, 0.3F, 0.9F};
    writePattern(*gpu, scratch("tiles.1001.png"), 8, 8, red);
    writePattern(*gpu, scratch("tiles.1012.png"), 16, 16, blue);
    auto store = material::TextureStore::create(*gpu->library);
    if (!store) FAIL(store.error().toString());
    const uint32_t id = (*store)->request(scratch("tiles.<UDIM>.png").string(), "raw");
    auto loaded = (*store)->commit();
    REQUIRE(loaded);
    CHECK(*loaded == 2);
    CHECK((*store)->info(id).udim);
    auto probe = gpu::ComputeKernel::create(*gpu->library, "athenea/test/material_textures", "materialTableProbe");
    REQUIRE(probe);
    gpu::Buffer probes = floatBuffer(*gpu->device, 3, "probes");
    const float probeData[12] = {0.5F, 0.5F, float(id), 0.0F, 1.5F, 1.25F, float(id), 0.0F,
                                 2.5F, 0.5F, float(id), 0.0F};
    REQUIRE(probes.write(*gpu->device, 0, sizeof(probeData), probeData));
    gpu::Buffer sampled = floatBuffer(*gpu->device, 3, "sampled");
    {
        gpu::CommandBatch batch(*gpu->device);
        probe->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            (*store)->bind(cursor["table"]);
            cursor["probes"].setBinding(probes.rhi());
            cursor["sampled"].setBinding(sampled.rhi());
            cursor["params"]["count"].setData(uint32_t{3});
        });
        REQUIRE(batch.submit(true));
    }
    float s[12] = {};
    REQUIRE(sampled.read(*gpu->device, 0, sizeof(s), s));
    std::printf("  uv (0.5, 0.5): %.3f %.3f %.3f; (1.5, 1.25): %.3f %.3f %.3f; (2.5, 0.5) found %.0f\n",
                double(s[0]), double(s[1]), double(s[2]), double(s[4]), double(s[5]), double(s[6]), double(s[11]));
    CHECK(s[3] == 1.0F);
    CHECK(s[0] == Catch::Approx(std::round(red[0] * 255.0F) / 255.0F).margin(1e-5F));
    CHECK(s[2] == Catch::Approx(std::round(red[2] * 255.0F) / 255.0F).margin(1e-5F));
    CHECK(s[7] == 1.0F);
    CHECK(s[6] == Catch::Approx(std::round(blue[2] * 255.0F) / 255.0F).margin(1e-5F));
    CHECK(s[11] == 0.0F);
}

// 8-bit sRGB has two routes into light: the fast one, where the texture stays
// 8-bit behind an sRGB view and the sampler decodes, and the one every other
// colour space takes, where the decode kernel calls the function OpenColorIO
// compiled for it and the texture holds half floats. Same file, both routes,
// level 0 and the last level compared on the device.
TEST_CASE("an 8-bit sRGB texture reads the same through its sRGB view and through the compiled sRGB function",
          "[material][texture][colour]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const uint32_t w = 37;
    const uint32_t h = 23;
    const fs::path png = scratch("pattern_routes.png");
    writePattern(*gpu, png, w, h);
    auto fast = material::TextureStore::create(*gpu->library);
    auto compiled = material::TextureStore::create(*gpu->library);
    if (!fast) FAIL(fast.error().toString());
    if (!compiled) FAIL(compiled.error().toString());
    (*compiled)->setSrgbFastPath(false);
    const uint32_t a = (*fast)->request(png.string(), "srgb_texture");
    const uint32_t b = (*compiled)->request(png.string(), "srgb_texture");
    REQUIRE((*fast)->commit());
    REQUIRE((*compiled)->commit());
    REQUIRE((*fast)->info(a).loaded);
    REQUIRE((*compiled)->info(b).loaded);
    std::printf("  fast: %s; compiled: %s\n", (*fast)->info(a).decode.c_str(), (*compiled)->info(b).decode.c_str());
    CHECK((*fast)->info(a).decode == "srgb view");
    CHECK((*compiled)->info(b).decode != "srgb view");
    const gpu::Texture& eight = (*fast)->slotTexture(0);
    const gpu::Texture& half = (*compiled)->slotTexture(0);
    CHECK(eight.desc().format == rhi::Format::RGBA8Unorm);
    CHECK(half.desc().format == rhi::Format::RGBA16Float);
    REQUIRE(eight.mipCount() == half.mipCount());
    auto agree = gpu::ComputeKernel::create(*gpu->library, "athenea/test/material_textures", "materialLevelsAgree");
    REQUIRE(agree);
    for (const uint32_t mip : {0u, eight.mipCount() - 1}) {
        // Level 0 is the file through two decodes: within 1/255 in light
        // (measured 4.9e-4). A mip is light averaged and then stored again,
        // in 8-bit sRGB on the fast route: half a code there is up to
        // 0.5/255 * 2.4 in light near white (measured 4.9e-3 at 1x1).
        const float tolerance = mip == 0 ? 1.0F / 255.0F : 2.0F / 255.0F;
        // The 8-bit texture as materials see it: through an sRGB view.
        rhi::TextureViewDesc srgbDesc;
        srgbDesc.format = rhi::Format::RGBA8UnormSrgb;
        srgbDesc.subresourceRange.layer = 0;
        srgbDesc.subresourceRange.layerCount = 1;
        srgbDesc.subresourceRange.mip = mip;
        srgbDesc.subresourceRange.mipCount = 1;
        rhi::ComPtr<rhi::ITextureView> decoded;
        REQUIRE(SLANG_SUCCEEDED(gpu->device->rhi()->createTextureView(eight.rhi(), srgbDesc, decoded.writeRef())));
        auto light = half.view(mip);
        REQUIRE(light);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "counts");
        {
            gpu::CommandBatch batch(*gpu->device);
            agree->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["level"].setBinding(decoded.get());
                cursor["other"].setBinding((*light).get());
                cursor["counts"].setBinding(counts.rhi());
                cursor["params"]["width"].setData(eight.width(mip));
                cursor["params"]["height"].setData(eight.height(mip));
                cursor["params"]["tolerance"].setData(tolerance);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[3] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        float worst = 0.0F;
        std::memcpy(&worst, &c[1], sizeof(worst));
        std::printf("  level %u: %u texels, worst %.2e in light, %u beyond %.0f/255\n", mip, c[0], double(worst), c[2],
                    double(tolerance * 255.0F));
        CHECK(c[0] == eight.width(mip) * eight.height(mip));
        CHECK(c[2] == 0);
        CHECK(worst < tolerance);
    }
}

// Not a check: how long a commit takes, by route. 2048 x 2048 files, eight at
// a time; the first of each route pays for the config and the compile.
TEST_CASE("texture commit time", "[.timing]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const uint32_t w = 2048, h = 2048, n = 8;
    std::vector<fs::path> eight, sixteen;
    for (uint32_t k = 0; k < n; ++k) {
        eight.push_back(scratch("timing8_" + std::to_string(k) + ".png"));
        writePattern(*gpu, eight.back(), w, h);
        // The same bytes, as half floats in an EXR: a file of another depth
        // for the timing, not a picture.
        sixteen.push_back(scratch("timing16_" + std::to_string(k) + ".exr"));
        HioImageSharedPtr in = HioImage::OpenForReading(eight.back().string());
        REQUIRE(in);
        std::vector<uint8_t> bytes(size_t{w} * h * 4);
        HioImage::StorageSpec spec;
        spec.width = int(w);
        spec.height = int(h);
        spec.depth = 1;
        spec.format = in->GetFormat();
        spec.flipped = false;
        spec.data = bytes.data();
        REQUIRE(in->Read(spec));
        HioImageSharedPtr out = HioImage::OpenForWriting(sixteen.back().string());
        spec.width = int(w / 2);
        spec.format = HioFormatFloat16Vec4;
        REQUIRE(out->Write(spec));
    }
    const auto time = [&](const char* label, const std::vector<fs::path>& files, const char* space, bool fastPath) {
        auto store = material::TextureStore::create(*gpu->library);
        REQUIRE(store);
        (*store)->setSrgbFastPath(fastPath);
        for (const fs::path& f : files) (void)(*store)->request(f.string(), space);
        const auto t0 = std::chrono::steady_clock::now();
        auto loaded = (*store)->commit();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        REQUIRE(loaded);
        std::printf("  %s: %zu files in %.1f ms (%.1f ms each)\n", label, *loaded, ms, ms / double(*loaded));
    };
    for (int round = 0; round < 3; ++round) {
        time("8-bit srgb (fast path)", eight, "srgb_texture", true);
        time("8-bit raw", eight, "raw", true);
        time("16-bit srgb", sixteen, "srgb_texture", true);
        time("8-bit srgb (compiled function)", eight, "srgb_texture", false);
        time("8-bit acescg", eight, "acescg", true);
    }
}
