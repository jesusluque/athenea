// Copyright (c) 2026 jesus luque.
//
// A SHADOW CATCHER (athenea mesh2splat --shadow-catcher; docs/decisions.md,
// task PLAY-G) DRAWN THROUGH HYDRA: a lattice of black gaussians over a plain
// backdrop, each covering what its open cells let through of the frame's
// light. The cells are written here, not baked: a cone about straight up is
// closed, so the share the catcher leaves is known in closed form -- under a
// plain dome 1 - sin^2 of the cone, under a light straight above nothing, and
// under both their irradiances weighed. The frames are measured on the device
// (`imageStats` over a window); nothing else is read back.
#include "../gpu/GpuTest.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include <pxr/base/plug/registry.h>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

[[maybe_unused]] const bool kPluginsRegistered = [] {
    pxr::PlugRegistry::GetInstance().RegisterPlugins(fs::path(ATHENEA_HYDRA_PLUGIN_DIR).string());
    return true;
}();

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "catcher";
    fs::create_directories(dir);
    return dir / name;
}

constexpr uint32_t kWidth = 160;
constexpr uint32_t kHeight = 120;
/// The lattice's cell, and the catcher's sigma: what catcherAlphaFor's
/// composited lattice assumes.
constexpr float kCell = 0.04F;
constexpr int kAcross = 110;   // 4.4 x 3.4 units: wider than the view at z = 0
constexpr int kDown = 85;
constexpr uint32_t kSide = 32;   // the TX grid a shadow catcher is baked on
constexpr uint32_t kWords = kSide * kSide / 32;

/// The octahedral cell's direction, as shaders/athenea/common/octahedral.slang
/// decodes it: an input written to match, not a result.
void cellDirection(uint32_t cell, double& x, double& y, double& z) {
    const double u = (static_cast<double>(cell % kSide) + 0.5) / kSide * 2.0 - 1.0;
    const double v = (static_cast<double>(cell / kSide) + 0.5) / kSide * 2.0 - 1.0;
    x = u;
    y = v;
    z = 1.0 - std::abs(u) - std::abs(v);
    const double t = std::max(-z, 0.0);
    x += x >= 0.0 ? -t : t;
    y += y >= 0.0 ? -t : t;
    const double length = std::sqrt(x * x + y * y + z * z);
    x /= length;
    y /= length;
    z /= length;
}

/// The catcher: a lattice on z = 0 facing +z, every gaussian's cells open but
/// those within `closedDegrees` of +z (none at 0).
void writeCatcher(gpu::ShaderLibrary& library, const fs::path& path, double closedDegrees) {
    io::RawSplats raw;
    raw.source = "catcher synthetic";
    raw.linear = true;
    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 14;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3; e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10; e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14; e.restPerColour = 0;
    std::vector<int32_t> bits;
    std::vector<uint32_t> pattern(kWords, 0u);
    const double closedCos = std::cos(closedDegrees * std::numbers::pi / 180.0);
    for (uint32_t cell = 0; cell < kSide * kSide; ++cell) {
        double x = 0.0, y = 0.0, z = 0.0;
        cellDirection(cell, x, y, z);
        const bool closed = closedDegrees > 0.0 && z > closedCos;
        if (!closed) {
            pattern[cell >> 5u] |= 1u << (cell & 31u);
        }
    }
    for (int j = 0; j < kDown; ++j) {
        for (int i = 0; i < kAcross; ++i) {
            raw.records.resize(raw.records.size() + 14, 0.0F);
            float* r = raw.records.data() + raw.records.size() - 14;
            r[0] = (static_cast<float>(i) - 0.5F * static_cast<float>(kAcross - 1)) * kCell;
            r[1] = (static_cast<float>(j) - 0.5F * static_cast<float>(kDown - 1)) * kCell;
            r[2] = 0.0F;
            r[3] = 8.0F;   // a logit: opacity 0.9997
            r[4] = std::log(kCell);
            r[5] = std::log(kCell);
            r[6] = std::log(kCell * 0.01F);
            r[7] = 1.0F;
            raw.count += 1;
            for (uint32_t w = 0; w < kWords; ++w) {
                bits.push_back(static_cast<int32_t>(pattern[w]));
            }
        }
    }
    // A transfer's direct half, which a catcher carries and does not read.
    const std::vector<float> direct(static_cast<size_t>(raw.count) * 16, 0.0F);
    fs::remove(path);
    usd::ExportOptions options;
    options.addCamera = false;
    options.linear = true;
    options.upAxis = 'y';
    options.relight = true;
    options.catcher = true;
    options.transferDirect = direct;
    options.transferCoefficients = 16;
    options.shadowBits = bits;
    options.shadowWords = kWords;
    REQUIRE(usd::writeParticleFieldStage(library, raw, path, options));
}

/// A shot: an opaque grey backdrop of plain gaussians at z = -0.3 (no light
/// changes it), the catcher at `catcher` over it if given, the lights, and a
/// camera four units up looking down -z.
void writeShot(const fs::path& path, const fs::path& catcher, const std::string& lights) {
    std::ostringstream out;
    out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
           "def Xform \"World\"\n{\n";
    const int across = 60, down = 46;
    const int count = across * down;
    out << "    def ParticleField3DGaussianSplat \"Backdrop\"\n    {\n        point3f[] positions = [";
    for (int k = 0; k < count; ++k) {
        out << (k ? ", " : "") << "(" << (k % across - 0.5 * (across - 1)) * 0.08 << ", "
            << (k / across - 0.5 * (down - 1)) * 0.08 << ", -0.3)";
    }
    out << "]\n        quatf[] orientations = [";
    for (int k = 0; k < count; ++k) out << (k ? ", " : "") << "(1, 0, 0, 0)";
    out << "]\n        float3[] scales = [";
    for (int k = 0; k < count; ++k) out << (k ? ", " : "") << "(0.08, 0.08, 0.001)";
    out << "]\n        float[] opacities = [";
    for (int k = 0; k < count; ++k) out << (k ? ", " : "") << "1";
    out << "]\n        int radiance:sphericalHarmonicsDegree = 0\n"
           "        float3[] radiance:sphericalHarmonicsCoefficients = [";
    for (int k = 0; k < count; ++k) out << (k ? ", " : "") << "(0.5, 0.5, 0.5)";
    out << "]\n    }\n";
    if (!catcher.empty()) {
        out << "    def \"Catcher\" ( references = @" << catcher.string() << "@</World/Splats> )\n    {\n    }\n";
    }
    out << lights;
    out << "    def Camera \"Shot\"\n    {\n"
           "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
           "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
           "        double3 xformOp:translate = (0, 0, 4)\n"
           "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
    std::ofstream file(path);
    file << out.str();
}

/// A white dome of radiance one: an irradiance of pi on the lattice.
const std::string kDome = "    def DomeLight \"Sky\"\n    {\n        float inputs:intensity = 1\n    }\n";
/// A parallel light straight down -z, an irradiance of pi facing it.
const std::string kSun = "    def DistantLight \"Sun\"\n    {\n        float inputs:intensity = 3.14159265\n"
                         "        float inputs:angle = 0\n    }\n";

/// A frame of the shot, on the device.
gpu::Buffer frameOf(test::Gpu& gpu, const fs::path& stage) {
    auto renderer = usd::StageRenderer::open(stage);
    if (!renderer) FAIL(renderer.error().toString());
    auto image = (*renderer)->render("/World/Shot", 0.0, kWidth, kHeight, "raster");
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    desc.label = "catcher.frame";
    auto frame = gpu::Buffer::create(*gpu.device, desc, image->rgba.data());
    REQUIRE(frame);
    return std::move(*frame);
}

/// The mean of a frame's red over its middle, on the device.
double middle(test::Gpu& gpu, const gpu::Buffer& frame) {
    auto stats = render::imageStats(*gpu.library, frame, kWidth, kHeight, 40, 30, 120, 90);
    REQUIRE(stats);
    return stats->mean[0];
}

/// The frames with the catcher and without it.
struct Pair {
    gpu::Buffer with;
    gpu::Buffer without;
};

Pair shots(test::Gpu& gpu, const std::string& name, double closedDegrees, const std::string& lights) {
    const fs::path catcher = scratch(name + "_catcher.usdc");
    writeCatcher(*gpu.library, catcher, closedDegrees);
    const fs::path with = scratch(name + "_with.usda");
    const fs::path without = scratch(name + "_without.usda");
    writeShot(with, catcher, lights);
    writeShot(without, {}, lights);
    return {frameOf(gpu, with), frameOf(gpu, without)};
}

/// What the catcher leaves of its backdrop: the middle with it over the
/// middle without.
double leaves(test::Gpu& gpu, const std::string& name, double closedDegrees, const std::string& lights) {
    const Pair pair = shots(gpu, name, closedDegrees, lights);
    const double under = middle(gpu, pair.without);
    REQUIRE(under > 0.05);
    return middle(gpu, pair.with) / under;
}

/// k a, on the device (light_layers_check).
gpu::Buffer scaled(test::Gpu& gpu, const gpu::Buffer& a, float k) {
    static gpu::ComputeKernel kCombine = test::kernel(gpu, "athenea/test/light_layers_check");
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{kWidth} * kHeight * 16;
    desc.elementBytes = 16;
    desc.label = "catcher.scaled";
    auto out = gpu::Buffer::create(*gpu.device, desc);
    REQUIRE(out);
    gpu::CommandBatch batch(*gpu.device);
    kCombine.dispatch(batch, {kWidth * kHeight, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["a"].setBinding(a.rhi());
        cursor["b"].setBinding(a.rhi());
        cursor["c"].setBinding(a.rhi());
        cursor["combined"].setBinding(out->rhi());
        cursor["params"]["pixels"].setData(kWidth * kHeight);
        cursor["params"]["ka"].setData(k);
        cursor["params"]["kb"].setData(0.0F);
        cursor["params"]["kc"].setData(0.0F);
    });
    REQUIRE(batch.submit(true));
    return std::move(*out);
}

/// The cosine-weighted share of the hemisphere a cone about the normal holds.
double coneShare(double degrees) {
    const double s = std::sin(degrees * std::numbers::pi / 180.0);
    return s * s;
}

}   // namespace

TEST_CASE("a shadow catcher under a plain dome leaves what its open cells let through",
          "[usd][gpu][splat][catcher]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const double open = leaves(*gpu, "dome_open", 0.0, kDome);
    INFO("every cell open: " << open);
    CHECK(open > 0.99);
    CHECK(open < 1.01);
    const double closed = leaves(*gpu, "dome_cone", 25.0, kDome);
    const double expected = 1.0 - coneShare(25.0);
    INFO("a 25 degree cone closed: " << closed << " against " << expected);
    CHECK(std::abs(closed - expected) < 0.035);
}

TEST_CASE("a shadow catcher takes a distant light through its bits, with or without a dome",
          "[usd][gpu][splat][catcher]") {
    ATHENEA_REQUIRE_GPU(gpu);
    // The light straight above, its cells closed: nothing reaches the ground.
    const double shadowed = leaves(*gpu, "sun_cone", 25.0, kSun);
    INFO("under the light alone, its cone closed: " << shadowed);
    CHECK(shadowed < 0.05);
    // Every cell open: the catcher is not there.
    const double open = leaves(*gpu, "sun_open", 0.0, kSun);
    INFO("under the light alone, every cell open: " << open);
    CHECK(open > 0.99);
    // Both, the light's irradiance the dome's: the dome's open share of
    // half the light.
    const double both = leaves(*gpu, "both_cone", 25.0, kDome + kSun);
    const double expected = 0.5 * (1.0 - coneShare(25.0));
    INFO("under the dome and the light, the cone closed: " << both << " against " << expected);
    CHECK(std::abs(both - expected) < 0.035);
}

// A FAINT SHADOW IS DRAWN SMOOTH. A 15 degree cone leaves 0.93 of the dome,
// an opacity of about 0.015 a gaussian: drawn whole (two pixels a sigma),
// none of them is near what the blend drops, so none is thinned to a lot --
// which drew them as a grain of three-per-cent dots. The frame with the
// catcher is the frame without it scaled by what it leaves, pixel by pixel.
TEST_CASE("a faint shadow catcher whose gaussians are drawn whole darkens smoothly, not as grain",
          "[usd][gpu][splat][catcher]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const Pair pair = shots(*gpu, "faint_cone", 15.0, kDome);
    const double under = middle(*gpu, pair.without);
    REQUIRE(under > 0.05);
    const double ratio = middle(*gpu, pair.with) / under;
    const double expected = 1.0 - coneShare(15.0);
    INFO("a 15 degree cone closed: " << ratio << " against " << expected);
    CHECK(std::abs(ratio - expected) < 0.02);
    const gpu::Buffer even = scaled(*gpu, pair.without, static_cast<float>(ratio));
    auto diff = render::compareImages(*gpu->library, pair.with, even, kWidth, kHeight);
    REQUIRE(diff);
    INFO("against the backdrop scaled by it: p99 " << diff->p99 << ", max " << diff->max << ", over 2 "
                                                  << diff->over2);
    CHECK(diff->p99 <= 2);
}
