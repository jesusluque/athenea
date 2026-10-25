// Copyright (c) 2026 jesus luque.
//
// THE BRIDGE TO ANOTHER PROCESS: handles carried by token, cameras with
// their instants, fences two processes share, frames read where they lie --
// and, end to end, a second process (athenea-mcp, started here) fed pages
// from shared memory and from this process's device, cameras with PTP
// instants, and read back colour and depth without a copy, checked by
// kernels against a direct draw of the epoch each frame says it showed, from
// the camera it says it was drawn from.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "athenea/bridge/CameraQueue.h"
#include "athenea/bridge/ClientPages.h"
#include "athenea/bridge/Frames.h"
#include "athenea/core/Platform.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/SharedFence.h"
#include "athenea/mcp/Server.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/StageRenderer.h"

#if defined(ATHENEA_TEST_SPZ)
#include "load-spz.h"
#include "athenea/io/Readers.h"
#endif

using namespace athenea;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

json call(mcp::Server& server, const std::string& tool, const json& arguments) {
    const auto reply = server.handle(json{{"jsonrpc", "2.0"},
                                          {"id", 3},
                                          {"method", "tools/call"},
                                          {"params", {{"name", tool}, {"arguments", arguments}}}}
                                         .dump());
    REQUIRE(reply.has_value());
    return json::parse(*reply);
}

std::string textOf(const json& answer) {
    std::string text;
    for (const json& block : answer["result"]["content"]) {
        text += block.value("text", std::string{});
    }
    return text;
}

bool isError(const json& answer) {
    return answer["result"].value("isError", false);
}

uint64_t systemNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
}

render::Camera cameraOf(const bridge::RingRecord& r) {
    render::Camera c;
    std::copy(std::begin(r.cameraToWorld), std::end(r.cameraToWorld), c.cameraToWorld.m.begin());
    c.lens.focal = r.focal;
    c.lens.haperture = r.haperture;
    c.lens.vaperture = r.vaperture;
    c.lens.apertureOffset[0] = r.windowX;
    c.lens.apertureOffset[1] = r.windowY;
    c.lens.nearZ = r.nearZ;
    c.lens.farZ = r.farZ;
    return c;
}

}   // namespace

// --- the system's side ---------------------------------------------------------------

TEST_CASE("handles another process fetches by a token, over Mach and over a socket", "[bridge][platform]") {
    for (const platform::HandleTransport transport : {platform::HandleTransport::Native, platform::HandleTransport::Socket}) {
        auto server = platform::HandleServer::create("athenea.test", transport);
        if (!server) FAIL(server.error().toString());
        INFO("server " << (*server)->name());
        auto descriptor = platform::duplicateOsHandle({platform::OsHandle::Kind::Fd, 2});
        REQUIRE(descriptor);
        (*server)->offer(42, {*descriptor});
        for (int fetch = 0; fetch < 2; ++fetch) {   // copies: an offer serves every fetch
            auto got = platform::fetchHandles((*server)->name(), 42);
            if (!got) FAIL(got.error().toString());
            REQUIRE(got->size() == 1);
            CHECK((*got)[0].kind == platform::OsHandle::Kind::Fd);
            for (const auto& h : *got) platform::closeOsHandle(h);
        }
        auto unknown = platform::fetchHandles((*server)->name(), 43);
        REQUIRE(!unknown);
        CHECK(unknown.error().code() == ErrorCode::NotFound);
        (*server)->withdraw(42);
        CHECK(!platform::fetchHandles((*server)->name(), 42));
        CHECK((*server)->served() == 2);
        CHECK((*server)->refused() == 2);
    }
    CHECK(!platform::fetchHandles("athenea.none.1.2", 1, std::chrono::milliseconds(200)));
    CHECK(!platform::fetchHandles("unix:/tmp/athenea-none.sock", 1, std::chrono::milliseconds(200)));
}

// --- cameras ---------------------------------------------------------------------------

TEST_CASE("a camera at an instant is the pose between the two reports that bracket it", "[bridge][camera]") {
    bridge::CameraQueue queue;
    render::Camera a = render::Camera::lookingAt({0.0, 0.0, 5.0}, {0.0, 0.0, 0.0});
    a.lens.focal = 30.0;
    render::Camera b = a;
    b.cameraToWorld = aofx::xform::translation({2.0, 0.0, 5.0}) * aofx::xform::rotationY(90.0);
    b.lens.focal = 40.0;
    REQUIRE(queue.push({1000, a}));
    REQUIRE(queue.push({2000, b}));
    auto before = queue.at(500);
    REQUIRE(before);
    REQUIRE(before->has_value());
    CHECK((*before)->samples == 1);
    CHECK((*before)->camera.lens.focal == 30.0);
    auto mid = queue.at(1500);
    REQUIRE(mid);
    REQUIRE(mid->has_value());
    CHECK((*mid)->samples == 2);
    CHECK(std::abs((*mid)->camera.lens.focal - 35.0) < 1e-12);
    const render::Mat4 expected = aofx::xform::translation({1.0, 0.0, 5.0}) * aofx::xform::rotationY(45.0);
    double worst = 0.0;
    for (size_t k = 0; k < 16; ++k) worst = std::max(worst, std::abs((*mid)->camera.cameraToWorld.m[k] - expected.m[k]));
    CHECK(worst < 1e-9);
    auto after = queue.at(2500);
    REQUIRE(after);
    CHECK((*after)->samples == 1);
    CHECK((*after)->camera.lens.focal == 40.0);
    CHECK(queue.size() == 1);   // the report before the instant is all that is kept

    // Refused, counted, never queued.
    render::Camera scaled = a;
    scaled.cameraToWorld = aofx::xform::scaling({2.0, 2.0, 2.0});
    render::Camera clipped = a;
    clipped.lens.farZ = clipped.lens.nearZ * 0.5;
    render::Camera nan = a;
    nan.cameraToWorld.at(0, 3) = std::nan("");
    render::Camera mirrored = a;
    mirrored.cameraToWorld = aofx::xform::scaling({-1.0, 1.0, 1.0});
    for (const render::Camera& bad : {scaled, clipped, nan, mirrored}) {
        CHECK(!queue.push({3000, bad}));
    }
    CHECK(queue.stats().refused == 4);
    CHECK(queue.size() == 1);

    // The frame thread is the first to ask; any other is refused.
    Result<std::optional<bridge::CameraAt>> fromElsewhere = Error(ErrorCode::Unknown, "not run");
    std::thread other([&] { fromElsewhere = queue.at(2600); });
    other.join();
    REQUIRE(!fromElsewhere);
    CHECK(fromElsewhere.error().toString().find("off the frame thread") != std::string::npos);
}

TEST_CASE("camera_set refuses what makes no frustum and counts it, and queues the rest", "[bridge][camera][mcp]") {
    mcp::Server server;
    const std::vector<double> pose{1, 0, 0, 0.5, 0, 1, 0, 0, 0, 0, 1, 4, 0, 0, 0, 1};
    const std::vector<json> good{
        {{"ptpTime", 100}, {"pose", pose}, {"focal", 30}},
        {{"ptpTime", 200}, {"position", {0, 0, 4}}, {"rotation", {0, 0, 0, 1}}},
        {{"ptpTime", 300}, {"pose", pose}, {"fx", 500}, {"fy", 500}, {"cx", 320}, {"cy", 180},
         {"resolution", {640, 360}}},
    };
    const std::vector<json> bad{
        {{"pose", pose}},                                               // no instant
        {{"ptpTime", -1}, {"pose", pose}},                              // an instant below 0
        {{"ptpTime", 1}, {"pose", std::vector<double>(15, 0.0)}},       // fifteen numbers
        {{"ptpTime", 1}, {"pose", {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1}}},   // scaled
        {{"ptpTime", 1}, {"position", {0, 0, 4}}, {"rotation", {0, 0, 0, 2}}},          // not unit
        {{"ptpTime", 1}, {"position", {0, nullptr, 4}}, {"rotation", {0, 0, 0, 1}}},    // NaN as JSON writes one
        {{"ptpTime", 1}, {"pose", pose}, {"fx", 500}, {"fy", 400}, {"cx", 0}, {"cy", 0}, {"resolution", {64, 64}}},
        {{"ptpTime", 1}, {"pose", pose}, {"focal", -3}},
        {{"ptpTime", 1}, {"pose", pose}, {"near", 10}, {"far", 1}},
        {{"ptpTime", 1}},                                               // no pose
    };
    for (const json& args : good) {
        const json answer = call(server, "camera_set", args);
        INFO(textOf(answer));
        CHECK(!isError(answer));
    }
    json last;
    for (const json& args : bad) {
        last = call(server, "camera_set", args);
        INFO(args.dump() << " -> " << textOf(last));
        CHECK(isError(last));
        CHECK(textOf(last).find("not queued") != std::string::npos);
    }
    const json counts = json::parse(textOf(call(server, "camera_set", good[0])));
    CHECK(counts["queued"] == 3);
    CHECK(counts["refused"] == bad.size());
    CHECK(counts["replaced"] == 1);
}

// --- fences and frames on one device ---------------------------------------------------

TEST_CASE("a fence made again from its port is raised and waited on by the device", "[bridge][gpu][fence]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (gpu->device->backend() != gpu::Backend::Metal) SKIP("shared events are Metal's here");
    gpu::Device& device = *gpu->device;
    auto made = gpu::SharedFence::create(device, 0);
    if (!made) FAIL(made.error().toString());
    auto port = made->exportHandle();
    if (!port) FAIL(port.error().toString());
    auto imported = gpu::SharedFence::import(device, *port);
    platform::closeOsHandle(*port);
    if (!imported) FAIL(imported.error().toString());
    const auto waitFor = [](const gpu::SharedFence& f, uint64_t v) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (f.value() < v && std::chrono::steady_clock::now() < until) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return f.value() >= v;
    };
    // The imported side raises it on the device: the maker sees it.
    {
        gpu::CommandBatch batch(device);
        REQUIRE(imported->signalAfter(batch, 5));
        CHECK(waitFor(*made, 5));
        CHECK(imported->value() == 5);
    }
    // The imported side waits on the device for the maker's host.
    auto done = gpu::SharedFence::create(device, 0);
    REQUIRE(done);
    {
        gpu::CommandBatch batch(device);
        REQUIRE(imported->waitBefore(batch, 9));
        REQUIRE(done->signalAfter(batch, 1));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(done->value() == 0);   // held on the device
        made->setValue(9);
        CHECK(waitFor(*done, 1));
    }
    // And the other way: the maker waits for what the imported side raises.
    {
        gpu::CommandBatch batch(device);
        REQUIRE(made->waitBefore(batch, 12));
        REQUIRE(done->signalAfter(batch, 2));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(done->value() == 1);
        imported->setValue(12);
        CHECK(waitFor(*done, 2));
    }
}

TEST_CASE("a frame published into a slot is read where it lies, and only the frame thread publishes",
          "[bridge][gpu][frames]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (gpu->device->backend() != gpu::Backend::Metal) SKIP("slots over shared pages are Metal's here");
    gpu::Device& device = *gpu->device;
    constexpr uint32_t w = 160, h = 120;
    test::CloudBuilder built = test::randomCloud(1500, 7);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto cloud = loader->upload(built.raw);
    REQUIRE(cloud);
    auto raster = render::TileRasterizer::create(*gpu->library);
    REQUIRE(raster);
    render::RenderSettings settings;
    settings.width = w;
    settings.height = h;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 6.0}, {0.0, 0.0, 0.0});
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::RenderTargets frames[2];
    REQUIRE(raster->render(render::projectionFor(camera, w, h), instances, settings, frames[0]));
    camera.cameraToWorld.at(0, 3) += 0.4;
    REQUIRE(raster->render(render::projectionFor(camera, w, h), instances, settings, frames[1]));

    auto server = platform::HandleServer::create("athenea.test");
    REQUIRE(server);
    for (const bool fence : {true, false}) {
        INFO((fence ? "fence" : "poll"));
        bridge::PublisherDesc desc;
        desc.width = w;
        desc.height = h;
        desc.slots = 2;
        desc.fenceSync = fence;
        desc.aovs = {{"color", bridge::Format::RGBA32F}, {"depth", bridge::Format::R32F}};
        auto publisher = bridge::FramePublisher::create(device, *gpu->library, **server, desc);
        if (!publisher) FAIL(publisher.error().toString());
        auto info = bridge::SubscriptionInfo::fromJson((*publisher)->info().toJson());
        REQUIRE(info);
        auto subscriber = bridge::FrameSubscriber::open(device, *info);
        if (!subscriber) FAIL(subscriber.error().toString());
        const auto sources = [&](const render::RenderTargets& t) {
            return std::vector<bridge::AovSource>{{&t.colour, 4, 0, nullptr}, {&t.depth, 1, 0, nullptr}};
        };
        for (uint64_t frame = 1; frame <= 4; ++frame) {
            bridge::RingRecord record;
            record.epoch = frame * 10;
            const auto s = sources(frames[frame % 2]);
            auto published = (*publisher)->publish(frame, s, record);
            REQUIRE(published);
            if (!fence && frame > 2 && !*published) {
                FAIL("a released slot was not reused");
            }
            // The engine's host brings the ring's `completed` word up as it
            // sees the frames fence rise (poll mode reads it).
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while ((*subscriber)->completed() < frame && std::chrono::steady_clock::now() < until) {
                (*publisher)->observe();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            REQUIRE((*subscriber)->completed() >= frame);
            const auto kept = (*subscriber)->record(frame);
            REQUIRE(kept.has_value());
            CHECK(kept->epoch == frame * 10);
            CHECK((*subscriber)->state(static_cast<uint32_t>(kept->slot)).written == 1);
            const uint32_t slot = static_cast<uint32_t>(kept->slot);
            auto diff = render::compareImages(*gpu->library, (*subscriber)->aov(slot, 0), frames[frame % 2].colour, w, h);
            REQUIRE(diff);
            CHECK(diff->max == 0);
            auto depth = render::countDifferent(*gpu->library, (*subscriber)->aov(slot, 1), frames[frame % 2].depth, w * h);
            REQUIRE(depth);
            CHECK(*depth == 0);
            gpu::CommandBatch reads(device);
            REQUIRE((*subscriber)->release(reads, frame));
        }
        if (!fence) {
            // A subscriber that has not released frame 4's slot: frame 6 is
            // skipped, not waited for.
            bridge::RingRecord record;
            REQUIRE((*publisher)->publish(5, sources(frames[0]), record));
            auto skipped = (*publisher)->publish(6, sources(frames[0]), record);
            REQUIRE(skipped);
            CHECK(!*skipped == false);   // frame 4 was released, so 6 may take its slot
            auto held = (*publisher)->publish(7, sources(frames[0]), record);
            REQUIRE(held);
            CHECK(!*held);               // frame 5 is not released: 7 is skipped
            CHECK((*publisher)->stats().skipped == 1);
        }
        // Only the thread that made it publishes.
        Result<bool> fromElsewhere = Error(ErrorCode::Unknown, "not run");
        std::thread other([&] {
            bridge::RingRecord record;
            fromElsewhere = (*publisher)->publish(100, sources(frames[0]), record);
        });
        other.join();
        REQUIRE(!fromElsewhere);
        CHECK(fromElsewhere.error().toString().find("off the frame thread") != std::string::npos);
        device.waitIdle();
    }
}

// A principal point off the centre, as camera_set's cx and cy make it: the
// lens's aperture offsets are UsdGeomCamera's own -- a camera of the caller's
// draws what a stage camera with those offsets draws, through Hydra and
// directly (projectionFor).
TEST_CASE("an aperture offset is the one a stage camera authors", "[bridge][gpu][camera]") {
    ATHENEA_REQUIRE_GPU(gpu);
    constexpr uint32_t w = 200, h = 150;
    const fs::path dir = fs::temp_directory_path() / "athenea-bridge-tests";
    fs::create_directories(dir);
    test::CloudBuilder built = test::randomCloud(1200, 31, 0.02F, 0.3F);
    usd::ExportOptions options;
    options.addCamera = false;
    const fs::path cloudStage = dir / "offset_cloud.usda";
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, built.raw, cloudStage, options));
    const fs::path stage = dir / "offset.usda";
    {
        std::ofstream out(stage);
        out << "#usda 1.0\n(\n    subLayers = [@" << cloudStage.string() << "@]\n)\n"
               "def Camera \"Shot\"\n{\n"
               "    float2 clippingRange = (0.1, 1000)\n    float focalLength = 30\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float horizontalApertureOffset = 2\n    float verticalApertureOffset = -1.5\n"
               "    double3 xformOp:translate = (0.3, 0.2, 6)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    }
    auto renderer = usd::StageRenderer::open(stage);
    if (!renderer) FAIL(renderer.error().toString());
    render::Camera camera;
    camera.cameraToWorld = aofx::xform::translation({0.3, 0.2, 6.0});
    camera.lens.focal = 30.0;
    camera.lens.haperture = 24.576;
    camera.lens.vaperture = 18.432;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    camera.lens.apertureOffset[0] = 2.0;
    camera.lens.apertureOffset[1] = -1.5;
    const auto upload = [&](const std::vector<float>& rgba) {
        gpu::BufferDesc desc;
        desc.bytes = rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, rgba.data());
        REQUIRE(made);
        return *made;
    };
    auto byPrim = (*renderer)->render("/Shot", 0.0, w, h);
    if (!byPrim) FAIL(byPrim.error().toString());
    auto byLens = (*renderer)->render(camera, 0.0, w, h);
    if (!byLens) FAIL(byLens.error().toString());
    render::Camera centred = camera;
    centred.lens.apertureOffset[0] = centred.lens.apertureOffset[1] = 0.0;
    auto byCentre = (*renderer)->render(centred, 0.0, w, h);
    REQUIRE(byCentre);
    const gpu::Buffer prim = upload(byPrim->rgba);
    const gpu::Buffer lens = upload(byLens->rgba);
    const gpu::Buffer centre = upload(byCentre->rgba);
    auto same = render::compareImages(*gpu->library, lens, prim, w, h);
    REQUIRE(same);
    auto moved = render::compareImages(*gpu->library, centre, prim, w, h);
    REQUIRE(moved);
    // Directly: the rasteriser through projectionFor.
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto cloud = loader->upload(built.raw);
    REQUIRE(cloud);
    auto raster = render::TileRasterizer::create(*gpu->library);
    REQUIRE(raster);
    render::RenderSettings settings;
    settings.width = w;
    settings.height = h;
    render::RenderTargets direct;
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    REQUIRE(raster->render(render::projectionFor(camera, w, h), instances, settings, direct));
    auto directDiff = render::compareImages(*gpu->library, direct.colour, prim, w, h);
    REQUIRE(directDiff);
    std::printf("  aperture offset: our lens against the stage camera p99 %u max %u; centred against it p99 %u max "
                "%u; directly p99 %u max %u\n",
                same->p99, same->max, moved->p99, moved->max, directDiff->p99, directDiff->max);
    CHECK(same->p99 == 0);   // GfCamera's float offsets against the prim's: a code value at most
    CHECK(same->max <= 1);
    CHECK(moved->max > 16);   // the offset moves the picture
    CHECK(directDiff->p99 <= 1);
    CHECK(directDiff->max <= 2);
}

#if defined(ATHENEA_TEST_SPZ)

namespace {

std::vector<uint8_t> spzPage(uint32_t count, uint64_t seed) {
    test::CloudBuilder built = test::randomCloud(count, seed, 0.02F, 0.3F);
    const auto& r = built.raw;
    const uint32_t stride = r.encoding.floatsPerRecord;
    spz::GaussianCloud cloud;
    cloud.numPoints = static_cast<int32_t>(r.count);
    cloud.shDegree = 0;
    cloud.antialiased = true;
    for (uint32_t i = 0; i < r.count; ++i) {
        const float* f = r.records.data() + size_t{i} * stride;
        cloud.positions.insert(cloud.positions.end(), {f[0], f[1], f[2]});
        cloud.alphas.push_back(f[3]);
        cloud.scales.insert(cloud.scales.end(), {f[4], f[5], f[6]});
        cloud.rotations.insert(cloud.rotations.end(), {f[8], f[9], f[10], f[7]});
        cloud.colors.insert(cloud.colors.end(), {f[11], f[12], f[13]});
    }
    spz::PackOptions options;
    options.version = 4;
    options.from = spz::CoordinateSystem::RDF;
    std::vector<uint8_t> bytes;
    REQUIRE(spz::saveSpz(cloud, options, &bytes));
    return bytes;
}

/// The engine, as another process: athenea-mcp on pipes.
struct Engine {
    platform::ChildProcess child;
    int                    id = 1;

    json call(const std::string& tool, const json& arguments) {
        REQUIRE(child.writeLine(json{{"jsonrpc", "2.0"},
                                     {"id", ++id},
                                     {"method", "tools/call"},
                                     {"params", {{"name", tool}, {"arguments", arguments}}}}
                                    .dump()));
        auto line = child.readLine(std::chrono::seconds(120));
        if (!line) FAIL(tool << ": " << line.error().toString());
        return json::parse(*line);
    }
};

/// Gaussians written by this process's device into the pages it shares.
void writeRecords(gpu::ShaderLibrary& library, gpu::CommandBatch& batch, const gpu::Buffer& records, uint32_t first,
                  uint32_t count, uint32_t seed) {
    static auto kernel = gpu::ComputeKernel::create(library, "athenea/test/bridge_records", "bridgeTestRecords");
    REQUIRE(kernel);
    kernel->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["records"].setBinding(records.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["first"].setData(first);
        cursor["params"]["seed"].setData(seed);
        cursor["params"]["stride"].setData(uint32_t{14});
    });
}

struct DepthCheck {
    uint32_t coverage = 0, apart = 0, drewA = 0, drewB = 0;
    float    maxA = 0, maxB = 0, minA = 0, minB = 0;
};

DepthCheck checkDepth(gpu::Device& device, gpu::ShaderLibrary& library, const gpu::Buffer& a, const gpu::Buffer& b,
                      uint32_t pixels) {
    static auto kernel = gpu::ComputeKernel::create(library, "athenea/test/bridge_depth_check", "bridgeDepthCheck");
    REQUIRE(kernel);
    const uint32_t initial[8] = {0, 0, 0, 0, 0, 0, 0xFFFFFFFFu, 0xFFFFFFFFu};
    gpu::BufferDesc desc;
    desc.bytes = sizeof initial;
    desc.elementBytes = 4;
    desc.label = "test.depth.counts";
    auto counts = gpu::Buffer::create(device, desc, initial);
    REQUIRE(counts);
    gpu::CommandBatch batch(device);
    kernel->dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["a"].setBinding(a.rhi());
        cursor["b"].setBinding(b.rhi());
        cursor["counts"].setBinding(counts->rhi());
        cursor["params"]["pixels"].setData(pixels);
        cursor["params"]["relative"].setData(1e-3F);
    });
    REQUIRE(batch.submit(true));
    uint32_t got[8] = {};
    REQUIRE(counts->read(device, 0, sizeof got, got));
    DepthCheck out{got[0], got[1], got[2], got[3]};
    std::memcpy(&out.maxA, &got[4], 4);
    std::memcpy(&out.maxB, &got[5], 4);
    std::memcpy(&out.minA, &got[6], 4);
    std::memcpy(&out.minB, &got[7], 4);
    return out;
}

}   // namespace

TEST_CASE("another process feeds pages and cameras and reads its frames where they lie",
          "[bridge][gpu][live][process]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = *gpu->device;
    if (device.backend() != gpu::Backend::Metal) SKIP("the bridge is checked on Metal");
    const fs::path program = platform::executableDir() / "athenea-mcp";
    if (!fs::exists(program)) SKIP("no athenea-mcp beside the tests");
    constexpr uint32_t kChunks = 2;
    constexpr uint32_t kSplats = 600;
    constexpr uint32_t w = 200, h = 150;
    const std::string stream = "athenea-test-bridge-" + std::to_string(platform::processId());
    const fs::path dir = fs::temp_directory_path() / "athenea-bridge-tests";
    fs::create_directories(dir);
    const fs::path stage = dir / "bridge.usda";
    {
        std::ofstream out(stage);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def ParticleField3DGaussianSplat \"Live\" (\n"
               "        prepend apiSchemas = [\"AtheneaLiveSourceAPI\"]\n    )\n    {\n"
               "        string primvars:athenea:live:stream = \"" << stream << "\" ( interpolation = \"constant\" )\n"
               "        int primvars:athenea:live:chunks = " << kChunks << " ( interpolation = \"constant\" )\n"
               "        int primvars:athenea:live:chunkSplats = " << kSplats << " ( interpolation = \"constant\" )\n"
               "        int primvars:athenea:live:degree = 0 ( interpolation = \"constant\" )\n"
               "        token primvars:athenea:live:sync = \"clock\" ( interpolation = \"constant\" )\n"
               "        float3[] extent = [(-2, -2, -2), (2, 2, 2)]\n"
               "    }\n}\n";
    }
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    // What each epoch holds, chunk by chunk: an SPZ page from the ring, or
    // records this process's device writes (a seed), or nothing.
    struct Chunk {
        std::vector<uint8_t> spz;
        uint32_t             seed = 0;   ///< device records when not 0
        uint32_t             first = 0;  ///< where in the shared pages
        std::optional<scene::GpuSplats> cloud;
    };
    std::vector<std::array<Chunk, kChunks>> epochs(4);
    epochs[1][0].spz = spzPage(500, 11);
    epochs[1][1].spz = spzPage(560, 12);
    epochs[2][0].spz = spzPage(520, 21);
    epochs[2][1].seed = 22;   // the device page whose fence comes after its commit
    epochs[2][1].first = 0;
    epochs[3][1].seed = 32;   // epoch 3 removes chunk 0
    epochs[3][1].first = 1024;
    for (size_t e = 1; e < epochs.size(); ++e) {
        for (Chunk& c : epochs[e]) {
            if (c.spz.empty()) continue;
            auto raw = io::readSpz(std::span<const uint8_t>(c.spz), "page");
            REQUIRE(raw);
            auto cloud = loader->upload(*raw, 3);
            REQUIRE(cloud);
            c.cloud = std::move(*cloud);
        }
    }

    // THE ENGINE, a process of its own.
    auto child = platform::ChildProcess::spawn({program.string()});
    if (!child) FAIL(child.error().toString());
    Engine engine{std::move(*child)};
    REQUIRE(engine.child.writeLine(json{{"jsonrpc", "2.0"}, {"id", 1}, {"method", "initialize"},
                                        {"params", {{"protocolVersion", "2025-06-18"}}}}
                                       .dump()));
    REQUIRE(engine.child.readLine());
    {
        const json opened = engine.call("open_stage", {{"stage", stage.string()}});
        INFO(opened.dump());
        REQUIRE(!isError(opened));
    }
    const json streamAnswer = engine.call(
        "stream_open", {{"name", stream}, {"chunks", kChunks}, {"ringBytes", 1 << 20}, {"sync", "clock"},
                        {"devicePages", {{"records", 4096}, {"degree", 0}}}});
    INFO(streamAnswer.dump());
    REQUIRE(!isError(streamAnswer));
    const json opened = json::parse(textOf(streamAnswer));
    auto ring = platform::SharedMemory::open(opened["ring"].get<std::string>(), opened["bytes"].get<size_t>());
    if (!ring) FAIL(ring.error().toString());
    bridge::ClientPages::Info pagesInfo;
    {
        const json& p = opened["devicePages"];
        pagesInfo.memory = p["memory"].get<std::string>();
        pagesInfo.bytes = p["bytes"].get<uint64_t>();
        pagesInfo.records = p["records"].get<uint64_t>();
        pagesInfo.floatsPerRecord = p["floatsPerRecord"].get<uint32_t>();
        pagesInfo.degree = p["degree"].get<uint32_t>();
        pagesInfo.server = p["server"].get<std::string>();
        pagesInfo.token = std::stoull(p["token"].get<std::string>(), nullptr, 16);
    }
    auto pages = bridge::ClientPagesView::open(device, pagesInfo);
    if (!pages) FAIL(pages.error().toString());
    // The device-page chunks' references: the same records, written apart.
    for (size_t e = 1; e < epochs.size(); ++e) {
        for (Chunk& c : epochs[e]) {
            if (c.seed == 0) continue;
            gpu::BufferDesc desc;
            desc.bytes = uint64_t{kSplats} * 14 * 4;
            desc.elementBytes = 4;
            desc.label = "test.records";
            auto records = gpu::Buffer::create(device, desc);
            REQUIRE(records);
            gpu::CommandBatch batch(device);
            writeRecords(*gpu->library, batch, *records, 0, kSplats, c.seed);
            REQUIRE(batch.submit(true));
            auto cloud = loader->upload(*records, kSplats, bridge::clientRecordEncoding(0), "device page", 3);
            REQUIRE(cloud);
            cloud->antialiased = true;
            c.cloud = std::move(*cloud);
        }
    }

    // CAMERAS with their instants: one pose now, another 4 s on.
    const uint64_t t0 = systemNs();
    const uint64_t t1 = t0 + 4'000'000'000ULL;
    render::Camera cam0 = render::Camera::lookingAt({0.0, 0.2, 6.0}, {0.0, 0.0, 0.0});
    cam0.lens.focal = 30.0;
    cam0.lens.vaperture = cam0.lens.haperture * h / w;
    render::Camera cam1 = render::Camera::lookingAt({0.8, 0.0, 5.5}, {0.1, 0.0, 0.0});
    cam1.lens = cam0.lens;
    {
        std::vector<double> m(cam0.cameraToWorld.m.begin(), cam0.cameraToWorld.m.end());
        const json a = engine.call("camera_set", {{"ptpTime", t0}, {"pose", m}, {"focal", 30.0},
                                                  {"vaperture", cam0.lens.vaperture}});
        REQUIRE(!isError(a));
        const bridge::Quat q = bridge::quatFromMatrix(cam1.cameraToWorld);
        const render::Vec3 p = cam1.cameraToWorld.translation();
        // The second in pixels: fx from the same lens, the principal point centred.
        const double fx = 30.0 / cam0.lens.haperture * w;
        const json b = engine.call("camera_set", {{"ptpTime", t1}, {"position", {p.x, p.y, p.z}},
                                                  {"rotation", {q.x, q.y, q.z, q.w}}, {"fx", fx}, {"fy", fx},
                                                  {"cx", w * 0.5}, {"cy", h * 0.5}, {"resolution", {w, h}}});
        INFO(b.dump());
        REQUIRE(!isError(b));
        cam1.cameraToWorld = bridge::matrixFromQuat(q, p);   // as the engine holds it
    }

    // THE FRAMES.
    const json subscribed = engine.call("render_subscribe", {{"aovs", {"color", "depth"}}, {"width", w},
                                                             {"height", h}, {"rate", 60}, {"sync", "fence"}});
    INFO(subscribed.dump());
    REQUIRE(!isError(subscribed));
    auto info = bridge::SubscriptionInfo::fromJson(textOf(subscribed));
    REQUIRE(info);
    auto frames = bridge::FrameSubscriber::open(device, *info);
    if (!frames) FAIL(frames.error().toString());

    auto raster = render::TileRasterizer::create(*gpu->library);
    REQUIRE(raster);
    render::RenderSettings settings;
    settings.width = w;
    settings.height = h;
    uint64_t released = 0;
    const auto releaseUpTo = [&](uint64_t frame) {
        if (frame <= released) return;
        gpu::CommandBatch batch(device);
        REQUIRE(frames->get()->release(batch, frame));
        released = frame;
    };
    struct Seen {
        int checked = 0, mismatched = 0, depthApart = 0, interpolated = 0, short_ = 0;
        uint32_t worstP99 = 0, worstMax = 0;
        uint64_t worstOver2 = 0;
        double cameraError = 0.0;
    } seen;
    // Waits for frames showing `epoch`, checks `want` of them, and answers how
    // long after `pushedNs` the first was signalled.
    const auto check = [&](uint32_t epoch, int want, uint64_t pushedNs) -> double {
        double latencyMs = -1.0;
        int checked = 0;
        uint64_t next = frames->get()->completed() + 1;
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (checked < want && std::chrono::steady_clock::now() < until) {
            auto done = frames->get()->waitFrame(next, std::chrono::seconds(10));
            if (!done) FAIL(done.error().toString());
            const uint64_t newest = *done;
            const uint64_t signalledNs = systemNs();
            const auto record = frames->get()->record(newest);
            next = newest + 1;
            if (!record.has_value() || record->epoch != epoch) {
                releaseUpTo(newest);
                continue;
            }
            if (latencyMs < 0.0) latencyMs = static_cast<double>(signalledNs - pushedNs) * 1e-6;
            // The reference: the epoch's chunks drawn directly from the camera
            // the record says the frame was drawn from.
            const render::Camera camera = cameraOf(*record);
            if (record->cameraSamples == 2) {
                const double t = static_cast<double>(record->instantNs - t0) / static_cast<double>(t1 - t0);
                const render::Camera expected = bridge::interpolateCamera(cam0, cam1, t);
                for (size_t k = 0; k < 16; ++k) {
                    seen.cameraError = std::max(seen.cameraError,
                                                std::abs(expected.cameraToWorld.m[k] - camera.cameraToWorld.m[k]));
                }
                ++seen.interpolated;
            }
            std::vector<render::SplatInstance> instances;
            for (const Chunk& c : epochs[epoch]) {
                if (c.cloud) instances.push_back({&*c.cloud, render::Mat4::identity()});
            }
            render::RenderTargets reference;
            REQUIRE(raster->render(render::projectionFor(camera, w, h), instances, settings, reference));
            const uint32_t slot = static_cast<uint32_t>(record->slot);
            auto diff = render::compareImages(*gpu->library, frames->get()->aov(slot, 0), reference.colour, w, h);
            REQUIRE(diff);
            const DepthCheck depth = checkDepth(device, *gpu->library, frames->get()->aov(slot, 1), reference.depth, w * h);
            if (seen.checked < 8 || diff->p99 > 1 || diff->max > 2) {
                std::printf("  frame %llu epoch %u samples %llu: colour p99 %u max %u over2 %llu; depth: drew %u / %u, "
                            "coverage apart %u, depth apart %u, range %.4g..%.4g / %.4g..%.4g\n",
                            static_cast<unsigned long long>(newest), epoch,
                            static_cast<unsigned long long>(record->cameraSamples), diff->p99, diff->max,
                            static_cast<unsigned long long>(diff->over2), depth.drewA,
                            depth.drewB, depth.coverage, depth.apart, depth.minA, depth.maxA, depth.minB, depth.maxB);
            }
            // What the device said of the frame beside it: a frame whose
            // pairs outgrew the rasteriser's buffers was drawn short.
            const bridge::FrameState state = frames->get()->state(slot);
            CHECK(state.written == 1);
            if (!state.whole()) {
                std::printf("  frame %llu epoch %u: drawn short (%u pairs, %u held), p99 %u max %u\n",
                            static_cast<unsigned long long>(newest), epoch, state.pairs, state.pairCapacity,
                            diff->p99, diff->max);
                ++seen.short_;
                releaseUpTo(newest);
                continue;
            }
            seen.worstP99 = std::max(seen.worstP99, diff->p99);
            seen.worstMax = std::max(seen.worstMax, diff->max);
            // Hydra's camera is single precision and the direct draw's double:
            // a splat on a pixel's edge may fall either side of it, so a
            // handful of pixels may part by more than 2 (over2), never the
            // image (p99).
            seen.worstOver2 = std::max<uint64_t>(seen.worstOver2, diff->over2);
            seen.mismatched += (diff->p99 <= 1 && diff->over2 <= 30) ? 0 : 1;
            seen.depthApart = std::max(seen.depthApart, static_cast<int>(depth.coverage + depth.apart));
            ++seen.checked;
            ++checked;
            releaseUpTo(newest);
        }
        CHECK(checked == want);
        return latencyMs;
    };
    const auto commit = [&](uint32_t epoch, uint64_t fence) {
        json list = json::array();
        uint64_t at = 0;
        for (uint32_t c = 0; c < kChunks; ++c) {
            const Chunk& chunk = epochs[epoch][c];
            if (!chunk.spz.empty()) {
                std::memcpy(ring->bytes().data() + at, chunk.spz.data(), chunk.spz.size());
                list.push_back({{"chunk", c}, {"op", "replace"}, {"offset", at}, {"bytes", chunk.spz.size()}});
                at += chunk.spz.size();
            } else if (chunk.seed != 0) {
                list.push_back({{"chunk", c}, {"op", "replace"}, {"device", {{"first", chunk.first}, {"count", kSplats}}}});
            } else {
                list.push_back({{"chunk", c}, {"op", "remove"}});
            }
            list.back()["ptpTime"] = systemNs();
        }
        json args{{"name", stream}, {"epoch", epoch}, {"pages", list}};
        if (fence > 0) args["fence"] = fence;
        const uint64_t pushed = systemNs();
        const json answer = engine.call("stream_commit", args);
        INFO(answer.dump());
        CHECK(!isError(answer));
        return pushed;
    };

    // Epoch 1: SPZ pages from the ring, under the moving camera.
    const double latency1 = check(1, 4, commit(1, 0));
    // Epoch 2: one chunk this process's device has not written yet. The
    // engine's device waits for the fence; the host writes the records 150 ms
    // after the commit and raises the fence on its device.
    const uint64_t pushed2 = commit(2, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    {
        gpu::CommandBatch batch(device);
        writeRecords(*gpu->library, batch, pages->records(), epochs[2][1].first, kSplats, epochs[2][1].seed);
        REQUIRE(pages->written().signalAfter(batch, 1));
    }
    const double latency2 = check(2, 4, pushed2);
    // Epoch 3: written and raised before the commit; chunk 0 removed.
    {
        gpu::CommandBatch batch(device);
        writeRecords(*gpu->library, batch, pages->records(), epochs[3][1].first, kSplats, epochs[3][1].seed);
        REQUIRE(pages->written().signalAfter(batch, 2));
    }
    const double latency3 = check(3, 4, commit(3, 2));

    // LATENCY, push to the AOVs' signal: SPZ epochs, two contents in turn,
    // each due at once.
    const std::array<Chunk, kChunks> contentA{epochs[1][0], epochs[1][1]};
    const std::array<Chunk, kChunks> contentB{epochs[2][0], epochs[1][1]};
    std::vector<double> latencies;
    for (uint32_t k = 0; k < 24; ++k) {
        const uint32_t epoch = static_cast<uint32_t>(epochs.size());
        epochs.push_back(k % 2 == 0 ? contentB : contentA);
        latencies.push_back(check(epoch, 1, commit(epoch, 0)));
    }
    std::sort(latencies.begin(), latencies.end());

    const json status = json::parse(textOf(engine.call("subscription_status", json::object())));
    const json unsubscribed = engine.call("render_unsubscribe", json::object());
    CHECK(!isError(unsubscribed));
    const uint64_t consumed = pages->consumed();
    CHECK(!isError(engine.call("stream_close", {{"name", stream}})));
    auto exited = engine.child.wait();
    REQUIRE(exited);

    std::printf("  bridge: %d frames checked against a direct draw (p99 %u, max %u, over2 %llu worst; %d not one whole "
                "epoch; %d drawn short, said so), %d interpolated cameras (worst %.2e off), depth pixels apart %d\n",
                seen.checked, seen.worstP99, seen.worstMax, static_cast<unsigned long long>(seen.worstOver2),
                seen.mismatched, seen.short_, seen.interpolated, seen.cameraError, seen.depthApart);
    std::printf("  bridge: push to AOV signal %.1f / %.1f (fence after commit) / %.1f ms; %zu SPZ epochs median "
                "%.1f ms, p95 %.1f\n",
                latency1, latency2, latency3, latencies.size(), latencies[latencies.size() / 2],
                latencies[latencies.size() * 95 / 100]);
    std::printf("  bridge: %s\n", status.dump().c_str());
    CHECK(seen.mismatched == 0);
    CHECK(seen.interpolated > 0);
    CHECK(seen.cameraError < 1e-6);
    CHECK(status["hostWaitsSteadyMax"] == 0);
    CHECK(status["steadyFrames"].get<uint64_t>() > 20);
    CHECK(status["failed"] == 0);
    CHECK(consumed >= 2);
    CHECK(*exited == 0);
}

#endif   // ATHENEA_TEST_SPZ
