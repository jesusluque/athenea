// Copyright (c) 2026 jesus luque.
//
// Gaussian splats as video planes: the planes kernels round-trip a cloud
// through a mosaic with no codec between (each attribute back within half a
// step of its quantisation, the position high parts exact); through the
// hardware codec where the platform has one (the high parts still exact);
// and the `.athv` container keeps what it is given. The measures are the
// compare kernel's counters.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <cstring>
#include <filesystem>
#include <vector>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Texture.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/video/Athv.h"
#include "athenea/video/Codec.h"
#include "athenea/video/Planes.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

/// A frame's side data on the device: its ranges, then its high parts.
gpu::Buffer sidePage(gpu::Device& device, const video::EncodedPlanes& planes) {
    std::vector<std::byte> bytes(planes.ranges.size() * sizeof(float) + planes.high.size() + 4);
    std::memcpy(bytes.data(), planes.ranges.data(), planes.ranges.size() * sizeof(float));
    if (!planes.high.empty()) {
        std::memcpy(bytes.data() + planes.ranges.size() * sizeof(float), planes.high.data(), planes.high.size());
    }
    bytes.resize((bytes.size() + 3) / 4 * 4);
    gpu::BufferDesc desc;
    desc.bytes = bytes.size();
    desc.elementBytes = 4;
    desc.label = "test.side";
    auto made = gpu::Buffer::create(device, desc, bytes.data());
    REQUIRE(made);
    return std::move(*made);
}

struct Encoded {
    video::PlaneLayout   layout;
    video::EncodedPlanes planes;
    gpu::Buffer          order, ranges;
};

Encoded encode(gpu::Device& device, video::PlanesCodec& codec, const scene::GpuSplats& cloud, uint32_t bits,
               bool hiInVideo) {
    Encoded e;
    auto layout = video::PlaneLayout::make(cloud.count, bits, cloud.restPerColour, hiInVideo,
                                           video::PlaneLayout::kBandSplats, 0);
    REQUIRE(layout);
    e.layout = *layout;
    REQUIRE(codec.ranges(cloud, e.layout, cloud.count, e.ranges));
    REQUIRE(codec.order(cloud, cloud.count, e.ranges, e.order));
    auto planes = codec.encode(cloud, e.layout, e.order, 0, e.ranges);
    REQUIRE(planes);
    e.planes = std::move(*planes);
    (void)device;
    return e;
}

/// The mosaic decoded back from `luma` and measured against `cloud`.
std::vector<video::AttributeError> roundTrip(gpu::Device& device, video::PlanesCodec& codec,
                                             const scene::GpuSplats& cloud, Encoded& e, rhi::ITexture* luma) {
    gpu::Buffer side = sidePage(device, e.planes);
    auto back = video::PlanesCodec::allocate(device, cloud.count, e.layout.restPerColour, "test.back");
    REQUIRE(back);
    {
        gpu::CommandBatch batch(device);
        REQUIRE(codec.decode(batch, e.layout, luma, side.rhi(), 0, e.layout.rangeBytes(), 0, cloud.count, *back, 0,
                             e.layout.restPerColour));
        REQUIRE(batch.submit(true));
    }
    auto errors = codec.compare(e.layout, *back, cloud, e.order, 0, e.ranges);
    REQUIRE(errors);
    return std::move(*errors);
}

}   // namespace

TEST_CASE("video planes give a cloud back within half a step, the position high parts exact",
          "[video][gpu][planes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = *gpu->device;
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto codec = video::PlanesCodec::create(*gpu->library);
    REQUIRE(codec);
    // Two chunks and a part: a band past the first, and a tile's spare rows.
    test::CloudBuilder built = test::randomCloud(70000, 17, 0.004F, 0.3F);
    auto cloud = loader->upload(built.raw, 3);
    REQUIRE(cloud);
    REQUIRE(cloud->restPerColour == 3);

    for (const uint32_t bits : {8u, 10u}) {
        for (const bool hiInVideo : {false, true}) {
            CAPTURE(bits, hiInVideo);
            Encoded e = encode(device, *codec, *cloud, bits, hiInVideo);
            REQUIRE(e.layout.chunks() == 3);
            REQUIRE(e.planes.luma.size() == e.layout.lumaBytes());
            REQUIRE(e.planes.high.size() == (hiInVideo ? 0u : 3u * e.layout.highChunkBytes()));
            // The mosaic as the decoder would leave it, without the codec.
            gpu::TextureDesc desc;
            desc.width = e.layout.width;
            desc.height = e.layout.height;
            desc.format = bits > 8 ? rhi::Format::R16Unorm : rhi::Format::R8Unorm;
            desc.label = "test.luma";
            auto luma = gpu::Texture::create(device, desc);
            REQUIRE(luma);
            REQUIRE(luma->upload(device, 0, 0, e.planes.luma));
            const std::vector<video::AttributeError> errors = roundTrip(device, *codec, *cloud, e, luma->rhi());
            REQUIRE(errors.size() == e.layout.attributes());
            for (size_t a = 0; a < errors.size(); ++a) {
                CAPTURE(a, errors[a].maxSteps, errors[a].meanSteps);
                // Half a step of the quantisation, and what packing the
                // decoded value as a half (and a rotation in 10-bit smallest
                // three, renormalised) adds to it -- at 10 bits a half's own
                // spacing is near a step.
                const bool rotation = a >= 7 && a < 11;
                const double most = rotation ? (bits > 8 ? 4.0 : 2.5) : (bits > 8 ? 1.25 : 0.75);
                CHECK(errors[a].maxSteps <= most);
                CHECK(errors[a].far == 0);
            }
        }
    }
}

TEST_CASE("the hardware video codec keeps the position high parts exact", "[video][gpu][codec]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = *gpu->device;
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto codec = video::PlanesCodec::create(*gpu->library);
    REQUIRE(codec);
    test::CloudBuilder built = test::randomCloud(40000, 23, 0.004F, 0.3F);
    auto cloud = loader->upload(built.raw, 3);
    REQUIRE(cloud);
    Encoded e = encode(device, *codec, *cloud, 8, false);
    video::EncoderSettings settings;
    settings.width = e.layout.width;
    settings.height = e.layout.height;
    settings.quality = 0.9;
    settings.keyInterval = 1;
    auto encoder = video::createHardwareEncoder(settings);
    if (!encoder && encoder.error().code() == ErrorCode::Unsupported) {
        SKIP("no hardware video encoder here: " << encoder.error().message());
    }
    REQUIRE(encoder);
    auto units = (*encoder)->encode(e.planes.luma, 0);
    REQUIRE(units);
    REQUIRE(units->size() == 1);
    REQUIRE(units->front().keyframe);
    REQUIRE_FALSE(units->front().parameterSets.empty());
    auto decoder = video::createHardwareDecoder(0x31637668, 8);
    REQUIRE(decoder);
    auto frame = (*decoder)->decode(units->front().unit, units->front().parameterSets, true);
    REQUIRE(frame);
    REQUIRE((*frame)->width() == e.layout.width);
    auto textures = video::createLumaTextures(device);
    REQUIRE(textures);
    auto luma = (*textures)->wrap(**frame);
    REQUIRE(luma);
    const std::vector<video::AttributeError> errors = roundTrip(device, *codec, *cloud, e, luma->texture.get());
    for (size_t a = 0; a < 3; ++a) {
        CAPTURE(a, errors[a].maxSteps);
        CHECK(errors[a].far == 0);   // the high parts went beside the video, exact
    }
    for (size_t a = 0; a < errors.size(); ++a) {
        CAPTURE(a, errors[a].maxSteps, errors[a].meanSteps);
        CHECK(errors[a].meanSteps < 8.0);   // a lossy codec at a high quality: near, not exact
    }
}

TEST_CASE("a .athv keeps its pictures, their side data and their slices", "[video][athv]") {
    const fs::path path = fs::temp_directory_path() / "athenea_test_video.athv";
    video::AthvHeader header;
    header.bits = 10;
    header.restPerColour = 3;
    header.chunkSplats = video::PlaneLayout::kBandSplats;
    header.maxSplats = 2 * video::PlaneLayout::kBandSplats;
    header.antialiased = true;
    header.rate = 24.0;
    header.rateNumerator = 24;
    header.rendition = 7;
    header.quality = 0.5F;
    {
        auto writer = video::AthvWriter::create(path, header);
        REQUIRE(writer);
        for (uint32_t k = 0; k < 3; ++k) {
            video::AthvFrame f;
            f.index = k;
            f.source = k / 2;   // pictures 0 and 1 are the slices of frame 0
            f.frame = 100 + k / 2;
            f.slice = k % 2;
            f.slices = k < 2 ? 2 : 1;
            f.firstSplat = (k % 2) * header.maxSplats;
            f.sourceSplats = 3 * header.chunkSplats;
            f.splats = k == 1 ? header.chunkSplats : header.maxSplats;
            f.keyframe = k != 2;
            f.ptpTime = 1000 + k;
            f.time = 1.0 + k;
            f.timecode = "00:00:00:0" + std::to_string(k);
            f.ranges.assign(46, 0.25F * static_cast<float>(k));
            f.high.assign(4096, std::byte{static_cast<uint8_t>(k)});
            if (f.keyframe) f.parameterSets = {{1, 2, 3}, {4, 5}, {6}};
            f.unit = {0, 0, 0, 2, 0x26, static_cast<uint8_t>(k)};
            REQUIRE(writer->get()->write(f));
        }
        REQUIRE(writer->get()->finish(0, 2));
    }
    auto reader = video::AthvReader::open(path);
    REQUIRE(reader);
    const video::AthvHeader& h = (*reader)->header();
    CHECK(h.bits == 10);
    CHECK(h.restPerColour == 3);
    CHECK(h.maxSplats == header.maxSplats);
    CHECK(h.antialiased);
    CHECK(h.rate == 24.0);
    CHECK(h.version == video::AthvHeader::kVersion);
    CHECK(h.rateNumerator == 24);
    CHECK(h.rateDenominator == 1);
    CHECK(h.rendition == 7);
    CHECK(h.sourceFrames == 2);
    CHECK(h.quality == 0.5F);
    REQUIRE((*reader)->frames() == 3);
    REQUIRE((*reader)->sourceFrames() == 2);
    CHECK((*reader)->slicesOf(0) == std::vector<uint32_t>{0, 1});
    CHECK((*reader)->slicesOf(1) == std::vector<uint32_t>{2});
    CHECK((*reader)->mostSourceSplats() == 3 * header.chunkSplats);
    for (uint32_t k = 0; k < 3; ++k) {
        auto f = (*reader)->read(k);
        REQUIRE(f);
        CHECK(f->index == k);
        CHECK(f->source == k / 2);
        CHECK(f->rendition == 7);
        CHECK(f->frame == 100 + k / 2);
        CHECK(f->slice == k % 2);
        CHECK(f->slices == (k < 2 ? 2u : 1u));
        CHECK(f->keyframe == (k != 2));
        CHECK(f->ptpTime == 1000 + k);
        CHECK(f->timecode == "00:00:00:0" + std::to_string(k));
        CHECK(f->ranges == std::vector<float>(46, 0.25F * static_cast<float>(k)));
        CHECK(f->high == std::vector<std::byte>(4096, std::byte{static_cast<uint8_t>(k)}));
        CHECK(f->parameterSets.size() == (k != 2 ? 3u : 0u));
        CHECK(f->unit == std::vector<uint8_t>{0, 0, 0, 2, 0x26, static_cast<uint8_t>(k)});
        if (k == 0) {
            // As Annex B: the parameter sets and the NAL unit, each after a start code.
            CHECK(video::annexB(*f) ==
                  std::vector<uint8_t>{0, 0, 0, 1, 1, 2, 3, 0, 0, 0, 1, 4, 5, 0, 0, 0, 1, 6, 0, 0, 0, 1, 0x26, 0});
        }
    }
    std::error_code ignored;
    fs::remove(path, ignored);
}

TEST_CASE("a picture held to a smaller size cuts the frame into slices that fit it", "[video][layout]") {
    // The web sparrow's frame: 300847 splats, degree 0 (14 planes) and 1 (23), in 1920 x 1080 at most.
    const uint32_t splats = 300847;
    for (const uint32_t rest : {0u, 3u}) {
        auto cap = video::PlaneLayout::sliceSplats(splats, 8, rest, false, video::PlaneLayout::kBandSplats, 0, 1920,
                                                   1080);
        REQUIRE(cap);
        REQUIRE(*cap < splats);
        REQUIRE(*cap % video::PlaneLayout::kBandSplats == 0);
        auto layout = video::PlaneLayout::make(*cap, 8, rest, false, video::PlaneLayout::kBandSplats, 0, 0, 1920,
                                               1080);
        REQUIRE(layout);
        CHECK(layout->width <= 1920);
        CHECK(layout->height <= 1080);
        // One more chunk would not fit.
        CHECK_FALSE(video::PlaneLayout::make(*cap + video::PlaneLayout::kBandSplats, 8, rest, false,
                                             video::PlaneLayout::kBandSplats, 0, 0, 1920, 1080));
        // A decoder rebuilds the same layout from the header alone (columns and the most a picture holds).
        auto again = video::PlaneLayout::make(*cap / 2, 8, rest, false, video::PlaneLayout::kBandSplats,
                                              layout->columns, *cap);
        REQUIRE(again);
        CHECK(again->width == layout->width);
        CHECK(again->height == layout->height);
    }
    // Degree 3 (59 planes) needs more than 1080 rows for even one chunk in three columns.
    CHECK_FALSE(video::PlaneLayout::sliceSplats(splats, 8, 15, false, video::PlaneLayout::kBandSplats, 0, 1920, 1080));
}
