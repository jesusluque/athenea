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
#include "athenea/render/ReferenceRenderer.h"
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
               bool hiInVideo, uint32_t positionBlock = 0, int positionHighBits = -1, bool chroma = false) {
    Encoded e;
    auto layout = video::PlaneLayout::make(cloud.count, bits, cloud.restPerColour, hiInVideo,
                                           video::PlaneLayout::kBandSplats, 0, 0, 0, 0, chroma);
    REQUIRE(layout);
    e.layout = *layout;
    e.layout.positionBlock = positionBlock;
    e.layout.positionHighBits = positionHighBits < 0 ? bits : static_cast<uint32_t>(positionHighBits);

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
                                             const scene::GpuSplats& cloud, Encoded& e, rhi::ITexture* luma,
                                             rhi::ITexture* chroma = nullptr) {
    gpu::Buffer side = sidePage(device, e.planes);
    auto back = video::PlanesCodec::allocate(device, cloud.count, e.layout.restPerColour, "test.back");
    REQUIRE(back);
    {
        gpu::CommandBatch batch(device);
        REQUIRE(codec.decode(batch, e.layout, luma, chroma, side.rhi(), 0, e.layout.rangeBytes(), 0, cloud.count,
                             *back, 0, e.layout.restPerColour));
        REQUIRE(batch.submit(true));
    }
    auto errors = codec.compare(e.layout, *back, cloud, e.order, 0, e.ranges, side, e.layout.rangeBytes());
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

TEST_CASE("positions over their block's box: the high parts exact, fewer of them or none", "[video][gpu][planes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = *gpu->device;
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto codec = video::PlanesCodec::create(*gpu->library);
    REQUIRE(codec);
    test::CloudBuilder built = test::randomCloud(70000, 17, 0.004F, 0.3F);
    auto cloud = loader->upload(built.raw, 3);
    REQUIRE(cloud);
    for (const uint32_t block : {256u, 4096u}) {
        for (const int high : {8, 4, 0}) {
            CAPTURE(block, high);
            Encoded e = encode(device, *codec, *cloud, 8, false, block, high);
            // Each chunk's side payload: its blocks' boxes, then its high parts.
            const uint64_t boxes = uint64_t{video::PlaneLayout::kBandSplats / block} * 12;
            REQUIRE(e.layout.highChunkBytes() == boxes + (high == 0 ? 0u : 3u * video::PlaneLayout::kBandSplats));
            REQUIRE(e.planes.high.size() == 3u * e.layout.highChunkBytes());
            gpu::TextureDesc desc;
            desc.width = e.layout.width;
            desc.height = e.layout.height;
            desc.format = rhi::Format::R8Unorm;
            desc.label = "test.luma";
            auto luma = gpu::Texture::create(device, desc);
            REQUIRE(luma);
            REQUIRE(luma->upload(device, 0, 0, e.planes.luma));
            const std::vector<video::AttributeError> errors = roundTrip(device, *codec, *cloud, e, luma->rhi());
            for (size_t a = 0; a < errors.size(); ++a) {
                CAPTURE(a, errors[a].maxSteps, errors[a].meanSteps);
                CHECK(errors[a].far == 0);
                if (a < 3 && high == 8) {
                    // 16 bits over a block's box are finer than over the frame's.
                    CHECK(errors[a].maxSteps <= 0.75);
                } else if (a >= 3) {
                    const bool rotation = a >= 7 && a < 11;
                    CHECK(errors[a].maxSteps <= (rotation ? 2.5 : 0.75));
                }
            }
        }
    }
}

TEST_CASE("colour in chroma: Y a plane, Co and Cg one a 2 x 2 block of four Morton neighbours",
          "[video][gpu][planes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = *gpu->device;
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto codec = video::PlanesCodec::create(*gpu->library);
    REQUIRE(codec);
    test::CloudBuilder built = test::randomCloud(70000, 17, 0.004F, 0.3F);
    auto cloud = loader->upload(built.raw, 3);
    REQUIRE(cloud);
    Encoded e = encode(device, *codec, *cloud, 8, false, 256, 8, true);
    // 3 position, 8 opacity / scale / rotation, and a Y plane for the base
    // colour and each of degree 1's three bases.
    REQUIRE(e.layout.planes() == 15);
    REQUIRE(e.planes.chroma.size() == e.layout.chromaBytes());
    gpu::TextureDesc desc;
    desc.width = e.layout.width;
    desc.height = e.layout.height;
    desc.format = rhi::Format::R8Unorm;
    desc.label = "test.luma";
    auto luma = gpu::Texture::create(device, desc);
    REQUIRE(luma);
    REQUIRE(luma->upload(device, 0, 0, e.planes.luma));
    desc.width /= 2;
    desc.height /= 2;
    desc.format = rhi::Format::RG8Unorm;
    desc.label = "test.chroma";
    auto chroma = gpu::Texture::create(device, desc);
    REQUIRE(chroma);
    REQUIRE(chroma->upload(device, 0, 0, e.planes.chroma));
    const std::vector<video::AttributeError> errors =
        roundTrip(device, *codec, *cloud, e, luma->rhi(), chroma->rhi());
    for (size_t a = 0; a < errors.size(); ++a) {
        CAPTURE(a, errors[a].maxSteps, errors[a].meanSteps);
        CHECK(errors[a].far == 0);
        const bool rotation = a >= 7 && a < 11;
        const bool colour = a >= 11;
        if (a < 11) {
            CHECK(errors[a].maxSteps <= (rotation ? 2.5 : 0.75));
        } else if (colour && (a - 11) % 3 == 0) {
            // Y back within half a step and the halves' rounding; Co and Cg
            // are their block's mean, which a random cloud's neighbours do
            // not share, so only their Y is bounded here.
            CHECK(errors[a].maxSteps <= 1.5);
        }
    }
}

TEST_CASE("the geometry layer carries position, scale and rotation exactly, as keys and as differences",
          "[video][gpu][planes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = *gpu->device;
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto codec = video::PlanesCodec::create(*gpu->library);
    REQUIRE(codec);
    test::CloudBuilder built = test::randomCloud(70000, 17, 0.004F, 0.3F);
    auto cloud = loader->upload(built.raw, 3);
    REQUIRE(cloud);
    auto layout = video::PlaneLayout::make(cloud->count, 8, cloud->restPerColour, false,
                                           video::PlaneLayout::kBandSplats, 0, 0, 0, 0, true, true);
    REQUIRE(layout);
    REQUIRE(layout->planes() == 1 + 1 + cloud->restPerColour);   // opacity, base colour Y, a Y a basis
    layout->positionHighBits = 0;
    gpu::Buffer ranges, order;
    REQUIRE(codec->ranges(*cloud, *layout, cloud->count, ranges));
    REQUIRE(codec->order(*cloud, cloud->count, ranges, order));
    auto planes = codec->encode(*cloud, *layout, order, 0, ranges);
    REQUIRE(planes);
    gpu::TextureDesc desc;
    desc.width = layout->width;
    desc.height = layout->height;
    desc.format = rhi::Format::R8Unorm;
    desc.label = "test.luma";
    auto luma = gpu::Texture::create(device, desc);
    REQUIRE(luma);
    REQUIRE(luma->upload(device, 0, 0, planes->luma));
    desc.width /= 2;
    desc.height /= 2;
    desc.format = rhi::Format::RG8Unorm;
    auto chroma = gpu::Texture::create(device, desc);
    REQUIRE(chroma);
    REQUIRE(chroma->upload(device, 0, 0, planes->chroma));
    gpu::BufferDesc words;
    words.bytes = uint64_t{cloud->count} * 16;
    words.elementBytes = 4;
    std::vector<std::byte> zeros(words.bytes);
    auto now = gpu::Buffer::create(device, words, zeros.data());
    auto before = gpu::Buffer::create(device, words, zeros.data());
    gpu::BufferDesc twoFrames = words;
    twoFrames.bytes *= 2;
    std::vector<std::byte> moreZeros(twoFrames.bytes);
    auto state = gpu::Buffer::create(device, twoFrames, moreZeros.data());
    REQUIRE(now);
    REQUIRE(before);
    REQUIRE(state);
    // Frame 0 a key, frame 1 the same cloud as differences (all zero), frame
    // 2 a staggered refresh: each comes back the same.
    for (const auto& [frame, interval, stagger] : {std::tuple{0u, 4u, false}, std::tuple{1u, 4u, false},
                                                   std::tuple{2u, 4u, true}}) {
        CAPTURE(frame, interval, stagger);
        const video::GeometryFrame at{frame, interval, stagger};
        auto geometry = codec->geometryEncode(*cloud, *layout, order, 0, ranges, at, *now, *before);
        REQUIRE(geometry);
        REQUIRE(geometry->size() == layout->chunks() * layout->geometryChunkBytes());
        video::EncodedPlanes side = *planes;
        side.high = *geometry;   // the side page helper lays ranges then these bytes
        gpu::Buffer page = sidePage(device, side);
        auto back = video::PlanesCodec::allocate(device, cloud->count, layout->restPerColour, "test.back");
        REQUIRE(back);
        {
            gpu::CommandBatch batch(device);
            REQUIRE(codec->geometryApply(batch, *layout, page.rhi(), 0, layout->rangeBytes(), at, 0, 0, cloud->count,
                                         *state));
            REQUIRE(codec->decode(batch, *layout, luma->rhi(), chroma->rhi(), page.rhi(), 0, layout->rangeBytes(), 0,
                                  cloud->count, *back, 0, layout->restPerColour, &*state, 0));
            REQUIRE(batch.submit(true));
        }
        auto errors = codec->compare(*layout, *back, *cloud, order, 0, ranges, page, layout->rangeBytes());
        REQUIRE(errors);
        for (size_t a = 0; a < 11; ++a) {
            CAPTURE(a, (*errors)[a].maxSteps);
            if (a < 3) {
                CHECK((*errors)[a].maxSteps <= 0.75);   // 16 bits over the range: half a step of 2 x 8 bits
            } else if (a >= 7) {
                CHECK((*errors)[a].maxSteps == 0.0);    // the rotation word itself
            }
        }
        std::swap(*now, *before);
    }
}

TEST_CASE("the geometry layer's motion: the decoder holds what the encoder held", "[video][gpu][planes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = *gpu->device;
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto codec = video::PlanesCodec::create(*gpu->library);
    REQUIRE(codec);
    test::CloudBuilder built = test::randomCloud(70000, 29, 0.004F, 0.3F);
    auto cloud = loader->upload(built.raw, 3);
    REQUIRE(cloud);
    auto layout = video::PlaneLayout::make(cloud->count, 8, cloud->restPerColour, false,
                                           video::PlaneLayout::kBandSplats, 0, 0, 0, 0, true, true);
    REQUIRE(layout);
    layout->positionHighBits = 0;
    layout->geometryPositionBits = 12;
    layout->geometryScaleBits = 8;
    layout->geometryRotationBits = 8;
    layout->geometryMotion = true;
    gpu::Buffer ranges, order;
    REQUIRE(codec->ranges(*cloud, *layout, cloud->count, ranges));
    REQUIRE(codec->order(*cloud, cloud->count, ranges, order));
    auto planes = codec->encode(*cloud, *layout, order, 0, ranges);
    REQUIRE(planes);
    gpu::BufferDesc words;
    words.bytes = uint64_t{cloud->count} * 16;
    words.elementBytes = 4;
    std::vector<std::byte> zeros(words.bytes * 2);
    auto now = gpu::Buffer::create(device, words, zeros.data());
    auto before = gpu::Buffer::create(device, words, zeros.data());
    gpu::BufferDesc twoFrames = words;
    twoFrames.bytes *= 2;
    auto state = gpu::Buffer::create(device, twoFrames, zeros.data());
    REQUIRE(now);
    REQUIRE(before);
    REQUIRE(state);
    // Keys every 3 frames, staggered or whole, with and without holds: each
    // frame's planes, cluster motion and keys bring the decoder's state to
    // the encoder's word for word.
    for (const auto& [frame, stagger, hold] : {std::tuple{0u, false, 0u}, std::tuple{1u, false, 0u},
                                               std::tuple{2u, false, 1u}, std::tuple{3u, false, 1u},
                                               std::tuple{4u, true, 2u}, std::tuple{5u, true, 0u}}) {
        CAPTURE(frame, stagger, hold);
        video::GeometryFrame at{frame, 3, stagger, 3};
        at.holdPosition = at.holdScale = at.holdRotation = hold;
        auto geometry = codec->geometryEncode(*cloud, *layout, order, 0, ranges, at, *now, *before);
        REQUIRE(geometry);
        REQUIRE(geometry->size() == layout->chunks() * layout->geometryChunkBytes());
        video::EncodedPlanes side = *planes;
        side.high = *geometry;
        gpu::Buffer page = sidePage(device, side);
        {
            gpu::CommandBatch batch(device);
            REQUIRE(codec->geometryApply(batch, *layout, page.rhi(), 0, layout->rangeBytes(), at, 0, 0, cloud->count,
                                         *state));
            REQUIRE(batch.submit(true));
        }
        auto different = render::countDifferent(*gpu->library, *now, *state, cloud->count * 4);
        REQUIRE(different);
        CHECK(*different == 0);
        std::swap(*now, *before);
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
    auto units = (*encoder)->encode(e.planes.luma, e.planes.chroma, 0);
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
