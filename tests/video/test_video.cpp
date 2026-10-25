// Copyright (c) 2026 jesus luque.
//
// Gaussian splats as video: the `.athv` container keeps what it is given.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "athenea/video/Athv.h"
#include "athenea/video/Planes.h"

using namespace athenea;
namespace fs = std::filesystem;

TEST_CASE("a .athv keeps its pictures, their side data and their slices", "[video][athv]") {
    const fs::path path = fs::temp_directory_path() / "athenea_test_video.athv";
    video::AthvHeader header;
    header.bits = 10;
    header.restPerColour = 3;
    header.chunkSplats = video::PlaneLayout::kBandSplats;
    header.maxSplats = 2 * video::PlaneLayout::kBandSplats;
    header.antialiased = true;
    header.rate = 24.0;
    {
        auto writer = video::AthvWriter::create(path, header);
        REQUIRE(writer);
        for (uint32_t k = 0; k < 3; ++k) {
            video::AthvFrame f;
            f.index = k;
            f.source = k / 2;   // pictures 0 and 1 are the slices of frame 0
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
        REQUIRE(writer->get()->finish(0));
    }
    auto reader = video::AthvReader::open(path);
    REQUIRE(reader);
    const video::AthvHeader& h = (*reader)->header();
    CHECK(h.bits == 10);
    CHECK(h.restPerColour == 3);
    CHECK(h.maxSplats == header.maxSplats);
    CHECK(h.antialiased);
    CHECK(h.rate == 24.0);
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
