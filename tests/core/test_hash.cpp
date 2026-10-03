// Copyright (c) 2026 jesus luque.
//
// The hash Cryptomatte names things by, against the numbers everyone else
// gets: MurmurHash3's own test vectors, and the specification's rule for
// making a hash safe to carry as a float.
#include "athenea/core/Hash.h"

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <map>
#include <cmath>

using namespace athenea;

TEST_CASE("MurmurHash3 x86 32 gives the reference's numbers", "[core][hash]") {
    CHECK(core::murmurHash3x86_32("", 0) == 0x00000000u);
    CHECK(core::murmurHash3x86_32("", 1) == 0x514E28B7u);
    CHECK(core::murmurHash3x86_32("", 0xFFFFFFFFu) == 0x81F16F39u);
    CHECK(core::murmurHash3x86_32("test", 0) == 0xBA6BD213u);
    CHECK(core::murmurHash3x86_32("Hello, world!", 0) == 0xC0363E43u);
    CHECK(core::murmurHash3x86_32("The quick brown fox jumps over the lazy dog", 0) == 0x2E4FF723u);
}

TEST_CASE("a Cryptomatte id is a hash whose bits are a normal float", "[core][hash][crypto]") {
    // The fixup: an exponent of all zeros or all ones has its lowest bit flipped.
    CHECK(core::cryptomatteFixup(0x00000000u) == 0x00800000u);
    CHECK(core::cryptomatteFixup(0x7F800000u) == 0x7F000000u);
    CHECK(core::cryptomatteFixup(0xFF800001u) == 0xFF000001u);
    CHECK(core::cryptomatteFixup(0x3F800000u) == 0x3F800000u);
    // And every name's id survives a float.
    for (const char* name : {"/World/Splats", "/root/Object_347/Object_199", "bunny", "", "a"}) {
        const uint32_t id = core::cryptomatteId(name);
        const float asFloat = std::bit_cast<float>(id);
        CHECK(std::isfinite(asFloat));
        CHECK(std::isnormal(asFloat));
        CHECK(std::bit_cast<uint32_t>(asFloat) == id);
        CHECK(core::hex8(id).size() == 8);
    }
    CHECK(core::hex8(0x0000ABCDu) == "0000abcd");
    CHECK(core::cryptomatteLayerKey("CryptoObject").size() == 7);
}

TEST_CASE("a manifest is written and read back", "[core][hash][crypto]") {
    const std::map<std::string, uint32_t> names{{"/World/Body", core::cryptomatteId("/World/Body")},
                                                {"/World/Wheel", core::cryptomatteId("/World/Wheel")}};
    const std::string json = core::cryptomatteManifest(names);
    CHECK(json.front() == '{');
    CHECK(json.back() == '}');
    CHECK(json.find("/World/Body") != std::string::npos);
    CHECK(core::parseCryptomatteManifest(json) == names);
    // What a reader hands us is read, not trusted: spaces are skipped, a hex
    // that is not eight digits is dropped, and the rest still comes back.
    const auto lenient = core::parseCryptomatteManifest(
        R"({ "/a" : "0000abcd" , "/short" : "abc" , "/b" : "3f800000" })");
    CHECK(lenient.size() == 2);
    CHECK(lenient.at("/a") == 0x0000ABCDu);
    CHECK(lenient.at("/b") == 0x3F800000u);
    CHECK(core::parseCryptomatteManifest("not a manifest").empty());
    CHECK(core::parseCryptomatteManifest("{}").empty());
}
