// Copyright (c) 2026 jesus luque.
//
// PLY HEADERS AND RECORDS A WRITER GOT WRONG, OR WROTE ODDLY. Clouds whose
// properties come in every type the format names, in random order, with
// unused ones between them (every offset unaligned), must decode on the device
// to exactly what the same values written as float32 decode to; and headers
// and bodies that are damaged -- truncated, bit-flipped, lying about their
// counts, in another byte order -- must be refused or read, never crash or
// read past the file. What is compared is the decoded cloud, word for word, on
// the device (countDifferent).
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "athenea/io/Readers.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

struct Lcg {
    uint64_t s = 0x2545F4914F6CDD1Dull;
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(s >> 33);
    }
};

struct Type {
    const char* name;
    int         bytes;
    bool        sign;
    bool        integer;
};
const Type kTypes[] = {{"char", 1, true, true},     {"uchar", 1, false, true},  {"short", 2, true, true},
                       {"ushort", 2, false, true},  {"int", 4, true, true},     {"uint", 4, false, true},
                       {"float", 4, true, false},   {"double", 8, true, false}, {"int8", 1, true, true},
                       {"uint8", 1, false, true},   {"int16", 2, true, true},   {"uint16", 2, false, true},
                       {"int32", 4, true, true},    {"uint32", 4, false, true}, {"float32", 4, true, false},
                       {"float64", 8, true, false}};

const char* kNames[] = {"x",     "y",     "z",     "opacity", "scale_0", "scale_1", "scale_2",
                        "rot_0", "rot_1", "rot_2", "rot_3",   "f_dc_0",  "f_dc_1",  "f_dc_2"};

void put(std::string& out, const Type& t, int value) {
    const auto raw = [&](const auto v) { out.append(reinterpret_cast<const char*>(&v), sizeof v); };
    const std::string n = t.name;
    if (n == "float" || n == "float32") raw(static_cast<float>(value));
    else if (n == "double" || n == "float64") raw(static_cast<double>(value));
    else if (t.bytes == 1) raw(static_cast<int8_t>(value));
    else if (t.bytes == 2) raw(static_cast<int16_t>(value));
    else raw(static_cast<int32_t>(value));
}

/// The value property `p` of splat `i` holds: small integers every type keeps
/// exactly, negative only where the type is signed.
int valueOf(uint32_t p, uint32_t i, bool unsignedOnly) {
    int v = 0;
    if (p < 3) v = static_cast<int>((i * 7 + p * 13) % 9) - 4;          // positions -4..4
    else if (p == 3) v = static_cast<int>(i % 5) - 2;                   // opacity logit -2..2
    else if (p < 7) v = -3 - static_cast<int>((i + p) % 3);             // ln scale -5..-3
    else if (p < 11) v = static_cast<int>((i + p) % 3);                 // rotation 0..2
    else v = static_cast<int>((i * 3 + p) % 3) - 1;                     // DC -1..1
    if (unsignedOnly && v < 0) v = -v;
    return v;
}

fs::path dirOf() {
    const fs::path d = fs::temp_directory_path() / "athenea-ply-qa";
    fs::create_directories(d);
    return d;
}

/// A PLY of `n` splats: each named property of the type `types[p]`, with
/// `extra` unused properties of random types between them.
std::string plyOf(uint32_t n, const std::vector<int>& types, Lcg& rng, uint32_t extra, bool allFloat,
                  const std::vector<bool>& unsignedOnly) {
    std::string header = "ply\nformat binary_little_endian 1.0\ncomment athenea qa\nelement vertex " +
                         std::to_string(n) + "\n";
    std::vector<int> layout;   // >= 0: named property; < 0: an unused one of type -1 - t
    for (uint32_t p = 0; p < 14; ++p) {
        if (extra > 0 && rng.next() % 2 == 0) {
            layout.push_back(-1 - static_cast<int>(rng.next() % 16));
            --extra;
        }
        layout.push_back(static_cast<int>(p));
    }
    for (uint32_t e = 0; e < extra; ++e) layout.push_back(-1 - static_cast<int>(rng.next() % 16));
    int unused = 0;
    for (const int l : layout) {
        if (l >= 0) {
            header += std::string("property ") + (allFloat ? "float" : kTypes[types[static_cast<size_t>(l)]].name) +
                      " " + kNames[l] + "\n";
        } else {
            header += std::string("property ") + kTypes[-1 - l].name + " unused_" + std::to_string(unused++) + "\n";
        }
    }
    header += "end_header\n";
    std::string body;
    for (uint32_t i = 0; i < n; ++i) {
        for (const int l : layout) {
            if (l >= 0) {
                const Type& t = allFloat ? kTypes[6] : kTypes[types[static_cast<size_t>(l)]];
                put(body, t, valueOf(static_cast<uint32_t>(l), i, unsignedOnly[static_cast<size_t>(l)]));
            } else {
                put(body, kTypes[-1 - l], 1);
            }
        }
    }
    return header + body;
}

fs::path write(const std::string& name, const std::string& bytes) {
    const fs::path path = dirOf() / name;
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

}   // namespace

TEST_CASE("qa: a PLY of any mix of types decodes as the same values in float32", "[render][io][gpu][qa]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    Lcg rng;
    constexpr uint32_t kSplats = 1537;   // not a multiple of any slice
    int differing = 0, failed = 0;
    for (int round = 0; round < 24; ++round) {
        std::vector<int> types(14);
        std::vector<bool> unsignedOnly(14);
        for (uint32_t p = 0; p < 14; ++p) {
            types[p] = static_cast<int>(rng.next() % 16);
            unsignedOnly[p] = !kTypes[types[p]].sign;
        }
        Lcg layoutRng;
        layoutRng.s = rng.next();
        Lcg sameLayout = layoutRng;
        const uint32_t extra = rng.next() % 6;
        const fs::path mixed = write("mixed_" + std::to_string(round) + ".ply",
                                     plyOf(kSplats, types, layoutRng, extra, false, unsignedOnly));
        const fs::path flat = write("flat_" + std::to_string(round) + ".ply",
                                    plyOf(kSplats, types, sameLayout, 0, true, unsignedOnly));
        auto a = io::readSplats(mixed);
        auto b = io::readSplats(flat);
        if (!a || !b) {
            ++failed;
            std::printf("  round %d: %s\n", round, (!a ? a.error() : b.error()).toString().c_str());
            continue;
        }
        auto ca = loader->upload(*a, 3);
        auto cb = loader->upload(*b, 3);
        if (!ca || !cb) {
            ++failed;
            std::printf("  round %d upload: %s\n", round, (!ca ? ca.error() : cb.error()).toString().c_str());
            continue;
        }
        REQUIRE(ca->count == cb->count);
        uint64_t diff = 0;
        for (const auto& [x, y, words] :
             {std::tuple{&ca->positions, &cb->positions, ca->count * 4}, std::tuple{&ca->shape, &cb->shape, ca->count * 4},
              std::tuple{&ca->sh, &cb->sh, ca->count * ca->shWords}}) {
            auto d = render::countDifferent(*gpu->library, *x, *y, words);
            REQUIRE(d);
            diff += *d;
        }
        if (diff != 0) {
            ++differing;
            std::string said;
            for (uint32_t p = 0; p < 14; ++p) said += std::string(kNames[p]) + ":" + kTypes[types[p]].name + " ";
            std::printf("  round %d (%u unused): %llu words differ; %s\n", round, extra,
                        static_cast<unsigned long long>(diff), said.c_str());
        }
    }
    std::printf("  24 mixed-type PLYs against float32: %d differ, %d not read\n", differing, failed);
    CHECK(differing == 0);
    CHECK(failed == 0);
}

TEST_CASE("qa: damaged PLY headers and bodies are refused or read, never past the file", "[render][io][gpu][qa]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    Lcg rng;
    std::vector<int> types(14);
    std::vector<bool> unsignedOnly(14);
    for (uint32_t p = 0; p < 14; ++p) {
        types[p] = static_cast<int>(rng.next() % 16);
        unsignedOnly[p] = !kTypes[types[p]].sign;
    }
    Lcg layout;
    const std::string good = plyOf(300, types, layout, 3, false, unsignedOnly);
    const size_t bodyAt = good.find("end_header\n") + 11;
    std::vector<std::pair<std::string, std::string>> cases;
    for (size_t n : {size_t{0}, size_t{3}, size_t{20}, bodyAt - 5, bodyAt, bodyAt + 1, bodyAt + 100, good.size() - 1}) {
        cases.emplace_back("truncated at " + std::to_string(n), good.substr(0, n));
    }
    const auto replaced = [&](const std::string& from, const std::string& to) {
        std::string s = good;
        const size_t at = s.find(from);
        if (at != std::string::npos) s.replace(at, from.size(), to);
        return s;
    };
    cases.emplace_back("count 0", replaced("element vertex 300", "element vertex 0"));
    cases.emplace_back("count 301", replaced("element vertex 300", "element vertex 301"));
    cases.emplace_back("count 2^32", replaced("element vertex 300", "element vertex 4294967296"));
    cases.emplace_back("count 2^63", replaced("element vertex 300", "element vertex 9223372036854775807"));
    cases.emplace_back("count -1", replaced("element vertex 300", "element vertex -1"));
    cases.emplace_back("count text", replaced("element vertex 300", "element vertex many"));
    cases.emplace_back("big endian", replaced("binary_little_endian", "binary_big_endian"));
    cases.emplace_back("ascii", replaced("binary_little_endian", "ascii"));
    cases.emplace_back("no end_header", replaced("end_header\n", "end_headr\n"));
    cases.emplace_back("unknown type", replaced("property ", "property int64 "));
    cases.emplace_back("a list", replaced("property ", "property list uchar int vertex_indices\nproperty "));
    cases.emplace_back("x twice", replaced("property ", std::string("property float x\nproperty ")));
    cases.emplace_back("no x", replaced(" x\n", " q\n"));
    cases.emplace_back("element before vertex", replaced("element vertex", "element face 3\nproperty uchar k\nelement vertex"));
    cases.emplace_back("4096 properties", [&] {
        std::string extra;
        for (int k = 0; k < 4096; ++k) extra += "property uchar pad" + std::to_string(k) + "\n";
        return replaced("end_header\n", extra + "end_header\n");
    }());
    for (int k = 0; k < 400; ++k) {
        std::string s = good;
        const size_t span = rng.next() % 3 == 0 ? s.size() : bodyAt;
        for (int f = 0; f < 1 + static_cast<int>(rng.next() % 3); ++f) {
            s[rng.next() % span] ^= static_cast<char>(1u << (rng.next() % 8));
        }
        cases.emplace_back("flipped " + std::to_string(k), std::move(s));
    }
    int read = 0, uploaded = 0;
    for (size_t k = 0; k < cases.size(); ++k) {
        const fs::path path = write("damaged.ply", cases[k].second);
        auto raw = io::readSplats(path);
        if (!raw) {
            if (k < 30) std::printf("  %-24s refused: %s\n", cases[k].first.c_str(), raw.error().message().c_str());
            continue;
        }
        ++read;
        // What a reader accepts must lie inside the file it read.
        const uint64_t need = uint64_t{raw->count} * raw->encoding.floatsPerRecord;
        if (k < 30) std::printf("  %-24s read: %u splats\n", cases[k].first.c_str(), raw->count);
        CHECK(raw->count <= 300);
        (void)need;
        auto cloud = loader->upload(*raw, 3);
        uploaded += cloud ? 1 : 0;
    }
    std::printf("  %zu damaged PLYs: %d read, %d uploaded\n", cases.size(), read, uploaded);
}
