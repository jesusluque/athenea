// Copyright (c) 2026 jesus luque.
//
// THE MCP STREAM TOOLS UNDER WHAT A CLIENT GETS WRONG: arguments of the wrong
// type, offsets and sizes past the ring, names nobody opened, two sessions on
// one stream, a ring a crashed server left behind, a close right after a
// commit. Every call must be answered; a refused commit must queue nothing;
// what is queued must be the bytes the producer wrote. No device: the queue a
// name stands for is drained and its pages read as bytes.
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "athenea/core/Hash.h"
#include "athenea/core/Platform.h"
#include "athenea/lod/SplatStream.h"
#include "athenea/mcp/Server.h"

using namespace athenea;
using nlohmann::json;

namespace {

json call(mcp::Server& server, const std::string& tool, const json& arguments) {
    const auto reply = server.handle(json{{"jsonrpc", "2.0"},
                                          {"id", 9},
                                          {"method", "tools/call"},
                                          {"params", {{"name", tool}, {"arguments", arguments}}}}
                                         .dump());
    REQUIRE(reply.has_value());
    return json::parse(*reply);
}

bool isError(const json& answer) {
    return !answer.contains("result") || answer["result"].value("isError", false);
}

std::string textOf(const json& answer) {
    std::string text;
    if (answer.contains("result")) {
        for (const json& block : answer["result"]["content"]) text += block.value("text", std::string{});
    }
    return text;
}

size_t drained(const std::string& name) {
    std::vector<lod::PageUpdate> out;
    lod::PageQueue::named(name)->drain(0, out);
    return out.size();
}

}   // namespace

TEST_CASE("qa: stream tools answer every malformed call and a refused commit queues nothing", "[mcp][live][qa]") {
    mcp::Server server;
    const std::string name = "athenea-qa-mcp-bad";
    drained(name);
    // stream_open with what it cannot take.
    int answered = 0;
    for (const json& args : {json{}, json{{"name", 7}}, json{{"name", nullptr}}, json{{"name", json::array()}},
                             json{{"name", ""}}, json{{"name", name}, {"ringBytes", 0}},
                             json{{"name", name}, {"ringBytes", -1}}, json{{"name", name}, {"ringBytes", "big"}},
                             json{{"name", name}, {"ringBytes", 1.0e300}}, json{{"name", name}, {"chunks", "x"}}}) {
        const json a = call(server, "stream_open", args);
        ++answered;
        CHECK(isError(a));
        if (!isError(a)) {
            std::printf("  stream_open %s accepted: %s\n", args.dump().c_str(), textOf(a).c_str());
            (void)call(server, "stream_close", {{"name", args.value("name", std::string{})}});
        }
    }
    const json opened = call(server, "stream_open", {{"name", name}, {"chunks", 4}, {"ringBytes", 4096}});
    REQUIRE(!isError(opened));
    const json ring = json::parse(textOf(opened));
    auto memory = platform::SharedMemory::open(ring["ring"].get<std::string>(), 4096);
    REQUIRE(memory);
    std::memset(memory->bytes().data(), 0x5A, 4096);

    const auto page = [](json fields) {
        json p{{"chunk", 0}, {"op", "replace"}, {"offset", 0}, {"bytes", 100}};
        for (auto& [k, v] : fields.items()) p[k] = v;
        return p;
    };
    struct Bad {
        const char* what;
        json        args;
    };
    const std::vector<Bad> bad{
        {"no name", {{"epoch", 1}, {"pages", {page({})}}}},
        {"a name not open", {{"name", "nobody"}, {"epoch", 1}, {"pages", {page({})}}}},
        {"epoch 0", {{"name", name}, {"epoch", 0}, {"pages", {page({})}}}},
        {"epoch -1", {{"name", name}, {"epoch", -1}, {"pages", {page({})}}}},
        {"epoch text", {{"name", name}, {"epoch", "one"}, {"pages", {page({})}}}},
        {"pages not a list", {{"name", name}, {"epoch", 1}, {"pages", 3}}},
        {"no pages", {{"name", name}, {"epoch", 1}, {"pages", json::array()}}},
        {"a page that is a number", {{"name", name}, {"epoch", 1}, {"pages", {5}}}},
        {"offset past the ring", {{"name", name}, {"epoch", 1}, {"pages", {page({{"offset", 5000}})}}}},
        {"bytes past the ring", {{"name", name}, {"epoch", 1}, {"pages", {page({{"offset", 4000}, {"bytes", 200}})}}}},
        {"offset + bytes wrapping", {{"name", name}, {"epoch", 1},
                                     {"pages", {page({{"offset", 18446744073709551600ull}, {"bytes", 32}})}}}},
        {"no bytes", {{"name", name}, {"epoch", 1}, {"pages", {page({{"bytes", 0}})}}}},
        {"negative bytes", {{"name", name}, {"epoch", 1}, {"pages", {page({{"bytes", -100}})}}}},
        {"chunk past the stream", {{"name", name}, {"epoch", 1}, {"pages", {page({{"chunk", 4}})}}}},
        {"chunk -1", {{"name", name}, {"epoch", 1}, {"pages", {page({{"chunk", -1}})}}}},
        {"bounds of text", {{"name", name}, {"epoch", 1},
                            {"pages", {page({{"bounds", {"a", "b", "c", "d", "e", "f"}}})}}}},
        {"bounds with a NaN (null)", {{"name", name}, {"epoch", 1},
                                      {"pages", {page({{"bounds", {0, 0, nullptr, 1, 1, 1}}})}}}},
        {"a good page, then one past the ring",
         {{"name", name}, {"epoch", 1}, {"pages", {page({}), page({{"chunk", 1}, {"offset", 4090}, {"bytes", 100}})}}}},
        {"an op nobody knows", {{"name", name}, {"epoch", 1}, {"pages", {page({{"op", "explode"}})}}}},
        {"fewer pages said than sent", {{"name", name}, {"epoch", 1}, {"epochPages", 1},
                                        {"pages", {page({}), page({{"chunk", 1}})}}}},
        {"one chunk twice", {{"name", name}, {"epoch", 1}, {"pages", {page({}), page({})}}}},
    };
    int accepted = 0;
    for (const Bad& b : bad) {
        const json a = call(server, "stream_commit", b.args);
        ++answered;
        const size_t queued = drained(name);
        if (!isError(a) || queued != 0) {
            ++accepted;
            std::printf("  stream_commit, %s: %s, %zu pages queued (%s)\n", b.what, isError(a) ? "refused" : "accepted",
                        queued, textOf(a).c_str());
        }
    }
    std::printf("  %d malformed calls answered; %d of %zu bad commits accepted or queued something\n", answered,
                accepted, bad.size());
    CHECK(accepted == 0);
    // Closed, the name is no stream and the ring is gone.
    CHECK(!isError(call(server, "stream_close", {{"name", name}})));
    CHECK(isError(call(server, "stream_commit", {{"name", name}, {"epoch", 2}, {"pages", {page({})}}})));
    CHECK(isError(call(server, "stream_close", {{"name", name}})));
    *memory = platform::SharedMemory();   // the producer lets go too (a Windows mapping lives while one is open)
    CHECK(!platform::SharedMemory::open(ring["ring"].get<std::string>(), 4096));
    CHECK(drained(name) == 0);
}

TEST_CASE("qa: a commit's pages are the bytes written, whatever the ring holds after", "[mcp][live][qa]") {
    mcp::Server server;
    const std::string name = "athenea-qa-mcp-copy";
    drained(name);
    const json opened = call(server, "stream_open", {{"name", name}, {"chunks", 2}, {"ringBytes", 1 << 16}});
    REQUIRE(!isError(opened));
    const json ring = json::parse(textOf(opened));
    auto memory = platform::SharedMemory::open(ring["ring"].get<std::string>(), 1 << 16);
    REQUIRE(memory);
    for (size_t k = 0; k < 2000; ++k) memory->bytes()[k] = static_cast<std::byte>(k * 31);
    const json a = call(server, "stream_commit",
                        {{"name", name}, {"epoch", 1},
                         {"pages", {{{"chunk", 0}, {"offset", 0}, {"bytes", 1000}},
                                    {{"chunk", 1}, {"offset", 1000}, {"bytes", 1000}}}}});
    REQUIRE(!isError(a));
    // The producer writes over the ring at once, and the stream is closed.
    std::memset(memory->bytes().data(), 0, 2000);
    REQUIRE(!isError(call(server, "stream_close", {{"name", name}})));
    std::vector<lod::PageUpdate> out;
    lod::PageQueue::named(name)->drain(0, out);
    REQUIRE(out.size() == 2);
    int wrong = 0;
    for (const auto& u : out) {
        REQUIRE(u.bytes.size() == 1000);
        for (size_t k = 0; k < 1000; ++k) {
            wrong += u.bytes[k] == static_cast<std::byte>((u.chunk * 1000 + k) * 31) ? 0 : 1;
        }
    }
    CHECK(wrong == 0);
}

TEST_CASE("qa: a second session opening a stream another has open is refused", "[mcp][live][qa]") {
    // Two producers on one stream push pages of their own epochs into one
    // queue; every epoch of one is late or dropped against the other's.
    mcp::Server a, b;
    const std::string name = "athenea-qa-mcp-two";
    const json first = call(a, "stream_open", {{"name", name}, {"ringBytes", 4096}});
    REQUIRE(!isError(first));
    const json second = call(b, "stream_open", {{"name", name}, {"ringBytes", 4096}});
    std::printf("  a second session's stream_open: %s\n", isError(second) ? "refused" : "accepted");
    CHECK(isError(second));
    (void)call(a, "stream_close", {{"name", name}});
    (void)call(b, "stream_close", {{"name", name}});
}

TEST_CASE("qa: a ring a crashed server left behind does not keep its stream from opening", "[mcp][live][qa]") {
    // A server killed while a stream was open leaves its POSIX ring (no
    // unlink); the next server process counts its rings from 1 again and
    // names them by the stream's hash. Those leftovers are made here, as a
    // crashed process would have left them; a fresh server must still open
    // the stream.
#if defined(_WIN32)
    SKIP("a Windows mapping goes with the last process that held it");
#endif
    const std::string name = "athenea-qa-mcp-crash";
    std::vector<platform::SharedMemory> leftovers;
    for (uint32_t serial = 1; serial <= 256; ++serial) {
        char ring[32];
        std::snprintf(ring, sizeof ring, "athlive-%08x-%u", core::murmurHash3x86_32(name), serial);
        auto made = platform::SharedMemory::create(ring, 4096);
        if (made) leftovers.push_back(std::move(*made));
    }
    mcp::Server server;
    const json opened = call(server, "stream_open", {{"name", name}, {"ringBytes", 4096}});
    std::printf("  with %zu leftover rings: %s\n", leftovers.size(), textOf(opened).c_str());
    CHECK(!isError(opened));
    (void)call(server, "stream_close", {{"name", name}});
}
