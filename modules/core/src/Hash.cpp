// Copyright (c) 2026 jesus luque.
#include "athenea/core/Hash.h"

#include <cstdio>
#include <map>

namespace athenea::core {

namespace {

constexpr uint32_t rotl32(uint32_t x, int r) noexcept { return (x << r) | (x >> (32 - r)); }

constexpr uint32_t fmix32(uint32_t h) noexcept {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

}   // namespace

uint32_t murmurHash3x86_32(std::string_view key, uint32_t seed) noexcept {
    const auto* data = reinterpret_cast<const unsigned char*>(key.data());
    const size_t length = key.size();
    const size_t blocks = length / 4;
    constexpr uint32_t c1 = 0xcc9e2d51u;
    constexpr uint32_t c2 = 0x1b873593u;
    uint32_t h1 = seed;
    for (size_t i = 0; i < blocks; ++i) {
        // Little-endian, whatever the machine: the reference reads it so.
        uint32_t k1 = uint32_t{data[i * 4]} | (uint32_t{data[i * 4 + 1]} << 8) |
                      (uint32_t{data[i * 4 + 2]} << 16) | (uint32_t{data[i * 4 + 3]} << 24);
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5u + 0xe6546b64u;
    }
    const unsigned char* tail = data + blocks * 4;
    uint32_t k1 = 0;
    switch (length & 3) {
    case 3: k1 ^= uint32_t{tail[2]} << 16; [[fallthrough]];
    case 2: k1 ^= uint32_t{tail[1]} << 8; [[fallthrough]];
    case 1:
        k1 ^= uint32_t{tail[0]};
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        break;
    default: break;
    }
    h1 ^= static_cast<uint32_t>(length);
    return fmix32(h1);
}

uint32_t cryptomatteFixup(uint32_t hash) noexcept {
    const uint32_t exponent = (hash >> 23) & 0xffu;
    if (exponent == 0 || exponent == 0xffu) {
        hash ^= 1u << 23;
    }
    return hash;
}

uint32_t cryptomatteId(std::string_view name) noexcept { return cryptomatteFixup(murmurHash3x86_32(name)); }

std::string hex8(uint32_t bits) {
    char out[9];
    std::snprintf(out, sizeof(out), "%08x", bits);
    return std::string(out);
}

std::string cryptomatteLayerKey(std::string_view layer) { return hex8(murmurHash3x86_32(layer)).substr(0, 7); }


std::map<std::string, uint32_t> parseCryptomatteManifest(std::string_view json) {
    // A flat scan for "key":"value" pairs: the manifest is one object of
    // strings, and a reader that walks it needs no parser.
    std::map<std::string, uint32_t> names;
    const auto readString = [&](size_t& at, std::string& into) {
        if (at >= json.size() || json[at] != '"') {
            return false;
        }
        ++at;
        into.clear();
        while (at < json.size() && json[at] != '"') {
            if (json[at] == '\\' && at + 1 < json.size()) {
                ++at;
            }
            into.push_back(json[at]);
            ++at;
        }
        if (at >= json.size()) {
            return false;
        }
        ++at;
        return true;
    };
    const auto skipSpace = [&](size_t& at) {
        while (at < json.size() && (json[at] == ' ' || json[at] == '\t' || json[at] == '\n' || json[at] == '\r')) {
            ++at;
        }
    };
    size_t at = json.find('{');
    if (at == std::string_view::npos) {
        return names;
    }
    ++at;
    while (at < json.size()) {
        skipSpace(at);
        if (at < json.size() && (json[at] == ',' || json[at] == '}')) {
            if (json[at] == '}') {
                break;
            }
            ++at;
            continue;
        }
        std::string name;
        if (!readString(at, name)) {
            break;
        }
        skipSpace(at);
        if (at >= json.size() || json[at] != ':') {
            break;
        }
        ++at;
        skipSpace(at);
        std::string hex;
        if (!readString(at, hex)) {
            break;
        }
        if (hex.size() != 8) {
            continue;
        }
        uint32_t bits = 0;
        bool ok = true;
        for (const char ch : hex) {
            const int digit = ch >= '0' && ch <= '9'   ? ch - '0'
                              : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10
                              : ch >= 'A' && ch <= 'F' ? ch - 'A' + 10
                                                       : -1;
            if (digit < 0) {
                ok = false;
                break;
            }
            bits = (bits << 4) | static_cast<uint32_t>(digit);
        }
        if (ok) {
            names[name] = bits;
        }
    }
    return names;
}

std::string cryptomatteManifest(const std::map<std::string, uint32_t>& names) {
    std::string out = "{";
    bool first = true;
    for (const auto& [name, id] : names) {
        out += first ? "\"" : ",\"";
        // A path holds no quote or backslash of its own, but a name that did
        // would break the object, so both are escaped as JSON escapes them.
        for (const char ch : name) {
            if (ch == '"' || ch == '\\') {
                out.push_back('\\');
            }
            out.push_back(ch);
        }
        out += "\":\"" + hex8(id) + "\"";
        first = false;
    }
    out += "}";
    return out;
}

}   // namespace athenea::core
