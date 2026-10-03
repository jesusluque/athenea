// Copyright (c) 2026 jesus luque.
//
// The hash Cryptomatte names things by. A matte's id is MurmurHash3_x86_32 of
// the object's name, read as the bits of a float32 -- with the exponent moved
// off 0 and 255 first, so the value is never a NaN, an infinity or a
// denormal, which an image pipeline would not carry intact. The manifest an
// EXR carries maps each name to the hex of exactly those bits. This is the
// specification's `mm3hash_float` and `id_to_hex_str` (Cryptomatte 1.2.0).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

namespace athenea::core {

/// MurmurHash3, the x86 32-bit variant, of `key` with `seed`.
[[nodiscard]] uint32_t murmurHash3x86_32(std::string_view key, uint32_t seed = 0) noexcept;

/// A hash made safe to carry as a float: an exponent of all zeros or all ones
/// has its lowest bit flipped, as Cryptomatte does.
[[nodiscard]] uint32_t cryptomatteFixup(uint32_t hash) noexcept;

/// The id Cryptomatte gives a name: the hash, fixed up. The bits of the float
/// the image carries, and the number the manifest writes in hex.
[[nodiscard]] uint32_t cryptomatteId(std::string_view name) noexcept;

/// Eight lowercase hex digits of `bits`, zero padded.
[[nodiscard]] std::string hex8(uint32_t bits);

/// The key a layer's metadata is filed under, `cryptomatte/<key>/...`: the
/// first seven hex digits of the raw hash of the layer's name.
[[nodiscard]] std::string cryptomatteLayerKey(std::string_view layer);

/// A manifest as the format writes one, `{"<name>":"<hex8>", ...}`, back into
/// names and ids. A pair whose hex is not eight hex digits is skipped: a
/// manifest is read, never trusted. Escapes are taken as the JSON ones a path
/// can hold (`\\`, `\"`, `\/`).
[[nodiscard]] std::map<std::string, uint32_t> parseCryptomatteManifest(std::string_view json);

/// The other direction: the manifest string for `names`, in the order a
/// `std::map` holds them.
[[nodiscard]] std::string cryptomatteManifest(const std::map<std::string, uint32_t>& names);

}   // namespace athenea::core
