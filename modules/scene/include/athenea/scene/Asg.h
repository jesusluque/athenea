// Copyright (c) 2026 jesus luque.
//
// ANCHORED SURFEL GAUSSIANS (`.asg`, 3Dspace's bitstream v1, proposal 002):
// a cloud as anchors -- a frame, a centre, ranges, a material, a Cryptomatte
// id, a transfer mean and a shadow pattern each -- and leaves of a few bits a
// gaussian read against their anchor. The CPU reads the header and works out
// where the anchor record's fields are (they depend on its flags), then hands
// the whole file to the device; the decode is a kernel, a thread a gaussian
// (scene/asg_decode.slang), into float records as io::SplatEncoding describes
// them and the arrays a TX cloud carries beside them.
#pragma once

#include <cstdint>
#include <filesystem>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/io/RawSplats.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::scene {

struct AsgCloud {
    uint32_t          count = 0;       ///< gaussians, in the file's order (grouped by anchor)
    uint32_t          anchors = 0;
    io::SplatEncoding encoding;        ///< of `records`
    gpu::Buffer       records;         ///< count * encoding.floatsPerRecord floats
    gpu::Buffer       crypto;          ///< uint a gaussian: its anchor's Cryptomatte id
    gpu::Buffer       thinWalled;      ///< int a gaussian, from its anchor's material
    uint32_t          directCount = 0; ///< transfer values a gaussian, direct block (0: none)
    uint32_t          indirectCount = 0;
    gpu::Buffer       direct;          ///< float * directCount
    gpu::Buffer       indirect;        ///< float * indirectCount
    uint32_t          shadowWords = 0; ///< 0: none
    gpu::Buffer       shadowBits;      ///< int * shadowWords, bytes as shadowBits primvars hold them
};

/// Whether `path` names an .asg file (by its extension).
[[nodiscard]] bool isAsg(const std::filesystem::path& path);

/// The file at `path` decoded on the device.
[[nodiscard]] Result<AsgCloud> decodeAsg(gpu::ShaderLibrary& library, const std::filesystem::path& path);

}   // namespace athenea::scene
