// Copyright (c) 2026 jesus luque.
//
// An SPZ page in a staging slot: what a loader thread writes after it has
// decompressed the page (LiveSource::fetch), and what the frame thread reads
// back to record its expansion and decode (StreamingPool). The header is
// bookkeeping -- counts and offsets -- and the streams follow it at
// ChunkFormat::kSpzHeaderBytes, as io::readSpz laid them out.
#pragma once

#include <cstdint>

#include "athenea/lod/ChunkSource.h"

namespace athenea::lod {

struct SpzStaged {
    static constexpr uint32_t kMagic = 0x5A50534C;   // "LSPZ"
    uint32_t magic = kMagic;
    uint32_t count = 0;
    uint32_t shDim = 0;
    uint32_t fractionalBits = 0;
    uint32_t smallestThree = 0;
    uint32_t flipYZ = 0;
    uint32_t antialiased = 0;
    uint32_t pad = 0;
    uint64_t offsets[6] = {};   // positions, alphas, colours, scales, rotations, sh: from the streams' start
};
static_assert(sizeof(SpzStaged) <= ChunkFormat::kSpzHeaderBytes);

}   // namespace athenea::lod
