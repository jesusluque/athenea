// Copyright (c) 2026 jesus luque.
//
// What the .athc reader and the .athz reader share inside the module: a
// streaming store of chunk slots, made empty.
#pragma once

#include <cstdint>
#include <string>

#include "athenea/core/Result.h"
#include "athenea/lod/ChunkSource.h"
#include "athenea/lod/Lod.h"

namespace athenea::lod::detail {

/// A store of `slots` chunks of `format`, and the per-chunk flags, all empty.
/// `count` is the cloud's splats; `chunks` its chunks.
[[nodiscard]] Result<void> makeStore(gpu::Device& device, LodCloud& lod, const ChunkFormat& format, uint64_t count,
                                     uint32_t chunks, uint32_t slots, const scene::Bounds& bounds,
                                     const std::string& name);

}   // namespace athenea::lod::detail
