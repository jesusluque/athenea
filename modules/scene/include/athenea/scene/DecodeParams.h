// Copyright (c) 2026 jesus luque.
//
// DecodeParams (shaders/athenea/scene/splat_encoding.slang), set by name from an
// io::SplatEncoding. One place, for every kernel that reads raw splat records:
// the loader's validate and decode, and the USD export.
#pragma once

#include <cstdint>

#include <slang-rhi/shader-cursor.h>

#include "athenea/io/RawSplats.h"

namespace athenea::scene {

/// Fills `cursor["params"]`.
void setDecodeParams(rhi::ShaderCursor cursor, const io::SplatEncoding& encoding, uint32_t count,
                     uint32_t base, uint32_t keepPerColour, uint32_t shWords, uint32_t recordBase = 0);

}   // namespace athenea::scene
