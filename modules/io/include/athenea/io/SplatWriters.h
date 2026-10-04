// Copyright (c) 2026 jesus luque.
//
// STANDARD GAUSSIAN SPLATTING FILES, WRITTEN (`athenea flatten`, task TXF).
// What reaches these functions is already every number the file holds, worked
// out on the device (shaders/athenea/splat/flatten_pack.slang); here they are
// framed -- a header, a container, a compression -- and written. ZSTD and
// gzip are byte coding, input and output, and nothing else is done to a value.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "athenea/core/Result.h"

namespace athenea::io {

/// A 3DGS PLY, binary little endian, INRIA's layout: x y z nx ny nz,
/// f_dc_0..2, f_rest_0..44 (all red, then green, then blue), opacity (a
/// logit), scale_0..2 (ln sigma), rot_0..3 (w x y z) -- 62 floats a gaussian.
/// `chunks` are written in order, one after another; `comments` go in the
/// header as `comment` lines.
[[nodiscard]] Result<void> writePly3dgs(const std::filesystem::path& path,
                                        std::span<const std::vector<float>> chunks,
                                        std::span<const std::string> comments);

/// What an SPZ header says.
struct SpzHeader {
    uint32_t version = 3;          ///< 3: gzip, one stream (Spark reads it); 4: ZSTD, six
    uint32_t count = 0;
    uint32_t shDegree = 3;
    uint32_t fractionalBits = 12;
    bool     antialiased = true;
};

/// An SPZ file of the six streams Niantic's format lays out -- positions (24
/// bit fixed point), alphas, colours, scales, rotations (smallest three),
/// harmonics -- already packed. Version 3 is gzip around one body, version 4
/// ZSTD a stream after a table of contents (third_party/spz's layout).
[[nodiscard]] Result<void> writeSpz(const std::filesystem::path& path, const SpzHeader& header,
                                    const std::array<std::vector<uint8_t>, 6>& streams);

/// Whether this build writes SPZ version 4 (it needs ZSTD).
[[nodiscard]] bool writesSpzVersion4() noexcept;

/// A glTF binary of one POINTS primitive with KHR_gaussian_splatting:
/// POSITION, KHR_gaussian_splatting:ROTATION (x y z w), :SCALE (linear),
/// :OPACITY (linear) and :SH_DEGREE_l_COEF_n for l to `degree`, all float.
/// `chunks` are clouds, each its blocks in that order (position 3, rotation 4,
/// scale 3, opacity 1, then each harmonic 3), each block as long as the
/// cloud; the attribute is the clouds' blocks one after another.
struct GlbGaussians {
    std::vector<uint32_t>           counts;   ///< gaussians of each chunk
    std::vector<std::vector<float>> chunks;
    std::array<float, 3>            min{};    ///< the positions' box, which glTF wants on POSITION
    std::array<float, 3>            max{};
    uint32_t                        degree = 3;
    std::string                     colorSpace = "srgb_rec709_display";
    std::string                     generator = "athenea flatten";
};
[[nodiscard]] Result<void> writeGlbGaussians(const std::filesystem::path& path, const GlbGaussians& gaussians);

}   // namespace athenea::io
