// Copyright (c) 2026 jesus luque.
//
// `.athv`: A VIDEO OF GAUSSIAN SPLATS AND WHAT GOES BESIDE IT. One file, a
// header and then a record a frame, each record whole in itself -- what a
// network sender cuts into packets and a receiver puts back:
//
//   header   "ATHV", version, the codec's FourCC, the plane layout (bits,
//            rest coefficients, chunk size, columns, whether the position high
//            parts are planes), the rate, the source's antialiasing flag
//   picture  "ATVF", its index, the source frame it belongs to and where in
//            it its splats start (a frame too big for one picture is cut
//            into slices, Planes.h), splats, key or not, PTP time (ns), time code
//            and timecode label, the frame's ranges (two floats an
//            attribute), the position high parts deflated (unless they are
//            planes), the codec's parameter sets on a key frame, and the
//            access unit: NAL units, each after its 4-byte big-endian length
//
// Little endian; every count and length is checked against the file before
// anything is read past it. The CPU reads, writes and inflates; the values
// are the device's (Planes.h).
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "athenea/core/Result.h"

namespace athenea::video {

struct AthvHeader {
    static constexpr uint32_t kMagic = 0x56485441;   // "ATHV"
    static constexpr uint32_t kVersion = 1;
    uint32_t codec = 0x31637668;   // 'hvc1'
    uint32_t bits = 8;
    uint32_t restPerColour = 0;
    uint32_t chunkSplats = 0;
    uint32_t columns = 0;
    uint32_t maxSplats = 0;        ///< the most a picture holds: what its tiles are sized for
    bool     hiInVideo = false;
    bool     antialiased = false;
    double   rate = 30.0;
};

struct AthvFrame {
    uint32_t                          index = 0;       ///< the picture's, in the file
    uint32_t                          source = 0;      ///< the source frame it is a slice of
    uint32_t                          firstSplat = 0;  ///< where in that frame its splats start
    uint32_t                          sourceSplats = 0;///< the source frame's splats
    uint32_t                          splats = 0;      ///< the picture's
    bool                              keyframe = false;
    uint64_t                          ptpTime = 0;
    double                            time = 0.0;      ///< the source's time code
    std::string                       timecode;        ///< HH:MM:SS:FF, a label
    std::vector<float>                ranges;
    std::vector<std::byte>            high;            ///< inflated
    uint64_t                          highStored = 0;  ///< bytes the high parts took in the file
    std::vector<std::vector<uint8_t>> parameterSets;   ///< VPS, SPS, PPS on a key frame
    std::vector<uint8_t>              unit;            ///< the access unit, length-prefixed NALs
    uint64_t                          recordBytes = 0; ///< the whole record in the file
};

class AthvWriter {
public:
    [[nodiscard]] static Result<std::unique_ptr<AthvWriter>> create(const std::filesystem::path& path,
                                                                   const AthvHeader& header);
    ~AthvWriter();
    /// Deflates `frame.high` and appends the record; `frame.highStored` and
    /// `frame.recordBytes` say what it took.
    [[nodiscard]] Result<void> write(AthvFrame& frame);
    /// The header again, with the most splats seen; the file closed and
    /// given its name (until then it is written under a partial one, and a
    /// writer dropped unfinished leaves nothing).
    [[nodiscard]] Result<void> finish(uint32_t maxSplats);

private:
    AthvWriter() = default;
    [[nodiscard]] Result<void> writeHeader();
    std::ofstream         file_;
    std::filesystem::path path_, partial_;
    AthvHeader            header_;
};

class AthvReader {
public:
    [[nodiscard]] static Result<std::unique_ptr<AthvReader>> open(const std::filesystem::path& path);
    ~AthvReader();
    [[nodiscard]] const AthvHeader& header() const noexcept { return header_; }
    /// Pictures in the file.
    [[nodiscard]] uint32_t frames() const noexcept { return static_cast<uint32_t>(offsets_.size()); }
    /// Source frames, and the pictures of each (its slices, in order).
    [[nodiscard]] uint32_t sourceFrames() const noexcept { return static_cast<uint32_t>(slices_.size()); }
    [[nodiscard]] const std::vector<uint32_t>& slicesOf(uint32_t source) const { return slices_.at(source); }
    /// The most splats a source frame holds.
    [[nodiscard]] uint32_t mostSourceSplats() const noexcept { return mostSource_; }
    /// Picture `k`, read and inflated. Any thread, one call at a time.
    [[nodiscard]] Result<AthvFrame> read(uint32_t k);
    /// Bytes the file holds.
    [[nodiscard]] uint64_t bytes() const noexcept { return size_; }
    /// Picture `k`'s time code and the bytes of its record, from the walk at open.
    [[nodiscard]] double time(uint32_t k) const noexcept { return k < times_.size() ? times_[k] : 0.0; }
    [[nodiscard]] uint64_t recordBytes(uint32_t k) const noexcept {
        return k >= offsets_.size() ? 0 : (k + 1 < offsets_.size() ? offsets_[k + 1] : size_) - offsets_[k];
    }

private:
    AthvReader() = default;
    std::ifstream         file_;
    std::string           name_;
    AthvHeader            header_;
    uint64_t              size_ = 0;
    std::vector<uint64_t> offsets_;
    std::vector<double>   times_;
    std::vector<std::vector<uint32_t>> slices_;
    uint32_t              mostSource_ = 0;
};

/// The access unit and, on a key frame, its parameter sets as an Annex B
/// stream (start codes), for a player or a probe to read.
[[nodiscard]] std::vector<uint8_t> annexB(const AthvFrame& frame);

}   // namespace athenea::video
