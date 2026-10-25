// Copyright (c) 2026 jesus luque.
//
// .athz: a cloud with its levels of detail in pages a GPU decodes, the
// coding on the device both ways (shaders/athenea/lod/athz_*.slang; research
// proposal 101, docs/decisions.md ".athz"). What it decodes to is what a
// .athc holds -- each merged level's groups and the cloud's own splats, the
// record a pool slot holds -- bit for bit: rendering does not change, only
// the bytes on disk and on the wire.
//
//   [0]      header, 64 bytes: 'ATHZ', version, flags, layers, counts, the
//            tables' offsets, a CRC-32 of itself
//   [64]     extensions: (u16 type, u16 flags, u32 length, payload)..., a
//            type 0 ends them; a reader skips the types it does not know
//   [4096]   asset table: the cloud's format and bounds, its levels, each
//            plane's coding (raw width or rANS table), the rANS tables'
//            frequencies, where each layer's section is
//   [..]     page table, 64 bytes a page: elements, band, octree level, the
//            page it hangs from, its box (f16, rounded outwards), its error,
//            each layer's offset and bytes in its section, a CRC-32
//   [4096k]  the finest merged level's group starts (raw)
//   [4096k]  sections G, A, T: the pages' coded bytes, a layer each, in band
//            order -- the coarsest merged level's pages first, the cloud's
//            own splats last -- so the first bands of every section are a
//            fixed cut by themselves (writeAthzPrefix)
//
// Each word of an element is coded after a transform the writer chose for it
// (athz_common.slang: a difference from the run's previous element or from the
// parent group, by words or by halves, an exclusive or, or against the
// coefficient before it), then split into byte planes, each raw or rANS.
// A page is `pageRuns` runs of `runGaussians` elements of one band; a layer
// of it is the run ends (u32 each), then each run: where a layer has planes
// coded by rANS, the stream's length in 16-bit words, the 32 lanes' states
// and the words; then its raw planes, bit-packed.
//
// Little-endian; every CRC is zlib's CRC-32 over the bytes as stored, which
// the reader checks before a page reaches the device: a corrupt or cut page
// is refused, never decoded.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "athenea/lod/ChunkSource.h"
#include "athenea/lod/Lod.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::lod {

/// A page's layers, as bits (`layerMask`, `readAthz`).
inline constexpr uint32_t kAthzGeometry = 1;     ///< G: position, shape but the colour, the tail word, normal, material
inline constexpr uint32_t kAthzAppearance = 2;   ///< A: the base colour, harmonics, emission
inline constexpr uint32_t kAthzTransfer = 4;     ///< T: transfer and its open directions
inline constexpr uint32_t kAthzTemporal = 8;     ///< K: none written yet
inline constexpr uint32_t kAthzAllLayers = 15;

/// LAYER K: the rig that moves a skinned cloud, once -- each splat's four
/// (joint, weight) float pairs and their gradients in the store's order, as
/// scene::SplatSkinner takes them -- and its joints at each frame, written in
/// halves (each matrix's first three columns; the fourth is 0 0 0 1).
struct AthzTemporal {
    const gpu::Buffer*    influences = nullptr;   ///< 8 words a store splat
    const gpu::Buffer*    gradients = nullptr;    ///< 3 words a store splat, or null
    uint32_t              joints = 0;
    std::vector<double>   times;
    std::vector<float>    xforms;                 ///< times * joints * 16, row major
    std::array<float, 16> geomBind{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

struct AthzSettings {
    /// Elements a run: 32, 64, 128 or 256 (a lane's symbols a plane are
    /// runGaussians / 32).
    uint32_t runGaussians = 256;
    /// Runs a page. A page is runGaussians * pageRuns elements, and the
    /// streaming unit: one chunk of a pool.
    uint32_t pageRuns = 12;
    /// log2 of a rANS table's size, 9 to 12 (12 fills WebGPU's 16 KiB of
    /// workgroup memory; 11 is what athz_decode11 decodes there).
    uint32_t tableLog = 11;
    /// false: every plane raw bit-packed (encoding 0).
    bool rans = true;
    /// Every plane raw at 8 bits from 0: the record's bytes transposed into
    /// planes and nothing more (the census's byte planes). Implies !rans.
    bool bytePlanes = false;
    /// THE TRANSFORMS a word may take before it is coded (athz_common.slang),
    /// a bit each: 1 none, 2 delta, 4 delta by halves, 8 xor, 16 parent, 32
    /// parent by halves, 64 xor with the parent, 128 each half against the
    /// coefficient before it, 256 the same on the differences from the
    /// parent. Each word takes the one whose planes measure least. 1: order 0
    /// on the words as they are.
    uint32_t transforms = 0x1FF;
    /// A LOSSY LAYER T: where set (a clustered PCA basis, f32, 256 clusters
    /// and 8 axes a block, as `athenea decimate --transfer-basis` takes), a
    /// TX transfer of 112 values is coded as its cluster id and 24 weights of
    /// 4 bits (transfer_codec.slang's codec), the basis kept once in the file;
    /// the decoder writes the transfer back from them. Empty: lossless.
    std::filesystem::path transferBasis;
    /// Layer K, where the cloud is skinned (null: none).
    const AthzTemporal* temporal = nullptr;
    /// The census: each plane's order-1 entropy under two contexts, measured.
    bool measureContexts = false;
    /// Not 0: the pages' histograms are counted in an order shuffled by this
    /// seed (a check that the tables do not depend on it).
    uint64_t histogramShuffle = 0;
    /// What the file says it was made from (an extension).
    std::string source;
};

/// What a plane cost (the census).
struct AthzPlaneStat {
    uint32_t kind = 0;       ///< 0 the merged levels, 1 the cloud's own splats
    uint32_t byte = 0;       ///< within an element
    uint32_t layer = 0;      ///< 0 G, 1 A, 2 T, 3 K
    uint32_t encoding = 0;   ///< 0 raw, 1 rANS
    uint32_t width = 0;      ///< bits a symbol, raw
    uint64_t symbols = 0;    ///< elements coded
    uint64_t rawBytes = 0;   ///< at `width`
    uint64_t ransBytes = 0;  ///< what its table's entropy says rANS takes (0: not tried)
    uint32_t transform = 0;  ///< its word's (AthzSettings::transforms' bit index)
    /// measureContexts: its residuals at their order-0 entropy, and under
    /// order 1 with the top two bits of the run's previous element's symbol,
    /// or of the element's previous byte, as context.
    uint64_t order0Bytes = 0;
    uint64_t contextElementBytes = 0;
    uint64_t contextByteBytes = 0;
};

struct AthzStats {
    uint64_t fileBytes = 0;
    std::array<uint64_t, 4> layerBytes{};   ///< each section
    uint64_t tableBytes = 0;    ///< header, asset table and page table
    uint64_t startsBytes = 0;
    uint32_t pages = 0;
    uint32_t elementBytes = 0;
    std::array<uint64_t, 2> elements{};     ///< merged groups, splats
    std::vector<AthzPlaneStat> planes;
};

/// Writes a cloud whose chunks are all on the device in order (slot c
/// chunk c, as a build or readAthc leaves it). Coded on the device; the
/// CPU frames and writes the bytes.
[[nodiscard]] Result<AthzStats> writeAthz(gpu::ShaderLibrary& library, const LodCloud& cloud,
                                          const std::filesystem::path& path, const AthzSettings& settings = {});

[[nodiscard]] bool isAthz(const std::filesystem::path& path);

/// The whole file onto the device, `layers` of it (G always; a layer left out
/// is zero): every page the file holds in its slot, chunk c in slot c. A file
/// cut to a prefix of its bands that keeps every merged level reads as a
/// cloud whose missing chunks are not resident.
[[nodiscard]] Result<LodCloud> readAthz(gpu::ShaderLibrary& library, const std::filesystem::path& path,
                                        uint32_t layers = kAthzAllLayers);

/// A FIXED CUT: the finest band whose pages are all in the file -- the
/// cloud's own splats, or else a merged level -- as a cloud by itself.
[[nodiscard]] Result<scene::GpuSplats> readAthzCut(gpu::ShaderLibrary& library, const std::filesystem::path& path,
                                                   uint32_t* band = nullptr);

/// The first `bands` bands of `in` (merged levels from the coarsest, then
/// the splats) as a file of their own: bytes copied, the tables cut short.
[[nodiscard]] Result<void> writeAthzPrefix(const std::filesystem::path& in, const std::filesystem::path& out,
                                           uint32_t bands);

/// A SKINNED .athz READ BACK: the cloud's own splats in their rest pose (the
/// store's order), layer K's influences beside them, and every frame's
/// joints in halves on the device; `athzFrameXforms` writes one frame's as
/// the skinner's floats.
struct AthzSkinned {
    scene::GpuSplats      rest;
    gpu::Buffer           influences;
    gpu::Buffer           gradients;   ///< invalid where the file has none
    uint32_t              joints = 0;
    std::vector<double>   times;
    gpu::Buffer           halves;
    std::array<float, 16> geomBind{};
};
[[nodiscard]] Result<AthzSkinned> readAthzSkinned(gpu::ShaderLibrary& library, const std::filesystem::path& path);
[[nodiscard]] Result<void> athzFrameXforms(gpu::ShaderLibrary& library, const AthzSkinned& skinned, uint32_t frame,
                                           gpu::Buffer& xforms);
/// A file's per-gaussian words (`words` a record) in the store's order:
/// `order` a store splat's splat of the decoded cloud, `origin` (or null) a
/// decoded splat's record.
[[nodiscard]] Result<gpu::Buffer> athzGatherToStore(gpu::ShaderLibrary& library, const gpu::Buffer& fileWords,
                                                    uint32_t words, const gpu::Buffer& order,
                                                    const gpu::Buffer* origin, uint32_t count);

/// THE PREFIX THAT FITS: the most pages, in band order, whose file is at
/// most `maxBytes` -- a fixed cut of that size. What it holds.
struct AthzPrefix {
    uint32_t pages = 0;
    uint64_t bytes = 0;
    uint64_t merged = 0;       ///< merged gaussians in its pages
    uint64_t splats = 0;       ///< the cloud's own splats in its pages
    bool     levelsWhole = false;   ///< every merged level is in it: it streams as a cloud
};
[[nodiscard]] Result<AthzPrefix> writeAthzPrefixBytes(const std::filesystem::path& in, const std::filesystem::path& out,
                                                      uint64_t maxBytes);

/// E1: how long the device takes to decode the file's splat pages (all
/// layers), in dispatches of at most `batchBytes` coded bytes, and in
/// dispatches of one page for the first `singlePages`. GPU time where the
/// backend gives it (gpu::GpuTimer). Each batch is checked: a corrupt run
/// fails it.
struct AthzDecodeTiming {
    uint32_t pages = 0;
    uint64_t codedBytes = 0;
    uint64_t decodedBytes = 0;
    double   milliseconds = 0.0;
    bool     single = false;      ///< one page alone
    bool     fromDevice = false;  ///< the GPU's clock (else the host's)
};
[[nodiscard]] Result<std::vector<AthzDecodeTiming>> timeAthzDecode(gpu::ShaderLibrary& library,
                                                                   const std::filesystem::path& path,
                                                                   uint64_t batchBytes, uint32_t singlePages);

/// The census's byte planes: for a file written with `bytePlanes`, plane b
/// of kind k (0 merged, 1 splats) at [k][b], every run's bytes of it one
/// after another (bytes copied out of the runs).
[[nodiscard]] Result<std::array<std::vector<std::vector<uint8_t>>, 2>> athzBytePlanes(
    const std::filesystem::path& path);

/// EXPORT: a cloud's own records as an SPZ file of `version` (3 gzip, 4
/// ZSTD where the build has it), every number worked out on the device --
/// the record turned into the flattened one (athz_check's athzFlat) and
/// packed by render::FlattenPack into SPZ's streams at 8 bits a harmonic --
/// and only framed and compressed here (io::writeSpz). Degree 3 streams,
/// the bands past the cloud's own zero. A cloud in linear light has its
/// base colour written as sRGB, which SPZ assumes; exact at degree 0.
[[nodiscard]] Result<void> writeSpzOfCloud(gpu::ShaderLibrary& library, const scene::GpuSplats& cloud,
                                           const std::filesystem::path& path, uint32_t version = 3);

/// A .athz's pages as a ChunkSource: the header and tables parsed, the file
/// mapped; the merged levels decoded on the device at openCloud, the splat
/// pages staged as they are coded and decoded by the device a batch at a
/// time (`decode`). A loader checks a page's CRC before it is staged.
class AthzSource final : public ChunkSource {
public:
    /// Any thread: reads the tables and maps the file. `layers` as readAthz's.
    [[nodiscard]] static Result<std::unique_ptr<AthzSource>> open(gpu::ShaderLibrary& library,
                                                                  const std::filesystem::path& path,
                                                                  uint32_t layers = kAthzAllLayers);
    ~AthzSource() override;

    [[nodiscard]] const ChunkFormat& format() const noexcept override;
    [[nodiscard]] uint32_t chunkCount() const noexcept override;
    [[nodiscard]] bool available(uint32_t chunk) const noexcept override;
    [[nodiscard]] scene::Bounds bounds(uint32_t chunk) const override;
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] Result<LodCloud> openCloud(gpu::Device& device) override;
    [[nodiscard]] ChunkRequest request(uint32_t chunk) const override;
    [[nodiscard]] Result<void> fetch(const ChunkRequest& request, std::span<std::byte> staging) override;
    [[nodiscard]] Result<void> decode(gpu::CommandBatch& batch, const gpu::Buffer& staging,
                                      std::span<const StagedChunk> chunks, LodCloud& cloud) override;

private:
    AthzSource();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace athenea::lod
