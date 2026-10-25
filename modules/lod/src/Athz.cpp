// Copyright (c) 2026 jesus luque.
#include "athenea/lod/Athz.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <numeric>
#include <optional>
#include <random>

#include <zlib.h>

#include "athenea/core/Log.h"
#include "athenea/core/Platform.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/GpuTimer.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/io/SplatWriters.h"
#include "athenea/render/Flatten.h"
#include "StoreDetail.h"

namespace athenea::lod {
namespace {

constexpr char     kMagic[4] = {'A', 'T', 'H', 'Z'};
constexpr uint16_t kVersion = 1;
constexpr uint64_t kPage = 4096;
constexpr uint32_t kHeaderBytes = 64;
constexpr uint32_t kEntryBytes = 64;
// Header flags (a bit this reader does not know is refused, as .athc's).
constexpr uint32_t kFlagTransfer = 1;    ///< layer T is there
constexpr uint32_t kFlagHarmonics = 2;   ///< layer A carries harmonics past the DC
constexpr uint32_t kFlagTemporal = 4;    ///< layer K is there
constexpr uint32_t kFlagLinear = 8;      ///< colours in linear light
constexpr uint32_t kFlagPredicted = 16;  ///< a plane is coded from its parent page (encoding 2): not read yet
constexpr uint32_t kFlagEmission = 32;   ///< elements carry RGB9E5 emission
constexpr uint32_t kKnownFlags = kFlagTransfer | kFlagHarmonics | kFlagTemporal | kFlagLinear | kFlagPredicted |
                                 kFlagEmission;
constexpr uint32_t kKinds = 2;           ///< 0 the merged levels' groups, 1 the cloud's own splats
constexpr uint32_t kNone = 0xFFFFFFFFu;
constexpr uint32_t kLanes = 32;
/// What the encoder's scratch may take at once.
constexpr uint64_t kScratchBytes = uint64_t{96} << 20;
/// What a read uploads at once.
constexpr uint64_t kReadBatchBytes = uint64_t{64} << 20;
constexpr uint32_t kTlvSource = 1;

uint64_t aligned(uint64_t at, uint64_t to) {
    return (at + to - 1) / to * to;
}

template <typename T>
void putAt(uint8_t* at, T v) {
    std::memcpy(at, &v, sizeof v);
}
template <typename T>
T getAt(const uint8_t* at) {
    T v{};
    std::memcpy(&v, at, sizeof v);
    return v;
}

uint32_t crcOf(const void* data, uint64_t bytes, uint32_t crc = 0) {
    const auto* p = static_cast<const Bytef*>(data);
    uLong c = crc;
    while (bytes > 0) {
        const auto n = static_cast<uInt>(std::min<uint64_t>(bytes, uint64_t{1} << 30));
        c = crc32(c, p, n);
        p += n;
        bytes -= n;
    }
    return static_cast<uint32_t>(c);
}

Error bad(const std::filesystem::path& path, const std::string& why) {
    return Error(ErrorCode::IoFailure, path.string() + ": not a readable .athz (" + why + ")");
}

// --- the header ---------------------------------------------------------------

struct Header {
    uint32_t flags = 0;
    uint32_t layerMask = 0;
    uint64_t gaussians = 0;
    uint32_t pages = 0;
    uint16_t runGaussians = 0;
    uint16_t pageRuns = 0;
    uint64_t assetOffset = 0;
    uint32_t assetBytes = 0;
    uint64_t pageTableOffset = 0;
    uint16_t entryBytes = kEntryBytes;
    uint32_t tlvOffset = 0;
};

std::array<uint8_t, kHeaderBytes> encodeHeader(const Header& h) {
    std::array<uint8_t, kHeaderBytes> b{};
    std::memcpy(b.data(), kMagic, 4);
    putAt<uint16_t>(b.data() + 4, kVersion);
    putAt<uint16_t>(b.data() + 6, kHeaderBytes);
    putAt<uint32_t>(b.data() + 8, h.flags);
    putAt<uint32_t>(b.data() + 12, h.layerMask);
    putAt<uint64_t>(b.data() + 16, h.gaussians);
    putAt<uint32_t>(b.data() + 24, h.pages);
    putAt<uint16_t>(b.data() + 28, h.runGaussians);
    putAt<uint16_t>(b.data() + 30, h.pageRuns);
    putAt<uint64_t>(b.data() + 32, h.assetOffset);
    putAt<uint32_t>(b.data() + 40, h.assetBytes);
    putAt<uint64_t>(b.data() + 44, h.pageTableOffset);
    putAt<uint16_t>(b.data() + 52, h.entryBytes);
    putAt<uint16_t>(b.data() + 54, 0);
    putAt<uint32_t>(b.data() + 56, h.tlvOffset);
    putAt<uint32_t>(b.data() + 60, crcOf(b.data(), 60));
    return b;
}

Result<Header> decodeHeader(std::span<const std::byte> bytes, const std::filesystem::path& path) {
    if (bytes.size() < kPage) {
        return bad(path, "shorter than its header");
    }
    const auto* b = reinterpret_cast<const uint8_t*>(bytes.data());
    if (std::memcmp(b, kMagic, 4) != 0) {
        return bad(path, "no ATHZ magic");
    }
    if (getAt<uint16_t>(b + 4) != kVersion) {
        return bad(path, "version " + std::to_string(getAt<uint16_t>(b + 4)) + ", this reads 1");
    }
    if (getAt<uint16_t>(b + 6) != kHeaderBytes || getAt<uint32_t>(b + 60) != crcOf(b, 60)) {
        return bad(path, "the header's CRC");
    }
    Header h;
    h.flags = getAt<uint32_t>(b + 8);
    h.layerMask = getAt<uint32_t>(b + 12);
    h.gaussians = getAt<uint64_t>(b + 16);
    h.pages = getAt<uint32_t>(b + 24);
    h.runGaussians = getAt<uint16_t>(b + 28);
    h.pageRuns = getAt<uint16_t>(b + 30);
    h.assetOffset = getAt<uint64_t>(b + 32);
    h.assetBytes = getAt<uint32_t>(b + 40);
    h.pageTableOffset = getAt<uint64_t>(b + 44);
    h.entryBytes = getAt<uint16_t>(b + 52);
    h.tlvOffset = getAt<uint32_t>(b + 56);
    if ((h.flags & ~kKnownFlags) != 0) {
        return bad(path, "unknown flag bits " + std::to_string(h.flags & ~kKnownFlags));
    }
    if ((h.flags & kFlagPredicted) != 0) {
        return bad(path, "planes coded from their parent page (encoding 2), which this version does not read");
    }
    if ((h.layerMask & kAthzGeometry) == 0 || (h.layerMask & ~uint32_t{kAthzAllLayers}) != 0) {
        return bad(path, "layers " + std::to_string(h.layerMask));
    }
    if (h.runGaussians == 0 || h.runGaussians % kLanes != 0 || h.runGaussians > 256 || h.pageRuns == 0 ||
        h.entryBytes != kEntryBytes) {
        return bad(path, "run and page sizes");
    }
    return h;
}

// --- the asset table ----------------------------------------------------------

struct LevelEntry {
    uint32_t level = 0;
    uint32_t groups = 0;
    uint32_t firstPage = 0;
    uint32_t pages = 0;
};

/// A plane's coding (athz_common.slang's uint4): encoding | width << 8 |
/// base << 16 | layer << 24, its table, its raw words' start in a run, the
/// encoder's estimate.
using PlaneEntry = std::array<uint32_t, 4>;

struct Asset {
    uint32_t planes = 0;   ///< a kind's: 4 * element words
    uint32_t tableLog = 11;
    uint32_t tables = 0;
    uint32_t count = 0;
    uint32_t finestGroups = 0;
    ChunkFormat format;    ///< what an element carries (payload Athz)
    float    boundsLo[3] = {0, 0, 0};
    float    extent = 0;
    scene::Bounds bounds;
    uint64_t assetId = 0;
    std::array<uint64_t, 4> sectionOffset{};
    std::array<uint64_t, 4> sectionBytes{};
    uint64_t startsOffset = 0;
    uint64_t startsBytes = 0;
    std::vector<LevelEntry> levels;
    std::vector<PlaneEntry> planeEntries;   ///< kKinds * planes
    std::vector<uint16_t>   frequencies;    ///< tables * 256
};

struct ByteWriter {
    std::vector<uint8_t> bytes;
    template <typename T>
    void put(T v) {
        const size_t at = bytes.size();
        bytes.resize(at + sizeof v);
        std::memcpy(bytes.data() + at, &v, sizeof v);
    }
};

struct ByteReader {
    const uint8_t* at;
    const uint8_t* end;
    bool           ok = true;
    template <typename T>
    T get() {
        T v{};
        if (static_cast<size_t>(end - at) < sizeof v) {
            ok = false;
            return v;
        }
        std::memcpy(&v, at, sizeof v);
        at += sizeof v;
        return v;
    }
};

uint64_t fnv1a(const uint8_t* data, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ data[i]) * 1099511628211ULL;
    }
    return h;
}

std::vector<uint8_t> encodeAsset(Asset& a) {
    const auto write = [&](uint64_t id) {
        ByteWriter w;
        w.put<uint32_t>(0);   // bytes, patched below
        w.put<uint32_t>(kKinds);
        w.put<uint32_t>(a.planes);
        w.put<uint32_t>(a.tableLog);
        w.put<uint32_t>(a.tables);
        w.put<uint32_t>(static_cast<uint32_t>(a.levels.size()));
        w.put<uint32_t>(a.count);
        w.put<uint32_t>(a.finestGroups);
        const ChunkFormat& f = a.format;
        w.put<uint32_t>(f.chunkSplats);
        w.put<uint32_t>(f.restPerColour);
        w.put<uint32_t>(f.shWords);
        w.put<uint32_t>(f.normals ? 1 : 0);
        w.put<uint32_t>(f.emission ? 1 : 0);
        w.put<uint32_t>(f.pbrWords);
        w.put<uint32_t>(f.lobesWords);
        w.put<uint32_t>(f.transferCount);
        w.put<uint32_t>(f.transferWords);
        w.put<uint32_t>(f.shadowWords);
        for (float v : a.boundsLo) w.put<float>(v);
        w.put<float>(a.extent);
        for (float v : a.bounds.min) w.put<float>(v);
        for (float v : a.bounds.max) w.put<float>(v);
        w.put<uint64_t>(id);
        for (uint64_t v : a.sectionOffset) w.put<uint64_t>(v);
        for (uint64_t v : a.sectionBytes) w.put<uint64_t>(v);
        w.put<uint64_t>(a.startsOffset);
        w.put<uint64_t>(a.startsBytes);
        for (const LevelEntry& l : a.levels) {
            w.put<uint32_t>(l.level);
            w.put<uint32_t>(l.groups);
            w.put<uint32_t>(l.firstPage);
            w.put<uint32_t>(l.pages);
        }
        for (const PlaneEntry& p : a.planeEntries) {
            for (uint32_t v : p) w.put<uint32_t>(v);
        }
        for (uint16_t v : a.frequencies) w.put<uint16_t>(v);
        while (w.bytes.size() % 64 != 0) w.put<uint8_t>(0);
        putAt<uint32_t>(w.bytes.data(), static_cast<uint32_t>(w.bytes.size()));
        return w.bytes;
    };
    if (a.assetId == 0) {
        const std::vector<uint8_t> anonymous = write(0);
        a.assetId = fnv1a(anonymous.data(), anonymous.size());
    }
    return write(a.assetId);
}

Result<Asset> decodeAsset(std::span<const std::byte> bytes, const Header& h, const std::filesystem::path& path) {
    if (h.assetOffset > bytes.size() || h.assetBytes > bytes.size() - h.assetOffset) {
        return bad(path, "asset table past the end");
    }
    const auto* base = reinterpret_cast<const uint8_t*>(bytes.data()) + h.assetOffset;
    ByteReader r{base, base + h.assetBytes};
    Asset a;
    const uint32_t size = r.get<uint32_t>();
    const uint32_t kinds = r.get<uint32_t>();
    a.planes = r.get<uint32_t>();
    a.tableLog = r.get<uint32_t>();
    a.tables = r.get<uint32_t>();
    const uint32_t levels = r.get<uint32_t>();
    a.count = r.get<uint32_t>();
    a.finestGroups = r.get<uint32_t>();
    ChunkFormat& f = a.format;
    f.payload = ChunkFormat::Payload::Athz;
    f.chunkSplats = r.get<uint32_t>();
    f.restPerColour = r.get<uint32_t>();
    f.shWords = r.get<uint32_t>();
    f.normals = r.get<uint32_t>() != 0;
    f.emission = r.get<uint32_t>() != 0;
    f.pbrWords = r.get<uint32_t>();
    f.lobesWords = r.get<uint32_t>();
    f.transferCount = r.get<uint32_t>();
    f.transferWords = r.get<uint32_t>();
    f.shadowWords = r.get<uint32_t>();
    f.linear = (h.flags & kFlagLinear) != 0;
    for (float& v : a.boundsLo) v = r.get<float>();
    a.extent = r.get<float>();
    for (float& v : a.bounds.min) v = r.get<float>();
    for (float& v : a.bounds.max) v = r.get<float>();
    a.assetId = r.get<uint64_t>();
    for (uint64_t& v : a.sectionOffset) v = r.get<uint64_t>();
    for (uint64_t& v : a.sectionBytes) v = r.get<uint64_t>();
    a.startsOffset = r.get<uint64_t>();
    a.startsBytes = r.get<uint64_t>();
    if (!r.ok || size != h.assetBytes || kinds != kKinds || levels == 0 || levels > 30 || a.count == 0 ||
        f.shWords == 0 || f.shWords > 64 || a.planes != 4 * (f.elementBytes() / 4) || a.tableLog < 9 ||
        a.tableLog > 12 || f.chunkSplats != uint32_t{h.runGaussians} * h.pageRuns || h.gaussians != a.count) {
        return bad(path, "inconsistent asset table");
    }
    a.levels.resize(levels);
    for (LevelEntry& l : a.levels) {
        l.level = r.get<uint32_t>();
        l.groups = r.get<uint32_t>();
        l.firstPage = r.get<uint32_t>();
        l.pages = r.get<uint32_t>();
    }
    a.planeEntries.resize(uint64_t{kKinds} * a.planes);
    for (PlaneEntry& p : a.planeEntries) {
        for (uint32_t& v : p) v = r.get<uint32_t>();
    }
    a.frequencies.resize(uint64_t{a.tables} * 256);
    for (uint16_t& v : a.frequencies) v = r.get<uint16_t>();
    if (!r.ok) {
        return bad(path, "asset table shorter than its counts");
    }
    // Every table fills its 2^tableLog slots, and every plane names its own.
    for (uint32_t t = 0; t < a.tables; ++t) {
        uint32_t sum = 0;
        for (uint32_t s = 0; s < 256; ++s) sum += a.frequencies[t * 256 + s];
        if (sum != (1u << a.tableLog)) {
            return bad(path, "rANS table " + std::to_string(t) + " does not fill its slots");
        }
    }
    for (const PlaneEntry& p : a.planeEntries) {
        const uint32_t encoding = p[0] & 0xFF, width = (p[0] >> 8) & 0xFF, layer = p[0] >> 24;
        if (encoding > 1 || width > 8 || layer > 3 || (encoding == 1 && p[1] >= a.tables) ||
            ((h.layerMask >> layer) & 1u) == 0) {
            return bad(path, "a plane's coding");
        }
    }
    uint32_t page = 0;
    for (size_t l = 0; l < a.levels.size(); ++l) {
        const LevelEntry& e = a.levels[l];
        if (e.groups == 0 || e.firstPage != page || e.pages != (e.groups + f.chunkSplats - 1) / f.chunkSplats ||
            (l > 0 && e.level != a.levels[l - 1].level + 1)) {
            return bad(path, "level " + std::to_string(l));
        }
        page += e.pages;
    }
    if (a.levels.back().groups != a.finestGroups || a.startsBytes != uint64_t{a.finestGroups} * 4 ||
        a.startsOffset > bytes.size() || a.startsBytes > bytes.size() - a.startsOffset) {
        return bad(path, "the finest level's starts");
    }
    for (uint32_t L = 0; L < 4; ++L) {
        if (a.sectionOffset[L] > bytes.size() || a.sectionBytes[L] > bytes.size() - a.sectionOffset[L]) {
            return bad(path, "section " + std::to_string(L) + " past the end: a file cut short");
        }
    }
    return a;
}

// --- what an element is ---------------------------------------------------------

/// Words of each array an element carries, in its order (athz_common's fields).
std::array<uint32_t, 10> fieldWords(const ChunkFormat& f) {
    return {4, 4, f.shWords, 1, f.normals ? 1u : 0u, f.emission ? 1u : 0u, f.pbrWords, f.lobesWords, f.transferWords,
            f.shadowWords};
}

/// The layer of an element's byte `b` (0 G, 1 A, 2 T): position, rotation,
/// scales, the tail word, normal and material are geometry; the base colour
/// (shape word 2's high half and word 3), harmonics and emission appearance;
/// the transfer and its open directions T.
uint32_t layerOfByte(const ChunkFormat& f, uint32_t b) {
    uint32_t w = b / 4;
    const uint32_t inWord = b % 4;
    const std::array<uint32_t, 10> words = fieldWords(f);
    uint32_t field = 0;
    for (; field < 10; ++field) {
        if (w < words[field]) break;
        w -= words[field];
    }
    switch (field) {
    case 0: return 0;
    case 1: return (w < 2 || (w == 2 && inWord < 2)) ? 0u : 1u;
    case 2: return 1;
    case 3: return 0;
    case 4: return 0;
    case 5: return 1;
    case 6: return 0;
    case 7: return 0;
    default: return 2;
    }
}

/// The format of a cloud's elements, as writeAthc decides it: normals,
/// emission, material and transfer where every level carries them as the
/// splats do.
ChunkFormat formatOfCloud(const LodCloud& cloud, uint32_t pageGaussians) {
    using Member = gpu::Buffer scene::GpuSplats::*;
    const auto everywhere = [&](Member m) {
        return (cloud.splats.*m).valid() &&
               std::all_of(cloud.levels.begin(), cloud.levels.end(), [&](const LodLevel& l) { return (l.gaussians.*m).valid(); });
    };
    ChunkFormat f;
    f.payload = ChunkFormat::Payload::Athz;
    f.chunkSplats = pageGaussians;
    f.restPerColour = cloud.splats.restPerColour;
    f.shWords = cloud.splats.shWords;
    f.linear = cloud.splats.linear;
    f.normals = everywhere(&scene::GpuSplats::normals);
    f.emission = everywhere(&scene::GpuSplats::emission);
    if (everywhere(&scene::GpuSplats::pbr)) {
        f.pbrWords = 1;
        f.lobesWords = everywhere(&scene::GpuSplats::lobes) ? 3 : 0;
    }
    if (everywhere(&scene::GpuSplats::transfer) && cloud.splats.transferWords > 0) {
        f.transferCount = cloud.splats.transferCount;
        f.transferWords = cloud.splats.transferWords;
        f.shadowWords = everywhere(&scene::GpuSplats::shadowBits) ? cloud.splats.shadowWords : 0;
    }
    return f;
}

/// An element's arrays, as buffers (the tail apart from the cloud's).
struct Arrays {
    std::array<const gpu::Buffer*, 10> field{};
};

Arrays arraysOf(const scene::GpuSplats& s, const gpu::Buffer& tail) {
    Arrays a;
    a.field = {&s.positions, &s.shape, &s.sh, &tail, &s.normals, &s.emission, &s.pbr, &s.lobes, &s.transfer,
               &s.shadowBits};
    return a;
}

constexpr std::array<const char*, 10> kSrcNames = {"srcPositions", "srcShape", "srcSh", "srcTail", "srcNormals",
                                                   "srcEmission", "srcPbr", "srcLobes", "srcTransfer", "srcShadow"};
constexpr std::array<const char*, 10> kDstNames = {"dstPositions", "dstShape", "dstSh", "dstTail", "dstNormals",
                                                   "dstEmission", "dstPbr", "dstLobes", "dstTransfer", "dstShadow"};

void setLayout(rhi::ShaderCursor p, const ChunkFormat& f) {
    p["shWords"].setData(f.shWords);
    p["normals"].setData(f.normals ? 1u : 0u);
    p["emission"].setData(f.emission ? 1u : 0u);
    p["pbrWords"].setData(f.pbrWords);
    p["lobesWords"].setData(f.lobesWords);
    p["transferWords"].setData(f.transferWords);
    p["shadowWords"].setData(f.shadowWords);
    p["words"].setData(static_cast<uint32_t>(f.elementBytes() / 4));
}

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t bytes, uint32_t element, const char* label,
                                 const void* initial = nullptr, gpu::Memory memory = gpu::Memory::Device) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(bytes, 4);
    desc.elementBytes = element;
    desc.label = label;
    desc.memory = memory;
    return gpu::Buffer::create(device, desc, initial);
}

Result<gpu::Buffer> zeros(gpu::Device& device, uint64_t words, const char* label) {
    const std::vector<uint32_t> z(std::max<uint64_t>(words, 1), 0u);
    return deviceBuffer(device, z.size() * 4, 4, label, z.data());
}

/// A cloud's arrays for `n` elements of format `f`, empty.
Result<scene::GpuSplats> emptyCloud(gpu::Device& device, const ChunkFormat& f, uint32_t n, const std::string& name,
                                    const scene::Bounds& bounds) {
    scene::GpuSplats s;
    s.source = name;
    s.count = n;
    s.declared = n;
    s.restPerColour = f.restPerColour;
    s.shWords = f.shWords;
    s.linear = f.linear;
    s.bounds = bounds;
    s.transferCount = f.transferCount;
    s.transferWords = f.transferWords;
    s.shadowWords = f.shadowWords;
    const auto make = [&](gpu::Buffer& into, uint64_t words, uint32_t element) -> Result<void> {
        auto b = deviceBuffer(device, uint64_t{n} * words * 4, element, "athz.cloud");
        if (!b) return std::move(b).error();
        into = std::move(*b);
        return ok();
    };
    ATHENEA_TRY(make(s.positions, 4, 16));
    ATHENEA_TRY(make(s.shape, 4, 4));
    ATHENEA_TRY(make(s.sh, f.shWords, 4));
    if (f.normals) ATHENEA_TRY(make(s.normals, 1, 4));
    if (f.emission) ATHENEA_TRY(make(s.emission, 1, 4));
    if (f.pbrWords) ATHENEA_TRY(make(s.pbr, f.pbrWords, 4));
    if (f.lobesWords) ATHENEA_TRY(make(s.lobes, f.lobesWords, 4));
    if (f.transferWords) ATHENEA_TRY(make(s.transfer, f.transferWords, 4));
    if (f.shadowWords) ATHENEA_TRY(make(s.shadowBits, f.shadowWords, 4));
    return s;
}

// --- a parsed file -------------------------------------------------------------

struct PageEntry {
    std::array<uint32_t, 6> info{};       ///< athzPageInfo's: count | band << 16 | level << 24, parent, box, eps
    std::array<uint32_t, 4> offset{};     ///< in each layer's section
    std::array<uint32_t, 4> bytes{};
    uint32_t                crc = 0;
    uint32_t                reserved = 0;
    [[nodiscard]] uint32_t count() const noexcept { return info[0] & 0xFFFF; }
    [[nodiscard]] uint32_t band() const noexcept { return (info[0] >> 16) & 0xFF; }
};
static_assert(sizeof(PageEntry) == kEntryBytes);

struct Parsed {
    std::filesystem::path  path;
    platform::MappedFile   file;
    Header                 header;
    Asset                  asset;
    std::vector<PageEntry> pages;
    uint32_t               leafFirstPage = 0;   ///< the splats' first page (after every merged level's)
    uint32_t               leafPages = 0;       ///< the splats' pages in the whole asset
    std::string            source;              ///< the provenance extension
    std::vector<uint8_t>   tlv;                 ///< the extensions as written
};

Result<std::unique_ptr<Parsed>> parse(const std::filesystem::path& path) {
    auto parsed = std::make_unique<Parsed>();
    Parsed& p = *parsed;
    p.path = path;
    auto file = platform::MappedFile::open(path);
    if (!file) return std::move(file).error();
    p.file = std::move(*file);
    const std::span<const std::byte> bytes = p.file.bytes();
    auto header = decodeHeader(bytes, path);
    if (!header) return std::move(header).error();
    p.header = *header;
    auto asset = decodeAsset(bytes, p.header, path);
    if (!asset) return std::move(asset).error();
    p.asset = std::move(*asset);
    const Asset& a = p.asset;
    p.leafFirstPage = a.levels.back().firstPage + a.levels.back().pages;
    p.leafPages = (a.count + a.format.chunkSplats - 1) / a.format.chunkSplats;
    const Header& h = p.header;
    if (h.pages > p.leafFirstPage + p.leafPages) {
        return bad(path, "more pages than the asset has");
    }
    if (h.pageTableOffset > bytes.size() || uint64_t{h.pages} * kEntryBytes > bytes.size() - h.pageTableOffset) {
        return bad(path, "page table past the end: a file cut short");
    }
    p.pages.resize(h.pages);
    std::memcpy(p.pages.data(), bytes.data() + h.pageTableOffset, p.pages.size() * kEntryBytes);
    for (uint32_t i = 0; i < h.pages; ++i) {
        const PageEntry& e = p.pages[i];
        const uint32_t band = i < p.leafFirstPage
                                  ? static_cast<uint32_t>(std::upper_bound(a.levels.begin(), a.levels.end(), i,
                                                                           [](uint32_t v, const LevelEntry& l) {
                                                                               return v < l.firstPage;
                                                                           }) -
                                                          a.levels.begin() - 1)
                                  : static_cast<uint32_t>(a.levels.size());
        const uint64_t elements = i < p.leafFirstPage ? a.levels[band].groups : a.count;
        const uint64_t first = uint64_t{i - (i < p.leafFirstPage ? a.levels[band].firstPage : p.leafFirstPage)} *
                               a.format.chunkSplats;
        if (e.band() != band || e.count() != std::min<uint64_t>(a.format.chunkSplats, elements - first)) {
            return bad(path, "page " + std::to_string(i) + "'s band or count");
        }
        for (uint32_t L = 0; L < 4; ++L) {
            const bool present = ((h.layerMask >> L) & 1u) != 0;
            if ((!present && e.bytes[L] != 0) ||
                (present && (uint64_t{e.offset[L]} + e.bytes[L] > a.sectionBytes[L] || e.bytes[L] < 4u * h.pageRuns))) {
                return bad(path, "page " + std::to_string(i) + " past its section");
            }
        }
    }
    // The extensions: walked, the provenance kept, the rest skipped.
    if (h.tlvOffset != 0) {
        uint64_t at = h.tlvOffset;
        const auto* b = reinterpret_cast<const uint8_t*>(bytes.data());
        const uint64_t start = at;
        while (at + 8 <= h.assetOffset) {
            const uint16_t type = getAt<uint16_t>(b + at);
            const uint32_t length = getAt<uint32_t>(b + at + 4);
            if (type == 0) {
                at += 8;
                break;
            }
            if (at + 8 + length > h.assetOffset) {
                return bad(path, "an extension past the header page");
            }
            if (type == kTlvSource) {
                p.source.assign(reinterpret_cast<const char*>(b + at + 8), length);
            }
            at += 8 + aligned(length, 8);
        }
        p.tlv.assign(b + start, b + at);
    }
    return parsed;
}

/// The page's bytes of layer L in the mapping.
const std::byte* pageBytes(const Parsed& p, uint32_t page, uint32_t L) {
    return p.file.bytes().data() + p.asset.sectionOffset[L] + p.pages[page].offset[L];
}

/// Whether the page's bytes are what its CRC says.
Result<void> checkPage(const Parsed& p, uint32_t page) {
    uint32_t crc = 0;
    for (uint32_t L = 0; L < 4; ++L) {
        if (((p.header.layerMask >> L) & 1u) != 0) {
            crc = crcOf(pageBytes(p, page, L), p.pages[page].bytes[L], crc);
        }
    }
    if (crc != p.pages[page].crc) {
        return Error::make(ErrorCode::IoFailure, "{}: page {} is corrupt (its CRC-32 does not match)", p.path.string(),
                           page);
    }
    return ok();
}

uint64_t codedBytes(const Parsed& p, uint32_t page) {
    uint64_t n = 0;
    for (uint32_t L = 0; L < 4; ++L) n += p.pages[page].bytes[L];
    return n;
}

// --- the decoder ---------------------------------------------------------------

class Decoder {
public:
    static Result<std::unique_ptr<Decoder>> create(gpu::ShaderLibrary& library, const Asset& asset,
                                                   uint32_t layerMask, uint32_t layers) {
        auto d = std::unique_ptr<Decoder>(new Decoder());
        d->device_ = &library.device();
        d->asset_ = &asset;
        d->layers_ = (layers | kAthzGeometry) & layerMask;
        auto kernel = gpu::ComputeKernel::create(
            library, asset.tableLog <= 11 ? "athenea/lod/athz_decode11" : "athenea/lod/athz_decode", "athzDecode");
        if (!kernel) return std::move(kernel).error();
        d->decode_ = std::move(*kernel);
        auto expand = gpu::ComputeKernel::create(library, "athenea/lod/athz_tables", "athzExpandTables");
        if (!expand) return std::move(expand).error();
        gpu::Device& device = *d->device_;
        for (uint32_t k = 0; k < kKinds; ++k) {
            std::vector<uint32_t> words;
            for (uint32_t b = 0; b < asset.planes; ++b) {
                const PlaneEntry& e = asset.planeEntries[k * asset.planes + b];
                words.insert(words.end(), e.begin(), e.end());
                if ((e[0] & 0xFF) == 1) {
                    d->hasRans_[k] |= 1u << (e[0] >> 24);
                }
            }
            auto planes = deviceBuffer(device, words.size() * 4, 16, "athz.planes", words.data());
            if (!planes) return std::move(planes).error();
            d->planes_[k] = std::move(*planes);
        }
        const uint64_t slots = uint64_t{std::max<uint32_t>(asset.tables, 1)} << asset.tableLog;
        auto tables = deviceBuffer(device, slots * 4, 4, "athz.tables");
        if (!tables) return std::move(tables).error();
        d->tables_ = std::move(*tables);
        if (asset.tables > 0) {
            auto freqs = deviceBuffer(device, asset.frequencies.size() * 2, 0, "athz.frequencies",
                                      asset.frequencies.data());
            if (!freqs) return std::move(freqs).error();
            gpu::CommandBatch batch(device);
            expand->dispatch(batch, {static_cast<uint32_t>(slots), 1, 1}, [&](rhi::ShaderCursor c) {
                c["frequencies"].setBinding(freqs->rhi());
                c["slots"].setBinding(d->tables_.rhi());
                c["params"]["tables"].setData(asset.tables);
                c["params"]["tableLog"].setData(asset.tableLog);
            });
            ATHENEA_TRY(batch.submit(true));
        }
        auto errors = zeros(device, 4, "athz.errors");
        if (!errors) return std::move(errors).error();
        d->errors_ = std::move(*errors);
        auto dummy = zeros(device, 4, "athz.dummy");
        if (!dummy) return std::move(dummy).error();
        d->dummy_ = std::move(*dummy);
        return d;
    }

    [[nodiscard]] uint32_t layers() const noexcept { return layers_; }

    /// Records the decode of the pages `list` (8 uints each: the source byte
    /// offsets of G, A, T, K, the element written from, its elements) of
    /// `kind` out of `source` into `dst`. One dispatch.
    Result<void> record(gpu::CommandBatch& batch, uint32_t kind, const gpu::Buffer& source,
                        std::span<const uint32_t> list, const Arrays& dst, uint32_t runGaussians, uint32_t pageRuns) {
        const auto pages = static_cast<uint32_t>(list.size() / 8);
        if (pages == 0) {
            return ok();
        }
        // The page list, uploaded into a buffer of its own: what the
        // dispatch reads stays as it was until the device has run it.
        auto made = deviceBuffer(*device_, list.size_bytes(), 4, "athz.pages", list.data());
        if (!made) return std::move(made).error();
        lists_.push_back(std::move(*made));
        if (lists_.size() > 64) {
            lists_.erase(lists_.begin());   // long run, long since read
        }
        const gpu::Buffer& pageList = lists_.back();
        const uint32_t runs = pages * pageRuns;
        const uint32_t groups = (runs + 7) / 8;
        decode_.dispatch(batch, {groups * 256, 1, 1}, [&](rhi::ShaderCursor c) {
            setLayout(c["params"], asset_->format);
            c["params"]["pages"].setData(pages);
            c["params"]["pageRuns"].setData(pageRuns);
            c["params"]["runGaussians"].setData(runGaussians);
            c["params"]["tableLog"].setData(asset_->tableLog);
            c["params"]["planes"].setData(asset_->planes);
            c["params"]["layers"].setData(layers_);
            c["params"]["hasRans"].setData(hasRans_[kind]);
            c["pageList"].setBinding(pageList.rhi());
            c["planes"].setBinding(planes_[kind].rhi());
            c["tables"].setBinding(tables_.rhi());
            c["source"].setBinding(source.rhi());
            for (size_t f = 0; f < kDstNames.size(); ++f) {
                const gpu::Buffer* b = dst.field[f];
                c[kDstNames[f]].setBinding((b != nullptr && b->valid() ? *b : dummy_).rhi());
            }
            c["errors"].setBinding(errors_.rhi());
        });
        return ok();
    }

    /// Runs found corrupt since the last call (a wait on the device).
    Result<uint32_t> takeErrors() {
        uint32_t n = 0;
        ATHENEA_TRY(errors_.read(*device_, 0, 4, &n));
        if (n != 0) {
            const uint32_t zero = 0;
            ATHENEA_TRY(errors_.write(*device_, 0, 4, &zero));
        }
        return n;
    }

private:
    Decoder() = default;
    gpu::Device*               device_ = nullptr;
    const Asset*               asset_ = nullptr;
    uint32_t                   layers_ = 0;
    std::array<uint32_t, 2>    hasRans_{};
    gpu::ComputeKernel         decode_;
    std::array<gpu::Buffer, 2> planes_;
    gpu::Buffer                tables_, errors_, dummy_;
    std::vector<gpu::Buffer>   lists_;
};

/// Pages [first, last) of the file uploaded together, and their page list:
/// each layer's bytes of a page where `list` says. CRCs checked first.
struct Upload {
    gpu::Buffer           source;
    std::vector<uint32_t> list;
    uint64_t              bytes = 0;
};

Result<Upload> uploadPages(gpu::Device& device, const Parsed& p, const std::vector<uint32_t>& pages,
                           const std::vector<uint32_t>& dstElement, uint32_t layers) {
    Upload u;
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < pages.size(); ++i) {
        const uint32_t page = pages[i];
        ATHENEA_TRY(checkPage(p, page));
        std::array<uint32_t, 4> src{kNone, kNone, kNone, kNone};
        for (uint32_t L = 0; L < 4; ++L) {
            if (((layers >> L) & 1u) == 0) continue;
            src[L] = static_cast<uint32_t>(bytes.size());
            const std::byte* from = pageBytes(p, page, L);
            bytes.insert(bytes.end(), reinterpret_cast<const uint8_t*>(from),
                         reinterpret_cast<const uint8_t*>(from) + p.pages[page].bytes[L]);
        }
        u.list.insert(u.list.end(), src.begin(), src.end());
        u.list.push_back(dstElement[i]);
        u.list.push_back(p.pages[page].count());
        u.list.push_back(0);
        u.list.push_back(0);
    }
    u.bytes = bytes.size();
    auto source = deviceBuffer(device, bytes.size(), 0, "athz.source", bytes.empty() ? nullptr : bytes.data());
    if (!source) return std::move(source).error();
    u.source = std::move(*source);
    return u;
}

/// Decodes `pages` (all of one kind) into `dst`, page i written from
/// element dstElement[i], in batches of at most kReadBatchBytes.
Result<void> decodePages(gpu::Device& device, Decoder& decoder, const Parsed& p, uint32_t kind,
                         const std::vector<uint32_t>& pages, const std::vector<uint32_t>& dstElement,
                         const Arrays& dst) {
    size_t at = 0;
    while (at < pages.size()) {
        std::vector<uint32_t> some, where;
        uint64_t bytes = 0;
        while (at < pages.size() && (some.empty() || bytes + codedBytes(p, pages[at]) <= kReadBatchBytes)) {
            bytes += codedBytes(p, pages[at]);
            some.push_back(pages[at]);
            where.push_back(dstElement[at]);
            ++at;
        }
        auto upload = uploadPages(device, p, some, where, decoder.layers());
        if (!upload) return std::move(upload).error();
        gpu::CommandBatch batch(device);
        ATHENEA_TRY(decoder.record(batch, kind, upload->source, upload->list, dst, p.header.runGaussians,
                                   p.header.pageRuns));
        ATHENEA_TRY(batch.submit(true));
        auto errors = decoder.takeErrors();
        if (!errors) return std::move(errors).error();
        if (*errors != 0) {
            return Error::make(ErrorCode::IoFailure, "{}: {} corrupt lanes in pages {} to {}", p.path.string(), *errors,
                               some.front(), some.back());
        }
    }
    return ok();
}

/// The merged levels and the finest level's starts on the device, as
/// cloudOf does for a .athc: every level's pages must be in the file.
Result<LodCloud> levelsOf(gpu::Device& device, Decoder& decoder, const Parsed& p) {
    const Asset& a = p.asset;
    if (p.header.pages < p.leafFirstPage) {
        return Error::make(ErrorCode::InvalidArgument,
                           "{}: a prefix of bands that stops inside the merged levels ({} of {} pages) is a fixed cut "
                           "(readAthzCut), not a cloud to stream",
                           p.path.string(), p.header.pages, p.leafFirstPage);
    }
    LodCloud lod;
    lod.count = a.count;
    lod.chunkSplats = a.format.chunkSplats;
    std::copy(a.boundsLo, a.boundsLo + 3, lod.boundsLo);
    lod.extent = a.extent;
    const std::string name = p.path.filename().string();
    for (const LevelEntry& e : a.levels) {
        LodLevel level;
        level.level = e.level;
        auto g = emptyCloud(device, a.format, e.groups, name, a.bounds);
        if (!g) return std::move(g).error();
        level.gaussians = std::move(*g);
        auto cells = deviceBuffer(device, uint64_t{e.groups} * 4, 4, "athz.cells");
        if (!cells) return std::move(cells).error();
        level.cells = std::move(*cells);
        std::vector<uint32_t> pages, where;
        for (uint32_t k = 0; k < e.pages; ++k) {
            pages.push_back(e.firstPage + k);
            where.push_back(k * a.format.chunkSplats);
        }
        ATHENEA_TRY(decodePages(device, decoder, p, 0, pages, where, arraysOf(level.gaussians, level.cells)));
        lod.levels.push_back(std::move(level));
    }
    auto starts = deviceBuffer(device, a.startsBytes, 4, "athz.starts", p.file.bytes().data() + a.startsOffset);
    if (!starts) return std::move(starts).error();
    lod.starts = std::move(*starts);
    return lod;
}

// --- the encoder ---------------------------------------------------------------

struct EncodeKernels {
    gpu::ComputeKernel histogram, normalize, encodeRuns, place, pageInfo;
    gpu::PrefixSum     prefix;
};

Result<EncodeKernels> encodeKernels(gpu::ShaderLibrary& library) {
    EncodeKernels k;
    const auto make = [&](gpu::ComputeKernel& into, const char* entry) -> Result<void> {
        auto made = gpu::ComputeKernel::create(library, "athenea/lod/athz_encode", entry);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(make(k.histogram, "athzHistogram"));
    ATHENEA_TRY(make(k.normalize, "athzNormalize"));
    ATHENEA_TRY(make(k.encodeRuns, "athzEncodeRuns"));
    ATHENEA_TRY(make(k.place, "athzPlace"));
    ATHENEA_TRY(make(k.pageInfo, "athzPageInfo"));
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    k.prefix = std::move(*prefix);
    return k;
}

/// One kind's elements, as the encoder reads them.
struct KindSource {
    Arrays                  arrays;
    std::vector<uint32_t>   pageList;   ///< uint4s: first element, elements, page in the file, band
    uint64_t                elements = 0;
};

void bindSources(rhi::ShaderCursor c, const Arrays& a, const gpu::Buffer& dummy) {
    for (size_t f = 0; f < kSrcNames.size(); ++f) {
        const gpu::Buffer* b = a.field[f];
        c[kSrcNames[f]].setBinding((b != nullptr && b->valid() ? *b : dummy).rhi());
    }
}

}   // namespace

bool isAthz(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext == ".athz";
}

Result<AthzStats> writeAthz(gpu::ShaderLibrary& library, const LodCloud& cloud, const std::filesystem::path& path,
                            const AthzSettings& settings) {
    gpu::Device& device = library.device();
    if (cloud.levels.empty() || cloud.count == 0) {
        return Error(ErrorCode::InvalidArgument, "a .athz holds a cloud with levels of detail");
    }
    for (uint32_t c = 0; c < cloud.chunks(); ++c) {
        if (cloud.slots[c] != static_cast<int32_t>(c) || cloud.chunkCount(c) != std::min(cloud.chunkSplats, cloud.count - c * cloud.chunkSplats)) {
            return Error(ErrorCode::InvalidArgument, "writing a .athz needs every chunk on the device, chunk c in slot c");
        }
    }
    const uint32_t R = settings.runGaussians;
    if (R == 0 || R % kLanes != 0 || R > 256 || settings.pageRuns == 0 || settings.pageRuns > 0xFFFF ||
        settings.tableLog < 9 || settings.tableLog > 12) {
        return Error(ErrorCode::InvalidArgument, "athz: runs of 32 to 256 in steps of 32, tables of 2^9 to 2^12");
    }
    const uint32_t PG = R * settings.pageRuns;
    if (PG > 0xFFFF) {
        return Error(ErrorCode::InvalidArgument, "athz: a page holds at most 65535 elements (its count is 16 bits)");
    }
    auto kernels = encodeKernels(library);
    if (!kernels) return std::move(kernels).error();
    auto dummy = zeros(device, 4, "athz.dummy");
    if (!dummy) return std::move(dummy).error();

    Asset asset;
    asset.format = formatOfCloud(cloud, PG);
    const ChunkFormat& f = asset.format;
    const uint32_t words = static_cast<uint32_t>(f.elementBytes() / 4);
    asset.planes = 4 * words;
    asset.tableLog = settings.tableLog;
    asset.count = cloud.count;
    asset.finestGroups = cloud.levels.back().gaussians.count;
    std::copy(cloud.boundsLo, cloud.boundsLo + 3, asset.boundsLo);
    asset.extent = cloud.extent;
    asset.bounds = cloud.splats.bounds;

    // THE TWO KINDS' SOURCES. The merged levels' arrays put end to end
    // (device copies), coarsest first; the splats are the store itself.
    std::array<KindSource, 2> kinds;
    scene::GpuSplats merged;
    gpu::Buffer      mergedCells;
    {
        uint64_t total = 0;
        for (const LodLevel& l : cloud.levels) total += l.gaussians.count;
        auto m = emptyCloud(device, f, static_cast<uint32_t>(total), "athz.merged", asset.bounds);
        if (!m) return std::move(m).error();
        merged = std::move(*m);
        auto cells = deviceBuffer(device, total * 4, 4, "athz.mergedCells");
        if (!cells) return std::move(cells).error();
        mergedCells = std::move(*cells);
        const Arrays to = arraysOf(merged, mergedCells);
        const std::array<uint32_t, 10> fw = fieldWords(f);
        gpu::CommandBatch batch(device);
        uint64_t at = 0;
        uint32_t page = 0;
        for (size_t li = 0; li < cloud.levels.size(); ++li) {
            const LodLevel& l = cloud.levels[li];
            const Arrays from = arraysOf(l.gaussians, l.cells);
            for (size_t k = 0; k < fw.size(); ++k) {
                if (fw[k] == 0) continue;
                const uint64_t per = uint64_t{fw[k]} * 4;
                batch.encoder()->copyBuffer(to.field[k]->rhi(), at * per, from.field[k]->rhi(), 0,
                                            uint64_t{l.gaussians.count} * per);
            }
            batch.markDirty();
            LevelEntry e;
            e.level = l.level;
            e.groups = l.gaussians.count;
            e.firstPage = page;
            e.pages = (e.groups + PG - 1) / PG;
            for (uint32_t k = 0; k < e.pages; ++k) {
                kinds[0].pageList.insert(kinds[0].pageList.end(),
                                         {static_cast<uint32_t>(at + uint64_t{k} * PG), std::min(PG, e.groups - k * PG),
                                          page + k, static_cast<uint32_t>(li)});
            }
            page += e.pages;
            at += l.gaussians.count;
            asset.levels.push_back(e);
        }
        ATHENEA_TRY(batch.submit(true));
        kinds[0].arrays = to;
        kinds[0].elements = total;
        const uint32_t leafPages = (cloud.count + PG - 1) / PG;
        for (uint32_t k = 0; k < leafPages; ++k) {
            kinds[1].pageList.insert(kinds[1].pageList.end(), {k * PG, std::min(PG, cloud.count - k * PG), page + k,
                                                               static_cast<uint32_t>(cloud.levels.size())});
        }
        kinds[1].arrays = arraysOf(cloud.splats, cloud.groups);
        kinds[1].elements = cloud.count;
    }
    const uint32_t totalPages = static_cast<uint32_t>((kinds[0].pageList.size() + kinds[1].pageList.size()) / 4);

    // Each byte's layer; what the file holds of each.
    std::vector<uint32_t> layerOf(asset.planes);
    uint32_t layerMask = 0;
    for (uint32_t b = 0; b < asset.planes; ++b) {
        layerOf[b] = layerOfByte(f, b);
        layerMask |= 1u << layerOf[b];
    }

    AthzStats stats;
    stats.elementBytes = static_cast<uint32_t>(f.elementBytes());
    stats.pages = totalPages;
    std::array<std::vector<uint8_t>, 4> sections;
    std::vector<PageEntry> entries(totalPages);

    for (uint32_t kind = 0; kind < kKinds; ++kind) {
        KindSource& ks = kinds[kind];
        const auto pages = static_cast<uint32_t>(ks.pageList.size() / 4);
        stats.elements[kind] = ks.elements;
        // THE TABLES: histograms (in a shuffled page order where asked: the
        // counts are sums), then each plane's range and frequencies.
        std::vector<uint32_t> order(ks.pageList);
        if (settings.histogramShuffle != 0) {
            std::vector<uint32_t> perm(pages);
            std::iota(perm.begin(), perm.end(), 0u);
            std::mt19937_64 rng(settings.histogramShuffle);
            std::shuffle(perm.begin(), perm.end(), rng);
            for (uint32_t i = 0; i < pages; ++i) {
                std::copy_n(ks.pageList.begin() + perm[i] * 4, 4, order.begin() + i * 4);
            }
        }
        std::vector<uint32_t> planeWords(uint64_t{asset.planes} * 4, 0u);
        for (uint32_t b = 0; b < asset.planes; ++b) planeWords[b * 4] = layerOf[b] << 24;
        auto planesIn = deviceBuffer(device, planeWords.size() * 4, 16, "athz.planesIn", planeWords.data());
        auto listShuffled = deviceBuffer(device, order.size() * 4, 16, "athz.order", order.data());
        auto histogram = zeros(device, uint64_t{asset.planes} * 256, "athz.histogram");
        auto planesOut = zeros(device, uint64_t{asset.planes} * 4, "athz.planesOut");
        auto freqs = zeros(device, uint64_t{asset.planes} * 256, "athz.freqs");
        for (auto* r : {&planesIn, &listShuffled, &histogram, &planesOut, &freqs}) {
            if (!*r) return std::move(*r).error();
        }
        {
            gpu::CommandBatch batch(device);
            const auto bindParams = [&](rhi::ShaderCursor c) {
                setLayout(c["params"], f);
                c["params"]["pages"].setData(pages);
                c["params"]["pageRuns"].setData(settings.pageRuns);
                c["params"]["runGaussians"].setData(R);
                c["params"]["tableLog"].setData(settings.tableLog);
                c["params"]["planes"].setData(asset.planes);
                c["params"]["allowRans"].setData(settings.rans && !settings.bytePlanes ? 1u : 0u);
                c["params"]["extent"].setData(cloud.extent);
            };
            if (pages > 0) {
                kernels->histogram.dispatch(batch, {pages * 256, asset.planes, 1}, [&](rhi::ShaderCursor c) {
                    bindParams(c);
                    c["pageList"].setBinding(listShuffled->rhi());
                    c["planes"].setBinding(planesIn->rhi());
                    bindSources(c, ks.arrays, *dummy);
                    c["histogram"].setBinding(histogram->rhi());
                });
            }
            kernels->normalize.dispatch(batch, {asset.planes, 1, 1}, [&](rhi::ShaderCursor c) {
                bindParams(c);
                c["planes"].setBinding(planesIn->rhi());
                c["histogram"].setBinding(histogram->rhi());
                c["planesOut"].setBinding(planesOut->rhi());
                c["freqs"].setBinding(freqs->rhi());
            });
            ATHENEA_TRY(batch.submit(true));
        }
        auto coded = planesOut->readAll<uint32_t>(device);
        if (!coded) return std::move(coded).error();
        auto tableFreqs = freqs->readAll<uint32_t>(device);
        if (!tableFreqs) return std::move(tableFreqs).error();
        // Bookkeeping: each plane's table and where its raw words go in a
        // run, layer by layer.
        std::array<uint32_t, 4> rawWords{}, ransPlanes{};
        uint32_t hasRans = 0;
        for (uint32_t b = 0; b < asset.planes; ++b) {
            PlaneEntry e{(*coded)[b * 4], kNone, 0, (*coded)[b * 4 + 3]};
            if (settings.bytePlanes) {
                e[0] = (layerOf[b] << 24) | (8u << 8);   // raw, 8 bits, from 0
            }
            const uint32_t L = e[0] >> 24, width = (e[0] >> 8) & 0xFF;
            if ((e[0] & 0xFF) == 1) {
                e[1] = asset.tables++;
                for (uint32_t s = 0; s < 256; ++s) {
                    asset.frequencies.push_back(static_cast<uint16_t>((*tableFreqs)[b * 256 + s] & 0xFFFF));
                }
                ++ransPlanes[L];
                hasRans |= 1u << L;
            } else {
                e[2] = rawWords[L];
                rawWords[L] += (R * width + 31) / 32;
            }
            asset.planeEntries.push_back(e);
            AthzPlaneStat st;
            st.kind = kind;
            st.byte = b;
            st.layer = L;
            st.encoding = e[0] & 0xFF;
            st.width = width;
            st.symbols = ks.elements;
            st.rawBytes = (ks.elements * width + 7) / 8;
            st.ransBytes = (*coded)[b * 4 + 3];
            stats.planes.push_back(st);
        }
        std::vector<uint32_t> planeTable;
        for (uint32_t b = 0; b < asset.planes; ++b) {
            const PlaneEntry& e = asset.planeEntries[kind * asset.planes + b];
            planeTable.insert(planeTable.end(), e.begin(), e.end());
        }
        auto planes = deviceBuffer(device, planeTable.size() * 4, 16, "athz.planes", planeTable.data());
        if (!planes) return std::move(planes).error();

        // THE RUNS, a layer at a time, in batches the scratch holds.
        for (uint32_t L = 0; L < 4; ++L) {
            if (((layerMask >> L) & 1u) == 0) continue;
            const uint32_t capWords = R * ransPlanes[L];
            const uint64_t perRun = 4 * (uint64_t{capWords} + rawWords[L] + kLanes + 4);
            const uint32_t perBatch = static_cast<uint32_t>(
                std::clamp<uint64_t>(kScratchBytes / (perRun * settings.pageRuns), 1, std::max(pages, 1u)));
            const uint32_t maxRuns = perBatch * settings.pageRuns;
            auto scratch = deviceBuffer(device, uint64_t{maxRuns} * std::max(capWords, 1u) * 4, 4, "athz.scratch");
            auto rawScratch = deviceBuffer(device, uint64_t{maxRuns} * std::max(rawWords[L], 1u) * 4, 4, "athz.raw");
            auto states = deviceBuffer(device, uint64_t{maxRuns} * kLanes * 4, 4, "athz.states");
            auto runWords = zeros(device, maxRuns, "athz.runWords");
            auto runBytes = zeros(device, maxRuns, "athz.runBytes");
            auto runOffsets = zeros(device, maxRuns, "athz.runOffsets");
            auto total = zeros(device, 1, "athz.total");
            for (auto* r : {&scratch, &rawScratch, &states, &runWords, &runBytes, &runOffsets, &total}) {
                if (!*r) return std::move(*r).error();
            }
            for (uint32_t p0 = 0; p0 < pages; p0 += perBatch) {
                const uint32_t n = std::min(perBatch, pages - p0);
                const uint32_t runs = n * settings.pageRuns;
                auto list = deviceBuffer(device, uint64_t{n} * 16, 16, "athz.list", ks.pageList.data() + uint64_t{p0} * 4);
                if (!list) return std::move(list).error();
                const auto bindParams = [&](rhi::ShaderCursor c) {
                    setLayout(c["params"], f);
                    c["params"]["pages"].setData(n);
                    c["params"]["pageRuns"].setData(settings.pageRuns);
                    c["params"]["runGaussians"].setData(R);
                    c["params"]["tableLog"].setData(settings.tableLog);
                    c["params"]["layer"].setData(L);
                    c["params"]["hasRans"].setData((hasRans >> L) & 1u);
                    c["params"]["rawWords"].setData(rawWords[L]);
                    c["params"]["capWords"].setData(capWords);
                    c["params"]["planes"].setData(asset.planes);
                    c["pageList"].setBinding(list->rhi());
                    c["planes"].setBinding(planes->rhi());
                    c["freqs"].setBinding(freqs->rhi());
                    c["scratch"].setBinding(scratch->rhi());
                    c["rawScratch"].setBinding(rawScratch->rhi());
                    c["states"].setBinding(states->rhi());
                    c["runWords"].setBinding(runWords->rhi());
                    c["runBytes"].setBinding(runBytes->rhi());
                    c["runOffsets"].setBinding(runOffsets->rhi());
                };
                gpu::CommandBatch batch(device);
                kernels->encodeRuns.dispatch(batch, {runs * kLanes, 1, 1}, [&](rhi::ShaderCursor c) {
                    bindParams(c);
                    bindSources(c, ks.arrays, *dummy);
                });
                ATHENEA_TRY(kernels->prefix.apply(batch, *runBytes, *runOffsets, *total, runs));
                ATHENEA_TRY(batch.submit(true));
                uint32_t bytes = 0;
                ATHENEA_TRY(total->read(device, 0, 4, &bytes));
                auto output = deviceBuffer(device, bytes, 0, "athz.output");
                if (!output) return std::move(output).error();
                kernels->place.dispatch(batch, {runs * kLanes, 1, 1}, [&](rhi::ShaderCursor c) {
                    bindParams(c);
                    c["output"].setBinding(output->rhi());
                });
                ATHENEA_TRY(batch.submit(true));
                std::vector<uint8_t> out(bytes);
                ATHENEA_TRY(output->read(device, 0, bytes, out.data()));
                std::vector<uint32_t> offsets(runs);
                ATHENEA_TRY(runOffsets->read(device, 0, uint64_t{runs} * 4, offsets.data()));
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t start = offsets[j * settings.pageRuns];
                    const uint32_t end = j + 1 < n ? offsets[(j + 1) * settings.pageRuns] : bytes;
                    PageEntry& e = entries[ks.pageList[(p0 + j) * 4 + 2]];
                    e.offset[L] = static_cast<uint32_t>(sections[L].size());
                    e.bytes[L] = end - start;
                    sections[L].insert(sections[L].end(), out.begin() + start, out.begin() + end);
                }
            }
        }

        // THE PAGES' BOXES, ERRORS AND PARENTS.
        std::vector<uint32_t> levelInfo;
        {
            uint32_t base = 0;
            for (const LevelEntry& e : asset.levels) {
                levelInfo.insert(levelInfo.end(), {base, e.groups, e.firstPage, e.level});
                base += e.groups;
            }
        }
        auto levels = deviceBuffer(device, levelInfo.size() * 4, 16, "athz.levels", levelInfo.data());
        auto list = deviceBuffer(device, ks.pageList.size() * 4, 16, "athz.list", ks.pageList.data());
        auto info = zeros(device, uint64_t{pages} * 8, "athz.pageInfo");
        for (auto* r : {&levels, &list, &info}) {
            if (!*r) return std::move(*r).error();
        }
        {
            gpu::CommandBatch batch(device);
            kernels->pageInfo.dispatch(batch, {std::max(pages, 1u), 1, 1}, [&](rhi::ShaderCursor c) {
                setLayout(c["params"], f);
                c["params"]["pages"].setData(pages);
                c["params"]["pageRuns"].setData(settings.pageRuns);
                c["params"]["runGaussians"].setData(R);
                c["params"]["levels"].setData(static_cast<uint32_t>(asset.levels.size()));
                c["params"]["leaves"].setData(kind);
                c["params"]["extent"].setData(cloud.extent);
                c["pageList"].setBinding(list->rhi());
                c["levelInfo"].setBinding(levels->rhi());
                c["pageInfo"].setBinding(info->rhi());
                bindSources(c, ks.arrays, *dummy);
            });
            ATHENEA_TRY(batch.submit(true));
        }
        auto pageInfo = info->readAll<uint32_t>(device);
        if (!pageInfo) return std::move(pageInfo).error();
        for (uint32_t j = 0; j < pages; ++j) {
            PageEntry& e = entries[ks.pageList[j * 4 + 2]];
            std::copy_n(pageInfo->begin() + j * 8, 6, e.info.begin());
        }
    }

    // Each page's CRC over its layers' bytes, as stored.
    for (PageEntry& e : entries) {
        uint32_t crc = 0;
        for (uint32_t L = 0; L < 4; ++L) {
            if (((layerMask >> L) & 1u) != 0) {
                crc = crcOf(sections[L].data() + e.offset[L], e.bytes[L], crc);
            }
        }
        e.crc = crc;
    }

    // THE FILE: header page (header, extensions), asset table, page table,
    // starts, then the sections, each on a page.
    std::vector<uint8_t> tlv;
    {
        ByteWriter w;
        std::string source = settings.source.substr(0, 2048);
        if (!source.empty()) {
            w.put<uint16_t>(kTlvSource);
            w.put<uint16_t>(0);
            w.put<uint32_t>(static_cast<uint32_t>(source.size()));
            w.bytes.insert(w.bytes.end(), source.begin(), source.end());
            while (w.bytes.size() % 8 != 0) w.put<uint8_t>(0);
        }
        w.put<uint16_t>(0);
        w.put<uint16_t>(0);
        w.put<uint32_t>(0);
        tlv = std::move(w.bytes);
    }
    std::vector<uint8_t> starts(uint64_t{asset.finestGroups} * 4);
    ATHENEA_TRY(cloud.starts.read(device, 0, starts.size(), starts.data()));
    asset.startsBytes = starts.size();
    // The asset table's size does not depend on the offsets it holds.
    const uint64_t assetBytes = encodeAsset(asset).size();
    asset.assetId = 0;
    const uint64_t pageTable = aligned(kPage + assetBytes, 64);
    asset.startsOffset = aligned(pageTable + uint64_t{totalPages} * kEntryBytes, kPage);
    uint64_t at = aligned(asset.startsOffset + asset.startsBytes, kPage);
    for (uint32_t L = 0; L < 4; ++L) {
        asset.sectionOffset[L] = ((layerMask >> L) & 1u) != 0 ? at : 0;
        asset.sectionBytes[L] = sections[L].size();
        at = aligned(at + sections[L].size(), kPage);
    }
    const std::vector<uint8_t> assetTable = encodeAsset(asset);
    Header h;
    h.flags = ((layerMask & kAthzTransfer) != 0 ? kFlagTransfer : 0) | (f.restPerColour > 0 ? kFlagHarmonics : 0) |
              (f.linear ? kFlagLinear : 0) | (f.emission ? kFlagEmission : 0);
    h.layerMask = layerMask;
    h.gaussians = cloud.count;
    h.pages = totalPages;
    h.runGaussians = static_cast<uint16_t>(R);
    h.pageRuns = static_cast<uint16_t>(settings.pageRuns);
    h.assetOffset = kPage;
    h.assetBytes = static_cast<uint32_t>(assetTable.size());
    h.pageTableOffset = pageTable;
    h.tlvOffset = kHeaderBytes;
    const auto header = encodeHeader(h);

    std::filesystem::path partial = path;
    partial += ".partial";
    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Error(ErrorCode::IoFailure, "cannot write " + partial.string());
    }
    uint64_t written = 0;
    const auto put = [&](const void* data, uint64_t size) {
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        written += size;
    };
    const auto padTo = [&](uint64_t offset) {
        static const char kZeros[kPage] = {};
        while (written < offset) put(kZeros, std::min<uint64_t>(kPage, offset - written));
    };
    put(header.data(), header.size());
    put(tlv.data(), std::min<uint64_t>(tlv.size(), kPage - kHeaderBytes));
    padTo(kPage);
    put(assetTable.data(), assetTable.size());
    padTo(pageTable);
    put(entries.data(), entries.size() * kEntryBytes);
    padTo(asset.startsOffset);
    put(starts.data(), starts.size());
    for (uint32_t L = 0; L < 4; ++L) {
        if (asset.sectionOffset[L] == 0) continue;
        padTo(asset.sectionOffset[L]);
        put(sections[L].data(), sections[L].size());
    }
    padTo(aligned(written, kPage));
    out.close();
    if (!out) {
        return Error(ErrorCode::IoFailure, "writing " + partial.string() + " failed");
    }
    std::error_code ec;
    std::filesystem::rename(partial, path, ec);
    if (ec) {
        return Error(ErrorCode::IoFailure, "cannot move " + partial.string() + " to " + path.string() + ": " + ec.message());
    }
    stats.fileBytes = written;
    for (uint32_t L = 0; L < 4; ++L) stats.layerBytes[L] = sections[L].size();
    stats.tableBytes = kPage + assetTable.size() + uint64_t{totalPages} * kEntryBytes;
    stats.startsBytes = starts.size();
    log::info("{}: {} splats and {} merged in {} pages of {} ({}x{}), tables 2^{}, {:.2f} MB", path.string(), cloud.count,
              stats.elements[0], totalPages, PG, settings.pageRuns, R, settings.tableLog,
              static_cast<double>(written) / (1 << 20));
    return stats;
}

Result<LodCloud> readAthz(gpu::ShaderLibrary& library, const std::filesystem::path& path, uint32_t layers) {
    gpu::Device& device = library.device();
    auto parsed = parse(path);
    if (!parsed) return std::move(parsed).error();
    const Parsed& p = **parsed;
    auto decoder = Decoder::create(library, p.asset, p.header.layerMask, layers);
    if (!decoder) return std::move(decoder).error();
    auto lod = levelsOf(device, **decoder, p);
    if (!lod) return std::move(lod).error();
    const Asset& a = p.asset;
    ATHENEA_TRY(detail::makeStore(device, *lod, a.format, a.count, p.leafPages, p.leafPages, a.bounds,
                                  path.filename().string()));
    std::vector<uint32_t> pages, where, resident(p.leafPages, 0);
    for (uint32_t c = 0; c < p.leafPages && p.leafFirstPage + c < p.header.pages; ++c) {
        pages.push_back(p.leafFirstPage + c);
        where.push_back(c * a.format.chunkSplats);
        lod->slots[c] = static_cast<int32_t>(c);
        resident[c] = 1;
    }
    ATHENEA_TRY(decodePages(device, **decoder, p, 1, pages, where, arraysOf(lod->splats, lod->groups)));
    ATHENEA_TRY(lod->resident.write(device, 0, resident.size() * 4, resident.data()));
    lod->streamed = pages.size() < p.leafPages;
    return std::move(*lod);
}

Result<scene::GpuSplats> readAthzCut(gpu::ShaderLibrary& library, const std::filesystem::path& path, uint32_t* band) {
    gpu::Device& device = library.device();
    auto parsed = parse(path);
    if (!parsed) return std::move(parsed).error();
    const Parsed& p = **parsed;
    const Asset& a = p.asset;
    auto decoder = Decoder::create(library, a, p.header.layerMask, kAthzAllLayers);
    if (!decoder) return std::move(decoder).error();
    // The finest band all of whose pages are here.
    const auto bands = static_cast<uint32_t>(a.levels.size() + 1);
    uint32_t chosen = bands;
    for (uint32_t b = bands; b-- > 0;) {
        const uint32_t first = b < a.levels.size() ? a.levels[b].firstPage : p.leafFirstPage;
        const uint32_t n = b < a.levels.size() ? a.levels[b].pages : p.leafPages;
        if (first + n <= p.header.pages) {
            chosen = b;
            break;
        }
    }
    if (chosen == bands) {
        return Error::make(ErrorCode::InvalidArgument, "{}: not even the coarsest level is whole", path.string());
    }
    const bool leaves = chosen == a.levels.size();
    const uint32_t first = leaves ? p.leafFirstPage : a.levels[chosen].firstPage;
    const uint32_t n = leaves ? p.leafPages : a.levels[chosen].pages;
    const uint32_t elements = leaves ? a.count : a.levels[chosen].groups;
    auto cloud = emptyCloud(device, a.format, elements, path.filename().string(), a.bounds);
    if (!cloud) return std::move(cloud).error();
    auto tail = deviceBuffer(device, uint64_t{elements} * 4, 4, "athz.tail");
    if (!tail) return std::move(tail).error();
    std::vector<uint32_t> pages, where;
    for (uint32_t k = 0; k < n; ++k) {
        pages.push_back(first + k);
        where.push_back(k * a.format.chunkSplats);
    }
    ATHENEA_TRY(decodePages(device, **decoder, p, leaves ? 1 : 0, pages, where, arraysOf(*cloud, *tail)));
    if (band != nullptr) {
        *band = chosen;
    }
    return std::move(*cloud);
}

Result<void> writeAthzPrefix(const std::filesystem::path& in, const std::filesystem::path& out, uint32_t bands) {
    auto parsed = parse(in);
    if (!parsed) return std::move(parsed).error();
    const Parsed& p = **parsed;
    Asset a = p.asset;
    const uint32_t keep = std::min(p.header.pages, bands < a.levels.size() ? a.levels[bands].firstPage
                                                                            : (bands == a.levels.size() ? p.leafFirstPage
                                                                                                        : p.header.pages));
    std::array<uint64_t, 4> bytes{};
    for (uint32_t L = 0; L < 4; ++L) {
        bytes[L] = keep < p.header.pages ? p.pages[keep].offset[L] : a.sectionBytes[L];
        a.sectionBytes[L] = bytes[L];
    }
    const uint64_t assetBytes = encodeAsset(a).size();
    const uint64_t pageTable = aligned(kPage + assetBytes, 64);
    const uint64_t startsOffset = aligned(pageTable + uint64_t{keep} * kEntryBytes, kPage);
    std::array<uint64_t, 4> oldOffset = a.sectionOffset;
    uint64_t at = aligned(startsOffset + a.startsBytes, kPage);
    for (uint32_t L = 0; L < 4; ++L) {
        if (a.sectionOffset[L] == 0) continue;
        a.sectionOffset[L] = at;
        at = aligned(at + bytes[L], kPage);
    }
    const uint64_t oldStarts = a.startsOffset;
    a.startsOffset = startsOffset;
    const std::vector<uint8_t> assetTable = encodeAsset(a);
    Header h = p.header;
    h.pages = keep;
    h.assetBytes = static_cast<uint32_t>(assetTable.size());
    h.pageTableOffset = pageTable;
    const auto header = encodeHeader(h);
    const std::byte* base = p.file.bytes().data();

    std::filesystem::path partial = out;
    partial += ".partial";
    std::ofstream o(partial, std::ios::binary | std::ios::trunc);
    if (!o) {
        return Error(ErrorCode::IoFailure, "cannot write " + partial.string());
    }
    uint64_t written = 0;
    const auto put = [&](const void* data, uint64_t size) {
        o.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        written += size;
    };
    const auto padTo = [&](uint64_t offset) {
        static const char kZeros[kPage] = {};
        while (written < offset) put(kZeros, std::min<uint64_t>(kPage, offset - written));
    };
    put(header.data(), header.size());
    put(base + kHeaderBytes, kPage - kHeaderBytes);   // the extensions as they were
    put(assetTable.data(), assetTable.size());
    padTo(pageTable);
    put(p.pages.data(), uint64_t{keep} * kEntryBytes);
    padTo(startsOffset);
    put(base + oldStarts, a.startsBytes);
    for (uint32_t L = 0; L < 4; ++L) {
        if (a.sectionOffset[L] == 0) continue;
        padTo(a.sectionOffset[L]);
        put(base + oldOffset[L], bytes[L]);
    }
    padTo(aligned(written, kPage));
    o.close();
    if (!o) {
        return Error(ErrorCode::IoFailure, "writing " + partial.string() + " failed");
    }
    std::error_code ec;
    std::filesystem::rename(partial, out, ec);
    if (ec) {
        return Error(ErrorCode::IoFailure, "cannot move " + partial.string() + " to " + out.string());
    }
    return ok();
}

Result<std::vector<AthzDecodeTiming>> timeAthzDecode(gpu::ShaderLibrary& library, const std::filesystem::path& path,
                                                     uint64_t batchBytes, uint32_t singlePages) {
    gpu::Device& device = library.device();
    auto parsed = parse(path);
    if (!parsed) return std::move(parsed).error();
    const Parsed& p = **parsed;
    const Asset& a = p.asset;
    auto decoder = Decoder::create(library, a, p.header.layerMask, kAthzAllLayers);
    if (!decoder) return std::move(decoder).error();
    const uint32_t present = p.header.pages > p.leafFirstPage ? p.header.pages - p.leafFirstPage : 0;
    auto store = emptyCloud(device, a.format, present * a.format.chunkSplats, "athz.timing", a.bounds);
    if (!store) return std::move(store).error();
    auto tail = deviceBuffer(device, uint64_t{present} * a.format.chunkSplats * 4, 4, "athz.tail");
    if (!tail) return std::move(tail).error();
    const Arrays dst = arraysOf(*store, *tail);
    std::vector<AthzDecodeTiming> out;
    const auto timeOne = [&](const std::vector<uint32_t>& pages, bool single) -> Result<void> {
        std::vector<uint32_t> where;
        uint64_t decoded = 0;
        for (uint32_t page : pages) {
            where.push_back((page - p.leafFirstPage) * a.format.chunkSplats);
            decoded += uint64_t{p.pages[page].count()} * a.format.elementBytes();
        }
        auto upload = uploadPages(device, p, pages, where, (*decoder)->layers());
        if (!upload) return std::move(upload).error();
        gpu::CommandBatch batch(device);
        ATHENEA_TRY(batch.submit(true));
        ATHENEA_TRY((*decoder)->record(batch, 1, upload->source, upload->list, dst, p.header.runGaussians,
                                       p.header.pageRuns));
        gpu::GpuTimer timer;
        ATHENEA_TRY(timer.run(batch));
        auto errors = (*decoder)->takeErrors();
        if (!errors) return std::move(errors).error();
        if (*errors != 0) {
            return Error::make(ErrorCode::IoFailure, "{}: {} corrupt lanes", path.string(), *errors);
        }
        out.push_back({static_cast<uint32_t>(pages.size()), upload->bytes, decoded, timer.milliseconds(), single,
                       timer.fromDevice()});
        return ok();
    };
    // A first dispatch, not counted: the pipeline's first use.
    if (present > 0) {
        ATHENEA_TRY(timeOne({p.leafFirstPage}, true));
        out.clear();
    }
    for (uint32_t c = 0; c < std::min(singlePages, present); ++c) {
        ATHENEA_TRY(timeOne({p.leafFirstPage + c}, true));
    }
    std::vector<uint32_t> pages;
    uint64_t bytes = 0;
    for (uint32_t c = 0; c < present; ++c) {
        const uint32_t page = p.leafFirstPage + c;
        if (!pages.empty() && bytes + codedBytes(p, page) > batchBytes) {
            ATHENEA_TRY(timeOne(pages, false));
            pages.clear();
            bytes = 0;
        }
        pages.push_back(page);
        bytes += codedBytes(p, page);
    }
    if (!pages.empty()) {
        ATHENEA_TRY(timeOne(pages, false));
    }
    return out;
}

Result<std::array<std::vector<std::vector<uint8_t>>, 2>> athzBytePlanes(const std::filesystem::path& path) {
    auto parsed = parse(path);
    if (!parsed) return std::move(parsed).error();
    const Parsed& p = **parsed;
    const Asset& a = p.asset;
    const uint32_t R = p.header.runGaussians;
    std::array<std::vector<std::vector<uint8_t>>, 2> planes;
    for (uint32_t k = 0; k < kKinds; ++k) {
        planes[k].resize(a.planes);
        for (uint32_t b = 0; b < a.planes; ++b) {
            const PlaneEntry& e = a.planeEntries[k * a.planes + b];
            if ((e[0] & 0xFF) != 0 || ((e[0] >> 8) & 0xFF) != 8 || ((e[0] >> 16) & 0xFF) != 0) {
                return Error::make(ErrorCode::InvalidArgument, "{}: not written as byte planes", path.string());
            }
        }
    }
    for (uint32_t page = 0; page < p.header.pages; ++page) {
        const uint32_t k = page < p.leafFirstPage ? 0 : 1;
        const PageEntry& e = p.pages[page];
        for (uint32_t b = 0; b < a.planes; ++b) {
            const PlaneEntry& pe = a.planeEntries[k * a.planes + b];
            const uint32_t L = pe[0] >> 24;
            const auto* layer = reinterpret_cast<const uint8_t*>(pageBytes(p, page, L));
            for (uint32_t r = 0; r < p.header.pageRuns; ++r) {
                const uint32_t first = r * R;
                if (first >= e.count()) break;
                const uint32_t start = r == 0 ? 4u * p.header.pageRuns : getAt<uint32_t>(layer + 4 * (r - 1));
                const uint8_t* run = layer + start + 4 * pe[2];
                const uint32_t here = std::min(R, e.count() - first);
                planes[k][b].insert(planes[k][b].end(), run, run + here);
            }
        }
    }
    return planes;
}

Result<void> writeSpzOfCloud(gpu::ShaderLibrary& library, const scene::GpuSplats& cloud,
                             const std::filesystem::path& path, uint32_t version) {
    gpu::Device& device = library.device();
    if (cloud.count == 0) {
        return Error(ErrorCode::InvalidArgument, "an SPZ of no splats");
    }
    if (version == 4 && !io::writesSpzVersion4()) {
        return Error(ErrorCode::Unsupported, "this build writes SPZ version 4 only with ZSTD");
    }
    auto flat = gpu::ComputeKernel::create(library, "athenea/lod/athz_check", "athzFlat");
    if (!flat) return std::move(flat).error();
    auto records = deviceBuffer(device, uint64_t{cloud.count} * render::kFlatStride * 4, 4, "athz.flat");
    if (!records) return std::move(records).error();
    {
        gpu::CommandBatch batch(device);
        flat->dispatch(batch, {cloud.count, 1, 1}, [&](rhi::ShaderCursor c) {
            c["positions"].setBinding(cloud.positions.rhi());
            c["shape"].setBinding(cloud.shape.rhi());
            c["sh"].setBinding(cloud.sh.rhi());
            c["flat"].setBinding(records->rhi());
            c["a"].setBinding(cloud.positions.rhi());
            c["b"].setBinding(cloud.positions.rhi());
            c["counters"].setBinding(records->rhi());
            c["params"]["count"].setData(cloud.count);
            c["params"]["shWords"].setData(cloud.shWords);
            c["params"]["restPerColour"].setData(cloud.restPerColour);
            c["params"]["toSrgb"].setData(cloud.linear ? 1u : 0u);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    if (cloud.linear && cloud.restPerColour > 0) {
        log::warn("{}: harmonics in linear light written as they are; SPZ's viewers read them as sRGB",
                  path.string());
    }
    auto pack = render::FlattenPack::create(library);
    if (!pack) return std::move(pack).error();
    std::vector<render::FlatCloud> clouds{{0, cloud.count, false, *records}};
    constexpr float kLogScaleFloor = -12.0F;   // as `athenea flatten`
    auto stats = pack->stats(clouds, kLogScaleFloor);
    if (!stats) return std::move(stats).error();
    auto streams = pack->spz(clouds, stats->fractionalBits, 8, 8);
    if (!streams) return std::move(streams).error();
    io::SpzHeader header;
    header.version = version;
    header.count = cloud.count;
    header.shDegree = 3;
    header.fractionalBits = stats->fractionalBits;
    header.antialiased = cloud.antialiased.value_or(true);
    return io::writeSpz(path, header, streams->streams);
}

// --- a file as a chunk source ------------------------------------------------------

struct AthzSource::Impl {
    gpu::ShaderLibrary*      library = nullptr;
    std::unique_ptr<Parsed>  parsed;
    ChunkFormat              format;
    uint32_t                 layers = kAthzAllLayers;
    std::unique_ptr<Decoder> decoder;
};

AthzSource::AthzSource() : impl_(std::make_unique<Impl>()) {}
AthzSource::~AthzSource() = default;

Result<std::unique_ptr<AthzSource>> AthzSource::open(gpu::ShaderLibrary& library, const std::filesystem::path& path,
                                                     uint32_t layers) {
    std::unique_ptr<AthzSource> source(new AthzSource());
    Impl& s = *source->impl_;
    s.library = &library;
    auto parsed = parse(path);
    if (!parsed) return std::move(parsed).error();
    s.parsed = std::move(*parsed);
    s.layers = (layers | kAthzGeometry) & s.parsed->header.layerMask;
    s.format = s.parsed->asset.format;
    // Staging holds every layer a page has (a loader copies them all, as
    // its CRC covers them all).
    for (uint32_t page = s.parsed->leafFirstPage; page < s.parsed->header.pages; ++page) {
        s.format.codedBytes = std::max(s.format.codedBytes, codedBytes(*s.parsed, page));
    }
    s.format.codedBytes = std::max<uint64_t>(s.format.codedBytes, 4);
    return source;
}

const ChunkFormat& AthzSource::format() const noexcept {
    return impl_->format;
}

uint32_t AthzSource::chunkCount() const noexcept {
    return impl_->parsed->leafPages;
}

bool AthzSource::available(uint32_t chunk) const noexcept {
    return impl_->parsed->leafFirstPage + chunk < impl_->parsed->header.pages;
}

scene::Bounds AthzSource::bounds(uint32_t /*chunk*/) const {
    return impl_->parsed->asset.bounds;
}

std::string AthzSource::name() const {
    return impl_->parsed->path.filename().string();
}

Result<LodCloud> AthzSource::openCloud(gpu::Device& device) {
    Impl& s = *impl_;
    auto decoder = Decoder::create(*s.library, s.parsed->asset, s.parsed->header.layerMask, s.layers);
    if (!decoder) return std::move(decoder).error();
    s.decoder = std::move(*decoder);
    return levelsOf(device, *s.decoder, *s.parsed);
}

ChunkRequest AthzSource::request(uint32_t chunk) const {
    const Parsed& p = *impl_->parsed;
    const uint32_t page = p.leafFirstPage + chunk;
    if (page >= p.header.pages) {
        return {chunk, 0, 0, 0};
    }
    return {chunk, 0, p.pages[page].count(), codedBytes(p, page)};
}

Result<void> AthzSource::fetch(const ChunkRequest& request, std::span<std::byte> staging) {
    const Parsed& p = *impl_->parsed;
    const uint32_t page = p.leafFirstPage + request.chunk;
    if (page >= p.header.pages) {
        return Error::make(ErrorCode::InvalidArgument, "{}: no page for chunk {}", name(), request.chunk);
    }
    if (staging.size() != codedBytes(p, page)) {
        return Error::make(ErrorCode::InvalidArgument, "{}: chunk {} is {} bytes, staged into {}", name(), request.chunk,
                           codedBytes(p, page), staging.size());
    }
    ATHENEA_TRY(checkPage(p, page));
    uint64_t at = 0;
    for (uint32_t L = 0; L < 4; ++L) {
        const uint32_t n = p.pages[page].bytes[L];
        std::memcpy(staging.data() + at, pageBytes(p, page, L), n);
        at += n;
    }
    return ok();
}

Result<void> AthzSource::decode(gpu::CommandBatch& batch, const gpu::Buffer& staging,
                                std::span<const StagedChunk> chunks, LodCloud& cloud) {
    Impl& s = *impl_;
    const Parsed& p = *s.parsed;
    if (!s.decoder) {
        return Error::make(ErrorCode::InvalidArgument, "{}: decode before openCloud", name());
    }
    std::vector<uint32_t> list;
    for (const StagedChunk& c : chunks) {
        const uint32_t page = p.leafFirstPage + c.request.chunk;
        uint64_t at = c.offset;
        std::array<uint32_t, 4> src{kNone, kNone, kNone, kNone};
        for (uint32_t L = 0; L < 4; ++L) {
            if (((s.layers >> L) & 1u) != 0) {
                src[L] = static_cast<uint32_t>(at);
            }
            at += p.pages[page].bytes[L];
        }
        list.insert(list.end(), src.begin(), src.end());
        list.push_back(c.slot * cloud.chunkSplats);
        list.push_back(p.pages[page].count());
        list.push_back(0);
        list.push_back(0);
    }
    return s.decoder->record(batch, 1, staging, list, arraysOf(cloud.splats, cloud.groups), p.header.runGaussians,
                             p.header.pageRuns);
}

}   // namespace athenea::lod
