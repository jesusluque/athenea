// Copyright (c) 2026 jesus luque.
#include "athenea/lod/Athc.h"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include "athenea/core/FrameThread.h"
#include "athenea/core/Log.h"
#include "athenea/core/Platform.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"

namespace athenea::lod {
namespace {

constexpr uint64_t kPage = 4096;
/// 2: `flags` says what a block carries besides its four arrays (the shading
/// normals, `kHasNormals`). A version 1 file has nothing besides, and its
/// `flags` is the zero its padding always was, so it reads unchanged.
/// A file is written as version 2 if and only if its `flags` is not zero:
/// one that carries nothing besides is a version 1 file, and says so, so a
/// reader that knows only version 1 opens it.
constexpr uint32_t kVersion = 2;
constexpr uint32_t kOldestVersion = 1;
constexpr uint32_t kHasNormals = 1;
/// Bit 1: the colours are linear light (`GpuSplats::linear`), which is
/// nothing a block carries but how every block's colours are read. Clear in
/// every file written before it, which were all captures' or encoded.
constexpr uint32_t kLinear = 2;
/// Bit 2: every block ends with the emitted radiance, one RGB9E5 word an
/// element, after the normals where those are there too. (Bit 1 is taken.)
constexpr uint32_t kHasEmission = 4;
/// Every bit this reader knows. A bit outside it is refused, not ignored: a
/// later bit may add a block, and a reader that skipped it would take every
/// block after it from the wrong place.
/// Bit 4: every block then carries the material (proposal 026): one `pbr`
/// word an element and, where the extra header says, the three of its layers.
constexpr uint32_t kHasMaterial = 16;
/// Bit 5: and a transfer (task TX): its words an element, then its open
/// directions' -- the counts in the extra header. Bit 3 is proposal 009's.
constexpr uint32_t kHasTransfer = 32;
constexpr uint32_t kKnownFlags = kHasNormals | kLinear | kHasEmission | kHasMaterial | kHasTransfer;
constexpr char     kMagic[4] = {'A', 'T', 'H', 'C'};
/// Uploads staged before a submit: the staging heap holds them until then.
constexpr uint64_t kStageBytes = uint64_t{256} << 20;

struct FileHeader {
    char     magic[4];
    uint32_t version;
    uint32_t count;           // splats
    uint32_t restPerColour;
    uint32_t shWords;
    uint32_t levels;
    uint32_t chunkSplats;
    uint32_t chunks;
    uint32_t finestGroups;
    float    boundsLo[3];
    float    extent;
    float    boundsMin[3];
    float    boundsMax[3];
    uint32_t flags;           // version 2: kHasNormals; 0 in a version 1 file
    uint64_t levelTable;      // LevelEntry[levels]
    uint64_t chunkTable;      // ChunkEntry[chunks]
    uint64_t starts;          // uint[finestGroups]
};
struct LevelEntry {
    uint32_t level;
    uint32_t groups;
    uint64_t offset;
};
struct ChunkEntry {
    uint64_t offset;
    uint32_t count;
    uint32_t pad;
};
static_assert(sizeof(FileHeader) == 104);
/// After the header, where bit 4 or 5 is set: how many words of each extra a
/// block element carries, in the order they follow the emission.
struct ExtraHeader {
    uint32_t pbrWords;        // 1 with bit 4
    uint32_t lobesWords;      // 3 where the cloud carries its layers, else 0
    uint32_t transferCount;   // the transfer's values (its layout: 9, 36, 84, 16, 64, 112 or 10)
    uint32_t transferWords;   // f16 pairs of them
    uint32_t shadowWords;     // its open directions: 2, 8 or 32
    uint32_t pad[3];
};
static_assert(sizeof(ExtraHeader) == 32);
constexpr uint64_t kExtraHeaderAt = sizeof(FileHeader);

/// A member of a cloud, as a pointer: named once, since MSVC does not parse
/// `gpu::Buffer scene::GpuSplats::*` as a template argument.
using CloudMember = gpu::Buffer scene::GpuSplats::*;

/// Each extra a block element carries, in order: the member of a cloud it
/// fills and its words.
std::vector<std::pair<CloudMember, uint32_t>> extrasOf(const ExtraHeader& x) {
    std::vector<std::pair<CloudMember, uint32_t>> out;
    if (x.pbrWords) out.emplace_back(&scene::GpuSplats::pbr, x.pbrWords);
    if (x.lobesWords) out.emplace_back(&scene::GpuSplats::lobes, x.lobesWords);
    if (x.transferWords) out.emplace_back(&scene::GpuSplats::transfer, x.transferWords);
    if (x.shadowWords) out.emplace_back(&scene::GpuSplats::shadowBits, x.shadowWords);
    return out;
}

uint32_t extraWords(const ExtraHeader& x) {
    return x.pbrWords + x.lobesWords + x.transferWords + x.shadowWords;
}

/// Bytes an element of a block takes, the extras included.
uint64_t elementBytes(uint32_t shWords, bool normals, bool emission);
bool keepsNormals(const FileHeader& h);
bool keepsEmission(const FileHeader& h);
uint64_t elementBytesOf(const FileHeader& h, const ExtraHeader& x) {
    return elementBytes(h.shWords, keepsNormals(h), keepsEmission(h)) + 4 * uint64_t{extraWords(x)};
}
static_assert(sizeof(LevelEntry) == 16);
static_assert(sizeof(ChunkEntry) == 16);

uint64_t aligned(uint64_t at) {
    return (at + kPage - 1) / kPage * kPage;
}

/// Bytes a level group or a splat takes: position, shape, SH, a uint, and
/// the shading normal's word where the file keeps one.
uint64_t elementBytes(uint32_t shWords, bool normals, bool emission) {
    return 16 + 16 + 4 * uint64_t{shWords} + 4 + (normals ? 4 : 0) + (emission ? 4 : 0);
}

bool keepsNormals(const FileHeader& h) {
    return (h.flags & kHasNormals) != 0;
}

bool keepsEmission(const FileHeader& h) {
    return (h.flags & kHasEmission) != 0;
}

struct Layout {
    FileHeader              header{};
    ExtraHeader             extra{};
    std::vector<LevelEntry> levels;
    std::vector<ChunkEntry> chunks;
};

Error bad(const std::filesystem::path& path, const std::string& why) {
    return Error(ErrorCode::IoFailure, path.string() + ": not a readable .athc (" + why + ")");
}

Result<Layout> parse(const platform::MappedFile& file, const std::filesystem::path& path) {
    const std::span<const std::byte> bytes = file.bytes();
    Layout layout;
    FileHeader& h = layout.header;
    if (bytes.size() < kPage) {
        return bad(path, "shorter than its header");
    }
    std::memcpy(&h, bytes.data(), sizeof h);
    if (std::memcmp(h.magic, kMagic, 4) != 0) {
        return bad(path, "no ATHC magic");
    }
    if (h.version < kOldestVersion || h.version > kVersion) {
        return bad(path, "version " + std::to_string(h.version) + ", this reads " + std::to_string(kOldestVersion) +
                             " to " + std::to_string(kVersion));
    }
    if (h.version < 2) {
        h.flags = 0;
    }
    if ((h.flags & (kHasMaterial | kHasTransfer)) != 0) {
        std::memcpy(&layout.extra, bytes.data() + kExtraHeaderAt, sizeof layout.extra);
        const ExtraHeader& x = layout.extra;
        if (((h.flags & kHasMaterial) != 0) != (x.pbrWords == 1) || (x.lobesWords != 0 && x.lobesWords != 3) ||
            ((h.flags & kHasTransfer) != 0) != (x.transferWords != 0) || x.transferWords > 64 ||
            (x.shadowWords != 0 && x.shadowWords != 2 && x.shadowWords != 8 && x.shadowWords != 32) ||
            x.transferWords != (x.transferCount + 1) / 2) {
            return bad(path, "inconsistent extra header");
        }
    }
    if ((h.flags & ~kKnownFlags) != 0) {
        std::string bits;
        for (uint32_t b = 0; b < 32; ++b) {
            if ((h.flags & ~kKnownFlags & (1u << b)) != 0) {
                bits += (bits.empty() ? "" : ", ") + std::to_string(b);
            }
        }
        return bad(path, "unknown flag bits " + bits +
                             "; this reads bits 0 (normals), 1 (linear), 2 (emission), 4 (material) and 5 (transfer)");
    }
    const auto within = [&](uint64_t offset, uint64_t size) {
        return offset <= bytes.size() && size <= bytes.size() - offset;
    };
    if (h.count == 0 || h.levels == 0 || h.chunkSplats == 0 || h.shWords == 0 || h.shWords > 64 ||
        h.chunks != (uint64_t{h.count} + h.chunkSplats - 1) / h.chunkSplats) {
        return bad(path, "inconsistent counts");
    }
    if (!within(h.levelTable, uint64_t{h.levels} * sizeof(LevelEntry)) ||
        !within(h.chunkTable, uint64_t{h.chunks} * sizeof(ChunkEntry)) ||
        !within(h.starts, uint64_t{h.finestGroups} * 4)) {
        return bad(path, "tables past the end");
    }
    layout.levels.resize(h.levels);
    layout.chunks.resize(h.chunks);
    std::memcpy(layout.levels.data(), bytes.data() + h.levelTable, layout.levels.size() * sizeof(LevelEntry));
    std::memcpy(layout.chunks.data(), bytes.data() + h.chunkTable, layout.chunks.size() * sizeof(ChunkEntry));
    const uint64_t per = elementBytesOf(h, layout.extra);
    for (size_t l = 0; l < layout.levels.size(); ++l) {
        const LevelEntry& e = layout.levels[l];
        if (e.groups == 0 || !within(e.offset, e.groups * per) ||
            (l > 0 && e.level != layout.levels[l - 1].level + 1) || e.level == 0 || e.level > 10) {
            return bad(path, "level " + std::to_string(l));
        }
    }
    if (layout.levels.back().groups != h.finestGroups) {
        return bad(path, "finest level's groups");
    }
    for (uint32_t c = 0; c < h.chunks; ++c) {
        const ChunkEntry& e = layout.chunks[c];
        const uint32_t expected = std::min(h.chunkSplats, h.count - c * h.chunkSplats);
        if (e.count != expected || !within(e.offset, e.count * per)) {
            return bad(path, "chunk " + std::to_string(c));
        }
    }
    return layout;
}

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                                 const void* initial = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, count > 0 ? initial : nullptr);
}

/// Where a block's arrays start, in bytes from the block: four, the shading
/// normals after them where the file keeps them, the emitted radiance after
/// those, then the extras (bits 4 and 5) in the extra header's order.
struct BlockOffsets {
    uint64_t positions = 0, shape = 0, sh = 0, tail = 0, normals = 0, emission = 0;
    bool     hasNormals = false, hasEmission = false;
    std::vector<uint64_t> extras;
};
BlockOffsets offsetsOf(uint64_t n, uint32_t shWords, bool normals, bool emission, const ExtraHeader& x) {
    BlockOffsets b;
    b.shape = n * 16;
    b.sh = b.shape + n * 16;
    b.tail = b.sh + n * 4 * uint64_t{shWords};
    b.hasNormals = normals;
    b.hasEmission = emission;
    b.normals = b.tail + n * 4;
    b.emission = b.normals + (normals ? n * 4 : 0);
    uint64_t next = b.emission + (emission ? n * 4 : 0);
    for (const auto& [member, words] : extrasOf(x)) {
        b.extras.push_back(next);
        next += n * 4 * uint64_t{words};
    }
    return b;
}

/// A block's arrays, where they are in the mapping: four, the shading
/// normals after them where the file keeps them (`normals` null otherwise),
/// and the emitted radiance after those (`emission` null otherwise).
struct Block {
    const std::byte* positions, *shape, *sh, *tail, *normals, *emission;
    /// The extras after them (bits 4 and 5), in the extra header's order.
    std::vector<const std::byte*> extras;
};
Block blockAt(const std::byte* at, uint64_t n, uint32_t shWords, bool normals, bool emission,
              const ExtraHeader& x = {}) {
    const BlockOffsets o = offsetsOf(n, shWords, normals, emission, x);
    Block b;
    b.positions = at + o.positions;
    b.shape = at + o.shape;
    b.sh = at + o.sh;
    b.tail = at + o.tail;
    b.normals = normals ? at + o.normals : nullptr;
    b.emission = emission ? at + o.emission : nullptr;
    for (const uint64_t e : o.extras) {
        b.extras.push_back(at + e);
    }
    return b;
}

/// What every chunk of a file holds, and back.
ChunkFormat formatOf(const Layout& layout) {
    const FileHeader& h = layout.header;
    const ExtraHeader& x = layout.extra;
    ChunkFormat f;
    f.chunkSplats = h.chunkSplats;
    f.restPerColour = h.restPerColour;
    f.shWords = h.shWords;
    f.normals = keepsNormals(h);
    f.emission = keepsEmission(h);
    f.linear = (h.flags & kLinear) != 0;
    f.pbrWords = x.pbrWords;
    f.lobesWords = x.lobesWords;
    f.transferCount = x.transferCount;
    f.transferWords = x.transferWords;
    f.shadowWords = x.shadowWords;
    return f;
}
ExtraHeader extraOf(const ChunkFormat& f) {
    ExtraHeader x{};
    x.pbrWords = f.pbrWords;
    x.lobesWords = f.lobesWords;
    x.transferCount = f.transferCount;
    x.transferWords = f.transferWords;
    x.shadowWords = f.shadowWords;
    return x;
}

/// The extra header's counts on a cloud that carries them.
void countsOf(const ExtraHeader& x, scene::GpuSplats& s) {
    s.transferCount = x.transferCount;
    s.transferWords = x.transferWords;
    s.shadowWords = x.shadowWords;
}

scene::GpuSplats splatsLike(const ChunkFormat& f, const scene::Bounds& bounds, const std::string& name) {
    scene::GpuSplats s;
    s.source = name;
    s.restPerColour = f.restPerColour;
    s.shWords = f.shWords;
    s.linear = f.linear;
    s.bounds = bounds;
    return s;
}

scene::Bounds boundsOf(const FileHeader& h) {
    scene::Bounds b;
    std::copy(h.boundsMin, h.boundsMin + 3, b.min.begin());
    std::copy(h.boundsMax, h.boundsMax + 3, b.max.begin());
    return b;
}

scene::GpuSplats splatsLike(const FileHeader& h, const ExtraHeader& x, const std::filesystem::path& path) {
    Layout layout;
    layout.header = h;
    layout.extra = x;
    return splatsLike(formatOf(layout), boundsOf(h), path.filename().string());
}

/// The cloud but its chunks: header fields, levels and starts on the device.
Result<LodCloud> cloudOf(gpu::Device& device, const platform::MappedFile& file, const Layout& layout,
                           const std::filesystem::path& path) {
    const FileHeader& h = layout.header;
    LodCloud lod;
    lod.count = h.count;
    lod.chunkSplats = h.chunkSplats;
    std::copy(h.boundsLo, h.boundsLo + 3, lod.boundsLo);
    lod.extent = h.extent;
    const std::byte* base = file.bytes().data();
    for (const LevelEntry& e : layout.levels) {
        const Block b = blockAt(base + e.offset, e.groups, h.shWords, keepsNormals(h), keepsEmission(h), layout.extra);
        LodLevel level;
        level.level = e.level;
        level.gaussians = splatsLike(h, layout.extra, path);
        level.gaussians.count = e.groups;
        level.gaussians.declared = e.groups;
        auto p = deviceBuffer(device, e.groups, 16, "athc.level", b.positions);
        if (!p) return std::move(p).error();
        auto s = deviceBuffer(device, uint64_t{e.groups} * 4, 4, "athc.level", b.shape);
        if (!s) return std::move(s).error();
        auto sh = deviceBuffer(device, uint64_t{e.groups} * h.shWords, 4, "athc.level", b.sh);
        if (!sh) return std::move(sh).error();
        auto cells = deviceBuffer(device, e.groups, 4, "athc.cells", b.tail);
        if (!cells) return std::move(cells).error();
        level.gaussians.positions = std::move(*p);
        level.gaussians.shape = std::move(*s);
        level.gaussians.sh = std::move(*sh);
        if (b.normals != nullptr) {
            auto normals = deviceBuffer(device, e.groups, 4, "athc.level", b.normals);
            if (!normals) return std::move(normals).error();
            level.gaussians.normals = std::move(*normals);
        }
        if (b.emission != nullptr) {
            auto emission = deviceBuffer(device, e.groups, 4, "athc.level", b.emission);
            if (!emission) return std::move(emission).error();
            level.gaussians.emission = std::move(*emission);
        }
        {
            size_t k = 0;
            for (const auto& [member, words] : extrasOf(layout.extra)) {
                auto extra = deviceBuffer(device, uint64_t{e.groups} * words, 4, "athc.level", b.extras[k++]);
                if (!extra) return std::move(extra).error();
                level.gaussians.*member = std::move(*extra);
            }
            countsOf(layout.extra, level.gaussians);
        }
        level.cells = std::move(*cells);
        lod.levels.push_back(std::move(level));
    }
    auto starts = deviceBuffer(device, h.finestGroups, 4, "athc.starts", base + h.starts);
    if (!starts) return std::move(starts).error();
    lod.starts = std::move(*starts);
    return lod;
}

/// A store of `slots` chunks of `format`, and the per-chunk flags, all empty.
/// `count` is the cloud's splats (its chunks', whole); `chunks` its chunks.
Result<void> makeStore(gpu::Device& device, LodCloud& lod, const ChunkFormat& format, uint64_t count,
                       uint32_t chunks, uint32_t slots, const scene::Bounds& bounds, const std::string& name) {
    const ExtraHeader x = extraOf(format);
    const uint64_t n = uint64_t{slots} * format.chunkSplats;
    lod.splats = splatsLike(format, bounds, name);
    lod.splats.count = static_cast<uint32_t>(std::min<uint64_t>(n, count));
    lod.splats.declared = static_cast<uint32_t>(count);
    auto p = deviceBuffer(device, n, 16, "athc.store");
    if (!p) return std::move(p).error();
    auto s = deviceBuffer(device, n * 4, 4, "athc.store");
    if (!s) return std::move(s).error();
    auto sh = deviceBuffer(device, n * format.shWords, 4, "athc.store");
    if (!sh) return std::move(sh).error();
    auto g = deviceBuffer(device, n, 4, "athc.groups");
    if (!g) return std::move(g).error();
    lod.splats.positions = std::move(*p);
    lod.splats.shape = std::move(*s);
    lod.splats.sh = std::move(*sh);
    if (format.normals) {
        auto normals = deviceBuffer(device, n, 4, "athc.store");
        if (!normals) return std::move(normals).error();
        lod.splats.normals = std::move(*normals);
    }
    if (format.emission) {
        auto emission = deviceBuffer(device, n, 4, "athc.store");
        if (!emission) return std::move(emission).error();
        lod.splats.emission = std::move(*emission);
    }
    for (const auto& [member, words] : extrasOf(x)) {
        auto extra = deviceBuffer(device, n * words, 4, "athc.store");
        if (!extra) return std::move(extra).error();
        lod.splats.*member = std::move(*extra);
    }
    countsOf(x, lod.splats);
    lod.groups = std::move(*g);
    lod.slots.assign(chunks, -1);
    const std::vector<uint32_t> none(chunks, 0);
    auto resident = gpu::Buffer::fromSpan(device, std::span<const uint32_t>(none), "athc.resident");
    if (!resident) return std::move(resident).error();
    lod.resident = std::move(*resident);
    return ok();
}

Result<void> makeStore(gpu::Device& device, LodCloud& lod, const Layout& layout, uint32_t slots,
                       const std::filesystem::path& path) {
    const FileHeader& h = layout.header;
    return makeStore(device, lod, formatOf(layout), h.count, h.chunks, slots, boundsOf(h), path.filename().string());
}

/// Records the copies of a chunk of `n` splats, staged at `from` in
/// `staging`, into slot `slot`: device copies, nothing on the CPU.
void placeStaged(gpu::CommandBatch& batch, LodCloud& lod, const ExtraHeader& x, uint32_t slot, uint64_t n,
                 rhi::IBuffer* staging, uint64_t from) {
    const uint64_t at = uint64_t{slot} * lod.chunkSplats;
    const uint32_t shWords = lod.splats.shWords;
    const BlockOffsets b = offsetsOf(n, shWords, lod.splats.hasNormals(), lod.splats.hasEmission(), x);
    rhi::ICommandEncoder* e = batch.encoder();
    e->copyBuffer(lod.splats.positions.rhi(), at * 16, staging, from + b.positions, n * 16);
    e->copyBuffer(lod.splats.shape.rhi(), at * 16, staging, from + b.shape, n * 16);
    e->copyBuffer(lod.splats.sh.rhi(), at * 4 * shWords, staging, from + b.sh, n * 4 * shWords);
    e->copyBuffer(lod.groups.rhi(), at * 4, staging, from + b.tail, n * 4);
    if (b.hasNormals) {
        e->copyBuffer(lod.splats.normals.rhi(), at * 4, staging, from + b.normals, n * 4);
    }
    if (b.hasEmission) {
        e->copyBuffer(lod.splats.emission.rhi(), at * 4, staging, from + b.emission, n * 4);
    }
    size_t k = 0;
    for (const auto& [member, words] : extrasOf(x)) {
        e->copyBuffer((lod.splats.*member).rhi(), at * 4 * words, staging, from + b.extras[k++], n * 4 * words);
    }
    batch.markDirty();
}

void setResident(gpu::CommandBatch& batch, LodCloud& lod, uint32_t chunk, bool resident) {
    const uint32_t value = resident ? 1 : 0;
    batch.encoder()->uploadBufferData(lod.resident.rhi(), uint64_t{chunk} * 4, 4, &value);
    batch.markDirty();
}

/// Records chunk `chunk`'s bytes (one block, `from`) into slot `slot`.
void place(gpu::CommandBatch& batch, LodCloud& lod, uint32_t chunk, uint32_t slot, const std::byte* from,
           const ExtraHeader& x) {
    const uint64_t n = lod.chunkCount(chunk);
    const uint64_t at = uint64_t{slot} * lod.chunkSplats;
    const Block b = blockAt(from, n, lod.splats.shWords, lod.splats.hasNormals(), lod.splats.hasEmission(), x);
    rhi::ICommandEncoder* e = batch.encoder();
    e->uploadBufferData(lod.splats.positions.rhi(), at * 16, n * 16, b.positions);
    e->uploadBufferData(lod.splats.shape.rhi(), at * 16, n * 16, b.shape);
    e->uploadBufferData(lod.splats.sh.rhi(), at * 4 * lod.splats.shWords, n * 4 * lod.splats.shWords, b.sh);
    e->uploadBufferData(lod.groups.rhi(), at * 4, n * 4, b.tail);
    if (b.normals != nullptr) {
        e->uploadBufferData(lod.splats.normals.rhi(), at * 4, n * 4, b.normals);
    }
    if (b.emission != nullptr) {
        e->uploadBufferData(lod.splats.emission.rhi(), at * 4, n * 4, b.emission);
    }
    {
        size_t k = 0;
        for (const auto& [member, words] : extrasOf(x)) {
            e->uploadBufferData((lod.splats.*member).rhi(), at * 4 * words, n * 4 * words, b.extras[k++]);
        }
    }
    const uint32_t one = 1;
    e->uploadBufferData(lod.resident.rhi(), uint64_t{chunk} * 4, 4, &one);
    batch.markDirty();
    lod.slots[chunk] = static_cast<int32_t>(slot);
}

}   // namespace

bool isAthc(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext == ".athc";
}

Result<void> writeAthc(gpu::Device& device, const LodCloud& cloud, const std::filesystem::path& path) {
    if (cloud.levels.empty() || cloud.count == 0) {
        return Error(ErrorCode::InvalidArgument, "a .athc holds a cloud with levels of detail");
    }
    for (uint32_t c = 0; c < cloud.chunks(); ++c) {
        if (cloud.slots[c] < 0) {
            return Error(ErrorCode::InvalidArgument, "writing a cloud whose chunks are not all on the device");
        }
    }
    const uint32_t shWords = cloud.splats.shWords;
    // Normals are written only where every level keeps them as the splats do.
    const bool normals = cloud.splats.hasNormals() &&
                         std::all_of(cloud.levels.begin(), cloud.levels.end(),
                                     [](const LodLevel& l) { return l.gaussians.hasNormals(); });
    // And the emission likewise.
    const bool emission = cloud.splats.hasEmission() &&
                          std::all_of(cloud.levels.begin(), cloud.levels.end(),
                                      [](const LodLevel& l) { return l.gaussians.hasEmission(); });
    // The material and the transfer, where every level carries them as the
    // splats do (bits 4 and 5).
    ExtraHeader x{};
    const auto everywhere = [&](CloudMember member) {
        return (cloud.splats.*member).valid() &&
               std::all_of(cloud.levels.begin(), cloud.levels.end(),
                           [&](const LodLevel& l) { return (l.gaussians.*member).valid(); });
    };
    if (everywhere(&scene::GpuSplats::pbr)) {
        x.pbrWords = 1;
        x.lobesWords = everywhere(&scene::GpuSplats::lobes) ? 3 : 0;
    }
    if (everywhere(&scene::GpuSplats::transfer) && cloud.splats.transferWords > 0) {
        x.transferCount = cloud.splats.transferCount;
        x.transferWords = cloud.splats.transferWords;
        x.shadowWords = everywhere(&scene::GpuSplats::shadowBits) ? cloud.splats.shadowWords : 0;
    }
    const uint64_t per = elementBytes(shWords, normals, emission) + 4 * uint64_t{extraWords(x)};
    FileHeader h{};
    std::memcpy(h.magic, kMagic, 4);
    h.version = kVersion;
    h.flags = (normals ? kHasNormals : 0u) | (cloud.splats.linear ? kLinear : 0u) |
              (emission ? kHasEmission : 0u) | (x.pbrWords ? kHasMaterial : 0u) | (x.transferWords ? kHasTransfer : 0u);
    h.count = cloud.count;
    h.restPerColour = cloud.splats.restPerColour;
    h.shWords = shWords;
    h.levels = static_cast<uint32_t>(cloud.levels.size());
    h.chunkSplats = cloud.chunkSplats;
    h.chunks = cloud.chunks();
    h.finestGroups = cloud.levels.back().gaussians.count;
    std::copy(cloud.boundsLo, cloud.boundsLo + 3, h.boundsLo);
    h.extent = cloud.extent;
    std::copy(cloud.splats.bounds.min.begin(), cloud.splats.bounds.min.end(), h.boundsMin);
    std::copy(cloud.splats.bounds.max.begin(), cloud.splats.bounds.max.end(), h.boundsMax);
    // Version 2 if and only if a block carries something besides.
    h.version = h.flags != 0 ? kVersion : kOldestVersion;
    h.levelTable = kPage;
    h.chunkTable = h.levelTable + uint64_t{h.levels} * sizeof(LevelEntry);
    h.starts = aligned(h.chunkTable + uint64_t{h.chunks} * sizeof(ChunkEntry));
    uint64_t at = aligned(h.starts + uint64_t{h.finestGroups} * 4);
    std::vector<LevelEntry> levels;
    for (const LodLevel& l : cloud.levels) {
        levels.push_back({l.level, l.gaussians.count, at});
        at = aligned(at + l.gaussians.count * per);
    }
    std::vector<ChunkEntry> chunks;
    for (uint32_t c = 0; c < h.chunks; ++c) {
        chunks.push_back({at, cloud.chunkCount(c), 0});
        at = aligned(at + cloud.chunkCount(c) * per);
    }

    // Beside the destination first: a failed write never leaves a file that
    // looks whole.
    std::filesystem::path partial = path;
    partial += ".partial";
    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Error(ErrorCode::IoFailure, "cannot write " + partial.string());
    }
    uint64_t written = 0;
    std::vector<char> bytes;
    const auto put = [&](const void* data, uint64_t size) {
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        written += size;
    };
    const auto padTo = [&](uint64_t offset) {
        static const char zeros[kPage] = {};
        while (written < offset) {
            put(zeros, std::min<uint64_t>(kPage, offset - written));
        }
    };
    const auto copy = [&](const gpu::Buffer& from, uint64_t offset, uint64_t size) -> Result<void> {
        bytes.resize(size);
        ATHENEA_TRY(from.read(device, offset, size, bytes.data()));
        put(bytes.data(), size);
        return ok();
    };
    put(&h, sizeof h);
    if ((h.flags & (kHasMaterial | kHasTransfer)) != 0) {
        put(&x, sizeof x);
    }
    padTo(h.levelTable);
    put(levels.data(), levels.size() * sizeof(LevelEntry));
    put(chunks.data(), chunks.size() * sizeof(ChunkEntry));
    padTo(h.starts);
    ATHENEA_TRY(copy(cloud.starts, 0, uint64_t{h.finestGroups} * 4));
    for (size_t l = 0; l < cloud.levels.size(); ++l) {
        const LodLevel& level = cloud.levels[l];
        const uint64_t n = level.gaussians.count;
        padTo(levels[l].offset);
        ATHENEA_TRY(copy(level.gaussians.positions, 0, n * 16));
        ATHENEA_TRY(copy(level.gaussians.shape, 0, n * 16));
        ATHENEA_TRY(copy(level.gaussians.sh, 0, n * 4 * shWords));
        ATHENEA_TRY(copy(level.cells, 0, n * 4));
        if (normals) {
            ATHENEA_TRY(copy(level.gaussians.normals, 0, n * 4));
        }
        if (emission) {
            ATHENEA_TRY(copy(level.gaussians.emission, 0, n * 4));
        }
        for (const auto& [member, words] : extrasOf(x)) {
            ATHENEA_TRY(copy(level.gaussians.*member, 0, n * 4 * words));
        }
    }
    for (uint32_t c = 0; c < h.chunks; ++c) {
        const uint64_t n = chunks[c].count;
        const uint64_t first = static_cast<uint64_t>(cloud.slots[c]) * cloud.chunkSplats;
        padTo(chunks[c].offset);
        ATHENEA_TRY(copy(cloud.splats.positions, first * 16, n * 16));
        ATHENEA_TRY(copy(cloud.splats.shape, first * 16, n * 16));
        ATHENEA_TRY(copy(cloud.splats.sh, first * 4 * shWords, n * 4 * shWords));
        ATHENEA_TRY(copy(cloud.groups, first * 4, n * 4));
        if (normals) {
            ATHENEA_TRY(copy(cloud.splats.normals, first * 4, n * 4));
        }
        if (emission) {
            ATHENEA_TRY(copy(cloud.splats.emission, first * 4, n * 4));
        }
        for (const auto& [member, words] : extrasOf(x)) {
            ATHENEA_TRY(copy(cloud.splats.*member, first * 4 * words, n * 4 * words));
        }
    }
    padTo(aligned(written));
    out.close();
    if (!out) {
        return Error(ErrorCode::IoFailure, "writing " + partial.string() + " failed");
    }
    std::error_code ec;
    std::filesystem::rename(partial, path, ec);
    if (ec) {
        return Error(ErrorCode::IoFailure, "cannot move " + partial.string() + " to " + path.string() + ": " +
                                             ec.message());
    }
    log::info("{}: {} splats in {} chunks, {} levels, {:.1f} MB", path.string(), h.count, h.chunks, h.levels,
              static_cast<double>(written) / (1 << 20));
    return ok();
}

Result<bool> migrateLrtc(const std::filesystem::path& in, const std::filesystem::path& out) {
    std::error_code ec;
    if (std::filesystem::equivalent(in, out, ec) ||
        std::filesystem::weakly_canonical(in, ec) == std::filesystem::weakly_canonical(out, ec)) {
        return Error(ErrorCode::InvalidArgument, out.string() + " is the input; a migration writes beside it");
    }
    auto file = platform::MappedFile::open(in);
    if (!file) return std::move(file).error();
    const std::span<const std::byte> bytes = file->bytes();
    if (bytes.size() < kPage) {
        return bad(in, "shorter than its header");
    }
    FileHeader h{};
    std::memcpy(&h, bytes.data(), sizeof h);
    // lucabRTrender's .lrtc is version 1 of this layout under another magic
    // (its Lrtc.cpp differs from Athc.cpp at e8ef1eb in the magic alone): the
    // header's last word was padding it always wrote as zero, which is
    // version 2's `flags` saying no normals. Only the header changes.
    constexpr char kOldMagic[4] = {'L', 'R', 'T', 'C'};
    bool converted = false;
    if (std::memcmp(h.magic, kOldMagic, 4) == 0) {
        if (h.version != 1) {
            return bad(in, ".lrtc version " + std::to_string(h.version) + "; lucabRTrender wrote version 1 only");
        }
        std::memcpy(h.magic, kMagic, 4);
        h.flags = 0;
        h.version = kOldestVersion;   // nothing besides, so version 1
        converted = true;
    } else if (std::memcmp(h.magic, kMagic, 4) != 0) {
        return bad(in, "neither LRTC nor ATHC magic");
    }
    std::filesystem::path partial = out;
    partial += ".partial";
    {
        std::ofstream o(partial, std::ios::binary | std::ios::trunc);
        if (!o) {
            return Error(ErrorCode::IoFailure, "cannot write " + partial.string());
        }
        o.write(reinterpret_cast<const char*>(&h), sizeof h);
        // The payload as it is, in slices: nothing in it is decoded.
        constexpr uint64_t kSlice = uint64_t{64} << 20;
        for (uint64_t at = sizeof h; at < bytes.size(); at += kSlice) {
            const uint64_t n = std::min<uint64_t>(kSlice, bytes.size() - at);
            o.write(reinterpret_cast<const char*>(bytes.data() + at), static_cast<std::streamsize>(n));
        }
        o.close();
        if (!o) {
            return Error(ErrorCode::IoFailure, "writing " + partial.string() + " failed");
        }
    }
    // What was written must parse as any .athc does before it takes the name.
    {
        auto written = platform::MappedFile::open(partial);
        if (!written) return std::move(written).error();
        auto layout = parse(*written, out);
        if (!layout) {
            std::filesystem::remove(partial, ec);
            return std::move(layout).error();
        }
    }
    std::filesystem::rename(partial, out, ec);
    if (ec) {
        return Error(ErrorCode::IoFailure, "cannot move " + partial.string() + " to " + out.string() + ": " +
                                             ec.message());
    }
    return converted;
}

Result<LodCloud> readAthc(gpu::Device& device, const std::filesystem::path& path) {
    auto file = platform::MappedFile::open(path);
    if (!file) return std::move(file).error();
    auto layout = parse(*file, path);
    if (!layout) return std::move(layout).error();
    auto lod = cloudOf(device, *file, *layout, path);
    if (!lod) return std::move(lod).error();
    const FileHeader& h = layout->header;
    ATHENEA_TRY(makeStore(device, *lod, *layout, h.chunks, path));
    gpu::CommandBatch batch(device);
    uint64_t staged = 0;
    for (uint32_t c = 0; c < h.chunks; ++c) {
        place(batch, *lod, c, c, file->bytes().data() + layout->chunks[c].offset, layout->extra);
        staged += lod->chunkCount(c) * elementBytesOf(h, layout->extra);
        if (staged >= kStageBytes) {
            ATHENEA_TRY(batch.submit(true));
            staged = 0;
        }
    }
    ATHENEA_TRY(batch.submit(true));
    return std::move(*lod);
}

// --- a file as a chunk source ------------------------------------------------

struct AthcSource::Impl {
    std::filesystem::path path;
    platform::MappedFile  file;
    Layout                layout;
    ChunkFormat           format;
};

AthcSource::AthcSource() : impl_(std::make_unique<Impl>()) {}
AthcSource::~AthcSource() = default;

Result<std::unique_ptr<AthcSource>> AthcSource::open(const std::filesystem::path& path) {
    std::unique_ptr<AthcSource> source(new AthcSource());
    Impl& s = *source->impl_;
    s.path = path;
    auto file = platform::MappedFile::open(path);
    if (!file) return std::move(file).error();
    s.file = std::move(*file);
    auto layout = parse(s.file, path);
    if (!layout) return std::move(layout).error();
    s.layout = std::move(*layout);
    s.format = formatOf(s.layout);
    return source;
}

const ChunkFormat& AthcSource::format() const noexcept {
    return impl_->format;
}

uint32_t AthcSource::chunkCount() const noexcept {
    return impl_->layout.header.chunks;
}

scene::Bounds AthcSource::bounds(uint32_t /*chunk*/) const {
    return boundsOf(impl_->layout.header);
}

std::string AthcSource::name() const {
    return impl_->path.filename().string();
}

Result<LodCloud> AthcSource::openCloud(gpu::Device& device) {
    return cloudOf(device, impl_->file, impl_->layout, impl_->path);
}

ChunkRequest AthcSource::request(uint32_t chunk) const {
    const ChunkEntry& e = impl_->layout.chunks[chunk];
    return {chunk, 0, e.count, e.count * impl_->format.elementBytes()};
}

Result<void> AthcSource::fetch(const ChunkRequest& request, std::span<std::byte> staging) {
    const Impl& s = *impl_;
    if (request.chunk >= s.layout.chunks.size()) {
        return Error::make(ErrorCode::InvalidArgument, "{}: no chunk {}", name(), request.chunk);
    }
    const ChunkEntry& e = s.layout.chunks[request.chunk];
    const uint64_t bytes = e.count * s.format.elementBytes();
    if (staging.size() != bytes) {
        return Error::make(ErrorCode::InvalidArgument, "{}: chunk {} is {} bytes, staged into {}", name(),
                           request.chunk, bytes, staging.size());
    }
    std::memcpy(staging.data(), s.file.bytes().data() + e.offset, bytes);
    return ok();
}

// --- streaming ---------------------------------------------------------------

namespace {

/// A chunk a loader has copied into staging slot `staging`.
struct Loaded {
    ChunkRequest request;
    uint32_t     staging = 0;
};

/// An epoch a live source handed over whole.
struct Epoch {
    uint64_t                 id = 0;
    uint64_t                 ptpTime = 0;
    std::vector<ChunkChange> changes;
};

}   // namespace

struct StreamingPool::Impl {
    gpu::Device*                 device = nullptr;
    std::unique_ptr<ChunkSource> source;
    ChunkFormat                  format;
    ExtraHeader                  extra{};
    bool                         live = false;
    FrameThread                  frameThread;
    LodCloud                     cloud;
    std::vector<int32_t>         owner;          // per slot: the chunk it shows, or -1
    std::vector<uint32_t>        wantedNow;      // per chunk: how much this frame wants it (CutStats::needs)
    std::vector<uint64_t>        lastWanted;     // per chunk: the frame it was last wanted
    std::vector<uint8_t>         inFlight;       // per chunk: queued, loading, or loaded and not placed
    std::vector<scene::Bounds>   shownBounds;    // per chunk
    uint64_t                     frame = 1;
    uint64_t                     budgetBytes = 0;   // StreamingSettings::bytesPerFrame
    Status                       counters;

    // STAGING: one upload buffer of `stagingSlots` whole chunks, mapped once
    // by the frame thread at open. A loader writes only the slot it took
    // from `freeStaging`; the frame thread records copies out of it and
    // gives it back once the fence says the device has run them.
    gpu::Buffer                                 staging;
    std::byte*                                  mapped = nullptr;
    uint64_t                                    stride = 0;
    rhi::ComPtr<rhi::IFence>                    fence;
    uint64_t                                    fenceValue = 0;
    std::vector<std::pair<uint32_t, uint64_t>>  retiring;   // frame thread: staging slot, fence value
    bool                                        recorded = false;   // the batch holds copies or flags

    // Shared with the loaders, under `mutex`.
    std::mutex                mutex;
    std::condition_variable   work, done;
    std::deque<ChunkRequest>  queue;
    std::vector<uint32_t>     freeStaging;
    std::vector<Loaded>       loaded;
    uint32_t                  busy = 0;   // loads a loader has taken
    bool                      stop = false;
    std::optional<Error>      failure;
    std::vector<std::thread>  loaders;
    Status                    reported;   // `counters` as the last update left them, for status()

    // A live source's epochs (frame thread).
    std::deque<Epoch>                      incoming;   // complete, not started
    std::optional<Epoch>                   building;   // being loaded beside the one shown
    std::vector<int32_t>                   nextSlots;  // per chunk: its slot in the epoch being built
    std::vector<uint32_t>                  nextCounts;
    std::vector<scene::Bounds>             nextBounds;
    std::unordered_map<uint64_t, uint32_t> targets;    // the building epoch's pages not placed: ticket -> slot
    std::vector<uint8_t>                   reserved;   // per slot: a target of the building epoch

    /// LOADER THREADS. Takes a request only with a staging slot for it, so a
    /// loader that has one always finishes.
    void load() {
        std::unique_lock lock(mutex);
        for (;;) {
            work.wait(lock, [&] { return stop || (!queue.empty() && !freeStaging.empty()); });
            if (stop) {
                return;
            }
            const ChunkRequest request = queue.front();
            queue.pop_front();
            const uint32_t slot = freeStaging.back();
            freeStaging.pop_back();
            ++busy;
            lock.unlock();
            Result<void> fetched =
                request.bytes <= stride
                    ? source->fetch(request, std::span<std::byte>(mapped + slot * stride, request.bytes))
                    : Result<void>(Error::make(ErrorCode::InvalidArgument, "{}: chunk {} of {} bytes, past a chunk's {}",
                                               source->name(), request.chunk, request.bytes, stride));
            lock.lock();
            --busy;
            if (fetched) {
                loaded.push_back({request, slot});
            } else {
                freeStaging.push_back(slot);
                if (!failure) {
                    failure = std::move(fetched).error();
                }
            }
            done.notify_all();
        }
    }

    /// Frame thread: staging slots whose copies the device has run go back
    /// to the loaders. `all`: the caller has waited for the queue.
    void retire(bool all) {
        uint64_t completed = 0;
        if (!all && SLANG_FAILED(fence->getCurrentValue(&completed))) {
            return;
        }
        std::lock_guard lock(mutex);
        size_t kept = 0;
        for (const auto& r : retiring) {
            if (all || r.second <= completed) {
                freeStaging.push_back(r.first);
            } else {
                retiring[kept++] = r;
            }
        }
        const bool freed = kept != retiring.size();
        retiring.resize(kept);
        if (freed) {
            work.notify_all();
        }
    }

    /// Frame thread: a staging slot nothing was copied out of.
    void release(uint32_t slot) {
        std::lock_guard lock(mutex);
        freeStaging.push_back(slot);
        work.notify_all();
    }

    /// Frame thread: submits what was recorded; the staging read by it is
    /// given back when the fence passes.
    Result<void> submit(gpu::CommandBatch& batch, bool wait) {
        if (!recorded && !wait) {
            retire(false);   // nothing to run: no submit a frame for nothing
            return ok();
        }
        // slang-rhi's CUDA fence takes its value when the host submits, not
        // when the device gets there: it cannot say a copy has run, so on
        // CUDA a batch that read staging is waited for, as every update was
        // before the staging ring.
        const bool waited = wait || device->backend() == gpu::Backend::CUDA;
        ATHENEA_TRY(batch.submitSignalling(fence.get(), ++fenceValue, waited));
        recorded = false;
        retire(waited);
        return ok();
    }

    /// Frame thread: whatever a loader failed with, once.
    Result<void> takeFailure() {
        std::lock_guard lock(mutex);
        if (failure) {
            Error e = std::move(*failure);
            failure.reset();
            return e;
        }
        return ok();
    }

    /// Frame thread: copies `l` into `slot`.
    void placeLoaded(gpu::CommandBatch& batch, const Loaded& l, uint32_t slot) {
        placeStaged(batch, cloud, extra, slot, l.request.splats, staging.rhi(), uint64_t{l.staging} * stride);
        retiring.emplace_back(l.staging, fenceValue + 1);
        recorded = true;
    }

    /// Frame thread: what status() answers from now on.
    void report() {
        std::lock_guard lock(mutex);
        reported = counters;
    }

    Result<uint32_t> updateOnDemand(bool wait, uint64_t budget);
    Result<uint32_t> updateLive(bool wait, uint64_t budget);
    Result<void>     startEpoch();
    void             publish(gpu::CommandBatch& batch);
};

StreamingPool::StreamingPool() : impl_(std::make_unique<Impl>()) {}

StreamingPool::~StreamingPool() {
    {
        std::lock_guard lock(impl_->mutex);
        impl_->stop = true;
    }
    impl_->work.notify_all();
    for (std::thread& t : impl_->loaders) {
        t.join();
    }
    if (impl_->mapped != nullptr) {
        (void)impl_->device->rhi()->unmapBuffer(impl_->staging.rhi());
    }
}

Result<std::unique_ptr<StreamingPool>> StreamingPool::open(gpu::Device& device, const std::filesystem::path& path,
                                                           const StreamingSettings& settings) {
    auto source = AthcSource::open(path);
    if (!source) return std::move(source).error();
    return open(device, std::unique_ptr<ChunkSource>(std::move(*source)), settings);
}

Result<std::unique_ptr<StreamingPool>> StreamingPool::open(gpu::Device& device, std::unique_ptr<ChunkSource> source,
                                                           const StreamingSettings& settings) {
    if (source == nullptr) {
        return Error(ErrorCode::InvalidArgument, "a streaming pool needs a chunk source");
    }
    std::unique_ptr<StreamingPool> pool(new StreamingPool());
    Impl& p = *pool->impl_;
    p.frameThread.bind();
    p.device = &device;
    p.source = std::move(source);
    p.format = p.source->format();
    p.extra = extraOf(p.format);
    p.live = p.source->live();
    const uint32_t chunks = p.source->chunkCount();
    if (chunks == 0 || p.format.chunkSplats == 0 || p.format.shWords == 0) {
        return Error::make(ErrorCode::InvalidArgument, "{}: no chunks to stream", p.source->name());
    }
    auto cloud = p.source->openCloud(device);
    if (!cloud) return std::move(cloud).error();
    p.cloud = std::move(*cloud);
    // A live store holds the epoch shown and the one being built beside it.
    const uint64_t most = p.live ? uint64_t{2} * chunks : chunks;
    const uint32_t slots =
        static_cast<uint32_t>(std::clamp<uint64_t>(settings.budgetSplats / p.format.chunkSplats, 1, most));
    ATHENEA_TRY(makeStore(device, p.cloud, p.format, p.cloud.count, chunks, slots,
                          p.live ? p.cloud.splats.bounds : p.source->bounds(0), p.source->name()));
    p.cloud.streamed = true;
    if (p.live) {
        // Nothing shown yet: every chunk empty until an epoch says otherwise.
        p.cloud.chunkCounts.assign(chunks, 0);
        p.cloud.splats.count = static_cast<uint32_t>(uint64_t{slots} * p.format.chunkSplats);
    }
    p.owner.assign(slots, -1);
    p.reserved.assign(slots, 0);
    p.wantedNow.assign(chunks, 0);
    p.lastWanted.assign(chunks, 0);
    p.inFlight.assign(chunks, 0);
    p.shownBounds.assign(chunks, p.live ? scene::Bounds{} : p.source->bounds(0));
    p.counters.slots = slots;
    p.reported = p.counters;
    p.budgetBytes = settings.bytesPerFrame;

    // The staging ring: upload memory the loaders write and the device copies
    // from, mapped here once for the pool's life. Every backend slang-rhi
    // builds here hands back the same pointer for good (Metal's shared
    // contents, CUDA's pinned host allocation, Vulkan's host-coherent
    // mapping): see docs/decisions.md, "Live splat streams".
    const uint32_t loaders = std::max<uint32_t>(settings.loaders, 1);
    const uint32_t stagingSlots = settings.stagingChunks != 0 ? settings.stagingChunks : 2 * loaders + 4;
    p.stride = (p.format.chunkBytes() + 255) / 256 * 256;
    gpu::BufferDesc desc;
    desc.bytes = p.stride * stagingSlots;
    desc.memory = gpu::Memory::Upload;
    desc.label = "stream.staging";
    auto staging = gpu::Buffer::create(device, desc);
    if (!staging) return std::move(staging).error();
    p.staging = std::move(*staging);
    void* mapped = nullptr;
    if (SLANG_FAILED(device.rhi()->mapBuffer(p.staging.rhi(), rhi::CpuAccessMode::Write, &mapped)) ||
        mapped == nullptr) {
        return Error::make(ErrorCode::DeviceFailure, "{}: cannot map the staging buffer", p.source->name());
    }
    p.mapped = static_cast<std::byte*>(mapped);
    rhi::FenceDesc fence;
    fence.initialValue = 0;
    if (SLANG_FAILED(device.rhi()->createFence(fence, p.fence.writeRef()))) {
        return Error::make(ErrorCode::DeviceFailure, "{}: cannot make the staging fence", p.source->name());
    }
    for (uint32_t k = 0; k < stagingSlots; ++k) {
        p.freeStaging.push_back(stagingSlots - 1 - k);
    }
    for (uint32_t k = 0; k < loaders; ++k) {
        p.loaders.emplace_back([&p] { p.load(); });
    }
    log::info("{}: streaming {} chunks of {} splats into {} slots{}", p.source->name(), chunks, p.format.chunkSplats,
              slots, p.live ? ", live" : "");
    return pool;
}

const LodCloud& StreamingPool::cloud() const noexcept {
    return impl_->cloud;
}

scene::Bounds StreamingPool::chunkBounds(uint32_t chunk) const {
    return chunk < impl_->shownBounds.size() ? impl_->shownBounds[chunk] : scene::Bounds{};
}

void StreamingPool::want(std::span<const uint32_t> needs) {
    Impl& p = *impl_;
    if (!p.frameThread.current() || p.live) {
        return;
    }
    const size_t n = std::min(needs.size(), p.wantedNow.size());
    for (size_t c = 0; c < n; ++c) {
        p.wantedNow[c] = std::max(p.wantedNow[c], needs[c]);
    }
}

Result<uint32_t> StreamingPool::update(bool wait) {
    Impl& p = *impl_;
    ATHENEA_TRY(p.frameThread.require("StreamingPool::update"));
    p.retire(false);
    return p.live ? p.updateLive(wait, p.budgetBytes) : p.updateOnDemand(wait, p.budgetBytes);
}

namespace {

/// Whether a chunk of `bytes` fits what is left of a frame's budget. The
/// first chunk of a frame always does: one larger than the budget would
/// otherwise never come.
bool withinBudget(uint64_t budget, uint64_t placed, uint64_t bytes) {
    return budget == 0 || placed == 0 || placed + bytes <= budget;
}

}   // namespace

Result<uint32_t> StreamingPool::Impl::updateOnDemand(bool wait, uint64_t budget) {
    LodCloud& lod = cloud;
    const uint32_t chunks = lod.chunks();
    const uint32_t slots = static_cast<uint32_t>(owner.size());
    for (uint32_t c = 0; c < chunks; ++c) {
        if (wantedNow[c] != 0) {
            lastWanted[c] = frame;
        }
    }
    // What a slot's place is worth keeping: a free slot least, then chunks not
    // wanted now (least recently wanted first), then by how much they are.
    const auto worth = [&](uint32_t slot) -> std::pair<uint64_t, uint64_t> {
        const int32_t chunk = owner[slot];
        if (chunk < 0) {
            return {0, 0};
        }
        const auto c = static_cast<uint32_t>(chunk);
        return wantedNow[c] == 0 ? std::pair<uint64_t, uint64_t>{1, lastWanted[c]}
                                 : std::pair<uint64_t, uint64_t>{2, wantedNow[c]};
    };
    // A chunk wanted `want` may take a slot worth `w`.
    const auto mayTake = [](uint32_t want, std::pair<uint64_t, uint64_t> w) {
        return w.first < 2 ? (w.first == 0 || want > 0) : uint64_t{want} > 2 * w.second;
    };
    std::vector<uint32_t> bySlotWorth(slots);
    const auto sortSlots = [&] {
        for (uint32_t k = 0; k < slots; ++k) {
            bySlotWorth[k] = k;
        }
        std::sort(bySlotWorth.begin(), bySlotWorth.end(), [&](uint32_t a, uint32_t b) { return worth(a) < worth(b); });
    };

    {
        std::unique_lock lock(mutex);
        // Queued chunks no longer wanted are not loaded after all.
        for (auto it = queue.begin(); it != queue.end();) {
            if (wantedNow[it->chunk] == 0) {
                inFlight[it->chunk] = 0;
                it = queue.erase(it);
            } else {
                ++it;
            }
        }
        // Missing chunks, most wanted first, each against the next cheapest
        // slot not already promised to a load in flight.
        std::vector<uint32_t> missing;
        for (uint32_t c = 0; c < chunks; ++c) {
            if (wantedNow[c] != 0 && lod.slots[c] < 0 && inFlight[c] == 0) {
                missing.push_back(c);
            }
        }
        std::sort(missing.begin(), missing.end(), [&](uint32_t a, uint32_t b) {
            return wantedNow[a] != wantedNow[b] ? wantedNow[a] > wantedNow[b] : a < b;
        });
        sortSlots();
        size_t next = queue.size() + busy + loaded.size();
        for (const uint32_t c : missing) {
            if (next >= slots || !mayTake(wantedNow[c], worth(bySlotWorth[next]))) {
                break;
            }
            inFlight[c] = 1;
            queue.push_back(source->request(c));
            ++next;
        }
        std::stable_sort(queue.begin(), queue.end(), [&](const ChunkRequest& a, const ChunkRequest& b) {
            return wantedNow[a.chunk] > wantedNow[b.chunk];
        });
        work.notify_all();
    }

    // Placed most wanted first, each in the cheapest slot it may take, as long
    // as the frame's bytes last. `wait`: round after round until every queued
    // chunk has come -- the staging holds only so many at once.
    uint32_t placed = 0;
    uint64_t placedBytes = 0;
    uint32_t deferred = 0;
    gpu::CommandBatch batch(*device);
    for (;;) {
        std::vector<Loaded> arrived;
        {
            std::unique_lock lock(mutex);
            if (wait) {
                done.wait(lock, [&] { return busy == 0 && (queue.empty() || freeStaging.empty()); });
            }
            arrived.swap(loaded);
        }
        std::sort(arrived.begin(), arrived.end(), [&](const Loaded& a, const Loaded& b) {
            return wantedNow[a.request.chunk] > wantedNow[b.request.chunk];
        });
        std::vector<Loaded> held;
        for (const Loaded& l : arrived) {
            const uint32_t chunk = l.request.chunk;
            if (lod.slots[chunk] >= 0) {
                inFlight[chunk] = 0;
                release(l.staging);
                continue;
            }
            uint32_t slot = 0;
            for (uint32_t k = 1; k < slots; ++k) {
                slot = worth(k) < worth(slot) ? k : slot;
            }
            const auto w = worth(slot);
            if (!(w.first == 0 || mayTake(wantedNow[chunk], w))) {
                inFlight[chunk] = 0;
                release(l.staging);
                continue;   // not worth a place now: asked for again when it is
            }
            if (!withinBudget(budget, placedBytes, l.request.bytes)) {
                held.push_back(l);   // the next frame's, still in flight
                continue;
            }
            inFlight[chunk] = 0;
            const int32_t evicted = owner[slot];
            if (evicted >= 0) {
                lod.slots[static_cast<uint32_t>(evicted)] = -1;
                setResident(batch, lod, static_cast<uint32_t>(evicted), false);
                recorded = true;
                ++counters.evictions;
            }
            placeLoaded(batch, l, slot);
            setResident(batch, lod, chunk, true);
            lod.slots[chunk] = static_cast<int32_t>(slot);
            owner[slot] = static_cast<int32_t>(chunk);
            ++placed;
            placedBytes += l.request.bytes;
        }
        deferred = static_cast<uint32_t>(held.size());
        bool more = false;
        {
            std::lock_guard lock(mutex);
            loaded.insert(loaded.end(), held.begin(), held.end());
            more = !queue.empty() || busy > 0;
        }
        if (!wait || !held.empty() || !more) {
            break;
        }
        // Another round: what this one placed is run, its staging freed.
        ATHENEA_TRY(submit(batch, true));
        ATHENEA_TRY(takeFailure());
    }
    ATHENEA_TRY(submit(batch, wait));
    ATHENEA_TRY(takeFailure());
    counters.loads += placed;
    counters.placedBytes = placedBytes;
    counters.deferred = deferred;

    uint32_t resident = 0;
    uint32_t missing = 0;
    for (uint32_t c = 0; c < chunks; ++c) {
        resident += lod.slots[c] >= 0 ? 1 : 0;
        missing += (wantedNow[c] != 0 && lod.slots[c] < 0) ? 1 : 0;
    }
    counters.resident = resident;
    counters.missing = missing;
    std::fill(wantedNow.begin(), wantedNow.end(), 0);
    ++frame;
    report();
    return placed;
}

Result<void> StreamingPool::Impl::startEpoch() {
    // Taken off `incoming` once it is started, or dropped when it does not fit.
    const Epoch& e = incoming.front();
    const uint32_t chunks = cloud.chunks();
    nextSlots = cloud.slots;
    nextCounts = cloud.chunkCounts;
    nextBounds = shownBounds;
    std::vector<uint32_t> free;
    for (uint32_t k = static_cast<uint32_t>(owner.size()); k-- > 0;) {
        if (owner[k] < 0 && reserved[k] == 0) {
            free.push_back(k);
        }
    }
    size_t needed = 0;
    for (const ChunkChange& c : e.changes) {
        needed += c.op == ChunkChange::Op::Remove ? 0 : 1;
        if (c.request.chunk >= chunks || c.request.splats > format.chunkSplats ||
            (c.op != ChunkChange::Op::Remove &&
             (c.request.splats == 0 || c.request.bytes != c.request.splats * format.elementBytes()))) {
            return Error::make(ErrorCode::InvalidArgument, "{}: epoch {} has a page that is not a chunk of its stream",
                               source->name(), e.id);
        }
    }
    if (needed > free.size()) {
        // Dropped, not waited for: the epoch shown stays, and a later epoch
        // that fits is shown. Its pages' bytes are given back to the source.
        log::warn("{}: epoch {} dropped: it writes {} chunks, and the store has {} slots free beside the epoch "
                  "shown; a live store needs room for two epochs (budgetSplats)",
                  source->name(), e.id, needed, free.size());
        for (const ChunkChange& c : e.changes) {
            if (c.op != ChunkChange::Op::Remove) {
                source->discard(c.request);
            }
        }
        ++counters.droppedNoRoom;
        incoming.pop_front();
        return ok();
    }
    {
        std::lock_guard lock(mutex);
        for (const ChunkChange& c : e.changes) {
            const uint32_t chunk = c.request.chunk;
            if (c.op == ChunkChange::Op::Remove) {
                nextSlots[chunk] = -1;
                nextCounts[chunk] = 0;
                continue;
            }
            const uint32_t slot = free.back();
            free.pop_back();
            reserved[slot] = 1;
            nextSlots[chunk] = static_cast<int32_t>(slot);
            nextCounts[chunk] = c.request.splats;
            nextBounds[chunk] = c.bounds;
            targets[c.request.ticket] = slot;
            queue.push_back(c.request);
        }
        work.notify_all();
    }
    building = std::move(incoming.front());
    incoming.pop_front();
    return ok();
}

void StreamingPool::Impl::publish(gpu::CommandBatch& batch) {
    const uint32_t chunks = cloud.chunks();
    for (uint32_t c = 0; c < chunks; ++c) {
        if ((cloud.slots[c] >= 0) != (nextSlots[c] >= 0)) {
            setResident(batch, cloud, c, nextSlots[c] >= 0);
            recorded = true;
        }
    }
    std::fill(owner.begin(), owner.end(), -1);
    std::fill(reserved.begin(), reserved.end(), 0);
    for (uint32_t c = 0; c < chunks; ++c) {
        if (nextSlots[c] >= 0) {
            owner[static_cast<uint32_t>(nextSlots[c])] = static_cast<int32_t>(c);
        }
    }
    cloud.slots = nextSlots;
    cloud.chunkCounts = nextCounts;
    shownBounds = nextBounds;
    counters.epoch = building->id;
    counters.ptpTime = building->ptpTime;
    ++counters.epochs;
    building.reset();
}

Result<uint32_t> StreamingPool::Impl::updateLive(bool wait, uint64_t budget) {
    std::vector<ChunkChange> changes;
    source->poll(changes);
    for (ChunkChange& c : changes) {
        if (incoming.empty() || incoming.back().id != c.epoch) {
            incoming.push_back({c.epoch, c.ptpTime, {}});
        }
        incoming.back().changes.push_back(std::move(c));
    }
    uint32_t placed = 0;
    uint64_t placedBytes = 0;
    uint32_t deferred = 0;
    gpu::CommandBatch batch(*device);
    for (;;) {
        while (!building && !incoming.empty()) {
            ATHENEA_TRY(startEpoch());
        }
        if (!building) {
            break;
        }
        std::vector<Loaded> arrived;
        {
            std::unique_lock lock(mutex);
            if (wait) {
                done.wait(lock, [&] { return busy == 0 && (queue.empty() || freeStaging.empty()); });
            }
            arrived.swap(loaded);
        }
        std::vector<Loaded> held;
        for (const Loaded& l : arrived) {
            const auto target = targets.find(l.request.ticket);
            if (target == targets.end()) {
                release(l.staging);   // cannot happen: every load is a target of the epoch built
                continue;
            }
            if (!withinBudget(budget, placedBytes, l.request.bytes)) {
                held.push_back(l);
                continue;
            }
            placeLoaded(batch, l, target->second);
            targets.erase(target);
            ++placed;
            placedBytes += l.request.bytes;
        }
        deferred = static_cast<uint32_t>(held.size());
        bool more = false;
        {
            std::lock_guard lock(mutex);
            loaded.insert(loaded.end(), held.begin(), held.end());
            more = !queue.empty() || busy > 0;
        }
        ATHENEA_TRY(takeFailure());
        if (targets.empty()) {
            // Every page of the epoch is in its slot (the copies come before
            // anything this frame records after them): show it.
            publish(batch);
            continue;
        }
        if (!wait || !held.empty() || !more) {
            break;
        }
        ATHENEA_TRY(submit(batch, true));
    }
    ATHENEA_TRY(submit(batch, wait));
    ATHENEA_TRY(takeFailure());
    counters.loads += placed;
    counters.placedBytes = placedBytes;
    counters.deferred = deferred;
    uint32_t resident = 0;
    for (const int32_t s : cloud.slots) {
        resident += s >= 0 ? 1 : 0;
    }
    counters.resident = resident;
    counters.missing = 0;
    counters.pendingEpochs = static_cast<uint32_t>(incoming.size() + (building ? 1 : 0));
    ++frame;
    report();
    return placed;
}

StreamingPool::Status StreamingPool::status() const {
    Impl& p = *impl_;
    std::lock_guard lock(p.mutex);
    Status s = p.reported;
    s.inFlight = static_cast<uint32_t>(p.queue.size() + p.busy + p.loaded.size());
    return s;
}

}   // namespace athenea::lod
