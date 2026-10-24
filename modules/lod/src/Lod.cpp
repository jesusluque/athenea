// Copyright (c) 2026 jesus luque.
#include "athenea/lod/Lod.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "athenea/core/Log.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::lod {
namespace {

constexpr uint32_t kLevels = 10;
constexpr uint32_t kMomentsHead = 13;

Result<gpu::Buffer> buffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

Result<scene::GpuSplats> packed(gpu::Device& device, uint32_t count, uint32_t restPerColour, uint32_t shWords,
                                const char* label, bool withNormals = false, bool withEmission = false) {
    scene::GpuSplats out;
    out.source = label;
    out.count = count;
    out.declared = count;
    out.restPerColour = restPerColour;
    out.shWords = shWords;
    auto p = buffer(device, count, 16, label);
    if (!p) return std::move(p).error();
    auto s = buffer(device, uint64_t{count} * 4, 4, label);
    if (!s) return std::move(s).error();
    auto h = buffer(device, uint64_t{count} * shWords, 4, label);
    if (!h) return std::move(h).error();
    out.positions = std::move(*p);
    out.shape = std::move(*s);
    out.sh = std::move(*h);
    if (withNormals) {
        auto n = buffer(device, count, 4, label);
        if (!n) return std::move(n).error();
        out.normals = std::move(*n);
    }
    if (withEmission) {
        auto e = buffer(device, count, 4, label);
        if (!e) return std::move(e).error();
        out.emission = std::move(*e);
    }
    return out;
}

/// WHAT A CLOUD CARRIES BESIDE ITS GAUSSIANS (lod_extras.slang): a buffer, its
/// words a gaussian, and how a level's group merges it -- 0 the first
/// gaussian's, 1 f16 pairs averaged by opacity, 2 bits set where half the
/// weight has them. The material and the matte's ids take one gaussian's; a
/// transfer of harmonics is averaged (it is linear in the sky) and a zonal
/// one, whose axes do not average, takes one gaussian's; the open directions
/// go by weight.
struct Extra {
    gpu::Buffer scene::GpuSplats::* member;
    uint32_t                        words;
    uint32_t                        mode;
};

std::vector<Extra> extrasOf(const scene::GpuSplats& cloud) {
    std::vector<Extra> out;
    if (cloud.hasPbr()) out.push_back({&scene::GpuSplats::pbr, 1, 0});
    if (cloud.hasLobes()) out.push_back({&scene::GpuSplats::lobes, 3, 0});
    if (cloud.hasCrypto()) out.push_back({&scene::GpuSplats::crypto, 1, 0});
    if (cloud.transfer.valid() && cloud.transferWords > 0) {
        out.push_back({&scene::GpuSplats::transfer, cloud.transferWords, cloud.isZonal() ? 0u : 1u});
    }
    if (cloud.hasShadowBits() && cloud.shadowWords > 0) {
        out.push_back({&scene::GpuSplats::shadowBits, cloud.shadowWords, 2});
    }
    return out;
}

/// `to` given room for `count` gaussians of every extra `like` carries, and
/// the counts that say what its transfer and bits are.
Result<void> allocateExtras(gpu::Device& device, const scene::GpuSplats& like, uint32_t count,
                            scene::GpuSplats& to) {
    for (const Extra& e : extrasOf(like)) {
        auto made = buffer(device, uint64_t{std::max(count, 1u)} * e.words, 4, "lod.extra");
        if (!made) return std::move(made).error();
        to.*e.member = std::move(*made);
    }
    to.transferCount = like.transferCount;
    to.transferWords = like.transferWords;
    to.shadowWords = like.shadowWords;
    return ok();
}

/// Whether `to` carries what `from` does, the same way.
bool sameExtras(const scene::GpuSplats& from, const scene::GpuSplats& to) {
    return from.hasPbr() == to.hasPbr() && from.hasLobes() == to.hasLobes() && from.hasCrypto() == to.hasCrypto() &&
           from.transferWords == to.transferWords && from.transferCount == to.transferCount &&
           from.shadowWords == to.shadowWords && from.transfer.valid() == to.transfer.valid() &&
           from.hasShadowBits() == to.hasShadowBits();
}

/// The shading normals' bindings: the cloud's own where it keeps them, and a
/// buffer of the right kind in their place where it does not, since every
/// name a shader declares must be bound. `params.normals` says which.
void bindNormals(rhi::ShaderCursor cursor, const scene::GpuSplats& from, const scene::GpuSplats& to) {
    const bool carried = from.hasNormals() && to.hasNormals();
    cursor["srcNormals"].setBinding(carried ? from.normals.rhi() : from.shape.rhi());
    cursor["normals"].setBinding(carried ? to.normals.rhi() : to.shape.rhi());
    cursor["params"]["normals"].setData(uint32_t{carried ? 1u : 0u});
}

/// The same for the radiance a gaussian gives off: `params.emission` says
/// whether it is carried.
void bindEmission(rhi::ShaderCursor cursor, const scene::GpuSplats& from, const scene::GpuSplats& to) {
    const bool carried = from.hasEmission() && to.hasEmission();
    cursor["srcEmission"].setBinding(carried ? from.emission.rhi() : from.shape.rhi());
    cursor["emission"].setBinding(carried ? to.emission.rhi() : to.shape.rhi());
    cursor["params"]["emission"].setData(uint32_t{carried ? 1u : 0u});
}

void setBounds(rhi::ShaderCursor p, const LodCloud& cloud) {
    p["boundsLoX"].setData(cloud.boundsLo[0]);
    p["boundsLoY"].setData(cloud.boundsLo[1]);
    p["boundsLoZ"].setData(cloud.boundsLo[2]);
    p["extent"].setData(cloud.extent);
}

}   // namespace

Result<LodBuilder> LodBuilder::create(gpu::ShaderLibrary& library) {
    LodBuilder b;
    b.device_ = &library.device();
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    b.sort_ = std::move(*sort);
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    b.prefix_ = std::move(*prefix);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    ATHENEA_TRY(make(b.morton_, "athenea/lod/lod_morton", "lodMorton"));
    ATHENEA_TRY(make(b.reorder_, "athenea/lod/lod_reorder", "lodReorder"));
    ATHENEA_TRY(make(b.boundaries_, "athenea/lod/lod_boundaries", "lodBoundaries"));
    ATHENEA_TRY(make(b.groups_, "athenea/lod/lod_groups", "lodGroups"));
    ATHENEA_TRY(make(b.leafMoments_, "athenea/lod/lod_leaf_moments", "lodLeafMoments"));
    ATHENEA_TRY(make(b.mergeMoments_, "athenea/lod/lod_merge_moments", "lodMergeMoments"));
    ATHENEA_TRY(make(b.finalize_, "athenea/lod/lod_finalize", "lodFinalize"));
    ATHENEA_TRY(make(b.extrasReorder_, "athenea/lod/lod_extras", "lodExtrasReorder"));
    ATHENEA_TRY(make(b.extrasMerge_, "athenea/lod/lod_extras", "lodExtrasMerge"));
    return b;
}

Result<LodCloud> LodBuilder::build(const scene::GpuSplats& cloud, const LodBuildSettings& settings) {
    gpu::Device& device = *device_;
    const uint32_t n = cloud.count;
    if (n == 0) {
        return Error(ErrorCode::InvalidArgument, "an empty cloud has no levels of detail");
    }
    LodCloud lod;
    for (int k = 0; k < 3; ++k) {
        lod.boundsLo[k] = cloud.bounds.min[static_cast<size_t>(k)];
        lod.extent = std::max(lod.extent, cloud.bounds.max[static_cast<size_t>(k)] - cloud.bounds.min[static_cast<size_t>(k)]);
    }
    lod.extent = std::max(lod.extent, 1e-6F);
    // Slightly past the bounds, so the far corner quantises inside the last cell.
    lod.extent *= 1.0001F;

    // Morton order.
    gpu::SortBuffers sorting;
    const auto assign = [&](gpu::Buffer& into, const char* label) -> Result<void> {
        auto made = buffer(device, n, 4, label);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(assign(sorting.keysLo, "lod.keys"));
    ATHENEA_TRY(assign(sorting.values, "lod.order"));
    ATHENEA_TRY(assign(sorting.scratchKeysLo, "lod.keys2"));
    ATHENEA_TRY(assign(sorting.scratchValues, "lod.order2"));
    const bool normals = cloud.hasNormals();
    const bool emission = cloud.hasEmission();
    auto splats = packed(device, n, cloud.restPerColour, cloud.shWords, "lod.splats", normals, emission);
    if (!splats) return std::move(splats).error();
    splats->linear = cloud.linear;   // reordered, in the space they were in
    splats->antialiased = cloud.antialiased;
    {
        gpu::CommandBatch batch(device);
        morton_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud.positions.rhi());
            cursor["keys"].setBinding(sorting.keysLo.rhi());
            cursor["values"].setBinding(sorting.values.rhi());
            cursor["params"]["count"].setData(n);
            setBounds(cursor["params"], lod);
        });
        ATHENEA_TRY(sort_.sort(batch, sorting, n, 30));
        reorder_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["order"].setBinding(sorting.values.rhi());
            cursor["srcPositions"].setBinding(cloud.positions.rhi());
            cursor["srcShape"].setBinding(cloud.shape.rhi());
            cursor["srcSh"].setBinding(cloud.sh.rhi());
            cursor["positions"].setBinding(splats->positions.rhi());
            cursor["shape"].setBinding(splats->shape.rhi());
            cursor["sh"].setBinding(splats->sh.rhi());
            bindNormals(cursor, cloud, *splats);
            bindEmission(cursor, cloud, *splats);
            cursor["params"]["count"].setData(n);
            cursor["params"]["shWords"].setData(cloud.shWords);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    // WHAT IT CARRIES BESIDE, into the same order.
    ATHENEA_TRY(allocateExtras(device, cloud, n, *splats));
    {
        gpu::CommandBatch batch(device);
        for (const Extra& e : extrasOf(cloud)) {
            extrasReorder_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                for (const char* unused : {"starts", "selected", "dest"}) {
                    cursor[unused].setBinding(sorting.values.rhi());
                }
                cursor["positions"].setBinding(cloud.positions.rhi());
                cursor["order"].setBinding(sorting.values.rhi());
                cursor["source"].setBinding((cloud.*e.member).rhi());
                cursor["target"].setBinding(((*splats).*e.member).rhi());
                cursor["params"]["count"].setData(n);
                cursor["params"]["words"].setData(e.words);
            });
        }
        ATHENEA_TRY(batch.submit(true));
    }
    splats->source = cloud.source;
    splats->bounds = cloud.bounds;
    lod.count = n;
    lod.splats = std::move(*splats);
    lod.order = sorting.values;
    const gpu::Buffer keys = sorting.keysLo;

    // Every level's groups, and how many there are: one small read each.
    struct Level {
        uint32_t    groups = 0;
        gpu::Buffer group, starts, cells;
    };
    std::array<Level, kLevels + 1> levels;
    auto boundary = buffer(device, n, 4, "lod.boundary");
    if (!boundary) return std::move(boundary).error();
    auto before = buffer(device, n, 4, "lod.before");
    if (!before) return std::move(before).error();
    auto total = buffer(device, 1, 4, "lod.total");
    if (!total) return std::move(total).error();
    const auto groupsOf = [&](uint32_t r) -> Result<void> {
        Level& level = levels[r];
        gpu::CommandBatch batch(device);
        boundaries_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["keys"].setBinding(keys.rhi());
            cursor["boundary"].setBinding(boundary->rhi());
            cursor["params"]["count"].setData(n);
            cursor["params"]["level"].setData(r);
        });
        ATHENEA_TRY(prefix_.apply(batch, *boundary, *before, *total, n));
        ATHENEA_TRY(batch.submit(true));
        ATHENEA_TRY(total->read(device, 0, sizeof(level.groups), &level.groups));
        auto group = buffer(device, n, 4, "lod.group");
        if (!group) return std::move(group).error();
        auto starts = buffer(device, level.groups, 4, "lod.starts");
        if (!starts) return std::move(starts).error();
        auto cells = buffer(device, level.groups, 4, "lod.cells");
        if (!cells) return std::move(cells).error();
        level.group = std::move(*group);
        level.starts = std::move(*starts);
        level.cells = std::move(*cells);
        gpu::CommandBatch write(device);
        groups_.dispatch(write, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["keys"].setBinding(keys.rhi());
            cursor["boundary"].setBinding(boundary->rhi());
            cursor["before"].setBinding(before->rhi());
            cursor["group"].setBinding(level.group.rhi());
            cursor["starts"].setBinding(level.starts.rhi());
            cursor["cells"].setBinding(level.cells.rhi());
            cursor["params"]["count"].setData(n);
            cursor["params"]["level"].setData(r);
        });
        return write.submit(true);
    };

    // The finest level worth storing: cells holding at least 1 / fraction
    // splats on average. Found coarse to fine; group counts only grow.
    const uint32_t coarsest = std::clamp<uint32_t>(settings.coarsestLevel, 1, kLevels);
    uint32_t finest = coarsest;
    for (uint32_t r = coarsest; r <= kLevels; ++r) {
        ATHENEA_TRY(groupsOf(r));
        if (static_cast<float>(levels[r].groups) > settings.maxGroupFraction * static_cast<float>(n) && r > coarsest) {
            levels[r] = Level{};
            break;
        }
        finest = r;
    }

    // Moments from the finest level up; a Gaussian per group at each.
    const uint32_t keep = cloud.restPerColour;
    // The shading normals' three moments, then the emission's three, each
    // where the cloud keeps it (lod_common.slang's layout).
    const uint32_t stride = kMomentsHead + keep * 3 + (normals ? 3u : 0u) + (emission ? 3u : 0u);
    gpu::Buffer fineMoments;
    std::vector<LodLevel> stored;
    for (uint32_t r = finest + 1; r-- > coarsest;) {
        Level& level = levels[r];
        auto moments = buffer(device, uint64_t{level.groups} * stride, 4, "lod.moments");
        if (!moments) return std::move(moments).error();
        auto gaussians = packed(device, level.groups, keep, cloud.shWords, "lod.merged", normals, emission);
        if (!gaussians) return std::move(gaussians).error();
        // Merged in the space the cloud keeps its colours in (lod_leaf_moments).
        gaussians->linear = cloud.linear;
        gaussians->antialiased = cloud.antialiased;
        gpu::CommandBatch batch(device);
        if (r == finest) {
            leafMoments_.dispatch(batch, {level.groups, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(lod.splats.positions.rhi());
                cursor["shape"].setBinding(lod.splats.shape.rhi());
                cursor["sh"].setBinding(lod.splats.sh.rhi());
                cursor["starts"].setBinding(level.starts.rhi());
                cursor["normals"].setBinding(normals ? lod.splats.normals.rhi() : lod.splats.shape.rhi());
                cursor["emission"].setBinding(emission ? lod.splats.emission.rhi() : lod.splats.shape.rhi());
                cursor["moments"].setBinding(moments->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["normals"].setData(uint32_t{normals ? 1u : 0u});
                p["emission"].setData(uint32_t{emission ? 1u : 0u});
                p["count"].setData(n);
                p["groups"].setData(level.groups);
                p["keep"].setData(keep);
                p["shWords"].setData(cloud.shWords);
                p["stride"].setData(stride);
            });
        } else {
            const Level& fine = levels[r + 1];
            mergeMoments_.dispatch(batch, {level.groups, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["starts"].setBinding(level.starts.rhi());
                cursor["fineGroup"].setBinding(fine.group.rhi());
                cursor["fineMoments"].setBinding(fineMoments.rhi());
                cursor["moments"].setBinding(moments->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(n);
                p["groups"].setData(level.groups);
                p["fineGroups"].setData(fine.groups);
                p["stride"].setData(stride);
            });
        }
        finalize_.dispatch(batch, {level.groups, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["moments"].setBinding(moments->rhi());
            cursor["positions"].setBinding(gaussians->positions.rhi());
            cursor["shape"].setBinding(gaussians->shape.rhi());
            cursor["sh"].setBinding(gaussians->sh.rhi());
            cursor["normals"].setBinding(normals ? gaussians->normals.rhi() : gaussians->shape.rhi());
            cursor["emission"].setBinding(emission ? gaussians->emission.rhi() : gaussians->shape.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["normals"].setData(uint32_t{normals ? 1u : 0u});
            p["emission"].setData(uint32_t{emission ? 1u : 0u});
            p["groups"].setData(level.groups);
            p["keep"].setData(keep);
            p["shWords"].setData(cloud.shWords);
            p["stride"].setData(stride);
        });
        // And what the cloud carries beside, merged over each group's run.
        ATHENEA_TRY(allocateExtras(device, lod.splats, level.groups, *gaussians));
        for (const Extra& e : extrasOf(lod.splats)) {
            extrasMerge_.dispatch(batch, {level.groups, 1, 1}, [&](rhi::ShaderCursor cursor) {
                for (const char* unused : {"order", "selected", "dest"}) {
                    cursor[unused].setBinding(level.starts.rhi());
                }
                cursor["starts"].setBinding(level.starts.rhi());
                cursor["positions"].setBinding(lod.splats.positions.rhi());
                cursor["source"].setBinding((lod.splats.*e.member).rhi());
                cursor["target"].setBinding(((*gaussians).*e.member).rhi());
                cursor["params"]["count"].setData(level.groups);
                cursor["params"]["words"].setData(e.words);
                cursor["params"]["mode"].setData(e.mode);
                cursor["params"]["splats"].setData(n);
            });
        }
        ATHENEA_TRY(batch.submit(true));
        gaussians->bounds = cloud.bounds;
        LodLevel out;
        out.level = r;
        out.gaussians = std::move(*gaussians);
        out.cells = level.cells;
        stored.push_back(std::move(out));
        if (r == finest) {
            lod.groups = level.group;
            lod.starts = level.starts;
        }
        fineMoments = std::move(*moments);
        if (r + 1 <= kLevels) {
            levels[r + 1] = Level{};   // its groups were only needed to build this one
        }
    }
    std::reverse(stored.begin(), stored.end());
    lod.levels = std::move(stored);

    // In memory, chunk c is slot c and every chunk is there.
    lod.chunkSplats = std::max<uint32_t>(settings.chunkSplats, 1);
    const uint32_t chunks = (n + lod.chunkSplats - 1) / lod.chunkSplats;
    lod.slots.resize(chunks);
    for (uint32_t c = 0; c < chunks; ++c) {
        lod.slots[c] = static_cast<int32_t>(c);
    }
    auto resident = gpu::Buffer::fromSpan(device, std::span<const uint32_t>(std::vector<uint32_t>(chunks, 1)),
                                          "lod.resident");
    if (!resident) return std::move(resident).error();
    lod.resident = std::move(*resident);
    log::info("{}: levels of detail {}..{}, {} merged Gaussians over {} splats", cloud.source, coarsest, finest,
              lod.mergedGaussians(), n);
    return lod;
}

struct CutSelector::Frame {
    scene::GpuSplats cloud;
    uint32_t         capacity = 0;
    /// Per part -- the levels, then the runs of the store the splats are
    /// drawn from -- which elements are drawn, where they go, how many.
    std::vector<gpu::Buffer> selected, dest, totals;
    gpu::Buffer      state;       ///< the finest level's per-group state
    gpu::Buffer      needs;       ///< per chunk
    gpu::Buffer      readback;    ///< every part's count, then the needs: one read
};

CutSelector::CutSelector() = default;
CutSelector::CutSelector(CutSelector&&) noexcept = default;
CutSelector& CutSelector::operator=(CutSelector&&) noexcept = default;
CutSelector::~CutSelector() = default;

Result<CutSelector> CutSelector::create(gpu::ShaderLibrary& library) {
    CutSelector c;
    c.device_ = &library.device();
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    c.prefix_ = std::move(*prefix);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    ATHENEA_TRY(make(c.cutGroups_, "athenea/lod/lod_cut", "lodCutGroups"));
    ATHENEA_TRY(make(c.cutFinest_, "athenea/lod/lod_cut", "lodCutFinest"));
    ATHENEA_TRY(make(c.cutSplats_, "athenea/lod/lod_cut", "lodCutSplats"));
    ATHENEA_TRY(make(c.chunkNeeds_, "athenea/lod/lod_cut", "lodChunkNeeds"));
    ATHENEA_TRY(make(c.gather_, "athenea/lod/lod_gather", "lodGather"));
    ATHENEA_TRY(make(c.extrasGather_, "athenea/lod/lod_extras", "lodExtrasGather"));
    return c;
}

namespace {

/// A run of consecutive chunks in consecutive slots: one dispatch draws it.
struct Run {
    uint32_t offset = 0;   ///< first splat in the store
    uint32_t count = 0;
};

std::vector<Run> runsOf(const LodCloud& lod) {
    std::vector<Run> runs;
    int32_t previous = -2;
    for (uint32_t c = 0; c < lod.chunks(); ++c) {
        const int32_t slot = lod.slots[c];
        if (slot < 0) {
            previous = -2;
            continue;
        }
        // A chunk holding fewer than chunkSplats (the last, or a live page)
        // leaves a gap before the next slot: the run ends with it.
        if (slot == previous + 1 && !runs.empty()) {
            runs.back().count += lod.chunkCount(c);
        } else {
            runs.push_back({static_cast<uint32_t>(slot) * lod.chunkSplats, lod.chunkCount(c)});
        }
        previous = lod.chunkCount(c) == lod.chunkSplats ? slot : -2;
    }
    return runs;
}

}   // namespace

Result<void> CutSelector::ensureCloud(const LodCloud& lod, Frame& frame, uint32_t drawn) {
    gpu::Device& device = *device_;
    if (frame.capacity < drawn || frame.cloud.restPerColour != lod.splats.restPerColour ||
        frame.cloud.shWords != lod.splats.shWords || !frame.cloud.positions.valid() ||
        frame.cloud.hasNormals() != lod.splats.hasNormals() ||
        frame.cloud.hasEmission() != lod.splats.hasEmission() || !sameExtras(lod.splats, frame.cloud)) {
        const uint32_t capacity = std::max(drawn, frame.capacity + frame.capacity / 2);
        auto made = packed(device, std::max(capacity, 1u), lod.splats.restPerColour, lod.splats.shWords,
                           "cut.frame", lod.splats.hasNormals(), lod.splats.hasEmission());
        if (!made) return std::move(made).error();
        ATHENEA_TRY(allocateExtras(device, lod.splats, std::max(capacity, 1u), *made));
        frame.cloud = std::move(*made);
        frame.capacity = std::max(capacity, 1u);
    }
    frame.cloud.source = lod.splats.source;
    frame.cloud.linear = lod.splats.linear;
    frame.cloud.antialiased = lod.splats.antialiased;
    frame.cloud.count = drawn;
    frame.cloud.declared = lod.count;
    frame.cloud.bounds = lod.splats.bounds;
    return ok();
}

Result<uint32_t> CutSelector::copyChunks(const LodCloud& lod, Frame& frame) {
    const std::vector<Run> runs = runsOf(lod);
    uint32_t drawn = 0;
    for (const Run& run : runs) {
        drawn += run.count;
    }
    if (drawn == 0) {
        return drawn;
    }
    ATHENEA_TRY(ensureCloud(lod, frame, drawn));
    gpu::CommandBatch batch(*device_);
    rhi::ICommandEncoder* e = batch.encoder();
    const auto copy = [&](const gpu::Buffer& from, const gpu::Buffer& to, uint64_t perSplat) {
        uint64_t at = 0;
        for (const Run& run : runs) {
            e->copyBuffer(to.rhi(), at * perSplat, from.rhi(), uint64_t{run.offset} * perSplat, run.count * perSplat);
            at += run.count;
        }
    };
    copy(lod.splats.positions, frame.cloud.positions, 16);
    copy(lod.splats.shape, frame.cloud.shape, 16);
    copy(lod.splats.sh, frame.cloud.sh, 4 * uint64_t{lod.splats.shWords});
    if (lod.splats.hasNormals()) {
        copy(lod.splats.normals, frame.cloud.normals, 4);
    }
    if (lod.splats.hasEmission()) {
        copy(lod.splats.emission, frame.cloud.emission, 4);
    }
    for (const Extra& x : extrasOf(lod.splats)) {
        copy(lod.splats.*x.member, frame.cloud.*x.member, 4 * uint64_t{x.words});
    }
    batch.markDirty();
    ATHENEA_TRY(batch.submit());
    return drawn;
}

Result<std::vector<render::SplatInstance>> CutSelector::select(const render::Projection& projection,
                                                               std::span<const LodInstance> instances,
                                                               float threshold, std::vector<CutStats>* stats) {
    gpu::Device& device = *device_;
    std::vector<render::SplatInstance> out;
    if (stats != nullptr) {
        stats->clear();
    }
    while (frames_.size() < instances.size()) {
        frames_.push_back(std::make_unique<Frame>());
    }
    for (size_t k = 0; k < instances.size(); ++k) {
        const LodInstance& instance = instances[k];
        if (instance.cloud == nullptr || instance.cloud->count == 0) {
            if (stats != nullptr) {
                stats->emplace_back();
            }
            continue;
        }
        const float cutThreshold = instance.threshold.value_or(threshold);
        const LodCloud& lod = *instance.cloud;
        Frame& frame = *frames_[k];
        if (lod.levels.empty()) {
            if (!lod.streamed) {
                return Error(ErrorCode::InvalidArgument, lod.splats.source + ": a cut needs a merged level");
            }
            // A live stream's: nothing to cut to, its chunks on the device
            // copied as they are into the frame's cloud.
            auto drawn = copyChunks(lod, frame);
            if (!drawn) return std::move(drawn).error();
            if (stats != nullptr) {
                CutStats s;
                s.splats = *drawn;
                s.available = *drawn;
                stats->push_back(std::move(s));
            }
            if (*drawn > 0) {
                out.push_back({&frame.cloud, instance.objectToWorld, instance.edit});
            }
            continue;
        }
        const std::vector<Run> runs = runsOf(lod);
        const size_t levels = lod.levels.size();
        const size_t parts = levels + runs.size();
        const uint32_t chunks = lod.chunks();
        const uint32_t finestGroups = lod.levels.back().gaussians.count;
        frame.selected.resize(parts);
        frame.dest.resize(parts);
        frame.totals.resize(parts);
        const auto countOf = [&](size_t part) {
            return part < levels ? lod.levels[part].gaussians.count : runs[part - levels].count;
        };
        const auto ensure = [&](gpu::Buffer& into, uint64_t count, const char* label) -> Result<void> {
            if (into.count() < std::max<uint64_t>(count, 1)) {
                auto made = buffer(device, count, 4, label);
                if (!made) return std::move(made).error();
                into = std::move(*made);
            }
            return ok();
        };
        for (size_t part = 0; part < parts; ++part) {
            ATHENEA_TRY(ensure(frame.selected[part], countOf(part), "cut.selected"));
            ATHENEA_TRY(ensure(frame.dest[part], countOf(part), "cut.dest"));
            ATHENEA_TRY(ensure(frame.totals[part], 1, "cut.total"));
        }
        ATHENEA_TRY(ensure(frame.state, finestGroups, "cut.state"));
        ATHENEA_TRY(ensure(frame.needs, chunks, "cut.needs"));
        const size_t wanted = parts + (lod.streamed ? chunks : 0);
        ATHENEA_TRY(ensure(frame.readback, wanted, "cut.readback"));

        // The eye in the cloud's space. Perspective size is edge over distance,
        // which a uniform scale leaves alone; orthographic size is not.
        const render::Mat4 toCloud = aofx::xform::inverseAffine(instance.objectToWorld);
        const render::Vec3 eye = toCloud.point(projection.eyeWorld);
        const double scale = std::max({aofx::xform::length(instance.objectToWorld.column(0)),
                                       aofx::xform::length(instance.objectToWorld.column(1)),
                                       aofx::xform::length(instance.objectToWorld.column(2))});
        const float pixelsPerUnit = static_cast<float>(projection.orthographic ? projection.focalX * scale
                                                                               : projection.focalX);
        const auto setCut = [&](rhi::ShaderCursor p, uint32_t count, uint32_t level, bool coarsest) {
            p["count"].setData(count);
            p["level"].setData(level);
            setBounds(p, lod);
            p["eyeX"].setData(static_cast<float>(eye.x));
            p["eyeY"].setData(static_cast<float>(eye.y));
            p["eyeZ"].setData(static_cast<float>(eye.z));
            p["pixelsPerUnit"].setData(pixelsPerUnit);
            p["threshold"].setData(cutThreshold);
            p["orthographic"].setData(uint32_t{projection.orthographic ? 1u : 0u});
            p["coarsest"].setData(uint32_t{coarsest ? 1u : 0u});
            p["chunkSplats"].setData(lod.chunkSplats);
            p["chunks"].setData(chunks);
            p["splats"].setData(lod.count);
        };
        {
            gpu::CommandBatch batch(device);
            for (size_t part = 0; part < levels; ++part) {
                const LodLevel& level = lod.levels[part];
                const uint32_t count = level.gaussians.count;
                const bool finest = part + 1 == levels;
                (finest ? cutFinest_ : cutGroups_).dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["cells"].setBinding(level.cells.rhi());
                    cursor["selected"].setBinding(frame.selected[part].rhi());
                    if (finest) {
                        cursor["starts"].setBinding(lod.starts.rhi());
                        cursor["residentChunks"].setBinding(lod.resident.rhi());
                        cursor["state"].setBinding(frame.state.rhi());
                    }
                    setCut(cursor["params"], count, level.level, part == 0);
                });
            }
            for (size_t part = levels; part < parts; ++part) {
                const Run& run = runs[part - levels];
                cutSplats_.dispatch(batch, {run.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["groups"].setBinding(lod.groups.rhi());
                    cursor["state"].setBinding(frame.state.rhi());
                    cursor["selected"].setBinding(frame.selected[part].rhi());
                    cursor["params"]["count"].setData(run.count);
                    cursor["params"]["offset"].setData(run.offset);
                });
            }
            if (lod.streamed) {
                chunkNeeds_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["cells"].setBinding(lod.levels.back().cells.rhi());
                    cursor["starts"].setBinding(lod.starts.rhi());
                    cursor["state"].setBinding(frame.state.rhi());
                    cursor["selected"].setBinding(frame.needs.rhi());
                    setCut(cursor["params"], finestGroups, lod.levels.back().level, false);
                });
            }
            for (size_t part = 0; part < parts; ++part) {
                ATHENEA_TRY(prefix_.apply(batch, frame.selected[part], frame.dest[part], frame.totals[part],
                                      countOf(part)));
            }
            // Every count into one buffer, so they come back in one read and
            // not one read a part (that was most of the cut's time).
            for (size_t part = 0; part < parts; ++part) {
                batch.encoder()->copyBuffer(frame.readback.rhi(), part * sizeof(uint32_t), frame.totals[part].rhi(),
                                            0, sizeof(uint32_t));
            }
            if (lod.streamed) {
                batch.encoder()->copyBuffer(frame.readback.rhi(), parts * sizeof(uint32_t), frame.needs.rhi(), 0,
                                            chunks * sizeof(uint32_t));
            }
            batch.markDirty();
            ATHENEA_TRY(batch.submit(true));
        }
        std::vector<uint32_t> read(wanted, 0);
        ATHENEA_TRY(frame.readback.read(device, 0, wanted * sizeof(uint32_t), read.data()));
        uint32_t drawn = 0;
        uint32_t splatsDrawn = 0;
        for (size_t part = 0; part < parts; ++part) {
            drawn += read[part];
            splatsDrawn += part < levels ? 0 : read[part];
        }
        ATHENEA_TRY(ensureCloud(lod, frame, drawn));
        {
            gpu::CommandBatch batch(device);
            uint32_t base = 0;
            for (size_t part = 0; part < parts; ++part) {
                if (read[part] == 0) {
                    continue;
                }
                const bool merged = part < levels;
                const scene::GpuSplats& source = merged ? lod.levels[part].gaussians : lod.splats;
                gather_.dispatch(batch, {countOf(part), 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["selected"].setBinding(frame.selected[part].rhi());
                    cursor["dest"].setBinding(frame.dest[part].rhi());
                    cursor["srcPositions"].setBinding(source.positions.rhi());
                    cursor["srcShape"].setBinding(source.shape.rhi());
                    cursor["srcSh"].setBinding(source.sh.rhi());
                    cursor["positions"].setBinding(frame.cloud.positions.rhi());
                    cursor["shape"].setBinding(frame.cloud.shape.rhi());
                    cursor["sh"].setBinding(frame.cloud.sh.rhi());
                    bindNormals(cursor, source, frame.cloud);
                    bindEmission(cursor, source, frame.cloud);
                    cursor["params"]["count"].setData(countOf(part));
                    cursor["params"]["base"].setData(base);
                    cursor["params"]["offset"].setData(merged ? 0u : runs[part - levels].offset);
                    cursor["params"]["shWords"].setData(lod.splats.shWords);
                });
                // And what the cloud carries beside, the same gaussians'.
                for (const Extra& e : extrasOf(lod.splats)) {
                    if (!(source.*e.member).valid()) {
                        continue;
                    }
                    extrasGather_.dispatch(batch, {countOf(part), 1, 1}, [&](rhi::ShaderCursor cursor) {
                        for (const char* unused : {"order", "starts"}) {
                            cursor[unused].setBinding(frame.dest[part].rhi());
                        }
                        cursor["positions"].setBinding(source.positions.rhi());
                        cursor["selected"].setBinding(frame.selected[part].rhi());
                        cursor["dest"].setBinding(frame.dest[part].rhi());
                        cursor["source"].setBinding((source.*e.member).rhi());
                        cursor["target"].setBinding((frame.cloud.*e.member).rhi());
                        cursor["params"]["count"].setData(countOf(part));
                        cursor["params"]["words"].setData(e.words);
                        cursor["params"]["base"].setData(base);
                        cursor["params"]["offset"].setData(merged ? 0u : runs[part - levels].offset);
                    });
                }
                base += read[part];
            }
            ATHENEA_TRY(batch.submit(true));
        }
        if (stats != nullptr) {
            CutStats s;
            s.splats = splatsDrawn;
            s.merged = drawn - splatsDrawn;
            s.available = lod.count;
            if (lod.streamed) {
                s.needs.resize(chunks);
                for (uint32_t c = 0; c < chunks; ++c) {
                    s.needs[c] = read[parts + c];
                }
            }
            stats->push_back(std::move(s));
        }
        if (drawn > 0) {
            out.push_back({&frame.cloud, instance.objectToWorld, instance.edit});
        }
    }
    return out;
}

Result<Decimator> Decimator::create(gpu::ShaderLibrary& library) {
    Decimator d;
    d.device_ = &library.device();
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    d.prefix_ = std::move(*prefix);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    ATHENEA_TRY(make(d.errorFinest_, "athenea/lod/lod_decimate", "lodErrorFinest"));
    ATHENEA_TRY(make(d.errorLevel_, "athenea/lod/lod_decimate", "lodErrorLevel"));
    ATHENEA_TRY(make(d.keepGroups_, "athenea/lod/lod_decimate", "lodDecimateGroups"));
    ATHENEA_TRY(make(d.keepSplats_, "athenea/lod/lod_decimate", "lodDecimateSplats"));
    ATHENEA_TRY(make(d.keepShape_, "athenea/lod/lod_decimate", "lodDecimateShape"));
    ATHENEA_TRY(make(d.recordOf_, "athenea/lod/lod_attributes", "lodRecordOf"));
    ATHENEA_TRY(make(d.gatherRanges_, "athenea/lod/lod_attributes", "lodGatherRanges"));
    ATHENEA_TRY(make(d.attributeKey_, "athenea/lod/lod_attributes", "lodAttributeKey"));
    ATHENEA_TRY(make(d.attributeMaterial_, "athenea/lod/lod_attributes", "lodAttributeMaterial"));
    ATHENEA_TRY(make(d.mergeAttribute_, "athenea/lod/lod_attributes", "lodMergeAttribute"));
    ATHENEA_TRY(make(d.gather_, "athenea/lod/lod_gather", "lodGather"));
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    d.sort_ = std::move(*sort);
    ATHENEA_TRY(make(d.pairInit_, "athenea/lod/lod_pairs", "pairInit"));
    ATHENEA_TRY(make(d.pairCodes_, "athenea/lod/lod_pairs", "pairCodes"));
    ATHENEA_TRY(make(d.pairBest_, "athenea/lod/lod_pairs", "pairBest"));
    ATHENEA_TRY(make(d.pairHistogram_, "athenea/lod/lod_pairs", "pairHistogram"));
    ATHENEA_TRY(make(d.pairMerge_, "athenea/lod/lod_pairs", "pairMerge"));
    ATHENEA_TRY(make(d.pairCompact_, "athenea/lod/lod_pairs", "pairCompact"));
    ATHENEA_TRY(make(d.pairRelabel_, "athenea/lod/lod_pairs", "pairRelabel"));
    ATHENEA_TRY(make(d.pairCount_, "athenea/lod/lod_pairs_final", "pairCount"));
    ATHENEA_TRY(make(d.pairScatter_, "athenea/lod/lod_pairs_final", "pairScatter"));
    ATHENEA_TRY(make(d.pairGather_, "athenea/lod/lod_pairs_final", "pairGather"));
    ATHENEA_TRY(make(d.pairFinalize_, "athenea/lod/lod_pairs_final", "pairFinalize"));
    ATHENEA_TRY(make(d.pairRootSizes_, "athenea/lod/lod_pairs", "pairRootSizes"));
    ATHENEA_TRY(make(d.pairTreeRoots_, "athenea/lod/lod_pairs", "pairTreeRoots"));
    ATHENEA_TRY(make(d.pairTreeOffsets_, "athenea/lod/lod_pairs", "pairTreeOffsets"));
    ATHENEA_TRY(make(d.pairTreeSizes_, "athenea/lod/lod_pairs", "pairTreeSizes"));
    ATHENEA_TRY(make(d.pairTreeLayout_, "athenea/lod/lod_pairs_final", "pairTreeLayout"));
    ATHENEA_TRY(make(d.pruneFold_, "athenea/lod/lod_prune", "pruneFold"));
    ATHENEA_TRY(make(d.pruneHistogram_, "athenea/lod/lod_prune", "pruneHistogram"));
    ATHENEA_TRY(make(d.pruneKeep_, "athenea/lod/lod_prune", "pruneKeep"));
    ATHENEA_TRY(make(d.pruneScatter_, "athenea/lod/lod_prune", "pruneScatter"));
    return d;
}

float Decimator::scoreOfBin(uint32_t bin) noexcept {
    return std::pow(10.0F, -12.0F + static_cast<float>(bin) * (18.0F / static_cast<float>(kScoreBins)));
}

Result<void> Decimator::foldScore(gpu::Buffer& frameCount, gpu::Buffer& score, uint32_t count, float scale,
                                  bool wide) {
    gpu::CommandBatch batch(*device_);
    pruneFold_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["frameCount"].setBinding(frameCount.rhi());
        cursor["score"].setBinding(score.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["scale"].setData(scale);
        cursor["params"]["wide"].setData(uint32_t{wide ? 1u : 0u});
    });
    return batch.submit(true);
}

Result<std::vector<uint32_t>> Decimator::scoreHistogram(const gpu::Buffer& score, uint32_t count) {
    gpu::Device& device = *device_;
    auto histogram = buffer(device, kScoreBins, 4, "prune.histogram");
    if (!histogram) return std::move(histogram).error();
    gpu::CommandBatch batch(device);
    batch.encoder()->clearBuffer(histogram->rhi(), 0, uint64_t{kScoreBins} * 4);
    batch.markDirty();
    pruneHistogram_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["score"].setBinding(score.rhi());
        cursor["histogram"].setBinding(histogram->rhi());
        cursor["params"]["count"].setData(count);
    });
    ATHENEA_TRY(batch.submit(true));
    std::vector<uint32_t> bins(kScoreBins, 0);
    ATHENEA_TRY(histogram->read(device, 0, uint64_t{kScoreBins} * 4, bins.data()));
    return bins;
}

Result<LodCloud> Decimator::prune(const LodCloud& lod, const gpu::Buffer& score, float threshold) {
    gpu::Device& device = *device_;
    const uint32_t n = lod.count;
    if (n == 0 || !lod.order.valid()) {
        return Error(ErrorCode::InvalidArgument, lod.splats.source + ": a prune needs the store's order");
    }
    const auto make = [&](uint64_t count, uint32_t element, const char* label) {
        return buffer(device, std::max<uint64_t>(count, 1), element, label);
    };
    auto keep = make(n, 4, "prune.keep");
    auto offsets = make(n, 4, "prune.offsets");
    auto total = make(1, 4, "prune.total");
    if (!keep || !offsets || !total) {
        return Error(ErrorCode::OutOfMemory, "prune: cannot allocate its flags");
    }
    const bool normals = lod.splats.hasNormals();
    const uint32_t shWords = lod.splats.shWords;
    {
        gpu::CommandBatch batch(device);
        pruneKeep_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["score"].setBinding(score.rhi());
            cursor["keep"].setBinding(keep->rhi());
            cursor["params"]["count"].setData(n);
            cursor["params"]["threshold"].setData(threshold);
        });
        batch.markDirty();
        ATHENEA_TRY(prefix_.apply(batch, *keep, *offsets, *total, n));
        ATHENEA_TRY(batch.submit(true));
    }
    uint32_t kept = 0;
    ATHENEA_TRY(total->read(device, 0, 4, &kept));
    if (kept == 0) {
        return Error(ErrorCode::InvalidArgument, lod.splats.source + ": no splat scored enough; nothing left to keep");
    }
    LodCloud out;
    out.count = kept;
    out.chunkSplats = kept;
    out.slots = {0};
    out.boundsLo[0] = lod.boundsLo[0];
    out.boundsLo[1] = lod.boundsLo[1];
    out.boundsLo[2] = lod.boundsLo[2];
    out.extent = lod.extent;
    out.splats.source = lod.splats.source;
    out.splats.count = kept;
    out.splats.declared = lod.splats.declared;
    out.splats.restPerColour = lod.splats.restPerColour;
    out.splats.shWords = shWords;
    out.splats.linear = lod.splats.linear;
    out.splats.antialiased = lod.splats.antialiased;
    out.splats.bounds = lod.splats.bounds;
    auto positions = make(kept, 16, "prune.positions");
    auto shape = make(uint64_t{kept} * 4, 4, "prune.shape");
    auto sh = make(uint64_t{kept} * std::max(shWords, 1u), 4, "prune.sh");
    auto normalWords = make(normals ? kept : 1, 4, "prune.normals");
    auto order = make(kept, 4, "prune.order");
    if (!positions || !shape || !sh || !normalWords || !order) {
        return Error(ErrorCode::OutOfMemory, "prune: cannot allocate the pruned store");
    }
    {
        gpu::CommandBatch batch(device);
        pruneScatter_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["storeOrder"].setBinding(lod.order.rhi());
            cursor["keep"].setBinding(keep->rhi());
            cursor["offsets"].setBinding(offsets->rhi());
            cursor["inPositions"].setBinding(lod.splats.positions.rhi());
            cursor["inShape"].setBinding(lod.splats.shape.rhi());
            cursor["inSh"].setBinding(shWords > 0 ? lod.splats.sh.rhi() : keep->rhi());
            cursor["inNormals"].setBinding(normals ? lod.splats.normals.rhi() : keep->rhi());
            cursor["outPositions"].setBinding(positions->rhi());
            cursor["outShape"].setBinding(shape->rhi());
            cursor["outSh"].setBinding(sh->rhi());
            cursor["outNormals"].setBinding(normalWords->rhi());
            cursor["outOrder"].setBinding(order->rhi());
            cursor["params"]["count"].setData(n);
            cursor["params"]["shWords"].setData(shWords);
            cursor["params"]["normals"].setData(uint32_t{normals ? 1u : 0u});
        });
        ATHENEA_TRY(batch.submit(true));
    }
    out.splats.positions = std::move(*positions);
    out.splats.shape = std::move(*shape);
    out.splats.sh = std::move(*sh);
    if (normals) {
        out.splats.normals = std::move(*normalWords);
    }
    out.order = std::move(*order);
    return out;
}

namespace {

constexpr uint32_t kNodeFloats = 24;       // lod_pairs.slang's
constexpr uint32_t kHistogramBins = 256;
constexpr float    kHistogramLo = -12.0F;  // log10 of the cheapest cost binned
constexpr float    kHistogramStep = 18.0F / kHistogramBins;
constexpr float    kNoMerge = 1.0e30F;

}   // namespace

Result<DecimateResult> Decimator::decimatePairs(const LodCloud& lod, const gpu::Buffer& origin,
                                                const PairSettings& settings, const DecimateCarried& carried,
                                                DecimateStats* stats, DecimateResult* treeOut) {
    gpu::Device& device = *device_;
    const uint32_t n = lod.count;
    if (n == 0) {
        return Error(ErrorCode::InvalidArgument, lod.splats.source + ": nothing to decimate");
    }
    if (!lod.order.valid() || !origin.valid()) {
        return Error(ErrorCode::InvalidArgument,
                     lod.splats.source + ": a decimation needs the cloud's order and origin (a build in memory)");
    }
    for (const int32_t slot : lod.slots) {
        if (slot < 0) {
            return Error(ErrorCode::InvalidArgument,
                         lod.splats.source + ": a decimation needs every chunk on the device");
        }
    }
    const uint32_t target = std::max(1u, static_cast<uint32_t>(std::ceil(n / std::max(settings.ratio, 1.0F))));
    const auto make = [&](uint64_t count, uint32_t element, const char* label) {
        return buffer(device, std::max<uint64_t>(count, 1), element, label);
    };
    auto recordOf = make(n, 4, "pairs.recordOf");
    auto nothing = make(4, 16, "pairs.nothing");
    auto nodesA = make(uint64_t{n} * kNodeFloats, 4, "pairs.nodes");
    auto nodesB = make(uint64_t{n} * kNodeFloats, 4, "pairs.nodes2");
    auto best = make(n, 4, "pairs.best");
    auto bestCost = make(n, 4, "pairs.bestCost");
    auto histogram = make(kHistogramBins, 4, "pairs.histogram");
    auto alive = make(n, 4, "pairs.alive");
    auto absorbedBy = make(n, 4, "pairs.absorbedBy");
    auto slotOf = make(n, 4, "pairs.slotOf");
    auto total = make(1, 4, "pairs.total");
    auto clusterOf = make(n, 4, "pairs.clusterOf");
    if (!recordOf || !nothing || !nodesA || !nodesB || !best || !bestCost || !histogram || !alive || !absorbedBy ||
        !slotOf || !total || !clusterOf) {
        return Error(ErrorCode::OutOfMemory, "decimate: cannot allocate the pairs' buffers");
    }
    gpu::SortBuffers sorting;
    for (auto [into, label] : {std::pair{&sorting.keysLo, "pairs.codes"}, std::pair{&sorting.values, "pairs.order"},
                               std::pair{&sorting.scratchKeysLo, "pairs.codes2"},
                               std::pair{&sorting.scratchValues, "pairs.order2"}}) {
        auto made = make(n, 4, label);
        if (!made) return std::move(made).error();
        *into = std::move(*made);
    }
    const gpu::Buffer& keys = carried.keys != nullptr ? *carried.keys : *nothing;
    const gpu::Buffer& material = carried.material != nullptr ? *carried.material : *nothing;
    const bool normals = lod.splats.hasNormals();
    const gpu::Buffer& normalWords = normals ? lod.splats.normals : *nothing;
    // The second Morton order's shift: half the finest cell a 30-bit code
    // tells apart, on every axis.
    const float shift = lod.extent / static_cast<float>(1u << 10) * 0.5F;
    // ABOVE WHAT THE GUARDS ALLOW, for a tree: once nothing more may merge,
    // the roots go on merging with the colour edge, the normals' angle, the
    // glass, the aspect and the pixel limit lifted, so that a far view has
    // coarse levels to cut to. The flat cloud is what stood before.
    bool relaxed = false;
    const auto setParams = [&](rhi::ShaderCursor p, uint32_t count) {
        p["count"].setData(count);
        p["relaxed"].setData(uint32_t{relaxed ? 1u : 0u});
        p["keyed"].setData(uint32_t{carried.keys != nullptr ? 1u : 0u});
        p["materialled"].setData(uint32_t{carried.material != nullptr ? 1u : 0u});
        p["normals"].setData(uint32_t{normals ? 1u : 0u});
        p["boundsLoX"].setData(lod.boundsLo[0] - shift);
        p["boundsLoY"].setData(lod.boundsLo[1] - shift);
        p["boundsLoZ"].setData(lod.boundsLo[2] - shift);
        p["extent"].setData(lod.extent + 2.0F * shift);
        p["boundsShiftX"].setData(shift);
        p["boundsShiftY"].setData(shift);
        p["boundsShiftZ"].setData(shift);
        p["pixelsPerUnit"].setData(settings.pixelsPerUnit);
        p["eyeX"].setData(settings.eye[0]);
        p["eyeY"].setData(settings.eye[1]);
        p["eyeZ"].setData(settings.eye[2]);
        p["edge"].setData(relaxed ? 2.0F : settings.edge);
        p["maxCos"].setData(relaxed ? -1.0F
                                    : static_cast<float>(std::cos(settings.maxAngle * 3.14159265358979 / 180.0)));
        p["colourWeight"].setData(settings.colourWeight);
        p["normalWeight"].setData(settings.normalWeight);
        p["materialWeight"].setData(settings.materialWeight);
        p["flatWeight"].setData(settings.flatWeight);
        p["highlightCone"].setData(settings.highlightCone);
        p["mergeGlass"].setData(uint32_t{settings.mergeGlass || relaxed ? 1u : 0u});
        p["maxAspect"].setData(relaxed ? 1.0e30F : settings.maxAspect);
        p["maxPixels"].setData(relaxed ? 0.0F : settings.maxPixels);
        p["histogramBins"].setData(kHistogramBins);
        p["histogramLo"].setData(kHistogramLo);
        p["histogramStep"].setData(kHistogramStep);
    };
    const auto bindAll = [&](rhi::ShaderCursor c, gpu::Buffer& nodes) {
        c["splatPositions"].setBinding(lod.splats.positions.rhi());
        c["splatShape"].setBinding(lod.splats.shape.rhi());
        c["splatNormals"].setBinding(normalWords.rhi());
        c["recordOf"].setBinding(recordOf->rhi());
        c["keys"].setBinding(keys.rhi());
        c["material"].setBinding(material.rhi());
        c["nodes"].setBinding(nodes.rhi());
        c["codes"].setBinding(sorting.keysLo.rhi());
        c["order"].setBinding(sorting.values.rhi());
        c["sorted"].setBinding(sorting.values.rhi());
        c["best"].setBinding(best->rhi());
        c["bestCost"].setBinding(bestCost->rhi());
        c["histogram"].setBinding(histogram->rhi());
        c["alive"].setBinding(alive->rhi());
        c["absorbedBy"].setBinding(absorbedBy->rhi());
        c["slotOf"].setBinding(slotOf->rhi());
        c["nodesIn"].setBinding(nodes.rhi());
        c["nodesOut"].setBinding(nodes.rhi());
        c["clusterOf"].setBinding(clusterOf->rhi());
    };
    // Every store splat a node, and its own cluster.
    {
        gpu::CommandBatch batch(device);
        recordOf_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["storeOrder"].setBinding(lod.order.rhi());
            cursor["origin"].setBinding(origin.rhi());
            cursor["recordOut"].setBinding(recordOf->rhi());
            cursor["params"]["count"].setData(n);
        });
        batch.markDirty();
        pairInit_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bindAll(cursor, *nodesA);
            setParams(cursor["params"], n);
        });
        ATHENEA_TRY(batch.submit(true));
        std::vector<uint32_t> identity(n);
        for (uint32_t s = 0; s < n; ++s) {
            identity[s] = s;   // bookkeeping: each splat its own cluster to start
        }
        gpu::BufferDesc desc;
        desc.bytes = uint64_t{n} * 4;
        desc.elementBytes = 4;
        desc.label = "pairs.identity";
        auto ident = gpu::Buffer::create(device, desc, identity.data());
        if (!ident) return std::move(ident).error();
        gpu::CommandBatch copy(device);
        copy.encoder()->copyBuffer(clusterOf->rhi(), 0, ident->rhi(), 0, uint64_t{n} * 4);
        copy.markDirty();
        ATHENEA_TRY(copy.submit(true));
    }
    // THE MERGE TREE, where asked for: every merge a node after the leaves.
    const bool tree = treeOut != nullptr;
    const uint32_t treeNodes = tree ? 2 * n : 1;
    auto treeParent = make(treeNodes, 4, "pairs.treeParent");
    auto treeChildren = make(n, 8, "pairs.treeChildren");
    auto treeCost = make(treeNodes, 4, "pairs.treeCost");
    auto treePass = make(n, 4, "pairs.treePass");
    auto treeMade = make(1, 4, "pairs.treeMade");
    auto treeStart = make(treeNodes, 4, "pairs.treeStart");
    auto treeSize = make(treeNodes, 4, "pairs.treeSize");
    auto flatClusterOf = make(n, 4, "pairs.flatClusterOf");
    if (!treeParent || !treeChildren || !treeCost || !treePass || !treeMade || !treeStart || !treeSize ||
        !flatClusterOf) {
        return Error(ErrorCode::OutOfMemory, "decimate: cannot allocate the merge tree");
    }
    if (tree) {
        std::vector<uint32_t> identity(treeNodes);
        for (uint32_t k = 0; k < treeNodes; ++k) {
            identity[k] = k;   // bookkeeping: every node its own parent until merged
        }
        ATHENEA_TRY(treeParent->write(device, 0, uint64_t{treeNodes} * 4, identity.data()));
        gpu::CommandBatch batch(device);
        batch.encoder()->clearBuffer(treeCost->rhi(), 0, uint64_t{treeNodes} * 4);
        batch.encoder()->clearBuffer(treeMade->rhi(), 0, 4);
        batch.markDirty();
        ATHENEA_TRY(batch.submit(true));
    }
    const auto bindTree = [&](rhi::ShaderCursor c) {
        c["treeParent"].setBinding(treeParent->rhi());
        c["treeChildren"].setBinding(treeChildren->rhi());
        c["treeCost"].setBinding(treeCost->rhi());
        c["treePass"].setBinding(treePass->rhi());
        c["treeMade"].setBinding(treeMade->rhi());
        c["treeStart"].setBinding(treeStart->rhi());
        c["treeSize"].setBinding(treeSize->rhi());
        c["rootStart"].setBinding(slotOf->rhi());
        c["params"]["tree"].setData(uint32_t{tree ? 1u : 0u});
        c["params"]["leaves"].setData(n);
    };
    // The tree goes on to `treeRatio` (one root a key at most); the flat cloud
    // is the state the passes were in when the count first reached `target`.
    const uint32_t stopAt =
        tree ? std::max(1u, static_cast<uint32_t>(std::ceil(n / std::max(settings.treeRatio, settings.ratio))))
             : target;
    gpu::Buffer* nodes = &*nodesA;
    gpu::Buffer* spare = &*nodesB;
    uint32_t count = n;
    uint32_t passes = 0;
    uint32_t flatCount = 0;
    uint32_t flatPasses = 0;
    const auto snapshot = [&]() -> Result<void> {
        gpu::CommandBatch copy(device);
        copy.encoder()->copyBuffer(flatClusterOf->rhi(), 0, clusterOf->rhi(), 0, uint64_t{n} * 4);
        copy.markDirty();
        ATHENEA_TRY(copy.submit(true));
        flatCount = count;
        flatPasses = passes;
        return ok();
    };
    if (count <= target) {
        ATHENEA_TRY(snapshot());
    }
    while (count > stopAt && passes < settings.passes) {
        ++passes;
        {
            gpu::CommandBatch batch(device);
            for (uint32_t order = 0; order < 2; ++order) {
                pairCodes_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    bindAll(cursor, *nodes);
                    bindTree(cursor);
                    setParams(cursor["params"], count);
                    cursor["params"]["shift"].setData(order);
                });
                batch.markDirty();
                ATHENEA_TRY(sort_.sort(batch, sorting, count, 30));
                batch.markDirty();
                pairBest_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    bindAll(cursor, *nodes);
                    bindTree(cursor);
                    setParams(cursor["params"], count);
                    cursor["params"]["shift"].setData(order);
                });
                batch.markDirty();
            }
            batch.encoder()->clearBuffer(histogram->rhi(), 0, uint64_t{kHistogramBins} * 4);
            batch.markDirty();
            pairHistogram_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                bindAll(cursor, *nodes);
                bindTree(cursor);
                setParams(cursor["params"], count);
            });
            ATHENEA_TRY(batch.submit(true));
        }
        std::vector<uint32_t> bins(kHistogramBins, 0);
        ATHENEA_TRY(histogram->read(device, 0, kHistogramBins * 4, bins.data()));
        uint64_t pairs = 0;
        for (const uint32_t b : bins) {
            pairs += b;
        }
        if (pairs == 0) {
            if (tree && settings.relaxTop && !relaxed) {
                if (flatCount == 0) {
                    ATHENEA_TRY(snapshot());
                }
                relaxed = true;
                continue;
            }
            break;   // nothing left may merge
        }
        // As many of the cheapest pairs as take the count to the target --
        // the flat cloud's, and past it the tree's.
        const uint64_t wanted = count - (flatCount == 0 ? target : stopAt);
        float threshold = kNoMerge;
        if (pairs > wanted) {
            uint64_t sum = 0;
            for (uint32_t b = 0; b < kHistogramBins; ++b) {
                sum += bins[b];
                if (sum >= wanted) {
                    threshold = std::pow(10.0F, kHistogramLo + static_cast<float>(b + 1) * kHistogramStep);
                    break;
                }
            }
        }
        // NO MERGE COSTS MORE THAN `maxCost` in the flat cloud: where every
        // pair left would, the flat cloud is what stands now -- the smooth
        // panels gone further than the ratio, the creases, edges and badges
        // kept -- and only the tree goes on.
        if (settings.maxCost > 0.0F && flatCount == 0) {
            uint64_t cheap = 0;
            for (uint32_t b = 0; b < kHistogramBins; ++b) {
                if (kHistogramLo + static_cast<float>(b + 1) * kHistogramStep > std::log10(settings.maxCost)) {
                    break;
                }
                cheap += bins[b];
            }
            if (cheap == 0) {
                ATHENEA_TRY(snapshot());
                if (!tree) {
                    break;
                }
            } else {
                threshold = std::min(threshold, settings.maxCost);
            }
        }
        {
            gpu::CommandBatch batch(device);
            pairMerge_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                bindAll(cursor, *nodes);
                bindTree(cursor);
                setParams(cursor["params"], count);
                cursor["params"]["threshold"].setData(threshold);
                cursor["params"]["pass"].setData(passes);
            });
            batch.markDirty();
            ATHENEA_TRY(prefix_.apply(batch, *alive, *slotOf, *total, count));
            batch.markDirty();
            pairCompact_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                bindAll(cursor, *nodes);
                bindTree(cursor);
                cursor["nodesIn"].setBinding(nodes->rhi());
                cursor["nodesOut"].setBinding(spare->rhi());
                setParams(cursor["params"], count);
            });
            pairRelabel_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                bindAll(cursor, *nodes);
                bindTree(cursor);
                setParams(cursor["params"], n);
            });
            ATHENEA_TRY(batch.submit(true));
        }
        uint32_t kept = count;
        ATHENEA_TRY(total->read(device, 0, 4, &kept));
        std::swap(nodes, spare);
        if (kept == count) {
            if (tree && settings.relaxTop && !relaxed) {
                if (flatCount == 0) {
                    ATHENEA_TRY(snapshot());
                }
                relaxed = true;
                continue;
            }
            break;
        }
        count = kept;
        if (flatCount == 0 && count <= target) {
            ATHENEA_TRY(snapshot());
        }
    }
    if (flatCount == 0) {
        ATHENEA_TRY(snapshot());   // the target was not reached: what could merge, merged
    }

    // OUTPUTS AS RUNS OF THE STORE, and each one's gaussian: `members` the
    // store in run order, `starts` and `counts` each output's run.
    const auto finalize = [&](uint32_t outputs, gpu::Buffer& members, gpu::Buffer& starts, gpu::Buffer& counts,
                              DecimateResult& into, uint32_t& surfelCount) -> Result<void> {
        auto permPositions = make(n, 16, "pairs.storePositions");
        auto permShape = make(uint64_t{n} * 4, 4, "pairs.storeShape");
        auto permRecordOf = make(n, 4, "pairs.storeRecordOf");
        auto ranges = make(outputs, 8, "pairs.ranges");
        auto surfels = make(1, 4, "pairs.surfels");
        auto coverage = make(outputs, 4, "pairs.coverage");
        auto hardness = make(outputs, 4, "pairs.hardness");
        if (!permPositions || !permShape || !permRecordOf || !ranges || !surfels || !coverage || !hardness) {
            return Error(ErrorCode::OutOfMemory, "decimate: cannot allocate the clusters");
        }
        auto out = packed(device, outputs, lod.splats.restPerColour, lod.splats.shWords, "pairs.cloud", normals, false);
        if (!out) return std::move(out).error();
        out->source = lod.splats.source;
        out->linear = lod.splats.linear;
    out->antialiased = lod.splats.antialiased;
        out->antialiased = lod.splats.antialiased;
        out->count = outputs;
        out->declared = outputs;
        out->bounds = lod.splats.bounds;
        gpu::CommandBatch batch(device);
        batch.encoder()->clearBuffer(surfels->rhi(), 0, 4);
        batch.markDirty();
        const auto bindFinal = [&](rhi::ShaderCursor c, uint32_t count) {
            c["counts"].setBinding(counts.rhi());
            c["starts"].setBinding(starts.rhi());
            c["members"].setBinding(members.rhi());
            c["storePositions"].setBinding(lod.splats.positions.rhi());
            c["storeShape"].setBinding(lod.splats.shape.rhi());
            c["storeSh"].setBinding(lod.splats.shWords > 0 ? lod.splats.sh.rhi() : nothing->rhi());
            c["storeNormals"].setBinding(normalWords.rhi());
            c["storeRecordOf"].setBinding(recordOf->rhi());
            c["permPositions"].setBinding(permPositions->rhi());
            c["permShape"].setBinding(permShape->rhi());
            c["permRecordOf"].setBinding(permRecordOf->rhi());
            c["ranges"].setBinding(ranges->rhi());
            c["outPositions"].setBinding(out->positions.rhi());
            c["outShape"].setBinding(out->shape.rhi());
            c["outSh"].setBinding(lod.splats.shWords > 0 ? out->sh.rhi() : nothing->rhi());
            c["outNormals"].setBinding(normals ? out->normals.rhi() : nothing->rhi());
            c["surfelCount"].setBinding(surfels->rhi());
            c["outCoverage"].setBinding(coverage->rhi());
            c["outHardness"].setBinding(hardness->rhi());
            rhi::ShaderCursor p = c["params"];
            p["count"].setData(count);
            p["shWords"].setData(lod.splats.shWords);
            p["keep"].setData(lod.splats.restPerColour);
            p["surfels"].setData(uint32_t{settings.surfels ? 1u : 0u});
            p["normals"].setData(uint32_t{normals ? 1u : 0u});
            p["flatTolerance"].setData(settings.flatTolerance);
            p["opacityRule"].setData(settings.opacityRule);
            p["hardness"].setData(uint32_t{settings.hardness ? 1u : 0u});
            p["maxHardness"].setData(settings.maxHardness);
        };
        pairGather_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor c) { bindFinal(c, n); });
        pairFinalize_.dispatch(batch, {outputs, 1, 1}, [&](rhi::ShaderCursor c) { bindFinal(c, outputs); });
        ATHENEA_TRY(batch.submit(true));
        ATHENEA_TRY(surfels->read(device, 0, 4, &surfelCount));
        into.cloud = std::move(*out);
        into.ranges = std::move(*ranges);
        into.recordOf = std::move(*permRecordOf);
        into.storePositions = std::move(*permPositions);
        into.storeShape = std::move(*permShape);
        into.coverage = std::move(*coverage);
        into.hardness = std::move(*hardness);
        return ok();
    };

    // The flat cloud: its clusters made runs (count, scan, scatter).
    const uint32_t clusters = flatCount;
    auto counts = make(clusters, 4, "pairs.counts");
    auto starts = make(clusters, 4, "pairs.starts");
    auto filled = make(clusters, 4, "pairs.filled");
    auto members = make(n, 4, "pairs.members");
    if (!counts || !starts || !filled || !members) {
        return Error(ErrorCode::OutOfMemory, "decimate: cannot allocate the clusters");
    }
    {
        gpu::CommandBatch batch(device);
        batch.encoder()->clearBuffer(counts->rhi(), 0, uint64_t{std::max(clusters, 1u)} * 4);
        batch.encoder()->clearBuffer(filled->rhi(), 0, uint64_t{std::max(clusters, 1u)} * 4);
        batch.markDirty();
        const auto bindRuns = [&](rhi::ShaderCursor c) {
            c["clusterOf"].setBinding(flatClusterOf->rhi());
            c["counts"].setBinding(counts->rhi());
            c["starts"].setBinding(starts->rhi());
            c["filled"].setBinding(filled->rhi());
            c["members"].setBinding(members->rhi());
            c["params"]["count"].setData(n);
        };
        pairCount_.dispatch(batch, {n, 1, 1}, bindRuns);
        batch.markDirty();
        ATHENEA_TRY(prefix_.apply(batch, *counts, *starts, *total, clusters));
        batch.markDirty();
        pairScatter_.dispatch(batch, {n, 1, 1}, bindRuns);
        ATHENEA_TRY(batch.submit(true));
    }
    DecimateResult result;
    uint32_t surfelCount = 0;
    ATHENEA_TRY(finalize(clusters, *members, *starts, *counts, result, surfelCount));

    if (tree) {
        // Sizes from the first pass on, the roots' starts, then each pass's
        // children placed from the last pass back; then the tree as a cloud.
        uint32_t made = 0;
        ATHENEA_TRY(treeMade->read(device, 0, 4, &made));
        const uint32_t outputs = n + made;
        auto treeMembers = make(n, 4, "pairs.treeMembers");
        auto treeStarts = make(outputs, 4, "pairs.treeStarts");
        auto treeCounts = make(outputs, 4, "pairs.treeCounts");
        auto parentOut = make(outputs, 4, "pairs.parent");
        auto costOut = make(outputs, 4, "pairs.cost");
        auto runOut = make(outputs, 8, "pairs.run");
        if (!treeMembers || !treeStarts || !treeCounts || !parentOut || !costOut || !runOut) {
            return Error(ErrorCode::OutOfMemory, "decimate: cannot allocate the merge tree");
        }
        {
            gpu::CommandBatch batch(device);
            const auto treeKernel = [&](gpu::ComputeKernel& k, uint32_t threads, uint32_t pass) {
                k.dispatch(batch, {std::max(threads, 1u), 1, 1}, [&](rhi::ShaderCursor cursor) {
                    bindAll(cursor, *nodes);
                    bindTree(cursor);
                    setParams(cursor["params"], threads);
                    cursor["params"]["pass"].setData(pass);
                });
                batch.markDirty();
            };
            for (uint32_t p = 1; p <= passes; ++p) {
                treeKernel(pairTreeSizes_, made, p);
            }
            treeKernel(pairRootSizes_, count, 0);
            ATHENEA_TRY(prefix_.apply(batch, *alive, *slotOf, *total, count));
            batch.markDirty();
            treeKernel(pairTreeRoots_, count, 0);
            for (uint32_t p = passes; p >= 1; --p) {
                treeKernel(pairTreeOffsets_, made, p);
            }
            pairTreeLayout_.dispatch(batch, {outputs, 1, 1}, [&](rhi::ShaderCursor c) {
                c["treeStart"].setBinding(treeStart->rhi());
                c["treeSize"].setBinding(treeSize->rhi());
                c["treeParent"].setBinding(treeParent->rhi());
                c["treeCost"].setBinding(treeCost->rhi());
                c["members"].setBinding(treeMembers->rhi());
                c["starts"].setBinding(treeStarts->rhi());
                c["counts"].setBinding(treeCounts->rhi());
                c["outParent"].setBinding(parentOut->rhi());
                c["outCost"].setBinding(costOut->rhi());
                c["outRun"].setBinding(runOut->rhi());
                c["params"]["count"].setData(outputs);
                c["params"]["treeLeaves"].setData(n);
            });
            ATHENEA_TRY(batch.submit(true));
        }
        uint32_t treeSurfels = 0;
        ATHENEA_TRY(finalize(outputs, *treeMembers, *treeStarts, *treeCounts, *treeOut, treeSurfels));
        treeOut->parent = std::move(*parentOut);
        treeOut->cost = std::move(*costOut);
        treeOut->run = std::move(*runOut);
        std::printf("decimate: the merge tree: %u leaves, %u merges, %u roots, %u passes\n", n, made, count, passes);
    }
    if (stats != nullptr) {
        stats->before = n;
        stats->merged = clusters;
        stats->splats = 0;
        stats->surfels = surfelCount;
        stats->passes = flatPasses;
    }
    return result;
}

Result<DecimateResult> Decimator::decimate(const LodCloud& lod, const gpu::Buffer& origin,
                                           const DecimateSettings& settings, const DecimateCarried& carried,
                                           DecimateStats* stats) {
    gpu::Device& device = *device_;
    if (lod.levels.empty() || lod.count == 0) {
        return Error(ErrorCode::InvalidArgument, lod.splats.source + ": a decimation needs a merged level");
    }
    if (!lod.order.valid() || !origin.valid()) {
        return Error(ErrorCode::InvalidArgument,
                     lod.splats.source + ": a decimation needs the cloud's order and origin (a build in memory)");
    }
    // Each store splat's record, once: everything a record carries is read
    // through it.
    auto recordOf = buffer(device, lod.count, 4, "decimate.recordOf");
    if (!recordOf) return std::move(recordOf).error();
    // Something bound where nothing is carried: the kernels ask the flags
    // before they read.
    auto nothing = buffer(device, 4, 16, "decimate.nothing");
    if (!nothing) return std::move(nothing).error();
    const gpu::Buffer& keys = carried.keys != nullptr ? *carried.keys : *nothing;
    const gpu::Buffer& material = carried.material != nullptr ? *carried.material : *nothing;
    std::vector<gpu::Buffer> groupRange(lod.levels.size());
    for (size_t k = 0; k < lod.levels.size(); ++k) {
        auto made = buffer(device, lod.levels[k].gaussians.count, 8, "decimate.groupRange");
        if (!made) return std::move(made).error();
        groupRange[k] = std::move(*made);
    }
    for (const int32_t slot : lod.slots) {
        if (slot < 0) {
            return Error(ErrorCode::InvalidArgument,
                         lod.splats.source + ": a decimation needs every chunk on the device");
        }
    }
    const size_t levels = lod.levels.size();
    const size_t parts = levels + 1;   // the levels, then the splats
    const auto countOf = [&](size_t part) {
        return part < levels ? lod.levels[part].gaussians.count : lod.count;
    };
    std::vector<gpu::Buffer> errors(levels), keptPositions(levels), keptShape(levels), selected(parts),
        dest(parts), totals(parts);
    for (size_t part = 0; part < parts; ++part) {
        if (part < levels) {
            auto e = buffer(device, countOf(part), 4, "decimate.errors");
            auto kp = buffer(device, countOf(part), 16, "decimate.positions");
            auto ks = buffer(device, uint64_t{countOf(part)} * 4, 4, "decimate.shape");
            if (!e || !kp || !ks) return Error(ErrorCode::OutOfMemory, "decimate: cannot allocate its buffers");
            errors[part] = std::move(*e);
            keptPositions[part] = std::move(*kp);
            keptShape[part] = std::move(*ks);
        }
        auto sel = buffer(device, countOf(part), 4, "decimate.selected");
        auto dst = buffer(device, countOf(part), 4, "decimate.dest");
        auto tot = buffer(device, 1, 4, "decimate.total");
        if (!sel || !dst || !tot) return Error(ErrorCode::OutOfMemory, "decimate: cannot allocate its buffers");
        selected[part] = std::move(*sel);
        dest[part] = std::move(*dst);
        totals[part] = std::move(*tot);
    }
    auto readback = buffer(device, parts, 4, "decimate.readback");
    if (!readback) return std::move(readback).error();

    const auto setParams = [&](rhi::ShaderCursor p, uint32_t count, uint32_t children, uint32_t parents) {
        p["count"].setData(count);
        p["children"].setData(children);
        p["splats"].setData(lod.count);
        p["parents"].setData(parents);
        p["colourTolerance"].setData(std::max(settings.colourTolerance, 1.0e-6F));
        p["flatTolerance"].setData(std::max(settings.flatTolerance, 1.0e-6F));
        p["reach"].setData(std::max(settings.reach, 1.0e-6F));
        p["outliers"].setData(std::max(settings.outliers, 1.0e-6F));
        p["extent"].setData(lod.extent);
        p["hasKey"].setData(uint32_t{carried.keys != nullptr ? 1u : 0u});
        p["hasMaterial"].setData(uint32_t{carried.material != nullptr ? 1u : 0u});
    };
    {
        gpu::CommandBatch batch(device);
        const LodLevel& finestLevel = lod.levels.back();
        const auto setWhere = [&](rhi::ShaderCursor p, size_t k) {
            p["finest"].setData(finestLevel.gaussians.count);
            p["below"].setData(static_cast<uint32_t>(levels - 1 - k));
            p["level"].setData(lod.levels[k].level);
        };
        const auto bindSplats = [&](rhi::ShaderCursor cursor) {
            cursor["finestCells"].setBinding(finestLevel.cells.rhi());
            cursor["starts"].setBinding(lod.starts.rhi());
            cursor["splatPositions"].setBinding(lod.splats.positions.rhi());
            cursor["splatShape"].setBinding(lod.splats.shape.rhi());
            cursor["recordOf"].setBinding(recordOf->rhi());
            cursor["keys"].setBinding(keys.rhi());
            cursor["material"].setBinding(material.rhi());
        };
        recordOf_.dispatch(batch, {lod.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["storeOrder"].setBinding(lod.order.rhi());
            cursor["origin"].setBinding(origin.rhi());
            cursor["recordOut"].setBinding(recordOf->rhi());
            cursor["params"]["count"].setData(lod.count);
        });
        batch.markDirty();
        // Each level's merges as they would be kept: widened to cover what
        // they replace (lodDecimateShape says why). The errors read them.
        for (size_t k = 0; k < levels; ++k) {
            const LodLevel& level = lod.levels[k];
            const uint32_t count = level.gaussians.count;
            keepShape_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(level.gaussians.positions.rhi());
                cursor["shape"].setBinding(level.gaussians.shape.rhi());
                cursor["cells"].setBinding(level.cells.rhi());
                bindSplats(cursor);
                cursor["keptPositions"].setBinding(keptPositions[k].rhi());
                cursor["keptShape"].setBinding(keptShape[k].rhi());
                cursor["groupRange"].setBinding(groupRange[k].rhi());
                rhi::ShaderCursor p = cursor["params"];
                setParams(p, count, 0u, 0u);
                setWhere(p, k);
            });
        }
        batch.markDirty();
        // The errors, finest first: each level's are never below its
        // children's, so a coarser level needs the finer one done.
        for (size_t k = levels; k-- > 0;) {
            const LodLevel& level = lod.levels[k];
            const bool finest = k + 1 == levels;
            const uint32_t count = level.gaussians.count;
            (finest ? errorFinest_ : errorLevel_).dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(level.gaussians.positions.rhi());
                cursor["shape"].setBinding(level.gaussians.shape.rhi());
                cursor["cells"].setBinding(level.cells.rhi());
                cursor["widePositions"].setBinding(keptPositions[k].rhi());
                cursor["wideShape"].setBinding(keptShape[k].rhi());
                bindSplats(cursor);
                if (finest) {
                    cursor["childPositions"].setBinding(lod.splats.positions.rhi());
                    cursor["childShape"].setBinding(lod.splats.shape.rhi());
                } else {
                    const LodLevel& finer = lod.levels[k + 1];
                    cursor["childPositions"].setBinding(finer.gaussians.positions.rhi());
                    cursor["childShape"].setBinding(finer.gaussians.shape.rhi());
                    cursor["childCells"].setBinding(finer.cells.rhi());
                    cursor["childErrors"].setBinding(errors[k + 1].rhi());
                }
                cursor["errors"].setBinding(errors[k].rhi());
                rhi::ShaderCursor p = cursor["params"];
                setParams(p, count, finest ? 0u : lod.levels[k + 1].gaussians.count, 0u);
                setWhere(p, k);
            });
            batch.markDirty();
        }
        for (size_t k = 0; k < levels; ++k) {
            const LodLevel& level = lod.levels[k];
            const uint32_t count = level.gaussians.count;
            keepGroups_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["cells"].setBinding(level.cells.rhi());
                cursor["errors"].setBinding(errors[k].rhi());
                if (k > 0) {
                    cursor["parentErrors"].setBinding(errors[k - 1].rhi());
                    cursor["parentCells"].setBinding(lod.levels[k - 1].cells.rhi());
                }
                cursor["selected"].setBinding(selected[k].rhi());
                setParams(cursor["params"], count, 0u, k > 0 ? lod.levels[k - 1].gaussians.count : 0u);
            });
        }
        keepSplats_.dispatch(batch, {lod.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["groupOf"].setBinding(lod.groups.rhi());
            cursor["finestErrors"].setBinding(errors[levels - 1].rhi());
            cursor["selected"].setBinding(selected[levels].rhi());
            setParams(cursor["params"], lod.count, 0u, 0u);
        });
        for (size_t part = 0; part < parts; ++part) {
            ATHENEA_TRY(prefix_.apply(batch, selected[part], dest[part], totals[part], countOf(part)));
        }
        for (size_t part = 0; part < parts; ++part) {
            batch.encoder()->copyBuffer(readback->rhi(), part * sizeof(uint32_t), totals[part].rhi(), 0,
                                        sizeof(uint32_t));
        }
        batch.markDirty();
        ATHENEA_TRY(batch.submit(true));
    }
    std::vector<uint32_t> read(parts, 0);
    ATHENEA_TRY(readback->read(device, 0, parts * sizeof(uint32_t), read.data()));
    uint32_t kept = 0;
    for (const uint32_t n : read) {
        kept += n;
    }
    auto out = packed(device, std::max(kept, 1u), lod.splats.restPerColour, lod.splats.shWords, "decimate.cloud",
                      lod.splats.hasNormals(), lod.splats.hasEmission());
    if (!out) return std::move(out).error();
    auto ranges = buffer(device, std::max(kept, 1u), 8, "decimate.ranges");
    if (!ranges) return std::move(ranges).error();
    out->source = lod.splats.source;
    out->linear = lod.splats.linear;
    out->antialiased = lod.splats.antialiased;
    out->count = kept;
    out->declared = kept;
    out->bounds = lod.splats.bounds;
    {
        gpu::CommandBatch batch(device);
        uint32_t base = 0;
        for (size_t part = 0; part < parts; ++part) {
            if (read[part] == 0) {
                continue;
            }
            const bool merged = part < levels;
            const scene::GpuSplats& source = merged ? lod.levels[part].gaussians : lod.splats;
            gather_.dispatch(batch, {countOf(part), 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["selected"].setBinding(selected[part].rhi());
                cursor["dest"].setBinding(dest[part].rhi());
                cursor["srcPositions"].setBinding(merged ? keptPositions[part].rhi() : source.positions.rhi());
                cursor["srcShape"].setBinding(merged ? keptShape[part].rhi() : source.shape.rhi());
                cursor["srcSh"].setBinding(source.sh.rhi());
                cursor["positions"].setBinding(out->positions.rhi());
                cursor["shape"].setBinding(out->shape.rhi());
                cursor["sh"].setBinding(out->sh.rhi());
                bindNormals(cursor, source, *out);
                bindEmission(cursor, source, *out);
                cursor["params"]["count"].setData(countOf(part));
                cursor["params"]["base"].setData(base);
                cursor["params"]["offset"].setData(0u);
                cursor["params"]["shWords"].setData(lod.splats.shWords);
            });
            gatherRanges_.dispatch(batch, {countOf(part), 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["gatherSelected"].setBinding(selected[part].rhi());
                cursor["gatherDest"].setBinding(dest[part].rhi());
                cursor["levelRange"].setBinding(merged ? groupRange[part].rhi() : nothing->rhi());
                cursor["ranges"].setBinding(ranges->rhi());
                cursor["params"]["count"].setData(countOf(part));
                cursor["params"]["base"].setData(base);   // the first output slot of this part
                cursor["params"]["mode"].setData(uint32_t{merged ? 0u : 1u});
            });
            base += read[part];
        }
        ATHENEA_TRY(batch.submit(true));
    }
    if (stats != nullptr) {
        stats->splats = read[levels];
        stats->merged = kept - read[levels];
        stats->before = lod.count;
    }
    DecimateResult result;
    result.cloud = std::move(*out);
    result.ranges = std::move(*ranges);
    result.recordOf = std::move(*recordOf);
    return result;
}

Result<void> Decimator::addKey(gpu::Buffer& keys, uint32_t records, const gpu::Buffer& values, uint32_t width,
                               AttributeMerge how, const gpu::Buffer* weights) {
    if (records == 0 || width == 0) {
        return ok();
    }
    gpu::CommandBatch batch(*device_);
    attributeKey_.dispatch(batch, {records, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["values"].setBinding(values.rhi());
        cursor["weights"].setBinding(weights != nullptr ? weights->rhi() : values.rhi());
        cursor["keyOut"].setBinding(keys.rhi());
        cursor["params"]["count"].setData(records);
        cursor["params"]["width"].setData(width);
        cursor["params"]["mode"].setData(static_cast<uint32_t>(how));
    });
    return batch.submit(true);
}

Result<void> Decimator::addMaterial(gpu::Buffer& material, uint32_t records, const gpu::Buffer& values,
                                    uint32_t component) {
    if (records == 0) {
        return ok();
    }
    gpu::CommandBatch batch(*device_);
    attributeMaterial_.dispatch(batch, {records, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["values"].setBinding(values.rhi());
        cursor["materialOut"].setBinding(material.rhi());
        cursor["params"]["count"].setData(records);
        cursor["params"]["mode"].setData(component);
    });
    return batch.submit(true);
}

Result<std::pair<gpu::Buffer, gpu::Buffer>> Decimator::mergeAttribute(const LodCloud& lod, const DecimateResult& kept,
                                                                      const gpu::Buffer& values, uint32_t width,
                                                                      AttributeMerge how, const gpu::Buffer* weights) {
    gpu::Device& device = *device_;
    const uint32_t count = kept.cloud.count;
    auto merged = buffer(device, uint64_t{std::max(count, 1u)} * std::max(width, 1u), 4, "decimate.attribute");
    // Only a rig's joints (mode 3) write weights beside them: elsewhere one
    // word, not a second attribute's worth (1.7 GB for a transfer's 48 floats
    // over a 4.4M-leaf tree, which ran a 5120 px part out of device memory).
    auto mergedWeights = buffer(device, weights != nullptr ? uint64_t{std::max(count, 1u)} * std::max(width, 1u) : 1u,
                                4, "decimate.weights");
    if (!merged || !mergedWeights) return Error(ErrorCode::OutOfMemory, "decimate: cannot allocate an attribute");
    if (count > 0 && width > 0) {
        gpu::CommandBatch batch(device);
        mergeAttribute_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["ranges"].setBinding(kept.ranges.rhi());
            cursor["recordOf"].setBinding(kept.recordOf.rhi());
            // A merge by pairs reordered the store (DecimateResult::storePositions).
            const bool reordered = kept.storePositions.valid();
            cursor["splatPositions"].setBinding(reordered ? kept.storePositions.rhi() : lod.splats.positions.rhi());
            cursor["splatShape"].setBinding(reordered ? kept.storeShape.rhi() : lod.splats.shape.rhi());
            cursor["values"].setBinding(values.rhi());
            cursor["weights"].setBinding(weights != nullptr ? weights->rhi() : values.rhi());
            cursor["merged"].setBinding(merged->rhi());
            cursor["mergedWeights"].setBinding(mergedWeights->rhi());
            cursor["keptShape"].setBinding(kept.cloud.shape.rhi());
            cursor["params"]["count"].setData(count);
            cursor["params"]["width"].setData(width);
            cursor["params"]["mode"].setData(static_cast<uint32_t>(how));
        });
        ATHENEA_TRY(batch.submit(true));
    }
    return std::pair{std::move(*merged), std::move(*mergedWeights)};
}

}   // namespace athenea::lod
