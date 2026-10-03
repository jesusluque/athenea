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
                                const char* label) {
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
    return out;
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
    auto splats = packed(device, n, cloud.restPerColour, cloud.shWords, "lod.splats");
    if (!splats) return std::move(splats).error();
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
            cursor["params"]["count"].setData(n);
            cursor["params"]["shWords"].setData(cloud.shWords);
        });
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
    const uint32_t stride = kMomentsHead + keep * 3;
    gpu::Buffer fineMoments;
    std::vector<LodLevel> stored;
    for (uint32_t r = finest + 1; r-- > coarsest;) {
        Level& level = levels[r];
        auto moments = buffer(device, uint64_t{level.groups} * stride, 4, "lod.moments");
        if (!moments) return std::move(moments).error();
        auto gaussians = packed(device, level.groups, keep, cloud.shWords, "lod.merged");
        if (!gaussians) return std::move(gaussians).error();
        gpu::CommandBatch batch(device);
        if (r == finest) {
            leafMoments_.dispatch(batch, {level.groups, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(lod.splats.positions.rhi());
                cursor["shape"].setBinding(lod.splats.shape.rhi());
                cursor["sh"].setBinding(lod.splats.sh.rhi());
                cursor["starts"].setBinding(level.starts.rhi());
                cursor["moments"].setBinding(moments->rhi());
                rhi::ShaderCursor p = cursor["params"];
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
            rhi::ShaderCursor p = cursor["params"];
            p["groups"].setData(level.groups);
            p["keep"].setData(keep);
            p["shWords"].setData(cloud.shWords);
            p["stride"].setData(stride);
        });
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
        if (slot == previous + 1 && !runs.empty()) {
            runs.back().count += lod.chunkCount(c);
        } else {
            runs.push_back({static_cast<uint32_t>(slot) * lod.chunkSplats, lod.chunkCount(c)});
        }
        previous = slot;
    }
    return runs;
}

}   // namespace

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
        if (lod.levels.empty()) {
            return Error(ErrorCode::InvalidArgument, lod.splats.source + ": a cut needs a merged level");
        }
        Frame& frame = *frames_[k];
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
        if (frame.capacity < drawn || frame.cloud.restPerColour != lod.splats.restPerColour ||
            frame.cloud.shWords != lod.splats.shWords || !frame.cloud.positions.valid()) {
            const uint32_t capacity = std::max(drawn, frame.capacity + frame.capacity / 2);
            auto made = packed(device, std::max(capacity, 1u), lod.splats.restPerColour, lod.splats.shWords,
                               "cut.frame");
            if (!made) return std::move(made).error();
            frame.cloud = std::move(*made);
            frame.capacity = std::max(capacity, 1u);
        }
        frame.cloud.source = lod.splats.source;
        frame.cloud.count = drawn;
        frame.cloud.declared = lod.count;
        frame.cloud.bounds = lod.splats.bounds;
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
                    cursor["params"]["count"].setData(countOf(part));
                    cursor["params"]["base"].setData(base);
                    cursor["params"]["offset"].setData(merged ? 0u : runs[part - levels].offset);
                    cursor["params"]["shWords"].setData(lod.splats.shWords);
                });
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
    return d;
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
    auto out = packed(device, std::max(kept, 1u), lod.splats.restPerColour, lod.splats.shWords, "decimate.cloud");
    if (!out) return std::move(out).error();
    auto ranges = buffer(device, std::max(kept, 1u), 8, "decimate.ranges");
    if (!ranges) return std::move(ranges).error();
    out->source = lod.splats.source;
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
    auto mergedWeights = buffer(device, uint64_t{std::max(count, 1u)} * std::max(width, 1u), 4, "decimate.weights");
    if (!merged || !mergedWeights) return Error(ErrorCode::OutOfMemory, "decimate: cannot allocate an attribute");
    if (count > 0 && width > 0) {
        gpu::CommandBatch batch(device);
        mergeAttribute_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["ranges"].setBinding(kept.ranges.rhi());
            cursor["recordOf"].setBinding(kept.recordOf.rhi());
            cursor["splatPositions"].setBinding(lod.splats.positions.rhi());
            cursor["splatShape"].setBinding(lod.splats.shape.rhi());
            cursor["values"].setBinding(values.rhi());
            cursor["weights"].setBinding(weights != nullptr ? weights->rhi() : values.rhi());
            cursor["merged"].setBinding(merged->rhi());
            cursor["mergedWeights"].setBinding(mergedWeights->rhi());
            cursor["params"]["count"].setData(count);
            cursor["params"]["width"].setData(width);
            cursor["params"]["mode"].setData(static_cast<uint32_t>(how));
        });
        ATHENEA_TRY(batch.submit(true));
    }
    return std::pair{std::move(*merged), std::move(*mergedWeights)};
}

}   // namespace athenea::lod
