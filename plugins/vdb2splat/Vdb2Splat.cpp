// Copyright (c) 2026 jesus luque.
//
// A volume into volumetric gaussians, as an AOFX effect (proposal 103): the
// leaves of a sparse grid in, gaussians whose sum is its extinction out. The
// method and the layouts are at the head of vdb2splat.slang; this file is
// the declaration and the order the kernels run in.
//
// WHY A GRID ARRIVES AS A PICTURE
//
// An effect is handed images and gives one back. The host (athenea
// vdb2splat) lays the grid's leaves out as one -- 513 entries a leaf, its
// origin and then its voxels -- and sets `leaves` to how many there are.
// The gaussians leave the same way, four entries each, and the effect
// attaches how many it wrote. What crosses to the processor here is counts:
// the box the leaves span (to size the scratch), how many blocks a
// threshold keeps and how many voxels a residual leaves under (to bisect
// the two thresholds on), and the totals of a compaction.
#include <aofx/Effect.h>
#include <aofx/Entry.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "aofx_kernels_vdb2splat.h"

namespace {

/// Must match `V2sParams` in vdb2splat.slang exactly.
struct Vdb2SplatUniforms {
    uint32_t leafCount = 0;
    uint32_t leafEntries = 513;
    uint32_t capacity = 0;
    uint32_t unused0 = 0;

    int32_t cellLo[4] = {0, 0, 0, 0};
    int32_t cellExtent[4] = {1, 1, 1, 0};

    uint32_t maxLevel = 6;
    uint32_t nodeCount = 0;
    uint32_t level = 0;
    uint32_t mode = 0;

    uint32_t coarseBase[4] = {0, 0, 0, 0};
    uint32_t coarseDims[4][4] = {};

    float    tau = 0.0F;
    float    emissionWeight = 1.0F;
    float    threshold = 0.0F;
    float    maxStd = 21.0F;

    float    minStd = 0.5F;
    float    learningRate = 0.1F;
    uint32_t step = 1;
    float    pruneMass = 1e-6F;

    uint32_t hasTemperature = 0;
    uint32_t hasColour = 0;
    uint32_t listBase = 0;
    uint32_t flagCount = 0;

    uint32_t bornCount = 0;
    uint32_t freeBase = 0;
    uint32_t outCount = 0;
    uint32_t pad = 0;
};
static_assert(sizeof(Vdb2SplatUniforms) == 208, "must match V2sParams exactly");

constexpr uint32_t kLeafEntries = 513;
constexpr uint32_t kNodesPerLeaf = 73;
constexpr uint32_t kCounterCount = 8;
constexpr uint32_t kCounterErrMax = 9;
constexpr uint32_t kCounterTotal = 10;
constexpr uint32_t kCounterStats = 12;

const char* const kKernels[] = {"v2sClear",       "v2sBounds",     "v2sLeafMap",          "v2sLeafNodes",
                                "v2sCoarseMoments", "v2sCoarseError", "v2sCoarseErrorGather", "v2sCutFlags",
                                "v2sCompact",     "v2sInitEmit",   "v2sReset",            "v2sPrepare",
                                "v2sForward",     "v2sLoss",       "v2sBackward",         "v2sAdam",
                                "v2sAttributes",  "v2sEm",         "v2sPrune",      "v2sUnder",            "v2sSpawn",
                                "v2sAliveFlags",  "v2sWrite"};

float floatOf(uint32_t bits) {
    float f = 0.0F;
    std::memcpy(&f, &bits, 4);
    return f;
}

class Vdb2Splat final : public aofx::Effect {
public:
    void describe(aofx::EffectDesc& into) override {
        into.identifier = "rt.sparrow.aofx.vdb2splat";
        into.label = "Volume to Gaussians";
        into.grouping = "Convert";
        into.description =
            "Fits a sparse volume's extinction with a sum of 3D gaussians, mu_t = sum sigma_i G_i, by "
            "moments, an error-driven cut through blocks of 2 to 128 voxels, and Adam refinement with "
            "densification; temperature and Cd are fitted after, as density-weighted means.";
        aofx::ClipDesc grid;
        grid.name = "Grid";
        grid.label = "Grid leaves";
        into.inputs.push_back(grid);

        const auto number = [&into](const char* name, const char* label, const char* hint, aofx::ParamType type,
                                    double value, double low, double high) {
            aofx::ParamDesc d;
            d.name = name;
            d.label = label;
            d.hint = hint;
            d.type = type;
            d.defaults = {value};
            d.hardMin = {low};
            d.hardMax = {high};
            into.params.push_back(d);
        };
        using T = aofx::ParamType;
        number("leaves", "Leaves", "How many leaves the Grid clip holds (the host sets it).", T::Integer, 0, 0, 1e9);
        number("target", "Gaussians", "The budget: how many gaussians the fit ends with, at most.", T::Integer, 20000,
               1, 1e8);
        number("emIterations", "EM steps", "Steps that move each gaussian to the moments of the density it is "
               "responsible for, before Adam's.", T::Integer, 40, 0, 100000);
        number("iterations", "Iterations", "Adam steps on the geometry, after the EM steps.", T::Integer, 100, 0,
               100000);
        number("attributeIterations", "Attribute steps", "Jacobi steps on temperature and Cd.", T::Integer, 20, 0,
               10000);
        number("maxBlock", "Largest block", "The largest block one gaussian starts from, in voxels a side (8 to "
               "128, a power of two); a third of it is the largest deviation a gaussian may take.",
               T::Integer, 64, 8, 128);
        number("emissionWeight", "Emission weight",
               "How much the temperature's spread in a block counts beside the density's error, in "
               "density units (the temperature scaled to the density's range).",
               T::Double, 1.0, 0.0, 1e6);
        number("initFraction", "Initial share", "The share of the budget the cut starts with; the rest is born "
               "where the residual is largest.", T::Double, 0.6, 0.01, 1.0);
        number("densifyEvery", "Densify every", "Iterations between densify-and-prune rounds (none after 70%).",
               T::Integer, 20, 1, 100000);
        number("learningRate", "Learning rate", "Adam's step: log units for sigma and the factor's diagonal, "
               "deviations for the centre. Decays tenfold over the run.", T::Double, 0.1, 0.0, 1.0);
        number("minStd", "Smallest deviation", "In voxels: below about half a voxel a gaussian sampled at the voxel centres no longer holds the mass its integral says, which the EM steps assume.", T::Double, 0.5, 0.05, 10.0);
        number("pruneMass", "Prune below", "A gaussian whose mass is under this share of the total goes.", T::Double,
               1e-6, 0.0, 1.0);
        number("hasTemperature", "Temperature", "The Grid clip carries a temperature (host).", T::Boolean, 0, 0, 1);
        number("hasColour", "Cd", "The Grid clip carries a Cd (host).", T::Boolean, 0, 0, 1);
    }

    [[nodiscard]] std::vector<aofx::KernelDesc> kernels() const override {
        std::vector<aofx::KernelDesc> out;
        for (const char* name : kKernels) {
            out.push_back(aofx::KernelDesc{name, name, k_vdb2splat, k_vdb2splatBytes});
        }
        return out;
    }

    bool process(const aofx::RenderRequest& request) override {
        Run run{request};
        return run.go();
    }

private:
    struct Run {
        const aofx::RenderRequest& request;
        aofx::Gpu*                 gpu = nullptr;
        Vdb2SplatUniforms          u;
        std::vector<aofx::Buffer>  buffers;
        aofx::Buffer               counters;
        std::vector<aofx::KernelId> ids;

        bool fail(const char* why) {
            request.complaint = why;
            return false;
        }

        /// A scratch of `quads` float4s, linear (rows of 4096).
        aofx::Buffer quads(uint64_t n) {
            n = std::max<uint64_t>(n, 1);
            const uint64_t width = std::min<uint64_t>(n, 4096);
            const uint64_t height = (n + width - 1) / width;
            return gpu->scratch(static_cast<int>(width), static_cast<int>(height));
        }

        bool dispatch(const char* name, uint64_t threads) {
            if (threads == 0) return true;
            const auto at = static_cast<size_t>(std::find_if(std::begin(kKernels), std::end(kKernels),
                                                             [&](const char* k) { return std::strcmp(k, name) == 0; }) -
                                                std::begin(kKernels));
            return gpu->run(ids[at], aofx::Grid{static_cast<uint32_t>(threads), 1, 1}, buffers, &u, sizeof(u));
        }

        bool readCounters(uint32_t* into, size_t count) {
            return gpu->read(counters, into, count * 4);
        }

        /// A pass that counts into kCounterCount, cleared first; the count.
        bool counted(const char* name, uint64_t threads, uint32_t& count) {
            const uint32_t mode = u.mode;
            u.mode = 1;
            if (!dispatch("v2sClear", 1)) return false;
            u.mode = mode;
            if (!dispatch(name, threads)) return false;
            uint32_t c[16] = {};
            if (!readCounters(c, 16)) return false;
            count = c[kCounterCount];
            return true;
        }

        /// The flagged among `flags[0, n)` into `list` from `base`; their count.
        bool compact(uint32_t n, uint32_t base, uint32_t& total) {
            u.flagCount = n;
            u.listBase = base;
            if (!dispatch("v2sCompact", 256)) return false;
            uint32_t c[16] = {};
            if (!readCounters(c, 16)) return false;
            total = c[kCounterTotal];
            return true;
        }

        bool lossPass(uint32_t leafThreads, float stats[4]) {
            u.mode = 1;
            if (!dispatch("v2sPrepare", u.capacity) || !dispatch("v2sForward", leafThreads) ||
                !dispatch("v2sLoss", 256)) {
                return false;
            }
            u.mode = 0;
            uint32_t c[16] = {};
            if (!readCounters(c, 16)) return false;
            for (int k = 0; k < 4; ++k) stats[k] = floatOf(c[kCounterStats + k]);
            return true;
        }

        bool go() {
            gpu = request.gpu;
            const aofx::InputPlane*  grid = request.input("Grid");
            const aofx::OutputPlane* target = request.output("Color");
            if (grid == nullptr || target == nullptr || !grid->buffer.isValid() || !target->buffer.isValid() ||
                gpu == nullptr) {
                return fail("vdb2splat needs its Grid input and somewhere to write");
            }
            if (grid->buffer.stride != grid->buffer.width || target->buffer.stride != target->buffer.width) {
                return fail("vdb2splat reads and writes pictures whose rows are not padded");
            }
            u.leafCount = static_cast<uint32_t>(std::max(request.number("leaves", 0.0), 0.0));
            if (u.leafCount == 0) return fail("vdb2splat was given no leaves: the host sets 'leaves'");
            if (static_cast<uint64_t>(grid->buffer.width) * grid->buffer.height < uint64_t{u.leafCount} * kLeafEntries) {
                return fail("vdb2splat: the Grid clip is smaller than its leaves");
            }
            const uint64_t room = static_cast<uint64_t>(target->buffer.width) * target->buffer.height / 4;
            u.capacity = static_cast<uint32_t>(
                std::min<uint64_t>(room, static_cast<uint64_t>(std::max(request.number("target", 20000.0), 1.0))));
            if (u.capacity == 0) return fail("vdb2splat was given no room to write a single gaussian");
            const uint32_t maxBlock = static_cast<uint32_t>(std::clamp(request.number("maxBlock", 64.0), 8.0, 128.0));
            u.maxLevel = 3;
            while ((2u << u.maxLevel) <= maxBlock && u.maxLevel < 7) ++u.maxLevel;
            u.maxStd = static_cast<float>(maxBlock) / 3.0F;
            u.minStd = static_cast<float>(std::clamp(request.number("minStd", 0.5), 0.05, 10.0));
            u.emissionWeight = static_cast<float>(std::max(request.number("emissionWeight", 1.0), 0.0));
            u.pruneMass = static_cast<float>(std::clamp(request.number("pruneMass", 1e-6), 0.0, 1.0));
            u.hasTemperature = request.number("hasTemperature", 0.0) >= 0.5 ? 1u : 0u;
            u.hasColour = request.number("hasColour", 0.0) >= 0.5 ? 1u : 0u;
            const uint32_t adamSteps = static_cast<uint32_t>(std::max(request.number("iterations", 100.0), 0.0));
            const uint32_t emSteps = static_cast<uint32_t>(std::max(request.number("emIterations", 40.0), 0.0));
            const uint32_t iterations = emSteps + adamSteps;
            const uint32_t attributeSteps =
                static_cast<uint32_t>(std::max(request.number("attributeIterations", 20.0), 0.0));
            const uint32_t densifyEvery = static_cast<uint32_t>(std::max(request.number("densifyEvery", 20.0), 1.0));
            const double initFraction = std::clamp(request.number("initFraction", 0.6), 0.01, 1.0);
            const float learningRate = static_cast<float>(std::clamp(request.number("learningRate", 0.1), 0.0, 1.0));

            for (const char* name : kKernels) {
                const aofx::KernelId id = gpu->load(name);
                if (id == aofx::kInvalidKernel) return false;
                ids.push_back(id);
            }
            counters = quads(4);
            if (!counters.isValid()) return fail("vdb2splat could not take its counters");
            // Bound in the kernels' order; what is not there yet is the counters.
            buffers = {grid->buffer, target->buffer, counters, counters, counters, counters, counters,
                       counters,     counters,       counters, counters, counters, counters};

            const uint32_t leafThreads = u.leafCount * 256;
            u.mode = 0;
            if (!dispatch("v2sClear", 1) || !dispatch("v2sBounds", u.leafCount)) return false;
            uint32_t c[16] = {};
            if (!readCounters(c, 16)) return false;
            for (int k = 0; k < 3; ++k) {
                u.cellLo[k] = static_cast<int32_t>(c[k]) - (1 << 30);
                u.cellExtent[k] = static_cast<int32_t>(c[3 + k]) - static_cast<int32_t>(c[k]);
            }
            const uint64_t cells = uint64_t(u.cellExtent[0]) * uint64_t(u.cellExtent[1]) * uint64_t(u.cellExtent[2]);
            if (cells == 0 || cells > (uint64_t{1} << 31)) return fail("vdb2splat: the leaves' box is empty or vast");
            // The coarse levels: each level's blocks over the box.
            uint32_t coarseTotal = 0;
            for (uint32_t level = 4; level <= u.maxLevel; ++level) {
                const int32_t span = 1 << (level - 3);
                u.coarseBase[level - 4] = coarseTotal;
                uint32_t blocks = 1;
                for (int k = 0; k < 3; ++k) {
                    u.coarseDims[level - 4][k] = static_cast<uint32_t>((u.cellExtent[k] + span - 1) / span);
                    blocks *= u.coarseDims[level - 4][k];
                }
                coarseTotal += blocks;
            }
            u.nodeCount = u.leafCount * kNodesPerLeaf + coarseTotal;
            const uint64_t voxels = uint64_t{u.leafCount} * 512;
            const uint64_t flagCount = std::max<uint64_t>({u.nodeCount, u.capacity, voxels});
            const uint32_t region = std::max(u.nodeCount, u.capacity);
            u.freeBase = region;

            const aofx::Buffer leafMap = quads((cells + 3) / 4);
            const aofx::Buffer nodes = quads(uint64_t{u.leafCount} * kNodesPerLeaf * 4);
            const aofx::Buffer coarse = quads(uint64_t{std::max(coarseTotal, 1u)} * 4);
            const aofx::Buffer partial = quads(uint64_t{u.leafCount} * 2);
            const aofx::Buffer flags = quads((flagCount + 3) / 4);
            const aofx::Buffer state = quads(uint64_t{u.capacity} * 4);
            const aofx::Buffer adam = quads(uint64_t{u.capacity} * 8);
            const aofx::Buffer grad = quads(uint64_t{u.capacity} * 4);
            const aofx::Buffer model = quads(voxels * 2);
            const aofx::Buffer list = quads((uint64_t{region} + u.capacity + 3) / 4);
            for (const aofx::Buffer* b : {&leafMap, &nodes, &coarse, &partial, &flags, &state, &adam, &grad, &model,
                                          &list}) {
                if (!b->isValid()) return fail("vdb2splat: no room on the device for its working buffers");
            }
            buffers = {grid->buffer, target->buffer, counters, leafMap, nodes, coarse, partial,
                       flags,        state,          adam,     grad,    model, list};

            // 1, 2: the blocks, their moments and their errors.
            u.mode = 0;
            if (!dispatch("v2sLeafMap", cells)) return false;
            u.mode = 1;
            if (!dispatch("v2sLeafMap", u.leafCount)) return false;
            u.mode = 0;
            if (!dispatch("v2sLeafNodes", leafThreads)) return false;
            for (uint32_t level = 4; level <= u.maxLevel; ++level) {
                u.level = level;
                const uint32_t* d = u.coarseDims[level - 4];
                if (!dispatch("v2sCoarseMoments", uint64_t{d[0]} * d[1] * d[2])) return false;
            }
            if (u.maxLevel >= 4) {
                if (!dispatch("v2sCoarseError", leafThreads)) return false;
                for (uint32_t level = 4; level <= u.maxLevel; ++level) {
                    u.level = level;
                    const uint32_t* d = u.coarseDims[level - 4];
                    if (!dispatch("v2sCoarseErrorGather", uint64_t{d[0]} * d[1] * d[2])) return false;
                }
            }
            if (!readCounters(c, 16)) return false;
            const float errMax = floatOf(c[kCounterErrMax]);

            // 3: the cut, tau bisected on its count.
            const uint32_t initTarget =
                std::max<uint32_t>(1, static_cast<uint32_t>(std::lround(u.capacity * initFraction)));
            double lo = std::max(static_cast<double>(errMax), 1e-30) * 1e-14;
            double hi = std::max(static_cast<double>(errMax), 1e-30) * 1.0001;
            uint32_t count = 0;
            u.tau = static_cast<float>(hi);
            if (!counted("v2sCutFlags", u.nodeCount, count)) return false;
            const uint32_t minimalCut = count;
            if (minimalCut > u.capacity) {
                // Fewer than the coarsest blocks would leave whole blocks of
                // the volume without a gaussian: refused, not cut short.
                request.complaint = "vdb2splat: the coarsest blocks need " + std::to_string(minimalCut) +
                                    " gaussians, more than the budget of " + std::to_string(u.capacity) +
                                    "; raise maxBlock or the budget";
                return false;
            }
            if (count <= initTarget) {
                for (int i = 0; i < 40; ++i) {
                    const double mid = std::sqrt(lo * hi);
                    u.tau = static_cast<float>(mid);
                    if (!counted("v2sCutFlags", u.nodeCount, count)) return false;
                    if (count > initTarget) lo = mid;
                    else hi = mid;
                }
            }
            u.tau = static_cast<float>(hi);
            if (!dispatch("v2sCutFlags", u.nodeCount)) return false;
            uint32_t cut = 0;
            if (!compact(u.nodeCount, 0, cut)) return false;
            u.bornCount = std::min(cut, u.capacity);
            if (!dispatch("v2sReset", u.capacity) || !dispatch("v2sInitEmit", u.bornCount)) return false;
            const uint32_t initial = u.bornCount;

            // 4, 5: refinement, densify and prune.
            const uint32_t lastDensify = static_cast<uint32_t>(iterations * 7 / 10);
            uint32_t roundsLeft = 0;
            for (uint32_t it = 0; it < lastDensify; ++it) {
                if ((it + 1) % densifyEvery == 0) ++roundsLeft;
            }
            float stats[4] = {};
            for (uint32_t it = 0; it < iterations; ++it) {
                u.mode = 0;
                if (!dispatch("v2sPrepare", u.capacity) || !dispatch("v2sForward", leafThreads)) return false;
                if (it < emSteps) {
                    // EM first: robust from a rough start, and a fixed point
                    // at a mixture that is the grid.
                    u.mode = 2;
                    if (!dispatch("v2sBackward", uint64_t{u.capacity} * 64) || !dispatch("v2sEm", u.capacity)) {
                        return false;
                    }
                    u.mode = 0;
                } else {
                    const uint32_t a = it - emSteps;
                    u.learningRate =
                        learningRate * static_cast<float>(std::pow(0.01, static_cast<double>(a) / adamSteps));
                    u.step = a + 1;
                    if (!dispatch("v2sBackward", uint64_t{u.capacity} * 64) || !dispatch("v2sAdam", u.capacity)) {
                        return false;
                    }
                }
                if ((it + 1) % densifyEvery != 0 || it >= lastDensify) continue;
                if (!lossPass(leafThreads, stats)) return false;
                if (!dispatch("v2sPrune", u.capacity)) return false;
                uint32_t free = 0;
                if (!compact(u.capacity, u.freeBase, free)) return false;
                const uint32_t want = roundsLeft > 0 ? (free + roundsLeft - 1) / roundsLeft : free;
                roundsLeft = roundsLeft > 0 ? roundsLeft - 1 : 0;
                if (want == 0) continue;
                // The threshold the residual must pass for `want` voxels at most.
                const float rhoMax = floatOf(c[6]);
                double tLo = 0.0, tHi = std::max(static_cast<double>(rhoMax), 1e-30);
                for (int i = 0; i < 24; ++i) {
                    const double mid = 0.5 * (tLo + tHi);
                    u.threshold = static_cast<float>(mid);
                    if (!counted("v2sUnder", voxels, count)) return false;
                    if (count > want) tLo = mid;
                    else tHi = mid;
                }
                u.threshold = static_cast<float>(tHi);
                u.mode = 1;
                if (!dispatch("v2sUnder", voxels)) return false;
                u.mode = 0;
                uint32_t chosen = 0;
                if (!compact(static_cast<uint32_t>(voxels), 0, chosen)) return false;
                u.bornCount = std::min({chosen, free, want});
                if (!dispatch("v2sSpawn", u.bornCount)) return false;
            }

            // 6: the attributes, the geometry fixed.
            if (u.hasTemperature != 0 || u.hasColour != 0) {
                for (uint32_t it = 0; it < attributeSteps; ++it) {
                    u.mode = 0;
                    if (!dispatch("v2sPrepare", u.capacity) || !dispatch("v2sForward", leafThreads)) return false;
                    u.mode = 1;
                    if (!dispatch("v2sBackward", uint64_t{u.capacity} * 64)) return false;
                    u.mode = 0;
                    if (!dispatch("v2sAttributes", u.capacity)) return false;
                }
            }

            // The output: the living, in slot order.
            if (!lossPass(leafThreads, stats)) return false;
            if (!dispatch("v2sAliveFlags", u.capacity)) return false;
            uint32_t written = 0;
            if (!compact(u.capacity, 0, written)) return false;
            u.outCount = written;
            u.mode = 0;
            if (!dispatch("v2sWrite", written)) return false;
            u.mode = 1;
            if (!dispatch("v2sWrite", written)) return false;
            uint32_t done[16] = {};
            if (!readCounters(done, 16)) return false;   // and waits for the last write
            // What went out: written, the budget, leaves, the fit's own
            // relative L2 error over the active voxels, the cut it started
            // from, the iterations, the coarsest level, and the smallest cut
            // the blocks allow.
            request.attach("gaussians", {static_cast<float>(written), static_cast<float>(u.capacity),
                                         static_cast<float>(u.leafCount), stats[3], static_cast<float>(initial),
                                         static_cast<float>(iterations), static_cast<float>(u.maxLevel),
                                         static_cast<float>(minimalCut)});
            return true;
        }
    };
};

}   // namespace

AOFX_EXPORT_EFFECTS(Vdb2Splat)
