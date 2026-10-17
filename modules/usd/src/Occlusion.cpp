// Copyright (c) 2026 jesus luque.
//
// WHERE A TREE'S NODES CANNOT BE SEEN FROM (occlusion_mask.slang): each
// node's mask and the codebook a player keeps them in, for one tree
// (`decimate --occlusion-mask`) or several sharing one codebook
// (`--occlusion-group`: an asset's pieces, whose occluders are the same car).

#include "athenea/usd/Export.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>

namespace athenea::usd {

namespace {

constexpr uint32_t kWords = 8;   // 16 x 16 cells

bool ends(const std::string& name, const std::string& tail) {
    return name.size() >= tail.size() && name.compare(name.size() - tail.size(), tail.size(), tail) == 0;
}

}  // namespace

OcclusionInput occlusionUpOf(const std::filesystem::path& path, OcclusionInput in) {
    const pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(path.string());
    const bool yUp = stage && pxr::UsdGeomGetStageUpAxis(stage) == pxr::UsdGeomTokens->y;
    in.up[0] = 0.0F;
    in.up[1] = yUp ? 1.0F : 0.0F;
    in.up[2] = yUp ? 0.0F : 1.0F;
    return in;
}

Result<OcclusionBook> occlusionBook(gpu::ShaderLibrary& library, const OcclusionInput& in, uint32_t clusters) {
    gpu::Device& device = library.device();
    const uint32_t records = static_cast<uint32_t>(in.parent.size());
    if (records == 0 || in.leaves.size() != records || clusters == 0) {
        return Error(ErrorCode::InvalidArgument, "occlusion: a tree and a codebook size");
    }
    const bool hasBits = in.bits.size() == size_t{records} * kWords;
    const bool hasGlass = in.glass.size() == records;
    const auto upload = [&](const std::vector<uint32_t>& words, size_t count, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<size_t>(count, 1) * 4;
        desc.elementBytes = 4;
        desc.label = label;
        return gpu::Buffer::create(device, desc, words.size() >= count && count > 0 ? words.data() : nullptr);
    };
    const std::vector<uint32_t> none;
    const auto zeros = [&](size_t count, const char* label) {
        return upload(std::vector<uint32_t>(std::max<size_t>(count, 1), 0u), count, label);
    };
    auto bits = hasBits ? upload(in.bits, in.bits.size(), "occlusion.bits") : zeros(kWords, "occlusion.bits");
    auto glass = hasGlass ? upload(in.glass, records, "occlusion.glass") : zeros(1, "occlusion.glass");
    auto parents = upload(in.parent, records, "occlusion.parent");
    auto leaves = upload(in.leaves, records, "occlusion.leaves");
    auto mask = zeros(size_t{records} * kWords, "occlusion.mask");
    auto moved = zeros(1, "occlusion.changed");
    auto centroids = zeros(size_t{clusters} * kWords, "occlusion.book");
    auto gathered = zeros(size_t{clusters} * kWords, "occlusion.next");
    auto memberCount = zeros(clusters, "occlusion.members");
    auto assigned = zeros(records, "occlusion.assigned");
    auto lostCells = zeros(3, "occlusion.lost");
    auto worthBins = zeros(64, "occlusion.worth");
    if (!bits || !glass || !parents || !leaves || !mask || !moved || !centroids || !gathered || !memberCount ||
        !assigned || !lostCells || !worthBins) {
        return Error(ErrorCode::OutOfMemory, "occlusion: the masks");
    }
    const char* names[] = {"occlusionLeaf",   "occlusionUp",   "occlusionSeed", "occlusionAssign",
                           "occlusionUpdate", "occlusionLost", "occlusionWorth"};
    std::vector<gpu::ComputeKernel> k;
    for (const char* name : names) {
        auto made = gpu::ComputeKernel::create(library, "athenea/lod/occlusion_mask", name);
        if (!made) return std::move(made).error();
        k.push_back(std::move(*made));
    }
    uint32_t subset = 0;
    uint32_t minBin = 0;
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["bits"].setBinding(bits->rhi());
        cursor["transmission"].setBinding(glass->rhi());
        cursor["parent"].setBinding(parents->rhi());
        cursor["leaves"].setBinding(leaves->rhi());
        cursor["mask"].setBinding(mask->rhi());
        cursor["changed"].setBinding(moved->rhi());
        cursor["centroids"].setBinding(centroids->rhi());
        cursor["next"].setBinding(gathered->rhi());
        cursor["members"].setBinding(memberCount->rhi());
        cursor["assigned"].setBinding(assigned->rhi());
        cursor["lost"].setBinding(lostCells->rhi());
        cursor["worthBins"].setBinding(worthBins->rhi());
        cursor["params"]["minBin"].setData(minBin);
        // An eye looked for down to ~15 degrees under the horizon.
        const float up[4] = {in.up[0], in.up[1], in.up[2], -0.25F};
        cursor["params"]["up"].setData(up, 16);
        cursor["params"]["count"].setData(records);
        cursor["params"]["words"].setData(kWords);
        cursor["params"]["side"].setData(uint32_t{16});
        cursor["params"]["clusters"].setData(clusters);
        cursor["params"]["hasBits"].setData(uint32_t{hasBits ? 1u : 0u});
        cursor["params"]["hasGlass"].setData(uint32_t{hasGlass ? 1u : 0u});
        cursor["params"]["subset"].setData(subset);
    };
    {
        gpu::CommandBatch batch(device);
        k[0].dispatch(batch, {records, 1, 1}, bind);
        ATHENEA_TRY(batch.submit(true));
    }
    // The climb, eight levels a submit, until no mask opens.
    OcclusionBook out;
    for (uint32_t open = 1; open != 0 && out.levels < 512; out.levels += 8) {
        const uint32_t zero = 0;
        ATHENEA_TRY(moved->write(device, 0, 4, &zero));
        gpu::CommandBatch batch(device);
        for (int r = 0; r < 8; ++r) {
            k[1].dispatch(batch, {records, 1, 1}, bind);
            batch.markDirty();
        }
        ATHENEA_TRY(batch.submit(true));
        ATHENEA_TRY(moved->read(device, 0, 4, &open));
    }
    // THE NODES WORTH LEAST, under kSpared of the whole between them, take
    // the all-open entry and stay out of the fit: the threshold picked from a
    // histogram of log2(worth), each bin counted at its middle.
    {
        gpu::CommandBatch batch(device);
        k[6].dispatch(batch, {records, 1, 1}, bind);
        ATHENEA_TRY(batch.submit(true));
    }
    uint32_t bins[64] = {};
    ATHENEA_TRY(worthBins->read(device, 0, sizeof(bins), bins));
    constexpr double kSpared = 0.02;
    double total = 0.0;
    for (uint32_t b = 1; b < 64; ++b) {
        total += bins[b] * std::exp2(double(b) - 0.5);
    }
    double below = 0.0;
    for (minBin = 1; minBin < 64; ++minBin) {
        const double here = bins[minBin] * std::exp2(double(minBin) - 0.5);
        if (below + here > kSpared * total) {
            break;
        }
        below += here;
    }
    for (uint32_t b = 0; b < minBin; ++b) {
        out.spared += bins[b];
    }
    {
        gpu::CommandBatch batch(device);
        k[2].dispatch(batch, {clusters, 1, 1}, bind);
        batch.markDirty();
        for (uint32_t round = 0; round < 8; ++round) {
            subset = round == 0 ? 0u : 1u;
            k[3].dispatch(batch, {records, 1, 1}, bind);
            batch.markDirty();
            k[4].dispatch(batch, {clusters, 1, 1}, bind);
            batch.markDirty();
        }
        k[5].dispatch(batch, {records, 1, 1}, bind);
        ATHENEA_TRY(batch.submit(true));
    }
    uint32_t lost[3] = {0, 0, 0};
    ATHENEA_TRY(lostCells->read(device, 0, 12, lost));
    out.reopened = lost[0];
    out.closed = lost[1];
    out.seenClosed = lost[2];
    auto index = assigned->readAll<uint32_t>(device);
    auto book = centroids->readAll<uint32_t>(device);
    if (!index || !book) return Error(ErrorCode::IoFailure, "occlusion: the masks back");
    index->resize(records);
    book->resize(size_t{clusters} * kWords);
    out.index = std::move(*index);
    out.book = std::move(*book);
    return out;
}

std::vector<GaussianArray> occlusionArrays(const OcclusionBook& book) {
    GaussianArray idx;
    idx.name = "primvars:athenea:splat:occlusionIndex";
    idx.typeName = "int[]";
    idx.integer = true;
    idx.interpolation = "vertex";
    idx.values.push_back(book.index);
    GaussianArray cb;
    cb.name = "primvars:athenea:splat:occlusionCodebook";
    cb.typeName = "int[]";
    cb.integer = true;
    cb.interpolation = "constant";
    cb.values.push_back(book.book);
    return {std::move(idx), std::move(cb)};
}

Result<OcclusionInput> occlusionInputOf(const std::vector<GaussianArray>& arrays, uint32_t records) {
    OcclusionInput in;
    for (const GaussianArray& a : arrays) {
        if (a.values.empty()) {
            continue;
        }
        if (ends(a.name, "athenea:splat:lodParent")) in.parent = a.values[0];
        if (ends(a.name, "athenea:splat:lodSize")) in.leaves = a.values[0];
        if (ends(a.name, "athenea:splat:transmission") && a.width == 1 && !a.integer) in.glass = a.values[0];
        if (ends(a.name, "athenea:splat:cryptoObject") && a.width == 1 && a.integer) in.part = a.values[0];
        if (ends(a.name, "athenea:splat:shadowBits")) {
            if (a.width != kWords) {
                return Error(ErrorCode::InvalidArgument,
                             "occlusion: reads 16 x 16 shadow bits (eight ints a gaussian), not " + a.name);
            }
            in.bits = a.values[0];
        }
    }
    if (in.parent.size() != records || in.leaves.size() != records) {
        return Error(ErrorCode::InvalidArgument, "occlusion: needs a merge tree (lodParent, lodSize)");
    }
    return in;
}

Result<void> occlusionMasksShared(gpu::ShaderLibrary& library, const std::vector<std::filesystem::path>& inputs,
                                  const std::vector<std::filesystem::path>& outputs, uint32_t clusters, bool byPart) {
    if (inputs.empty() || inputs.size() != outputs.size()) {
        return Error(ErrorCode::InvalidArgument, "occlusion: as many outputs as trees");
    }
    // EACH UNIT ITS SHARE BY NEED (research): fitted alone at the whole
    // size, a unit uses as many entries as its masks are varied -- a
    // grille's 2773 on a third of a fender's nodes -- while one fit over all
    // handed entries out by how many nodes each brought (the grille got 363).
    // So each unit is fitted once alone, the codebook shared out in
    // proportion to the entries each used, and each fitted again within its
    // share; its indices then point past the shares before. A unit is a tree
    // or, `byPart`, one part of a tree: a part's merges never leave it, so its
    // nodes are a tree of their own once renumbered (the whole Corvette's
    // groups of parts gave a hidden panel 90 entries for 47k culled nodes).
    struct Unit {
        size_t                t = 0;
        std::vector<uint32_t> nodes;   // its nodes in the tree's order
        OcclusionInput        in;
        uint32_t              used = 0;
        uint32_t              share = 0;
    };
    std::vector<std::string> where(inputs.size());
    std::vector<uint32_t> counts(inputs.size(), 0u);
    std::vector<Unit> units;
    for (size_t t = 0; t < inputs.size(); ++t) {
        auto raw = readParticleFieldRecords(inputs[t], {}, &where[t]);
        if (!raw) return std::move(raw).error();
        auto arrays = readGaussianArrays(inputs[t], where[t], raw->count);
        if (!arrays) return std::move(arrays).error();
        auto in = occlusionInputOf(*arrays, raw->count);
        if (!in) return std::move(in).error();
        OcclusionInput tree = occlusionUpOf(inputs[t], std::move(*in));
        counts[t] = raw->count;
        if (!byPart || tree.part.size() != raw->count) {
            Unit u;
            u.t = t;
            u.nodes.resize(raw->count);
            for (uint32_t g = 0; g < raw->count; ++g) u.nodes[g] = g;
            u.in = std::move(tree);
            units.push_back(std::move(u));
            continue;
        }
        // Its parts, each renumbered: bookkeeping over the tree's indices.
        std::map<uint32_t, std::vector<uint32_t>> byId;
        for (uint32_t g = 0; g < raw->count; ++g) byId[tree.part[g]].push_back(g);
        std::vector<uint32_t> local(raw->count, 0u);
        for (auto& [id, nodes] : byId) {
            Unit u;
            u.t = t;
            for (uint32_t k = 0; k < nodes.size(); ++k) local[nodes[k]] = k;
            const bool bits = tree.bits.size() == size_t{raw->count} * kWords;
            const bool glass = tree.glass.size() == raw->count;
            for (uint32_t g : nodes) {
                const uint32_t up = tree.parent[g];
                u.in.parent.push_back(up < raw->count && tree.part[up] == id ? local[up] : local[g]);
                u.in.leaves.push_back(tree.leaves[g]);
                if (bits) u.in.bits.insert(u.in.bits.end(), tree.bits.begin() + size_t{g} * kWords,
                                           tree.bits.begin() + size_t{g + 1} * kWords);
                if (glass) u.in.glass.push_back(tree.glass[g]);
            }
            std::copy(std::begin(tree.up), std::end(tree.up), std::begin(u.in.up));
            u.nodes = std::move(nodes);
            units.push_back(std::move(u));
        }
    }
    for (Unit& u : units) {
        auto alone = occlusionBook(library, u.in, clusters);
        if (!alone) return std::move(alone).error();
        std::vector<uint8_t> seen(clusters, 0u);
        for (uint32_t e : alone->index) {
            if (e < clusters && !seen[e]) {
                seen[e] = 1u;
                ++u.used;
            }
        }
    }
    // Its need: the entries it uses. Weighed by how closed its nodes are
    // where an eye looks from, the door (closed, culling little) kept its
    // share and the fender (culling 26 times more) lost half (T41): how
    // closed is not how much is culled, which takes views to know.
    double need = 0.0;
    for (const Unit& u : units) need += u.used;
    const uint32_t least = byPart ? 2u : 16u;   // entry 0 and one: a unit that culls nothing still has a place
    const uint64_t floor = uint64_t{least} * units.size();
    const uint32_t spare = clusters > floor ? clusters - uint32_t(floor) : 0u;
    for (Unit& u : units) {
        u.share = least + (need > 0.0 ? uint32_t(double(spare) * u.used / need) : spare / uint32_t(units.size()));
    }
    OcclusionBook book;
    std::vector<std::vector<uint32_t>> indices(inputs.size());
    for (size_t t = 0; t < inputs.size(); ++t) indices[t].assign(counts[t], 0u);
    for (const Unit& u : units) {
        auto one = occlusionBook(library, u.in, u.share);
        if (!one) return std::move(one).error();
        const uint32_t base = static_cast<uint32_t>(book.book.size() / kWords);
        for (size_t k = 0; k < u.nodes.size(); ++k) indices[u.t][u.nodes[k]] = one->index[k] + base;
        book.book.insert(book.book.end(), one->book.begin(), one->book.end());
        book.closed += one->closed;
        book.reopened += one->reopened;
        if (!byPart) {
            std::printf("occlusion: %s uses %u entries alone, %u in the shared codebook; reopens %.1f%% of its closed "
                        "cells\n", inputs[u.t].filename().string().c_str(), u.used, u.share,
                        one->closed > 0 ? 100.0 * one->reopened / one->closed : 0.0);
        }
    }
    std::printf("occlusion: %zu trees, %zu units, a codebook of %zu shared by need; it reopens %llu of their %llu "
                "closed cells (%.1f%%)\n", inputs.size(), units.size(), book.book.size() / kWords,
                static_cast<unsigned long long>(book.reopened), static_cast<unsigned long long>(book.closed),
                book.closed > 0 ? 100.0 * double(book.reopened) / double(book.closed) : 0.0);
    for (size_t t = 0; t < inputs.size(); ++t) {
        OcclusionBook one;
        one.index = std::move(indices[t]);
        one.book = book.book;
        const std::vector<float> keep(one.index.size(), std::numeric_limits<float>::quiet_NaN());
        // A tree written over itself (the output is the input): to a file
        // beside it, then renamed over it -- a close-up level's trees are
        // tens of GB together, and a copy of each would not fit beside them.
        std::error_code same;
        const bool inPlace = std::filesystem::equivalent(inputs[t], outputs[t], same);
        const std::filesystem::path to =
            inPlace ? std::filesystem::path(outputs[t]).concat(".occlusion.usdc") : outputs[t];
        ATHENEA_TRY(writeTreeOpacities(inputs[t], to, where[t], keep, "", "", occlusionArrays(one)));
        if (inPlace) {
            std::error_code moved;
            std::filesystem::rename(to, outputs[t], moved);
            if (moved) return Error(ErrorCode::IoFailure, "occlusion: cannot replace " + outputs[t].string());
        }
        std::printf("occlusion: wrote %s\n", outputs[t].string().c_str());
    }
    return ok();
}

}  // namespace athenea::usd
