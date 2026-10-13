// Copyright (c) 2026 jesus luque.
//
// A CLOUD DECIMATED, AND EVERYTHING IT CARRIED KEPT: the pipeline behind
// `athenea decimate`, here so that a test can run what the command runs.
#include "athenea/usd/Export.h"

#include <limits>
#include <map>
#include <set>
#include <sstream>
#include "athenea/core/Hash.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu/algo/RadixSort.h"
#include "athenea/lod/Lod.h"
#include "athenea/render/Camera.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

#include <pxr/base/gf/camera.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>

namespace athenea::usd {

Result<lod::DecimateStats> decimateStage(gpu::ShaderLibrary& library, const std::filesystem::path& input,
                                     const std::filesystem::path& output, const DecimateStageOptions& o) {
    gpu::Device& device = library.device();
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return std::move(loader).error();
    const std::string ext = std::filesystem::path(input).extension().string();
    const bool usd = ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz";
    Result<io::RawSplats> raw = Error(ErrorCode::InternalError, "unread");
    std::string where;
    std::vector<GaussianArray> arrays;
    if (usd) {
        bool moved = false;
        raw = readParticleFieldRecords(input, o.prim, &where, &moved);
        if (!raw) return std::move(raw).error();
        std::printf("decimate: %s, %u gaussians\n", where.c_str(), raw->count);
        auto read = readGaussianArrays(input, where, raw->count);
        if (!read) return std::move(read).error();
        arrays = std::move(*read);
        // A TREE BUILT AGAIN FROM A TREE'S LEAVES: the old tree's bookkeeping
        // (its links, pages, costs, coverage and occlusion) belongs to the
        // nodes it made and is made anew. Left in, every integer array would
        // become a merge key one to a gaussian, and nothing would merge.
        if (o.pairs && !o.treeOutput.empty()) {
            const auto old = [](const std::string& name) {
                for (const char* tail : {"athenea:splat:lodParent", "athenea:splat:lodStart", "athenea:splat:lodSize",
                                         "athenea:splat:lodPage", "athenea:splat:lodPageInfo",
                                         "athenea:splat:lodPageBounds", "athenea:splat:lodCost",
                                         "athenea:splat:lodSpill", "athenea:splat:coverageMass",
                                         "athenea:splat:occlusionIndex", "athenea:splat:occlusionCodebook"}) {
                    const std::string end(tail);
                    if (name.size() >= end.size() && name.compare(name.size() - end.size(), end.size(), end) == 0) {
                        return true;
                    }
                }
                return false;
            };
            const size_t before = arrays.size();
            std::erase_if(arrays, [&](const GaussianArray& a) { return old(a.name); });
            if (arrays.size() != before) {
                std::printf("decimate: %zu arrays of an earlier tree left out\n", before - arrays.size());
            }
        }
        if (o.treeReport) {
            // WHERE A TREE STOPPED MERGING: its roots counted a part
            // (cryptoObject), the parts named from the field's manifest --
            // bookkeeping over the tree's own arrays.
            const auto tail = [](const std::string& name, const std::string& end) {
                return name.size() >= end.size() && name.compare(name.size() - end.size(), end.size(), end) == 0;
            };
            const GaussianArray* up = nullptr;
            const GaussianArray* ids = nullptr;
            const GaussianArray* size = nullptr;
            for (const GaussianArray& a : arrays) {
                if (tail(a.name, "athenea:splat:lodParent")) up = &a;
                if (tail(a.name, "athenea:splat:lodSize")) size = &a;
                if (tail(a.name, "athenea:splat:cryptoObject")) ids = &a;
            }
            if (up == nullptr || ids == nullptr || size == nullptr) {
                return Error(ErrorCode::InvalidArgument, "decimate: --tree-report needs a tree with cryptoObject");
            }
            std::map<uint32_t, std::string> names;
            if (const pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(input.string())) {
                const pxr::UsdAttribute manifest =
                    stage->GetPrimAtPath(pxr::SdfPath(where)).GetAttribute(pxr::TfToken("primvars:athenea:splat:cryptoManifest"));
                std::string json;
                if (manifest && manifest.Get(&json)) {
                    for (const auto& [name, id] : core::parseCryptomatteManifest(json)) names[id] = name;
                }
            }
            std::map<uint32_t, std::pair<uint32_t, uint32_t>> perPart;   // roots, leaves
            for (uint32_t g = 0; g < raw->count; ++g) {
                auto& entry = perPart[ids->values[0][g]];
                if (up->values[0][g] == g || up->values[0][g] >= raw->count) ++entry.first;
                if (size->values[0][g] <= 1) ++entry.second;
            }
            std::vector<std::pair<uint32_t, uint32_t>> order;
            for (const auto& [id, counts] : perPart) order.emplace_back(counts.first, id);
            std::sort(order.rbegin(), order.rend());
            for (size_t k = 0; k < std::min<size_t>(order.size(), 15); ++k) {
                const uint32_t id = order[k].second;
                std::printf("decimate: %8u roots of %8u leaves  %s\n", perPart[id].first, perPart[id].second,
                            names.count(id) ? names[id].c_str() : std::to_string(id).c_str());
            }
            return lod::DecimateStats{};
        }
        if (o.objectGroups > 1) {
            // ONE GROUP OF AN ASSET'S PARTS (a whole car will not merge in one
            // pass: its tree's attributes outgrow the device): the gaussians
            // whose part (cryptoObject) falls in group `objectGroup` of
            // `objectGroups`, by its id. Bookkeeping -- which records stay --
            // before anything is uploaded; a part's merges never cross parts,
            // so its tree is what the whole asset's would hold for it.
            const GaussianArray* ids = nullptr;
            for (const GaussianArray& a : arrays) {
                if (a.integer && a.width == 1 && a.name.size() >= 12 &&
                    a.name.compare(a.name.size() - 12, 12, "cryptoObject") == 0) {
                    ids = &a;
                }
            }
            if (ids == nullptr || ids->values.empty()) {
                return Error(ErrorCode::InvalidArgument, "decimate: --object-groups needs primvars:athenea:splat:cryptoObject");
            }
            std::vector<uint32_t> keep;
            for (uint32_t g = 0; g < raw->count; ++g) {
                if (ids->values[0][g] % o.objectGroups == o.objectGroup) keep.push_back(g);
            }
            const uint32_t stride = raw->encoding.floatsPerRecord;
            std::vector<float> records;
            records.reserve(size_t{keep.size()} * stride);
            for (uint32_t g : keep) {
                records.insert(records.end(), raw->records.begin() + size_t{g} * stride,
                               raw->records.begin() + size_t{g + 1} * stride);
            }
            raw->records = std::move(records);
            for (GaussianArray& a : arrays) {
                for (std::vector<uint32_t>& values : a.values) {
                    std::vector<uint32_t> kept;
                    kept.reserve(size_t{keep.size()} * a.width);
                    for (uint32_t g : keep) {
                        kept.insert(kept.end(), values.begin() + size_t{g} * a.width,
                                    values.begin() + size_t{g + 1} * a.width);
                    }
                    values = std::move(kept);
                }
            }
            std::printf("decimate: group %u of %u by part: %zu of %u gaussians\n", o.objectGroup, o.objectGroups,
                        keep.size(), raw->count);
            raw->count = static_cast<uint32_t>(keep.size());
        }
        for (const GaussianArray& array : arrays) {
            std::printf("decimate: carries %s (%u a gaussian%s)\n", array.name.c_str(), array.width,
                        array.times.empty() ? "" : ", sampled in time");
        }
    } else {
        raw = scene::readSplatRecords(*loader, input, o.degree);
    }
    if (!raw) return std::move(raw).error();
    if (!o.transferBasis.empty()) {
        // THE TRANSFER THROUGH ITS CODEC AND BACK (transfer_codec.slang),
        // before anything reads it.
        GaussianArray* blocks[3] = {nullptr, nullptr, nullptr};
        const char* names[3] = {"athenea:splat:transferDirect", "athenea:splat:transferIndirect",
                                "athenea:splat:transferReflected"};
        const uint32_t dims[3] = {16, 48, 48};
        for (GaussianArray& a : arrays) {
            for (int b = 0; b < 3; ++b) {
                const std::string tail(names[b]);
                if (a.name.size() >= tail.size() &&
                    a.name.compare(a.name.size() - tail.size(), tail.size(), tail) == 0 && a.width == dims[b]) {
                    blocks[b] = &a;
                }
            }
        }
        if (blocks[0] == nullptr || blocks[1] == nullptr || blocks[2] == nullptr) {
            return Error(ErrorCode::InvalidArgument, "decimate: --transfer-basis needs a degree 3 transfer");
        }
        const uint64_t c = o.transferClusters, k = o.transferComponents;
        uint64_t floats = c * 112 + 3;
        uint32_t meanAt[3] = {0, 0, 0};
        for (int b = 0; b < 3; ++b) {
            meanAt[b] = static_cast<uint32_t>(floats);
            floats += c * dims[b] + c * dims[b] * k + c * k;
        }
        std::vector<uint32_t> words(floats);
        std::ifstream in(o.transferBasis, std::ios::binary | std::ios::ate);
        if (!in || static_cast<uint64_t>(in.tellg()) != floats * 4) {
            return Error(ErrorCode::InvalidArgument, "decimate: " + o.transferBasis.string() +
                                                         " is not a basis of that many clusters and axes");
        }
        in.seekg(0);
        in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(floats * 4));
        const auto put = [&](const std::vector<uint32_t>& v, const char* label) {
            gpu::BufferDesc desc;
            desc.bytes = std::max<size_t>(v.size(), 1) * 4;
            desc.elementBytes = 4;
            desc.label = label;
            return gpu::Buffer::create(device, desc, v.empty() ? nullptr : v.data());
        };
        auto basisBuffer = put(words, "decimate.transferBasis");
        auto d0 = put(blocks[0]->values[0], "decimate.transferDirect");
        auto d1 = put(blocks[1]->values[0], "decimate.transferIndirect");
        auto d2 = put(blocks[2]->values[0], "decimate.transferReflected");
        auto clipped = put(std::vector<uint32_t>(1, 0u), "decimate.transferClipped");
        if (!basisBuffer || !d0 || !d1 || !d2 || !clipped) {
            return Error(ErrorCode::OutOfMemory, "decimate: the transfer codec");
        }
        auto codec = gpu::ComputeKernel::create(library, "athenea/lod/transfer_codec", "transferCodec");
        if (!codec) return std::move(codec).error();
        const uint32_t n = raw->count;
        {
            gpu::CommandBatch batch(device);
            codec->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["basis"].setBinding(basisBuffer->rhi());
                cursor["direct"].setBinding(d0->rhi());
                cursor["indirect"].setBinding(d1->rhi());
                cursor["reflected"].setBinding(d2->rhi());
                cursor["counters"].setBinding(clipped->rhi());
                cursor["params"]["count"].setData(n);
                cursor["params"]["clusters"].setData(o.transferClusters);
                cursor["params"]["components"].setData(o.transferComponents);
                cursor["params"]["levels"].setData(o.transferBits == 4 ? 7.0F : 127.0F);
                cursor["params"]["meanAt0"].setData(meanAt[0]);
                cursor["params"]["meanAt1"].setData(meanAt[1]);
                cursor["params"]["meanAt2"].setData(meanAt[2]);
                cursor["params"]["normAt"].setData(static_cast<uint32_t>(c * 112));
            });
            ATHENEA_TRY(batch.submit(true));
        }
        // Back into the arrays the output is written from.
        for (int b = 0; b < 3; ++b) {
            const gpu::Buffer& from = b == 0 ? *d0 : (b == 1 ? *d1 : *d2);
            auto back = from.readAll<uint32_t>(device);
            if (!back) return std::move(back).error();
            back->resize(blocks[b]->values[0].size());
            blocks[b]->values[0] = std::move(*back);
        }
        uint32_t clips = 0;
        ATHENEA_TRY(clipped->read(device, 0, 4, &clips));
        std::printf("decimate: the transfer coded with %s (%llu clusters, %llu axes, %u bit weights: %llu B a "
                    "gaussian), %u weights clipped\n",
                    o.transferBasis.string().c_str(), static_cast<unsigned long long>(c),
                    static_cast<unsigned long long>(k), o.transferBits,
                    static_cast<unsigned long long>((c > 256 ? 2 : 1) + (3 * k * o.transferBits + 7) / 8), clips);
    }
    if (o.shadowCodebook > 0) {
        // THE SHADOW BITS FROM A CODEBOOK (shadow_vq.slang), before anything
        // reads them.
        GaussianArray* shadow = nullptr;
        for (GaussianArray& a : arrays) {
            const std::string tail("shadowBits");
            if (a.integer && a.name.size() >= tail.size() &&
                a.name.compare(a.name.size() - tail.size(), tail.size(), tail) == 0) {
                shadow = &a;
            }
        }
        if (shadow == nullptr || shadow->width == 0 || shadow->width > 8) {
            return Error(ErrorCode::InvalidArgument, "decimate: --shadow-codebook needs shadow bits (up to 8 words)");
        }
        const uint32_t n = raw->count, c = o.shadowCodebook, words = shadow->width;
        const auto put = [&](uint64_t count, const uint32_t* data, const char* label) {
            gpu::BufferDesc desc;
            desc.bytes = std::max<uint64_t>(count, 1) * 4;
            desc.elementBytes = 4;
            desc.label = label;
            return gpu::Buffer::create(device, desc, data);
        };
        std::vector<uint32_t> zeros(std::max<uint64_t>(uint64_t{c} * words * 32, 257), 0u);
        auto bitsBuffer = put(uint64_t{n} * words, shadow->values[0].data(), "decimate.shadowBits");
        auto centroids = put(uint64_t{c} * words, zeros.data(), "decimate.shadowCentroids");
        auto assigned = put(n, nullptr, "decimate.shadowAssigned");
        auto ones = put(uint64_t{c} * words * 32, zeros.data(), "decimate.shadowOnes");
        auto members = put(c, zeros.data(), "decimate.shadowMembers");
        auto distances = put(257, zeros.data(), "decimate.shadowDistances");
        if (!bitsBuffer || !centroids || !assigned || !ones || !members || !distances) {
            return Error(ErrorCode::OutOfMemory, "decimate: the shadow codebook");
        }
        const auto kernel = [&](const char* entry) {
            return gpu::ComputeKernel::create(library, "athenea/lod/shadow_vq", entry);
        };
        auto seed = kernel("vqSeed");
        auto assign = kernel("vqAssign");
        auto update = kernel("vqUpdate");
        auto clear = kernel("vqClearMembers");
        auto apply = kernel("vqApply");
        if (!seed || !assign || !update || !clear || !apply) {
            return Error(ErrorCode::InternalError, "decimate: the shadow codebook's kernels");
        }
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["bits"].setBinding(bitsBuffer->rhi());
            cursor["centroids"].setBinding(centroids->rhi());
            cursor["assigned"].setBinding(assigned->rhi());
            cursor["ones"].setBinding(ones->rhi());
            cursor["members"].setBinding(members->rhi());
            cursor["distances"].setBinding(distances->rhi());
            cursor["params"]["count"].setData(n);
            cursor["params"]["clusters"].setData(c);
            cursor["params"]["words"].setData(words);
        };
        {
            gpu::CommandBatch batch(device);
            seed->dispatch(batch, {c, 1, 1}, bind);
            ATHENEA_TRY(batch.submit(true));
        }
        for (uint32_t round = 0; round < o.shadowRounds; ++round) {
            gpu::CommandBatch batch(device);
            clear->dispatch(batch, {c, 1, 1}, bind);
            batch.markDirty();
            assign->dispatch(batch, {n, 1, 1}, bind);
            batch.markDirty();
            if (round + 1 < o.shadowRounds) {
                update->dispatch(batch, {c * words, 1, 1}, bind);
            }
            ATHENEA_TRY(batch.submit(true));
        }
        {
            gpu::CommandBatch batch(device);
            apply->dispatch(batch, {n, 1, 1}, bind);
            ATHENEA_TRY(batch.submit(true));
        }
        auto back = bitsBuffer->readAll<uint32_t>(device);
        if (!back) return std::move(back).error();
        back->resize(shadow->values[0].size());
        shadow->values[0] = std::move(*back);
        std::vector<uint32_t> histogram(257, 0u);
        ATHENEA_TRY(distances->read(device, 0, 257 * 4, histogram.data()));
        uint64_t seen = 0;
        uint32_t p50 = 0, p90 = 0, p99 = 0;
        for (uint32_t d = 0; d < 257; ++d) {
            seen += histogram[d];
            if (p50 == 0 && seen * 2 >= n) p50 = d + 1;
            if (p90 == 0 && seen * 10 >= uint64_t{n} * 9) p90 = d + 1;
            if (p99 == 0 && seen * 100 >= uint64_t{n} * 99) p99 = d + 1;
        }
        std::printf("decimate: the shadow bits from %u centroids (%u rounds): bits off, median %u, p90 %u, p99 %u "
                    "of %u\n", c, o.shadowRounds, p50 - 1, p90 - 1, p99 - 1, words * 32);
    }
    if (o.quantizeChunk > 0) {
        // THE GEOMETRY AT A PHONE CHUNK'S BITS (geometry_quantize.slang).
        GaussianArray* curvatureArray = nullptr;
        for (GaussianArray& a : arrays) {
            const std::string tail("athenea:splat:curvature");
            if (!a.integer && a.width == 3 && a.name.size() >= tail.size() &&
                a.name.compare(a.name.size() - tail.size(), tail.size(), tail) == 0) {
                curvatureArray = &a;
            }
        }
        const io::SplatEncoding& e = raw->encoding;
        const uint32_t n = raw->count;
        const uint32_t chunks = (n + o.quantizeChunk - 1) / o.quantizeChunk;
        const auto put = [&](uint64_t count, const void* data, const char* label) {
            gpu::BufferDesc desc;
            desc.bytes = std::max<uint64_t>(count, 1) * 4;
            desc.elementBytes = 4;
            desc.label = label;
            return gpu::Buffer::create(device, desc, data);
        };
        auto records = put(raw->records.size(), raw->records.data(), "decimate.quantizeRecords");
        auto curv = put(curvatureArray ? curvatureArray->values[0].size() : 1,
                        curvatureArray ? curvatureArray->values[0].data() : nullptr, "decimate.quantizeCurvature");
        auto ranges = put(uint64_t{chunks} * 16, nullptr, "decimate.quantizeRanges");
        if (!records || !curv || !ranges) return Error(ErrorCode::OutOfMemory, "decimate: the quantizer");
        auto rangesKernel = gpu::ComputeKernel::create(library, "athenea/lod/geometry_quantize", "quantizeRanges");
        auto recordsKernel = gpu::ComputeKernel::create(library, "athenea/lod/geometry_quantize", "quantizeRecords");
        if (!rangesKernel || !recordsKernel) return Error(ErrorCode::InternalError, "decimate: the quantizer's kernels");
        const bool hasNormal = e.normal != io::SplatEncoding::kNoField;
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["records"].setBinding(records->rhi());
            cursor["curvature"].setBinding(curv->rhi());
            cursor["ranges"].setBinding(ranges->rhi());
            auto p = cursor["params"];
            p["count"].setData(n);
            p["chunk"].setData(o.quantizeChunk);
            p["stride"].setData(e.floatsPerRecord);
            p["hasNormal"].setData(uint32_t{hasNormal ? 1u : 0u});
            p["x"].setData(e.x);
            p["y"].setData(e.y);
            p["z"].setData(e.z);
            p["opacity"].setData(e.opacity);
            p["s0"].setData(e.scale0);
            p["s1"].setData(e.scale1);
            p["s2"].setData(e.scale2);
            p["rw"].setData(e.rotW);
            p["rx"].setData(e.rotX);
            p["ry"].setData(e.rotY);
            p["rz"].setData(e.rotZ);
            p["dc0"].setData(e.dc0);
            p["dc1"].setData(e.dc1);
            p["dc2"].setData(e.dc2);
            p["normal"].setData(hasNormal ? e.normal : 0u);
            p["hasCurvature"].setData(uint32_t{curvatureArray != nullptr ? 1u : 0u});
            p["linearValues"].setData(uint32_t{e.opacity_ == io::SplatEncoding::Opacity::Linear ? 1u : 0u});
        };
        {
            gpu::CommandBatch batch(device);
            rangesKernel->dispatch(batch, {chunks, 1, 1}, bind);
            batch.markDirty();
            recordsKernel->dispatch(batch, {n, 1, 1}, bind);
            ATHENEA_TRY(batch.submit(true));
        }
        auto back = records->readAll<float>(device);
        if (!back) return std::move(back).error();
        back->resize(raw->records.size());
        raw->records = std::move(*back);
        if (curvatureArray != nullptr) {
            auto bent = curv->readAll<uint32_t>(device);
            if (!bent) return std::move(bent).error();
            bent->resize(curvatureArray->values[0].size());
            curvatureArray->values[0] = std::move(*bent);
        }
        std::printf("decimate: the geometry quantised in %u chunks of %u (24 B a gaussian)\n", chunks,
                    o.quantizeChunk);
    }
    auto cloud = loader->upload(*raw, o.degree);
    if (!cloud) return std::move(cloud).error();
    auto builder = lod::LodBuilder::create(library);
    if (!builder) return std::move(builder).error();
    auto lod = builder->build(*cloud);
    if (!lod) return std::move(lod).error();
    auto decimator = lod::Decimator::create(library);
    if (!decimator) return std::move(decimator).error();

    // HOW EACH ARRAY MERGES, by what it is: a rig's joints with their
    // weights, bits as bits, an int as what may not be merged across (an
    // id, a part, a sheet), and a float as a mean -- metallic, roughness
    // and transmission compared as a colour is, and the radiance a gaussian
    // gives off (`athenea:splat:emission`) merged as one: the mean, weighed
    // by what each covers, is the light the merged one gives off.
    const auto ends = [](const std::string& name, const char* tail) {
        const std::string t(tail);
        return name.size() >= t.size() && name.compare(name.size() - t.size(), t.size(), t) == 0;
    };
    struct Plan {
        size_t              array = 0;
        lod::AttributeMerge how = lod::AttributeMerge::Mean;
        int                 weightsOf = -1;   // for a rig's joints: the weights array
        bool                skip = false;     // a rig's weights, merged with its joints
    };
    std::vector<Plan> plans(arrays.size());
    for (size_t k = 0; k < arrays.size(); ++k) {
        plans[k].array = k;
        const GaussianArray& a = arrays[k];
        if (ends(a.name, "skel:jointIndices")) {
            for (size_t w = 0; w < arrays.size(); ++w) {
                if (ends(arrays[w].name, "skel:jointWeights") && arrays[w].width == a.width) {
                    plans[k].how = lod::AttributeMerge::Rig;
                    plans[k].weightsOf = static_cast<int>(w);
                }
            }
            if (plans[k].weightsOf < 0) {
                plans[k].how = lod::AttributeMerge::First;
            }
        } else if (a.integer) {
            plans[k].how = ends(a.name, "shadowBits") ? lod::AttributeMerge::Bits : lod::AttributeMerge::First;
        } else if (ends(a.name, "athenea:splat:normal") && a.width == 3) {
            plans[k].how = lod::AttributeMerge::Normal;   // a direction: averaged, and a unit again
        } else if (ends(a.name, "athenea:splat:transferZonal") && a.width == 10) {
            plans[k].how = lod::AttributeMerge::Zonal;   // axes in each gaussian's own frame
        } else if (ends(a.name, "athenea:splat:curvature") && a.width == 3) {
            plans[k].how = lod::AttributeMerge::Curvature;   // in the frame's first two axes
        }
    }
    for (Plan& plan : plans) {
        if (plan.how == lod::AttributeMerge::Rig) {
            plans[static_cast<size_t>(plan.weightsOf)].skip = true;
        }
    }
    const auto upload = [&](const std::vector<uint32_t>& words, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<size_t>(words.size(), 1) * 4;
        desc.elementBytes = 4;
        desc.label = label;
        return gpu::Buffer::create(device, desc, words.empty() ? nullptr : words.data());
    };
    const uint32_t records = raw->count;
    auto keysMade = upload(std::vector<uint32_t>(records, 0u), "decimate.keys");
    if (!keysMade) return std::move(keysMade).error();
    gpu::Buffer keys = std::move(*keysMade);
    auto materialMade = upload(std::vector<uint32_t>(size_t{records} * 4, 0u), "decimate.material");
    if (!materialMade) return std::move(materialMade).error();
    gpu::Buffer material = std::move(*materialMade);
    bool keyed = false;
    bool materialled = false;
    for (const Plan& plan : plans) {
        const GaussianArray& a = arrays[plan.array];
        if (plan.skip) {
            continue;
        }
        auto valuesMade = upload(a.values[0], "decimate.values");
        if (!valuesMade) return std::move(valuesMade).error();
        const gpu::Buffer values = std::move(*valuesMade);
        if (plan.how == lod::AttributeMerge::First || plan.how == lod::AttributeMerge::Rig) {
            std::optional<gpu::Buffer> weights;
            if (plan.how == lod::AttributeMerge::Rig) {
                auto w = upload(arrays[static_cast<size_t>(plan.weightsOf)].values[0], "decimate.weights");
                if (!w) return std::move(w).error();
                weights = std::move(*w);
            }
            if (auto added = decimator->addKey(keys, records, values, a.width, plan.how,
                                               weights ? &*weights : nullptr);
                !added) {
                return std::move(added).error();
            }
            keyed = true;
        }
        // A LAMP'S LIGHT KEEPS ITS PLACE: no merge between what a lamp group
        // lights and what it lights four times less, so that a merged disc
        // does not spread one member's headlight over the rest at night.
        if (o.lightBands && a.name.find("athenea:splat:lightLayer:") != std::string::npos && a.width == 3 &&
            !a.integer) {
            if (auto added = decimator->addKey(keys, records, values, 3, lod::AttributeMerge::LightBand); !added) {
                return std::move(added).error();
            }
            keyed = true;
        }
        for (const auto& [tail, component] : {std::pair{"athenea:splat:metallic", 0u},
                                              std::pair{"athenea:splat:roughness", 1u},
                                              std::pair{"athenea:splat:transmission", 2u}}) {
            if (ends(a.name, tail) && a.width == 1 && !a.integer) {
                if (auto added = decimator->addMaterial(material, records, values, component); !added) {
                    return std::move(added).error();
                }
                materialled = true;
            }
        }
    }

    const lod::DecimateSettings& settings = o.settings;
    lod::DecimateCarried carried;
    carried.keys = keyed ? &keys : nullptr;
    carried.material = materialled ? &material : nullptr;
    lod::DecimateStats stats;
    lod::PairSettings pairSettings = o.pairSettings;
    if (o.pairs && usd) {
        // The camera, read on the host: where it stands and how many pixels a
        // unit of length covers at a distance of one.
        const pxr::UsdStageRefPtr stage =
            pxr::UsdStage::Open(o.cameraStage.empty() ? input.string() : o.cameraStage.string());
        const std::string path = o.camera.empty() ? std::string("/World/Camera") : o.camera;
        const pxr::UsdGeomCamera camera(stage ? stage->GetPrimAtPath(pxr::SdfPath(path)) : pxr::UsdPrim());
        if (camera) {
            const pxr::GfCamera gf = camera.GetCamera(pxr::UsdTimeCode::Default());
            const pxr::GfVec3d eye = gf.GetTransform().ExtractTranslation();
            pairSettings.eye[0] = static_cast<float>(eye[0]);
            pairSettings.eye[1] = static_cast<float>(eye[1]);
            pairSettings.eye[2] = static_cast<float>(eye[2]);
            pairSettings.pixelsPerUnit = gf.GetHorizontalAperture() > 0.0F
                                             ? gf.GetFocalLength() / gf.GetHorizontalAperture() *
                                                   static_cast<float>(o.cameraPixels)
                                             : 0.0F;
            std::printf("decimate: seen from %s, %.0f pixels a unit at a distance of one\n", path.c_str(),
                        pairSettings.pixelsPerUnit);
        } else if (!o.camera.empty()) {
            return Error(ErrorCode::InvalidArgument, "decimate: no camera at " + o.camera);
        }
    }
    // A CLOUD PRUNED BY WHAT THE FRAMES SAY, before anything merges: the
    // store drawn from an orbit round its bounds, the rasteriser counting
    // what each gaussian gave every frame (lod_prune.slang says the rest).
    std::optional<lod::LodCloud> pruned;
    if (o.dropHidden || o.pruneError > 1.0F) {
        auto rasterizer = render::TileRasterizer::create(library);
        if (!rasterizer) return std::move(rasterizer).error();
        bool zUp = false;
        if (usd) {
            const pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(input.string());
            zUp = stage && pxr::UsdGeomGetStageUpAxis(stage) == pxr::UsdGeomTokens->z;
        }
        const auto& b = cloud->bounds;
        const render::Vec3 centre{0.5 * (b.min[0] + b.max[0]), 0.5 * (b.min[1] + b.max[1]),
                                  0.5 * (b.min[2] + b.max[2])};
        const double radius = 0.5 * std::sqrt(double(b.max[0] - b.min[0]) * (b.max[0] - b.min[0]) +
                                               double(b.max[1] - b.min[1]) * (b.max[1] - b.min[1]) +
                                               double(b.max[2] - b.min[2]) * (b.max[2] - b.min[2]));
        // Directions over the sphere (a Fibonacci spiral), the elevation over
        // the horizon of the stage's up axis held at or above `hiddenBelow`,
        // each seen from three distances, 90 degrees across.
        const auto cameras = [&](uint32_t views) {
            std::vector<render::Camera> made;
            const double lowest = std::sin(std::clamp(double(o.hiddenBelow), -90.0, 89.0) * M_PI / 180.0);
            const double golden = M_PI * (3.0 - std::sqrt(5.0));
            for (uint32_t k = 0; k < views; ++k) {
                const double up = lowest + (1.0 - lowest) * (k + 0.5) / views;   // sin(elevation)
                const double ring = std::sqrt(std::max(0.0, 1.0 - up * up));
                const double az = golden * k;
                const render::Vec3 dir = zUp ? render::Vec3{ring * std::cos(az), ring * std::sin(az), up}
                                             : render::Vec3{ring * std::cos(az), up, ring * std::sin(az)};
                const render::Vec3 upVec = std::abs(up) > 0.99 ? render::Vec3{1.0, 0.0, 0.0}
                                           : zUp               ? render::Vec3{0.0, 0.0, 1.0}
                                                               : render::Vec3{0.0, 1.0, 0.0};
                for (const double shell : {2.5, 1.6, 1.15}) {
                    render::Camera camera = render::Camera::lookingAt(centre + dir * (radius * shell), centre, upVec);
                    camera.lens.focal = 12.0;
                    camera.lens.haperture = 24.0;
                    camera.lens.vaperture = 24.0;
                    camera.lens.nearZ = std::max(1e-4, radius * 1e-3);
                    camera.lens.farZ = radius * 10.0;
                    made.push_back(camera);
                }
            }
            return made;
        };
        const auto made = [&](uint64_t count, const char* label) {
            gpu::BufferDesc desc;
            desc.bytes = std::max<uint64_t>(count, 1) * 4;
            desc.elementBytes = 4;
            desc.label = label;
            return gpu::Buffer::create(device, desc, nullptr);
        };
        // Every store splat's score over the orbit: mode 1 the weight it
        // blended with, in pixels; mode 2 the squared error its removal would
        // leave, each frame first drawn whole to know what is behind it.
        const auto scored = [&](const lod::LodCloud& from, uint32_t mode, uint32_t views,
                                uint32_t size) -> Result<gpu::Buffer> {
            const uint32_t n = from.count;
            auto frameCount = made(uint64_t{n} * 2, "decimate.frameCount");
            auto score = made(n, "decimate.score");
            auto whole = made(uint64_t{size} * size * 4, "decimate.whole");
            if (!frameCount || !score || !whole) return Error(ErrorCode::OutOfMemory, "decimate: the score");
            {
                gpu::CommandBatch clear(device);
                clear.encoder()->clearBuffer(frameCount->rhi(), 0, uint64_t{std::max(n, 1u)} * 8);
                clear.encoder()->clearBuffer(score->rhi(), 0, uint64_t{std::max(n, 1u)} * 4);
                clear.markDirty();
                ATHENEA_TRY(clear.submit(true));
            }
            const render::SplatInstance instance{&from.splats, render::Mat4::identity(), {}};
            render::RenderSettings draw;
            draw.width = size;
            draw.height = size;
            render::RenderTargets targets;
            for (const render::Camera& camera : cameras(views)) {
                if (mode == 2u) {
                    draw.contribution = nullptr;
                    auto frame = rasterizer->render(camera, std::span(&instance, 1), draw, targets);
                    if (!frame) return std::move(frame).error();
                    gpu::CommandBatch copy(device);
                    copy.encoder()->copyBuffer(whole->rhi(), 0, targets.colour.rhi(), 0, uint64_t{size} * size * 16);
                    copy.markDirty();
                    ATHENEA_TRY(copy.submit(true));
                }
                draw.contribution = &*frameCount;
                draw.contributionMode = mode;
                draw.finalColour = mode == 2u ? &*whole : nullptr;
                auto frame = rasterizer->render(camera, std::span(&instance, 1), draw, targets);
                if (!frame) return std::move(frame).error();
                ATHENEA_TRY(decimator->foldScore(*frameCount, *score, n,
                                                 mode == 2u ? 1.0F / 16777216.0F : 1.0F / 4096.0F, mode == 2u));
            }
            return std::move(*score);
        };
        const lod::LodCloud* current = &*lod;
        if (o.dropHidden) {
            auto score = scored(*current, 1u, o.hiddenViews, o.hiddenSize);
            if (!score) return std::move(score).error();
            auto kept = decimator->prune(*current, *score, o.hiddenPixels);
            if (!kept) return std::move(kept).error();
            std::printf("decimate: %u gaussians seen from %u views of %u (%.1f%% left out as hidden)\n", kept->count,
                        o.hiddenViews * 3, current->count,
                        100.0 * (1.0 - double(kept->count) / std::max(current->count, 1u)));
            pruned = std::move(*kept);
            current = &*pruned;
        }
        if (o.pruneError > 1.0F) {
            // REMOVED BY THE ERROR IT LEAVES, in rounds: each round scores
            // what is left and drops the cheapest, so the count falls
            // geometrically to the target -- one round's removals change
            // what the next one's cost (GaussianPOP re-scores for that).
            const uint32_t start = current->count;
            const uint32_t target = std::max(1u, static_cast<uint32_t>(std::ceil(start / o.pruneError)));
            const uint32_t rounds = std::max(o.pruneRounds, 1u);
            for (uint32_t r = 1; r <= rounds && current->count > target; ++r) {
                const uint32_t goal = std::max(
                    target, static_cast<uint32_t>(std::lround(start * std::pow(double(target) / start, double(r) / rounds))));
                auto score = scored(*current, 2u, o.pruneViews, o.pruneSize);
                if (!score) return std::move(score).error();
                auto bins = decimator->scoreHistogram(*score, current->count);
                if (!bins) return std::move(bins).error();
                const uint64_t drop = current->count - goal;
                uint64_t sum = 0;
                float threshold = 0.0F;
                for (uint32_t k = 0; k < lod::Decimator::kScoreBins; ++k) {
                    sum += (*bins)[k];
                    if (sum >= drop) {
                        threshold = lod::Decimator::scoreOfBin(k + 1);
                        break;
                    }
                }
                auto kept = decimator->prune(*current, *score, threshold);
                if (!kept) return std::move(kept).error();
                std::printf("decimate: error round %u: %u gaussians of %u (threshold %.3g)\n", r, kept->count,
                            current->count, threshold);
                pruned = std::move(*kept);
                current = &*pruned;
            }
        }
    }
    if (o.cutBudget > 0 || o.cutPixels > 0.0F || o.cutError > 0.0F || o.treeCoverage || o.occlusionClusters > 0 ||
        o.treePages > 0 || !o.markClosed.empty() || !o.markMaterialRoughness.empty()) {
        // A CUT THROUGH THE MERGE TREE the input is (lod_tree_cut.slang):
        // each gaussian's width on the camera's screen, then the threshold
        // searched whose cut holds the budget, and the store pruned to it.
        const GaussianArray* parentArray = nullptr;
        const GaussianArray* leavesArray = nullptr;
        const GaussianArray* costArray = nullptr;
        const GaussianArray* startArray = nullptr;
        const GaussianArray* roughArray = nullptr;
        const GaussianArray* spillArray = nullptr;
        for (const GaussianArray& a : arrays) {
            if (ends(a.name, "athenea:splat:lodParent")) parentArray = &a;
            if (ends(a.name, "athenea:splat:lodSize")) leavesArray = &a;
            if (ends(a.name, "athenea:splat:lodCost")) costArray = &a;
            if (ends(a.name, "athenea:splat:lodStart")) startArray = &a;
            if (ends(a.name, "athenea:splat:lodSpill") && !a.integer) spillArray = &a;
            if (ends(a.name, "athenea:splat:roughness") && a.width == 1 && !a.integer) roughArray = &a;
        }
        if (parentArray == nullptr || leavesArray == nullptr || !(pairSettings.pixelsPerUnit > 0.0F)) {
            return Error(ErrorCode::InvalidArgument,
                         "decimate: a cut needs a merge tree (decimate --tree) and a camera (--ratio 1)");
        }
        const lod::LodCloud& from = pruned ? *pruned : *lod;
        const uint32_t n = from.count;
        // The tree's indices count the file's gaussians, which is more than
        // the store's where the loader dropped some (under 1/255): what is
        // kept a file gaussian is sized by them. A dropped merge has no centre
        // of its own, so a tree is best cut after --tree-coverage, which
        // floors every merge where the loader keeps it.
        const uint32_t fileCount = raw->count;
        auto widthKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeCutWidth");
        if (!widthKernel) return std::move(widthKernel).error();
        auto keepKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeCutKeep");
        if (!keepKernel) return std::move(keepKernel).error();
        auto parents = upload(parentArray->values[0], "decimate.cutParent");
        auto leaves = upload(leavesArray->values[0], "decimate.cutLeaves");
        auto costs = upload(costArray != nullptr ? costArray->values[0] : std::vector<uint32_t>(fileCount, 0u),
                            "decimate.cutCost");
        // A merge's spill (treeCoverage), as the file has it, or 0 until a
        // --tree-coverage measures it.
        auto spills = upload(spillArray != nullptr && spillArray->values[0].size() >= fileCount
                                 ? spillArray->values[0]
                                 : std::vector<uint32_t>(fileCount, 0u),
                             "decimate.cutSpill");
        auto widths = upload(std::vector<uint32_t>(fileCount, 0u), "decimate.cutWidth");
        auto score = upload(std::vector<uint32_t>(n, 0u), "decimate.cutScore");
        auto counter = upload(std::vector<uint32_t>(1, 0u), "decimate.cutCount");
        auto errorBits = upload(std::vector<uint32_t>(fileCount, 0u), "decimate.cutErrors");
        auto radiusBits = upload(std::vector<uint32_t>(fileCount, 0u), "decimate.cutRadii");
        auto centres = upload(std::vector<uint32_t>(uint64_t{fileCount} * 4, 0u), "decimate.cutCentres");
        auto changed = upload(std::vector<uint32_t>(1, 0u), "decimate.cutChanged");
        auto spreadBits = upload(std::vector<uint32_t>(fileCount, 0u), "decimate.cutSpread");
        auto storeOfs = upload(std::vector<uint32_t>(fileCount, 0u), "decimate.cutStoreOf");
        auto starts = upload(startArray != nullptr ? startArray->values[0] : std::vector<uint32_t>(fileCount, 0u),
                             "decimate.cutRunStart");
        if (!storeOfs || !starts) return Error(ErrorCode::OutOfMemory, "decimate: the cut's coverage");
        auto coneBits = upload(std::vector<uint32_t>(fileCount, 0u), "decimate.cutCone");
        auto looks = upload(std::vector<uint32_t>(uint64_t{fileCount} * 4, 0u), "decimate.cutLook");
        auto facings = upload(std::vector<uint32_t>(uint64_t{fileCount} * 4, 0u), "decimate.cutFacing");
        const float half = 0.5F;
        uint32_t halfBits;
        std::memcpy(&halfBits, &half, 4);
        auto roughs = upload(roughArray != nullptr ? roughArray->values[0] : std::vector<uint32_t>(fileCount, halfBits),
                             "decimate.cutRoughness");
        // WHICH NODES' BACKS MAY GO (--cut-cull-back): a closed shell's
        // (primvars:athenea:splat:closed), not glass (transmission over a
        // half). Without the closed mark, every node, as before.
        std::vector<uint32_t> cullableWords(fileCount, 1u);
        {
            const GaussianArray* closedArray = nullptr;
            const GaussianArray* glassArray = nullptr;
            for (const GaussianArray& a : arrays) {
                if (ends(a.name, "athenea:splat:closed") && a.integer && a.width == 1) closedArray = &a;
                if (ends(a.name, "athenea:splat:transmission") && !a.integer && a.width == 1) glassArray = &a;
            }
            for (uint32_t g = 0; g < fileCount; ++g) {
                const bool closedHere = closedArray == nullptr || closedArray->values[0][g] != 0u;
                float t = 0.0F;
                if (glassArray != nullptr) std::memcpy(&t, &glassArray->values[0][g], 4);
                cullableWords[g] = closedHere && !(t > 0.5F) ? 1u : 0u;
            }
        }
        auto cullables = upload(cullableWords, "decimate.cutCullable");
        // THE OCCLUSION MASKS (--cut-occlusion): each node's entry and the
        // codebook, a constant of the field.
        std::vector<uint32_t> occlusionWords(1, 0u), occlusionBookWords(8, 0xFFFFFFFFu);
        if (o.cutOcclusion) {
            for (const GaussianArray& a : arrays) {
                if (ends(a.name, "athenea:splat:occlusionIndex") && a.integer && a.width == 1) {
                    occlusionWords = a.values[0];
                }
            }
            const pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(input.string());
            pxr::VtIntArray book;
            if (!stage || occlusionWords.size() != fileCount ||
                !stage->GetPrimAtPath(pxr::SdfPath(where))
                     .GetAttribute(pxr::TfToken("primvars:athenea:splat:occlusionCodebook"))
                     .Get(&book)) {
                return Error(ErrorCode::InvalidArgument, "decimate: --cut-occlusion needs the tree's occlusion masks");
            }
            occlusionBookWords.assign(book.begin(), book.end());
        }
        auto culledMarks = upload(std::vector<uint32_t>(fileCount, 0u), "decimate.cutCulled");
        auto occlusionIndexBuf = upload(occlusionWords, "decimate.cutOcclusionIndex");
        auto occlusionBookBuf = upload(occlusionBookWords, "decimate.cutOcclusionBook");
        if (!spreadBits || !coneBits || !looks || !facings || !roughs) {
            return Error(ErrorCode::OutOfMemory, "decimate: the cut's bounds");
        }
        if (!errorBits || !radiusBits || !centres || !changed) return Error(ErrorCode::OutOfMemory, "decimate: the cut");
        if (!parents || !leaves || !costs || !spills || !widths || !score || !counter) {
            return Error(ErrorCode::OutOfMemory, "decimate: the cut");
        }
        float threshold = 1.0F;
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["storeOrder"].setBinding(from.order.rhi());
            cursor["positions"].setBinding(from.splats.positions.rhi());
            cursor["shape"].setBinding(from.splats.shape.rhi());
            cursor["parent"].setBinding(parents->rhi());
            cursor["leaves"].setBinding(leaves->rhi());
            cursor["cost"].setBinding(costs->rhi());
            cursor["spill"].setBinding(spills->rhi());
            cursor["params"]["spillWeight"].setData(o.cutSpillWeight);
            cursor["params"]["spillFloor"].setData(o.cutSpillFloor);
            cursor["params"]["relaxedShare"].setData(0.25F);
            cursor["width"].setBinding(widths->rhi());
            cursor["score"].setBinding(score->rhi());
            cursor["counter"].setBinding(counter->rhi());
            cursor["errorBits"].setBinding(errorBits->rhi());
            cursor["radiusBits"].setBinding(radiusBits->rhi());
            cursor["centre"].setBinding(centres->rhi());
            cursor["changed"].setBinding(changed->rhi());
            cursor["spreadBits"].setBinding(spreadBits->rhi());
            cursor["coneBits"].setBinding(coneBits->rhi());
            cursor["look"].setBinding(looks->rhi());
            cursor["facing"].setBinding(facings->rhi());
            cursor["roughness"].setBinding(roughs->rhi());
            cursor["cullable"].setBinding(cullables->rhi());
            cursor["occlusionIndex"].setBinding(occlusionIndexBuf->rhi());
            cursor["culledMark"].setBinding(culledMarks->rhi());
            cursor["occlusionBook"].setBinding(occlusionBookBuf->rhi());
            cursor["params"]["occlusion"].setData(uint32_t{o.cutOcclusion ? 1u : 0u});
            cursor["storeOf"].setBinding(storeOfs->rhi());
            cursor["runStart"].setBinding(starts->rhi());
            cursor["outPositions"].setBinding(from.splats.positions.rhi());
            cursor["params"]["maxRun"].setData(uint32_t{2048});
            cursor["storeNormals"].setBinding(from.splats.hasNormals() ? from.splats.normals.rhi() : parents->rhi());
            cursor["params"]["hasNormals"].setData(uint32_t{from.splats.hasNormals() ? 1u : 0u});
            cursor["params"]["colourWeight"].setData(10.0F);
            cursor["params"]["normalWeight"].setData(1.0F);
            cursor["params"]["guard"].setData(o.cutGuard);
            cursor["params"]["floorSize"].setData(2.0F);
            cursor["params"]["cullBack"].setData(uint32_t{o.cutCullBack && o.cutBounds ? 1u : 0u});
            cursor["params"]["count"].setData(n);
            cursor["params"]["fileCount"].setData(fileCount);
            cursor["params"]["threshold"].setData(threshold);
            cursor["params"]["pixelsPerUnit"].setData(pairSettings.pixelsPerUnit);
            cursor["params"]["maxDepth"].setData(uint32_t{256});
            const float eye[4] = {pairSettings.eye[0], pairSettings.eye[1], pairSettings.eye[2], 1.0F};
            cursor["params"]["eye"].setData(eye, 16);
        };
        // WHAT THE CUT KEEPS RESIDENT, paged: the pages are bookkeeping over
        // the tree's shape (its parents, leaf counts and runs), a page of at
        // most `cutPages` gaussians; the GPU counts the ones the cut touches.
        const auto measurePages = [&](uint32_t kept) -> Result<void> {
            if (startArray == nullptr) {
                return Error(ErrorCode::InvalidArgument, "decimate: --cut-pages needs the tree's lodStart");
            }
            const std::vector<uint32_t>& up = parentArray->values[0];
            const std::vector<uint32_t>& size = leavesArray->values[0];
            const std::vector<uint32_t>& first = startArray->values[0];
            const uint32_t limit = o.cutPages;
            const auto nodes = [&](uint32_t i) { return 2u * std::max(size[i], 1u) - 1u; };
            const auto isRoot = [&](uint32_t i) { return up[i] == i || up[i] >= fileCount; };
            const auto byLeaf = [&](uint32_t a, uint32_t b) { return first[a] < first[b]; };
            // Each node's children, in leaf order.
            std::vector<uint32_t> childStart(fileCount + 1, 0u);
            for (uint32_t i = 0; i < fileCount; ++i) {
                if (!isRoot(i)) ++childStart[up[i] + 1];
            }
            for (uint32_t i = 0; i < fileCount; ++i) childStart[i + 1] += childStart[i];
            std::vector<uint32_t> children(childStart[fileCount]);
            {
                std::vector<uint32_t> at(childStart.begin(), childStart.end() - 1);
                for (uint32_t i = 0; i < fileCount; ++i) {
                    if (!isRoot(i)) children[at[up[i]]++] = i;
                }
            }
            // PAGES BY LEVEL (Nanite's way, research's side thinking): a page
            // is the top of a group of sibling subtrees, taken breadth first
            // up to `limit` nodes; its frontier, in leaf order, is packed into
            // groups -- a subtree as large as a page alone, small ones together
            // -- each the roots of a child page. The piece's roots are the
            // first group. A coarse cut stays in the coarse pages.
            std::vector<uint32_t> pageOfNode(fileCount, 0u);
            std::vector<uint32_t> pageSizes;
            std::vector<uint32_t> pageParent;
            std::vector<std::pair<std::vector<uint32_t>, uint32_t>> groups;   // roots, parent page
            {
                std::vector<uint32_t> roots;
                for (uint32_t i = 0; i < fileCount; ++i) {
                    if (isRoot(i)) roots.push_back(i);
                }
                std::sort(roots.begin(), roots.end(), byLeaf);
                groups.push_back({roots, 0u});
            }
            for (size_t g = 0; g < groups.size(); ++g) {
                const uint32_t page = static_cast<uint32_t>(pageSizes.size());
                pageSizes.push_back(0u);
                pageParent.push_back(groups[g].second == 0u && g == 0 ? 0u : groups[g].second);
                std::vector<uint32_t> level = groups[g].first;
                std::vector<uint32_t> frontier;
                while (!level.empty()) {
                    std::vector<uint32_t> below;
                    for (uint32_t i : level) {
                        if (pageSizes[page] < limit) {
                            pageOfNode[i] = page;
                            ++pageSizes[page];
                            below.insert(below.end(), children.begin() + childStart[i],
                                         children.begin() + childStart[i + 1]);
                        } else {
                            frontier.push_back(i);
                        }
                    }
                    level = std::move(below);
                }
                std::sort(frontier.begin(), frontier.end(), byLeaf);
                std::vector<uint32_t> group;
                uint32_t held = 0;
                for (uint32_t i : frontier) {
                    const uint32_t weight = std::min(nodes(i), limit);
                    if (!group.empty() && held + weight > limit) {
                        groups.push_back({group, page});
                        group.clear();
                        held = 0;
                    }
                    group.push_back(i);
                    held += weight;
                }
                if (!group.empty()) {
                    groups.push_back({group, page});
                }
            }
            const uint32_t pages = static_cast<uint32_t>(pageSizes.size());
            auto pageOf = upload(pageOfNode, "decimate.pageOf");
            auto parentPages = upload(pageParent, "decimate.pageParent");
            auto sizes = upload(pageSizes, "decimate.pageSize");
            auto touched = upload(std::vector<uint32_t>(pages, 0u), "decimate.pageTouched");
            auto counts = upload(std::vector<uint32_t>(2, 0u), "decimate.pageCounts");
            auto keptSizes = upload(std::vector<uint32_t>(32, 0u), "decimate.keptSizes");
            auto touchKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_pages", "treePagesTouched");
            auto countKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_pages", "treePagesCount");
            if (!pageOf || !parentPages || !sizes || !touched || !counts || !touchKernel || !countKernel) {
                return Error(ErrorCode::InternalError, "decimate: the pages");
            }
            const auto bindPages = [&](rhi::ShaderCursor cursor) {
                // lod_pages' own names only (its module is apart from the cut's).
                cursor["storeOrder"].setBinding(from.order.rhi());
                cursor["parent"].setBinding(parents->rhi());
                cursor["leaves"].setBinding(leaves->rhi());
                cursor["score"].setBinding(score->rhi());
                cursor["errorBits"].setBinding(errorBits->rhi());
                cursor["radiusBits"].setBinding(radiusBits->rhi());
                cursor["centre"].setBinding(centres->rhi());
                cursor["runStart"].setBinding(starts->rhi());
                cursor["facing"].setBinding(facings->rhi());
                cursor["params"]["facingBuckets"].setData(uint32_t{o.pagesByFacing ? 1u : 0u});
                cursor["params"]["count"].setData(n);
                cursor["params"]["fileCount"].setData(fileCount);
                cursor["params"]["threshold"].setData(threshold);
                cursor["params"]["pixelsPerUnit"].setData(pairSettings.pixelsPerUnit);
                const float eye[4] = {pairSettings.eye[0], pairSettings.eye[1], pairSettings.eye[2], 1.0F};
                cursor["params"]["eye"].setData(eye, 16);
                cursor["pageOf"].setBinding(pageOf->rhi());
                cursor["pageParent"].setBinding(parentPages->rhi());
                cursor["pageSize"].setBinding(sizes->rhi());
                cursor["touched"].setBinding(touched->rhi());
                cursor["pageCounts"].setBinding(counts->rhi());
                cursor["keptSizes"].setBinding(keptSizes->rhi());
                cursor["params"]["pageCount"].setData(pages);
            };
            {
                gpu::CommandBatch batch(device);
                touchKernel->dispatch(batch, {n, 1, 1}, bindPages);
                batch.markDirty();
                countKernel->dispatch(batch, {pages, 1, 1}, bindPages);
                ATHENEA_TRY(batch.submit(true));
            }
            uint32_t resident[2] = {0, 0};
            ATHENEA_TRY(counts->read(device, 0, 8, resident));
            // And by error: a band an octave of the parent's error, each band's
            // nodes in leaf order cut into pages of `limit`.
            {
                auto hi = upload(std::vector<uint32_t>(n, 0u), "decimate.bandHi");
                auto lo = upload(std::vector<uint32_t>(n, 0u), "decimate.bandLo");
                auto values = upload(std::vector<uint32_t>(n, 0u), "decimate.bandValues");
                auto scratchHi = upload(std::vector<uint32_t>(n, 0u), "decimate.bandScratchHi");
                auto scratchLo = upload(std::vector<uint32_t>(n, 0u), "decimate.bandScratchLo");
                auto scratchValues = upload(std::vector<uint32_t>(n, 0u), "decimate.bandScratchValues");
                auto bandCount = upload(std::vector<uint32_t>(64, 0u), "decimate.bandCount");
                auto keysKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_pages", "treeBandKeys");
                auto pagesKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_pages", "treeBandPages");
                auto pagesCount = gpu::ComputeKernel::create(library, "athenea/lod/lod_pages", "treeErrorPagesCount");
                auto sorter = gpu::RadixSort::create(library);
                if (!hi || !lo || !values || !scratchHi || !scratchLo || !scratchValues || !bandCount || !keysKernel ||
                    !pagesKernel || !pagesCount || !sorter) {
                    return Error(ErrorCode::InternalError, "decimate: the pages by error");
                }
                gpu::SortBuffers sorting;
                sorting.keysLo = *lo;
                sorting.keysHi = *hi;
                sorting.values = *values;
                sorting.scratchKeysLo = *scratchLo;
                sorting.scratchKeysHi = *scratchHi;
                sorting.scratchValues = *scratchValues;
                const auto bindSort = [&](rhi::ShaderCursor cursor) {
                    bindPages(cursor);
                    cursor["sortHi"].setBinding(hi->rhi());
                    cursor["sortLo"].setBinding(lo->rhi());
                    cursor["sortValues"].setBinding(values->rhi());
                    cursor["bandCount"].setBinding(bandCount->rhi());
                };
                {
                    gpu::CommandBatch batch(device);
                    keysKernel->dispatch(batch, {n, 1, 1}, bindSort);
                    batch.markDirty();
                    ATHENEA_TRY(sorter->sort(batch, sorting, n, 60));
                    ATHENEA_TRY(batch.submit(true));
                }
                const uint32_t pages = (n + limit - 1) / limit;
                std::vector<uint32_t> starts(64, 0u), firstPage(64, 0u);   // unused: the pages cross the bands
                auto bandStarts = upload(starts, "decimate.bandStart");
                auto bandPages = upload(firstPage, "decimate.bandPage");
                auto keySize = upload(std::vector<uint32_t>(std::max(pages, 1u), 0u), "decimate.errorPageSize");
                auto keyNeeded = upload(std::vector<uint32_t>(std::max(pages, 1u), 0u), "decimate.errorPageNeeded");
                auto byError = upload(std::vector<uint32_t>(5, 0u), "decimate.errorPageCounts");
                // What the culls leave out, so a walk stops there.
                {
                    auto mark = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeCulledMark");
                    if (!mark) return std::move(mark).error();
                    gpu::CommandBatch batch(device);
                    mark->dispatch(batch, {n, 1, 1}, bind);
                    ATHENEA_TRY(batch.submit(true));
                }
                if (!bandStarts || !bandPages || !keySize || !keyNeeded || !byError) {
                    return Error(ErrorCode::OutOfMemory, "decimate: the pages by error");
                }
                const auto bindKeys = [&](rhi::ShaderCursor cursor) {
                    bindSort(cursor);
                    cursor["sortHi"].setBinding(sorting.keysHi.rhi());
                    cursor["sortLo"].setBinding(sorting.keysLo.rhi());
                    cursor["sortValues"].setBinding(sorting.values.rhi());
                    cursor["bandStart"].setBinding(bandStarts->rhi());
                    cursor["bandPage"].setBinding(bandPages->rhi());
                    cursor["errorPageSize"].setBinding(keySize->rhi());
                    cursor["errorPageNeeded"].setBinding(keyNeeded->rhi());
                    cursor["pageCounts"].setBinding(byError->rhi());
                    cursor["culledMark"].setBinding(culledMarks->rhi());
                    cursor["params"]["pageCount"].setData(pages);
                    cursor["params"]["maxRun"].setData(limit);
                };
                {
                    gpu::CommandBatch batch(device);
                    pagesKernel->dispatch(batch, {n, 1, 1}, bindKeys);
                    batch.markDirty();
                    pagesCount->dispatch(batch, {pages, 1, 1}, bindKeys);
                    ATHENEA_TRY(batch.submit(true));
                }
                uint32_t e[5] = {0, 0, 0, 0, 0};
                ATHENEA_TRY(byError->read(device, 0, 20, e));
                std::printf("decimate: paged by error (coarsest first, %u a page): %u pages, %u at most; the cut "
                            "needs %u holding %u (%.2fx); %u nodes needed, a walk stopping at what the culls leave out "
                            "(%.2fx)\n",
                            limit, e[2], e[3], e[0], e[1], kept > 0 ? double(e[1]) / kept : 0.0, e[4],
                            kept > 0 ? double(e[4]) / kept : 0.0);
            }
            uint32_t bins[32] = {};
            ATHENEA_TRY(keptSizes->read(device, 0, sizeof(bins), bins));
            std::printf("decimate: the cut's nodes by log2(leaves):");
            for (uint32_t b = 0; b < 32; ++b) {
                if (bins[b] > 0) std::printf(" %u:%u", b, bins[b]);
            }
            std::printf("\n");
            std::printf("decimate: paged by level at %u gaussians: %u pages, the root page %u; the cut's %u gaussians "
                        "need %u pages holding %u (%.2fx)\n",
                        limit, pages, pageSizes[0], kept, resident[0], resident[1],
                        kept > 0 ? double(resident[1]) / kept : 0.0);
            return ok();
        };
        {
            gpu::CommandBatch batch(device);
            widthKernel->dispatch(batch, {n, 1, 1}, bind);
            ATHENEA_TRY(batch.submit(true));
        }
        // EACH NODE'S ERROR AND SPHERE, climbed until every parent covers its
        // children (lod_tree_cut.slang, research 100): with `bounds` the colour
        // spread and normal cone in it. In world units, no view.
        const auto errorsUp = [&](bool bounds) -> Result<uint32_t> {
                auto own = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeErrorOwn");
                auto up = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeErrorUp");
                if (!own || !up) return Error(ErrorCode::InternalError, "decimate: the error cut's kernels");
                {
                    gpu::CommandBatch batch(device);
                    own->dispatch(batch, {n, 1, 1}, bind);
                    ATHENEA_TRY(batch.submit(true));
                }
                // Climb a kernel up the tree until no parent is raised.
                const auto climb = [&](gpu::ComputeKernel& kernel) -> Result<uint32_t> {
                    uint32_t rounds = 0;
                    for (; rounds < 512; ++rounds) {
                        const uint32_t zero = 0;
                        ATHENEA_TRY(changed->write(device, 0, 4, &zero));
                        gpu::CommandBatch batch(device);
                        kernel.dispatch(batch, {fileCount, 1, 1}, bind);
                        ATHENEA_TRY(batch.submit(true));
                        uint32_t raised = 0;
                        ATHENEA_TRY(changed->read(device, 0, 4, &raised));
                        if (raised == 0) {
                            break;
                        }
                    }
                    return rounds;
                };
                uint32_t rounds = 0;
                if (bounds) {
                    // The bounds and the sphere first, then each node's own error
                    // with them, then the error's climb.
                    auto boundsUp = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeBoundsUp");
                    auto bounded = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeErrorBounded");
                    auto errorClimb = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeErrorClimb");
                    if (!boundsUp || !bounded || !errorClimb) {
                        return Error(ErrorCode::InternalError, "decimate: the bounds' kernels");
                    }
                    auto first = climb(*boundsUp);
                    if (!first) return std::move(first).error();
                    {
                        gpu::CommandBatch batch(device);
                        bounded->dispatch(batch, {n, 1, 1}, bind);
                        ATHENEA_TRY(batch.submit(true));
                    }
                    auto second = climb(*errorClimb);
                    if (!second) return std::move(second).error();
                    rounds = *first + *second;
                } else {
                    auto both = climb(*up);
                    if (!both) return std::move(both).error();
                    rounds = *both;
                }
            return rounds;
        };
        // THE TREE'S PAGES BY ERROR, written beside it (lod_pages.slang): the
        // errors climbed with the bounds, each node's band of its parent's
        // error, the bands' nodes in leaf order cut into pages, and each
        // page's table row.
        const auto pageTable = [&](std::vector<GaussianArray>& added) -> Result<void> {
            if (startArray == nullptr) {
                return Error(ErrorCode::InvalidArgument, "decimate: --tree-pages needs the tree's lodStart");
            }
            auto climbed = errorsUp(true);
            if (!climbed) return std::move(climbed).error();
            const uint32_t limit = o.treePages;
            auto hi = upload(std::vector<uint32_t>(n, 0u), "decimate.bandHi");
            auto lo = upload(std::vector<uint32_t>(n, 0u), "decimate.bandLo");
            auto values = upload(std::vector<uint32_t>(n, 0u), "decimate.bandValues");
            auto scratchHi = upload(std::vector<uint32_t>(n, 0u), "decimate.bandScratchHi");
            auto scratchLo = upload(std::vector<uint32_t>(n, 0u), "decimate.bandScratchLo");
            auto scratchValues = upload(std::vector<uint32_t>(n, 0u), "decimate.bandScratchValues");
            auto bandCount = upload(std::vector<uint32_t>(64, 0u), "decimate.bandCount");
            auto keysKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_pages", "treeBandKeys");
            auto tableKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_pages", "treeBandTable");
            auto outKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_pages", "treeBandTableOut");
            auto sorter = gpu::RadixSort::create(library);
            if (!hi || !lo || !values || !scratchHi || !scratchLo || !scratchValues || !bandCount || !keysKernel ||
                !tableKernel || !outKernel || !sorter) {
                return Error(ErrorCode::InternalError, "decimate: the page table");
            }
            gpu::SortBuffers sorting;
            sorting.keysLo = *lo;
            sorting.keysHi = *hi;
            sorting.values = *values;
            sorting.scratchKeysLo = *scratchLo;
            sorting.scratchKeysHi = *scratchHi;
            sorting.scratchValues = *scratchValues;
            uint32_t pages = 0;
            const auto take = [](Result<gpu::Buffer> made) { return made ? std::move(*made) : gpu::Buffer{}; };
            std::optional<gpu::Buffer> bandStarts, bandPages, filePage, info, box, error, bounds;
            const auto bindTable = [&](rhi::ShaderCursor cursor) {
                cursor["storeOrder"].setBinding(from.order.rhi());
                cursor["parent"].setBinding(parents->rhi());
                cursor["leaves"].setBinding(leaves->rhi());
                cursor["errorBits"].setBinding(errorBits->rhi());
                cursor["radiusBits"].setBinding(radiusBits->rhi());
                cursor["centre"].setBinding(centres->rhi());
                cursor["runStart"].setBinding(starts->rhi());
                cursor["facing"].setBinding(facings->rhi());
                cursor["params"]["facingBuckets"].setData(uint32_t{o.pagesByFacing ? 1u : 0u});
                cursor["sortHi"].setBinding(sorting.keysHi.rhi());
                cursor["sortLo"].setBinding(sorting.keysLo.rhi());
                cursor["sortValues"].setBinding(sorting.values.rhi());
                cursor["bandCount"].setBinding(bandCount->rhi());
                if (bandStarts) {
                    cursor["bandStart"].setBinding(bandStarts->rhi());
                    cursor["bandPage"].setBinding(bandPages->rhi());
                    cursor["filePage"].setBinding(filePage->rhi());
                    cursor["pageInfo"].setBinding(info->rhi());
                    cursor["pageBox"].setBinding(box->rhi());
                    cursor["pageError"].setBinding(error->rhi());
                    cursor["pageBounds"].setBinding(bounds->rhi());
                }
                cursor["params"]["count"].setData(n);
                cursor["params"]["fileCount"].setData(fileCount);
                cursor["params"]["maxRun"].setData(limit);
                cursor["params"]["pageCount"].setData(pages);
            };
            {
                gpu::CommandBatch batch(device);
                keysKernel->dispatch(batch, {n, 1, 1}, bindTable);
                batch.markDirty();
                ATHENEA_TRY(sorter->sort(batch, sorting, n, 60));
                ATHENEA_TRY(batch.submit(true));
            }
            pages = (n + limit - 1) / limit;
            std::vector<uint32_t> startsOfBands(64, 0u), reversed(64, 0u);   // unused by the table
            const uint32_t one = std::max(pages, 1u);
            const uint32_t big = 0xFFFFFFFFu;
            bandStarts = take(upload(startsOfBands, "decimate.bandStart"));
            bandPages = take(upload(reversed, "decimate.bandPage"));
            filePage = take(upload(std::vector<uint32_t>(fileCount, big), "decimate.filePage"));
            std::vector<uint32_t> infoInit(size_t{one} * 3, 0u);
            for (uint32_t p = 0; p < one; ++p) infoInit[p * 3 + 2] = big;
            info = take(upload(infoInit, "decimate.pageInfo"));
            std::vector<uint32_t> boxInit(size_t{one} * 6, 0u);
            for (uint32_t p = 0; p < one; ++p) {
                for (int k = 0; k < 3; ++k) boxInit[p * 6 + k] = big;
            }
            box = take(upload(boxInit, "decimate.pageBox"));
            std::vector<uint32_t> errorInit(size_t{one} * 2, 0u);
            for (uint32_t p = 0; p < one; ++p) errorInit[p * 2 + 1] = big;
            error = take(upload(errorInit, "decimate.pageError"));
            bounds = take(upload(std::vector<uint32_t>(size_t{one} * 8, 0u), "decimate.pageBounds"));
            if (!bandStarts->valid() || !bandPages->valid() || !filePage->valid() || !info->valid() ||
                !box->valid() || !error->valid() || !bounds->valid()) {
                return Error(ErrorCode::OutOfMemory, "decimate: the page table");
            }
            {
                gpu::CommandBatch batch(device);
                tableKernel->dispatch(batch, {n, 1, 1}, bindTable);
                batch.markDirty();
                outKernel->dispatch(batch, {one, 1, 1}, bindTable);
                ATHENEA_TRY(batch.submit(true));
            }
            auto pageOf = filePage->readAll<uint32_t>(device);
            auto infoBack = info->readAll<uint32_t>(device);
            auto boundsBack = bounds->readAll<uint32_t>(device);
            if (!pageOf || !infoBack || !boundsBack) return Error(ErrorCode::IoFailure, "decimate: the page table back");
            pageOf->resize(fileCount);
            infoBack->resize(size_t{pages} * 3);
            // A page's first node in page order (coarsest first): what a packer
            // writing the pages one after another starts it at.
            for (uint32_t p = 0, at = 0; p < pages; ++p) {
                (*infoBack)[size_t{p} * 3 + 2] = at;
                at += (*infoBack)[size_t{p} * 3 + 1];
            }
            boundsBack->resize(size_t{pages} * 8);
            const auto array = [](const char* name, const char* type, const char* interpolation,
                                  std::vector<uint32_t> words) {
                GaussianArray a;
                a.name = name;
                a.typeName = type;
                a.integer = std::string(type) == "int[]";
                a.interpolation = interpolation;
                a.values.push_back(std::move(words));
                return a;
            };
            added.push_back(array("primvars:athenea:splat:lodPage", "int[]", "vertex", std::move(*pageOf)));
            added.push_back(array("primvars:athenea:splat:lodPageInfo", "int[]", "constant", std::move(*infoBack)));
            added.push_back(array("primvars:athenea:splat:lodPageBounds", "float[]", "constant", std::move(*boundsBack)));
            std::printf("decimate: %u pages by error of %u gaussians (coarsest first)\n", pages, limit);
            return ok();
        };
        // EVERY MERGE'S REAL COVERAGE (treeCoverage): its opacity, the rest NaN.
        const auto treeCoverageOf = [&](std::vector<float>& opacities) -> Result<void> {
            if (startArray == nullptr) {
                return Error(ErrorCode::InvalidArgument, "decimate: --tree-coverage needs the tree's lodStart");
            }
            const float one = 1.0F;
            uint32_t oneBits;
            std::memcpy(&oneBits, &one, 4);
            ATHENEA_TRY(score->write(device, 0, uint64_t{n} * 4, std::vector<uint32_t>(n, oneBits).data()));
            auto storeOfKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeStoreOf");
            auto coverage = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeCoverage");
            if (!storeOfKernel || !coverage) return Error(ErrorCode::InternalError, "decimate: the coverage kernels");
            {
                gpu::CommandBatch batch(device);
                storeOfKernel->dispatch(batch, {n, 1, 1}, bind);
                batch.markDirty();
                coverage->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    bind(cursor);
                    cursor["params"]["maxRun"].setData(uint32_t{4096});
                });
                ATHENEA_TRY(batch.submit(true));
            }
            // Back in the file's order: the tree's own arrays are file
            // indices, so it is written as it came, only its merges'
            // opacities changed -- and the mass-over-area array renamed, so
            // no reader uncaps the opacity from it.
            auto toFile = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeCoverageToFile");
            auto floorKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeCoverageFloor");
            // A record per file gaussian, not per store splat: the loader
            // drops what draws nothing (a merge of opacity 0), and those
            // keep what the file says (NaN: not written).
            const uint32_t records = static_cast<uint32_t>(opacities.size());
            gpu::BufferDesc fileDesc;
            fileDesc.bytes = uint64_t{records} * 4;
            fileDesc.elementBytes = 4;
            fileDesc.label = "decimate.fileOpacity";
            auto fileOpacity = gpu::Buffer::create(device, fileDesc, std::vector<uint32_t>(records, 0x7FC00000u).data());
            if (!toFile || !floorKernel || !fileOpacity) return Error(ErrorCode::InternalError, "decimate: the coverage's write");
            {
                gpu::CommandBatch batch(device);
                toFile->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    bind(cursor);
                    cursor["fileOpacity"].setBinding(fileOpacity->rhi());
                });
                batch.markDirty();
                floorKernel->dispatch(batch, {records, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    bind(cursor);
                    cursor["fileOpacity"].setBinding(fileOpacity->rhi());
                    cursor["params"]["count"].setData(records);
                });
                ATHENEA_TRY(batch.submit(true));
            }
            auto read = fileOpacity->readAll<float>(device);
            if (!read) return std::move(read).error();
            read->resize(records);
            opacities = std::move(*read);
            std::printf("decimate: every merge's opacity is what its leaves cover\n");
            return ok();
        };
        // EACH NODE'S OCCLUSION MASK (occlusion_mask.slang, Occlusion.cpp).
        const auto occlusionMasks = [&](uint32_t records, std::vector<GaussianArray>& added) -> Result<void> {
            auto in = occlusionInputOf(arrays, records);
            if (!in) return std::move(in).error();
            auto book = occlusionBook(library, occlusionUpOf(input, std::move(*in)), o.occlusionClusters);
            if (!book) return std::move(book).error();
            std::printf("decimate: occlusion masks climbed in %u levels; %u nodes worth least left open; %llu closed "
                        "cells over the nodes, a codebook of %u opens %llu of them (%.1f%%)\n",
                        book->levels, book->spared, static_cast<unsigned long long>(book->closed), o.occlusionClusters,
                        static_cast<unsigned long long>(book->reopened),
                        book->closed > 0 ? 100.0 * double(book->reopened) / double(book->closed) : 0.0);
            for (GaussianArray& a : occlusionArrays(*book)) {
                added.push_back(std::move(a));
            }
            return ok();
        };
        if (o.treeCoverage || o.occlusionClusters > 0 || o.treePages > 0 || !o.markClosed.empty() ||
            !o.markMaterialRoughness.empty()) {
            // THE WHOLE TREE WRITTEN AGAIN, in the file's order (its own
            // arrays are file indices), with what a player cuts it by.
            const uint32_t records = raw->count;
            std::vector<float> opacities(records, std::numeric_limits<float>::quiet_NaN());
            std::vector<GaussianArray> added;
            if (o.treeCoverage) {
                ATHENEA_TRY(treeCoverageOf(opacities));
                // And each merge's spill, beside its coverage, for a cut to
                // weigh (--cut-spill).
                auto spilled = spills->readAll<uint32_t>(device);
                if (!spilled) return std::move(spilled).error();
                spilled->resize(records);
                GaussianArray lodSpill;
                lodSpill.name = "primvars:athenea:splat:lodSpill";
                lodSpill.typeName = "float[]";
                lodSpill.interpolation = "vertex";
                lodSpill.values.push_back(std::move(*spilled));
                added.push_back(std::move(lodSpill));
            }
            if (o.occlusionClusters > 0) {
                ATHENEA_TRY(occlusionMasks(records, added));
            }
            if (o.treePages > 0) {
                ATHENEA_TRY(pageTable(added));
            }
            if (!o.markClosed.empty()) {
                // WHICH PARTS ARE CLOSED SHELLS, from a list of their prim
                // paths (one a line): primvars:athenea:splat:closed a node, by
                // its part (cryptoObject) -- bookkeeping.
                std::ifstream list(o.markClosed);
                if (!list) return Error(ErrorCode::IoFailure, "decimate: cannot read " + o.markClosed.string());
                std::set<uint32_t> closedIds;
                for (std::string line; std::getline(list, line);) {
                    if (!line.empty()) closedIds.insert(core::cryptomatteId(line));
                }
                const GaussianArray* ids = nullptr;
                for (const GaussianArray& a : arrays) {
                    if (ends(a.name, "athenea:splat:cryptoObject") && a.integer && a.width == 1) ids = &a;
                }
                if (ids == nullptr) return Error(ErrorCode::InvalidArgument, "decimate: --mark-closed needs cryptoObject");
                GaussianArray closed;
                closed.name = "primvars:athenea:splat:closed";
                closed.typeName = "int[]";
                closed.integer = true;
                closed.interpolation = "vertex";
                std::vector<uint32_t> marks(records, 0u);
                uint32_t marked = 0;
                for (uint32_t g = 0; g < records; ++g) {
                    marks[g] = closedIds.count(ids->values[0][g]) ? 1u : 0u;
                    marked += marks[g];
                }
                closed.values.push_back(std::move(marks));
                added.push_back(std::move(closed));
                std::printf("decimate: %u of %u gaussians in the %zu closed parts listed\n", marked, records, closedIds.size());
            }
            if (!o.markMaterialRoughness.empty()) {
                // EACH PART'S MATERIAL ROUGHNESS, before the conversion widened
                // it, from a list of "prim-path roughness" lines:
                // primvars:athenea:splat:materialRoughness a node, by its part
                // (cryptoObject); -1 for a part the list does not name (read
                // as sharp) -- bookkeeping.
                std::ifstream list(o.markMaterialRoughness);
                if (!list) {
                    return Error(ErrorCode::IoFailure, "decimate: cannot read " + o.markMaterialRoughness.string());
                }
                std::map<uint32_t, float> byId;
                for (std::string line; std::getline(list, line);) {
                    std::istringstream fields(line);
                    std::string path;
                    float roughness = -1.0F;
                    if (fields >> path >> roughness) byId.emplace(core::cryptomatteId(path), roughness);
                }
                const GaussianArray* ids = nullptr;
                for (const GaussianArray& a : arrays) {
                    if (ends(a.name, "athenea:splat:cryptoObject") && a.integer && a.width == 1) ids = &a;
                }
                if (ids == nullptr) {
                    return Error(ErrorCode::InvalidArgument, "decimate: --mark-material-roughness needs cryptoObject");
                }
                GaussianArray own;
                own.name = "primvars:athenea:splat:materialRoughness";
                own.typeName = "float[]";
                own.interpolation = "vertex";
                std::vector<uint32_t> words(records);
                uint32_t named = 0;
                for (uint32_t g = 0; g < records; ++g) {
                    const auto found = byId.find(ids->values[0][g]);
                    const float value = found != byId.end() ? found->second : -1.0F;
                    named += found != byId.end() ? 1u : 0u;
                    std::memcpy(&words[g], &value, 4);
                }
                own.values.push_back(std::move(words));
                added.push_back(std::move(own));
                std::printf("decimate: %u of %u gaussians in the %zu parts whose material roughness is listed\n",
                            named, records, byId.size());
            }
            if (!usd) {
                return Error(ErrorCode::InvalidArgument, "decimate: --tree-coverage and --occlusion-mask write a USD tree");
            }
            ATHENEA_TRY(writeTreeOpacities(input, output, where, opacities,
                                           o.treeCoverage ? "primvars:athenea:splat:coverage" : "",
                                           "primvars:athenea:splat:coverageMass", added));
            std::printf("decimate: wrote the tree %s (%u gaussians, in the file's order)\n", output.string().c_str(),
                        records);
            return stats;
        } else if (o.cutError > 0.0F) {
            // THE CUT BY ERROR (lod_tree_cut.slang, research 100): each node's
            // own error and sphere, climbed until every parent covers its
            // children, then the one cut at the threshold.
            auto keepByError = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeErrorKeep");
            if (!keepByError) return Error(ErrorCode::InternalError, "decimate: the error cut's kernels");
            auto climbed = errorsUp(o.cutBounds);
            if (!climbed) return std::move(climbed).error();
            const uint32_t rounds = *climbed;
            const auto keepAt = [&](float t) -> Result<uint32_t> {
                threshold = t;
                const uint32_t zero = 0;
                ATHENEA_TRY(counter->write(device, 0, 4, &zero));
                gpu::CommandBatch batch(device);
                keepByError->dispatch(batch, {n, 1, 1}, bind);
                ATHENEA_TRY(batch.submit(true));
                uint32_t held = 0;
                ATHENEA_TRY(counter->read(device, 0, 4, &held));
                return held;
            };
            float at = o.cutError;
            if (o.cutBudget > 0) {
                // With a budget too, the error threshold that fills it,
                // bisected in its logarithm (the count falls as it rises).
                double low = std::log(1.0e-4), high = std::log(1.0e4);
                for (int k = 0; k < 40; ++k) {
                    const double mid = 0.5 * (low + high);
                    auto held = keepAt(static_cast<float>(std::exp(mid)));
                    if (!held) return std::move(held).error();
                    (*held > o.cutBudget ? low : high) = mid;
                }
                at = static_cast<float>(std::exp(high));
            }
            auto keptAt = keepAt(at);
            if (!keptAt) return std::move(keptAt).error();
            const uint32_t kept = *keptAt;
            if (o.cutCoverage) {
                if (startArray == nullptr) {
                    return Error(ErrorCode::InvalidArgument, "decimate: --cut-coverage needs the tree's lodStart");
                }
                auto storeOfKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeStoreOf");
                auto coverage = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeCoverage");
                if (!storeOfKernel || !coverage) return Error(ErrorCode::InternalError, "decimate: the coverage kernels");
                gpu::CommandBatch batch(device);
                storeOfKernel->dispatch(batch, {n, 1, 1}, bind);
                batch.markDirty();
                coverage->dispatch(batch, {n, 1, 1}, bind);
                ATHENEA_TRY(batch.submit(true));
                std::printf("decimate: the kept merges' opacity is what their leaves cover\n");
            }
            std::printf("decimate: the cut by error from the camera at %.3g pixels holds %u of the tree's %u "
                        "gaussians (%u rounds up the tree)\n", at, kept, n, rounds);
            if (o.cutPages > 0) {
                ATHENEA_TRY(measurePages(kept));
            }
            auto cut = decimator->prune(from, *score, 0.5F);
            if (!cut) return std::move(cut).error();
            pruned = std::move(*cut);
        } else {
        const auto cutAt = [&](float t) -> Result<uint32_t> {
            threshold = t;
            const uint32_t zero = 0;
            ATHENEA_TRY(counter->write(device, 0, 4, &zero));
            gpu::CommandBatch batch(device);
            keepKernel->dispatch(batch, {n, 1, 1}, bind);
            ATHENEA_TRY(batch.submit(true));
            uint32_t kept = 0;
            ATHENEA_TRY(counter->read(device, 0, 4, &kept));
            return kept;
        };
        // The count falls as the threshold rises: bisected in its logarithm
        // for the smallest threshold whose cut fits.
        double low = std::log(1.0e-3), high = std::log(o.cutPixels > 0.0F ? double(o.cutPixels) : 1.0e5);
        for (int k = 0; k < 40 && o.cutPixels <= 0.0F; ++k) {
            const double mid = 0.5 * (low + high);
            auto kept = cutAt(static_cast<float>(std::exp(mid)));
            if (!kept) return std::move(kept).error();
            (*kept > o.cutBudget ? low : high) = mid;
        }
        auto kept = cutAt(static_cast<float>(std::exp(high)));
        if (!kept) return std::move(kept).error();
        std::printf("decimate: the cut from the camera at %.3g pixels holds %u of the tree's %u gaussians\n",
                    std::exp(high), *kept, n);
        auto cut = decimator->prune(from, *score, 0.5F);
        if (!cut) return std::move(cut).error();
        pruned = std::move(*cut);
        }
    }
    const lod::LodCloud& store = pruned ? *pruned : *lod;
    lod::DecimateResult treeResult;
    const bool tree = o.pairs && !o.treeOutput.empty();
    auto decimated = o.pairs ? decimator->decimatePairs(store, cloud->origin, pairSettings, carried, &stats,
                                                        tree ? &treeResult : nullptr)
                             : decimator->decimate(store, cloud->origin, settings, carried, &stats);
    if (!decimated) return std::move(decimated).error();
    // The arrays as read (a merge tree's last write releases each once it is on
    // the device: a 4.4M-leaf part at 5120 px held them three times over in
    // the memory the GPU shares, and ran out writing its tree).
    std::vector<GaussianArray>& source = arrays;

    // ONE RESULT WRITTEN: every array, and every sample of it, merged over
    // what each of its gaussians stands for, beside what the merge adds.
    const auto write = [&](lod::DecimateResult& result, const std::filesystem::path& to,
                           std::vector<GaussianArray> added, bool surfel, bool lodTree) -> Result<void> {
        // Every array's values are replaced by its merge (a rig's weights by
        // its joints'): the metadata alone is copied.
        std::vector<GaussianArray> merged;
        merged.reserve(source.size());
        for (const GaussianArray& from : source) {
            GaussianArray a;
            a.name = from.name;
            a.typeName = from.typeName;
            a.integer = from.integer;
            a.width = from.width;
            a.elementSize = from.elementSize;
            a.interpolation = from.interpolation;
            a.times = from.times;
            a.values.resize(from.values.size());
            merged.push_back(std::move(a));
        }
        auto kept = loader->records(result.cloud);
        if (!kept) return std::move(kept).error();
        const uint32_t count = result.cloud.count;
        for (const Plan& plan : plans) {
            if (plan.skip) {
                continue;
            }
            GaussianArray& a = merged[plan.array];
            GaussianArray* w = plan.weightsOf >= 0 ? &merged[static_cast<size_t>(plan.weightsOf)] : nullptr;
            for (size_t v = 0; v < a.values.size(); ++v) {
                auto valuesMade = upload(source[plan.array].values[v], "decimate.values");
                if (!valuesMade) return std::move(valuesMade).error();
                const gpu::Buffer values = std::move(*valuesMade);
                std::optional<gpu::Buffer> weights;
                if (w != nullptr) {
                    const GaussianArray& sw = source[static_cast<size_t>(plan.weightsOf)];
                    auto made = upload(sw.values[std::min(v, sw.values.size() - 1)], "decimate.weights");
                    if (!made) return std::move(made).error();
                    weights = std::move(*made);
                }
                if (lodTree) {
                    // The tree is the last write: what is on the device is not
                    // needed on the host again.
                    std::vector<uint32_t>().swap(source[plan.array].values[v]);
                }
                auto m = decimator->mergeAttribute(store, result, values, a.width, plan.how,
                                                   weights ? &*weights : nullptr);
                if (!m) return std::move(m).error();
                auto back = m->first.readAll<uint32_t>(device);
                if (!back) return std::move(back).error();
                back->resize(size_t{count} * a.width);
                a.values[v] = std::move(*back);
                if (w != nullptr && v < w->values.size()) {
                    auto backWeights = m->second.readAll<uint32_t>(device);
                    if (!backWeights) return std::move(backWeights).error();
                    backWeights->resize(size_t{count} * a.width);
                    w->values[v] = std::move(*backWeights);
                }
            }
        }
        for (GaussianArray& a : added) {
            std::erase_if(merged, [&](const GaussianArray& m) { return m.name == a.name; });
            merged.push_back(std::move(a));
        }
        ExportOptions options;
        options.surfel = surfel;
        options.lodTree = lodTree;
        options.maxDegree = o.degree;
        options.addCamera = o.addCamera;
        if (usd) {
            // The gaussians themselves to a stage of their own, and from it
            // into a copy of the stage they came from, beside everything else.
            // Only that stage's arrays are read: the copy keeps the source's
            // metersPerUnit and upAxis, so this one's (metres) are never seen.
            const std::filesystem::path gaussians =
                std::filesystem::temp_directory_path() / ("athenea_decimate_" + to.stem().string() + ".usdc");
            options.addCamera = false;
            ATHENEA_TRY(writeParticleFieldStage(library, *kept, gaussians, options));
            auto written = writeDecimatedStage(input, to, where, gaussians, merged);
            std::filesystem::remove(gaussians);
            return written;
        }
        return writeParticleFieldStage(library, *kept, to, options);
    };
    const auto floats = [&](const gpu::Buffer& b, uint32_t count, const char* name,
                            const char* type) -> Result<GaussianArray> {
        auto words = b.readAll<uint32_t>(device);
        if (!words) return std::move(words).error();
        words->resize(count);
        GaussianArray a;
        a.name = name;
        a.typeName = type;
        a.interpolation = "vertex";
        a.values.push_back(std::move(*words));
        return a;
    };
    std::vector<GaussianArray> added;
    // A cloud that already carries what each gaussian covers -- a merge tree
    // cut (`cutPixels`, `cutBudget`), whose merges' coverage is not held under
    // 0.99 -- keeps its own: written again from a pass that merged nothing it
    // would be the capped opacity, and a reader that draws coverage past one
    // (sparkwebGPU) found the cut's merges too thin.
    const bool carriesCoverage = std::any_of(arrays.begin(), arrays.end(), [&](const GaussianArray& a) {
        return ends(a.name, "athenea:splat:coverage") || ends(a.name, "athenea:splat:coverageMass");
    });
    if (o.pairs && decimated->coverage.valid() && !carriesCoverage) {
        // What each merge covers, beside the opacity held at 0.99.
        auto a = floats(decimated->coverage, decimated->cloud.count, "primvars:athenea:splat:coverage", "float[]");
        if (!a) return std::move(a).error();
        added.push_back(std::move(*a));
    }
    if (o.pairs && o.pairSettings.hardness && decimated->hardness.valid()) {
        // How hard each merge's edge is (packing's hard surfel; 2 a gaussian).
        auto a = floats(decimated->hardness, decimated->cloud.count, "primvars:athenea:splat:hardness", "float[]");
        if (!a) return std::move(a).error();
        added.push_back(std::move(*a));
    }
    ATHENEA_TRY(write(*decimated, output, std::move(added), o.pairs && stats.surfels > 0, false));
    if (tree) {
        // THE MERGE TREE, a file of its own: every merge a gaussian after the
        // leaves, its parent and its cost (`athenea:splat:lodParent`,
        // `lodCost`), the leaves in depth-first order.
        const uint32_t n = treeResult.cloud.count;
        std::vector<GaussianArray> treeAdded;
        for (auto [buffer, name, type] :
             {std::tuple{&treeResult.coverage, "primvars:athenea:splat:coverage", "float[]"},
              std::tuple{&treeResult.parent, "primvars:athenea:splat:lodParent", "int[]"},
              std::tuple{&treeResult.cost, "primvars:athenea:splat:lodCost", "float[]"}}) {
            auto a = floats(*buffer, n, name, type);
            if (!a) return std::move(a).error();
            treeAdded.push_back(std::move(*a));
        }
        // Each node's run of leaves, as two ints: where it starts, how many.
        {
            auto words = treeResult.run.readAll<uint32_t>(device);
            if (!words) return std::move(words).error();
            words->resize(size_t{n} * 2);
            GaussianArray start, size;
            start.name = "primvars:athenea:splat:lodStart";
            size.name = "primvars:athenea:splat:lodSize";
            for (GaussianArray* a : {&start, &size}) {
                a->typeName = "int[]";
                a->interpolation = "vertex";
                a->values.emplace_back(n);
            }
            for (uint32_t k = 0; k < n; ++k) {
                start.values[0][k] = (*words)[size_t{k} * 2];
                size.values[0][k] = (*words)[size_t{k} * 2 + 1];
            }
            treeAdded.push_back(std::move(start));
            treeAdded.push_back(std::move(size));
        }
        ATHENEA_TRY(write(treeResult, o.treeOutput, std::move(treeAdded), true, true));
        std::printf("decimate: wrote the merge tree, %u gaussians, to %s\n", n, o.treeOutput.string().c_str());
    }
    return stats;
}

}   // namespace athenea::usd
