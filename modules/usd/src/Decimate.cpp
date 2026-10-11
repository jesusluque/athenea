// Copyright (c) 2026 jesus luque.
//
// A CLOUD DECIMATED, AND EVERYTHING IT CARRIED KEPT: the pipeline behind
// `athenea decimate`, here so that a test can run what the command runs.
#include "athenea/usd/Export.h"

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
    if (o.cutBudget > 0 || o.cutPixels > 0.0F || o.cutError > 0.0F || o.treeCoverage) {
        // A CUT THROUGH THE MERGE TREE the input is (lod_tree_cut.slang):
        // each gaussian's width on the camera's screen, then the threshold
        // searched whose cut holds the budget, and the store pruned to it.
        const GaussianArray* parentArray = nullptr;
        const GaussianArray* leavesArray = nullptr;
        const GaussianArray* costArray = nullptr;
        const GaussianArray* startArray = nullptr;
        const GaussianArray* roughArray = nullptr;
        for (const GaussianArray& a : arrays) {
            if (ends(a.name, "athenea:splat:lodParent")) parentArray = &a;
            if (ends(a.name, "athenea:splat:lodSize")) leavesArray = &a;
            if (ends(a.name, "athenea:splat:lodCost")) costArray = &a;
            if (ends(a.name, "athenea:splat:lodStart")) startArray = &a;
            if (ends(a.name, "athenea:splat:roughness") && a.width == 1 && !a.integer) roughArray = &a;
        }
        if (parentArray == nullptr || leavesArray == nullptr || !(pairSettings.pixelsPerUnit > 0.0F)) {
            return Error(ErrorCode::InvalidArgument,
                         "decimate: a cut needs a merge tree (decimate --tree) and a camera (--ratio 1)");
        }
        const lod::LodCloud& from = pruned ? *pruned : *lod;
        const uint32_t n = from.count;
        auto widthKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeCutWidth");
        if (!widthKernel) return std::move(widthKernel).error();
        auto keepKernel = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeCutKeep");
        if (!keepKernel) return std::move(keepKernel).error();
        auto parents = upload(parentArray->values[0], "decimate.cutParent");
        auto leaves = upload(leavesArray->values[0], "decimate.cutLeaves");
        auto costs = upload(costArray != nullptr ? costArray->values[0] : std::vector<uint32_t>(n, 0u),
                            "decimate.cutCost");
        auto widths = upload(std::vector<uint32_t>(n, 0u), "decimate.cutWidth");
        auto score = upload(std::vector<uint32_t>(n, 0u), "decimate.cutScore");
        auto counter = upload(std::vector<uint32_t>(1, 0u), "decimate.cutCount");
        auto errorBits = upload(std::vector<uint32_t>(n, 0u), "decimate.cutErrors");
        auto radiusBits = upload(std::vector<uint32_t>(n, 0u), "decimate.cutRadii");
        auto centres = upload(std::vector<uint32_t>(uint64_t{n} * 4, 0u), "decimate.cutCentres");
        auto changed = upload(std::vector<uint32_t>(1, 0u), "decimate.cutChanged");
        auto spreadBits = upload(std::vector<uint32_t>(n, 0u), "decimate.cutSpread");
        auto storeOfs = upload(std::vector<uint32_t>(n, 0u), "decimate.cutStoreOf");
        auto starts = upload(startArray != nullptr ? startArray->values[0] : std::vector<uint32_t>(n, 0u),
                             "decimate.cutRunStart");
        if (!storeOfs || !starts) return Error(ErrorCode::OutOfMemory, "decimate: the cut's coverage");
        auto coneBits = upload(std::vector<uint32_t>(n, 0u), "decimate.cutCone");
        auto looks = upload(std::vector<uint32_t>(uint64_t{n} * 4, 0u), "decimate.cutLook");
        auto facings = upload(std::vector<uint32_t>(uint64_t{n} * 4, 0u), "decimate.cutFacing");
        const float half = 0.5F;
        uint32_t halfBits;
        std::memcpy(&halfBits, &half, 4);
        auto roughs = upload(roughArray != nullptr ? roughArray->values[0] : std::vector<uint32_t>(n, halfBits),
                             "decimate.cutRoughness");
        if (!spreadBits || !coneBits || !looks || !facings || !roughs) {
            return Error(ErrorCode::OutOfMemory, "decimate: the cut's bounds");
        }
        if (!errorBits || !radiusBits || !centres || !changed) return Error(ErrorCode::OutOfMemory, "decimate: the cut");
        if (!parents || !leaves || !costs || !widths || !score || !counter) {
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
            cursor["storeOf"].setBinding(storeOfs->rhi());
            cursor["runStart"].setBinding(starts->rhi());
            cursor["outPositions"].setBinding(from.splats.positions.rhi());
            cursor["params"]["maxRun"].setData(uint32_t{2048});
            cursor["storeNormals"].setBinding(from.splats.hasNormals() ? from.splats.normals.rhi() : parents->rhi());
            cursor["params"]["hasNormals"].setData(uint32_t{from.splats.hasNormals() ? 1u : 0u});
            cursor["params"]["colourWeight"].setData(10.0F);
            cursor["params"]["normalWeight"].setData(1.0F);
            cursor["params"]["guard"].setData(4.0F);
            cursor["params"]["floorSize"].setData(2.0F);
            cursor["params"]["cullBack"].setData(uint32_t{o.cutCullBack && o.cutBounds ? 1u : 0u});
            cursor["params"]["count"].setData(n);
            cursor["params"]["threshold"].setData(threshold);
            cursor["params"]["pixelsPerUnit"].setData(pairSettings.pixelsPerUnit);
            cursor["params"]["maxDepth"].setData(uint32_t{256});
            const float eye[4] = {pairSettings.eye[0], pairSettings.eye[1], pairSettings.eye[2], 1.0F};
            cursor["params"]["eye"].setData(eye, 16);
        };
        {
            gpu::CommandBatch batch(device);
            widthKernel->dispatch(batch, {n, 1, 1}, bind);
            ATHENEA_TRY(batch.submit(true));
        }
        if (o.treeCoverage) {
            // EVERY MERGE'S REAL COVERAGE, the whole tree kept.
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
            const uint32_t records = raw->count;
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
            auto opacities = fileOpacity->readAll<float>(device);
            if (!opacities) return std::move(opacities).error();
            opacities->resize(records);
            if (!usd) {
                return Error(ErrorCode::InvalidArgument, "decimate: --tree-coverage writes a USD tree");
            }
            ATHENEA_TRY(writeTreeOpacities(input, output, where, *opacities, "primvars:athenea:splat:coverage",
                                           "primvars:athenea:splat:coverageMass"));
            std::printf("decimate: every merge's opacity is what its leaves cover (%u gaussians, in the file's "
                        "order); wrote %s\n", records, output.string().c_str());
            return stats;
        } else if (o.cutError > 0.0F) {
            // THE CUT BY ERROR (lod_tree_cut.slang, research 100): each node's
            // own error and sphere, climbed until every parent covers its
            // children, then the one cut at the threshold.
            auto own = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeErrorOwn");
            auto up = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeErrorUp");
            auto keepByError = gpu::ComputeKernel::create(library, "athenea/lod/lod_tree_cut", "treeErrorKeep");
            if (!own || !up || !keepByError) return Error(ErrorCode::InternalError, "decimate: the error cut's kernels");
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
                    kernel.dispatch(batch, {n, 1, 1}, bind);
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
            if (o.cutBounds) {
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
    const std::vector<GaussianArray> source = arrays;

    // ONE RESULT WRITTEN: every array, and every sample of it, merged over
    // what each of its gaussians stands for, beside what the merge adds.
    const auto write = [&](lod::DecimateResult& result, const std::filesystem::path& to,
                           std::vector<GaussianArray> added, bool surfel, bool lodTree) -> Result<void> {
        std::vector<GaussianArray> merged = source;
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
