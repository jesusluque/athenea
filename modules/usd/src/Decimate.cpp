// Copyright (c) 2026 jesus luque.
//
// A CLOUD DECIMATED, AND EVERYTHING IT CARRIED KEPT: the pipeline behind
// `athenea decimate`, here so that a test can run what the command runs.
#include "athenea/usd/Export.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/lod/Lod.h"
#include "athenea/scene/GpuClouds.h"

#include <pxr/base/gf/camera.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>

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
        const pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(input.string());
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
    lod::DecimateResult treeResult;
    const bool tree = o.pairs && !o.treeOutput.empty();
    auto decimated = o.pairs ? decimator->decimatePairs(*lod, cloud->origin, pairSettings, carried, &stats,
                                                        tree ? &treeResult : nullptr)
                             : decimator->decimate(*lod, cloud->origin, settings, carried, &stats);
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
                auto m = decimator->mergeAttribute(*lod, result, values, a.width, plan.how,
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
    if (o.pairs && decimated->coverage.valid()) {
        // What each merge covers, beside the opacity held at 0.99.
        auto a = floats(decimated->coverage, decimated->cloud.count, "primvars:athenea:splat:coverage", "float[]");
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
        ATHENEA_TRY(write(treeResult, o.treeOutput, std::move(treeAdded), true, true));
        std::printf("decimate: wrote the merge tree, %u gaussians, to %s\n", n, o.treeOutput.string().c_str());
    }
    return stats;
}

}   // namespace athenea::usd
