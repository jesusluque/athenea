// Copyright (c) 2026 jesus luque.
//
// `athenea lightmap`: the light of a stage's meshes, path traced into
// textures over a set of texture coordinates of their own (`st_lightmap`),
// and a stage that shows them unlit (`primvars:athenea:unlit`). The processor
// reads the stage, frames the files and keeps count; which texel each
// triangle covers, the rays, the bake, and the dilation of the islands'
// margins are the device's (shaders/athenea/usd/lightmap.slang).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "Commands.h"
#include "athenea/geom/Mesh.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu/algo/PrefixSum.h"
#include "athenea/gpu_host/Context.h"
#include "athenea/io/Exr.h"
#include "athenea/technique/Denoiser.h"
#include "athenea/technique/PathTracer.h"
#include "athenea/usd/MeshStage.h"
#include "athenea/usd/StageRenderer.h"

namespace athenea::cli {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t elements, uint32_t elementBytes, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(elements, 1) * elementBytes;
    desc.elementBytes = elementBytes;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

/// A prim path's components: "/root/a/b" -> {"root", "a", "b"}.
std::vector<std::string> componentsOf(const std::string& path) {
    std::vector<std::string> out;
    std::stringstream in(path);
    std::string part;
    while (std::getline(in, part, '/')) {
        if (!part.empty()) {
            out.push_back(part);
        }
    }
    return out;
}

/// The overs a layer needs to reach each of `meshes` (a prim path and its
/// atlas), each mesh given the lines `body(atlas)` returns, nested as the
/// paths are.
struct OverTree {
    std::map<std::string, OverTree> children;
    int32_t                         atlas = -1;
};

void writeOvers(std::ostream& out, const OverTree& node, int depth, const std::string& uv) {
    const std::string indent(static_cast<size_t>(depth) * 4, ' ');
    for (const auto& [name, child] : node.children) {
        out << indent << "over \"" << name << "\"";
        if (child.atlas >= 0) {
            out << " (\n" << indent << "    prepend apiSchemas = [\"MaterialBindingAPI\"]\n" << indent << ")";
        }
        out << "\n" << indent << "{\n";
        if (child.atlas >= 0) {
            out << indent << "    int primvars:athenea:unlit = 1 (\n"
                << indent << "        interpolation = \"constant\"\n"
                << indent << "    )\n"
                << indent << "    rel material:binding = </AtheneaLightmaps/atlas" << child.atlas << ">\n";
        }
        writeOvers(out, child, depth + 1, uv);
        out << indent << "}\n";
    }
}

}   // namespace

void addLightmap(CLI::App& app) {
    struct Options {
        std::string stage, output = "lightmap", uv = "st_lightmap", prim;
        uint32_t size = 1024, samples = 64, bounces = 3, dilate = 8, extraSamples = 0, passSamples = 32,
                 maxPasses = 16;
        bool noDenoise = false;
        double time = 0.0;
        bool defaultLights = false;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand(
        "lightmap", "a stage's meshes path traced into textures over their own texture coordinates, and a stage "
                    "that shows them unlit");
    cmd->add_option("stage", o->stage, ".usd / .usda / .usdc")->required();
    cmd->add_option("-o,--output", o->output, "a directory: the atlases (lightmap_<n>.exr) and lightmapped.usda");
    cmd->add_option("--uv", o->uv, "the primvar of the meshes' lightmap coordinates");
    cmd->add_option("--size", o->size, "an atlas's side, texels")->check(CLI::Range(16u, 16384u));
    cmd->add_option("--samples", o->samples, "paths a texel")->check(CLI::Range(1u, 65536u));
    cmd->add_option("--bounces", o->bounces, "bounces after the first hit")->check(CLI::Range(0u, 16u));
    cmd->add_option("--dilate", o->dilate, "rings of texels the islands are grown by into their margins");
    cmd->add_option("--extra-samples", o->extraSamples,
                    "paths a texel on average added after the first pass, where the light is noisiest");
    cmd->add_option("--pass-samples", o->passSamples, "--extra-samples: paths a texel an extra pass");
    cmd->add_option("--max-passes", o->maxPasses, "--extra-samples: extra passes a texel at most");
    cmd->add_flag("--no-denoise", o->noDenoise, "leave the atlas as traced (by default OIDN denoises it)");
    cmd->add_option("--prim", o->prim, "only meshes at or under this prim path");
    cmd->add_option("--time", o->time, "USD time code");
    cmd->add_flag("--default-lights", o->defaultLights, "a dome and a sun, for a stage with no lights");
    cmd->callback([o] {
        const auto started = Clock::now();
        const fs::path out = o->output;
        std::error_code error;
        fs::create_directories(out, error);
        if (error) {
            cli::err("lightmap: cannot make %s: %s\n", out.string().c_str(), error.message().c_str());
            throw CLI::RuntimeError(1);
        }
        gpu_host::Context* context = gpu_host::installProcessContext();
        if (context == nullptr || context->compute() == nullptr) {
            cli::err("no GPU compute device (gpe has no backend here)\n");
            throw CLI::RuntimeError(1);
        }
        gpu::ShaderLibrary library(context->deviceShared());
        struct Atlas {
            int32_t                  index = 0;
            uint32_t                 texels = 0;   // claimed
            std::vector<std::string> meshes;
        };
        std::vector<Atlas> atlases;
        Result<void> inside = ok();
        auto ran = context->run([&] {
            inside = [&]() -> Result<void> {
                gpu::Device& device = library.device();
                auto stage = usd::MeshStage::open(o->stage);
                if (!stage) return std::move(stage).error();
                auto builder = geom::MeshBuilder::create(library);
                if (!builder) return std::move(builder).error();
                usd::MeshStageOptions options;
                options.prim = o->prim;
                options.time = o->time;
                options.extraUv = o->uv;
                auto meshes = stage->read(*builder, options);
                if (!meshes) return std::move(meshes).error();
                std::map<int32_t, std::vector<const usd::StageMesh*>> byAtlas;
                for (const usd::StageMesh& m : *meshes) {
                    if (m.hasExtraUv && m.path.find('[') == std::string::npos) {
                        byAtlas[std::max(m.lightmapAtlas, 0)].push_back(&m);
                    }
                }
                if (byAtlas.empty()) {
                    return Error::make(ErrorCode::NotFound, "lightmap: no mesh of {} carries '{}'", o->stage, o->uv);
                }
                auto clear = gpu::ComputeKernel::create(library, "athenea/usd/lightmap", "lightmapClear");
                auto coverKernel = gpu::ComputeKernel::create(library, "athenea/usd/lightmap", "lightmapCover");
                auto flag = gpu::ComputeKernel::create(library, "athenea/usd/lightmap", "lightmapFlag");
                auto raysKernel = gpu::ComputeKernel::create(library, "athenea/usd/lightmap", "lightmapRays");
                auto scatter = gpu::ComputeKernel::create(library, "athenea/usd/lightmap", "lightmapScatter");
                auto dilate = gpu::ComputeKernel::create(library, "athenea/usd/lightmap", "lightmapDilate");
                auto guide = gpu::ComputeKernel::create(library, "athenea/usd/lightmap", "lightmapGuides");
                auto clearGuides = gpu::ComputeKernel::create(library, "athenea/usd/lightmap", "lightmapClearGuides");
                auto mask = gpu::ComputeKernel::create(library, "athenea/usd/lightmap", "lightmapMask");
                std::unique_ptr<technique::Denoiser> denoiser;
                auto prefix = gpu::PrefixSum::create(library);
                if (!clear || !coverKernel || !flag || !raysKernel || !scatter || !dilate || !prefix || !guide ||
                    !clearGuides || !mask) {
                    return Error(ErrorCode::ShaderFailure, "lightmap: cannot make its kernels");
                }
                auto renderer = usd::StageRenderer::open(o->stage, context->deviceShared());
                if (!renderer) return std::move(renderer).error();
                if (o->defaultLights) {
                    ATHENEA_TRY((*renderer)->setDefaultLights(true));
                }
                const uint32_t size = o->size;
                const uint32_t texels = size * size;
                for (const auto& [index, list] : byAtlas) {
                    auto owner = deviceBuffer(device, texels, 4, "lightmap.owner");
                    auto points = deviceBuffer(device, texels, 16, "lightmap.points");
                    auto normals = deviceBuffer(device, texels, 16, "lightmap.normals");
                    auto image = deviceBuffer(device, texels, 16, "lightmap.image");
                    auto other = deviceBuffer(device, texels, 16, "lightmap.dilated");
                    auto flags = deviceBuffer(device, texels, 4, "lightmap.flags");
                    auto starts = deviceBuffer(device, texels, 4, "lightmap.starts");
                    auto total = deviceBuffer(device, 1, 4, "lightmap.total");
                    if (!owner || !points || !normals || !image || !other || !flags || !starts || !total) {
                        return Error(ErrorCode::OutOfMemory, "lightmap: cannot hold an atlas");
                    }
                    const auto bindAtlas = [&](rhi::ShaderCursor c, uint32_t count) {
                        c["atlas"]["texels"].setData(texels);
                        c["atlas"]["size"].setData(size);
                        c["atlas"]["count"].setData(count);
                    };
                    {
                        gpu::CommandBatch batch(device);
                        clear->dispatch(batch, {texels, 1, 1}, [&](rhi::ShaderCursor c) {
                            c["owner"].setBinding(owner->rhi());
                            c["image"].setBinding(image->rhi());
                            bindAtlas(c, 0);
                        });
                        ATHENEA_TRY(batch.submit(true));
                    }
                    Atlas made;
                    made.index = index;
                    uint32_t number = 0;
                    for (const usd::StageMesh* m : list) {
                        const geom::GpuPrimvar* uv = m->mesh.primvar(o->uv);
                        const geom::GpuPrimvar* n = m->mesh.primvar("normals");
                        const auto interpolation = [](const geom::GpuPrimvar* p) -> uint32_t {
                            if (p == nullptr) return 0u;
                            if (p->interpolation == geom::Interpolation::Vertex ||
                                p->interpolation == geom::Interpolation::Varying) {
                                return 1u;
                            }
                            return p->interpolation == geom::Interpolation::FaceVarying ? 2u : 0u;
                        };
                        if (interpolation(uv) == 0 || m->mesh.triangles == 0) {
                            continue;
                        }
                        made.meshes.push_back(m->path);
                        gpu::CommandBatch batch(device);
                        coverKernel->dispatch(batch, {m->mesh.triangles, 1, 1}, [&](rhi::ShaderCursor c) {
                            rhi::ShaderCursor p = c["cover"];
                            p["row0"].setData(m->toWorld.data(), sizeof(float) * 4);
                            p["row1"].setData(m->toWorld.data() + 4, sizeof(float) * 4);
                            p["row2"].setData(m->toWorld.data() + 8, sizeof(float) * 4);
                            p["normal0"].setData(m->normalToWorld.data(), sizeof(float) * 4);
                            p["normal1"].setData(m->normalToWorld.data() + 4, sizeof(float) * 4);
                            p["normal2"].setData(m->normalToWorld.data() + 8, sizeof(float) * 4);
                            p["triangles"].setData(m->mesh.triangles);
                            p["size"].setData(size);
                            p["uvInterpolation"].setData(interpolation(uv));
                            p["hasNormals"].setData(interpolation(n));
                            p["mesh"].setData(number);
                            p["offset"].setData(1.0e-3F);
                            c["positions"].setBinding(m->mesh.positions.rhi());
                            c["indices"].setBinding(m->mesh.indices.rhi());
                            c["corners"].setBinding(m->mesh.triangleCorners.rhi());
                            c["uvs"].setBinding(uv->values.rhi());
                            c["normals"].setBinding(n != nullptr ? n->values.rhi() : uv->values.rhi());
                            c["owner"].setBinding(owner->rhi());
                            c["texelPoint"].setBinding(points->rhi());
                            c["texelNormal"].setBinding(normals->rhi());
                        });
                        ATHENEA_TRY(batch.submit(true));
                        ++number;
                    }
                    // The claimed texels, counted and laid out as bake rays.
                    {
                        gpu::CommandBatch batch(device);
                        flag->dispatch(batch, {texels, 1, 1}, [&](rhi::ShaderCursor c) {
                            c["owner"].setBinding(owner->rhi());
                            c["flags"].setBinding(flags->rhi());
                            bindAtlas(c, 0);
                        });
                        ATHENEA_TRY(prefix->apply(batch, *flags, *starts, *total, texels));
                        ATHENEA_TRY(batch.submit(true));
                    }
                    uint32_t count = 0;
                    ATHENEA_TRY(total->read(device, 0, 4, &count));
                    made.texels = count;
                    if (count == 0) {
                        atlases.push_back(std::move(made));
                        continue;
                    }
                    auto rays = deviceBuffer(device, uint64_t{count} * 3, 16, "lightmap.rays");
                    auto texelOf = deviceBuffer(device, count, 4, "lightmap.texelOf");
                    if (!rays || !texelOf) return Error(ErrorCode::OutOfMemory, "lightmap: cannot hold its rays");
                    {
                        gpu::CommandBatch batch(device);
                        raysKernel->dispatch(batch, {texels, 1, 1}, [&](rhi::ShaderCursor c) {
                            c["owner"].setBinding(owner->rhi());
                            c["starts"].setBinding(starts->rhi());
                            c["texelPoint"].setBinding(points->rhi());
                            c["texelNormal"].setBinding(normals->rhi());
                            c["rays"].setBinding(rays->rhi());
                            c["texelOf"].setBinding(texelOf->rhi());
                            bindAtlas(c, count);
                        });
                        ATHENEA_TRY(batch.submit(true));
                    }
                    // THE BAKE, adaptive where --extra-samples asks: a first
                    // pass at every texel, then passes where the light is
                    // noisiest for its cost (StageRenderer::bakeSplitOnDevice).
                    const auto bakeStart = Clock::now();
                    usd::BakeOptions bake;
                    bake.samples = o->samples;
                    bake.bounces = o->bounces;
                    bake.degree = 0;
                    bake.extraSamples = o->extraSamples;
                    bake.passSamples = o->passSamples;
                    bake.maxPasses = o->maxPasses;
                    auto split = (*renderer)->bakeSplitOnDevice(*rays, count, o->time, bake);
                    if (!split) return std::move(split).error();
                    auto baked = (*renderer)->combineBake(*split, *rays);
                    if (!baked) return std::move(baked).error();
                    const double bakeMs = msSince(bakeStart);
                    cli::out("lightmap: atlas %d, %u texels of %u baked (%u paths each, %.1f more on average in %u "
                             "passes, %u bounces) in %.0f ms\n",
                             index, count, texels, o->samples,
                             static_cast<double>(split->extraPoints) * o->passSamples / std::max(count, 1u),
                             split->extraPasses, o->bounces, bakeMs);
                    // The guides: the material at each texel, read by one path.
                    auto guides = (*renderer)->bakePointsOnDevice(*rays, count, o->time, 1, 0, 0,
                                                                  /*transfer=*/true, 0, 0, 0, /*material=*/true);
                    if (!guides) return std::move(guides).error();
                    const uint32_t transferEntries = 1 + technique::transferPlanes(true, 0);
                    const uint32_t stride = transferEntries + technique::bakeMaterialPlanes(true, true);
                    auto albedo = deviceBuffer(device, texels, 16, "lightmap.albedo");
                    auto normal = deviceBuffer(device, texels, 16, "lightmap.normal");
                    auto aux = deviceBuffer(device, uint64_t{texels} * 2, 16, "lightmap.aux");
                    auto denoised = deviceBuffer(device, texels, 16, "lightmap.denoised");
                    if (!albedo || !normal || !aux || !denoised) {
                        return Error(ErrorCode::OutOfMemory, "lightmap: cannot hold the denoiser's guides");
                    }
                    {
                        gpu::CommandBatch batch(device);
                        clearGuides->dispatch(batch, {texels, 1, 1}, [&](rhi::ShaderCursor c) {
                            c["albedoImage"].setBinding(albedo->rhi());
                            c["normalImage"].setBinding(normal->rhi());
                            bindAtlas(c, count);
                        });
                        scatter->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) {
                            c["baked"].setBinding(baked->rhi());
                            c["texelOf"].setBinding(texelOf->rhi());
                            c["image"].setBinding(image->rhi());
                            bindAtlas(c, count);
                        });
                        guide->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) {
                            c["material"].setBinding(guides->rhi());
                            c["materialStride"].setData(stride);
                            c["materialFirst"].setData(transferEntries);
                            c["texelOf"].setBinding(texelOf->rhi());
                            c["texelNormal"].setBinding(normals->rhi());
                            c["albedoImage"].setBinding(albedo->rhi());
                            c["normalImage"].setBinding(normal->rhi());
                            bindAtlas(c, count);
                        });
                        ATHENEA_TRY(batch.submit(true));
                    }
                    // Grown into the margins, a ring a pass, ping-ponged and
                    // left in `target`: before the denoiser, so it sees no
                    // black around an island, and after it.
                    const auto grow = [&](gpu::Buffer& target, uint32_t rings) -> Result<void> {
                        gpu::Buffer* from = &target;
                        gpu::Buffer* into = &*other;
                        for (uint32_t pass = 0; pass < rings; ++pass) {
                            gpu::CommandBatch batch(device);
                            dilate->dispatch(batch, {size, size, 1}, [&](rhi::ShaderCursor c) {
                                c["imageIn"].setBinding(from->rhi());
                                c["image"].setBinding(into->rhi());
                                bindAtlas(c, count);
                            });
                            ATHENEA_TRY(batch.submit(true));
                            std::swap(from, into);
                        }
                        if (from != &target) {
                            gpu::CommandBatch batch(device);
                            batch.encoder()->copyBuffer(target.rhi(), 0, from->rhi(), 0, uint64_t{texels} * 16);
                            batch.markDirty();
                            ATHENEA_TRY(batch.submit(true));
                        }
                        return ok();
                    };
                    ATHENEA_TRY(grow(*image, o->dilate));
                    gpu::Buffer* result = &*image;
                    if (!o->noDenoise && technique::denoiserBuilt()) {
                        ATHENEA_TRY(grow(*albedo, o->dilate));
                        ATHENEA_TRY(grow(*normal, o->dilate));
                        {
                            gpu::CommandBatch batch(device);
                            batch.encoder()->copyBuffer(aux->rhi(), 0, albedo->rhi(), 0, uint64_t{texels} * 16);
                            batch.encoder()->copyBuffer(aux->rhi(), uint64_t{texels} * 16, normal->rhi(), 0,
                                                        uint64_t{texels} * 16);
                            batch.markDirty();
                            ATHENEA_TRY(batch.submit(true));
                        }
                        if (!denoiser) {
                            auto made = technique::Denoiser::create(library);
                            if (!made) return std::move(made).error();
                            denoiser = std::make_unique<technique::Denoiser>(std::move(*made));
                        }
                        const auto denoiseStart = Clock::now();
                        ATHENEA_TRY(denoiser->denoise(*image, &*aux, 0, uint64_t{texels} * 16, *denoised, size, size));
                        cli::out("lightmap: atlas %d denoised (%s) in %.0f ms\n", index,
                                 denoiser->description().c_str(), msSince(denoiseStart));
                        {
                            gpu::CommandBatch batch(device);
                            mask->dispatch(batch, {texels, 1, 1}, [&](rhi::ShaderCursor c) {
                                c["owner"].setBinding(owner->rhi());
                                c["image"].setBinding(denoised->rhi());
                                bindAtlas(c, count);
                            });
                            ATHENEA_TRY(batch.submit(true));
                        }
                        ATHENEA_TRY(grow(*denoised, o->dilate));
                        result = &*denoised;
                    }
                    gpu::Buffer* from = result;
                    auto pixels = from->readAll<float>(device);
                    if (!pixels) return std::move(pixels).error();
                    const fs::path file = out / ("lightmap_" + std::to_string(index) + ".exr");
                    ATHENEA_TRY(io::writeExr(file, size, size, *pixels, {}, true));
                    cli::out("lightmap: wrote %s (%u meshes)\n", file.string().c_str(),
                             static_cast<unsigned>(made.meshes.size()));
                    atlases.push_back(std::move(made));
                }
                return ok();
            }();
        });
        if (!ran) {
            cli::fail(ran.error());
        }
        if (!inside) {
            cli::fail(inside.error());
        }
        // THE STAGE THAT SHOWS THEM: the source as a sublayer, every baked
        // mesh unlit and bound to its atlas, read at its own coordinates.
        const fs::path layer = out / "lightmapped.usda";
        std::ofstream file(layer);
        file << "#usda 1.0\n# Copyright (c) 2026 jesus luque.\n(\n    subLayers = [\n        @"
             << fs::absolute(o->stage).string() << "@\n    ]\n)\n\n";
        OverTree root;
        for (const Atlas& a : atlases) {
            for (const std::string& path : a.meshes) {
                OverTree* node = &root;
                for (const std::string& part : componentsOf(path)) {
                    node = &node->children[part];
                }
                node->atlas = a.index;
            }
        }
        writeOvers(file, root, 0, o->uv);
        file << "\ndef Scope \"AtheneaLightmaps\"\n{\n";
        for (const Atlas& a : atlases) {
            if (a.texels == 0) {
                continue;
            }
            const std::string name = "atlas" + std::to_string(a.index);
            const std::string base = "/AtheneaLightmaps/" + name;
            file << "    def Material \"" << name << "\"\n    {\n"
                 << "        token outputs:surface.connect = <" << base << "/Surface.outputs:surface>\n"
                 << "        def Shader \"Surface\"\n        {\n"
                 << "            uniform token info:id = \"UsdPreviewSurface\"\n"
                 << "            color3f inputs:diffuseColor = (0, 0, 0)\n"
                 << "            int inputs:useSpecularWorkflow = 1\n"
                 << "            color3f inputs:specularColor = (0, 0, 0)\n"
                 << "            color3f inputs:emissiveColor.connect = <" << base << "/Light.outputs:rgb>\n"
                 << "            token outputs:surface\n        }\n"
                 << "        def Shader \"Light\"\n        {\n"
                 << "            uniform token info:id = \"UsdUVTexture\"\n"
                 << "            asset inputs:file = @./lightmap_" << a.index << ".exr@\n"
                 << "            token inputs:sourceColorSpace = \"raw\"\n"
                 << "            float2 inputs:st.connect = <" << base << "/Coordinates.outputs:result>\n"
                 << "            float3 outputs:rgb\n        }\n"
                 << "        def Shader \"Coordinates\"\n        {\n"
                 << "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
                 << "            string inputs:varname = \"" << o->uv << "\"\n"
                 << "            float2 outputs:result\n        }\n    }\n";
        }
        file << "}\n";
        file.close();
        cli::out("lightmap: wrote %s in %.1f s\n", layer.string().c_str(), msSince(started) / 1000.0);
    });
}

}   // namespace athenea::cli
