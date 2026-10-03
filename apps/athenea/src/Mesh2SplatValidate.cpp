// Copyright (c) 2026 jesus luque.
//
// `athenea mesh2splat --validate`: Mesh2SplatValidate.h says what it is. The
// processor here opens files, writes them, keeps the table and counts; every
// pixel is the device's (athenea/usd/m2s_validate.slang, the Measure effect).
#include "Mesh2SplatValidate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>

#include "aofx/Effect.h"
#include "athenea/aofx/EffectRender.h"
#include "athenea/core/Hash.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu_host/Context.h"
#include "athenea/gpu_host/ImageStorage.h"
#include "athenea/image/Image.h"
#include "athenea/io/Exr.h"
#include "athenea/io/Png.h"
#include "athenea/usd/MeshStage.h"
#include "athenea/usd/StageRenderer.h"

namespace athenea::cli {
namespace {

namespace fs = std::filesystem;

/// A slang-rhi view of an image's own pixels (CmdMesh2Splat's, again).
Result<gpu::Buffer> viewOf(gpu_host::Context& context, const image::ImagePtr& image, const char* label) {
    gpu_host::ImageStorage* storage = context.sharedStorage();
    if (storage == nullptr) {
        return Error(ErrorCode::DeviceFailure, "the AOFX host has no device image storage");
    }
    const uint64_t buffer = storage->bufferFor(image->address());
    if (buffer == 0) {
        return Error::make(ErrorCode::DeviceFailure, "the {} picture is not on the device", label);
    }
    return context.renderView(buffer, image->sizeBytes(), 16, label);
}

/// A frame as a picture the Measure effect reads: its rows copied as they are.
Result<image::ImagePtr> pictureOf(const usd::StageImage& frame) {
    auto image = image::Image::create({0, 0, static_cast<int32_t>(frame.width), static_cast<int32_t>(frame.height)});
    if (!image) return std::move(image).error();
    auto floats = (*image)->floats();
    const auto stride = static_cast<size_t>((*image)->stride());
    for (uint32_t y = 0; y < frame.height; ++y) {
        std::copy_n(frame.rgba.begin() + static_cast<std::ptrdiff_t>(size_t{y} * frame.width * 4),
                    size_t{frame.width} * 4, floats.begin() + static_cast<std::ptrdiff_t>(size_t{y} * stride * 4));
    }
    return *image;
}

/// A name a file can have: the material prim's last element.
std::string fileNameOf(const std::string& material) {
    if (material.empty()) {
        return "no_material";
    }
    std::string name = material.substr(material.find_last_of('/') + 1);
    for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
            c = '_';
        }
    }
    return name;
}

/// A stage that is the source with `meshes` switched off and `cloud` beside
/// them: nested overs down to each mesh, as USD composes a deactivation.
std::string composedStage(const std::string& source, const std::vector<std::string>& meshes, const std::string& cloud,
                          char upAxis, double metersPerUnit) {
    struct Node {
        std::map<std::string, Node> children;
        bool off = false;
    };
    Node root;
    for (const std::string& mesh : meshes) {
        Node* at = &root;
        std::stringstream parts(mesh.substr(1));
        std::string part;
        while (std::getline(parts, part, '/')) {
            at = &at->children[part];
        }
        at->off = true;
    }
    std::ostringstream out;
    out << "#usda 1.0\n(\n    metersPerUnit = " << metersPerUnit << "\n    upAxis = \"" << (upAxis == 'z' ? "Z" : "Y")
        << "\"\n    subLayers = [@" << source << "@]\n)\n\n";
    const std::function<void(const Node&, const std::string&)> emit = [&](const Node& node, const std::string& in) {
        for (const auto& [name, child] : node.children) {
            out << in << "over \"" << name << "\"" << (child.off ? " (\n" + in + "    active = false\n" + in + ")" : "")
                << "\n" << in << "{\n";
            emit(child, in + "    ");
            out << in << "}\n";
        }
    };
    emit(root, "");
    // Typeless, so the reference gives it the ParticleField it is.
    out << "def \"AtheneaValidateCloud\" (\n    prepend references = @" << cloud
        << "@</World/Splats>\n)\n{\n}\n";
    return out.str();
}

struct Measured {
    double relMse = 0.0;
    double p99 = 0.0;
    double mean[3] = {0.0, 0.0, 0.0};
    double reference[3] = {0.0, 0.0, 0.0};
};

std::string jsonOf(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

}   // namespace

Result<void> validateConversion(const ValidateJob& job, gpu_host::Context& context, gpu::ShaderLibrary& library,
                                aofx::Effect& measure, const ValidateConvert& convert) {
    const fs::path dir(job.directory);
    std::error_code made;
    fs::create_directories(dir / "clouds", made);
    if (made) {
        return Error::make(ErrorCode::InvalidArgument, "--validate {}: {}", job.directory, made.message());
    }
    const fs::path source = fs::absolute(job.stage);
    std::string camera = job.camera;
    if (camera.empty()) {
        auto first = usd::stageFirstCamera(source);
        if (!first) return std::move(first).error();
        if (first->empty()) {
            return Error::make(ErrorCode::InvalidArgument,
                               "--validate: '{}' has no camera; name one (--validate-camera)", job.stage);
        }
        camera = *first;
    }
    auto groups = usd::stageMaterialGroups(source, job.prim, job.hidden, job.time);
    if (!groups) return std::move(groups).error();
    if (!job.materials.empty()) {
        std::erase_if(*groups, [&](const usd::MaterialGroup& g) {
            return std::none_of(job.materials.begin(), job.materials.end(), [&](const std::string& want) {
                return want == g.material || want == fileNameOf(g.material);
            });
        });
    }
    if (groups->empty()) {
        return Error(ErrorCode::NotFound, "--validate: no material to measure");
    }
    auto meshStage = usd::MeshStage::open(source);
    if (!meshStage) return std::move(meshStage).error();
    const char upAxis = meshStage->upAxis();
    const double metersPerUnit = meshStage->metersPerUnit();
    std::printf("validate: %zu materials, %ux%u through %s\n", groups->size(), job.width, job.height,
                camera.c_str());

    gpu::Device& device = library.device();
    const uint32_t w = job.width, h = job.height, pixels = w * h;
    // THE GT, ONCE: path traced and kept beside the rest, and read again by
    // the next validation of the same stage at the same size.
    usd::StageImage gt;
    usd::StageImage meshFrame;
    std::vector<float> matte;
    uint32_t layers = 0;
    Result<void> inside = ok();
    auto ran = context.run([&] {
        inside = [&]() -> Result<void> {
            const fs::path gtFile = dir / "gt.exr";
            if (auto kept = io::readExr(gtFile); kept && kept->width == w && kept->height == h) {
                gt = {w, h, std::move(kept->rgba)};
                std::printf("validate: GT read from %s\n", gtFile.string().c_str());
            } else {
                auto renderer = usd::StageRenderer::open(source, context.deviceShared());
                if (!renderer) return std::move(renderer).error();
                (*renderer)->setPathSamples(std::min(job.gtPaths, 64u));
                (*renderer)->setPathTotal(job.gtPaths);
                (*renderer)->setPathBounces(job.gtBounces);
                auto traced = (*renderer)->render(camera, job.time, w, h, "rt");
                if (!traced) return std::move(traced).error();
                gt = std::move(*traced);
                ATHENEA_TRY(io::writeExr(gtFile, w, h, gt.rgba, {}, false));
                std::printf("validate: GT path traced, %u paths a pixel, into %s\n", job.gtPaths,
                            gtFile.string().c_str());
            }
            // The stage as meshes, rasterised, with the matte that says whose
            // each pixel is.
            auto renderer = usd::StageRenderer::open(source, context.deviceShared());
            if (!renderer) return std::move(renderer).error();
            (*renderer)->requestOutputs({"CryptoObject00", "CryptoObject01", "CryptoObject02"});
            auto raster = (*renderer)->render(camera, job.time, w, h, "raster");
            if (!raster) return std::move(raster).error();
            meshFrame = std::move(*raster);
            ATHENEA_TRY(io::writeExr(dir / "mesh.exr", w, h, meshFrame.rgba, {}, false));
            for (const char* aov : {"CryptoObject00", "CryptoObject01", "CryptoObject02"}) {
                auto bytes = (*renderer)->mappedOutput(aov);
                if (!bytes) return std::move(bytes).error();
                if (bytes->size() < size_t{pixels} * 16) {
                    return Error::make(ErrorCode::DeviceFailure, "--validate: the frame kept no {}", aov);
                }
                const size_t at = matte.size();
                matte.resize(at + size_t{pixels} * 4);
                std::memcpy(matte.data() + at, bytes->data(), size_t{pixels} * 16);
                ++layers;
            }
            return ok();
        }();
    });
    if (!ran) return std::move(ran).error();
    ATHENEA_TRY(inside);

    auto gtPicture = pictureOf(gt);
    if (!gtPicture) return std::move(gtPicture).error();
    auto meshPicture = pictureOf(meshFrame);
    if (!meshPicture) return std::move(meshPicture).error();

    struct Row {
        std::string material;
        uint32_t    meshes = 0;
        uint32_t    splats = 0;
        double      cover = 0.0;
        Measured    cloud;
        Measured    mesh;
        std::string error;
    };
    std::vector<Row> rows;
    for (const usd::MaterialGroup& group : *groups) {
        Row row;
        row.material = group.material.empty() ? "(none)" : group.material;
        row.meshes = static_cast<uint32_t>(group.meshes.size());
        const std::string name = fileNameOf(group.material);
        const fs::path cloud = fs::absolute(dir / "clouds" / (name + ".usdc"));
        // ONLY THIS MATERIAL'S MESHES CONVERTED: every other mesh of the stage
        // is left out of the conversion, and stays a mesh in the frame.
        std::vector<std::string> hidden = job.hidden;
        for (const usd::MaterialGroup& other : *groups) {
            if (&other == &group) continue;
            for (const std::string& mesh : other.meshes) {
                if (!std::binary_search(group.meshes.begin(), group.meshes.end(), mesh)) {
                    hidden.push_back(mesh);
                }
            }
        }
        std::sort(hidden.begin(), hidden.end());
        hidden.erase(std::unique(hidden.begin(), hidden.end()), hidden.end());
        auto converted = convert(hidden, cloud.string());
        if (!converted) {
            row.error = converted.error().toString();
            std::printf("validate: %s: %s\n", row.material.c_str(), row.error.c_str());
            rows.push_back(row);
            continue;
        }
        row.splats = *converted;
        const fs::path composed = dir / "clouds" / (name + ".usda");
        {
            std::ofstream out(composed);
            out << composedStage(source.string(), group.meshes, cloud.string(), upAxis, metersPerUnit);
        }
        Result<void> measured = ok();
        auto drew = context.run([&] {
            measured = [&]() -> Result<void> {
                auto renderer = usd::StageRenderer::open(composed, context.deviceShared());
                if (!renderer) return std::move(renderer).error();
                auto frame = (*renderer)->render(camera, job.time, w, h, "raster");
                if (!frame) return std::move(frame).error();
                ATHENEA_TRY(io::writeExr(dir / (name + "_gs.exr"), w, h, frame->rgba, {}, false));
                auto cloudPicture = pictureOf(*frame);
                if (!cloudPicture) return std::move(cloudPicture).error();

                // THE MASK: the material's meshes' ids in the mesh frame's matte.
                std::vector<uint32_t> ids;
                for (const std::string& mesh : group.meshes) {
                    ids.push_back(core::cryptomatteId(mesh));
                }
                std::sort(ids.begin(), ids.end());
                auto idBuffer = gpu::Buffer::fromSpan<uint32_t>(device, ids, "validate.ids");
                if (!idBuffer) return std::move(idBuffer).error();
                auto matteBuffer = gpu::Buffer::fromSpan<float>(device, matte, "validate.matte");
                if (!matteBuffer) return std::move(matteBuffer).error();
                gpu::BufferDesc desc;
                desc.bytes = uint64_t{pixels} * 4;
                desc.elementBytes = 4;
                desc.label = "validate.mask";
                auto mask = gpu::Buffer::create(device, desc);
                if (!mask) return std::move(mask).error();
                const std::array<uint32_t, 8> start{0u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u, 0u, 0u, 0u};
                auto counters = gpu::Buffer::fromSpan<uint32_t>(device, start, "validate.counters");
                if (!counters) return std::move(counters).error();
                const auto kernel = [&](const char* entry) {
                    return gpu::ComputeKernel::create(library, "athenea/usd/m2s_validate", entry);
                };
                auto maskKernel = kernel("validateMask");
                if (!maskKernel) return std::move(maskKernel).error();
                auto apply = kernel("validateApply");
                if (!apply) return std::move(apply).error();
                auto triptych = kernel("validateTriptych");
                if (!triptych) return std::move(triptych).error();
                // Every name a kernel declares is bound, as the rule is.
                const auto common = [&](rhi::ShaderCursor c) {
                    c["matte"].setBinding(matteBuffer->rhi());
                    c["ids"].setBinding(idBuffer->rhi());
                    c["mask"].setBinding(mask->rhi());
                    c["counters"].setBinding(counters->rhi());
                    c["params"]["width"].setData(w);
                    c["params"]["height"].setData(h);
                    c["params"]["idCount"].setData(static_cast<uint32_t>(ids.size()));
                    c["params"]["layers"].setData(layers);
                };
                {
                    gpu::CommandBatch batch(device);
                    maskKernel->dispatch(batch, {w, h, 1}, [&](rhi::ShaderCursor c) {
                        common(c);
                        for (const char* unused : {"frame", "masked", "gt", "meshFrame", "cloud", "tripLinear",
                                                   "tripBytes"}) {
                            c[unused].setBinding(mask->rhi());
                        }
                    });
                    ATHENEA_TRY(batch.submit(true));
                }
                std::array<uint32_t, 8> counted{};
                ATHENEA_TRY(counters->read(device, 0, sizeof(counted), counted.data()));
                const double maskSum = static_cast<double>(counted[0]) / 256.0;
                if (!(maskSum > 0.0) || counted[3] <= counted[1] || counted[4] <= counted[2]) {
                    return Error(ErrorCode::NotFound, "the material is not on screen");
                }
                const uint32_t x0 = counted[1], y0 = counted[2], x1 = counted[3], y1 = counted[4];
                const double windowPixels = static_cast<double>(x1 - x0) * (y1 - y0);
                row.cover = maskSum / static_cast<double>(pixels);

                // Each frame times the mask, into a picture the effect reads.
                const auto maskedOf = [&](const image::ImagePtr& picture) -> Result<image::ImagePtr> {
                    auto out = image::Image::create({0, 0, static_cast<int32_t>(w), static_cast<int32_t>(h)});
                    if (!out) return std::move(out).error();
                    auto from = viewOf(context, picture, "validate.frame");
                    if (!from) return std::move(from).error();
                    auto to = viewOf(context, *out, "validate.masked");
                    if (!to) return std::move(to).error();
                    gpu::CommandBatch batch(device);
                    apply->dispatch(batch, {w, h, 1}, [&](rhi::ShaderCursor c) {
                        common(c);
                        c["frame"].setBinding(from->rhi());
                        c["masked"].setBinding(to->rhi());
                        c["params"]["srcStride"].setData(static_cast<uint32_t>(picture->stride()));
                        c["params"]["dstStride"].setData(static_cast<uint32_t>((*out)->stride()));
                        for (const char* unused : {"gt", "meshFrame", "cloud", "tripLinear", "tripBytes"}) {
                            c[unused].setBinding(mask->rhi());
                        }
                    });
                    ATHENEA_TRY(batch.submit(true));
                    (*out)->deviceWrote();
                    return *out;
                };
                auto gtMasked = maskedOf(*gtPicture);
                if (!gtMasked) return std::move(gtMasked).error();
                const auto measureOf = [&](const image::ImagePtr& picture, Measured& m) -> Result<void> {
                    auto masked = maskedOf(picture);
                    if (!masked) return std::move(masked).error();
                    aofx_host::EffectJob effectJob;
                    effectJob.inputs.push_back({"Source", *masked});
                    effectJob.inputs.push_back({"Reference", *gtMasked});
                    effectJob.params.push_back(aofx::ParamValue{
                        "window",
                        {static_cast<double>(x0), static_cast<double>(y0), static_cast<double>(x1),
                         static_cast<double>(y1)},
                        {}});
                    auto rendered = aofx_host::renderEffect(context, measure, effectJob);
                    if (!rendered) return std::move(rendered).error();
                    const std::vector<float>* src = (*rendered)->attached("source");
                    const std::vector<float>* ref = (*rendered)->attached("reference");
                    const std::vector<float>* hdr = (*rendered)->attached("hdr");
                    if (src == nullptr || ref == nullptr || hdr == nullptr || src->size() < 14 || ref->size() < 14 ||
                        hdr->size() < 6) {
                        return Error(ErrorCode::DeviceFailure, "the Measure effect attached no statistics");
                    }
                    // The error is over the whole frame, zero outside the
                    // mask; what it is over the material is that over its
                    // share of the frame.
                    const double all = (static_cast<double>((*hdr)[4]) * 16777216.0 + (*hdr)[5]);
                    m.relMse = all > 0.0 ? static_cast<double>((*hdr)[3]) / maskSum : 0.0;
                    m.p99 = (*hdr)[1];
                    for (int c = 0; c < 3; ++c) {
                        m.mean[c] = static_cast<double>((*src)[8 + c]) / maskSum;
                        m.reference[c] = static_cast<double>((*ref)[8 + c]) / maskSum;
                    }
                    (void)windowPixels;
                    return ok();
                };
                ATHENEA_TRY(measureOf(*cloudPicture, row.cloud));
                ATHENEA_TRY(measureOf(*meshPicture, row.mesh));

                // GT, the mesh and the cloud, side by side over the material's
                // box and a margin.
                const uint32_t pad = 16;
                const uint32_t cx0 = x0 > pad ? x0 - pad : 0, cy0 = y0 > pad ? y0 - pad : 0;
                const uint32_t cw = std::min(w, x1 + pad) - cx0, ch = std::min(h, y1 + pad) - cy0;
                auto gtView = viewOf(context, *gtPicture, "validate.gt");
                auto meshView = viewOf(context, *meshPicture, "validate.mesh");
                auto cloudView = viewOf(context, *cloudPicture, "validate.cloud");
                if (!gtView || !meshView || !cloudView) {
                    return Error(ErrorCode::DeviceFailure, "--validate: a frame is not on the device");
                }
                desc.bytes = uint64_t{cw} * 3 * ch * 16;
                desc.elementBytes = 16;
                desc.label = "validate.tripLinear";
                auto linear = gpu::Buffer::create(device, desc);
                if (!linear) return std::move(linear).error();
                desc.bytes = uint64_t{cw} * 3 * ch * 4;
                desc.elementBytes = 4;
                desc.label = "validate.tripBytes";
                auto bytes = gpu::Buffer::create(device, desc);
                if (!bytes) return std::move(bytes).error();
                {
                    gpu::CommandBatch batch(device);
                    triptych->dispatch(batch, {cw * 3, ch, 1}, [&](rhi::ShaderCursor c) {
                        common(c);
                        c["frame"].setBinding(gtView->rhi());
                        c["masked"].setBinding(linear->rhi());
                        c["gt"].setBinding(gtView->rhi());
                        c["meshFrame"].setBinding(meshView->rhi());
                        c["cloud"].setBinding(cloudView->rhi());
                        c["tripLinear"].setBinding(linear->rhi());
                        c["tripBytes"].setBinding(bytes->rhi());
                        c["params"]["srcStride"].setData(static_cast<uint32_t>((*gtPicture)->stride()));
                        c["params"]["x0"].setData(cx0);
                        c["params"]["y0"].setData(cy0);
                        c["params"]["cutWidth"].setData(cw);
                        c["params"]["cutHeight"].setData(ch);
                    });
                    ATHENEA_TRY(batch.submit(true));
                }
                auto linearRead = linear->readAll<float>(device);
                if (!linearRead) return std::move(linearRead).error();
                auto bytesRead = bytes->readAll<uint8_t>(device);
                if (!bytesRead) return std::move(bytesRead).error();
                ATHENEA_TRY(io::writeExr(dir / (name + ".exr"), cw * 3, ch, *linearRead, {}, true));
                ATHENEA_TRY(io::writePng(dir / (name + ".png"), cw * 3, ch, *bytesRead));
                return ok();
            }();
        });
        if (!drew) return std::move(drew).error();
        if (!measured) {
            row.error = measured.error().toString();
        }
        if (row.error.empty()) {
            std::printf("validate: %-36s %9u splats  relMSE %8.4f  (mesh raster %8.4f)  p99 %.3f  mean %.4f %.4f "
                        "%.4f  GT %.4f %.4f %.4f\n",
                        row.material.c_str(), row.splats, row.cloud.relMse, row.mesh.relMse, row.cloud.p99,
                        row.cloud.mean[0], row.cloud.mean[1], row.cloud.mean[2], row.cloud.reference[0],
                        row.cloud.reference[1], row.cloud.reference[2]);
        } else {
            std::printf("validate: %-36s %s\n", row.material.c_str(), row.error.c_str());
        }
        rows.push_back(row);
    }

    // THE TABLE, as JSON beside the pictures.
    std::ostringstream json;
    json << "{\n  \"stage\": " << jsonOf(source.string()) << ",\n  \"camera\": " << jsonOf(camera)
         << ",\n  \"width\": " << w << ",\n  \"height\": " << h << ",\n  \"gtPaths\": " << job.gtPaths
         << ",\n  \"materials\": [\n";
    for (size_t k = 0; k < rows.size(); ++k) {
        const Row& r = rows[k];
        const auto triple = [](const double v[3]) {
            std::ostringstream o;
            o << "[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
            return o.str();
        };
        json << "    {\"material\": " << jsonOf(r.material) << ", \"meshes\": " << r.meshes
             << ", \"splats\": " << r.splats << ", \"cover\": " << r.cover;
        if (r.error.empty()) {
            json << ", \"relMSE\": " << r.cloud.relMse << ", \"p99\": " << r.cloud.p99
                 << ", \"mean\": " << triple(r.cloud.mean) << ", \"meanGT\": " << triple(r.cloud.reference)
                 << ", \"meshRelMSE\": " << r.mesh.relMse << ", \"meshMean\": " << triple(r.mesh.mean);
        } else {
            json << ", \"error\": " << jsonOf(r.error);
        }
        json << "}" << (k + 1 < rows.size() ? "," : "") << "\n";
    }
    json << "  ]\n}\n";
    std::ofstream(dir / "validate.json") << json.str();
    std::printf("validate: wrote %s\n", (dir / "validate.json").string().c_str());
    return ok();
}

}   // namespace athenea::cli
