// Copyright (c) 2026 jesus luque.
//
// MaterialX documents through MaterialCompiler into Slang the device compiles:
// the surface shaders USD brings (UsdPreviewSurface, standard_surface,
// OpenPBR, glTF PBR) and an unlit texture graph.
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/material/MaterialCompiler.h"
#include "athenea/material/TextureStore.h"

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>

using namespace athenea;

namespace {

std::unique_ptr<material::MaterialCompiler> compiler(test::Gpu& gpu) {
    std::vector<std::filesystem::path> shaders;
    for (const std::string& path : gpu.device->shaderSearchPaths()) {
        shaders.emplace_back(path);
    }
    auto made = material::MaterialCompiler::create({ATHENEA_MATERIALX_ROOT}, shaders);
    if (!made) FAIL(made.error().toString());
    return std::move(*made);
}

std::string surface(const std::string& node, const std::string& inputs) {
    return "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
           "  <" + node + " name=\"shader\" type=\"surfaceshader\">\n" + inputs +
           "  </" + node + ">\n"
           "  <surfacematerial name=\"material\" type=\"material\">\n"
           "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
           "  </surfacematerial>\n</materialx>\n";
}

}   // namespace

TEST_CASE("a compute kernel's screen derivatives come from its 2x2 quad", "[material][derivatives]") {
    ATHENEA_REQUIRE_GPU(gpu);
    // heighttonormal, and so bump, differences the height across the quad the
    // thread sits in: that costs nothing where the quad's lanes are a 2x2
    // block of pixels, which is what this measures on the device at hand.
    auto kernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/quad_derivatives", "quadDerivatives");
    if (!kernel) FAIL(kernel.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 1, "counts");
    const uint32_t w = 161;   // partial quads at both edges
    const uint32_t h = 99;
    {
        gpu::CommandBatch batch(*gpu->device);
        kernel->dispatch(batch, {w, h, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["counts"].setBinding(counts.rhi());
            cursor["field"]["ax"].setData(2.0F);
            cursor["field"]["ay"].setData(-3.0F);
            cursor["field"]["width"].setData(w);
            cursor["field"]["height"].setData(h);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t differ = 0;
    REQUIRE(counts.read(*gpu->device, 0, sizeof(differ), &differ));
    std::printf("  quad derivatives: %u of %u threads differ from the gradient the field was made with\n", differ,
                w * h);
    CHECK(differ == 0);
}

TEST_CASE("MaterialX surface shaders compile to Slang modules the device loads", "[material][materialx]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto mx = compiler(*gpu);
    struct Case {
        const char* name;
        std::string xml;
    };
    const std::vector<Case> cases{
        {"UsdPreviewSurface",
         surface("UsdPreviewSurface", "    <input name=\"diffuseColor\" type=\"color3\" value=\"0.8, 0.2, 0.1\" />\n"
                                      "    <input name=\"roughness\" type=\"float\" value=\"0.4\" />\n")},
        {"standard_surface", surface("standard_surface", "    <input name=\"coat\" type=\"float\" value=\"0.5\" />\n")},
        {"open_pbr_surface", surface("open_pbr_surface", "")},
        {"gltf_pbr", surface("gltf_pbr", "    <input name=\"metallic\" type=\"float\" value=\"1.0\" />\n")},
        {"unlit texture",
         "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
         "  <image name=\"tex\" type=\"color3\">\n"
         "    <input name=\"file\" type=\"filename\" value=\"wood.png\" colorspace=\"srgb_texture\" />\n"
         "    <input name=\"uaddressmode\" type=\"string\" value=\"clamp\" />\n"
         "  </image>\n"
         "  <surface_unlit name=\"shader\" type=\"surfaceshader\">\n"
         "    <input name=\"emission_color\" type=\"color3\" nodename=\"tex\" />\n"
         "  </surface_unlit>\n"
         "  <surfacematerial name=\"material\" type=\"material\">\n"
         "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
         "  </surfacematerial>\n</materialx>\n"},
    };
    for (const Case& c : cases) {
        SECTION(c.name) {
            auto compiled = mx->compileXml(c.xml);
            if (!compiled) FAIL(compiled.error().toString());
            size_t textures = 0;
            size_t primvars = 0;
            for (const auto& slot : compiled->slots) {
                textures += slot.kind == material::MaterialSlot::Kind::Texture ? 1 : 0;
                primvars += slot.kind == material::MaterialSlot::Kind::Primvar ? 1 : 0;
            }
            std::printf("  %s: %s, %zu lines, %u blob words, %zu textures, %zu primvars\n", c.name,
                        compiled->module.c_str(), static_cast<size_t>(std::count(compiled->source.begin(),
                                                                                 compiled->source.end(), '\n')),
                        compiled->words, textures, primvars);
            if (const char* dump = std::getenv("ATHENEA_TEST_DUMP"); dump != nullptr) {
                std::FILE* f = std::fopen((std::string(dump) + "/" + compiled->module + ".slang").c_str(), "w");
                if (f != nullptr) {
                    std::fwrite(compiled->source.data(), 1, compiled->source.size(), f);
                    std::fclose(f);
                }
            }
            auto loaded = gpu->library->loadSource(compiled->module, compiled->source, {});
            if (!loaded) FAIL(loaded.error().toString());
        }
    }
}

namespace {

std::string replaceAll(std::string text, const std::string& from, const std::string& to) {
    for (size_t at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
    return text;
}

const char* kProbe = R"(
import athenea.material.material_runtime;
import LOBES_MODULE;
import REFERENCE_MODULE;

struct ProbeParams {
    uint count;
    uint referenceBlob;
    uint seed;
    uint pad0;
};

RWStructuredBuffer<float4>  ours;
RWStructuredBuffer<float4>  theirs;
ConstantBuffer<ProbeParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void materialProbe(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    Rng rng = Rng(i, params.seed);
    const float3 n = normalize(sampleUniformSphere(float2(rng.next(), rng.next())));
    const Frame frame = Frame(n, float3(0.3, 0.9, 0.1));
    const float3 v = frame.toWorld(sampleCosineHemisphere(float2(rng.next(), rng.next())));
    const float3 l = frame.toWorld(sampleCosineHemisphere(float2(rng.next(), rng.next())));
    MaterialInputs inputs;
    inputs.positionWorld = float3(0.0);
    inputs.normalWorld = n;
    inputs.tangentWorld = frame.x;
    inputs.bitangentWorld = frame.y;
    inputs.positionObject = float3(0.0);
    inputs.normalObject = n;
    inputs.tangentObject = frame.x;
    inputs.bitangentObject = frame.y;
    inputs.viewPosition = v * 4.0;
    inputs.uvDx = float2(0.0);
    inputs.uvDy = float2(0.0);
    inputs.frame = 0.0;
    inputs.time = 0.0;
    inputs.inside = false;
    inputs.worldFromObject0 = float4(1.0, 0.0, 0.0, 0.0);
    inputs.worldFromObject1 = float4(0.0, 1.0, 0.0, 0.0);
    inputs.worldFromObject2 = float4(0.0, 0.0, 1.0, 0.0);
    inputs.mesh.firstPoint = 0;
    inputs.mesh.points = 0;
    inputs.mesh.firstTriangle = 0;
    inputs.mesh.triangles = 0;
    inputs.mesh.hasNormals = 0;
    inputs.mesh.nodeBase = 0;
    inputs.mesh.slotBase = 0;
    inputs.mesh.subsetBase = 0;
    inputs.mesh.boundsLo = float4(0.0);
    inputs.mesh.boundsHi = float4(0.0);
    inputs.triangle = 0;
    inputs.points = uint3(0);
    inputs.weights = float3(1.0, 0.0, 0.0);
    inputs.displayColor = float4(0.18, 0.18, 0.18, 1.0);
    LOBES_FUNCTION(inputs, 0);
    const LobeStack stack = gAtheneaResult;
    ours[i] = float4(stackEval(stack, v, l), float(stack.count));
    gAtheneaReferenceL = l;
    gAtheneaReferenceResponse = float3(0.0);
    REFERENCE_FUNCTION(inputs, params.referenceBlob);
    theirs[i] = float4(gAtheneaReferenceResponse, 0.0);
}
)";

}   // namespace

TEST_CASE("a surface shader's lobes evaluate as MaterialX's own genglsl closures respond", "[material][materialx]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto mx = compiler(*gpu);
    auto textures = material::TextureStore::create(*gpu->library);
    if (!textures) FAIL(textures.error().toString());
    struct Case {
        const char* name;
        std::string xml;
    };
    const auto closure = [](const std::string& node, const std::string& inputs) {
        return "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n  <" + node + " name=\"b\" type=\"BSDF\">\n" +
               inputs + "  </" + node + ">\n"
               "  <surface name=\"shader\" type=\"surfaceshader\">\n"
               "    <input name=\"bsdf\" type=\"BSDF\" nodename=\"b\" />\n  </surface>\n"
               "  <surfacematerial name=\"material\" type=\"material\">\n"
               "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
               "  </surfacematerial>\n</materialx>\n";
    };
    const std::vector<Case> cases{
        {"oren_nayar_diffuse_bsdf, roughness 0.5",
         closure("oren_nayar_diffuse_bsdf", "    <input name=\"color\" type=\"color3\" value=\"0.6, 0.3, 0.2\" />\n"
                                            "    <input name=\"roughness\" type=\"float\" value=\"0.5\" />\n")},
        {"generalized_schlick_bsdf, alpha 0.0625",
         closure("generalized_schlick_bsdf", "    <input name=\"color0\" type=\"color3\" value=\"0.3, 0.5, 0.2\" />\n"
                                             "    <input name=\"roughness\" type=\"vector2\" value=\"0.0625, 0.0625\" />\n")},
        {"conductor_bsdf, alpha 0.2",
         closure("conductor_bsdf", "    <input name=\"roughness\" type=\"vector2\" value=\"0.2, 0.2\" />\n")},
        {"dielectric_bsdf R, alpha 0.3",
         closure("dielectric_bsdf", "    <input name=\"roughness\" type=\"vector2\" value=\"0.3, 0.3\" />\n")},
        {"sheen_bsdf",
         closure("sheen_bsdf", "    <input name=\"roughness\" type=\"float\" value=\"0.4\" />\n")},
        {"layer: schlick over oren_nayar",
         "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
         "  <oren_nayar_diffuse_bsdf name=\"d\" type=\"BSDF\"><input name=\"color\" type=\"color3\" value=\"0.6, 0.3, 0.2\" /></oren_nayar_diffuse_bsdf>\n"
         "  <generalized_schlick_bsdf name=\"s\" type=\"BSDF\"><input name=\"color0\" type=\"color3\" value=\"0.04, 0.04, 0.04\" /><input name=\"roughness\" type=\"vector2\" value=\"0.1, 0.1\" /></generalized_schlick_bsdf>\n"
         "  <layer name=\"b\" type=\"BSDF\"><input name=\"top\" type=\"BSDF\" nodename=\"s\" /><input name=\"base\" type=\"BSDF\" nodename=\"d\" /></layer>\n"
         "  <surface name=\"shader\" type=\"surfaceshader\"><input name=\"bsdf\" type=\"BSDF\" nodename=\"b\" /></surface>\n"
         "  <surfacematerial name=\"material\" type=\"material\"><input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" /></surfacematerial>\n"
         "</materialx>\n"},
        {"mix and multiply: 0.3 schlick, 0.7 of 0.5 oren_nayar",
         "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
         "  <oren_nayar_diffuse_bsdf name=\"d\" type=\"BSDF\"><input name=\"color\" type=\"color3\" value=\"0.6, 0.3, 0.2\" /></oren_nayar_diffuse_bsdf>\n"
         "  <multiply name=\"m\" type=\"BSDF\"><input name=\"in1\" type=\"BSDF\" nodename=\"d\" /><input name=\"in2\" type=\"float\" value=\"0.5\" /></multiply>\n"
         "  <generalized_schlick_bsdf name=\"s\" type=\"BSDF\"><input name=\"color0\" type=\"color3\" value=\"0.04, 0.04, 0.04\" /><input name=\"roughness\" type=\"vector2\" value=\"0.1, 0.1\" /></generalized_schlick_bsdf>\n"
         "  <mix name=\"b\" type=\"BSDF\"><input name=\"fg\" type=\"BSDF\" nodename=\"s\" /><input name=\"bg\" type=\"BSDF\" nodename=\"m\" /><input name=\"mix\" type=\"float\" value=\"0.3\" /></mix>\n"
         "  <surface name=\"shader\" type=\"surfaceshader\"><input name=\"bsdf\" type=\"BSDF\" nodename=\"b\" /></surface>\n"
         "  <surfacematerial name=\"material\" type=\"material\"><input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" /></surfacematerial>\n"
         "</materialx>\n"},
        {"UsdPreviewSurface, metallic 0.3, clearcoat 0.6",
         surface("UsdPreviewSurface", "    <input name=\"diffuseColor\" type=\"color3\" value=\"0.8, 0.2, 0.1\" />\n"
                                      "    <input name=\"roughness\" type=\"float\" value=\"0.4\" />\n"
                                      "    <input name=\"metallic\" type=\"float\" value=\"0.3\" />\n"
                                      "    <input name=\"clearcoat\" type=\"float\" value=\"0.6\" />\n"
                                      "    <input name=\"clearcoatRoughness\" type=\"float\" value=\"0.2\" />\n")},
        {"UsdPreviewSurface, specular workflow",
         surface("UsdPreviewSurface", "    <input name=\"useSpecularWorkflow\" type=\"integer\" value=\"1\" />\n"
                                      "    <input name=\"specularColor\" type=\"color3\" value=\"0.3, 0.5, 0.2\" />\n"
                                      "    <input name=\"roughness\" type=\"float\" value=\"0.25\" />\n")},
        {"standard_surface, metalness 0.2, coat 0.5, sheen 0.3",
         surface("standard_surface", "    <input name=\"base_color\" type=\"color3\" value=\"0.6, 0.4, 0.3\" />\n"
                                     "    <input name=\"specular_roughness\" type=\"float\" value=\"0.35\" />\n"
                                     "    <input name=\"metalness\" type=\"float\" value=\"0.2\" />\n"
                                     "    <input name=\"coat\" type=\"float\" value=\"0.5\" />\n"
                                     "    <input name=\"sheen\" type=\"float\" value=\"0.3\" />\n")},
        {"open_pbr_surface, coat 0.4, fuzz 0.3",
         surface("open_pbr_surface", "    <input name=\"base_color\" type=\"color3\" value=\"0.7, 0.5, 0.2\" />\n"
                                     "    <input name=\"specular_roughness\" type=\"float\" value=\"0.3\" />\n"
                                     "    <input name=\"coat_weight\" type=\"float\" value=\"0.4\" />\n"
                                     "    <input name=\"fuzz_weight\" type=\"float\" value=\"0.3\" />\n")},
        {"gltf_pbr, metallic 0.5, clearcoat 0.4",
         surface("gltf_pbr", "    <input name=\"base_color\" type=\"color3\" value=\"0.9, 0.6, 0.3\" />\n"
                             "    <input name=\"metallic\" type=\"float\" value=\"0.5\" />\n"
                             "    <input name=\"roughness\" type=\"float\" value=\"0.3\" />\n"
                             "    <input name=\"clearcoat\" type=\"float\" value=\"0.4\" />\n")},
    };
    constexpr uint32_t kSamples = 1u << 16;
    for (const Case& c : cases) {
        SECTION(c.name) {
            auto lobes = mx->compileXml(c.xml);
            auto reference = mx->compileXml(c.xml, {}, material::ClosureVariant::GenglslReference);
            if (!lobes) FAIL(lobes.error().toString());
            if (!reference) FAIL(reference.error().toString());
            if (const char* dump = std::getenv("ATHENEA_TEST_DUMP"); dump != nullptr) {
                for (const auto* m : {&*lobes, &*reference}) {
                    if (std::FILE* f = std::fopen((std::string(dump) + "/" + m->module + ".slang").c_str(), "w")) {
                        std::fwrite(m->source.data(), 1, m->source.size(), f);
                        std::fclose(f);
                    }
                }
            }
            REQUIRE(gpu->library->loadSource(lobes->module, lobes->source, {}));
            // genglsl's reflection pass scales what lies beneath a specular
            // layer by an albedo compensated with the Fresnel at the light's
            // half vector, so its throughput depends on the light; its
            // indirect pass uses the Fresnel at the view, which is what a
            // lobe stack can carry. The reference takes the view's here.
            std::string referenceSource = reference->source;
            referenceSource = replaceAll(
                referenceSource, "float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, safeColor0, safeColor90) * comp;",
                "float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, safeColor0, safeColor90) * "
                "mx_ggx_energy_compensation(NdotV, avgAlpha, mx_compute_fresnel(NdotV, fd));");
            referenceSource = replaceAll(
                referenceSource, "float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;",
                "float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * "
                "mx_ggx_energy_compensation(NdotV, avgAlpha, mx_compute_fresnel(NdotV, fd));");
            // genglsl's subsurface approximation reads screen-space curvature
            // (fwidth), which compute has not; these cases carry no subsurface.
            referenceSource = replaceAll(referenceSource, "float curvature = length(fwidth(N)) / length(fwidth(P));",
                                         "float curvature = 0.0;");
            const std::string referenceModule = reference->module + "_view_throughput";
            if (auto loaded = gpu->library->loadSource(referenceModule, referenceSource, {}); !loaded) {
                FAIL(loaded.error().toString());
            }
            const auto noSlot = [](const std::string&) { return uint32_t{0xFFFFFFFF}; };
            std::vector<float> blob = material::MaterialCompiler::parameters(*lobes, **textures, noSlot);
            const std::vector<float> theirsBlob = material::MaterialCompiler::parameters(*reference, **textures, noSlot);
            const uint32_t referenceBlob = static_cast<uint32_t>(blob.size());
            blob.insert(blob.end(), theirsBlob.begin(), theirsBlob.end());
            REQUIRE((*textures)->commit());
            std::string source = replaceAll(kProbe, "LOBES_MODULE", lobes->module);
            source = replaceAll(source, "REFERENCE_MODULE", referenceModule);
            source = replaceAll(source, "LOBES_FUNCTION", lobes->function);
            source = replaceAll(source, "REFERENCE_FUNCTION", reference->function);
            const std::string probeName = "athenea_probe_" + lobes->module.substr(8) + "_" + reference->module.substr(8);
            if (auto probe = gpu->library->loadSource(probeName, source, {"materialProbe"}); !probe) {
                FAIL(probe.error().toString());
            }
            auto probe = gpu::ComputeKernel::create(*gpu->library, probeName, "materialProbe");
            auto compare = gpu::ComputeKernel::create(*gpu->library, "athenea/test/material_compare", "materialCompare");
            if (!probe) FAIL(probe.error().toString());
            REQUIRE(compare);
            const auto floats = [&](uint64_t count, uint32_t element, const float* data, const char* label) {
                gpu::BufferDesc desc;
                desc.bytes = std::max<uint64_t>(count, 1) * element;
                desc.elementBytes = element;
                desc.label = label;
                auto made = gpu::Buffer::create(*gpu->device, desc, data);
                REQUIRE(made);
                return *made;
            };
            gpu::Buffer blobBuffer = floats(blob.size(), 4, blob.data(), "material.blob");
            gpu::Buffer ours = floats(kSamples, 16, nullptr, "probe.ours");
            gpu::Buffer theirs = floats(kSamples, 16, nullptr, "probe.theirs");
            gpu::Buffer placeholder4 = floats(1, 16, nullptr, "probe.placeholder4");
            gpu::Buffer placeholder = test::uintBuffer(*gpu->device, 4, "probe.placeholder");
            {
                gpu::CommandBatch batch(*gpu->device);
                probe->dispatch(batch, {kSamples, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["gMaterialBlob"].setBinding(blobBuffer.rhi());
                    (*textures)->bind(cursor["gTextures"]);
                    cursor["primvarRecords"].setBinding(placeholder.rhi());
                    cursor["primvarValues"].setBinding(placeholder4.rhi());
                    cursor["primvarSlots"].setBinding(placeholder.rhi());
                    cursor["triangleCorners"].setBinding(placeholder.rhi());
                    cursor["triangleFaces"].setBinding(placeholder.rhi());
                    cursor["ours"].setBinding(ours.rhi());
                    cursor["theirs"].setBinding(theirs.rhi());
                    cursor["params"]["count"].setData(kSamples);
                    cursor["params"]["referenceBlob"].setData(referenceBlob);
                    cursor["params"]["seed"].setData(uint32_t{77});
                });
                REQUIRE(batch.submit(true));
            }
            gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "compare.counts");
            gpu::Buffer worst = floats(1, 16, nullptr, "compare.worst");
            {
                gpu::CommandBatch batch(*gpu->device);
                compare->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["ours"].setBinding(ours.rhi());
                    cursor["theirs"].setBinding(theirs.rhi());
                    cursor["counts"].setBinding(counts.rhi());
                    cursor["worst"].setBinding(worst.rhi());
                    cursor["params"]["count"].setData(kSamples);
                    cursor["params"]["relative"].setData(1e-3F);
                    cursor["params"]["floor"].setData(1e-4F);
                });
                REQUIRE(batch.submit(true));
            }
            uint32_t n[3] = {};
            float w[4] = {};
            REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
            REQUIRE(worst.read(*gpu->device, 0, sizeof(w), w));
            std::printf("  %s: %u lobes; %u of %u samples lit, mean response %.4f; %u components beyond 1e-3, "
                        "worst %.2e (%.5f against %.5f)\n",
                        c.name, n[2], n[1], kSamples, double(w[3]), n[0], double(w[0]), double(w[1]), double(w[2]));
            CHECK(n[1] > kSamples / 2);
            CHECK(n[0] == 0);
        }
    }
}

namespace {

/// One point of a compiled material, its lobe stack read back as a converted
/// gaussian reads it (athenea/material/material_sample), against what the
/// material was authored with: a counter of fields that came back otherwise.
const char* kReadBack = R"(
import athenea.material.material_runtime;
import athenea.material.material_sample;
import LOBES_MODULE;
// The closures' marks, as the bake asks for them.
export static const bool kAtheneaMarkLobes = true;

struct ReadBackParams {
    uint   count;
    uint   seed;
    float  tolerance;
    uint   pad0;
    float4 albedo;     // rgb, the diffuse share
    float4 surface;    // roughness, metallic, coat roughness (-1: none), 0
    float4 tint;       // the tint over the reflection
};

RWStructuredBuffer<uint>       misses;   // [albedo, diffuse share, roughness, metallic, coat roughness, seen, tint]
ConstantBuffer<ReadBackParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void materialReadBack(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    Rng rng = Rng(i, params.seed);
    const float3 n = normalize(sampleUniformSphere(float2(rng.next(), rng.next())));
    const Frame frame = Frame(n, float3(0.3, 0.9, 0.1));
    const float3 v = frame.toWorld(sampleCosineHemisphere(float2(rng.next(), rng.next())));
    MaterialInputs inputs;
    inputs.positionWorld = float3(0.0);
    inputs.normalWorld = n;
    inputs.tangentWorld = frame.x;
    inputs.bitangentWorld = frame.y;
    inputs.positionObject = float3(0.0);
    inputs.normalObject = n;
    inputs.tangentObject = frame.x;
    inputs.bitangentObject = frame.y;
    inputs.viewPosition = v * 4.0;
    inputs.uvDx = float2(0.0);
    inputs.uvDy = float2(0.0);
    inputs.frame = 0.0;
    inputs.time = 0.0;
    inputs.inside = false;
    inputs.worldFromObject0 = float4(1.0, 0.0, 0.0, 0.0);
    inputs.worldFromObject1 = float4(0.0, 1.0, 0.0, 0.0);
    inputs.worldFromObject2 = float4(0.0, 0.0, 1.0, 0.0);
    inputs.mesh.firstPoint = 0;
    inputs.mesh.points = 0;
    inputs.mesh.firstTriangle = 0;
    inputs.mesh.triangles = 0;
    inputs.mesh.hasNormals = 0;
    inputs.mesh.nodeBase = 0;
    inputs.mesh.slotBase = 0;
    inputs.mesh.subsetBase = 0;
    inputs.mesh.boundsLo = float4(0.0);
    inputs.mesh.boundsHi = float4(0.0);
    inputs.triangle = 0;
    inputs.points = uint3(0);
    inputs.weights = float3(1.0, 0.0, 0.0);
    inputs.displayColor = float4(0.18, 0.18, 0.18, 1.0);
    LOBES_FUNCTION(inputs, 0);
    const MaterialSample m = materialSampleOf(gAtheneaResult, false);
    const float t = params.tolerance;
    if (m.albedoSeen < 0.5 || any(abs(m.albedo - params.albedo.xyz) > t)) {
        InterlockedAdd(misses[0], 1u);
    }
    if (abs(m.diffuseShare - params.albedo.w) > t) {
        InterlockedAdd(misses[1], 1u);
    }
    if (m.roughnessSeen < 0.5 || abs(m.roughness - params.surface.x) > t) {
        InterlockedAdd(misses[2], 1u);
    }
    // The metal's share of the base is a share of weights, which the
    // specular over the diffuse moves: only all or nothing is held.
    if ((params.surface.y >= 0.5) != (m.metallic >= 0.5)) {
        InterlockedAdd(misses[3], 1u);
    }
    const bool wantCoat = params.surface.z >= 0.0;
    if ((m.coatSeen > 0.5) != wantCoat || (wantCoat && abs(m.coatRoughness - params.surface.z) > t)) {
        InterlockedAdd(misses[4], 1u);
    }
    InterlockedAdd(misses[5], 1u);
    if (m.reflectionTintSeen < 0.5 || any(abs(m.reflectionTint - params.tint.xyz) > t)) {
        InterlockedAdd(misses[6], 1u);
    }
}
)";

}   // namespace

// A MATERIAL READ BACK INTO A GAUSSIAN'S NUMBERS (StageMaterial::sampled):
// the lobe stack a surface shader builds, reduced by material_sample.slang,
// gives back the base colour times the coat colour over it, the specular's
// or the metal's roughness and the coat's -- whatever the view, since none
// of these depends on it. What a TX bake does at each gaussian of a material
// whose fields the stage reader cannot say.
TEST_CASE("a surface shader's lobes read back as the inputs a gaussian carries", "[material][materialx][readback]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto mx = compiler(*gpu);
    auto textures = material::TextureStore::create(*gpu->library);
    if (!textures) FAIL(textures.error().toString());
    struct Case {
        const char*          name;
        std::string          xml;
        std::array<float, 4> albedo;    // rgb, diffuse share
        std::array<float, 4> surface;   // roughness, metallic, coat roughness (-1: none), 0
        std::array<float, 4> tint;      // the coat's colour over the metal or the specular
    };
    const std::vector<Case> cases{
        {"standard_surface: a dielectric under a tinted coat",
         surface("standard_surface", "    <input name=\"base_color\" type=\"color3\" value=\"0.6, 0.4, 0.3\" />\n"
                                     "    <input name=\"base\" type=\"float\" value=\"0.8\" />\n"
                                     "    <input name=\"specular_roughness\" type=\"float\" value=\"0.35\" />\n"
                                     "    <input name=\"coat\" type=\"float\" value=\"1\" />\n"
                                     "    <input name=\"coat_color\" type=\"color3\" value=\"1, 0.5, 0.25\" />\n"
                                     "    <input name=\"coat_roughness\" type=\"float\" value=\"0.1\" />\n"),
         {0.6F, 0.2F, 0.075F, 1.0F},
         {0.35F, 0.0F, 0.1F, 0.0F},
         {1.0F, 0.5F, 0.25F, 0.0F}},
        {"standard_surface: a pale metal under a gold coat, as the brass example",
         surface("standard_surface", "    <input name=\"base_color\" type=\"color3\" value=\"0.8, 0.8, 0.8\" />\n"
                                     "    <input name=\"base\" type=\"float\" value=\"1\" />\n"
                                     "    <input name=\"metalness\" type=\"float\" value=\"1\" />\n"
                                     "    <input name=\"specular\" type=\"float\" value=\"0\" />\n"
                                     "    <input name=\"specular_roughness\" type=\"float\" value=\"0.3\" />\n"
                                     "    <input name=\"coat\" type=\"float\" value=\"1\" />\n"
                                     "    <input name=\"coat_color\" type=\"color3\" value=\"0.9, 0.6, 0.2\" />\n"
                                     "    <input name=\"coat_roughness\" type=\"float\" value=\"0.05\" />\n"),
         {0.72F, 0.48F, 0.16F, 0.0F},
         {0.3F, 1.0F, 0.05F, 0.0F},
         {0.9F, 0.6F, 0.2F, 0.0F}},
        {"open_pbr_surface: a plain dielectric, no coat",
         surface("open_pbr_surface", "    <input name=\"base_color\" type=\"color3\" value=\"0.3, 0.5, 0.7\" />\n"
                                     "    <input name=\"specular_roughness\" type=\"float\" value=\"0.4\" />\n"),
         {0.3F, 0.5F, 0.7F, 1.0F},
         {0.4F, 0.0F, -1.0F, 0.0F},
         {1.0F, 1.0F, 1.0F, 0.0F}},
    };
    constexpr uint32_t kPoints = 4096;
    for (const Case& c : cases) {
        SECTION(c.name) {
            auto lobes = mx->compileXml(c.xml);
            if (!lobes) FAIL(lobes.error().toString());
            REQUIRE(gpu->library->loadSource(lobes->module, lobes->source, {}));
            const auto noSlot = [](const std::string&) { return uint32_t{0xFFFFFFFF}; };
            std::vector<float> blob = material::MaterialCompiler::parameters(*lobes, **textures, noSlot);
            REQUIRE((*textures)->commit());
            std::string source = replaceAll(kReadBack, "LOBES_MODULE", lobes->module);
            source = replaceAll(source, "LOBES_FUNCTION", lobes->function);
            const std::string name = "athenea_readback_" + lobes->module.substr(8);
            if (auto loaded = gpu->library->loadSource(name, source, {"materialReadBack"}); !loaded) {
                FAIL(loaded.error().toString());
            }
            auto kernel = gpu::ComputeKernel::create(*gpu->library, name, "materialReadBack");
            if (!kernel) FAIL(kernel.error().toString());
            gpu::BufferDesc desc;
            desc.bytes = std::max<uint64_t>(blob.size(), 1) * 4;
            desc.elementBytes = 4;
            desc.label = "readback.blob";
            auto blobBuffer = gpu::Buffer::create(*gpu->device, desc, blob.data());
            REQUIRE(blobBuffer);
            desc.bytes = 16;
            desc.elementBytes = 16;
            desc.label = "readback.placeholder4";
            auto placeholder4 = gpu::Buffer::create(*gpu->device, desc);
            REQUIRE(placeholder4);
            gpu::Buffer placeholder = test::uintBuffer(*gpu->device, 4, "readback.placeholder");
            gpu::Buffer misses = test::uintBuffer(*gpu->device, 7, "readback.misses");
            {
                gpu::CommandBatch batch(*gpu->device);
                kernel->dispatch(batch, {kPoints, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["gMaterialBlob"].setBinding(blobBuffer->rhi());
                    (*textures)->bind(cursor["gTextures"]);
                    cursor["primvarRecords"].setBinding(placeholder.rhi());
                    cursor["primvarValues"].setBinding(placeholder4->rhi());
                    cursor["primvarSlots"].setBinding(placeholder.rhi());
                    cursor["triangleCorners"].setBinding(placeholder.rhi());
                    cursor["triangleFaces"].setBinding(placeholder.rhi());
                    cursor["misses"].setBinding(misses.rhi());
                    cursor["params"]["count"].setData(kPoints);
                    cursor["params"]["seed"].setData(uint32_t{91});
                    cursor["params"]["tolerance"].setData(2e-3F);
                    cursor["params"]["albedo"].setData(c.albedo.data(), sizeof(float) * 4);
                    cursor["params"]["surface"].setData(c.surface.data(), sizeof(float) * 4);
                    cursor["params"]["tint"].setData(c.tint.data(), sizeof(float) * 4);
                });
                REQUIRE(batch.submit(true));
            }
            uint32_t n[7] = {};
            REQUIRE(misses.read(*gpu->device, 0, sizeof(n), n));
            std::printf("  %s: of %u points, %u albedo, %u diffuse share, %u roughness, %u metallic, %u coat and "
                        "%u tint readings off\n",
                        c.name, n[5], n[0], n[1], n[2], n[3], n[4], n[6]);
            CHECK(n[5] == kPoints);
            CHECK(n[0] == 0);
            CHECK(n[1] == 0);
            CHECK(n[2] == 0);
            CHECK(n[3] == 0);
            CHECK(n[4] == 0);
            CHECK(n[6] == 0);
        }
    }
}

// Materials that differ only in their values share a module, and a
// material's name is not a value it has: hdMtlx names the renderable after
// the material prim, and the generator named the surface's variables after
// it, so every material on a stage compiled a module of its own.
TEST_CASE("materials that differ in name and values alone share one module", "[material][materialx][share]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto c = compiler(*gpu);
    const auto named = [&](const std::string& name, const std::string& colour) {
        std::string xml = surface("UsdPreviewSurface",
                                  "    <input name=\"diffuseColor\" type=\"color3\" value=\"" + colour + "\" />\n");
        const std::string from = "name=\"material\"";
        xml.replace(xml.find(from), from.size(), "name=\"" + name + "\"");
        auto compiled = c->compileXml(xml, name, material::ClosureVariant::Lobes);
        if (!compiled) FAIL(compiled.error().toString());
        return std::move(*compiled);
    };
    const material::CompiledMaterial back = named("back", "0.8, 0.1, 0.1");
    const material::CompiledMaterial uvgrid = named("uvgrid", "0.1, 0.8, 0.1");
    std::printf("  'back' is %s, 'uvgrid' is %s\n", back.module.c_str(), uvgrid.module.c_str());
    CHECK(back.module == uvgrid.module);
    CHECK(back.source == uvgrid.source);
}

// THE COMPILER'S DEFINITIONS WIN OVER A DOCUMENT'S.
//
// A document hdMtlx builds carries the libraries of the host's USD, and a
// host may be older than the compiler's (Blender's 1.39.4). The compiler
// used to copy the document and import its own libraries after it, and an
// import skips what is there already: the host's definitions and
// implementations stayed. Here the document brings an implementation of
// standard_surface under the library's own name that draws a blue diffuse
// instead -- what an older host's definition looks like to the compiler --
// and the material must compile exactly as it does from its XML alone, with
// the compiler's libraries.
TEST_CASE("a document's own definitions give way to the compiler's libraries", "[material][materialx][libraries]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto c = compiler(*gpu);
    const std::string xml =
        surface("standard_surface", "    <input name=\"base_color\" type=\"color3\" value=\"0.8, 0.5, 0.2\" />\n");
    auto alone = c->compileXml(xml, "material");
    if (!alone) FAIL(alone.error().toString());

    namespace mx = MaterialX;
    mx::DocumentPtr document = mx::createDocument();
    mx::readFromXmlString(document, xml);
    mx::DocumentPtr decoy = mx::createDocument();
    mx::readFromXmlString(decoy,
                          "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
                          "  <nodegraph name=\"NG_host_standard_surface\">\n"
                          "    <oren_nayar_diffuse_bsdf name=\"diffuse\" type=\"BSDF\">\n"
                          "      <input name=\"color\" type=\"color3\" value=\"0, 0, 1\" />\n"
                          "    </oren_nayar_diffuse_bsdf>\n"
                          "    <surface name=\"shell\" type=\"surfaceshader\">\n"
                          "      <input name=\"bsdf\" type=\"BSDF\" nodename=\"diffuse\" />\n"
                          "    </surface>\n"
                          "    <output name=\"out\" type=\"surfaceshader\" nodename=\"shell\" />\n"
                          "  </nodegraph>\n"
                          "  <implementation name=\"IMPL_standard_surface_surfaceshader_101\"\n"
                          "                  nodedef=\"ND_standard_surface_surfaceshader\"\n"
                          "                  nodegraph=\"NG_host_standard_surface\" />\n"
                          "</materialx>\n");
    document->importLibrary(decoy);
    REQUIRE(document->getImplementation("IMPL_standard_surface_surfaceshader_101"));
    auto given = c->compileDocument(std::shared_ptr<void>(document), "material");
    if (!given) FAIL(given.error().toString());
    std::printf("  from its XML %s, from a document with the host's implementation %s\n", alone->module.c_str(),
                given->module.c_str());
    CHECK(given->module == alone->module);
    CHECK(given->source == alone->source);
}

TEST_CASE("a UsdPreviewSurface's opacity is coverage: no transmission lobe, and the material cuts",
          "[material][materialx][coverage]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto mx = compiler(*gpu);
    // MaterialX's own graph makes an opacity under one a dielectric
    // transmission at the surface's ior -- a sphere of opacity 0 came out a
    // grey lens of the sky. The engine's graph (athenea_usd_preview_surface.mtlx)
    // has no transmission: the lobes are the opaque surface's and `opacity`
    // goes out as the surface's opacity, for the passes to cut by.
    auto compiled = mx->compileXml(surface("UsdPreviewSurface",
                                           "    <input name=\"diffuseColor\" type=\"color3\" value=\"0.8, 0.2, 0.1\" />\n"
                                           "    <input name=\"opacity\" type=\"float\" value=\"0.5\" />\n"));
    if (!compiled) FAIL(compiled.error().toString());
    CHECK(compiled->source.find("transmission_bsdf") == std::string::npos);
    CHECK(compiled->source.find("opacity_presence_out") != std::string::npos);
    // The reference variant keeps MaterialX's graph, as genglsl does.
    auto reference = mx->compileXml(surface("UsdPreviewSurface",
                                            "    <input name=\"opacity\" type=\"float\" value=\"0.5\" />\n"),
                                    {}, material::ClosureVariant::GenglslReference);
    if (!reference) FAIL(reference.error().toString());
    CHECK(reference->source.find("transmission_bsdf") != std::string::npos);
    std::printf("  coverage graph: %zu lines; MaterialX's: %zu lines\n",
                static_cast<size_t>(std::count(compiled->source.begin(), compiled->source.end(), '\n')),
                static_cast<size_t>(std::count(reference->source.begin(), reference->source.end(), '\n')));
}

namespace {

/// The stacks two compiled materials leave at the same shading points, word
/// by word: both evaluated in one thread, the second against the first.
const char* kStackCompare = R"(
import athenea.material.material_runtime;
import NODE_MODULE;
import GRAPH_MODULE;
export static const bool kAtheneaMarkLobes = MARK_LOBES;
// Every lobe a surface can push kept (OpenPBR eleven, Standard Surface
// eight), so none is dropped as the
// lightest and the stack's order is the order they were pushed in; a device
// that gives a thread less stack keeps its own counts.
#ifndef ATHENEA_MAX_LOBES
export static const uint kBuildLobes = 16;
export static const uint kMaxLobes = 12;
#endif

struct StackCompareParams {
    uint  count;
    uint  graphBlob;
    uint  seed;
    float tolerance;
};

// [points, stacks of another count, lobes compared, kind/flags/scatter words that differ, floats beyond the
//  tolerance, marks beyond it, emission components beyond it, opacities beyond it, the worst relative
//  difference (a float's bits), floats that differ at all, the kinds seen (a bit each), the deepest stack]
RWStructuredBuffer<uint>           counts;
ConstantBuffer<StackCompareParams> params;

static uint gBeyond;
static uint gBits;
static float gWorst;

void differ(float a, float b) {
    const float error = abs(a - b) / max(max(abs(a), abs(b)), 1.0e-3);
    // Written so that a NaN on either side counts.
    if (!(error <= params.tolerance)) {
        gBeyond += 1;
    }
    if (asuint(a) != asuint(b)) {
        gBits += 1;
    }
    gWorst = max(gWorst, error);
}
void differ(float2 a, float2 b) { differ(a.x, b.x); differ(a.y, b.y); }
void differ(float3 a, float3 b) { differ(a.x, b.x); differ(a.y, b.y); differ(a.z, b.z); }

[shader("compute")]
[numthreads(64, 1, 1)]
void stackCompare(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    Rng rng = Rng(i, params.seed);
    const float3 n = normalize(sampleUniformSphere(float2(rng.next(), rng.next())));
    const Frame frame = Frame(n, float3(0.3, 0.9, 0.1));
    float3 v = frame.toWorld(sampleCosineHemisphere(float2(rng.next(), rng.next())));
    // A quarter of the points seen from inside, a quarter from behind.
    if ((i & 3) == 3) {
        v = -v;
    }
    MaterialInputs inputs;
    inputs.positionWorld = float3(0.0);
    inputs.normalWorld = n;
    inputs.tangentWorld = frame.x;
    inputs.bitangentWorld = frame.y;
    inputs.positionObject = float3(0.0);
    inputs.normalObject = n;
    inputs.tangentObject = frame.x;
    inputs.bitangentObject = frame.y;
    inputs.viewPosition = v * 4.0;
    inputs.uvDx = float2(0.0);
    inputs.uvDy = float2(0.0);
    inputs.frame = 0.0;
    inputs.time = 0.0;
    inputs.inside = (i & 3) == 2;
    inputs.worldFromObject0 = float4(1.0, 0.0, 0.0, 0.0);
    inputs.worldFromObject1 = float4(0.0, 1.0, 0.0, 0.0);
    inputs.worldFromObject2 = float4(0.0, 0.0, 1.0, 0.0);
    inputs.mesh.firstPoint = 0;
    inputs.mesh.points = 0;
    inputs.mesh.firstTriangle = 0;
    inputs.mesh.triangles = 0;
    inputs.mesh.hasNormals = 0;
    inputs.mesh.nodeBase = 0;
    inputs.mesh.slotBase = 0;
    inputs.mesh.subsetBase = 0;
    inputs.mesh.boundsLo = float4(0.0);
    inputs.mesh.boundsHi = float4(0.0);
    inputs.triangle = 0;
    inputs.points = uint3(0);
    inputs.weights = float3(1.0, 0.0, 0.0);
    inputs.displayColor = float4(0.18, 0.18, 0.18, 1.0);
    NODE_FUNCTION(inputs, 0);
    LobeStack node;
    copyStack(node, gAtheneaResult);
    GRAPH_FUNCTION(inputs, params.graphBlob);

    InterlockedAdd(counts[0], 1u);
    if (node.count != gAtheneaResult.count) {
        InterlockedAdd(counts[1], 1u);
    }
    const uint lobes = min(node.count, gAtheneaResult.count);
    uint words = 0;
    uint kinds = 0;
    gBeyond = 0;
    gBits = 0;
    gWorst = 0.0;
    uint marks = 0;
    for (uint k = 0; k < lobes; ++k) {
        const Lobe a = node.lobes[k];
        const Lobe b = gAtheneaResult.lobes[k];
        words += (a.kind != b.kind ? 1u : 0u) + (a.flags != b.flags ? 1u : 0u) + (a.scatter != b.scatter ? 1u : 0u) +
                 (a.pad0 != b.pad0 ? 1u : 0u);
        kinds |= 1u << min(a.kind, 31u);
        differ(a.weight, b.weight);
        differ(a.roughness, b.roughness);
        differ(a.colour0, b.colour0);
        differ(a.ior, b.ior);
        differ(a.colour1, b.colour1);
        differ(a.exponent, b.exponent);
        differ(a.colour2, b.colour2);
        differ(a.alpha, b.alpha);
        differ(a.normal, b.normal);
        differ(a.tangent, b.tangent);
        // The marks a conversion reads: the tint (pad1, pad2) and the layers over (pad3); pad4 with them.
        const uint before = gBeyond;
        differ(a.pad1, b.pad1);
        differ(a.pad2, b.pad2);
        differ(a.pad3, b.pad3);
        differ(a.pad4, b.pad4);
        marks += gBeyond - before;
    }
    InterlockedAdd(counts[2], lobes);
    InterlockedAdd(counts[3], words);
    InterlockedAdd(counts[4], gBeyond - marks);
    InterlockedAdd(counts[5], marks);
    const uint lobeBeyond = gBeyond;
    differ(node.emission, gAtheneaResult.emission);
    InterlockedAdd(counts[6], gBeyond - lobeBeyond);
    const uint emissionBeyond = gBeyond;
    differ(node.opacity, gAtheneaResult.opacity);
    InterlockedAdd(counts[7], gBeyond - emissionBeyond);
    InterlockedMax(counts[8], asuint(gWorst));
    InterlockedAdd(counts[9], gBits);
    InterlockedOr(counts[10], kinds);
    InterlockedMax(counts[11], lobes);
    if (any(node.emission != float3(0.0))) {
        InterlockedAdd(counts[12], 1u);
    }
    if (node.opacity != 1.0) {
        InterlockedAdd(counts[13], 1u);
    }
}
)";

struct RouteInput {
    const char* name;
    const char* type;
    const char* value;
};
struct RouteCase {
    const char*             name;
    std::vector<RouteInput> inputs;
};
/// What the sets of a comparison reached between them.
struct RouteTotals {
    uint32_t kinds = 0;      ///< the kinds of lobe seen, a bit each
    uint32_t deepest = 0;    ///< the deepest stack
    uint32_t emitting = 0;   ///< points whose surface emits
    uint32_t cut = 0;        ///< points whose surface is not opaque
};

/// A SURFACE'S ONE NODE HELD TO MATERIALX'S GRAPH: the surface `category`
/// compiled by both routes of `route` in one process, each set of inputs
/// evaluated by both in one thread at the same shading points and the two
/// stacks compared word by word (kStackCompare) -- the count, each lobe's
/// kind, flags and scatter, its weight and every parameter, the emission
/// and the opacity -- without the marks and with them (kAtheneaMarkLobes, as
/// a bake builds its stacks), which adds the tint and the layers over.
/// `nodeCall` is the node's call as the generated source spells it, which
/// one route must hold and the other must not.
RouteTotals compareRoutes(test::Gpu* gpu, const char* category,
                          material::SurfaceRoute material::MaterialCompilerOptions::* route, const char* nodeCall,
                          const std::vector<RouteCase>& cases) {
    std::vector<std::filesystem::path> shaders;
    for (const std::string& path : gpu->device->shaderSearchPaths()) {
        shaders.emplace_back(path);
    }
    const auto compiler = [&](material::SurfaceRoute which) {
        material::MaterialCompilerOptions options;
        options.*route = which;
        auto made = material::MaterialCompiler::create({ATHENEA_MATERIALX_ROOT}, shaders, options);
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    };
    auto nodeCompiler = compiler(material::SurfaceRoute::Node);
    auto graphCompiler = compiler(material::SurfaceRoute::Graph);
    auto textures = material::TextureStore::create(*gpu->library);
    if (!textures) FAIL(textures.error().toString());
    constexpr uint32_t kPoints = 1u << 14;
    constexpr float kTolerance = 1e-5F;
    constexpr uint32_t kCounters = 14;
    RouteTotals total;
    for (const RouteCase& c : cases) {
        std::string inputs;
        for (const RouteInput& input : c.inputs) {
            inputs += std::string("    <input name=\"") + input.name + "\" type=\"" + input.type + "\" value=\"" +
                      input.value + "\" />\n";
        }
        const std::string xml = surface(category, inputs);
        auto node = nodeCompiler->compileXml(xml);
        auto graph = graphCompiler->compileXml(xml);
        if (!node) FAIL(node.error().toString());
        if (!graph) FAIL(graph.error().toString());
        // Each route is the route it says: one call of the node and no layer
        // on one, layers and no node on the other.
        REQUIRE(node->source.find(nodeCall) != std::string::npos);
        REQUIRE(node->source.find("athenea_layer_bsdf(") == std::string::npos);
        REQUIRE(graph->source.find(nodeCall) == std::string::npos);
        REQUIRE(graph->source.find("athenea_layer_bsdf(") != std::string::npos);
        REQUIRE(node->module != graph->module);
        // A DOCUMENT THAT CARRIES MATERIALX'S OWN GRAPH -- hdMtlx's does, with
        // the libraries its USD loaded -- still compiles to the node: the
        // compiler's graph takes the place of the document's. Standard
        // Surface's graph names no nodedef (an implementation names it), and
        // was left in the document, so `athenea stage` drew MaterialX's graph
        // while this test, which compiles from XML, compared the node.
        {
            MaterialX::DocumentPtr carried = MaterialX::createDocument();
            MaterialX::readFromXmlString(carried, xml);
            carried->importLibrary(std::static_pointer_cast<MaterialX::Document>(graphCompiler->libraries()));
            auto given = nodeCompiler->compileDocument(std::shared_ptr<void>(carried), "material");
            if (!given) FAIL(given.error().toString());
            INFO(c.name << ", from a document carrying MaterialX's graph");
            CHECK(given->module == node->module);
        }
        REQUIRE(gpu->library->loadSource(node->module, node->source, {}));
        REQUIRE(gpu->library->loadSource(graph->module, graph->source, {}));
        const auto noSlot = [](const std::string&) { return uint32_t{0xFFFFFFFF}; };
        std::vector<float> blob = material::MaterialCompiler::parameters(*node, **textures, noSlot);
        const std::vector<float> graphBlob = material::MaterialCompiler::parameters(*graph, **textures, noSlot);
        const uint32_t graphAt = static_cast<uint32_t>(blob.size());
        blob.insert(blob.end(), graphBlob.begin(), graphBlob.end());
        REQUIRE((*textures)->commit());
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(blob.size(), 1) * 4;
        desc.elementBytes = 4;
        desc.label = "stacks.blob";
        auto blobBuffer = gpu::Buffer::create(*gpu->device, desc, blob.data());
        REQUIRE(blobBuffer);
        desc.bytes = 16;
        desc.elementBytes = 16;
        desc.label = "stacks.placeholder4";
        auto placeholder4 = gpu::Buffer::create(*gpu->device, desc);
        REQUIRE(placeholder4);
        gpu::Buffer placeholder = test::uintBuffer(*gpu->device, 4, "stacks.placeholder");
        for (const bool marks : {false, true}) {
            std::string source = replaceAll(kStackCompare, "NODE_MODULE", node->module);
            source = replaceAll(source, "GRAPH_MODULE", graph->module);
            source = replaceAll(source, "NODE_FUNCTION", node->function);
            source = replaceAll(source, "GRAPH_FUNCTION", graph->function);
            source = replaceAll(source, "MARK_LOBES", marks ? "true" : "false");
            const std::string name = "athenea_stacks_" + node->module.substr(8) + "_" + graph->module.substr(8) +
                                     (marks ? "_marks" : "");
            if (auto loaded = gpu->library->loadSource(name, source, {"stackCompare"}); !loaded) {
                FAIL(loaded.error().toString());
            }
            auto kernel = gpu::ComputeKernel::create(*gpu->library, name, "stackCompare");
            if (!kernel) FAIL(kernel.error().toString());
            gpu::Buffer counts = test::uintBuffer(*gpu->device, kCounters, "stacks.counts");
            {
                gpu::CommandBatch batch(*gpu->device);
                kernel->dispatch(batch, {kPoints, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["gMaterialBlob"].setBinding(blobBuffer->rhi());
                    (*textures)->bind(cursor["gTextures"]);
                    cursor["primvarRecords"].setBinding(placeholder.rhi());
                    cursor["primvarValues"].setBinding(placeholder4->rhi());
                    cursor["primvarSlots"].setBinding(placeholder.rhi());
                    cursor["triangleCorners"].setBinding(placeholder.rhi());
                    cursor["triangleFaces"].setBinding(placeholder.rhi());
                    cursor["counts"].setBinding(counts.rhi());
                    cursor["params"]["count"].setData(kPoints);
                    cursor["params"]["graphBlob"].setData(graphAt);
                    cursor["params"]["seed"].setData(uint32_t{53});
                    cursor["params"]["tolerance"].setData(kTolerance);
                });
                REQUIRE(batch.submit(true));
            }
            uint32_t n[kCounters] = {};
            REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
            float worst = 0.0F;
            std::memcpy(&worst, &n[8], sizeof(worst));
            std::printf("  %s%s: %u points, %u lobes (at most %u a stack, kinds 0x%x); %u counts, %u words, %u floats, "
                        "%u marks, %u emission and %u opacity values off; %u floats differ at all, the worst by %.2e\n",
                        c.name, marks ? ", marked" : "", n[0], n[2], n[11], n[10], n[1], n[3], n[4], n[5], n[6], n[7],
                        n[9], double(worst));
            INFO(c.name << (marks ? ", marked" : ""));
            CHECK(n[0] == kPoints);
            CHECK(n[2] >= kPoints);   // a lobe at least at every point
            CHECK(n[1] == 0);
            CHECK(n[3] == 0);
            CHECK(n[4] == 0);
            CHECK(n[5] == 0);
            CHECK(n[6] == 0);
            CHECK(n[7] == 0);
            total.kinds |= n[10];
            total.deepest = std::max(total.deepest, n[11]);
            total.emitting += n[12];
            total.cut += n[13];
        }
    }
    return total;
}

}   // namespace

// THE ONE NODE HELD TO MATERIALX'S GRAPH. OpenPBR Surface's lobes are built
// by athenea_open_pbr_bsdf, written from MaterialX 1.39.5's nodegraph
// (docs/decisions.md, 16 Nov); a MaterialX whose graph changed would leave
// the node as it is, and nothing else compares the two. Both are compiled
// here (MaterialCompilerOptions::openPbr) and evaluated in one thread at the
// same shading points, and the stacks compared word by word: the count, each
// lobe's kind, flags and scatter, its weight and every parameter, the
// emission and the opacity -- without the marks and with them
// (kAtheneaMarkLobes, as a bake builds its stacks), which adds the tint and
// the layers over. Nineteen parameter sets, each reaching a part of the
// graph: the same the two routes were first compared on as pictures.
TEST_CASE("OpenPBR's one node builds the stack MaterialX's graph builds", "[material][materialx][openpbr]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const std::vector<RouteCase> cases{
        {"a tinted coat with darkening and its own index",
         {{"base_color", "color3", "0.6, 0.03, 0.03"}, {"specular_roughness", "float", "0.5"},
          {"coat_weight", "float", "1"}, {"coat_color", "color3", "0.9, 0.6, 0.2"},
          {"coat_darkening", "float", "1"}, {"coat_roughness", "float", "0.1"}, {"coat_ior", "float", "1.6"}}},
        {"half a coat, anisotropic, over half a metal",
         {{"base_color", "color3", "0.1, 0.4, 0.7"}, {"base_metalness", "float", "0.5"},
          {"coat_weight", "float", "0.5"}, {"coat_roughness", "float", "0.2"},
          {"coat_roughness_anisotropy", "float", "0.7"}, {"coat_darkening", "float", "0.5"},
          {"coat_color", "color3", "0.7, 0.9, 0.8"}}},
        {"fuzz",
         {{"base_color", "color3", "0.15, 0.05, 0.2"}, {"specular_roughness", "float", "0.7"},
          {"fuzz_weight", "float", "1"}, {"fuzz_color", "color3", "0.9, 0.8, 1"},
          {"fuzz_roughness", "float", "0.5"}}},
        {"fuzz over a coat",
         {{"base_color", "color3", "0.2, 0.5, 0.1"}, {"fuzz_weight", "float", "0.6"},
          {"fuzz_color", "color3", "1, 0.9, 0.6"}, {"fuzz_roughness", "float", "0.3"},
          {"coat_weight", "float", "0.8"}, {"coat_roughness", "float", "0.05"}}},
        {"a thin film over a metal",
         {{"base_color", "color3", "0.9, 0.9, 0.9"}, {"base_metalness", "float", "1"},
          {"specular_roughness", "float", "0.15"}, {"thin_film_weight", "float", "0.6"},
          {"thin_film_thickness", "float", "0.5"}, {"thin_film_ior", "float", "1.4"}}},
        {"a thin film of weight one over a plastic",
         {{"base_color", "color3", "0.05, 0.05, 0.05"}, {"specular_roughness", "float", "0.1"},
          {"thin_film_weight", "float", "1"}, {"thin_film_thickness", "float", "0.4"},
          {"thin_film_ior", "float", "1.3"}}},
        {"a thin film over glass",
         {{"base_color", "color3", "1, 1, 1"}, {"transmission_weight", "float", "1"},
          {"specular_roughness", "float", "0"}, {"thin_film_weight", "float", "0.5"},
          {"thin_film_thickness", "float", "0.6"}}},
        {"subsurface 1",
         {{"base_color", "color3", "0.8, 0.8, 0.8"}, {"subsurface_weight", "float", "1"},
          {"subsurface_color", "color3", "0.9, 0.5, 0.4"}, {"subsurface_radius", "float", "0.1"},
          {"subsurface_radius_scale", "color3", "1, 0.5, 0.25"}}},
        {"subsurface 0.5",
         {{"base_color", "color3", "0.2, 0.6, 0.8"}, {"subsurface_weight", "float", "0.5"},
          {"subsurface_color", "color3", "0.9, 0.8, 0.3"}, {"subsurface_scatter_anisotropy", "float", "0.3"}}},
        {"a thin wall's subsurface",
         {{"base_color", "color3", "0.8, 0.8, 0.8"}, {"subsurface_weight", "float", "1"},
          {"subsurface_color", "color3", "0.4, 0.8, 0.3"}, {"subsurface_scatter_anisotropy", "float", "0.4"},
          {"geometry_thin_walled", "boolean", "true"}}},
        {"a thin wall's glass",
         {{"base_color", "color3", "1, 1, 1"}, {"transmission_weight", "float", "1"},
          {"transmission_color", "color3", "0.9, 0.7, 0.9"}, {"specular_roughness", "float", "0"},
          {"geometry_thin_walled", "boolean", "true"}}},
        {"transmission with dispersion",
         {{"base_color", "color3", "1, 1, 1"}, {"transmission_weight", "float", "1"},
          {"specular_roughness", "float", "0"}, {"specular_ior", "float", "1.6"},
          {"transmission_dispersion_scale", "float", "1"},
          {"transmission_dispersion_abbe_number", "float", "20"}}},
        {"transmission with depth and scatter",
         {{"base_color", "color3", "1, 1, 1"}, {"transmission_weight", "float", "1"},
          {"transmission_color", "color3", "0.4, 0.7, 0.9"}, {"transmission_depth", "float", "0.2"},
          {"transmission_scatter", "color3", "0.3, 0.2, 0.1"},
          {"transmission_scatter_anisotropy", "float", "0.2"}, {"specular_roughness", "float", "0.05"}}},
        {"half transmission",
         {{"base_color", "color3", "0.8, 0.3, 0.1"}, {"transmission_weight", "float", "0.5"},
          {"specular_roughness", "float", "0.1"}}},
        {"emission under a tinted coat",
         {{"base_color", "color3", "0.1, 0.1, 0.1"}, {"emission_luminance", "float", "3"},
          {"emission_color", "color3", "0.3, 0.6, 1"}, {"coat_weight", "float", "1"},
          {"coat_color", "color3", "1, 0.5, 0.5"}, {"coat_roughness", "float", "0.1"}}},
        {"opacity 0.5",
         {{"base_color", "color3", "0.7, 0.7, 0.1"}, {"geometry_opacity", "float", "0.5"}}},
        {"an anisotropic metal",
         {{"base_color", "color3", "0.95, 0.8, 0.5"}, {"base_metalness", "float", "1"},
          {"specular_roughness", "float", "0.3"}, {"specular_roughness_anisotropy", "float", "0.8"}}},
        {"half metal with a specular colour and weight",
         {{"base_color", "color3", "0.8, 0.5, 0.3"}, {"base_weight", "float", "0.7"},
          {"base_metalness", "float", "0.5"}, {"specular_weight", "float", "0.5"},
          {"specular_color", "color3", "0.6, 0.8, 1"}, {"specular_roughness", "float", "0.25"},
          {"base_diffuse_roughness", "float", "0.6"}}},
        {"every weight fractional",
         {{"base_color", "color3", "0.6, 0.4, 0.5"}, {"base_weight", "float", "0.9"},
          {"base_metalness", "float", "0.3"}, {"base_diffuse_roughness", "float", "0.4"},
          {"specular_weight", "float", "0.8"}, {"specular_color", "color3", "0.9, 1, 0.8"},
          {"specular_roughness", "float", "0.2"}, {"specular_roughness_anisotropy", "float", "0.3"},
          {"transmission_weight", "float", "0.3"}, {"transmission_color", "color3", "0.8, 0.9, 0.6"},
          {"subsurface_weight", "float", "0.3"}, {"subsurface_color", "color3", "0.9, 0.4, 0.4"},
          {"fuzz_weight", "float", "0.5"}, {"fuzz_color", "color3", "0.8, 0.8, 1"},
          {"fuzz_roughness", "float", "0.4"}, {"coat_weight", "float", "0.7"},
          {"coat_color", "color3", "0.9, 0.8, 0.7"}, {"coat_roughness", "float", "0.1"},
          {"coat_darkening", "float", "0.8"}, {"thin_film_weight", "float", "0.4"},
          {"thin_film_thickness", "float", "0.5"}, {"emission_luminance", "float", "0.5"},
          {"emission_color", "color3", "1, 0.8, 0.6"}, {"geometry_opacity", "float", "0.9"}}},
    };
    const RouteTotals total = compareRoutes(gpu, "open_pbr_surface", &material::MaterialCompilerOptions::openPbr,
                                            "athenea_open_pbr_bsdf(", cases);
    // That the sets reached the graph's parts: a diffuse, a translucent, a
    // dielectric, a Schlick metal and a sheen among the lobes, a stack of
    // nine at least, a surface that emits and one that is not opaque.
    std::printf("  over the %zu sets: kinds 0x%x, the deepest stack %u lobes, %u points emitting, %u not opaque\n",
                cases.size(), total.kinds, total.deepest, total.emitting, total.cut);
    CHECK((total.kinds & 0xDAu) == 0xDAu);   // kLobeOrenNayar, Translucent, Dielectric, Schlick, Sheen
    CHECK(total.deepest >= 9);
    CHECK(total.emitting > 0);
    CHECK(total.cut > 0);
}

// THE SAME FOR AUTODESK STANDARD SURFACE, whose lobes athenea_standard_surface_bsdf
// builds, written from MaterialX 1.39.5's NG_standard_surface_surfaceshader_100
// (docs/decisions.md, 17 Nov). The sets reach each lobe and each combinator
// of that graph: the diffuse, the metal's conductor, the reflection with its
// anisotropy and index, the transmission with every input it has, the two
// subsurfaces a thin wall chooses between, the sheen, the coat with its
// colour, roughness, index and what it does to the colours and roughnesses
// under it, a thin film, emission, opacity -- and the transmission a coat
// spares where no reflection lies between them.
TEST_CASE("Standard Surface's one node builds the stack MaterialX's graph builds",
          "[material][materialx][standardsurface]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const std::vector<RouteCase> cases{
        {"a rough diffuse base",
         {{"base", "float", "0.8"}, {"base_color", "color3", "0.6, 0.3, 0.1"}, {"diffuse_roughness", "float", "0.6"},
          {"specular_roughness", "float", "0.4"}}},
        {"a diffuse base with no reflection over it",
         {{"base", "float", "1"}, {"base_color", "color3", "0.2, 0.5, 0.7"}, {"specular", "float", "0"}}},
        {"an anisotropic metal, its tangent turned",
         {{"base", "float", "1"}, {"base_color", "color3", "0.95, 0.8, 0.5"}, {"metalness", "float", "1"},
          {"specular_roughness", "float", "0.3"}, {"specular_anisotropy", "float", "0.8"},
          {"specular_rotation", "float", "0.2"}}},
        {"half metal with a specular colour and weight",
         {{"base", "float", "0.7"}, {"base_color", "color3", "0.8, 0.5, 0.3"}, {"metalness", "float", "0.5"},
          {"specular", "float", "0.5"}, {"specular_color", "color3", "0.6, 0.8, 1"},
          {"specular_roughness", "float", "0.25"}, {"diffuse_roughness", "float", "0.3"}}},
        {"a reflection with anisotropy and its own index",
         {{"base_color", "color3", "0.05, 0.05, 0.05"}, {"specular", "float", "0.6"},
          {"specular_color", "color3", "1, 0.9, 0.7"}, {"specular_roughness", "float", "0.15"},
          {"specular_IOR", "float", "1.8"}, {"specular_anisotropy", "float", "0.6"},
          {"specular_rotation", "float", "0.4"}}},
        {"glass",
         {{"base_color", "color3", "1, 1, 1"}, {"transmission", "float", "1"},
          {"transmission_color", "color3", "0.9, 0.95, 1"}, {"specular_roughness", "float", "0"},
          {"specular_IOR", "float", "1.52"}}},
        {"transmission with depth and scatter",
         {{"transmission", "float", "1"}, {"transmission_color", "color3", "0.4, 0.7, 0.9"},
          {"transmission_depth", "float", "0.2"}, {"transmission_scatter", "color3", "0.3, 0.2, 0.1"},
          {"transmission_scatter_anisotropy", "float", "0.2"}, {"specular_roughness", "float", "0.05"}}},
        {"transmission with dispersion and extra roughness",
         {{"transmission", "float", "1"}, {"specular_roughness", "float", "0.1"}, {"specular_IOR", "float", "1.6"},
          {"transmission_dispersion", "float", "20"}, {"transmission_extra_roughness", "float", "0.3"},
          {"specular_anisotropy", "float", "0.4"}}},
        {"half transmission",
         {{"base_color", "color3", "0.8, 0.3, 0.1"}, {"transmission", "float", "0.5"},
          {"specular_roughness", "float", "0.1"}}},
        {"glass with no reflection, under a coat at its index",
         {{"transmission", "float", "1"}, {"specular", "float", "0"}, {"specular_IOR", "float", "1.5"},
          {"coat", "float", "1"}, {"coat_IOR", "float", "1.5"}, {"coat_roughness", "float", "0.05"}}},
        {"glass under a coat at another index",
         {{"transmission", "float", "1"}, {"transmission_color", "color3", "0.8, 1, 0.8"},
          {"specular_IOR", "float", "1.45"}, {"coat", "float", "0.8"}, {"coat_IOR", "float", "1.7"},
          {"coat_color", "color3", "1, 0.8, 0.6"}}},
        {"subsurface 1",
         {{"base_color", "color3", "0.8, 0.8, 0.8"}, {"subsurface", "float", "1"},
          {"subsurface_color", "color3", "0.9, 0.5, 0.4"}, {"subsurface_radius", "color3", "1, 0.5, 0.25"},
          {"subsurface_scale", "float", "0.1"}, {"subsurface_anisotropy", "float", "0.3"}}},
        {"subsurface 0.5",
         {{"base_color", "color3", "0.2, 0.6, 0.8"}, {"subsurface", "float", "0.5"},
          {"subsurface_color", "color3", "0.9, 0.8, 0.3"}, {"diffuse_roughness", "float", "0.5"}}},
        {"a thin wall's subsurface",
         {{"base_color", "color3", "0.8, 0.8, 0.8"}, {"subsurface", "float", "0.8"},
          {"subsurface_color", "color3", "0.4, 0.8, 0.3"}, {"thin_walled", "boolean", "true"}}},
        {"a thin wall's glass",
         {{"transmission", "float", "1"}, {"transmission_color", "color3", "0.9, 0.7, 0.9"},
          {"specular_roughness", "float", "0"}, {"thin_walled", "boolean", "true"}}},
        {"sheen",
         {{"base_color", "color3", "0.15, 0.05, 0.2"}, {"specular_roughness", "float", "0.7"},
          {"sheen", "float", "1"}, {"sheen_color", "color3", "0.9, 0.8, 1"}, {"sheen_roughness", "float", "0.5"}}},
        {"sheen over half a subsurface, half transmitting",
         {{"base_color", "color3", "0.5, 0.2, 0.2"}, {"sheen", "float", "0.6"},
          {"sheen_color", "color3", "1, 0.9, 0.6"}, {"sheen_roughness", "float", "0.2"},
          {"subsurface", "float", "0.5"}, {"subsurface_color", "color3", "0.9, 0.6, 0.5"},
          {"transmission", "float", "0.5"}}},
        {"a tinted coat, rough, anisotropic, at its own index",
         {{"base_color", "color3", "0.6, 0.03, 0.03"}, {"specular_roughness", "float", "0.5"},
          {"coat", "float", "1"}, {"coat_color", "color3", "0.9, 0.6, 0.2"}, {"coat_roughness", "float", "0.2"},
          {"coat_IOR", "float", "1.6"}, {"coat_anisotropy", "float", "0.7"}, {"coat_rotation", "float", "0.3"}}},
        {"half a coat that changes the colours and roughnesses under it",
         {{"base_color", "color3", "0.1, 0.4, 0.7"}, {"metalness", "float", "0.5"},
          {"specular_roughness", "float", "0.3"}, {"coat", "float", "0.5"},
          {"coat_color", "color3", "0.7, 0.9, 0.8"}, {"coat_roughness", "float", "0.4"},
          {"coat_affect_color", "float", "0.8"}, {"coat_affect_roughness", "float", "0.9"},
          {"subsurface", "float", "0.4"}, {"subsurface_color", "color3", "0.8, 0.3, 0.2"},
          {"transmission", "float", "0.2"}, {"transmission_extra_roughness", "float", "0.1"}}},
        {"a thin film over a metal",
         {{"base_color", "color3", "0.9, 0.9, 0.9"}, {"metalness", "float", "1"},
          {"specular_roughness", "float", "0.15"}, {"thin_film_thickness", "float", "500"},
          {"thin_film_IOR", "float", "1.4"}}},
        {"a thin film over a plastic",
         {{"base_color", "color3", "0.05, 0.05, 0.05"}, {"specular_roughness", "float", "0.1"},
          {"thin_film_thickness", "float", "400"}, {"thin_film_IOR", "float", "1.3"}}},
        {"emission under a tinted coat",
         {{"base_color", "color3", "0.1, 0.1, 0.1"}, {"emission", "float", "3"},
          {"emission_color", "color3", "0.3, 0.6, 1"}, {"coat", "float", "1"},
          {"coat_color", "color3", "1, 0.5, 0.5"}, {"coat_roughness", "float", "0.1"}}},
        {"opacity 0.5",
         {{"base_color", "color3", "0.7, 0.7, 0.1"}, {"opacity", "color3", "0.5, 0.5, 0.5"}}},
        {"every weight fractional",
         {{"base", "float", "0.9"}, {"base_color", "color3", "0.6, 0.4, 0.5"}, {"diffuse_roughness", "float", "0.4"},
          {"metalness", "float", "0.3"}, {"specular", "float", "0.8"}, {"specular_color", "color3", "0.9, 1, 0.8"},
          {"specular_roughness", "float", "0.2"}, {"specular_anisotropy", "float", "0.3"},
          {"specular_rotation", "float", "0.1"}, {"transmission", "float", "0.3"},
          {"transmission_color", "color3", "0.8, 0.9, 0.6"}, {"transmission_extra_roughness", "float", "0.2"},
          {"subsurface", "float", "0.3"}, {"subsurface_color", "color3", "0.9, 0.4, 0.4"},
          {"sheen", "float", "0.5"}, {"sheen_color", "color3", "0.8, 0.8, 1"}, {"sheen_roughness", "float", "0.4"},
          {"coat", "float", "0.7"}, {"coat_color", "color3", "0.9, 0.8, 0.7"}, {"coat_roughness", "float", "0.1"},
          {"coat_affect_color", "float", "0.5"}, {"coat_affect_roughness", "float", "0.5"},
          {"thin_film_thickness", "float", "300"}, {"emission", "float", "0.5"},
          {"emission_color", "color3", "1, 0.8, 0.6"}, {"opacity", "color3", "0.9, 0.9, 0.9"}}},
        {"every weight fractional, a thin wall",
         {{"base", "float", "0.9"}, {"base_color", "color3", "0.6, 0.4, 0.5"}, {"metalness", "float", "0.3"},
          {"specular", "float", "0.8"}, {"specular_roughness", "float", "0.2"}, {"transmission", "float", "0.3"},
          {"transmission_color", "color3", "0.8, 0.9, 0.6"}, {"subsurface", "float", "0.3"},
          {"subsurface_color", "color3", "0.9, 0.4, 0.4"}, {"sheen", "float", "0.5"},
          {"coat", "float", "0.7"}, {"coat_color", "color3", "0.9, 0.8, 0.7"}, {"thin_walled", "boolean", "true"}}},
        {"weights past one and under zero",
         {{"base", "float", "1.3"}, {"base_color", "color3", "0.6, 0.4, 0.5"}, {"metalness", "float", "1.2"},
          {"specular", "float", "1.4"}, {"transmission", "float", "-0.2"}, {"subsurface", "float", "1.5"},
          {"sheen", "float", "1.2"}, {"coat", "float", "1.5"}, {"coat_color", "color3", "1.4, 0.8, -0.2"}}},
    };
    const RouteTotals total =
        compareRoutes(gpu, "standard_surface", &material::MaterialCompilerOptions::standardSurface,
                      "athenea_standard_surface_bsdf(", cases);
    // That the sets reached the graph's parts: a diffuse, a translucent, a
    // dielectric, a conductor and a sheen among the lobes, a stack of seven
    // (every lobe but the one of the two subsurfaces the wall does not
    // choose), a surface that emits and one that is not opaque.
    std::printf("  over the %zu sets: kinds 0x%x, the deepest stack %u lobes, %u points emitting, %u not opaque\n",
                cases.size(), total.kinds, total.deepest, total.emitting, total.cut);
    CHECK((total.kinds & 0xBAu) == 0xBAu);   // kLobeOrenNayar, Translucent, Dielectric, Conductor, Sheen
    CHECK(total.deepest >= 7);
    CHECK(total.emitting > 0);
    CHECK(total.cut > 0);
}
