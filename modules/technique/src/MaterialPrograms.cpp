// Copyright (c) 2026 jesus luque.
#include "athenea/technique/MaterialPrograms.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <functional>
#include <map>
#include <set>

#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::technique {

namespace {

/// Asking a sample's material whether it is there at all. A UsdPreviewSurface
/// with an `opacityThreshold` has opacity 0 or 1; one without has what its
/// opacity says, which is coverage: the raster route cuts by a pixel's lot
/// (a hash, so a frame is one image), the path tracer only what is fully
/// gone, and draws its own lot a sample. Only rows flagged as cutouts pay.
const char* kCutout = R"(
uint lotHash(uint input) {
    const uint state = input * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// A pixel's lot for one surface: the pixel, the instance and the triangle
/// folded through the hash, so a frame is still one image -- and every layer
/// of a stack draws a lot of its own, as the path tracer's samples do. One
/// lot a pixel for every layer made the layers' coverage one coverage: two
/// cards of opacity one half in a row showed what was behind them half the
/// time instead of a quarter, and a feathered belly dozens of cards deep
/// showed the background through a card's soft edge wherever the frontmost
/// card's did.
float pixelLot(uint2 pixel, uint4 seen) {
    uint word = lotHash(pixel.y * 65536u + pixel.x);
    word = lotHash(word + seen.x);
    word = lotHash(word + seen.y);
    // Never exactly 0: an opacity of 0 always cuts, one of 1 never does.
    return (float(word >> 8) + 0.5) * (1.0 / 16777216.0);
}

public bool materialCuts(CameraParams camera, uint2 pixel, uint4 seen) {
    const Surface s = surfaceAt(camera, pixel.x, pixel.y, seen);
    // An invisible face cuts as a cutout does, whatever its material. Its
    // flag rides in the subset word's top bit: Metal allows a kernel 31
    // buffers, and the path tracer had no room for one more.
    if ((triangleSubsets[s.mesh.firstTriangle + s.triangle] & 0x80000000u) != 0) {
        return true;
    }
    const MaterialRecord m = materials[materialRowOf(s)];
    if ((m.flags & kMaterialCutout) == 0) {
        return false;
    }
    // A UsdPreviewSurface in transparent opacity mode is never cut by the
    // traced route: the path tracer keeps that surface, draws its specular
    // whole and lets (1 - opacity) of the light straight through itself. The
    // raster's lot keeps it with the tracer's probability, max(opacity,
    // 1/20), and its shading weighs the kept sample as the tracer does.
    const bool transparent = (m.flags & kMaterialTransparent) != 0;
    if (lookup.alphaDither == 0 && transparent) {
        return false;
    }
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, pixel.x, pixel.y, s, lookup.time);
    evaluateMaterial(m.function, inputs, m.blob);
    const float kept = transparent ? max(gAtheneaResult.opacity, kTransparentKeep) : gAtheneaResult.opacity;
    return kept < (lookup.alphaDither != 0 ? pixelLot(pixel, seen) : 1.0 / 512.0);
}
)";

/// HOW MANY LOBES A MATERIAL CAN PUSH: its generated source walked from its
/// function down, a BSDF node's call one lobe (UsdPreviewSurface's node five),
/// a function of the graph's as many as its body's calls push. An upper
/// bound -- both sides of a branch count -- and kBuildLobes (16) where the
/// source is not what the generator writes (a call to itself).
uint32_t lobesPushed(const std::string& source, const std::string& entry) {
    static const std::map<std::string, uint32_t> kLeaves = {
        {"athenea_oren_nayar_diffuse_bsdf", 1}, {"athenea_burley_diffuse_bsdf", 1},
        {"athenea_translucent_bsdf", 1},        {"athenea_subsurface_bsdf", 1},
        {"athenea_dielectric_bsdf", 1},         {"athenea_conductor_bsdf", 1},
        {"athenea_generalized_schlick_bsdf", 1}, {"athenea_sheen_bsdf", 1},
        {"athenea_chiang_hair_bsdf", 1},        {"athenea_preview_bsdf", 5},
    };
    constexpr uint32_t kMost = 16;
    // The functions the source defines, by their bodies: a definition opens
    // at the start of a line, and its body is the braces after its
    // parameters (a declaration, ending in ';' first, has none).
    std::map<std::string, std::string> bodies;
    size_t at = 0;
    while (at < source.size()) {
        const size_t end = source.find('\n', at);
        const size_t lineEnd = end == std::string::npos ? source.size() : end;
        const std::string line = source.substr(at, lineEnd - at);
        const size_t lineStart = at;
        at = lineEnd + 1;
        const size_t paren = line.find('(');
        if (line.empty() || line[0] == ' ' || line[0] == '\t' || line[0] == '/' || line[0] == '#' ||
            paren == std::string::npos) {
            continue;
        }
        size_t nameStart = paren;
        while (nameStart > 0 &&
               (std::isalnum(static_cast<unsigned char>(line[nameStart - 1])) || line[nameStart - 1] == '_')) {
            --nameStart;
        }
        if (nameStart == paren || nameStart == 0 || line[nameStart - 1] != ' ') {
            continue;
        }
        const size_t open = source.find_first_of("{;", lineStart + paren);
        if (open == std::string::npos || source[open] == ';') {
            continue;
        }
        size_t close = open;
        for (int depth = 0; close < source.size(); ++close) {
            depth += source[close] == '{' ? 1 : source[close] == '}' ? -1 : 0;
            if (depth == 0) {
                break;
            }
        }
        bodies[line.substr(nameStart, paren - nameStart)] = source.substr(open, close - open);
        at = std::max(at, close + 1);
    }
    std::map<std::string, uint32_t> known(kLeaves);
    std::set<std::string> walking;
    std::function<uint32_t(const std::string&)> pushes = [&](const std::string& name) -> uint32_t {
        if (const auto k = known.find(name); k != known.end()) {
            return k->second;
        }
        const auto body = bodies.find(name);
        if (body == bodies.end()) {
            return 0;
        }
        if (!walking.insert(name).second) {
            return kMost;
        }
        uint64_t sum = 0;
        const std::string& text = body->second;
        for (size_t k = 0; k < text.size(); ++k) {
            if (!(std::isalpha(static_cast<unsigned char>(text[k])) || text[k] == '_') ||
                (k > 0 && (std::isalnum(static_cast<unsigned char>(text[k - 1])) || text[k - 1] == '_'))) {
                continue;
            }
            size_t e = k;
            while (e < text.size() && (std::isalnum(static_cast<unsigned char>(text[e])) || text[e] == '_')) {
                ++e;
            }
            if (e < text.size() && text[e] == '(') {
                sum += pushes(text.substr(k, e - k));
            }
            k = e;
        }
        walking.erase(name);
        const uint32_t total = static_cast<uint32_t>(std::min<uint64_t>(sum, kMost));
        known[name] = total;
        return total;
    };
    return pushes(entry);
}

}   // namespace

Result<MaterialPrograms> MaterialPrograms::create(gpu::ShaderLibrary& library) {
    MaterialPrograms programs;
    programs.library_ = &library;
    ATHENEA_TRY(programs.setModules({}));
    return programs;
}

Result<void> MaterialPrograms::setModules(std::span<const material::CompiledMaterial> modules,
                                          std::span<const material::CompiledMaterial> opacities) {
    std::string signature;
    for (const material::CompiledMaterial& m : modules) {
        signature += m.module + ";";
    }
    for (const material::CompiledMaterial& m : opacities) {
        signature += m.module + ",";
    }
    if (signature == signature_) {
        return ok();
    }
    for (const material::CompiledMaterial& m : modules) {
        auto loaded = library_->loadSource(m.module, m.source, {});
        if (!loaded) return std::move(loaded).error();
    }
    for (const material::CompiledMaterial& m : opacities) {
        if (m.module.empty()) {
            continue;
        }
        auto loaded = library_->loadSource(m.module, m.source, {});
        if (!loaded) return std::move(loaded).error();
    }
    std::string source = "__exported import athenea.material.material_runtime;\n"
                         "__exported import athenea.technique.material_lookup;\n";
    for (const material::CompiledMaterial& m : modules) {
        source += "import " + m.module + ";\n";
    }
    for (const material::CompiledMaterial& m : opacities) {
        if (!m.module.empty()) {
            source += "import " + m.module + ";\n";
        }
    }
    source += "\npublic void evaluateMaterial(uint function, MaterialInputs inputs, uint blob) {\n    switch (function) {\n";
    for (size_t k = 0; k < modules.size(); ++k) {
        source += "    case " + std::to_string(k + 1) + ": " + modules[k].function + "(inputs, blob); break;\n";
    }
    source += "    default: atheneaFallbackMaterial(inputs); break;\n    }\n}\n";
    // The lobe stacks sized to what these materials push (material_runtime's
    // kBuildLobes, lobes' kMaxLobes): never past the device's own counts. The
    // fallback pushes one.
    uint32_t lobes = 1;
    for (const material::CompiledMaterial& m : modules) {
        lobes = std::max(lobes, lobesPushed(m.source, m.function));
    }
    const std::string count = std::to_string(lobes);
    const auto atMost = [&](const char* most) {
        return "(" + count + "u < " + most + " ? " + count + "u : " + most + ")";
    };
    source += "#ifdef ATHENEA_BUILD_LOBES\n"
              "export static const uint kBuildLobes = " + atMost("uint(ATHENEA_BUILD_LOBES)") + ";\n"
              "export static const uint kMaxLobes = " + atMost("uint(ATHENEA_MAX_LOBES)") + ";\n"
              "#else\n"
              "export static const uint kBuildLobes = " + atMost("16u") + ";\n"
              "export static const uint kMaxLobes = " + atMost("8u") + ";\n"
              "#endif\n";
    // WHAT A SHADOW RAY ASKS A CUT-OUT: its opacity alone, from the variant
    // compiled without the BSDF (ClosureVariant::Opacity), so asking costs
    // the compiler a small function a cut-out and not a second copy of every
    // material. A module no cut-out uses has none, and is opaque.
    source += "\npublic float evaluateOpacity(uint function, MaterialInputs inputs, uint blob) {\n    switch (function) {\n";
    for (size_t k = 0; k < opacities.size() && k < modules.size(); ++k) {
        if (!opacities[k].module.empty()) {
            source += "    case " + std::to_string(k + 1) + ": " + opacities[k].function +
                      "(inputs, blob); return gAtheneaOpacity;\n";
        }
    }
    source += "    default: return 1.0;\n    }\n}\n";
    source += kCutout;
    // Named after what it dispatches to: one module per distinct set.
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : signature) {
        hash = (hash ^ c) * 1099511628211ULL;
    }
    std::array<char, 40> name{};
    std::snprintf(name.data(), name.size(), "athenea_materials_%016llx", static_cast<unsigned long long>(hash));
    auto loaded = library_->loadSource(name.data(), source, {});
    if (!loaded) return std::move(loaded).error();
    module_ = name.data();
    signature_ = signature;
    return ok();
}

void bindScene(rhi::ShaderCursor cursor, const world::GpuScene& scene) {
    cursor["positions"].setBinding(scene.positions().rhi());
    cursor["indices"].setBinding(scene.indices().rhi());
    cursor["meshes"].setBinding(scene.meshRecords().rhi());
    cursor["instances"].setBinding(scene.instanceRecords().rhi());
    cursor["primvarRecords"].setBinding(scene.primvarRecords().rhi());
    cursor["primvarValues"].setBinding(scene.primvarValues().rhi());
    cursor["primvarSlots"].setBinding(scene.primvarSlots().rhi());
    cursor["triangleCorners"].setBinding(scene.triangleCorners().rhi());
    cursor["triangleFaces"].setBinding(scene.triangleFaces().rhi());
}

void bindMaterialFrame(rhi::ShaderCursor cursor, const MaterialFrame& frame, const render::Projection& projection) {
    bindScene(cursor, *frame.scene);
    cursor["materials"].setBinding(frame.records->rhi());
    cursor["triangleSubsets"].setBinding(frame.scene->triangleSubsets().rhi());
    cursor["subsetRows"].setBinding(frame.scene->subsetRows().rhi());
    cursor["gMaterialBlob"].setBinding(frame.blob->rhi());
    frame.textures->bind(cursor["gTextures"]);
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    cursor["toWorld"]["row0"].setData(toWorld.data(), sizeof(float) * 4);
    cursor["toWorld"]["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
    cursor["toWorld"]["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
    cursor["lookup"]["time"].setData(frame.time);
    cursor["lookup"]["alphaDither"].setData(frame.alphaDither);
}

}   // namespace athenea::technique
