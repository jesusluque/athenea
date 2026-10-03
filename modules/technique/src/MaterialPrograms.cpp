// Copyright (c) 2026 jesus luque.
#include "athenea/technique/MaterialPrograms.h"

#include <array>
#include <cstdio>

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
