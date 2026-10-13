// Copyright (c) 2026 jesus luque.
#include "athenea/usd/LightGroups.h"

#include <algorithm>
#include <cstring>

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/usd/sdf/copyUtils.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdVol/particleField3DGaussianSplat.h>
#include <pxr/usd/usdVol/tokens.h>

#include "athenea/core/Log.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/usd/PrimData.h"

namespace athenea::usd {

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

constexpr const char* kTargetPrefix = "athenea:lightState:target:";

/// A float attribute, however it was typed; `fallback` where it is not there.
float floatAttr(const UsdPrim& prim, const char* name, float fallback) {
    const UsdAttribute attr = prim.GetAttribute(TfToken(name));
    VtValue value;
    if (!attr || !attr.Get(&value)) {
        return fallback;
    }
    if (value.IsHolding<float>()) return value.UncheckedGet<float>();
    if (value.IsHolding<double>()) return static_cast<float>(value.UncheckedGet<double>());
    if (value.IsHolding<int>()) return static_cast<float>(value.UncheckedGet<int>());
    return fallback;
}

bool colourAttr(const UsdPrim& prim, const char* name, std::array<float, 3>& into) {
    const UsdAttribute attr = prim.GetAttribute(TfToken(name));
    VtValue value;
    if (!attr || !attr.Get(&value) || !value.IsHolding<GfVec3f>()) {
        return false;
    }
    const GfVec3f v = value.UncheckedGet<GfVec3f>();
    into = {v[0], v[1], v[2]};
    return true;
}

/// The field at `prim` on `stage`, or its first.
UsdVolParticleField3DGaussianSplat fieldOf(const UsdStageRefPtr& stage, const std::string& prim) {
    if (!prim.empty()) {
        return UsdVolParticleField3DGaussianSplat(stage->GetPrimAtPath(SdfPath(prim)));
    }
    for (const UsdPrim& candidate : stage->Traverse()) {
        if (candidate.IsA<UsdVolParticleField3DGaussianSplat>()) {
            return UsdVolParticleField3DGaussianSplat(candidate);
        }
    }
    return UsdVolParticleField3DGaussianSplat();
}

/// The float attribute, or its half twin, as the field holds it.
VtValue eitherOf(const UsdPrim& prim, const TfToken& full, const TfToken& half) {
    VtValue value;
    if (const UsdAttribute attr = prim.GetAttribute(full); attr && attr.Get(&value, UsdTimeCode::EarliestTime()) &&
                                                           !streamOf(value).empty()) {
        return value;
    }
    value = VtValue();
    if (const UsdAttribute attr = prim.GetAttribute(half); attr) {
        attr.Get(&value, UsdTimeCode::EarliestTime());
    }
    return value;
}

/// One cloud as `lightLayerFromCloud` reads it: its positions and harmonics
/// as the file holds them, and the space its colours are in.
struct CloudArrays {
    UsdStageRefPtr stage;
    std::string    field;
    VtValue        positions;
    VtValue        sh;
    uint32_t       count = 0;
    uint32_t       coefficients = 1;
    bool           linear = false;
};

Result<CloudArrays> cloudArrays(const std::filesystem::path& path, const std::string& prim) {
    CloudArrays c;
    c.stage = UsdStage::Open(path.string());
    if (!c.stage) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}'", path.string());
    }
    const UsdVolParticleField3DGaussianSplat field = fieldOf(c.stage, prim);
    if (!field) {
        return Error::make(ErrorCode::NotFound, "'{}' has no gaussian ParticleField{}", path.string(),
                           prim.empty() ? std::string() : " at " + prim);
    }
    c.field = field.GetPath().GetString();
    const UsdPrim held = field.GetPrim();
    c.positions = eitherOf(held, UsdVolTokens->positions, UsdVolTokens->positionsh);
    c.sh = eitherOf(held, UsdVolTokens->radianceSphericalHarmonicsCoefficients,
                    UsdVolTokens->radianceSphericalHarmonicsCoefficientsh);
    int degree = 0;
    field.GetRadianceSphericalHarmonicsDegreeAttr().Get(&degree);
    c.coefficients = static_cast<uint32_t>((degree + 1) * (degree + 1));
    size_t n = 0;
    const scene::FloatStream positions = streamOf(c.positions, &n);
    c.count = static_cast<uint32_t>(n);
    if (c.count == 0 || positions.isDouble) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': {} has no positions in float or half", path.string(),
                           c.field);
    }
    const scene::FloatStream sh = streamOf(c.sh);
    if (sh.isDouble || sh.values() < uint64_t{c.count} * c.coefficients * 3) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': {} carries {} harmonic values for {} gaussians at degree {}",
                           path.string(), c.field, sh.values(), c.count, degree);
    }
    bool linear = false;
    const UsdAttribute said = held.GetAttribute(TfToken("primvars:athenea:splat:linear"));
    c.linear = said && said.Get(&linear) && linear;
    return c;
}

Result<gpu::Buffer> upload(gpu::Device& device, const scene::FloatStream& stream, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>((stream.bytes.size() + 3) / 4 * 4, 4);
    desc.elementBytes = 4;
    desc.label = label;
    auto made = gpu::Buffer::create(device, desc);
    if (!made) return std::move(made).error();
    if (!stream.empty()) {
        ATHENEA_TRY(made->write(device, 0, stream.bytes.size(), stream.bytes.data()));
    }
    return made;
}

}   // namespace

const LightState* LightRig::state(const std::string& name) const {
    for (const LightState& s : states) {
        if (s.name == name) {
            return &s;
        }
    }
    return nullptr;
}

std::map<std::string, float> LightRig::levelsFor(const std::string& stateName,
                                                 const std::map<std::string, float>& levels) const {
    const LightState* chosen = stateName.empty() ? nullptr : state(stateName);
    std::map<std::string, float> out;
    for (const LightGroup& group : groups) {
        float level = group.initialOn ? 1.0F : 0.0F;
        if (chosen != nullptr) {
            const auto target = chosen->targets.find(group.name);
            level = target != chosen->targets.end() ? target->second : 0.0F;
        }
        if (const auto said = levels.find(group.name); said != levels.end()) {
            level = said->second;
        }
        out[group.name] = level;
    }
    return out;
}

std::map<std::string, std::array<float, 3>> LightRig::weights(const std::string& stateName,
                                                              const std::map<std::string, float>& levels) const {
    const std::map<std::string, float> level = levelsFor(stateName, levels);
    std::map<std::string, std::array<float, 3>> out;
    for (const LightGroup& group : groups) {
        const float scale = level.at(group.name) * group.luminance;
        out[group.name] = {scale * group.colour[0], scale * group.colour[1], scale * group.colour[2]};
    }
    return out;
}

LightRig readLightRig(const UsdStageRefPtr& stage) {
    LightRig rig;
    if (!stage) {
        return rig;
    }
    for (const UsdPrim& prim : stage->Traverse()) {
        const UsdPrim parent = prim.GetParent();
        if (!parent) {
            continue;
        }
        const std::string& under = parent.GetName().GetString();
        const std::string name = prim.GetName().GetString();
        if (under == "LightGroups") {
            const bool known = std::any_of(rig.groups.begin(), rig.groups.end(),
                                           [&](const LightGroup& g) { return g.name == name; });
            if (known) {
                continue;
            }
            LightGroup group;
            group.name = name;
            group.prim = prim.GetPath().GetString();
            group.luminance = floatAttr(prim, "athenea:lightGroup:emissionLuminance",
                                        floatAttr(prim, "athenea:lightGroup:radiance", 1.0F));
            if (!colourAttr(prim, "athenea:lightGroup:emissionColor", group.colour)) {
                colourAttr(prim, "athenea:lightGroup:color", group.colour);
            }
            bool on = false;
            if (const UsdAttribute attr = prim.GetAttribute(TfToken("athenea:lightGroup:initialOn"));
                attr && attr.Get(&on)) {
                group.initialOn = on;
            }
            rig.groups.push_back(std::move(group));
        } else if (under == "LightStates") {
            const bool known = std::any_of(rig.states.begin(), rig.states.end(),
                                           [&](const LightState& s) { return s.name == name; });
            if (known) {
                continue;
            }
            LightState state;
            state.name = name;
            state.prim = prim.GetPath().GetString();
            const std::string prefix = kTargetPrefix;
            for (const UsdAttribute& attr : prim.GetAuthoredAttributes()) {
                const std::string& attrName = attr.GetName().GetString();
                if (attrName.size() > prefix.size() && attrName.compare(0, prefix.size(), prefix) == 0) {
                    state.targets[attrName.substr(prefix.size())] = floatAttr(prim, attrName.c_str(), 0.0F);
                }
            }
            rig.states.push_back(std::move(state));
        }
    }
    return rig;
}

Result<LightMergeReport> mergeLightLayers(gpu::ShaderLibrary& library, const LightMergeOptions& options) {
    gpu::Device& device = library.device();
    auto kernel = gpu::ComputeKernel::create(library, "athenea/scene/light_layers", "lightLayerFromCloud");
    if (!kernel) return std::move(kernel).error();
    auto base = cloudArrays(options.base, options.prim);
    if (!base) return std::move(base).error();
    const uint32_t n = base->count;
    auto basePositions = upload(device, streamOf(base->positions), "lights.base.positions");
    if (!basePositions) return std::move(basePositions).error();
    LightMergeReport report;
    report.gaussians = n;

    // Each layer against the base, its light made on the device and read back
    // as the array the file will hold. Nothing is written until every layer
    // has been checked: a refused layer leaves no file behind.
    std::vector<std::pair<std::string, VtVec3fArray>> written;
    for (const LightLayerFile& file : options.layers) {
        if (file.group.empty() || TfToken(file.group) != TfToken(TfMakeValidIdentifier(file.group))) {
            return Error::make(ErrorCode::InvalidArgument, "'{}' is not a group name a primvar can carry", file.group);
        }
        if (std::any_of(written.begin(), written.end(), [&](const auto& w) { return w.first == file.group; })) {
            return Error::make(ErrorCode::InvalidArgument, "group '{}' given twice", file.group);
        }
        auto layer = cloudArrays(file.path, {});
        if (!layer) return std::move(layer).error();
        if (layer->count != n) {
            return Error::make(ErrorCode::InvalidArgument,
                               "'{}' ({}): {} gaussians where the base '{}' has {}; a layer must be the same "
                               "conversion of the same geometry",
                               file.path.string(), file.group, layer->count, options.base.string(), n);
        }
        const scene::FloatStream positionStream = streamOf(layer->positions);
        const scene::FloatStream shStream = streamOf(layer->sh);
        auto positions = upload(device, positionStream, "lights.layer.positions");
        if (!positions) return std::move(positions).error();
        auto sh = upload(device, shStream, "lights.layer.sh");
        if (!sh) return std::move(sh).error();
        gpu::BufferDesc lightDesc;
        lightDesc.bytes = uint64_t{n} * 12;
        lightDesc.elementBytes = 4;
        lightDesc.label = "lights.layer.light";
        auto light = gpu::Buffer::create(device, lightDesc);
        if (!light) return std::move(light).error();
        const std::array<uint32_t, 2> zeros{0, 0};
        auto counters = gpu::Buffer::fromSpan<uint32_t>(device, zeros, "lights.layer.counters");
        if (!counters) return std::move(counters).error();
        gpu::CommandBatch batch(device);
        kernel->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["basePositions"].setBinding(basePositions->rhi());
            cursor["layerPositions"].setBinding(positions->rhi());
            cursor["layerSh"].setBinding(sh->rhi());
            cursor["layerLight"].setBinding(light->rhi());
            cursor["counters"].setBinding(counters->rhi());
            auto p = cursor["mergeParams"];
            p["count"].setData(n);
            p["baseHalf"].setData(uint32_t{streamOf(base->positions).half ? 1u : 0u});
            p["layerHalf"].setData(uint32_t{positionStream.half ? 1u : 0u});
            p["shHalf"].setData(uint32_t{shStream.half ? 1u : 0u});
            p["coefficients"].setData(layer->coefficients);
            p["linearCloud"].setData(uint32_t{layer->linear ? 1u : 0u});
            p["tolerance"].setData(options.tolerance);
            p["floor"].setData(options.dropBelow);
        });
        ATHENEA_TRY(batch.submit(true));
        std::array<uint32_t, 2> counted{0, 0};
        ATHENEA_TRY(counters->read(device, 0, sizeof(counted), counted.data()));
        if (counted[0] != 0) {
            return Error::make(ErrorCode::InvalidArgument,
                               "'{}' ({}): {} of its {} gaussians are not where the base's are (tolerance {}); a "
                               "layer must be the same conversion of the same geometry",
                               file.path.string(), file.group, counted[0], n, options.tolerance);
        }
        LightMergeReport::Layer said;
        said.group = file.group;
        said.lit = counted[1];
        said.dropped = counted[1] == 0;
        report.layers.push_back(said);
        if (said.dropped) {
            log::warn("lights: '{}' ({}) sends out nothing above {}: dropped", file.path.string(), file.group,
                      options.dropBelow);
            continue;
        }
        if (!layer->linear) {
            log::warn("lights: '{}' does not say its colours are light (primvars:athenea:splat:linear); read as a "
                      "capture's sRGB",
                      file.path.string());
        }
        // The light as the device wrote it, three floats a gaussian: copied
        // into the array, not computed.
        VtVec3fArray values(n);
        ATHENEA_TRY(light->read(device, 0, uint64_t{n} * 12, values.data()));
        written.emplace_back(file.group, std::move(values));
    }

    // A copy of the base, then the layers on its field and the sidecar's
    // Scopes under it, saved once.
    const SdfLayerRefPtr baseLayer = base->stage->GetRootLayer();
    if (!baseLayer->Export(options.output.string())) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}'", options.output.string());
    }
    UsdStageRefPtr out = UsdStage::Open(options.output.string());
    if (!out) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}' after writing it", options.output.string());
    }
    const UsdPrim field = out->GetPrimAtPath(SdfPath(base->field));
    if (!field) {
        return Error::make(ErrorCode::InternalError, "'{}': no {} in the copy", options.output.string(), base->field);
    }
    UsdGeomPrimvarsAPI primvars(field);
    for (const auto& [group, values] : written) {
        primvars
            .CreatePrimvar(TfToken("athenea:splat:lightLayer:" + group), SdfValueTypeNames->Color3fArray,
                           UsdGeomTokens->vertex)
            .Set(values);
    }
    if (!options.lights.empty()) {
        const SdfLayerRefPtr sidecar = SdfLayer::FindOrOpen(options.lights.string());
        if (!sidecar) {
            return Error::make(ErrorCode::IoFailure, "cannot open the lights sidecar '{}'", options.lights.string());
        }
        const SdfLayerHandle into = out->GetRootLayer();
        std::vector<SdfPath> scopes;
        sidecar->Traverse(SdfPath::AbsoluteRootPath(), [&](const SdfPath& path) {
            if (path.IsPrimPath()) {
                const std::string& name = path.GetName();
                if (name == "LightGroups" || name == "LightStates" || name == "LightRules") {
                    scopes.push_back(path);
                }
            }
        });
        for (const SdfPath& from : scopes) {
            const SdfPath to = field.GetPath().AppendChild(TfToken(from.GetName()));
            // A base merged before carries the Scopes already: this sidecar's replace them.
            if (const SdfPrimSpecHandle there = into->GetPrimAtPath(to)) {
                into->GetPrimAtPath(to.GetParentPath())->RemoveNameChild(there);
            }
            if (!SdfCopySpec(sidecar, from, into, to)) {
                return Error::make(ErrorCode::IoFailure, "cannot copy {} of '{}' to {}", from.GetString(),
                                   options.lights.string(), to.GetString());
            }
            report.copied.push_back(to.GetString());
        }
        if (scopes.empty()) {
            log::warn("lights: '{}' has no LightGroups, LightStates or LightRules Scope", options.lights.string());
        }
    }
    if (!out->GetRootLayer()->Save()) {
        return Error::make(ErrorCode::IoFailure, "cannot save '{}'", options.output.string());
    }
    return report;
}

}   // namespace athenea::usd
