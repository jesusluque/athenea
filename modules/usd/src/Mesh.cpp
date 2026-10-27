// Copyright (c) 2026 jesus luque.
#include "Mesh.h"
#include "Instancer.h"

#include <algorithm>
#include <cstring>

#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/lightSchema.h>
// Windows' headers define `interface` as `struct`, and the material network
// schema names a parameter so.
#if defined(_WIN32) && defined(interface)
#undef interface
#endif
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/materialConnectionSchema.h>
#include <pxr/imaging/hd/materialNetworkSchema.h>
#include <pxr/imaging/hd/materialNodeParameterSchema.h>
#include <pxr/imaging/hd/materialNodeSchema.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/subdivTags.h>
#include <pxr/imaging/pxOsd/tokens.h>

#include "RenderParam.h"
#include "athenea/core/Log.h"
#include "athenea/usd/HydraCamera.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// A constant primvar's value (or the first of an array), as float3.
bool firstVec3(VtValue const& value, GfVec3f* out) {
    if (value.IsHolding<GfVec3f>()) {
        *out = value.UncheckedGet<GfVec3f>();
        return true;
    }
    if (value.IsHolding<VtVec3fArray>() && !value.UncheckedGet<VtVec3fArray>().empty()) {
        *out = value.UncheckedGet<VtVec3fArray>()[0];
        return true;
    }
    return false;
}

bool firstFloat(VtValue const& value, float* out) {
    if (value.IsHolding<float>()) {
        *out = value.UncheckedGet<float>();
        return true;
    }
    if (value.IsHolding<VtFloatArray>() && !value.UncheckedGet<VtFloatArray>().empty()) {
        *out = value.UncheckedGet<VtFloatArray>()[0];
        return true;
    }
    return false;
}

/// A string a primvar or attribute holds, as a token or a string.
std::string textOf(VtValue const& value) {
    if (value.IsHolding<TfToken>()) return value.UncheckedGet<TfToken>().GetString();
    if (value.IsHolding<std::string>()) return value.UncheckedGet<std::string>();
    if (value.IsHolding<VtArray<TfToken>>() && !value.UncheckedGet<VtArray<TfToken>>().empty()) {
        return value.UncheckedGet<VtArray<TfToken>>()[0].GetString();
    }
    if (value.IsHolding<VtArray<std::string>>() && !value.UncheckedGet<VtArray<std::string>>().empty()) {
        return value.UncheckedGet<VtArray<std::string>>()[0];
    }
    return {};
}

/// UsdLux's MESHLIGHTAPI ON A MESH, as Hydra delivers it: UsdImaging makes
/// no meshLight prim of its own (OpenUSD 26.08: the spike in
/// docs/decisions.md, mesh lights), it leaves the prim a mesh and gives it a
/// `light` container (`isLight`, `materialSyncMode`) and a material network
/// of its own whose `light` terminal is a MeshLight node carrying the
/// LightAPI's inputs. GetLightParamValue reads only the container, so the
/// inputs are read here from the terminal scene index, as Light.cpp reads a
/// dome's colour space.
std::optional<athenea::usd::MeshLook::MeshLight> meshLightOf(HdSceneDelegate* delegate, const SdfPath& id) {
    const HdSceneIndexBaseRefPtr index = delegate->GetRenderIndex().GetTerminalSceneIndex();
    if (!index) {
        return std::nullopt;
    }
    const HdSceneIndexPrim prim = index->GetPrim(id);
    if (!prim.dataSource) {
        return std::nullopt;
    }
    const HdContainerDataSourceHandle container =
        HdContainerDataSource::Cast(prim.dataSource->Get(HdLightSchemaTokens->light));
    if (!container) {
        return std::nullopt;
    }
    const auto sampled = [&](const TfToken& name) -> VtValue {
        if (const HdSampledDataSourceHandle ds = HdSampledDataSource::Cast(container->Get(name))) {
            return ds->GetValue(0.0F);
        }
        return {};
    };
    const VtValue isLight = sampled(HdTokens->isLight);
    if (!isLight.IsHolding<bool>() || !isLight.UncheckedGet<bool>()) {
        return std::nullopt;
    }
    athenea::usd::MeshLook::MeshLight out;
    const std::string mode = textOf(sampled(HdTokens->materialSyncMode));
    out.mode = mode == "independent" ? 1u : mode == "noMaterialResponse" ? 2u : 0u;
    const HdMaterialSchema material = HdMaterialSchema::GetFromParent(prim.dataSource);
    if (!material.IsDefined()) {
        return out;
    }
    const HdMaterialNetworkSchema network = material.GetMaterialNetwork();
    const HdMaterialConnectionSchema terminal = network.GetTerminals().Get(HdMaterialTerminalTokens->light);
    if (!terminal.IsDefined() || !terminal.GetUpstreamNodePath()) {
        return out;
    }
    const TfToken node = terminal.GetUpstreamNodePath()->GetTypedValue(0.0F);
    const HdMaterialNodeParameterContainerSchema parameters = network.GetNodes().Get(node).GetParameters();
    const auto parameter = [&](const TfToken& name) -> VtValue {
        for (const TfToken& key : {name, TfToken("inputs:" + name.GetString())}) {
            const HdMaterialNodeParameterSchema p = parameters.Get(key);
            if (p.IsDefined() && p.GetValue()) {
                return p.GetValue()->GetValue(0.0F);
            }
        }
        return {};
    };
    float value = 0.0F;
    if (firstFloat(parameter(HdLightTokens->intensity), &value)) out.intensity = value;
    if (firstFloat(parameter(HdLightTokens->exposure), &value)) out.exposure = value;
    if (firstFloat(parameter(HdLightTokens->colorTemperature), &value)) out.temperature = value;
    GfVec3f colour;
    if (firstVec3(parameter(HdLightTokens->color), &colour)) out.colour = {colour[0], colour[1], colour[2]};
    const VtValue normalize = parameter(HdLightTokens->normalize);
    out.normalize = normalize.IsHolding<bool>() && normalize.UncheckedGet<bool>();
    const VtValue temperature = parameter(HdLightTokens->enableColorTemperature);
    out.enableTemperature = temperature.IsHolding<bool>() && temperature.UncheckedGet<bool>();
    return out;
}

}   // namespace

TfTokenVector const& HdAtheneaMesh::GetBuiltinPrimvarNames() const {
    static const TfTokenVector names{HdTokens->points, HdTokens->normals, HdTokens->displayColor,
                                     HdTokens->displayOpacity};
    return names;
}

HdDirtyBits HdAtheneaMesh::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::Clean | HdChangeTracker::InitRepr | HdChangeTracker::DirtyPoints |
           HdChangeTracker::DirtyTopology | HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility |
           HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyDoubleSided |
           HdChangeTracker::DirtyDisplayStyle | HdChangeTracker::DirtyRenderTag |
           HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex | HdChangeTracker::DirtyMaterialId;
}

void HdAtheneaMesh::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
                     TfToken const&) {
    SdfPath const& id = GetId();
    auto* param = static_cast<HdAtheneaRenderParam*>(renderParam);
    auto* engine = param->GetEngine();
    if (engine == nullptr) {
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        return;
    }
    // The camera's shutter, when it is open for a while: transforms and
    // points are sampled at its open and close as well as at the frame.
    const float shutterOpen = static_cast<float>(param->GetShutterOpen());
    const float shutterClose = static_cast<float>(param->GetShutterClose());
    const bool shutter = shutterClose > shutterOpen;
    _UpdateInstancer(delegate, dirtyBits);
    HdInstancer::_SyncInstancerAndParents(delegate->GetRenderIndex(), GetInstancerId());
    std::optional<std::vector<athenea::usd::InstancerLink>> instancing;
    if ((*dirtyBits & (HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex)) != 0) {
        // The chain of instancers above this prototype, innermost first, with
        // the elements each level takes of the one above.
        std::vector<athenea::usd::InstancerLink> chain;
        SdfPath child = id;
        SdfPath instancer = GetInstancerId();
        while (!instancer.IsEmpty()) {
            chain.push_back({instancer, delegate->GetInstanceIndices(instancer, child)});
            HdInstancer* level = delegate->GetRenderIndex().GetInstancer(instancer);
            child = instancer;
            instancer = level != nullptr ? level->GetParentId() : SdfPath();
        }
        instancing = std::move(chain);
    }

    std::optional<athenea::usd::MeshArrays> arrays;
    if ((*dirtyBits & (HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyTopology |
                       HdChangeTracker::DirtyDisplayStyle | HdChangeTracker::DirtyNormals |
                       HdChangeTracker::DirtyPrimvar)) != 0) {
        athenea::usd::MeshArrays a;
        const HdMeshTopology topology = GetMeshTopology(delegate);
        a.faceVertexCounts = topology.GetFaceVertexCounts();
        a.faceVertexIndices = topology.GetFaceVertexIndices();
        a.holeIndices = topology.GetHoleIndices();
        a.invisibleFaces = topology.GetInvisibleFaces();
        a.scheme = topology.GetScheme();
        const PxOsdSubdivTags& tags = topology.GetSubdivTags();
        a.creaseIndices = tags.GetCreaseIndices();
        a.creaseLengths = tags.GetCreaseLengths();
        a.creaseSharpnesses = tags.GetCreaseWeights();
        a.cornerIndices = tags.GetCornerIndices();
        a.cornerSharpnesses = tags.GetCornerWeights();
        a.leftHanded = topology.GetOrientation() != HdTokens->rightHanded;
        for (const HdGeomSubset& subset : topology.GetGeomSubsets()) {
            if (subset.type == HdGeomSubset::TypeFaceSet) {
                a.subsets.push_back({subset.indices, subset.materialId});
            }
        }
        a.points = delegate->Get(id, HdTokens->points);
        // Skinned: the points are an ext computation's output. Its inputs --
        // the aggregator's (rest points, bindings, blend shapes) and its own
        // (the animation's transforms and weights) -- go to the engine, which
        // skins on the device. Hydra never runs the computation.
        for (const HdExtComputationPrimvarDescriptor& computed :
             delegate->GetExtComputationPrimvarDescriptors(id, HdInterpolationVertex)) {
            if (computed.name != HdTokens->points) {
                continue;
            }
            const SdfPath& computation = computed.sourceComputationId;
            athenea::usd::SkinningArrays skin;
            const auto read = [&](const SdfPath& from, const TfToken& name) {
                return delegate->GetExtComputationInput(from, name);
            };
            // The computation's own scene inputs.
            for (const TfToken& name : delegate->GetExtComputationSceneInputNames(computation)) {
                const VtValue value = read(computation, name);
                const std::string& n = name.GetString();
                if (n == "blendShapeWeights") skin.blendShapeWeights = value;
                else if (n == "skinningXforms") skin.skinningXforms = value;
                else if (n == "skinningDualQuats") { skin.skinningDualQuats = value; skin.dualQuaternion = true; }
                else if (n == "skinningScaleXforms") skin.skinningScaleXforms = value;
                else if (n == "skelLocalToWorld") skin.skelLocalToWorld = value;
                else if (n == "primWorldToLocal") skin.primWorldToLocal = value;
            }
            // And what it takes from its aggregator, by the aggregator's outputs.
            for (const HdExtComputationInputDescriptor& input :
                 delegate->GetExtComputationInputDescriptors(computation)) {
                const VtValue value = read(input.sourceComputationId, input.sourceComputationOutputName);
                const std::string& n = input.name.GetString();
                if (n == "restPoints") skin.restPoints = value;
                else if (n == "geomBindXform") skin.geomBindXform = value;
                else if (n == "influences") skin.influences = value;
                else if (n == "numInfluencesPerComponent" && value.IsHolding<int>())
                    skin.numInfluencesPerComponent = value.UncheckedGet<int>();
                else if (n == "hasConstantInfluences" && value.IsHolding<bool>())
                    skin.hasConstantInfluences = value.UncheckedGet<bool>();
                else if (n == "blendShapeOffsets") skin.blendShapeOffsets = value;
                else if (n == "blendShapeOffsetRanges") skin.blendShapeOffsetRanges = value;
            }
            if (!skin.restPoints.IsEmpty()) {
                a.points = skin.restPoints;
                a.skinning = std::move(skin);
            }
            break;
        }
        if (shutter) {
            float times[2] = {0.0F, 0.0F};
            VtValue values[2];
            const size_t n = delegate->SamplePrimvar(id, HdTokens->points, shutterOpen, shutterClose, 2, times, values);
            if (n >= 2 && values[0] != values[n - 1]) {
                a.pointsStart = values[0];
                a.pointsEnd = values[n - 1];
                a.pointsTimeStart = times[0];
                a.pointsTimeEnd = times[n - 1];
            }
        }
        // Smooth normals where Storm computes them: a subdivision scheme that
        // is not none or bilinear, and no flat shading asked for.
        // Every numeric primvar, as it is (indices resolved on the device).
        for (const HdInterpolation interpolation : {HdInterpolationConstant, HdInterpolationUniform,
                                                    HdInterpolationVarying, HdInterpolationVertex,
                                                    HdInterpolationFaceVarying}) {
            for (const HdPrimvarDescriptor& descriptor : GetPrimvarDescriptors(delegate, interpolation)) {
                if (descriptor.name == HdTokens->points) {
                    continue;
                }
                athenea::usd::PrimvarArrays primvar;
                primvar.name = descriptor.name.GetString();
                primvar.interpolation = static_cast<uint32_t>(interpolation);
                primvar.values = descriptor.indexed
                                     ? delegate->GetIndexedPrimvar(id, descriptor.name, &primvar.indices)
                                     : GetPrimvar(delegate, descriptor.name);
                uint32_t components = 0;
                if (!athenea::usd::primvarStreamOf(primvar.values, &components).empty()) {
                    a.primvars.push_back(std::move(primvar));
                }
            }
        }
        const HdDisplayStyle style = GetDisplayStyle(delegate);
        a.smoothNormals = !style.flatShadingEnabled && topology.GetScheme() != PxOsdOpenSubdivTokens->none &&
                          topology.GetScheme() != PxOsdOpenSubdivTokens->bilinear;
        a.refineLevel = std::max(style.refineLevel, 0);
        a.topologyChanged = (*dirtyBits & HdChangeTracker::DirtyTopology) != 0;
        arrays = std::move(a);
    }
    std::optional<athenea::usd::MeshLook> look;
    if (*dirtyBits & HdChangeTracker::DirtyMaterialId) {
        const SdfPath bound = delegate->GetMaterialId(id);
        athenea::log::debug("hdAthenea: {} material from Hydra: {}", id.GetString(),
                        bound.IsEmpty() ? std::string("<none>") : bound.GetString());
        SetMaterialId(bound);
    }
    if ((*dirtyBits & (HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyDoubleSided |
                       HdChangeTracker::DirtyMaterialId)) != 0) {
        athenea::usd::MeshLook l;
        l.material = GetMaterialId();
        GfVec3f colour;
        if (firstVec3(delegate->Get(id, HdTokens->displayColor), &colour)) {
            l.displayColor = {colour[0], colour[1], colour[2]};
        }
        float opacity = 1.0F;
        if (firstFloat(delegate->Get(id, HdTokens->displayOpacity), &opacity)) {
            l.displayOpacity = opacity;
        }
        l.doubleSided = IsDoubleSided(delegate);
        float plate = 0.0F;
        const VtValue plateValue = delegate->Get(id, TfToken("athenea:plate"));
        if (plateValue.IsHolding<int>()) {
            plate = static_cast<float>(plateValue.UncheckedGet<int>());
        } else if (plateValue.IsHolding<bool>()) {
            plate = plateValue.UncheckedGet<bool>() ? 1.0F : 0.0F;
        } else {
            firstFloat(plateValue, &plate);
        }
        l.plate = plate >= 1.5F ? 2u : plate >= 0.5F ? 1u : 0u;
        float unlit = 0.0F;
        const VtValue unlitValue = delegate->Get(id, TfToken("athenea:unlit"));
        if (unlitValue.IsHolding<int>()) {
            unlit = static_cast<float>(unlitValue.UncheckedGet<int>());
        } else if (unlitValue.IsHolding<bool>()) {
            unlit = unlitValue.UncheckedGet<bool>() ? 1.0F : 0.0F;
        } else {
            firstFloat(unlitValue, &unlit);
        }
        l.unlit = unlit >= 0.5F;
        // What the light linking scene index resolved this prim's collections
        // into. Empty unless something links to it.
        const VtArray<TfToken> categories = delegate->GetCategories(id);
        l.categories.assign(categories.begin(), categories.end());
        // The coordinate systems bound to it: each a coordSys prim (hdsi's,
        // "<target>.__coordSys:NAME" -- "__coordSys_NAME" once the emulation
        // has made a prim name of it -- or a host's "<prim>.coordSys:NAME"),
        // its name the id's last part past that prefix, its transform the
        // prim's.
        if (const HdIdVectorSharedPtr bindings = delegate->GetCoordSysBindings(id)) {
            for (const SdfPath& coordSys : *bindings) {
                std::string name = coordSys.GetName();
                for (const char* prefix : {"__coordSys", "coordSys"}) {
                    const size_t length = std::strlen(prefix);
                    if (name.rfind(prefix, 0) == 0 && name.size() > length + 1 &&
                        (name[length] == ':' || name[length] == '_')) {
                        name = name.substr(length + 1);
                        break;
                    }
                }
                l.coordSys.push_back({name, athenea::usd::fromUsd(delegate->GetTransform(coordSys))});
            }
        }
        l.meshLight = meshLightOf(delegate, id);
        l.lightGroup = textOf(delegate->Get(id, TfToken("athenea:lightGroup")));
        look = l;
    }
    athenea::render::Mat4 transform;
    std::optional<athenea::usd::MeshTransforms> shutterTransforms;
    const bool transformDirty = HdChangeTracker::IsTransformDirty(*dirtyBits, id);
    if (transformDirty) {
        transform = athenea::usd::fromUsd(delegate->GetTransform(id));
        athenea::usd::MeshTransforms at;
        if (shutter) {
            float times[2] = {0.0F, 0.0F};
            GfMatrix4d values[2];
            const size_t n = delegate->SampleTransform(id, shutterOpen, shutterClose, 2, times, values);
            if (n >= 2 && values[0] != values[n - 1]) {
                at.start = athenea::usd::fromUsd(values[0]);
                at.end = athenea::usd::fromUsd(values[n - 1]);
                at.timeStart = times[0];
                at.timeEnd = times[n - 1];
            }
        }
        shutterTransforms = at;
    }
    std::optional<bool> visible;
    if (HdChangeTracker::IsVisibilityDirty(*dirtyBits, id)) {
        _UpdateVisibility(delegate, dirtyBits);
        visible = IsVisible() && HdAtheneaInstancersVisible(delegate, GetInstancerId());
    }
    engine->setMesh(id, GetPrimId(), GetRenderTag(), std::move(arrays), transformDirty ? &transform : nullptr,
                    visible, look, std::move(instancing), shutterTransforms);
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdAtheneaMesh::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdAtheneaRenderParam*>(renderParam)->GetEngine()) {
        engine->remove(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
