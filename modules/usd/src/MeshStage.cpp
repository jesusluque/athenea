// Copyright (c) 2026 jesus luque.
#include "athenea/usd/MeshStage.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/base/gf/interval.h>
#include <pxr/usd/usd/editContext.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdSkel/animMapper.h>
#include <pxr/usd/usdSkel/bakeSkinning.h>
#include <pxr/usd/usdSkel/binding.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/skeleton.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/utils.h>

#include "athenea/core/Log.h"
#include "athenea/geom/Mesh.h"
#include "athenea/usd/PrimData.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace athenea::usd {
namespace {

/// One input of a shader, followed as far as it goes: a value, or the file a
/// texture reads.
struct Resolved {
    bool         hasValue = false;
    VtValue      value;
    StageTexture texture;
};

[[nodiscard]] bool isImageNode(const TfToken& id) {
    const std::string& name = id.GetString();
    return name.rfind("ND_image_", 0) == 0 || name.rfind("ND_tiledimage_", 0) == 0 ||
           name == "UsdUVTexture";
}

[[nodiscard]] bool isNormalMapNode(const TfToken& id) {
    return id.GetString().rfind("ND_normalmap", 0) == 0;
}

/// The file a texture node reads, with what its colour space says.
[[nodiscard]] StageTexture fileOf(const UsdShadeShader& shader) {
    StageTexture texture;
    const UsdShadeInput file = shader.GetInput(TfToken("file"));
    if (!file) {
        return texture;
    }
    SdfAssetPath asset;
    if (!file.Get(&asset)) {
        return texture;
    }
    // Resolved where the resolver could, authored otherwise: the same rule
    // Light.cpp and Volume.cpp follow for every asset this engine opens.
    texture.file = !asset.GetResolvedPath().empty() ? asset.GetResolvedPath() : asset.GetAssetPath();
    const std::string space = file.GetAttr().GetColorSpace().GetString();
    texture.srgb = space.find("srgb") != std::string::npos || space.find("sRGB") != std::string::npos;
    if (space.empty()) {
        // UsdUVTexture says it another way.
        TfToken source;
        if (const UsdShadeInput hint = shader.GetInput(TfToken("sourceColorSpace"))) {
            hint.Get(&source);
            texture.srgb = source == TfToken("sRGB");
        }
    }
    // How a texel is read: UsdUVTexture's own scale and bias, which a height
    // map needs -- written 0..1, it stands for whatever they say.
    for (const auto& [name, into] : {std::pair{"scale", &texture.scale}, std::pair{"bias", &texture.bias}}) {
        GfVec4f value;
        if (const UsdShadeInput held = shader.GetInput(TfToken(name)); held && held.Get(&value)) {
            *into = {value[0], value[1], value[2], value[3]};
        }
    }
    // WHICH COORDINATES. `inputs:st` leads back to a primvar reader, whose
    // `varname` is the primvar. The sparrow's feather cards read their colour
    // off one atlas by `st` and their shape and normal off another by
    // `UVMap_001`, and read all by the first, the cards were cut to the
    // wrong texels.
    if (const UsdShadeInput st = shader.GetInput(TfToken("st"))) {
        for (const UsdAttribute& attribute : st.GetValueProducingAttributes()) {
            if (UsdShadeUtils::GetType(attribute.GetName()) != UsdShadeAttributeType::Output) {
                continue;
            }
            const UsdShadeShader reader(attribute.GetPrim());
            if (!reader) {
                continue;
            }
            if (const UsdShadeInput varname = reader.GetInput(TfToken("varname"))) {
                std::string name;
                TfToken token;
                if (varname.Get(&name)) {
                    texture.uvSet = name;
                } else if (varname.Get(&token)) {
                    texture.uvSet = token.GetString();
                }
            }
            break;
        }
    }
    // MaterialX says it with `inputs:texcoord`, back to a geometric property
    // reader whose `geomprop` is the primvar.
    if (const UsdShadeInput texcoord = shader.GetInput(TfToken("texcoord")); texcoord && texture.uvSet.empty()) {
        for (const UsdAttribute& attribute : texcoord.GetValueProducingAttributes()) {
            if (UsdShadeUtils::GetType(attribute.GetName()) != UsdShadeAttributeType::Output) {
                continue;
            }
            const UsdShadeShader reader(attribute.GetPrim());
            if (!reader) {
                continue;
            }
            if (const UsdShadeInput geomprop = reader.GetInput(TfToken("geomprop"))) {
                std::string name;
                TfToken token;
                if (geomprop.Get(&name)) {
                    texture.uvSet = name;
                } else if (geomprop.Get(&token)) {
                    texture.uvSet = token.GetString();
                }
            }
            break;
        }
    }
    return texture;
}

/// Follows `input` through node graphs, interface inputs and a normal map to
/// whatever finally produces it. Depth-limited: a graph may be a cycle, and
/// a converter chain may be long, and neither is worth following forever --
/// this reads what a splat can carry, not what a shading point would.
[[nodiscard]] Resolved resolve(const UsdShadeInput& input, int depth = 0) {
    Resolved out;
    if (!input || depth > 8) {
        return out;
    }
    const UsdShadeAttributeVector producing = input.GetValueProducingAttributes();
    for (const UsdAttribute& attribute : producing) {
        if (UsdShadeUtils::GetType(attribute.GetName()) == UsdShadeAttributeType::Output) {
            const UsdShadeShader shader(attribute.GetPrim());
            if (!shader) {
                continue;
            }
            TfToken id;
            shader.GetShaderId(&id);
            if (isImageNode(id)) {
                out.texture = fileOf(shader);
                // `outputs:a` is not `outputs:rgb`. A cut-out mask is read off
                // one channel of a map that holds something else, and which
                // channel is part of the connection, not of the file.
                const std::string output = attribute.GetBaseName().GetString();
                if (output.size() == 1 && output.find_first_of("rgba") == 0) {
                    out.texture.channel = output[0];
                }
                return out;
            }
            if (isNormalMapNode(id)) {
                // A normal map node stands between the file and the surface.
                // What we want is the file; what the node does to it -- the
                // tangent frame -- the conversion does for itself.
                return resolve(shader.GetInput(TfToken("in")), depth + 1);
            }
            // Something computed: a mix, a multiply, a noise. There is no
            // answer here without shading the point, so the caller's default
            // stands and the material says so in the log.
            continue;
        }
        const UsdShadeInput held(attribute);
        if (held && held.Get(&out.value)) {
            out.hasValue = !out.value.IsEmpty();
            if (out.hasValue) {
                return out;
            }
        }
    }
    // Unconnected, with its own value.
    if (!out.hasValue && input.Get(&out.value)) {
        out.hasValue = !out.value.IsEmpty();
    }
    return out;
}

void takeFloat(const Resolved& resolved, float& into) {
    if (!resolved.hasValue) {
        return;
    }
    if (resolved.value.IsHolding<float>()) {
        into = resolved.value.UncheckedGet<float>();
    } else if (resolved.value.IsHolding<double>()) {
        into = static_cast<float>(resolved.value.UncheckedGet<double>());
    }
}

void takeColour(const Resolved& resolved, std::array<float, 3>& into) {
    if (!resolved.hasValue) {
        return;
    }
    if (resolved.value.IsHolding<GfVec3f>()) {
        const GfVec3f& v = resolved.value.UncheckedGet<GfVec3f>();
        into = {v[0], v[1], v[2]};
    } else if (resolved.value.IsHolding<GfVec3d>()) {
        const GfVec3d& v = resolved.value.UncheckedGet<GfVec3d>();
        into = {static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])};
    }
}

/// What a material says, as far as a gaussian can carry it.
///
/// Two vocabularies are read: MaterialX's `standard_surface`, which is what
/// every asset built from a .mtlx arrives as, and `UsdPreviewSurface`, which is
/// what a file written for the viewport uses. They name the same things
/// differently and nothing else in this engine has had to ask them directly --
/// every other route goes through Hydra, which hands over a network.
[[nodiscard]] StageMaterial materialOf(const UsdPrim& prim) {
    StageMaterial out;
    const UsdShadeMaterialBindingAPI binding(prim);
    const UsdShadeMaterial material = binding.ComputeBoundMaterial();
    if (!material) {
        return out;
    }
    out.path = material.GetPath().GetString();
    UsdShadeShader surface = material.ComputeSurfaceSource({TfToken("mtlx")});
    if (!surface) {
        surface = material.ComputeSurfaceSource();
    }
    if (!surface) {
        return out;
    }
    TfToken id;
    surface.GetShaderId(&id);
    // WHICH SURFACE, AND THEREFORE WHAT ITS INPUTS ARE CALLED.
    //
    // Three vocabularies reach this, and they agree on almost no name.
    // MaterialX's `standard_surface` is what an asset built from a .mtlx
    // carries; **OpenPBR** is what Blender 5 writes when it is asked for a
    // MaterialX network, and it calls the same things `base_metalness` and
    // `transmission_weight`; `UsdPreviewSurface` is what everything else
    // writes. Read with the wrong set the inputs simply are not there and
    // every material comes back at its defaults, in silence -- which is how
    // a concept car whose glass is `transmission = 1` arrived opaque.
    const bool preview = id == TfToken("UsdPreviewSurface");
    const bool openPbr = id == TfToken("ND_open_pbr_surface_surfaceshader") ||
                         id == TfToken("open_pbr_surface");

    const auto read = [&surface](const char* name) { return resolve(surface.GetInput(TfToken(name))); };
    const Resolved colour = read(preview ? "diffuseColor" : "base_color");
    takeColour(colour, out.baseColour);
    out.albedo = colour.texture;

    // A MAP IS THE VALUE, NOT A FACTOR ON THE DEFAULT. The conversion
    // multiplies a material's constant into its map, which is glTF's
    // convention where the constant defaults to one. USD's is that a
    // connection replaces the value: an input connected to a texture has no
    // constant to speak of, and reading none left `roughness` at this
    // struct's own default of 0.5 -- so the sparrow's roughness map, which
    // runs to 1, was halved everywhere, and its head shone like a marble.
    // A metallic map with the default of 0 in front of it would have been
    // erased outright. Connected, the constant is one.
    const Resolved metallic = read(preview ? "metallic" : (openPbr ? "base_metalness" : "metalness"));
    takeFloat(metallic, out.metallic);
    out.metallicMap = metallic.texture;
    if (!out.metallicMap.empty()) {
        out.metallic = 1.0F;
    }

    const Resolved roughness = read(preview ? "roughness" : "specular_roughness");
    takeFloat(roughness, out.roughness);
    out.roughnessMap = roughness.texture;
    if (!out.roughnessMap.empty()) {
        out.roughness = 1.0F;
    }

    out.normal = read(openPbr ? "geometry_normal" : "normal").texture;

    if (preview) {
        // UsdPreviewSurface has no transmission. What it has is an opacity,
        // and a surface you can see through is one whose opacity is less than
        // one -- so that is read as transmission, which is the only place a
        // gaussian can put it.
        const Resolved resolved = read("opacity");
        float opacity = 1.0F;
        takeFloat(resolved, opacity);
        out.transmission = std::clamp(1.0F - opacity, 0.0F, 1.0F);
        // A MAP ON THE OPACITY IS A CUT-OUT, NOT A TRANSMISSION. Where it
        // reads low the surface is not there; where it reads high it is
        // opaque. Carrying it as transmission would make a feather a pane of
        // glass shaped like a rectangle, which is what the wings were.
        out.opacityMap = resolved.texture;
        takeFloat(read("ior"), out.ior);
    } else {
        takeFloat(read(openPbr ? "specular_ior" : "specular_IOR"), out.ior);
        if (openPbr) {
            const Resolved thin = read("geometry_thin_walled");
            if (thin.hasValue && thin.value.IsHolding<bool>()) {
                out.thinWalled = thin.value.UncheckedGet<bool>();
            } else if (thin.hasValue && thin.value.IsHolding<int>()) {
                out.thinWalled = thin.value.UncheckedGet<int>() != 0;
            }
        }
        takeFloat(read(openPbr ? "transmission_weight" : "transmission"), out.transmission);
        takeColour(read("transmission_color"), out.transmissionColour);
        // THE SAME CUT-OUT, IN MATERIALX. `opacity` (`geometry_opacity` in
        // OpenPBR) is coverage there too -- where it reads low the surface is
        // not there -- and it is not transmission, which has an input of its
        // own. So a glass feather keeps its shape: the transmission makes it
        // glass and the map still cuts the card. An image node gives its
        // first channel, not an alpha, so that is the channel read.
        StageTexture cut = read(openPbr ? "geometry_opacity" : "opacity").texture;
        if (!cut.empty() && cut.channel == 0) {
            cut.channel = 'r';
        }
        out.opacityMap = cut;
    }

    // DISPLACEMENT: a height along the normal, which a gaussian can carry for
    // almost nothing -- it is where it stands -- and a mesh only by being cut
    // into triangles finer than the relief.
    const auto takeHeight = [&out](const Resolved& height, float scale) {
        StageTexture map = height.texture;
        if (!map.empty()) {
            if (map.channel == 0) {
                map.channel = 'r';   // an image node gives one value on its first channel
            }
            // The texel's own scale and bias, on the channel read, and the
            // node's scale over both.
            const size_t c = map.channel == 'g' ? 1 : map.channel == 'b' ? 2 : map.channel == 'a' ? 3 : 0;
            out.displacementScale = map.scale[c] * scale;
            out.displacementBias = map.bias[c] * scale;
            out.displacementMap = map;
        } else {
            float constant = 0.0F;
            takeFloat(height, constant);
            out.displacementBias = constant * scale;
        }
    };
    if (preview) {
        takeHeight(read("displacement"), 1.0F);
    } else if (const UsdShadeShader displacement = material.ComputeDisplacementSource({TfToken("mtlx")})) {
        TfToken kind;
        displacement.GetShaderId(&kind);
        if (kind == TfToken("ND_displacement_float")) {
            float scale = 1.0F;
            takeFloat(resolve(displacement.GetInput(TfToken("scale"))), scale);
            takeHeight(resolve(displacement.GetInput(TfToken("displacement"))), scale);
        } else {
            athenea::log::info("mesh2splat: '{}' displaces by a vector ({}), which is not read; only a height is",
                           out.path, kind.GetString());
        }
    }
    return out;
}

/// An affine map's three rows, in the order a kernel applies them: row `r`
/// holds the coefficients of the r-th coordinate of the result. USD stores a
/// matrix for row vectors (`v' = v M`), so a row here is a column there.
/// Where a prototype's mesh stands, once per instance of the PointInstancer
/// that holds it: empty for a mesh no instancer holds. The prototype root's
/// own transform is in the instance's (IncludeProtoXform), so the mesh is
/// taken relative to that root, then placed by each instance and the
/// instancer. The innermost instancer only: one nested inside another's
/// prototype is not expanded twice.
struct InstancedAt {
    bool                    instanced = false;   ///< under a prototype of some instancer
    std::vector<GfMatrix4d> toWorld;             ///< one a drawn instance
};

[[nodiscard]] InstancedAt instancesOf(const UsdPrim& prim, UsdGeomXformCache& transforms, UsdTimeCode at) {
    InstancedAt out;
    for (UsdPrim child = prim, up = prim.GetParent(); up; child = up, up = up.GetParent()) {
        const UsdGeomPointInstancer instancer(up);
        if (!instancer) {
            continue;
        }
        SdfPathVector prototypes;
        instancer.GetPrototypesRel().GetForwardedTargets(&prototypes);
        // Which prototype `prim` is under: the target at or above it.
        int which = -1;
        UsdPrim root;
        for (size_t k = 0; k < prototypes.size(); ++k) {
            if (prim.GetPath().HasPrefix(prototypes[k])) {
                which = static_cast<int>(k);
                root = prim.GetStage()->GetPrimAtPath(prototypes[k]);
                break;
            }
        }
        if (which < 0 || !root) {
            continue;   // under the instancer but no prototype of it
        }
        out.instanced = true;
        VtArray<GfMatrix4d> placed;
        if (!instancer.ComputeInstanceTransformsAtTime(&placed, at, at)) {
            return out;
        }
        VtIntArray indices;
        instancer.GetProtoIndicesAttr().Get(&indices, at);
        const GfMatrix4d relative =
            transforms.GetLocalToWorldTransform(prim) * transforms.GetLocalToWorldTransform(root).GetInverse();
        const GfMatrix4d parent = transforms.GetLocalToWorldTransform(up);
        for (size_t i = 0; i < placed.size() && i < indices.size(); ++i) {
            if (indices[i] == which) {
                out.toWorld.push_back(relative * placed[i] * parent);
            }
        }
        return out;
    }
    return out;
}

[[nodiscard]] std::array<float, 12> rowsOf(const GfMatrix4d& m) {
    std::array<float, 12> rows{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rows[static_cast<size_t>(r) * 4 + static_cast<size_t>(c)] = static_cast<float>(m[c][r]);
        }
        rows[static_cast<size_t>(r) * 4 + 3] = static_cast<float>(m[3][r]);
    }
    return rows;
}

/// The map a normal takes: the inverse transpose of the linear part. In USD's
/// row-vector convention that is the inverse's own rows, which is why this is
/// three lines rather than a transpose and an inverse.
[[nodiscard]] std::array<float, 12> normalRowsOf(const GfMatrix4d& m) {
    const GfMatrix4d inverse = m.GetInverse();
    std::array<float, 12> rows{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rows[static_cast<size_t>(r) * 4 + static_cast<size_t>(c)] = static_cast<float>(inverse[r][c]);
        }
    }
    return rows;
}

}   // namespace

struct MeshStage::Impl {
    UsdStageRefPtr        stage;
    std::string           source;
    std::optional<double> posedAt;   ///< the time the skinning was baked for
};

/// Puts a skinned stage in the pose it holds at `time`.
///
/// A conversion reads `UsdGeomMesh` directly, and a skinned mesh's `points`
/// attribute does not animate: the deformation is the skeleton's, and USD
/// resolves it through UsdSkel. `UsdSkelBakeSkinning` writes the posed points
/// as time samples on the meshes themselves, so the reads below need to know
/// nothing about skinning -- which is what keeps this away from Hydra and from
/// a second implementation of UsdSkel.
///
/// It is baked into the **session layer** and for one instant only, so the
/// file on disk is untouched and the cost is one pose and not a range. A stage
/// with no SkelRoot in it comes back unchanged.
Result<void> poseStage(const UsdStageRefPtr& stage, double time) {
    UsdEditContext edit(stage, stage->GetSessionLayer());
    if (!UsdSkelBakeSkinning(stage->Traverse(), GfInterval(time, time))) {
        return Error::make(ErrorCode::InvalidArgument, "cannot pose the stage's skinning at time {}",
                           time);
    }
    return ok();
}

Result<MeshStage> MeshStage::open(const std::filesystem::path& path) {
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    if (!stage) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': not a stage this can open", path.string());
    }
    MeshStage held;
    held.impl_ = std::make_unique<Impl>();
    held.impl_->stage = stage;
    held.impl_->source = path.string();
    return held;
}

MeshStage::MeshStage(MeshStage&&) noexcept = default;
MeshStage& MeshStage::operator=(MeshStage&&) noexcept = default;
MeshStage::~MeshStage() = default;

Result<std::vector<float>> MeshStage::skeletonTransforms(const std::string& skeleton,
                                                         const std::vector<double>& times) const {
    if (impl_ == nullptr) {
        return Error(ErrorCode::InvalidArgument, "no stage");
    }
    const UsdPrim prim = impl_->stage->GetPrimAtPath(SdfPath(skeleton));
    if (!prim || !prim.IsA<UsdSkelSkeleton>()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': not a Skeleton on this stage", skeleton);
    }
    UsdSkelCache cache;
    const UsdSkelSkeletonQuery query = cache.GetSkelQuery(UsdSkelSkeleton(prim));
    if (!query) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': its skeleton cannot be queried", skeleton);
    }
    VtTokenArray joints;
    UsdSkelSkeleton(prim).GetJointsAttr().Get(&joints);
    const size_t count = joints.size();
    if (count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': a skeleton with no joints", skeleton);
    }
    std::vector<float> out(times.size() * count * 16, 0.0F);
    for (size_t frame = 0; frame < times.size(); ++frame) {
        VtMatrix4fArray xforms;
        if (!query.ComputeSkinningTransforms(&xforms, UsdTimeCode(times[frame])) ||
            xforms.size() != count) {
            return Error::make(ErrorCode::InvalidArgument, "'{}': no joint transforms at time {}",
                               skeleton, times[frame]);
        }
        float* held = out.data() + frame * count * 16;
        // A copy, not a computation: USD laid the matrices out and they go
        // over as they are.
        std::memcpy(held, xforms.data(), count * 16 * sizeof(float));
    }
    return out;
}

Result<std::vector<std::string>> MeshStage::joints(const std::string& skeleton) const {
    if (impl_ == nullptr) {
        return Error(ErrorCode::InvalidArgument, "no stage");
    }
    const UsdPrim prim = impl_->stage->GetPrimAtPath(SdfPath(skeleton));
    if (!prim || !prim.IsA<UsdSkelSkeleton>()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': not a Skeleton on this stage", skeleton);
    }
    VtTokenArray held;
    UsdSkelSkeleton(prim).GetJointsAttr().Get(&held);
    std::vector<std::string> out;
    out.reserve(held.size());
    for (const TfToken& token : held) {
        out.push_back(token.GetString());
    }
    return out;
}

double MeshStage::timeCodesPerSecond() const {
    if (impl_ == nullptr) {
        return 24.0;
    }
    return impl_->stage->GetTimeCodesPerSecond();
}

std::pair<double, double> MeshStage::timeRange() const {
    if (impl_ == nullptr) {
        return {0.0, 0.0};
    }
    return {impl_->stage->GetStartTimeCode(), impl_->stage->GetEndTimeCode()};
}

char MeshStage::upAxis() const {
    if (impl_ == nullptr) {
        return 'y';
    }
    return UsdGeomGetStageUpAxis(impl_->stage) == UsdGeomTokens->z ? 'z' : 'y';
}

std::string MeshStage::source() const {
    return impl_ == nullptr ? std::string{} : impl_->source;
}

/// The joints a stage's meshes are carried by, resolved once for the whole
/// stage: one skinning query a skinnable prim, and the skeleton each is bound
/// to. Inherited bindings are UsdSkel's business, not ours.
struct SkelBindings {
    UsdSkelCache                                cache;
    std::map<SdfPath, UsdSkelSkinningQuery>     queries;
    std::map<SdfPath, UsdSkelSkeleton>          skeletons;
};

SkelBindings resolveSkinning(const UsdStageRefPtr& stage) {
    SkelBindings out;
    for (const UsdPrim& prim : stage->Traverse()) {
        if (!prim.IsA<UsdSkelRoot>()) {
            continue;
        }
        const UsdSkelRoot root(prim);
        if (!out.cache.Populate(root, UsdTraverseInstanceProxies())) {
            continue;
        }
        std::vector<UsdSkelBinding> bindings;
        if (!out.cache.ComputeSkelBindings(root, &bindings, UsdTraverseInstanceProxies())) {
            continue;
        }
        for (const UsdSkelBinding& binding : bindings) {
            for (const UsdSkelSkinningQuery& query : binding.GetSkinningTargets()) {
                const SdfPath at = query.GetPrim().GetPath();
                out.queries.emplace(at, query);
                out.skeletons.emplace(at, binding.GetSkeleton());
            }
        }
    }
    return out;
}

/// One mesh's influences, in the SKELETON'S joint order.
///
/// A mesh may name its own subset of the skeleton's joints (`skel:joints`),
/// and then the indices UsdSkel hands back are into that subset. The mapper
/// that USD keeps for it goes the other way -- skeleton order to the mesh's --
/// so it is run over the identity to get the mesh index's skeleton index, and
/// the influences are written in the skeleton's order. One cloud then has one
/// joint order whatever mixture of meshes it came from.
StageSkinning skinningOf(const UsdSkelSkinningQuery& query, const UsdSkelSkeleton& skeleton,
                         size_t points, double time, const GfMatrix4d& toWorld) {
    StageSkinning out;
    if (!query.HasJointInfluences()) {
        return out;
    }
    VtIntArray   indices;
    VtFloatArray weights;
    if (!query.ComputeVaryingJointInfluences(points, &indices, &weights)) {
        return out;
    }
    const int perPoint = query.GetNumInfluencesPerComponent();
    if (perPoint <= 0 || indices.size() != weights.size() ||
        indices.size() != points * static_cast<size_t>(perPoint)) {
        return out;
    }
    VtTokenArray joints;
    skeleton.GetJointsAttr().Get(&joints);

    // The mesh's joint index to the skeleton's.
    std::vector<int> toSkeleton;
    if (const UsdSkelAnimMapperRefPtr& mapper = query.GetJointMapper()) {
        VtIntArray identity(joints.size());
        for (size_t k = 0; k < joints.size(); ++k) {
            identity[k] = static_cast<int>(k);
        }
        VtIntArray mapped;
        if (mapper->Remap(identity, &mapped)) {
            toSkeleton.assign(mapped.begin(), mapped.end());
        }
    }

    out.bound = true;
    out.skeleton = skeleton.GetPrim().GetPath().GetString();
    out.joints.reserve(joints.size());
    for (const TfToken& joint : joints) {
        out.joints.push_back(joint.GetString());
    }
    out.perPoint = static_cast<uint32_t>(perPoint);
    out.influences.resize(indices.size() * 2);
    for (size_t k = 0; k < indices.size(); ++k) {
        int joint = indices[k];
        if (!toSkeleton.empty()) {
            joint = joint >= 0 && static_cast<size_t>(joint) < toSkeleton.size() ? toSkeleton[joint] : 0;
        }
        out.influences[k * 2] = static_cast<float>(std::max(joint, 0));
        out.influences[k * 2 + 1] = weights[k];
    }
    // OUT OF THE CLOUD'S SPACE, NOT OUT OF THE MESH'S.
    //
    // The conversion packs its triangles in world space, so the gaussians
    // stand there and not in the mesh's own space, while UsdSkel's bind
    // transform starts from the mesh's. The two are composed here, once, so
    // that what the file carries is the one matrix a renderer needs: the
    // cloud's own space into the space the joints are measured from.
    const GfMatrix4d bind = toWorld.GetInverse() * query.GetGeomBindTransform(UsdTimeCode(time));
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            out.geomBindTransform[static_cast<size_t>(row) * 4 + static_cast<size_t>(column)] =
                static_cast<float>(bind[row][column]);
        }
    }
    out.dualQuaternion = query.GetSkinningMethod() == UsdSkelTokens->dualQuaternion;
    return out;
}

Result<std::vector<StageMesh>> MeshStage::read(geom::MeshBuilder& builder, const MeshStageOptions& options) {
    if (impl_ == nullptr) {
        return Error(ErrorCode::InvalidArgument, "no stage");
    }
    const SdfPath under = options.prim.empty() ? SdfPath::AbsoluteRootPath() : SdfPath(options.prim);
    if (!options.prim.empty() && !under.IsAbsolutePath()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': not an absolute prim path", options.prim);
    }
    const UsdTimeCode at(options.time);
    // POSED, OR CARRIED, AND NEVER BOTH. A cloud that keeps its joints is
    // built in the bind pose, because that is where the skeleton's transforms
    // expect to find it; posing the stage first would skin it twice.
    if (!options.skinned && (!impl_->posedAt.has_value() || *impl_->posedAt != options.time)) {
        ATHENEA_TRY(poseStage(impl_->stage, options.time));
        impl_->posedAt = options.time;
    }
    if (!options.hidden.empty()) {
        const UsdEditContext session(impl_->stage, impl_->stage->GetSessionLayer());
        for (const std::string& path : options.hidden) {
            const UsdGeomImageable imageable(impl_->stage->GetPrimAtPath(SdfPath(path)));
            if (!imageable) {
                return Error::make(ErrorCode::NotFound, "--hide {}: no prim that can be seen there", path);
            }
            imageable.CreateVisibilityAttr().Set(UsdGeomTokens->invisible);
        }
    }
    const SkelBindings bindings = options.skinned ? resolveSkinning(impl_->stage) : SkelBindings{};
    UsdGeomXformCache transforms(at);
    std::vector<StageMesh> meshes;
    for (const UsdPrim& prim : impl_->stage->Traverse()) {
        if (!prim.IsA<UsdGeomMesh>() || !prim.GetPath().HasPrefix(under)) {
            continue;
        }
        // What the renderer draws, and nothing else: an invisible mesh, or
        // one under an invisible prim or instancer, makes no gaussians.
        if (UsdGeomImageable(prim).ComputeVisibility(at) == UsdGeomTokens->invisible) {
            continue;
        }
        // A prototype stands where its instancer puts it, once an instance,
        // and never at its own place (the renderer does not draw it there).
        const InstancedAt instances = instancesOf(prim, transforms, at);
        if (instances.instanced && instances.toWorld.empty()) {
            continue;
        }
        const UsdGeomMesh mesh(prim);
        // Held for as long as the build takes: every stream below is a span
        // over one of these arrays.
        VtVec3fArray points;
        VtIntArray   counts;
        VtIntArray   indices;
        VtIntArray   holes;
        mesh.GetPointsAttr().Get(&points, at);
        mesh.GetFaceVertexCountsAttr().Get(&counts, at);
        mesh.GetFaceVertexIndicesAttr().Get(&indices, at);
        mesh.GetHoleIndicesAttr().Get(&holes, at);
        if (points.empty() || counts.empty() || indices.empty()) {
            athenea::log::info("mesh2splat: '{}' has no geometry, skipped", prim.GetPath().GetString());
            continue;
        }
        TfToken orientation;
        mesh.GetOrientationAttr().Get(&orientation);
        TfToken scheme;
        mesh.GetSubdivisionSchemeAttr().Get(&scheme);

        VtVec3fArray normals;
        TfToken      normalsInterpolation = UsdGeomTokens->vertex;
        const bool   hasNormals = mesh.GetNormalsAttr().Get(&normals, at) && !normals.empty();
        if (hasNormals) {
            normalsInterpolation = mesh.GetNormalsInterpolation();
        }
        const UsdGeomPrimvarsAPI primvars(prim);
        VtVec2fArray uvs;
        VtIntArray   uvIndices;
        TfToken      uvInterpolation = UsdGeomTokens->faceVarying;
        bool         hasUvs = false;
        std::string  primary;
        for (const char* name : {"st", "st0", "uv", "UVMap"}) {
            const UsdGeomPrimvar primvar = primvars.GetPrimvar(TfToken(name));
            if (primvar && primvar.Get(&uvs, at) && !uvs.empty()) {
                primary = name;
                uvInterpolation = primvar.GetInterpolation();
                // AT THE SAME TIME AS THE VALUES. Blender writes
                // `primvars:st:indices` as time samples when the export
                // carries animation, and `GetIndices` without a time asks the
                // default, which such an attribute has not got: the indices
                // came back empty and the face-varying values were then read
                // positionally, so the sparrow's wings sampled one texel and
                // came out a flat mauve.
                primvar.GetIndices(&uvIndices, at);
                hasUvs = true;
                break;
            }
        }
        if (!hasUvs && !options.withoutTexcoords) {
            athenea::log::info("mesh2splat: '{}' has no texture coordinates, skipped",
                           prim.GetPath().GetString());
            continue;
        }
        // A SECOND SET, where a map of the material reads by a primvar that is
        // not the first: carried as `st2`, and the conversion samples that map
        // by it. One second set; a third map's would have to be a third.
        StageMaterial material = materialOf(prim);
        std::string second;
        for (const StageTexture* texture : {&material.albedo, &material.normal, &material.metallicMap,
                                            &material.roughnessMap, &material.opacityMap,
                                            &material.displacementMap}) {
            if (!texture->empty() && !texture->uvSet.empty() && texture->uvSet != primary) {
                second = texture->uvSet;
                break;
            }
        }
        VtVec2fArray uvs2;
        VtIntArray   uv2Indices;
        TfToken      uv2Interpolation = UsdGeomTokens->faceVarying;
        bool         hasUvs2 = false;
        if (!second.empty()) {
            const UsdGeomPrimvar primvar = primvars.GetPrimvar(TfToken(second));
            if (primvar && primvar.Get(&uvs2, at) && !uvs2.empty()) {
                uv2Interpolation = primvar.GetInterpolation();
                primvar.GetIndices(&uv2Indices, at);
                hasUvs2 = true;
            } else {
                athenea::log::info("mesh2splat: '{}' reads a map by '{}', which the mesh has not got; read by '{}'",
                               prim.GetPath().GetString(), second, primary);
                second.clear();
            }
        }

        const auto interpolationOf = [](const TfToken& token) {
            if (token == UsdGeomTokens->constant) return geom::Interpolation::Constant;
            if (token == UsdGeomTokens->uniform) return geom::Interpolation::Uniform;
            if (token == UsdGeomTokens->varying) return geom::Interpolation::Varying;
            if (token == UsdGeomTokens->faceVarying) return geom::Interpolation::FaceVarying;
            return geom::Interpolation::Vertex;
        };

        std::vector<geom::PrimvarInput> inputs;
        if (hasNormals) {
            geom::PrimvarInput primvar;
            primvar.name = "normals";
            primvar.interpolation = interpolationOf(normalsInterpolation);
            primvar.components = 3;
            primvar.values = {std::as_bytes(std::span<const GfVec3f>(normals.cdata(), normals.size())), false};
            inputs.push_back(std::move(primvar));
        }
        if (hasUvs) {
            geom::PrimvarInput primvar;
            primvar.name = "st";
            primvar.interpolation = interpolationOf(uvInterpolation);
            primvar.components = 2;
            primvar.values = {std::as_bytes(std::span<const GfVec2f>(uvs.cdata(), uvs.size())), false};
            primvar.indices = std::span<const int32_t>(uvIndices.cdata(), uvIndices.size());
            inputs.push_back(std::move(primvar));
        }
        if (hasUvs2) {
            geom::PrimvarInput primvar;
            primvar.name = "st2";
            primvar.interpolation = interpolationOf(uv2Interpolation);
            primvar.components = 2;
            primvar.values = {std::as_bytes(std::span<const GfVec2f>(uvs2.cdata(), uvs2.size())), false};
            primvar.indices = std::span<const int32_t>(uv2Indices.cdata(), uv2Indices.size());
            inputs.push_back(std::move(primvar));
        }

        geom::MeshInput input;
        input.source = prim.GetPath().GetString();
        input.points = {std::as_bytes(std::span<const GfVec3f>(points.cdata(), points.size())), false};
        input.faceVertexCounts = std::span<const int32_t>(counts.cdata(), counts.size());
        input.faceVertexIndices = std::span<const int32_t>(indices.cdata(), indices.size());
        input.holeIndices = std::span<const int32_t>(holes.cdata(), holes.size());
        input.leftHanded = orientation == UsdGeomTokens->leftHanded;
        // Hydra's rule, and `geom::MeshBuilder`'s: a mesh that is subdivided
        // gets smooth normals, one that is not keeps its facets.
        input.smoothNormals =
            !hasNormals && scheme != UsdGeomTokens->none && scheme != UsdGeomTokens->bilinear;
        input.primvars = inputs;

        auto built = builder.build(input);
        if (!built) {
            return std::move(built).error();
        }
        StageMesh out;
        out.path = input.source;
        out.mesh = std::move(*built);
        const GfMatrix4d toWorld = transforms.GetLocalToWorldTransform(prim);
        out.toWorld = rowsOf(toWorld);
        out.normalToWorld = normalRowsOf(toWorld);
        out.displacementUnit = static_cast<float>(std::cbrt(std::abs(toWorld.GetDeterminant3())));
        out.material = std::move(material);
        out.uv2 = hasUvs2 ? second : std::string();
        if (instances.instanced) {
            // One entry an instance, sharing the mesh on the device: its
            // buffers are counted references, so a copy costs no memory.
            for (size_t i = 0; i < instances.toWorld.size(); ++i) {
                StageMesh placed = out;
                placed.path = out.path + "[" + std::to_string(i) + "]";
                placed.toWorld = rowsOf(instances.toWorld[i]);
                placed.normalToWorld = normalRowsOf(instances.toWorld[i]);
                meshes.push_back(std::move(placed));
            }
            continue;
        }
        if (options.skinned) {
            const auto query = bindings.queries.find(prim.GetPath());
            const auto skeleton = bindings.skeletons.find(prim.GetPath());
            if (query != bindings.queries.end() && skeleton != bindings.skeletons.end()) {
                out.skinning = skinningOf(query->second, skeleton->second, points.size(), options.time,
                                          toWorld);
            }
        }
        meshes.push_back(std::move(out));
    }
    if (meshes.empty()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no meshes to convert", impl_->source);
    }
    return meshes;
}

}   // namespace athenea::usd
