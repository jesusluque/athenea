// Copyright (c) 2026 jesus luque.
//
// The meshes of a stage, read without Hydra.
//
// WHY NOT HYDRA
//
// Every other route into this engine goes through Hydra, and should: a render
// wants instancing, visibility, purposes, time samples and a render index that
// knows what changed. A conversion wants none of that. It wants the meshes a
// file holds, once, with their world transforms and what their materials say
// -- and it wants them from a process that is not rendering anything, so there
// is no render index to ask.
//
// So this walks the stage itself: `UsdGeomMesh` prims, `UsdGeomXformCache` for
// the transforms, `UsdShadeMaterialBindingAPI` for the material, and the
// arrays handed straight to `geom::MeshBuilder`, which triangulates them on
// the device in `HdMeshUtil`'s order -- the same triangles a render would have
// drawn, from the same kernel.
//
// What a material says is narrowed here to what a gaussian can carry: a
// colour, a metallic, a roughness, a normal, a transmission, the light it
// gives off, and the files those come from. A MaterialX graph is not evaluated -- that is
// `material::MaterialCompiler`'s work and it needs a shading point -- so a
// value that is computed rather than authored comes back as the texture it
// is read from, or as the default.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <utility>
#include <memory>
#include <string>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/geom/Mesh.h"

namespace athenea::geom {
class MeshBuilder;
}

namespace athenea::usd {

/// A file a material names, and whether its values are sRGB encoded (which is
/// what USD's `colorSpace` says, and what MaterialX's `colorspace` attribute
/// says for these assets).
struct StageTexture {
    std::string file;
    bool        srgb = false;
    /// WHICH CHANNEL THE SURFACE TOOK, because a mask is usually somebody
    /// else's alpha: `'a'` for the alpha, `'r'`, `'g'`, `'b'` for one
    /// component, `0` for the colour. The sparrow's feathers are flat cards
    /// whose shape lives entirely in the alpha of the map that also holds
    /// their normals.
    char        channel = 0;
    /// The primvar the texture's coordinates are read from (`inputs:st` back
    /// to a primvar reader's `varname`); empty where nothing says.
    std::string uvSet;
    /// UsdUVTexture's `scale` and `bias`, a texel read as `texel * scale +
    /// bias`: how a height map written as 0..1 says it stands for -1..1.
    std::array<float, 4> scale{1.0F, 1.0F, 1.0F, 1.0F};
    std::array<float, 4> bias{0.0F, 0.0F, 0.0F, 0.0F};

    [[nodiscard]] bool empty() const noexcept { return file.empty(); }
};

/// What one material says, in the words a splat understands.
struct StageMaterial {
    std::string            path;   ///< the prim, for messages
    std::array<float, 3>   baseColour{1.0F, 1.0F, 1.0F};
    float                  metallic = 0.0F;
    float                  roughness = 0.5F;
    /// Ours, not mesh2splat's: how much of what stands behind the surface
    /// comes through it, and the colour it takes doing so.
    float                  transmission = 0.0F;
    std::array<float, 3>   transmissionColour{1.0F, 1.0F, 1.0F};
    /// The dielectric's index (`specular_ior`, `specular_IOR`, `ior`).
    float                  ior = 1.5F;
    /// A sheet, not a solid (OpenPBR's `geometry_thin_walled`): what it
    /// transmits leaves along the direction it came in.
    bool                   thinWalled = false;
    /// ITS METAL IS A SCHLICK, not a conductor: OpenPBR's (MaterialX's
    /// `generalized_schlick_bsdf`, the base colour head on and the specular
    /// colour at 82 degrees) and glTF's. standard_surface's and
    /// UsdPreviewSurface's metals are conductors of an artistic index. The two
    /// part most for a dark metal at an angle: a car's paint (base 0.05)
    /// reflects 0.17 at 60 degrees as a conductor and 0.08 as a Schlick.
    bool                   schlickMetal = false;
    StageTexture           albedo;
    StageTexture           normal;
    StageTexture           metallicMap;
    StageTexture           roughnessMap;
    /// HOW MUCH OF WHAT STANDS BEHIND THE SURFACE IT COVERS, as a constant:
    /// MaterialX's `opacity` (OpenPBR `geometry_opacity`, glTF `alpha` in
    /// BLEND), or UsdPreviewSurface's in `presence` mode. Coverage, not
    /// transmission: the mesh is drawn there by that lot. One where an input
    /// is connected to `opacityMap`, which is then the value. (A
    /// UsdPreviewSurface in its default `transparent` mode carries its
    /// constant as a thin wall's `transmission` instead.)
    float                  opacity = 1.0F;
    /// The opacity read as coverage, a value a point: where it is below the
    /// conversion's cut the surface is not there at all, and above it the
    /// surface covers what it reads. It is not the same thing as
    /// `transmission`, which is a surface you see through.
    StageTexture           opacityMap;
    /// A CUT-OUT BY THRESHOLD (UsdPreviewSurface's `opacityThreshold`, glTF's
    /// `alpha_cutoff` in MASK): the opacity is either there, whole, or not,
    /// by whether it reads at least this. 0: no threshold, the opacity is
    /// coverage. A constant is resolved here (to an `opacity` of 0 or 1), so
    /// this is only ever set beside an `opacityMap`.
    float                  opacityThreshold = 0.0F;
    /// A HEIGHT ALONG THE NORMAL: UsdPreviewSurface's `displacement`, or a
    /// MaterialX `displacement` node's `displacement` times its `scale`. The
    /// surface stands `map * displacementScale + displacementBias` off the
    /// mesh, in the mesh's own units; the map's channel is one value, and a
    /// material with no map but a constant moves the whole surface.
    StageTexture           displacementMap;
    float                  displacementScale = 1.0F;
    float                  displacementBias = 0.0F;
    /// Whether the material displaces at all: a map, or a constant that is
    /// not zero.
    [[nodiscard]] bool displaces() const noexcept {
        return !displacementMap.empty() || displacementBias != 0.0F;
    }
    /// THE LIGHT IT GIVES OFF BY ITSELF, linear radiance: the colour times
    /// the weight, in each vocabulary's words -- standard_surface's
    /// `emission` x `emission_color`, OpenPBR's `emission_luminance` x
    /// `emission_color` (nits, which the mesh is rendered with as they are),
    /// glTF's `emissive` x `emissive_strength`, UsdPreviewSurface's
    /// `emissiveColor`. With `emissionMap` the map is the colour and this is
    /// what multiplies it (the weight, where the colour is the map; the
    /// colour, where the weight is), as for every other map here.
    std::array<float, 3>   emission{0.0F, 0.0F, 0.0F};
    /// A map on the colour (rgb, `channel` 0) or on the weight (one channel).
    StageTexture           emissionMap;
    /// Whether anything is given off at all.
    [[nodiscard]] bool emits() const noexcept {
        return emission[0] > 0.0F || emission[1] > 0.0F || emission[2] > 0.0F;
    }
    /// WHAT IT LAYERS OVER ITS BASE (proposal 026), as constants: the
    /// dielectric reflection's weight and tint (OpenPBR `specular_weight`
    /// and `specular_color`, standard_surface `specular` and
    /// `specular_color`, glTF `specular` and `specular_color`; the tint is
    /// also a metal's edge colour); a clear coat (`coat_weight` /`coat` /
    /// `clearcoat`, its roughness and index); and a sheen, its colour times
    /// its weight. The specular's index is `ior`. A map on any of these is
    /// not read -- the constant the input would have stands, and the log
    /// says so. Each surface's own defaults, which is what the mesh is
    /// rendered with.
    float                  specularWeight = 1.0F;
    std::array<float, 3>   specularColour{1.0F, 1.0F, 1.0F};
    float                  coatWeight = 0.0F;
    float                  coatRoughness = 0.0F;
    float                  coatIor = 1.5F;
    /// OpenPBR's `coat_darkening` (1 by default there): the base under the
    /// coat darkened by what the coat's inside reflects back into it. 0 for
    /// every other vocabulary, whose coat has none.
    float                  coatDarkening = 0.0F;
    std::array<float, 3>   sheenColour{0.0F, 0.0F, 0.0F};
    float                  sheenRoughness = 0.3F;
    /// The sheen's weight alone (`sheenColour` is it times the colour), for
    /// a map on the colour or on the weight.
    float                  sheenWeight = 0.0F;
    /// MAPS ON THE LAYERS (task TX): which input a map stands for, and the
    /// map. Sampled per gaussian by the conversion, as the base's are, in
    /// place of the input's constant; the first three a material has.
    enum class LayerTarget : uint32_t {
        SpecularWeight = 1, SpecularColour = 2, CoatWeight = 3, CoatRoughness = 4,
        SheenColour = 5, SheenWeight = 6, SheenRoughness = 7
    };
    struct LayerMap {
        LayerTarget  target = LayerTarget::SpecularWeight;
        StageTexture texture;
    };
    std::vector<LayerMap>  layerMaps;
    /// Whether any of it differs from the plain specular every gaussian has
    /// without them (weight one, white, an index of 1.5, no coat, no sheen):
    /// the conversion then writes them (`io::SplatEncoding::lobes`).
    [[nodiscard]] bool layered() const noexcept {
        const auto white = [](const std::array<float, 3>& c) {
            return c[0] == 1.0F && c[1] == 1.0F && c[2] == 1.0F;
        };
        return !layerMaps.empty() || specularWeight != 1.0F || !white(specularColour) || ior != 1.5F || coatWeight > 0.0F ||
               sheenColour[0] > 0.0F || sheenColour[1] > 0.0F || sheenColour[2] > 0.0F;
    }
};

/// WHAT CARRIES A MESH WHEN ITS SKELETON MOVES.
///
/// The joints each of its points is held by and how much, in the skeleton's
/// own joint order, and the transform out of the mesh's space into the bind
/// space those joints are measured from. It is the same thing
/// `geom::SkinningInput` takes, read here out of USD rather than out of
/// Hydra's ext computation, so that a conversion can give each gaussian the
/// influences of the triangle it stands on.
struct StageSkinning {
    bool                  bound = false;
    /// The Skeleton prim's path: what the cloud is written against.
    std::string           skeleton;
    /// Its joints, in order, which is the order the indices below are in.
    std::vector<std::string> joints;
    uint32_t              perPoint = 0;   ///< influences a point
    /// `(joint, weight)`, `perPoint` of them a point, point major -- the
    /// layout `geom::SkinningInput::influences` reads.
    std::vector<float>    influences;
    /// Mesh space to bind space, row major, four rows of four.
    std::array<float, 16> geomBindTransform{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    /// Dual quaternion rather than linear blend, if the mesh asked for it.
    bool                  dualQuaternion = false;
};

/// A GEOMSUBSET THAT BINDS A MATERIAL OF ITS OWN: one of the mesh's
/// `materialBind` family. Its faces are the mesh's triangles whose
/// `GpuMesh::triangleSubsets` is its index plus one; the faces no subset
/// claims keep the mesh's own material.
struct StageSubset {
    std::string   path;       ///< the GeomSubset prim, for messages
    StageMaterial material;   ///< what it binds (the mesh's, where it binds none)
};

/// THE MESHES A MATERIAL IS ON, as a stage binds them (`athenea mesh2splat
/// --validate`): every visible mesh at or under `prim` whose own binding, or
/// a GeomSubset's of its `materialBind` family, names the material. A mesh of
/// several materials is in each of their groups. "" names the meshes bound to
/// nothing. Read on the processor: it is the file's bindings, no geometry.
struct MaterialGroup {
    std::string              material;   ///< its prim path, or "" for none
    std::vector<std::string> meshes;     ///< the meshes, sorted
};
[[nodiscard]] Result<std::vector<MaterialGroup>> stageMaterialGroups(const std::filesystem::path& path,
                                                                     const std::string& prim,
                                                                     const std::vector<std::string>& hidden,
                                                                     double time);

/// Every UsdLux light of a stage, and whether it is a dome.
[[nodiscard]] Result<std::vector<std::pair<std::string, bool>>> stageLights(const std::filesystem::path& path);

/// The first camera of a stage, in traversal order, or "" where it has none.
[[nodiscard]] Result<std::string> stageFirstCamera(const std::filesystem::path& path);

/// One mesh of the stage, already on the device.
struct StageMesh {
    std::string           path;
    geom::GpuMesh         mesh;
    /// Object to world, row major: three rows of four, the fourth implied.
    std::array<float, 12> toWorld{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    /// Its inverse transpose, for the normals; the same twelve numbers.
    std::array<float, 12> normalToWorld{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                                        0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    StageMaterial         material;
    /// How long the mesh's unit is in the world: a height is authored in the
    /// mesh's own units, and a mesh scaled by ten displaces ten times as far.
    /// The cube root of the transform's volume, which is the scale where the
    /// scale is uniform.
    float                 displacementUnit = 1.0F;
    /// Filled when `MeshStageOptions::skinned` asked for it and the mesh is
    /// bound to a skeleton.
    StageSkinning         skinning;
    /// The primvar carried as the mesh's second set of texture coordinates
    /// (`st2`), where a map of its material reads by one that is not the
    /// first; empty otherwise.
    std::string           uv2;
    /// Its GeomSubsets of the `materialBind` family, in the order the mesh
    /// was built with them (`mesh.subsets` of them). Empty: one material.
    std::vector<StageSubset> subsets;
};

/// A camera of the stage, as a conversion that sizes its cells by what that
/// camera sees reads it: where it stands and the lens it looks through.
struct StageCamera {
    /// Camera to world, row major: three rows of four, the position in the
    /// fourth column (as `StageMesh::toWorld`).
    std::array<float, 12> toWorld{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    float                 focalLength = 50.0F;          ///< in the aperture's units (tenths of a scene unit)
    float                 horizontalAperture = 20.955F;
    float                 nearClip = 1.0F;              ///< the clipping range's near end, scene units
};

struct MeshStageOptions {
    /// A prim path: only meshes at or under it are read. Empty is the lot.
    std::string prim;
    /// Meshes with no `st` primvar are read anyway; their gaussians then take
    /// the triangle's own size rather than a texel's. False leaves them out.
    bool        withoutTexcoords = true;
    /// The USD time code the geometry is read at: the pose a conversion turns
    /// into gaussians. A skinned stage is posed for it first, into a session
    /// layer, so the file on disk is not touched; a stage with no animation in
    /// it reads exactly as it did, because an attribute with no time samples
    /// answers with its default whatever time is asked for.
    double      time = 0.0;
    /// Read the **bind** pose and each point's joint influences, rather than
    /// the pose at `time`. What a cloud that is to be carried by a skeleton
    /// needs: its gaussians stand where the skeleton's transforms expect
    /// them, and posing the stage first would apply the skinning twice.
    bool        skinned = false;
    /// Prims made invisible for this read, as session opinions: a host's
    /// hidden prims, so a conversion holds what that host draws. Nothing on
    /// disk changes.
    std::vector<std::string> hidden;
    /// Materials (prim paths, or their names) whose glass is a sheet, though
    /// the material does not say so: a windscreen modelled as one surface
    /// with a solid glass bound to it. Read as thin-walled.
    std::vector<std::string> thinGlass;
};

/// Opens `path` and reads its meshes. The stage stays open for as long as this
/// object does, because the arrays handed to the builder are spans over it.
class MeshStage {
public:
    [[nodiscard]] static Result<MeshStage> open(const std::filesystem::path& path);

    MeshStage(MeshStage&&) noexcept;
    MeshStage& operator=(MeshStage&&) noexcept;
    ~MeshStage();

    /// Builds every mesh the options ask for. The order is the stage's.
    [[nodiscard]] Result<std::vector<StageMesh>> read(geom::MeshBuilder& builder,
                                                      const MeshStageOptions& options = {});

    /// The joints of `skeleton` at each of `times`, in the skeleton's own
    /// order: `times.size() * joints * 16` floats, row major as USD holds
    /// them. What a cloud that carries its rig writes, and the only thing
    /// about such a cloud that changes from one frame to the next.
    [[nodiscard]] Result<std::vector<float>> skeletonTransforms(const std::string& skeleton,
                                                                const std::vector<double>& times) const;

    /// The stage's own time range, for a conversion that was given none.
    [[nodiscard]] std::pair<double, double> timeRange() const;

    /// The stage's own rate. A layer that does not say is taken at 24 by USD,
    /// and composing it under a root that says 30 scales every time sample it
    /// holds by 30/24 -- so a written cloud has to carry the rate it was
    /// sampled at or it plays slow.
    [[nodiscard]] double timeCodesPerSecond() const;

    /// The joints of a Skeleton prim, as the paths its `joints` attribute
    /// holds, in its order -- which is the order a cloud's joint indices use.
    /// Bookkeeping over names, for partitioning a rig into parts.
    [[nodiscard]] Result<std::vector<std::string>> joints(const std::string& skeleton) const;

    /// Which way it stood: 'y' or 'z'. The gaussians are in that stage's
    /// world space, so the cloud written from them has to say the same.
    [[nodiscard]] char upAxis() const;
    /// The stage's metersPerUnit, USD's fallback included.
    [[nodiscard]] double metersPerUnit() const;

    /// The `UsdGeomCamera` at `path`, at `time`: its world transform, focal
    /// length, horizontal aperture and near clip, as authored.
    [[nodiscard]] Result<StageCamera> camera(const std::string& path, double time) const;

    /// The layers the stage was opened from, for a message.
    [[nodiscard]] std::string source() const;

    struct Impl;

private:
    MeshStage() = default;
    std::unique_ptr<Impl> impl_;
};

}   // namespace athenea::usd
