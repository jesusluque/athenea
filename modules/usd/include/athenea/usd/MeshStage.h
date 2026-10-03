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
    StageTexture           albedo;
    StageTexture           normal;
    StageTexture           metallicMap;
    StageTexture           roughnessMap;
    /// A cut-out: where this reads below a half the surface is not there at
    /// all. It is not the same thing as `transmission`, which is a surface
    /// you see through.
    StageTexture           opacityMap;
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

    /// The layers the stage was opened from, for a message.
    [[nodiscard]] std::string source() const;

    struct Impl;

private:
    MeshStage() = default;
    std::unique_ptr<Impl> impl_;
};

}   // namespace athenea::usd
