// Copyright (c) 2026 jesus luque.
//
// mesh2splat's conversion, as an AOFX effect: a textured mesh in, gaussian
// splats out. The algorithm is mesh2splat's (Electronic Arts; BSD 3-Clause
// with a clause on EA's marks -- the notice and the licence are at the head of
// mesh2splat.slang, the file derived from it, and in THIRD_PARTY_NOTICES.md);
// this file is the declaration and the plumbing.
//
// WHY A MESH ARRIVES AS A PICTURE
//
// An AOFX effect is handed images and gives one back. That is the contract,
// and a mesh is not an image -- so the host packs the triangles into one:
// six float4 entries a triangle, `(position, u)` and `(normal, v)` for each of
// its three corners, laid out row by row. The gaussians leave the same way, as
// records in the engine's own encoding, four entries each and six where the
// PBR channels are asked for. What the host cannot know beforehand is how many
// there will be, so it gives a budget (`maxSplats`) and this effect attaches
// the count it actually wrote.
//
// Nothing here knows about USD, the engine, or where the mesh came from. The
// host reads the stage, and this runs on whatever device the machine has.
#include <aofx/Effect.h>
#include <aofx/Entry.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "aofx_kernels_mesh2splat.h"

namespace {

/// Must match `M2sParams` in mesh2splat.slang exactly.
struct Mesh2SplatUniforms {
    uint32_t triangles = 0;
    uint32_t meshWidth = 0;
    uint32_t meshStride = 0;
    uint32_t recordPixels = 4;

    uint32_t dstWidth = 0;
    uint32_t dstHeight = 0;
    uint32_t dstStride = 0;
    uint32_t maxSplats = 0;

    float boundsMin[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float boundsExtent[4] = {1.0F, 1.0F, 1.0F, 1.0F};

    uint32_t resolution = 512;
    uint32_t maxCells = 1u << 16;
    float    sigmaX = 0.65F;
    float    sigmaY = 0.65F;

    float    flatness = 0.1F;
    float    opacity = 1.0F;
    float    transmission = 0.0F;
    uint32_t useNormalMap = 0;

    float materialColour[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float transmissionColour[4] = {1.0F, 1.0F, 1.0F, 1.0F};

    uint32_t countWidth = 1;
    float    metallic = 0.0F;
    float    roughness = 0.5F;
    uint32_t countStride = 1;

    uint32_t hasAlbedo = 0;
    uint32_t albedoWidth = 0;
    uint32_t albedoHeight = 0;
    uint32_t albedoStride = 0;

    uint32_t hasNormal = 0;
    uint32_t normalWidth = 0;
    uint32_t normalHeight = 0;
    uint32_t normalStride = 0;

    uint32_t hasMetallicRoughness = 0;
    uint32_t mrWidth = 0;
    uint32_t mrHeight = 0;
    uint32_t mrStride = 0;

    uint32_t hasInfluences = 0;
    /// 0 none, else the channel the cut-out is read from: 1 r, 2 g, 3 b, 4 a.
    uint32_t opacityChannel = 0;
    uint32_t opacityWidth = 0;
    uint32_t opacityHeight = 0;

    uint32_t opacityStride = 0;
    float    opacityCut = 0.5F;
    /// A second set of texture coordinates (the Texcoord2 clip), and which
    /// maps read by it rather than by the mesh clip's own.
    uint32_t hasUv2 = 0;
    uint32_t albedoUv2 = 0;

    uint32_t normalUv2 = 0;
    uint32_t mrUv2 = 0;
    uint32_t opacityUv2 = 0;
    /// Triangles before this are an earlier slice's: a mesh whose gaussians
    /// do not fit one run's picture is converted in slices, in its own order.
    uint32_t firstTriangle = 0;

    /// The cell bounded in world units (0: no bound), and whether every
    /// triangle's cell is measured against the box's longest side.
    float    cellMin = 0.0F;
    float    cellMax = 0.0F;
    uint32_t cellByLongest = 0;
    /// What a fully transmitting material still stops, 1 keeping all of it.
    float    glassOpacity = 1.0F;

    /// A height along the normal: the Displacement clip read on this channel
    /// (1 r, 2 g, 3 b, 4 a; 0 no map, the bias alone), as `texel * scale +
    /// bias` in world units.
    uint32_t displaceChannel = 0;
    uint32_t displaceWidth = 0;
    uint32_t displaceHeight = 0;
    uint32_t displaceStride = 0;

    float    displaceScale = 1.0F;
    float    displaceBias = 0.0F;
    /// 0: no displacement. Otherwise the most gaussians a cell is split into
    /// along each of its two axes where the relief stretches it.
    uint32_t displaceRefine = 0;
    uint32_t displaceUv2 = 0;

    /// 0: a gaussian a cell. Otherwise how far the maps may move across a
    /// block of cells for the block to be one gaussian, and the largest block
    /// (2^levels cells a side).
    float    simplify = 0.0F;
    uint32_t simplifyLevels = 3;
    /// 1: the cut-out is a threshold (UsdPreviewSurface's `opacityThreshold`,
    /// glTF's MASK), and what it keeps is whole rather than its value.
    uint32_t opacityBinary = 0;
    /// 1: `opacity`, the mask and `glassOpacity` are the surface's coverage,
    /// and each gaussian takes what one of the stack over a point needs for
    /// it; 0: each gaussian's own opacity, as before.
    uint32_t coverage = 1;

    /// What the surface gives off: the Emission clip, where there is one,
    /// times `emissionColour`, written in a record entry of its own.
    uint32_t hasEmission = 0;
    uint32_t emissionWidth = 0;
    uint32_t emissionHeight = 0;
    uint32_t emissionStride = 0;

    float    emissionColour[4] = {0.0F, 0.0F, 0.0F, 0.0F};

    /// 0: not written. Otherwise the entry of a record it goes in: the last.
    uint32_t emissionEntry = 0;
    /// 0: the clip's rgb; else one channel as grey, 1 r, 2 g, 3 b, 4 a.
    uint32_t emissionChannel = 0;
    uint32_t emissionUv2 = 0;
    uint32_t emissionPad = 0;

    /// 1: the glass is a sheet, and covers what it reflects at `ior` rather
    /// than `glassOpacity`.
    uint32_t thinWall = 0;
    float    ior = 1.5F;
    /// The material's own coverage (its constant opacity), multiplied into
    /// `opacity`: the user's and the material's are two numbers.
    float    materialOpacity = 1.0F;
    uint32_t coveragePad = 0;
};
static_assert(sizeof(Mesh2SplatUniforms) == 368, "must match M2sParams exactly");

class Mesh2Splat final : public aofx::Effect {
public:
    void describe(aofx::EffectDesc& into) override {
        into.identifier = "rt.sparrow.aofx.mesh2splat";
        into.label = "Mesh to Splats";
        into.grouping = "Convert";
        into.description =
            "Turns a textured mesh into 3D gaussian splats: a gaussian per texel of the "
            "triangle's own projection, as wide as that texel is in the world. Based on "
            "mesh2splat (Electronic Arts, BSD-3-Clause+).";

        // The mesh is a picture of numbers, not a picture: the host packs it,
        // and nothing upstream of this node should be asked to show it.
        const auto clip = [&into](const char* name, const char* label, bool optional) {
            aofx::ClipDesc desc;
            desc.name = name;
            desc.label = label;
            desc.optional = optional;
            into.inputs.push_back(desc);
        };
        clip("Mesh", "Mesh", false);
        clip("Albedo", "Albedo", true);
        clip("Normal", "Normal map", true);
        clip("MetallicRoughness", "Metallic-roughness", true);
        // A cut-out, not a transmission: where it reads below the cut there is
        // no surface, so no gaussian is counted and none is written.
        clip("Opacity", "Opacity cut-out", true);
        // Not a picture of colour: the joints each corner of each triangle is
        // carried by, laid out exactly as the Mesh clip is.
        clip("Influences", "Joint influences", true);
        // A mesh with two sets of texture coordinates: the first rides in the
        // Mesh clip, this is the second, laid out exactly as the Mesh clip is,
        // and `albedoUv2`, `normalUv2`, `mrUv2` and `opacityUv2` say which
        // maps are read by it.
        clip("Texcoord2", "Second texture coordinates", true);
        // A height map: each gaussian stands off the surface by what it reads
        // there, along the normal, and turns with the relief.
        clip("Displacement", "Displacement height", true);
        // What the surface gives off by itself: multiplied into
        // `emissionColour`, its rgb or one channel of it (`emissionChannel`).
        clip("Emission", "Emission", true);

        aofx::ParamDesc opacityChannel;
        opacityChannel.name = "opacityChannel";
        opacityChannel.label = "Cut-out channel";
        opacityChannel.hint =
            "Which channel of the opacity clip holds the cut-out: 1 red, 2 green, 3 blue, 4 "
            "alpha. A feather atlas is usually the alpha of a map that holds something else.";
        opacityChannel.type = aofx::ParamType::Integer;
        opacityChannel.defaults = {4.0};
        opacityChannel.hardMin = {1.0};
        opacityChannel.hardMax = {4.0};
        into.params.push_back(opacityChannel);

        aofx::ParamDesc opacityCut;
        opacityCut.name = "opacityCut";
        opacityCut.label = "Cut at";
        opacityCut.hint =
            "Below this the surface is not there. A cut-out is not a transmission: a gaussian "
            "under the cut is not written at all, so the budget goes to the surface that exists.";
        opacityCut.type = aofx::ParamType::Double;
        opacityCut.defaults = {0.5};
        opacityCut.hardMin = {0.0};
        opacityCut.hardMax = {1.0};
        into.params.push_back(opacityCut);

        aofx::ParamDesc opacityBinary;
        opacityBinary.name = "opacityBinary";
        opacityBinary.label = "Cut-out is a threshold";
        opacityBinary.hint =
            "Where the cut is the material's own threshold (UsdPreviewSurface's opacityThreshold, "
            "glTF's MASK), what it keeps is the whole surface rather than the value it read.";
        opacityBinary.type = aofx::ParamType::Boolean;
        opacityBinary.defaults = {0.0};
        into.params.push_back(opacityBinary);

        aofx::ParamDesc coverage;
        coverage.name = "coverage";
        coverage.label = "Opacity is coverage";
        coverage.hint =
            "Opacity, the cut-out's value and the glass opacity say how much of what stands behind "
            "the surface it covers, and each gaussian takes what one of the several over a point "
            "needs for that. Off, they are each gaussian's own opacity, and a surface of overlapping "
            "gaussians covers far more than they say: a mask of 0.5 covered 96%.";
        coverage.type = aofx::ParamType::Boolean;
        coverage.defaults = {1.0};
        into.params.push_back(coverage);

        aofx::ParamDesc thinWall;
        thinWall.name = "thinWall";
        thinWall.label = "Thin wall";
        thinWall.hint =
            "The glass is a sheet: it sends what it does not reflect straight on, so it covers only what "
            "it reflects head on at its index, 2R/(1+R), instead of the glass opacity.";
        thinWall.type = aofx::ParamType::Boolean;
        thinWall.defaults = {0.0};
        into.params.push_back(thinWall);

        aofx::ParamDesc ior;
        ior.name = "ior";
        ior.label = "Index";
        ior.hint = "The thin wall's index of refraction.";
        ior.type = aofx::ParamType::Double;
        ior.defaults = {1.5};
        ior.hardMin = {1.0};
        into.params.push_back(ior);

        aofx::ParamDesc materialOpacity;
        materialOpacity.name = "materialOpacity";
        materialOpacity.label = "Material opacity";
        materialOpacity.hint = "The material's own constant coverage, multiplied into the opacity.";
        materialOpacity.type = aofx::ParamType::Double;
        materialOpacity.defaults = {1.0};
        materialOpacity.hardMin = {0.0};
        materialOpacity.hardMax = {1.0};
        into.params.push_back(materialOpacity);

        aofx::ParamDesc triangles;
        triangles.name = "triangles";
        triangles.label = "Triangles";
        triangles.hint = "How many triangles the mesh image holds.";
        triangles.type = aofx::ParamType::Integer;
        triangles.defaults = {0.0};
        triangles.hardMin = {0.0};
        triangles.hardMax = {1.0e9};
        into.params.push_back(triangles);

        aofx::ParamDesc resolution;
        resolution.name = "resolution";
        resolution.label = "Resolution";
        resolution.hint =
            "Cells across the longest side of the box the density is measured over -- the model's, "
            "or the mesh's own where the host attaches it. This is the density: it is the resolution "
            "mesh2splat renders the projected triangles at, and a gaussian stands on every cell "
            "one covers.";
        resolution.type = aofx::ParamType::Integer;
        resolution.defaults = {512.0};
        resolution.hardMin = {1.0};
        resolution.hardMax = {65536.0};
        resolution.displayMin = {32.0};
        resolution.displayMax = {4096.0};
        into.params.push_back(resolution);

        aofx::ParamDesc maxSplats;
        maxSplats.name = "maxSplats";
        maxSplats.label = "Most splats";
        maxSplats.hint =
            "The budget. Past it nothing more is written and the count says how many were "
            "wanted, so a second run can be given room.";
        maxSplats.type = aofx::ParamType::Integer;
        maxSplats.defaults = {1000000.0};
        maxSplats.hardMin = {1.0};
        into.params.push_back(maxSplats);

        aofx::ParamDesc sigma;
        sigma.name = "sigma";
        sigma.label = "Sigma";
        sigma.hint = "How wide a gaussian is against its texel. 0.65 is mesh2splat's.";
        sigma.type = aofx::ParamType::Double;
        sigma.dimension = 2;
        sigma.defaults = {0.65, 0.65};
        sigma.hardMin = {0.001};
        sigma.hardMax = {4.0};
        into.params.push_back(sigma);

        aofx::ParamDesc flatness;
        flatness.name = "flatness";
        flatness.label = "Thickness";
        flatness.hint = "The third size, as a fraction of the smaller of the other two. A length instead of a fraction makes how thin a gaussian is depend on how big the model is, and a ray tracer sees a ghost where a rasteriser sees a surface.";
        flatness.type = aofx::ParamType::Double;
        flatness.defaults = {0.1};
        flatness.hardMin = {0.0};
        into.params.push_back(flatness);

        aofx::ParamDesc opacity;
        opacity.name = "opacity";
        opacity.label = "Opacity";
        opacity.type = aofx::ParamType::Double;
        opacity.defaults = {1.0};
        opacity.hardMin = {0.0};
        opacity.hardMax = {1.0};
        into.params.push_back(opacity);

        aofx::ParamDesc colour;
        colour.name = "materialColour";
        colour.label = "Colour";
        colour.hint = "Multiplied into the albedo, as their material factor is.";
        colour.type = aofx::ParamType::Colour;
        colour.dimension = 3;
        colour.defaults = {1.0, 1.0, 1.0};
        into.params.push_back(colour);

        aofx::ParamDesc transmission;
        transmission.name = "transmission";
        transmission.label = "Transmission";
        transmission.hint =
            "What the material lets through, which a gaussian cannot: it keeps 1 - this much "
            "opacity and takes the transmission colour. Not part of mesh2splat.";
        transmission.type = aofx::ParamType::Double;
        transmission.defaults = {0.0};
        transmission.hardMin = {0.0};
        transmission.hardMax = {1.0};
        into.params.push_back(transmission);

        aofx::ParamDesc glass;
        glass.name = "glassOpacity";
        glass.label = "Glass Opacity";
        glass.hint =
            "What a fully transmitting material still covers, scaled by how much it transmits: 1 "
            "keeps the surface whole, which is what a gaussian did before this existed, and low "
            "is a window you can see through. A translucent material is not a transparent one, "
            "so the default keeps everything and the caller says otherwise.";
        glass.type = aofx::ParamType::Double;
        glass.defaults = {1.0};
        glass.hardMin = {0.0};
        glass.hardMax = {1.0};
        into.params.push_back(glass);

        aofx::ParamDesc metallic;
        metallic.name = "metallic";
        metallic.label = "Metallic";
        metallic.hint =
            "What the material is worth on its own, multiplied into the map where there is one, as "
            "glTF multiplies its factors.";
        metallic.type = aofx::ParamType::Double;
        metallic.defaults = {0.0};
        metallic.hardMin = {0.0};
        metallic.hardMax = {1.0};
        into.params.push_back(metallic);

        aofx::ParamDesc roughness;
        roughness.name = "roughness";
        roughness.label = "Roughness";
        roughness.hint = "The same, for how rough the surface is. A gaussian with none is plastic.";
        roughness.type = aofx::ParamType::Double;
        roughness.defaults = {0.5};
        roughness.hardMin = {0.0};
        roughness.hardMax = {1.0};
        into.params.push_back(roughness);

        aofx::ParamDesc tint;
        tint.name = "transmissionColour";
        tint.label = "Transmission colour";
        tint.type = aofx::ParamType::Colour;
        tint.dimension = 3;
        tint.defaults = {1.0, 1.0, 1.0};
        into.params.push_back(tint);

        aofx::ParamDesc useNormalMap;
        useNormalMap.name = "useNormalMap";
        useNormalMap.label = "Turn by the normal map";
        useNormalMap.hint =
            "Orient each gaussian by the sampled normal rather than the surface's own. Off, the "
            "map is still carried in the PBR channels.";
        useNormalMap.type = aofx::ParamType::Boolean;
        useNormalMap.defaults = {0.0};
        into.params.push_back(useNormalMap);

        aofx::ParamDesc pbr;
        pbr.name = "writePbr";
        pbr.label = "Write PBR channels";
        pbr.hint =
            "Six entries a record instead of four: the shading normal, metallic, roughness and "
            "the texture coordinate travel with every gaussian.";
        pbr.type = aofx::ParamType::Boolean;
        pbr.defaults = {1.0};
        into.params.push_back(pbr);

        aofx::ParamDesc joints;
        joints.name = "writeInfluences";
        joints.label = "Write joint influences";
        joints.hint =
            "Three entries more a record: the four joints the gaussian is carried by and how much, "
            "blended from the corners of its triangle, and how those weights change across it "
            "(three joints' gradients along its two axes, as halves; the fourth's is minus their "
            "sum). What a cloud a skeleton deforms needs, and it wants the Influences clip and the "
            "PBR entries with it.";
        joints.type = aofx::ParamType::Boolean;
        joints.defaults = {0.0};
        into.params.push_back(joints);

        // Which maps read by the second set of coordinates (the Texcoord2
        // clip) rather than by the Mesh clip's own.
        for (const char* name : {"albedoUv2", "normalUv2", "mrUv2", "opacityUv2", "displaceUv2", "emissionUv2"}) {
            aofx::ParamDesc bySecond;
            bySecond.name = name;
            bySecond.label = std::string(name) + ": read by the second coordinates";
            bySecond.hint = "The map is sampled by the Texcoord2 clip's coordinates, not the Mesh clip's.";
            bySecond.type = aofx::ParamType::Boolean;
            bySecond.defaults = {0.0};
            into.params.push_back(bySecond);
        }

        // THE LIGHT IT GIVES OFF. Additive, as the ABI asks: a host that sends
        // none of these gets records exactly as before.
        aofx::ParamDesc emission;
        emission.name = "writeEmission";
        emission.label = "Write emission";
        emission.hint =
            "One entry more a record, the last: the linear radiance the surface gives off there "
            "(emissionColour, times the Emission clip where there is one).";
        emission.type = aofx::ParamType::Boolean;
        emission.defaults = {0.0};
        into.params.push_back(emission);

        aofx::ParamDesc emissionColour;
        emissionColour.name = "emissionColour";
        emissionColour.label = "Emission";
        emissionColour.hint =
            "What the material gives off, linear and unbounded: its colour times its weight. "
            "Multiplied into the Emission clip where there is one.";
        emissionColour.type = aofx::ParamType::Colour;
        emissionColour.dimension = 3;
        emissionColour.defaults = {0.0, 0.0, 0.0};
        into.params.push_back(emissionColour);

        aofx::ParamDesc emissionChannel;
        emissionChannel.name = "emissionChannel";
        emissionChannel.label = "Emission channel";
        emissionChannel.hint =
            "0 reads the Emission clip's colour; 1 red, 2 green, 3 blue, 4 alpha read that channel "
            "as grey -- a map on the emission's weight rather than its colour.";
        emissionChannel.type = aofx::ParamType::Integer;
        emissionChannel.defaults = {0.0};
        emissionChannel.hardMin = {0.0};
        emissionChannel.hardMax = {4.0};
        into.params.push_back(emissionChannel);

        aofx::ParamDesc displace;
        displace.name = "displace";
        displace.label = "Displace";
        displace.hint =
            "Stand each gaussian off the surface by the height the material gives: the "
            "Displacement clip on `displaceChannel`, times `displaceScale`, plus `displaceBias`, "
            "in world units. Three entries more a record: the point it stands on with the "
            "height, the surface's normal, and the relief's.";
        displace.type = aofx::ParamType::Boolean;
        displace.defaults = {0.0};
        into.params.push_back(displace);
        for (const auto& [name, label, fallback] :
             {std::tuple{"displaceChannel", "Height channel", 1.0}, std::tuple{"displaceScale", "Height scale", 1.0},
              std::tuple{"displaceBias", "Height bias", 0.0},
              std::tuple{"displaceRefine", "Most gaussians a cell, along each axis", 8.0}}) {
            aofx::ParamDesc desc;
            desc.name = name;
            desc.label = label;
            desc.hint = std::string(name) == "displaceRefine"
                            ? "Where the relief stretches a cell, it is split into as many gaussians "
                              "along each axis as the stretch asks for, up to this."
                            : "How the Displacement clip is read into a height.";
            desc.type = std::string(name) == "displaceChannel" || std::string(name) == "displaceRefine"
                            ? aofx::ParamType::Integer
                            : aofx::ParamType::Double;
            desc.defaults = {fallback};
            into.params.push_back(desc);
        }

        aofx::ParamDesc simplify;
        simplify.name = "simplify";
        simplify.label = "Simplify";
        simplify.hint =
            "Where the surface is the same across a block of cells -- colour, metallic, roughness, "
            "cut-out and normal each moving by no more than this -- the block is one gaussian of "
            "its size. 0 is a gaussian a cell.";
        simplify.type = aofx::ParamType::Double;
        simplify.defaults = {0.0};
        simplify.hardMin = {0.0};
        into.params.push_back(simplify);
        aofx::ParamDesc simplifyLevels;
        simplifyLevels.name = "simplifyLevels";
        simplifyLevels.label = "Largest block";
        simplifyLevels.hint = "The largest block that may become one gaussian is 2^this cells a side.";
        simplifyLevels.type = aofx::ParamType::Integer;
        simplifyLevels.defaults = {3.0};
        simplifyLevels.hardMin = {1.0};
        simplifyLevels.hardMax = {5.0};
        into.params.push_back(simplifyLevels);

        aofx::ParamDesc first;
        first.name = "firstTriangle";
        first.label = "First triangle";
        first.hint = "Triangles before this are not converted: a mesh whose gaussians do not fit one "
                     "run is converted in slices, each starting where the budget cut the last.";
        first.type = aofx::ParamType::Integer;
        first.defaults = {0.0};
        first.hardMin = {0.0};
        into.params.push_back(first);

        aofx::ParamDesc cells;
        cells.name = "maxCells";
        cells.label = "Most cells a triangle";
        cells.hint =
            "One thread walks one triangle's cells, so a triangle far larger than the rest is "
            "bounded here. What it costs is counted and reported.";
        cells.type = aofx::ParamType::Integer;
        cells.defaults = {65536.0};
        cells.hardMin = {1.0};
        into.params.push_back(cells);

        // The cell, bounded in world units: additive parameters, as the ABI
        // asks; a host that does not send them gets the unbounded cell.
        aofx::ParamDesc cellMin;
        cellMin.name = "cellMin";
        cellMin.label = "Smallest cell";
        cellMin.hint = "No cell finer than this, in the mesh's units; 0 leaves it to the resolution.";
        cellMin.type = aofx::ParamType::Double;
        cellMin.defaults = {0.0};
        cellMin.hardMin = {0.0};
        into.params.push_back(cellMin);

        aofx::ParamDesc cellMax;
        cellMax.name = "cellMax";
        cellMax.label = "Largest cell";
        cellMax.hint = "No cell coarser than this, in the mesh's units; 0 leaves it to the resolution.";
        cellMax.type = aofx::ParamType::Double;
        cellMax.defaults = {0.0};
        cellMax.hardMin = {0.0};
        into.params.push_back(cellMax);

        aofx::ParamDesc byLongest;
        byLongest.name = "cellByLongest";
        byLongest.label = "One cell a mesh";
        byLongest.hint =
            "Measure every triangle's cell against the box's longest side, so a mesh has one cell "
            "whichever way its triangles face. Off, a triangle's cell is the wider of its "
            "projection's two ranges, as mesh2splat has it.";
        byLongest.type = aofx::ParamType::Boolean;
        byLongest.defaults = {0.0};
        into.params.push_back(byLongest);
    }

    [[nodiscard]] std::vector<aofx::KernelDesc> kernels() const override {
        return {aofx::KernelDesc{"m2sClear", "m2sClear", k_mesh2splat, k_mesh2splatBytes},
                aofx::KernelDesc{"m2sCount", "m2sCount", k_mesh2splat, k_mesh2splatBytes},
                aofx::KernelDesc{"m2sScan", "m2sScan", k_mesh2splat, k_mesh2splatBytes},
                aofx::KernelDesc{"m2sEmit", "m2sEmit", k_mesh2splat, k_mesh2splatBytes}};
    }

    bool process(const aofx::RenderRequest& request) override {
        const aofx::InputPlane*  meshPlane = request.input("Mesh");
        const aofx::OutputPlane* target = request.output("Color");
        if (meshPlane == nullptr || target == nullptr || !meshPlane->buffer.isValid() ||
            !target->buffer.isValid() || request.gpu == nullptr) {
            request.complaint = "mesh2splat needs its Mesh input and somewhere to write";
            return false;
        }
        const auto triangles = static_cast<uint32_t>(std::max(request.number("triangles", 0.0), 0.0));
        if (triangles == 0) {
            request.complaint = "mesh2splat was given no triangles: the host sets 'triangles'";
            return false;
        }

        Mesh2SplatUniforms uniforms;
        uniforms.triangles = triangles;
        uniforms.meshWidth = static_cast<uint32_t>(meshPlane->buffer.width);
        uniforms.meshStride = static_cast<uint32_t>(meshPlane->buffer.stride);
        uniforms.recordPixels = request.number("writePbr", 1.0) >= 0.5 ? 6U : 4U;
        // The joints a gaussian is carried by ride in three more entries --
        // two of (joint, weight) and one of the weights' gradients -- and
        // they need the PBR ones ahead of them: 4, 6, 9 and nothing between.
        const aofx::InputPlane* carried = request.input("Influences");
        const bool withJoints = request.number("writeInfluences", 0.0) >= 0.5 && carried != nullptr &&
                                carried->buffer.isValid();
        if (withJoints) {
            uniforms.recordPixels = 9U;
            uniforms.hasInfluences = 1U;
        }
        // A displaced gaussian carries three entries more: where the surface
        // under it is and how high it stands, the surface's normal, and the
        // relief's -- what a bake starts from, and what moving the relief
        // later needs.
        const aofx::InputPlane* heights = request.input("Displacement");
        const bool withHeights = heights != nullptr && heights->buffer.isValid();
        const bool displaces = request.number("displace", 0.0) >= 0.5;
        if (displaces) {
            uniforms.recordPixels += 3U;
            uniforms.displaceRefine = static_cast<uint32_t>(
                std::clamp(request.number("displaceRefine", 8.0), 1.0, 64.0));
            uniforms.displaceScale = static_cast<float>(request.number("displaceScale", 1.0));
            uniforms.displaceBias = static_cast<float>(request.number("displaceBias", 0.0));
            if (withHeights) {
                uniforms.displaceChannel =
                    static_cast<uint32_t>(std::clamp(request.number("displaceChannel", 1.0), 1.0, 4.0));
                uniforms.displaceWidth = static_cast<uint32_t>(heights->buffer.width);
                uniforms.displaceHeight = static_cast<uint32_t>(heights->buffer.height);
                uniforms.displaceStride = static_cast<uint32_t>(heights->buffer.stride);
            }
        }
        // And what the surface gives off, in one entry after all of those.
        const aofx::InputPlane* emitted = request.input("Emission");
        const bool withEmissionMap = emitted != nullptr && emitted->buffer.isValid();
        if (request.number("writeEmission", 0.0) >= 0.5) {
            uniforms.emissionEntry = uniforms.recordPixels;
            uniforms.recordPixels += 1U;
            for (int k = 0; k < 3; ++k) {
                uniforms.emissionColour[k] = static_cast<float>(
                    std::max(request.number("emissionColour", 0.0, static_cast<size_t>(k)), 0.0));
            }
            if (withEmissionMap) {
                uniforms.hasEmission = 1U;
                uniforms.emissionWidth = static_cast<uint32_t>(emitted->buffer.width);
                uniforms.emissionHeight = static_cast<uint32_t>(emitted->buffer.height);
                uniforms.emissionStride = static_cast<uint32_t>(emitted->buffer.stride);
                uniforms.emissionChannel =
                    static_cast<uint32_t>(std::clamp(request.number("emissionChannel", 0.0), 0.0, 4.0));
            }
        }
        uniforms.dstWidth = static_cast<uint32_t>(target->buffer.width);
        uniforms.dstHeight = static_cast<uint32_t>(target->buffer.height);
        uniforms.dstStride = static_cast<uint32_t>(target->buffer.stride);

        // The budget is what was asked for, and never more than the picture it
        // is written into can hold.
        const auto room = static_cast<uint64_t>(uniforms.dstWidth) * uniforms.dstHeight /
                          uniforms.recordPixels;
        const auto asked = static_cast<uint64_t>(std::max(request.number("maxSplats", 1.0e6), 1.0));
        uniforms.maxSplats = static_cast<uint32_t>(std::min(room, asked));
        if (uniforms.maxSplats == 0) {
            request.complaint = "mesh2splat was given no room to write a single splat";
            return false;
        }

        // The model's box, from the numbers the host attached to the mesh
        // image: six floats, low corner then high.
        float low[3] = {0.0F, 0.0F, 0.0F};
        float high[3] = {1.0F, 1.0F, 1.0F};
        if (const std::vector<float>* box = meshPlane->value("bounds");
            box != nullptr && box->size() >= 6) {
            for (size_t k = 0; k < 3; ++k) {
                low[k] = (*box)[k];
                high[k] = (*box)[k + 3];
            }
        }
        float longest = 0.0F;
        for (int k = 0; k < 3; ++k) {
            uniforms.boundsMin[k] = low[k];
            uniforms.boundsExtent[k] = high[k] - low[k];
            longest = std::max(longest, uniforms.boundsExtent[k]);
        }
        uniforms.boundsExtent[3] = longest > 0.0F ? longest : 1.0F;

        uniforms.resolution =
            static_cast<uint32_t>(std::clamp(request.number("resolution", 512.0), 1.0, 65536.0));
        uniforms.maxCells =
            static_cast<uint32_t>(std::max(request.number("maxCells", 65536.0), 1.0));
        uniforms.cellMin = static_cast<float>(std::max(request.number("cellMin", 0.0), 0.0));
        uniforms.cellMax = static_cast<float>(std::max(request.number("cellMax", 0.0), 0.0));
        uniforms.cellByLongest = request.number("cellByLongest", 0.0) != 0.0 ? 1u : 0u;
        uniforms.sigmaX = static_cast<float>(request.number("sigma", 0.65, 0));
        uniforms.sigmaY = static_cast<float>(request.number("sigma", 0.65, 1));
        uniforms.flatness = static_cast<float>(std::max(request.number("flatness", 0.1), 0.0));
        uniforms.opacity =
            static_cast<float>(std::clamp(request.number("opacity", 1.0), 0.0, 1.0));
        uniforms.transmission =
            static_cast<float>(std::clamp(request.number("transmission", 0.0), 0.0, 1.0));
        uniforms.glassOpacity =
            static_cast<float>(std::clamp(request.number("glassOpacity", 1.0), 0.0, 1.0));
        uniforms.useNormalMap = request.number("useNormalMap", 0.0) >= 0.5 ? 1U : 0U;
        uniforms.metallic = static_cast<float>(std::clamp(request.number("metallic", 0.0), 0.0, 1.0));
        uniforms.roughness = static_cast<float>(std::clamp(request.number("roughness", 0.5), 0.0, 1.0));
        for (int k = 0; k < 3; ++k) {
            uniforms.materialColour[k] =
                static_cast<float>(request.number("materialColour", 1.0, static_cast<size_t>(k)));
            uniforms.transmissionColour[k] = static_cast<float>(
                request.number("transmissionColour", 1.0, static_cast<size_t>(k)));
        }

        // Every map the kernel declares must be bound whether or not it is
        // there; `has...` is what stops it being read.
        const aofx::InputPlane* albedo = request.input("Albedo");
        const aofx::InputPlane* normal = request.input("Normal");
        const aofx::InputPlane* mr = request.input("MetallicRoughness");
        const aofx::InputPlane* cut = request.input("Opacity");
        const auto describe = [](const aofx::InputPlane* plane, uint32_t& has, uint32_t& width,
                                 uint32_t& height, uint32_t& stride) {
            const bool there = plane != nullptr && plane->buffer.isValid();
            has = there ? 1U : 0U;
            if (there) {
                width = static_cast<uint32_t>(plane->buffer.width);
                height = static_cast<uint32_t>(plane->buffer.height);
                stride = static_cast<uint32_t>(plane->buffer.stride);
            }
        };
        describe(albedo, uniforms.hasAlbedo, uniforms.albedoWidth, uniforms.albedoHeight,
                 uniforms.albedoStride);
        describe(normal, uniforms.hasNormal, uniforms.normalWidth, uniforms.normalHeight,
                 uniforms.normalStride);
        describe(mr, uniforms.hasMetallicRoughness, uniforms.mrWidth, uniforms.mrHeight,
                 uniforms.mrStride);
        uint32_t hasCut = 0;
        describe(cut, hasCut, uniforms.opacityWidth, uniforms.opacityHeight,
                 uniforms.opacityStride);
        uniforms.opacityChannel =
            hasCut != 0 ? static_cast<uint32_t>(request.number("opacityChannel", 4.0)) : 0U;
        uniforms.opacityCut = static_cast<float>(request.number("opacityCut", 0.5));
        uniforms.opacityBinary = request.number("opacityBinary", 0.0) >= 0.5 ? 1U : 0U;
        uniforms.coverage = request.number("coverage", 1.0) >= 0.5 ? 1U : 0U;
        uniforms.thinWall = request.number("thinWall", 0.0) >= 0.5 ? 1U : 0U;
        uniforms.ior = static_cast<float>(std::max(request.number("ior", 1.5), 1.0));
        uniforms.materialOpacity =
            static_cast<float>(std::clamp(request.number("materialOpacity", 1.0), 0.0, 1.0));
        const aofx::InputPlane* uv2 = request.input("Texcoord2");
        const bool withUv2 = uv2 != nullptr && uv2->buffer.isValid();
        uniforms.hasUv2 = withUv2 ? 1U : 0U;
        const auto byUv2 = [&](const char* name) {
            return withUv2 && request.number(name, 0.0) >= 0.5 ? 1U : 0U;
        };
        uniforms.albedoUv2 = byUv2("albedoUv2");
        uniforms.normalUv2 = byUv2("normalUv2");
        uniforms.mrUv2 = byUv2("mrUv2");
        uniforms.opacityUv2 = byUv2("opacityUv2");
        uniforms.displaceUv2 = byUv2("displaceUv2");
        uniforms.emissionUv2 = byUv2("emissionUv2");
        uniforms.simplify = static_cast<float>(std::max(request.number("simplify", 0.0), 0.0));
        uniforms.simplifyLevels = static_cast<uint32_t>(std::clamp(request.number("simplifyLevels", 3.0), 1.0, 5.0));
        uniforms.firstTriangle = static_cast<uint32_t>(
            std::clamp(request.number("firstTriangle", 0.0), 0.0, static_cast<double>(triangles)));

        // Five numbers: splats wanted, splats written, triangles with no
        // frame, cells the per-triangle bound left out, and the first
        // triangle the budget cut into. Two pixels hold eight.
        const aofx::Buffer counters = request.gpu->scratch(2, 1);
        if (!counters.isValid()) {
            request.complaint = "mesh2splat could not take the four numbers it counts with";
            return false;
        }
        // And one number a triangle: how many gaussians it stands, and then
        // where they start. Four to a pixel, in rows no wider than a picture
        // comfortably is.
        const uint32_t countPixels = (triangles + 3) / 4;
        const uint32_t countWidth = std::min<uint32_t>(countPixels, 4096);
        const uint32_t countHeight = (countPixels + countWidth - 1) / countWidth;
        const aofx::Buffer cellCounts = request.gpu->scratch(countWidth, countHeight);
        if (!cellCounts.isValid()) {
            request.complaint = "mesh2splat could not take a number for each of its triangles";
            return false;
        }
        uniforms.countWidth = static_cast<uint32_t>(cellCounts.width);
        uniforms.countStride = static_cast<uint32_t>(cellCounts.stride);

        const aofx::KernelId clear = request.gpu->load("m2sClear");
        const aofx::KernelId count = request.gpu->load("m2sCount");
        const aofx::KernelId scan = request.gpu->load("m2sScan");
        const aofx::KernelId emit = request.gpu->load("m2sEmit");
        if (clear == aofx::kInvalidKernel || count == aofx::kInvalidKernel ||
            scan == aofx::kInvalidKernel || emit == aofx::kInvalidKernel) {
            return false;
        }
        const std::vector<aofx::Buffer> buffers{
            meshPlane->buffer,
            albedo != nullptr && albedo->buffer.isValid() ? albedo->buffer : meshPlane->buffer,
            normal != nullptr && normal->buffer.isValid() ? normal->buffer : meshPlane->buffer,
            mr != nullptr && mr->buffer.isValid() ? mr->buffer : meshPlane->buffer,
            cut != nullptr && cut->buffer.isValid() ? cut->buffer : meshPlane->buffer,
            withJoints ? carried->buffer : meshPlane->buffer,
            counters,
            cellCounts,
            target->buffer,
            withUv2 ? uv2->buffer : meshPlane->buffer,
            withHeights ? heights->buffer : meshPlane->buffer,
            uniforms.hasEmission != 0 ? emitted->buffer : meshPlane->buffer};
        // Count, then settle where each triangle's gaussians start, then
        // write. The order of the output is the mesh's own, which is what
        // lets a gaussian be followed from one frame to the next.
        if (!request.gpu->run(clear, aofx::Grid{1, 1, 1}, buffers, &uniforms, sizeof(uniforms))) {
            return false;
        }
        if (!request.gpu->run(count, aofx::Grid{triangles, 1, 1}, buffers, &uniforms,
                              sizeof(uniforms))) {
            return false;
        }
        if (!request.gpu->run(scan, aofx::Grid{256, 1, 1}, buffers, &uniforms, sizeof(uniforms))) {
            return false;
        }
        if (!request.gpu->run(emit, aofx::Grid{triangles, 1, 1}, buffers, &uniforms,
                              sizeof(uniforms))) {
            return false;
        }

        uint32_t counted[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (!request.gpu->read(counters, counted, sizeof(counted))) {
            request.complaint = "mesh2splat could not read back how many splats it wrote";
            return false;
        }
        // What went out, in the order the host reads it: how many splats are
        // there, how many the mesh wanted, how many triangles had no frame to
        // stand a gaussian on, how many cells the per-triangle bound left
        // unwalked, and how long a record is.
        // And sixth, the first triangle the budget cut into (the triangle
        // count when nothing was): where a host's next slice starts.
        request.attach("splats", {static_cast<float>(counted[1]), static_cast<float>(counted[0]),
                                  static_cast<float>(counted[2]), static_cast<float>(counted[3]),
                                  static_cast<float>(uniforms.recordPixels), static_cast<float>(counted[4]),
                                  static_cast<float>(counted[5])});
        // A budget too small is not a failure: the splats that fit are real,
        // and the host is told what was wanted so it can say so or run again
        // with room. `complaint` is not the place for it -- the host ignores
        // that when `process` succeeds.
        return true;
    }
};

}   // namespace

AOFX_EXPORT_EFFECTS(Mesh2Splat)
