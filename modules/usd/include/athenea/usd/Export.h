// Copyright (c) 2026 jesus luque.
#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/io/RawSplats.h"
#include "athenea/lod/Lod.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::usd {

/// WHAT A CLOUD CARRIED BY A SKELETON WRITES BESIDE ITS GAUSSIANS.
///
/// Four joints a gaussian and how much each carries it, the transform out of
/// the cloud's space into the one the joints are measured from, and the
/// joints' own transforms at each of a set of instants. That last is the only
/// thing that changes from frame to frame, and for sixty joints it is four
/// kilobytes a frame -- which is why a cloud that carries its rig is
/// megabytes where one that carries per-frame arrays is hundreds of them.
struct SplatSkinning {
    /// The Skeleton prim's path: what `skel:skeleton` names.
    std::string           skeleton;
    /// Its joints by name, in the order the influences count them
    /// (`skel:joints`); empty means the Skeleton's own order.
    std::vector<std::string> jointNames;
    /// `(joint, weight)` four times a gaussian, in the order the records are.
    std::vector<float>    influences;
    std::array<float, 16> geomBindTransform{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    uint32_t              joints = 0;
    /// The time codes the transforms below were read at.
    std::vector<double>   times;
    /// What those codes are worth. Written on the stage, because a layer that
    /// is silent is read at 24 and stretched under a root that says otherwise.
    double                timeCodesPerSecond = 24.0;
    /// `times.size() * joints * 16` floats, row major as USD holds them.
    std::vector<float>    xforms;

    [[nodiscard]] bool valid() const noexcept {
        return joints > 0 && !times.empty() && !influences.empty() &&
               xforms.size() == times.size() * joints * 16;
    }
};

struct ExportOptions {
    uint32_t maxDegree = 3;
    /// Adds /World/Camera framing the cloud, so the stage renders as it is.
    bool     addCamera = true;
    /// COLMAP-trained clouds are y-down; turn them over on the prim's xform.
    double   rotateXDegrees = 0.0;
    /// Writes `primvars:athenea:splat:relight = 1` (AtheneaSplatLightingAPI): the
    /// colours are an albedo the scene's lights are to light, not radiance
    /// somebody captured. True for a cloud converted from a mesh, false for a
    /// capture, which carries the light it was shot under.
    bool     relight = false;
    /// Writes `primvars:athenea:splat:litBody = 1`: the colours are the light on
    /// the material's body, not an albedo, so a frame that relights this
    /// cloud adds the polish and nothing else. What `athenea mesh2splat` bakes.
    bool     litBody = false;
    /// Writes `primvars:athenea:splat:linear = 1`: the colours are linear
    /// light, not the sRGB a capture was trained in, and are drawn as they
    /// are. Also written whenever the records say so (`io::RawSplats::linear`,
    /// or colours encoded as `LinearLight`), so a cloud read back and written
    /// again keeps its space.
    bool     linear = false;
    /// Which way the stage the gaussians came from stood: 'y' or 'z'. They
    /// are in that stage's world space, so a cloud written as Y-up when they
    /// were laid out Z-up lies on its side -- which is what every asset out
    /// of Blender did.
    char     upAxis = 'y';
    /// The stage's linear unit, as UsdGeom keeps it. Written always: a stage
    /// that leaves it out is read as centimetres (USD's fallback is 0.01),
    /// and a cloud converted from a model in metres came out a hundredth of
    /// its size in any application that honours the unit.
    double   metersPerUnit = 1.0;
    /// The rig the cloud is carried by, or nothing. When it is there the
    /// stage takes a time range and the joints' transforms as time samples.
    const SplatSkinning* skinning = nullptr;
    /// WHICH PRIM EACH GAUSSIAN CAME FROM, as Cryptomatte names one: one id a
    /// record of `raw` (AtheneaSplatCryptomatteAPI), and what those ids are
    /// called. `athenea mesh2splat` knows both, since it converted the prims; a
    /// capture has neither, and is left out of the matte. Empty: nothing is
    /// written and the schema is not applied.
    std::span<const uint32_t>       cryptoObject;
    std::map<std::string, uint32_t> cryptoManifest;
    /// HOW MUCH OF AN ENVIRONMENT REACHES EACH GAUSSIAN: nine floats a record
    /// of the direct half (visibility times the cosine, the surface's own
    /// albedo taken as one), and twenty-seven more of the indirect one where
    /// it was baked. With them a cloud is lit by a sky it was never baked
    /// under, so the colours it carries are an albedo and nothing else.
    std::span<const float>          transferDirect;
    std::span<const float>          transferIndirect;
    /// WHICH WAYS OUT ARE OPEN: two ints a record, sixty-four bits of an
    /// 8 x 8 octahedral grid over the sphere, set where a traced ray found
    /// nothing. What lets a sun cast a hard shadow on a relit cloud.
    std::span<const int32_t>        shadowBits;
    /// One int a record, 1 where the gaussian came from a thin-walled glass.
    std::span<const int32_t>        thinWalled;
    /// `primvars:athenea:splat:ior` when above 1: the index the cloud's
    /// transmitting gaussians refract by. Not written at 0.
    float                           ior = 0.0F;
};

/// Writes `raw` as a UsdVolParticleField3DGaussianSplat at /World/Splats in a
/// new stage (.usda, .usdc or .usd). The values are computed on the GPU.
[[nodiscard]] Result<void> writeParticleFieldStage(gpu::ShaderLibrary& library,
                                                   const io::RawSplats& raw,
                                                   const std::filesystem::path& path,
                                                   const ExportOptions& options = {});

/// RECORDS THAT NEVER LEFT THE DEVICE: `count` of them in `records`, laid
/// out as `encoding` says, on the device of the library they are written
/// with. What `athenea mesh2splat` hands over -- the conversion, the bake and
/// the export's decode all on the device, and what crosses to the processor
/// only the values a USD array holds.
struct DeviceSplatRecords {
    std::string       source;
    uint32_t          count = 0;
    io::SplatEncoding encoding;
    gpu::Buffer       records;   ///< count * encoding.floatsPerRecord floats
    bool              linear = false;   ///< as io::RawSplats::linear
};

/// The same stage from records on the device.
[[nodiscard]] Result<void> writeParticleFieldStage(gpu::ShaderLibrary& library,
                                                   const DeviceSplatRecords& records,
                                                   const std::filesystem::path& path,
                                                   const ExportOptions& options = {});

/// A ParticleField read back as records: the first
/// UsdVolParticleField3DGaussianSplat of the stage at `path`, or `prim`
/// where it names one, at its default time. Position, linear opacity, linear
/// scales, rotation w x y z, the constant term as 3DGS keeps it and the rest
/// rgb per basis -- the values the file holds, laid out and not computed.
/// `where` says which prim was read, and `moved` whether that prim stands
/// under a transform that is not the identity (its records are in its own
/// space).
[[nodiscard]] Result<io::RawSplats> readParticleFieldRecords(const std::filesystem::path& path,
                                                             const std::string& prim = {},
                                                             std::string* where = nullptr, bool* moved = nullptr);

/// AN ARRAY A GAUSSIAN ELSE THE FIELD CARRIES: a vertex primvar (a
/// material's values, an id, a rig, a transfer, which parts see the sky) or
/// any other attribute one or more values a gaussian long, as 32-bit words --
/// floats as floats (halves and doubles widened), ints as ints -- a gaussian
/// at a time. What a decimation merges and writes back as the file had it.
struct GaussianArray {
    std::string name;           ///< the attribute's full name
    std::string typeName;       ///< its SdfValueTypeName, to write it back as it was
    bool        integer = false;
    uint32_t    width = 1;      ///< words a gaussian: components times the primvar's elementSize
    uint32_t    elementSize = 1;
    std::string interpolation;  ///< a primvar's; empty for a plain attribute
    /// The time codes it is sampled at; empty when it has a default alone.
    std::vector<double> times;
    /// The default's words (or the earliest sample's), then a sample's each.
    std::vector<std::vector<uint32_t>> values;
};

/// Every array a gaussian long on the field at `prim` (the first
/// ParticleField where empty), of `count` gaussians, besides the schema's
/// own positions, orientations, scales, opacities and harmonics -- which a
/// decimation writes from the gaussians themselves -- unless those are
/// sampled in time, when they are listed too.
[[nodiscard]] Result<std::vector<GaussianArray>> readGaussianArrays(const std::filesystem::path& path,
                                                                    const std::string& prim, uint32_t count);

/// THE STAGE AS IT WAS, WITH FEWER GAUSSIANS. The root layer of `input` is
/// copied to `output` -- the rig and its animation, the lights, the camera,
/// the Cryptomatte manifest, the variants, every constant primvar -- its
/// relative asset paths anchored to where it came from, and on the field at
/// `prim` the schema's arrays are replaced by those of the ParticleField at
/// /World/Splats of `gaussians` (a stage `writeParticleFieldStage` wrote)
/// and every array in `arrays` by its merged words.
[[nodiscard]] Result<void> writeDecimatedStage(const std::filesystem::path& input, const std::filesystem::path& output,
                                               const std::string& prim, const std::filesystem::path& gaussians,
                                               const std::vector<GaussianArray>& arrays);

/// What `decimateStage` is asked.
struct DecimateStageOptions {
    std::string           prim;          ///< the ParticleField to read; empty for the first
    uint32_t              degree = 3;    ///< harmonics kept
    bool                  addCamera = true;   ///< for a splat file only: a stage keeps its own
    lod::DecimateSettings settings;
};

/// FEWER GAUSSIANS, AND EVERYTHING THEY CARRIED. A stage's ParticleField --
/// or a splat file -- decimated (lod::Decimator), every array a gaussian
/// long merged over what each kept gaussian stands for (a rig as a rig, bits
/// as bits, an id as what may not be merged across, a float as a mean), and
/// written into a copy of the stage it came from (`writeDecimatedStage`).
[[nodiscard]] Result<lod::DecimateStats> decimateStage(gpu::ShaderLibrary& library, const std::filesystem::path& input,
                                                       const std::filesystem::path& output,
                                                       const DecimateStageOptions& options = {});

/// One level of a cloud converted at several cells: the stage it was written
/// to (its cloud at /World/Splats) and its cell, in its own units.
struct LodLevelFile {
    std::filesystem::path path;
    double                cell = 0.0;
};

/// A stage that draws `levels` as the levels of detail of one cloud: each
/// referenced as a ParticleField of its own under /World -- the finest at
/// /World/Splats, where a rig that names that prim still finds it -- with
/// `athenea:lod:group` saying they are one and `athenea:lod:cell` each one's cell.
/// Up axis and time range are the finest level's. Finest first.
[[nodiscard]] Result<void> writeLodAssembly(const std::filesystem::path& path,
                                            const std::vector<LodLevelFile>& levels, const std::string& group);

/// Writes a baked visibility onto the ParticleField at `prim` of the stage at
/// `path`, in place: `primvars:athenea:splat:visibilityParts` (12 floats a part)
/// and `primvars:athenea:splat:visibilityTexels` (two f16 a word), as
/// `athenea visibility` bakes them and `HdAtheneaParticleField` reads them back.
[[nodiscard]] Result<void> writeVisibility(const std::filesystem::path& path, const std::string& prim,
                                           std::span<const float> parts, std::span<const int32_t> texels,
                                           std::span<const int32_t> partOf, std::span<const int32_t> ambient);

}   // namespace athenea::usd
