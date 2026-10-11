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
    /// How those weights change across each gaussian: three words a gaussian
    /// in the same order, each two halves (d w_k / d u, d w_k / d v) along its
    /// rest axes for the first three joints, the fourth's minus their sum.
    /// Written as `primvars:athenea:splat:jointWeightGradients`; empty writes
    /// nothing, and the cloud is carried by the blend of its joints alone.
    std::vector<uint32_t> weightGradients;
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
    /// Writes `primvars:athenea:splat:surfel = 1`: some gaussians are surfels,
    /// a scale of exactly 0 (`athenea mesh2splat --flatness 0`, a merge that
    /// keeps discs flat), for a reader that draws them only when asked --
    /// Spark's enable2DGS.
    bool     surfel = false;
    /// Writes `primvars:athenea:splat:lodTree = 1`: the gaussians are a merge
    /// tree (`athenea decimate --tree`), its nodes each beside its parent --
    /// to be cut, not drawn whole.
    bool     lodTree = false;
    /// Writes `primvars:athenea:splat:catcher = 1`: the gaussians are a
    /// shadow catcher lying on a ground (`athenea mesh2splat --shadow-catcher`).
    /// A frame draws them black, each covering what the object took of the
    /// light reaching it, so they darken whatever is drawn behind them.
    bool     catcher = false;
    /// Writes `primvars:athenea:splat:lightCatcher = 1`: the gaussians lie on
    /// a ground and carry light on it (`athenea mesh2splat --light-catcher`).
    /// A frame draws only the lamps' layers they carry, added over whatever
    /// is behind them: they never darken it (docs/decisions.md, light groups).
    bool     lightCatcher = false;
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
    /// Coefficients a record of `transferDirect`: 9 (degree 2) or 16 (degree
    /// 3, a TX transfer's); the indirect half is three times as many.
    uint32_t                        transferCoefficients = 9;
    /// THE REFLECTED FIELD of a TX transfer (docs/decisions.md, task TX): 48
    /// floats a record, sixteen rgb coefficients of what arrives at the
    /// gaussian by direction after meeting the scene, under a white sky of
    /// radiance one. Written only beside `transferIndirect`.
    std::span<const float>          transferReflected;
    /// THE SAME, AS ZONAL LOBES IN EACH GAUSSIAN'S FRAME: ten floats a record,
    /// two lobes of an axis (the octahedral square's u, v) and three zonal
    /// coefficients. Written in place of `transferDirect` -- what a cloud a
    /// skeleton carries keeps, since the lobes turn with the gaussian.
    std::span<const float>          transferZonal;
    /// WHICH WAYS OUT ARE OPEN: two ints a record, sixty-four bits of an
    /// 8 x 8 octahedral grid over the sphere, set where a traced ray found
    /// nothing. What lets a sun cast a hard shadow on a relit cloud. Over the
    /// world's sphere beside `transferDirect`, over the gaussian's own frame
    /// beside `transferZonal`.
    std::span<const int32_t>        shadowBits;
    /// Ints a record of `shadowBits`: 2, or 8 or 32 for the 256 or 1024 bits
    /// of a 16 x 16 or 32 x 32 grid over the whole sphere (task TX), which a
    /// reader tells apart by the count.
    uint32_t                        shadowWords = 2;
    /// One int a record, 1 where the gaussian came from a thin-walled glass.
    std::span<const int32_t>        thinWalled;
    /// One int a record, 1 where the gaussian's mesh is closed -- every edge
    /// shared by two triangles -- so its back is hidden by its own front
    /// (`primvars:athenea:splat:closed`, what lets a player cull a coarse
    /// node facing away).
    std::span<const int32_t>        closed;
    /// One int a record, 1 where the gaussian's metal is a Schlick (OpenPBR,
    /// glTF) rather than a conductor (`primvars:athenea:splat:schlickMetal`).
    std::span<const int32_t>        schlickMetal;
    /// One float a record, the material's own roughness before the conversion
    /// widened it by the spread of normals (`primvars:athenea:splat:materialRoughness`):
    /// what says whether a frame's reflected field answers the polish.
    std::span<const float>          materialRoughness;
    /// Three floats a record, the surface's shape operator in the gaussian's
    /// own two axes (uu, uv, vv): `primvars:athenea:splat:curvature`, what a
    /// frame turns a TX transfer's reflection across the gaussian by.
    std::span<const float>          curvature;
    /// `primvars:athenea:splat:ior` when above 1: the index the cloud's
    /// transmitting gaussians refract by. Not written at 0.
    float                           ior = 0.0F;
};

/// WHAT A TRANSFER BAKE HANDS THE FILE, as the arrays a stage keeps
/// (`ExportOptions::transferDirect` and the rest).
struct TransferArrays {
    std::vector<float>   direct;
    std::vector<float>   bounced;
    std::vector<float>   reflected;
    std::vector<int32_t> shadowBits;
    /// Zonal lobes in each gaussian's frame, ten floats a gaussian, in place
    /// of the rest (`ExportOptions::transferZonal`).
    std::vector<float>   zonal;
    uint32_t             coefficients = 9;
    uint32_t             shadowWords = 2;
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

/// A MERGE TREE WITH ITS OPACITIES REPLACED (`decimate --tree-coverage`):
/// the root layer of `input` copied to `output` as `writeDecimatedStage`
/// does, and on the field at `prim` each opacity where `opacities` (file
/// order, one a gaussian) holds a number rather than NaN; nothing reordered,
/// since the tree's arrays are file indices. `renameFrom`, where the source
/// has it, is moved to `renameTo`. Each of `added` (int[], one word an
/// element) becomes a primvar of its interpolation.
[[nodiscard]] Result<void> writeTreeOpacities(const std::filesystem::path& input, const std::filesystem::path& output,
                                              const std::string& prim, const std::vector<float>& opacities,
                                              const std::string& renameFrom, const std::string& renameTo,
                                              const std::vector<GaussianArray>& added = {});

/// WHERE A TREE'S NODES CANNOT BE SEEN FROM (occlusion_mask.slang). The
/// tree's own arrays in the file's order: its parents and leaf counts, its
/// TX shadow bits (eight words a gaussian; empty: every mask open) and its
/// transmission (float bits; empty: no glass).
struct OcclusionInput {
    std::vector<uint32_t> parent;
    std::vector<uint32_t> leaves;
    std::vector<uint32_t> bits;
    std::vector<uint32_t> glass;
    /// A node's part (cryptoObject); empty where the field has none.
    std::vector<uint32_t> part;
    /// The stage's up axis: the codebook is fitted on the cells an eye above
    /// the ground looks from (`occlusionUpOf`).
    float                 up[3] = {0.0F, 0.0F, 1.0F};
};
/// The up axis of the stage at `path` (Z unless it says Y).
[[nodiscard]] OcclusionInput occlusionUpOf(const std::filesystem::path& path, OcclusionInput in);
/// Each node's entry in a codebook of masks (eight words an entry, a cell
/// open = 1) that never closes what a node has open; `closed` cells over the
/// nodes and how many of them the codebook `reopened`.
struct OcclusionBook {
    std::vector<uint32_t> index;
    std::vector<uint32_t> book;
    uint32_t              closed = 0;
    uint32_t              reopened = 0;
    uint32_t              levels = 0;
    uint32_t              spared = 0;   ///< nodes worth least, given the all-open entry 0
    uint32_t              seenClosed = 0;   ///< cells closed over the nodes where an eye looks from
};
[[nodiscard]] Result<OcclusionBook> occlusionBook(gpu::ShaderLibrary& library, const OcclusionInput& in,
                                                  uint32_t clusters);
/// A tree's input out of the arrays `readGaussianArrays` read.
[[nodiscard]] Result<OcclusionInput> occlusionInputOf(const std::vector<GaussianArray>& arrays, uint32_t records);
/// `primvars:athenea:splat:occlusionIndex` (vertex) and
/// `occlusionCodebook` (constant), to write beside a tree.
[[nodiscard]] std::vector<GaussianArray> occlusionArrays(const OcclusionBook& book);
/// SEVERAL TREES, ONE CODEBOOK (an asset's pieces: their occluders are the
/// same): each written to its output with its own indices and the codebook.
/// With `byPart` the codebook is shared out a part (cryptoObject) of each
/// tree rather than a tree: a group of many parts handed its entries to
/// whichever brought the most nodes.
[[nodiscard]] Result<void> occlusionMasksShared(gpu::ShaderLibrary& library,
                                                const std::vector<std::filesystem::path>& inputs,
                                                const std::vector<std::filesystem::path>& outputs, uint32_t clusters,
                                                bool byPart = false);

/// What `decimateStage` is asked.
struct DecimateStageOptions {
    std::string           prim;          ///< the ParticleField to read; empty for the first
    uint32_t              degree = 3;    ///< harmonics kept
    bool                  addCamera = true;   ///< for a splat file only: a stage keeps its own
    lod::DecimateSettings settings;
    /// MERGE BY PAIRS (lod::Decimator::decimatePairs) instead of cutting the
    /// octree: `pairSettings.ratio` times fewer, the cheapest merges first.
    bool                  pairs = false;
    lod::PairSettings     pairSettings;
    /// The camera the cloud is made for, a prim of the input stage: a merge
    /// it would see under half a pixel costs nothing of its shape. Empty:
    /// /World/Camera where the stage has one, else none.
    std::string           camera;
    uint32_t              cameraPixels = 1920;   ///< across the camera's image
    /// The stage `camera` is read from (empty: the input's own).
    std::filesystem::path cameraStage;
    /// With `pairs`: the whole merge tree written here as well -- the leaves
    /// in depth-first order, then every merge, each with its parent and cost.
    std::filesystem::path treeOutput;
    /// A CUT THROUGH A MERGE TREE (`input` written by `treeOutput`): for
    /// `camera` at `cameraPixels`, the leaves and merges a renderer would
    /// draw so that at most this many are drawn -- each leaf once, by itself
    /// or by the highest merge over it no wider on screen than a threshold
    /// searched for the budget (lod_tree_cut.slang). 0: no cut.
    uint32_t              cutBudget = 0;
    /// Or the threshold itself, in pixels: one number for every piece cut, so
    /// that pieces cut apart are cut as one tree would be. 0: searched.
    float                 cutPixels = 0.0F;
    /// Or a cut by error (research 100): each node's error in metres, held
    /// monotonic up the tree, projected at the nearest point of its sphere;
    /// a node is in the cut when that is no more than this many pixels and
    /// its parent's is more. 0: by size.
    float                 cutError = 0.0F;
    /// With `cutError`: each node's error also bounds how far its leaves'
    /// colours and normals stray from its own (research 100 §3.2, the weights
    /// of its prototype: colour 10, normals 1, a merge past the guards x4 and
    /// at least twice its size).
    bool                  cutBounds = false;
    /// With `cutBounds`: a node whose whole normal cone faces away from the
    /// camera is left out (for a closed, one-sided part only: glass, sheets and
    /// two-sided parts keep their backs).
    bool                  cutCullBack = false;
    /// With `cutError`: each kept merge's opacity becomes what its leaves,
    /// composited on its plane as the raster does, really cover -- not mass
    /// over its moment ellipse, which counts overlapping tails twice.
    bool                  cutCoverage = false;
    /// The input is a merge tree: every merge's opacity becomes what its leaves
    /// really cover (as `cutCoverage`, for every node, runs up to 4096), and the
    /// whole tree is written -- so a player cutting it reads the coverage.
    bool                  treeCoverage = false;
    /// The input is a merge tree with a TX transfer's shadow bits: each node
    /// gets an occlusion mask (occlusion_mask.slang) -- the cells any leaf
    /// under it is seen from, one cell of margin, all open on glass -- kept as
    /// an index a gaussian into a codebook of this many entries
    /// (`primvars:athenea:splat:occlusionIndex`, `occlusionCodebook`), and the
    /// whole tree is written. 0: none.
    uint32_t              occlusionClusters = 0;
    /// With `cutError`: the tree paged at this many gaussians a page (its
    /// piece's root page every node whose subtree is larger, the rest whole
    /// subtrees packed in leaf order), and how many the cut's pages hold. 0: none.
    uint32_t              cutPages = 0;
    /// With `cutBounds`: a merge past the guards (cost negative: across a
    /// fold, a colour edge, glass with paint) counts this many times its own
    /// error, and at least twice its size.
    float                 cutGuard = 4.0F;
    /// With `cutError`: leave out a node whose occlusion mask is closed toward
    /// the camera (the eye must stand outside the asset's box).
    bool                  cutOcclusion = false;
    /// Pages by error with each band's nodes grouped by the way they face (6
    /// buckets) before where they stand.
    bool                  pagesByFacing = false;
    /// The input is a merge tree: write `primvars:athenea:splat:closed` a node
    /// from this list of closed parts' prim paths (one a line), by its
    /// cryptoObject, and the whole tree.
    std::filesystem::path markClosed;
    /// The input is a merge tree: write `primvars:athenea:splat:materialRoughness`
    /// a node from this list of "prim-path roughness" lines (a part's
    /// material's own roughness), by its cryptoObject, and the whole tree.
    std::filesystem::path markMaterialRoughness;
    /// w_s: a merge's spill (`lodSpill`, written by --tree-coverage: the share
    /// of what it paints that its leaves do not) in the cut's error, beside its
    /// cost, colour spread and normal cone; 0 leaves it out.
    float                 cutSpillWeight = 0.0F;
    /// E0: only the spill past it counts (a Gaussian never fits its leaves'
    /// union: a panel's merges spill ~0.16, a grille's holes ~0.75).
    float                 cutSpillFloor = 0.0F;
    /// The input is a merge tree: write its pages by error (lod_pages.slang)
    /// of this many gaussians -- `primvars:athenea:splat:lodPage` a node, and
    /// a page's band, nodes and first node in page order -- coarsest first --
    /// (`lodPageInfo`, 3 ints)
    /// and the box holding its nodes' spheres with its parents' largest and
    /// smallest error (`lodPageBounds`, 8 floats) -- and the whole tree. 0: none.
    uint32_t              treePages = 0;
    /// Only the gaussians of one group of the asset's parts: those whose
    /// cryptoObject modulo `objectGroups` is `objectGroup` (an asset too large
    /// for one pass, decimated a group at a time). 0 or 1: all.
    /// The input is a merge tree: print its roots a part and stop.
    bool                  treeReport = false;
    uint32_t              objectGroups = 0;
    uint32_t              objectGroup = 0;
    /// THE TRANSFER THROUGH A CODEC AND BACK (transfer_codec.slang): a
    /// clustered PCA basis -- `transferClusters` centroids, `transferComponents`
    /// axes a block, f32 -- and weights of `transferBits` (8 or 4), applied to
    /// the transfer before anything else, so the output shows what a file
    /// holding one cluster id and the weights would draw. Empty: none.
    std::filesystem::path transferBasis;
    uint32_t              transferClusters = 256;
    uint32_t              transferComponents = 8;
    uint32_t              transferBits = 8;
    /// THE SHADOW BITS FROM A CODEBOOK (shadow_vq.slang): `shadowCodebook`
    /// centroids found by Hamming k-means over `shadowRounds` rounds, each
    /// gaussian's bits replaced by its centroid's before anything else. 0: none.
    uint32_t              shadowCodebook = 0;
    uint32_t              shadowRounds = 12;
    /// THE GEOMETRY AT A PHONE CHUNK'S BITS (geometry_quantize.slang): in
    /// chunks of this many gaussians, positions, scales, rotation, opacity,
    /// colour, normal and curvature quantised as research 097's S0 stream
    /// (23 bytes) and decoded again. 0: as they are.
    uint32_t              quantizeChunk = 0;
    /// WHAT NO CAMERA SEES, LEFT OUT FIRST (lod::Decimator::prune): the cloud
    /// drawn from `hiddenViews` directions round its bounds, at three
    /// distances (2.5, 1.6 and 1.15 times the bounding sphere's radius,
    /// 90 degrees across, `hiddenSize` pixels square), and a gaussian that
    /// gave less than `hiddenPixels` of a pixel to all of them together
    /// dropped before anything merges. `hiddenBelow`: the lowest elevation a
    /// view is taken from, in degrees over the stage's horizon (-90: all
    /// round; a car on a road is never seen from under it).
    bool                  dropHidden = false;
    uint32_t              hiddenViews = 256;
    uint32_t              hiddenSize = 1024;
    float                 hiddenPixels = 0.02F;
    float                 hiddenBelow = -90.0F;
    /// FEWER BY THE ERROR THEY LEAVE (after `dropHidden`, before merging):
    /// over `pruneRounds` rounds the store is drawn from `pruneViews`
    /// directions (three distances each, `pruneSize` pixels square), each
    /// gaussian scored by the squared error taking it away would leave, and
    /// the cheapest dropped, until `pruneError` times fewer are left. 0: off.
    /// A cloud with lamp groups' light layers: no merge across a fourfold
    /// change of any group's light (lod::AttributeMerge::LightBand).
    bool                  lightBands = true;
    float                 pruneError = 0.0F;
    uint32_t              pruneRounds = 8;
    uint32_t              pruneViews = 64;
    uint32_t              pruneSize = 512;
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
