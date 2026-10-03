// Copyright (c) 2026 jesus luque.
//
// `athenea mesh2splat`: a USD stage of meshes in, a stage of gaussian splats out,
// converted by Electronic Arts' mesh2splat running as an AOFX effect.
//
// WHO DOES WHAT, AND WHY IT IS SPLIT THIS WAY
//
// The effect knows nothing about USD. It is handed a picture of triangles, up
// to three maps, and a handful of numbers, and it writes gaussian records into
// the picture it is given -- which is all an AOFX effect can be handed, and
// which is exactly why it also runs unchanged in openFXplayer.
//
// So the stage is this side's work: open it, read its meshes
// (`usd::MeshStage`), triangulate them on the device (`geom::MeshBuilder`),
// pack the triangles into a picture (`athenea/usd/mesh_pack`), turn every texture
// a material names into rows of linear float4 (`athenea/usd/texture_rows`, through
// `material::TextureStore`), run the effect once per mesh, and write what comes
// back as a `UsdVolParticleField3DGaussianSplat` (`usd::writeParticleFieldStage`).
//
// The pictures live in the AOFX host's own image storage, which on a device
// with unified memory is the same memory a kernel reads. So the triangles are
// written where the effect will read them, with nothing copied: the host's
// share of the work is opening files and counting, and every number is a
// kernel's.
//
// And the records stay there. Each run's picture is laid out into the
// cloud's own buffers by a kernel (`athenea/usd/mesh2splat_gather`), the bake's
// rays are set up from them on the device (`mesh2splat_span`), the path
// tracer answers into a device buffer that a kernel writes back into the
// records (`mesh2splat_bake`), and the export decodes them where they are
// (`usd::DeviceSplatRecords`). What crosses to the processor is counts, and
// at the very end the values a USD array holds -- which are the processor's
// business by definition.
#include <algorithm>
#include <bit>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Commands.h"
#include "aofx/Effect.h"
#include "athenea/aofx/EffectRegistry.h"
#include "athenea/aofx/EffectRender.h"
#include "athenea/core/Hash.h"
#include "athenea/core/Platform.h"
#include "athenea/geom/Mesh.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/algo/PrefixSum.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu_host/Context.h"
#include "athenea/gpu_host/ImageStorage.h"
#include "athenea/image/Image.h"
#include "athenea/lod/Athc.h"
#include "athenea/technique/PathTracer.h"
#include "athenea/lod/Lod.h"
#include "athenea/material/TextureStore.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/MeshStage.h"
#include "athenea/usd/StageRenderer.h"

namespace athenea::cli {
namespace {

/// A glass that is a sheet: it transmits, and its surface says it is thin.
bool thinGlass(const athenea::usd::StageMaterial& material) {
    return material.thinWalled && material.transmission > 0.0F;
}


/// Rest coefficients a colour, by degree: (degree + 1)^2 - 1.
constexpr uint32_t kRestPerDegree[4] = {0, 3, 8, 15};

/// Entries a row of every picture this command makes. A multiple of four, so
/// that the padding an image adds to its rows is none -- and small enough that
/// a mesh of a few triangles is not a picture one pixel tall and a million
/// wide, which no pool would allocate.
constexpr uint32_t kRowEntries = 4096;

/// A picture tall enough to hold `entries` of them.
[[nodiscard]] image::PixelRect pictureFor(uint64_t entries) {
    const uint64_t rows = (entries + kRowEntries - 1) / kRowEntries;
    return {0, 0, static_cast<int32_t>(kRowEntries), static_cast<int32_t>(std::max<uint64_t>(rows, 1))};
}

/// A slang-rhi view of an image's own pixels: the same memory, no copy. Only
/// where the host's storage put the image on the device, which is every
/// backend gpe adopts.
[[nodiscard]] Result<gpu::Buffer> viewOf(gpu_host::Context& context, const image::ImagePtr& image,
                                         const char* label) {
    gpu_host::ImageStorage* storage = context.sharedStorage();
    if (storage == nullptr) {
        return Error(ErrorCode::DeviceFailure, "the AOFX host has no device image storage");
    }
    const uint64_t buffer = storage->bufferFor(image->address());
    if (buffer == 0) {
        return Error::make(ErrorCode::DeviceFailure,
                           "the {} picture is on the heap, not the device: {} by {}, {:.1f} MB -- the "
                           "pool would not serve it",
                           label, image->bounds().width(), image->bounds().height(),
                           static_cast<double>(image->sizeBytes()) / (1024.0 * 1024.0));
    }
    return context.renderView(buffer, image->sizeBytes(), 16, label);
}

struct Options {
    std::vector<std::string> hidden;
    std::string              stage;
    std::string              output = "splats.usda";
    std::string              prim;
    uint32_t                 resolution = 512;
    uint32_t                 lodLevels = 1;
    /// WHICH BOX THE DENSITY IS MEASURED OVER. "per-model": `resolution`
    /// cells across the whole converted set's longest side, one cell for
    /// everything, so a floor in the stage dilutes the car on it (a 16-unit
    /// floor gave a 1.9-unit car an eighth of the cells it converts alone
    /// with). "per-mesh": each mesh's own box, its cell held between
    /// `cellMin` and `cellMax` in world units (0: derived -- the model's
    /// cell, and an eighth of it).
    std::string              density = "per-model";
    double                   cellMin = 0.0;
    double                   cellMax = 0.0;
    uint64_t                 maxSplats = 2000000;
    /// A CAMERA DECIDES THE CELL: one cell, one pixel of it where each mesh
    /// is nearest (Mesh2GS). Empty: the density's box does.
    std::string              cellFromCamera;
    uint32_t                 cameraPixels = 1920;
    // Their 0.65 is the width they chose for their own renderer; traced here
    // it leaves a converted surface 30% transparent (docs/decisions.md has
    // the table). 1.0 closes it to 91%, 1.2 to 96%.
    double                   sigma = 1.0;
    double                   flatness = 0.1;
    double                   opacity = 1.0;
    double                   minOpacity = 0.6;
    /// A cut-out map reads below this where there is no surface.
    double                   opacityCut = 0.5;
    uint32_t                 maxCells = 1u << 18;
    /// A MAP NO BIGGER THAN THIS, AND A CEILING BY DEFAULT.
    ///
    /// A map travels to the effect as a float4 picture, sixteen bytes a texel
    /// where the file holds one: a 4k map is 268 MB on the device and a car
    /// with fifteen of them does not fit the pool at all. The conversion
    /// samples a map once a cell, and at resolution 512 the whole model is
    /// 512 cells across, so most of a 4k map is thrown away before it is
    /// looked at. 0 reads them at their own size.
    uint32_t                 textureSize = 1024;
    bool                     noTextures = false;
    bool                     normalMapTurns = false;
    /// Leave the material's displacement out: every gaussian on the flat mesh.
    bool                     noDisplacement = false;
    /// The most gaussians the relief may split one cell into, along each of
    /// its two axes.
    uint32_t                 displaceRefine = 8;
    /// FEWER GAUSSIANS WHERE THE SURFACE IS THE SAME: a block of cells whose
    /// maps move by no more than this is one gaussian of its size (0: a
    /// gaussian a cell), up to 2^simplifyLevels cells a side.
    double                   simplify = 0.0;
    uint32_t                 simplifyLevels = 3;
    bool                     addCamera = true;
    /// Bake the path tracer's answer into the gaussians (the default), or
    /// carry the material and be relit.
    bool                     bake = true;
    /// BAKE THE TRANSFER RATHER THAN THE LIGHT: how much of an environment
    /// reaches each gaussian, so the cloud can be lit by any sky afterwards.
    /// Excludes the radiance bake, which keeps one sky's light instead.
    bool                     transfer = false;
    bool                     indirect = true;
    /// THE TRANSFER'S OPEN DIRECTIONS, cells a side of an octahedral grid
    /// over the whole sphere: 16 (256 bits a gaussian) or 32 (1024), which is
    /// what a reflection's occlusion and a glass's view through are read from
    /// (task TX); 0, the first transfer's 64 over the half a gaussian faces.
    uint32_t                 transferCells = 16;
    /// PATHS A GAUSSIAN, 256 on average since every gaussian is blended in
    /// linear light: these everywhere, then `bakeExtra` more shared out
    /// where the noise is (docs/decisions.md, "The bake's grain"). A transfer
    /// takes the two together.
    uint32_t                 bakeSamples = 128;
    uint32_t                 bakeBounces = 3;
    /// How much of the direction the light leaves in the cloud carries: 0 is
    /// a colour, 1 to 3 are harmonics. Two is where a highlight starts to
    /// look like one.
    uint32_t                 bakeDegree = 2;
    /// ADAPTIVE: paths a gaussian on average added after the first pass,
    /// where the relative variance per cost says they are worth most.
    uint32_t                 bakeExtra = 128;
    uint32_t                 bakePassSamples = 64;
    /// A-TROUS PASSES of the splat bake filter (0: none), over the whole
    /// light unless `bakeFilterIndirectOnly`.
    uint32_t                 bakeFilter = 3;
    double                   bakeFilterLuminance = 4.0;
    /// The indirect half alone, as proposal 012 had it, rather than the
    /// whole light (which is where the grain was, measured on the pawn).
    bool                     bakeFilterIndirectOnly = false;
    bool                     defaultLights = false;
    /// Carry the skeleton: the gaussians are built in the bind pose and each
    /// keeps the joints that move it.
    bool                     skinned = false;
    /// START:END[:STEP] in time codes; empty is the stage's own range.
    std::string              range;
    double                   time = 0.0;
    std::vector<std::string> paths;
};

/// One map the conversion reads, as a picture. Several materials name the same
/// file, and a metallic and a roughness file become one picture, so these are
/// kept by what went into them.
struct MapKey {
    std::string metallic;   ///< also the only file, for a one-file map
    std::string roughness;
    bool        packed = false;

    friend bool operator<(const MapKey& a, const MapKey& b) {
        if (a.metallic != b.metallic) return a.metallic < b.metallic;
        if (a.roughness != b.roughness) return a.roughness < b.roughness;
        return static_cast<int>(a.packed) < static_cast<int>(b.packed);
    }
};

class Converter {
public:
    Converter(gpu_host::Context& context, gpu::ShaderLibrary& library, const Options& options)
        : context_(&context), library_(&library), options_(&options) {}

    [[nodiscard]] Result<void> prepare() {
        auto pack = gpu::ComputeKernel::create(*library_, "athenea/usd/mesh_pack", "meshPack");
        if (!pack) return std::move(pack).error();
        pack_ = std::move(*pack);
        auto chunks = gpu::ComputeKernel::create(*library_, "athenea/usd/mesh_pack", "streamBoundsChunks");
        if (!chunks) return std::move(chunks).error();
        chunks_ = std::move(*chunks);
        auto reduce = gpu::ComputeKernel::create(*library_, "athenea/scene/bounds_reduce", "boundsReduce");
        if (!reduce) return std::move(reduce).error();
        reduce_ = std::move(*reduce);
        auto slices = gpu::ComputeKernel::create(*library_, "athenea/scene/bounds_reduce_slices", "boundsReduceSlices");
        if (!slices) return std::move(slices).error();
        reduceSlices_ = std::move(*slices);
        auto rows = gpu::ComputeKernel::create(*library_, "athenea/usd/texture_rows", "textureRows");
        if (!rows) return std::move(rows).error();
        rows_ = std::move(*rows);
        const auto make = [&](const char* module, const char* entry, gpu::ComputeKernel& into) -> Result<void> {
            auto made = gpu::ComputeKernel::create(*library_, module, entry);
            if (!made) return std::move(made).error();
            into = std::move(*made);
            return ok();
        };
        ATHENEA_TRY(make("athenea/usd/mesh2splat_gather", "m2sGather", gather_));
        ATHENEA_TRY(make("athenea/usd/mesh2splat_gather", "m2sNoInfluence", noInfluence_));
        ATHENEA_TRY(make("athenea/usd/mesh2splat_span", "m2sRecordChunks", recordChunks_));
        ATHENEA_TRY(make("athenea/usd/mesh2splat_span", "m2sRaySpan", raySpan_));
        ATHENEA_TRY(make("athenea/usd/mesh2splat_bake", "m2sBakeInto", bakeInto_));
        ATHENEA_TRY(make("athenea/usd/mesh2splat_bake", "m2sTransferInto", transferInto_));
        ATHENEA_TRY(make("athenea/usd/mesh2splat_subset", "m2sSubsetFlags", subsetFlags_));
        ATHENEA_TRY(make("athenea/usd/mesh2splat_subset", "m2sSubsetScatter", subsetScatter_));
        ATHENEA_TRY(make("athenea/usd/mesh2splat_cells", "m2sCells", cells_));
        auto prefix = gpu::PrefixSum::create(*library_);
        if (!prefix) return std::move(prefix).error();
        prefix_ = std::move(*prefix);
        auto textures = material::TextureStore::create(*library_);
        if (!textures) return std::move(textures).error();
        textures_ = std::move(*textures);
        return ok();
    }

    /// A piece that is a whole mesh, not one of its subsets.
    static constexpr uint32_t kWhole = 0xFFFFFFFF;

    /// WHAT ONE RUN OF THE EFFECT CONVERTS: a mesh; or, where its GeomSubsets
    /// bind materials of their own, the triangles of one subset, or those no
    /// subset claims -- each with the material it is bound to. A mesh of
    /// three materials used to be converted with the mesh's alone.
    struct Piece {
        size_t                    mesh = 0;          ///< which of the stage's meshes
        uint32_t                  subset = kWhole;   ///< kWhole, 0 (no subset's), or the subset's index + 1
        std::string               path;              ///< for the log: the mesh, or the GeomSubset
        const usd::StageMaterial* material = nullptr;
        uint32_t                  triangles = 0;
        gpu::Buffer               list;              ///< the subset's triangles, in the mesh's order
    };
    /// The pieces of every mesh, and for each subset the list of its
    /// triangles, made on the device (`athenea/usd/mesh2splat_subset`); what
    /// crosses back is how many each holds.
    [[nodiscard]] Result<void> makePieces(const std::vector<usd::StageMesh>& meshes) {
        gpu::Device& device = library_->device();
        pieces_.clear();
        std::vector<gpu::Buffer> totals;
        for (size_t m = 0; m < meshes.size(); ++m) {
            const usd::StageMesh& one = meshes[m];
            const uint32_t triangles = one.mesh.triangles;
            if (one.subsets.empty() || triangles == 0 || !one.mesh.triangleSubsets.valid()) {
                pieces_.push_back({m, kWhole, one.path, &one.material, triangles, {}});
                totals.emplace_back();
                continue;
            }
            // Every subset, and then what none of them claims.
            for (uint32_t k = 0; k <= one.subsets.size(); ++k) {
                const bool rest = k == one.subsets.size();
                const uint32_t wanted = rest ? 0u : k + 1;
                const auto words = [&](uint64_t n, const char* label) {
                    gpu::BufferDesc desc;
                    desc.bytes = std::max<uint64_t>(n, 1) * 4;
                    desc.elementBytes = 4;
                    desc.label = label;
                    return gpu::Buffer::create(device, desc);
                };
                auto flags = words(triangles, "mesh2splat.subsetFlags");
                auto offsets = words(triangles, "mesh2splat.subsetOffsets");
                auto list = words(triangles, "mesh2splat.subsetList");
                auto total = words(1, "mesh2splat.subsetTotal");
                if (!flags || !offsets || !list || !total) {
                    return Error(ErrorCode::OutOfMemory, "mesh2splat: cannot list a subset's triangles");
                }
                const auto bind = [&](rhi::ShaderCursor cursor) {
                    cursor["triangleSubsets"].setBinding(one.mesh.triangleSubsets.rhi());
                    cursor["flags"].setBinding(flags->rhi());
                    cursor["offsets"].setBinding(offsets->rhi());
                    cursor["list"].setBinding(list->rhi());
                    cursor["subset"]["triangles"].setData(triangles);
                    cursor["subset"]["wanted"].setData(wanted);
                };
                // A batch a list: the prefix sum's scratch is its own, and
                // is not to be shared between lists in flight.
                gpu::CommandBatch batch(device);
                subsetFlags_.dispatch(batch, {triangles, 1, 1}, bind);
                ATHENEA_TRY(prefix_.apply(batch, *flags, *offsets, *total, triangles));
                subsetScatter_.dispatch(batch, {triangles, 1, 1}, bind);
                ATHENEA_TRY(batch.submit(true));
                pieces_.push_back({m, wanted, rest ? one.path : one.subsets[k].path,
                                   rest ? &one.material : &one.subsets[k].material, 0, std::move(*list)});
                totals.push_back(std::move(*total));
            }
        }
        for (size_t p = 0; p < pieces_.size(); ++p) {
            if (totals[p].valid()) {
                ATHENEA_TRY(totals[p].read(device, 0, sizeof(uint32_t), &pieces_[p].triangles));
            }
        }
        return ok();
    }

    /// Every piece's triangles into a picture of its own, and the model's box
    /// -- which is what the projection grid is measured against -- folded from
    /// those pictures on the device, with each mesh's own beside it.
    [[nodiscard]] Result<void> packMeshes(std::vector<usd::StageMesh>& meshes) {
        gpu::Device& device = library_->device();
        ATHENEA_TRY(makePieces(meshes));
        const size_t pieces = pieces_.size();
        streams_.resize(pieces);
        skins_.resize(pieces);
        uv2s_.resize(pieces);
        triangles_.resize(pieces);
        uint64_t chunkTotal = 0;
        std::vector<uint32_t> chunkFirst(pieces, 0);
        std::vector<uint32_t> chunkCount(pieces, 0);
        // A mesh's pieces are consecutive, so its chunks are one run: the
        // box it is measured over is folded from that run.
        std::vector<uint32_t> meshChunkFirst(meshes.size(), 0);
        std::vector<uint32_t> meshChunkCount(meshes.size(), 0);
        for (size_t k = 0; k < pieces; ++k) {
            const uint64_t entries = uint64_t{pieces_[k].triangles} * 6;
            chunkFirst[k] = static_cast<uint32_t>(chunkTotal);
            chunkCount[k] = static_cast<uint32_t>((entries + kChunkEntries - 1) / kChunkEntries);
            const size_t m = pieces_[k].mesh;
            if (meshChunkCount[m] == 0) {
                meshChunkFirst[m] = chunkFirst[k];
            }
            meshChunkCount[m] += chunkCount[k];
            chunkTotal += chunkCount[k];
        }
        if (chunkTotal == 0) {
            return Error(ErrorCode::InvalidArgument, "no triangles to convert");
        }
        gpu::BufferDesc desc;
        desc.bytes = chunkTotal * 2 * 16;
        desc.elementBytes = 16;
        desc.label = "mesh2splat.extents";
        auto extents = gpu::Buffer::create(device, desc);
        if (!extents) return std::move(extents).error();
        // The model's box in the first two entries, then each mesh's own:
        // the per-mesh density is measured over those.
        desc.bytes = (1 + meshes.size()) * 2 * 16;
        desc.label = "mesh2splat.bounds";
        auto bounds = gpu::Buffer::create(device, desc);
        if (!bounds) return std::move(bounds).error();
        std::vector<uint32_t> sliceRuns;
        sliceRuns.reserve(meshes.size() * 2);
        for (size_t m = 0; m < meshes.size(); ++m) {
            sliceRuns.push_back(meshChunkFirst[m]);
            sliceRuns.push_back(meshChunkCount[m]);
        }
        auto sliceBuffer = gpu::Buffer::fromSpan<uint32_t>(device, sliceRuns, "mesh2splat.slices");
        if (!sliceBuffer) return std::move(sliceBuffer).error();

        gpu::CommandBatch batch(device);
        for (size_t k = 0; k < pieces; ++k) {
            const Piece& piece = pieces_[k];
            const usd::StageMesh& owner = meshes[piece.mesh];
            const geom::GpuMesh& mesh = owner.mesh;
            triangles_[k] = piece.triangles;
            if (piece.triangles == 0) {
                continue;
            }
            const uint64_t entries = uint64_t{piece.triangles} * 6;
            auto picture = image::Image::create(pictureFor(entries));
            if (!picture) return std::move(picture).error();
            streams_[k] = *picture;
            auto view = viewOf(*context_, streams_[k], "mesh2splat.stream");
            if (!view) return std::move(view).error();

            // The joints each corner is carried by, in a second picture of
            // exactly the same shape, so the two are addressed alike and the
            // effect needs no second set of dimensions.
            const usd::StageSkinning& skin = owner.skinning;
            gpu::Buffer influences;
            std::optional<gpu::Buffer> skinView;
            if (skin.bound && !skin.influences.empty()) {
                gpu::BufferDesc held;
                held.bytes = skin.influences.size() * 4;
                held.elementBytes = 8;
                held.label = "mesh2splat.influences";
                auto made = gpu::Buffer::create(device, held, skin.influences.data());
                if (!made) return std::move(made).error();
                influences = std::move(*made);
                auto second = image::Image::create(pictureFor(entries));
                if (!second) return std::move(second).error();
                skins_[k] = *second;
                auto held2 = viewOf(*context_, skins_[k], "mesh2splat.skin");
                if (!held2) return std::move(held2).error();
                skinView = std::move(*held2);
            }

            const geom::GpuPrimvar* normals = mesh.primvar("normals");
            const geom::GpuPrimvar* uvs = mesh.primvar("st");
            // The second set of texture coordinates, where a material reads
            // some of its maps by one: a third picture of the same shape.
            const geom::GpuPrimvar* uvs2 = mesh.primvar("st2");
            std::optional<gpu::Buffer> uv2View;
            if (uvs2 != nullptr) {
                auto third = image::Image::create(pictureFor(entries));
                if (!third) return std::move(third).error();
                uv2s_[k] = *third;
                auto held3 = viewOf(*context_, uv2s_[k], "mesh2splat.uv2");
                if (!held3) return std::move(held3).error();
                uv2View = std::move(*held3);
            }
            const uint32_t stride = static_cast<uint32_t>(streams_[k]->stride());
            const uint32_t chunkThreads = chunkCount[k];
            const auto bind = [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(mesh.positions.rhi());
                cursor["indices"].setBinding(mesh.indices.rhi());
                cursor["triangleCorners"].setBinding(mesh.triangleCorners.rhi());
                cursor["triangleFaces"].setBinding(mesh.triangleFaces.rhi());
                // Every buffer the kernel declares must be bound, there or
                // not: a missing primvar is said with its count, not by
                // leaving a binding empty.
                cursor["normals"].setBinding(normals != nullptr ? normals->values.rhi()
                                                                : mesh.positions.rhi());
                cursor["uvs"].setBinding(uvs != nullptr ? uvs->values.rhi() : mesh.positions.rhi());
                cursor["uvs2"].setBinding(uvs2 != nullptr ? uvs2->values.rhi() : mesh.positions.rhi());
                cursor["influences"].setBinding(influences.valid() ? influences.rhi()
                                                                   : mesh.positions.rhi());
                cursor["skinStream"].setBinding(skinView ? skinView->rhi() : view->rhi());
                cursor["uv2Stream"].setBinding(uv2View ? uv2View->rhi() : view->rhi());
                cursor["stream"].setBinding(view->rhi());
                cursor["extents"].setBinding(extents->rhi());
                cursor["pack"]["triangles"].setData(piece.triangles);
                cursor["pack"]["listed"].setData(piece.list.valid() ? 1u : 0u);
                cursor["triangleList"].setBinding(piece.list.valid() ? piece.list.rhi() : mesh.indices.rhi());
                cursor["pack"]["destFirst"].setData(uint32_t{0});
                cursor["pack"]["normalMode"].setData(normals != nullptr
                                                         ? static_cast<uint32_t>(normals->interpolation)
                                                         : kNoPrimvar);
                cursor["pack"]["normalCount"].setData(normals != nullptr ? normals->count : 0U);
                cursor["pack"]["uvMode"].setData(uvs != nullptr ? static_cast<uint32_t>(uvs->interpolation)
                                                                : kNoPrimvar);
                cursor["pack"]["uvCount"].setData(uvs != nullptr ? uvs->count : 0U);
                cursor["pack"]["uv2Mode"].setData(uvs2 != nullptr ? static_cast<uint32_t>(uvs2->interpolation)
                                                                  : kNoPrimvar);
                cursor["pack"]["uv2Count"].setData(uvs2 != nullptr ? uvs2->count : 0U);
                cursor["pack"]["points"].setData(mesh.points);
                cursor["pack"]["corners"].setData(mesh.corners);
                cursor["pack"]["destWidth"].setData(kRowEntries);
                cursor["pack"]["destStride"].setData(stride);
                cursor["pack"]["chunkFirst"].setData(uint32_t{0});
                cursor["pack"]["perPoint"].setData(skinView ? skin.perPoint : 0U);
                for (uint32_t r = 0; r < 3; ++r) {
                    const std::string world = "toWorld" + std::to_string(r);
                    const std::string normal = "normalTo" + std::to_string(r);
                    cursor["pack"][world.c_str()].setData(owner.toWorld.data() + r * 4, 16);
                    cursor["pack"][normal.c_str()].setData(owner.normalToWorld.data() + r * 4, 16);
                }
                cursor["pack"]["entries"].setData(static_cast<uint32_t>(entries));
                cursor["pack"]["chunkSize"].setData(kChunkEntries);
                cursor["pack"]["chunkCount"].setData(chunkThreads);
            };
            pack_.dispatch(batch, {piece.triangles, 1, 1}, bind);
            // Every mesh's chunks go into one buffer, each at its own offset,
            // so that one reduce at the end gives the model's box.
            const uint32_t first = chunkFirst[k];
            chunks_.dispatch(batch, {chunkThreads, 1, 1}, [&](rhi::ShaderCursor cursor) {
                bind(cursor);
                cursor["pack"]["chunkFirst"].setData(first);
            });
        }
        const uint32_t total = static_cast<uint32_t>(chunkTotal);
        reduce_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["extents"].setBinding(extents->rhi());
            cursor["result"].setBinding(bounds->rhi());
            cursor["params"]["count"].setData(total * 2);
            cursor["params"]["chunkSize"].setData(kChunkEntries);
            cursor["params"]["chunkCount"].setData(total);
        });
        reduceSlices_.dispatch(batch, {static_cast<uint32_t>(meshes.size()), 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["extents"].setBinding(extents->rhi());
            cursor["slices"].setBinding(sliceBuffer->rhi());
            cursor["result"].setBinding(bounds->rhi());
            cursor["params"]["slices"].setData(static_cast<uint32_t>(meshes.size()));
            cursor["params"]["resultFirst"].setData(2u);
        });
        ATHENEA_TRY(batch.submit(true));
        for (const image::ImagePtr& stream : streams_) {
            if (stream) {
                stream->deviceWrote();
            }
        }
        for (const image::ImagePtr& skin : skins_) {
            if (skin) {
                skin->deviceWrote();
            }
        }
        for (const image::ImagePtr& uv2 : uv2s_) {
            if (uv2) {
                uv2->deviceWrote();
            }
        }
        auto box = bounds->readAll<float>(device);
        if (!box) return std::move(box).error();
        for (int axis = 0; axis < 3; ++axis) {
            boundsMin_[axis] = (*box)[static_cast<size_t>(axis)];
            boundsMax_[axis] = (*box)[4 + static_cast<size_t>(axis)];
        }
        meshBounds_.resize(meshes.size());
        for (size_t k = 0; k < meshes.size(); ++k) {
            const size_t at = (1 + k) * 8;
            for (size_t axis = 0; axis < 3; ++axis) {
                meshBounds_[k][axis] = (*box)[at + axis];
                meshBounds_[k][3 + axis] = (*box)[at + 4 + axis];
            }
            // A mesh with no triangles folded nothing: give it the model's box.
            if (meshChunkCount[k] == 0) {
                for (size_t axis = 0; axis < 3; ++axis) {
                    meshBounds_[k][axis] = boundsMin_[axis];
                    meshBounds_[k][3 + axis] = boundsMax_[axis];
                }
            }
        }
        boxes_ = std::move(*bounds);
        return ok();
    }

    /// The camera a conversion sizes its cells by (`--cell-from-camera`),
    /// and how many pixels across its image is.
    void setCamera(const usd::StageCamera& camera, uint32_t pixels) {
        camera_ = camera;
        cameraPixels_ = pixels;
    }

    /// Every piece's cell, its factor and what the effect is given for it,
    /// worked out on the device (`athenea/usd/mesh2splat_cells` says what
    /// each value is); read back as the answer.
    [[nodiscard]] Result<void> deriveCells() {
        gpu::Device& device = library_->device();
        const uint32_t pieces = static_cast<uint32_t>(pieces_.size());
        std::vector<uint32_t> meshOf(pieces, 0);
        for (uint32_t p = 0; p < pieces; ++p) {
            meshOf[p] = static_cast<uint32_t>(pieces_[p].mesh);
        }
        auto meshBuffer = gpu::Buffer::fromSpan<uint32_t>(device, meshOf, "mesh2splat.pieceMesh");
        auto wantBuffer = gpu::Buffer::fromSpan<uint32_t>(device, wantedBy_, "mesh2splat.wanted");
        auto shareBuffer = gpu::Buffer::fromSpan<uint32_t>(device, shareOf_, "mesh2splat.share");
        gpu::BufferDesc desc;
        desc.bytes = (3 + uint64_t{pieces} * 5) * 4;
        desc.elementBytes = 4;
        desc.label = "mesh2splat.cells";
        auto cells = gpu::Buffer::create(device, desc);
        if (!meshBuffer || !wantBuffer || !shareBuffer || !cells) {
            return Error(ErrorCode::OutOfMemory, "mesh2splat: cannot work out the cells");
        }
        gpu::CommandBatch batch(device);
        cells_.dispatch(batch, {std::max(pieces, 1u), 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["bounds"].setBinding(boxes_.rhi());
            cursor["pieceMesh"].setBinding(meshBuffer->rhi());
            cursor["wanted"].setBinding(wantBuffer->rhi());
            cursor["share"].setBinding(shareBuffer->rhi());
            cursor["cells"].setBinding(cells->rhi());
            rhi::ShaderCursor c = cursor["params"];
            c["pieces"].setData(pieces);
            c["resolution"].setData(options_->resolution);
            c["cellMin"].setData(static_cast<float>(options_->cellMin));
            c["cellMax"].setData(static_cast<float>(options_->cellMax));
            c["perMesh"].setData(perMesh_ ? 1u : 0u);
            c["camera"].setData(camera_ ? 1u : 0u);
            if (camera_) {
                c["aperture"].setData(camera_->horizontalAperture);
                c["focal"].setData(camera_->focalLength);
                c["pixels"].setData(static_cast<float>(cameraPixels_));
                c["near"].setData(camera_->nearClip);
                c["toWorld0"].setData(camera_->toWorld.data(), 16);
                c["toWorld1"].setData(camera_->toWorld.data() + 4, 16);
                c["toWorld2"].setData(camera_->toWorld.data() + 8, 16);
            }
        });
        ATHENEA_TRY(batch.submit(true));
        auto read = cells->readAll<float>(device);
        if (!read) return std::move(read).error();
        cellValues_ = std::move(*read);
        modelCell_ = static_cast<double>(cellValues_[0]);
        cellMin_ = static_cast<double>(cellValues_[1]);
        cellMax_ = static_cast<double>(cellValues_[2]);
        return ok();
    }

    /// The maps every material names -- a mesh's and its subsets' -- decoded
    /// once and handed over as pictures.
    [[nodiscard]] Result<void> loadTextures() {
        if (options_->noTextures) {
            return ok();
        }
        const auto ask = [&](const usd::StageTexture& texture) {
            if (!texture.empty() && !ids_.contains(texture.file)) {
                ids_[texture.file] = textures_->request(
                    texture.file, texture.srgb ? "srgb_texture" : "raw");
            }
        };
        for (const Piece& piece : pieces_) {
            const usd::StageMaterial& material = *piece.material;
            ask(material.albedo);
            ask(material.normal);
            ask(material.metallicMap);
            ask(material.roughnessMap);
            // The cut-out too: it was never asked for, and passed only while
            // it happened to be the normal map's file (the sparrow's feathers
            // read their alpha off it). With the normal map repaired into a
            // file of its own, the cut silently went.
            ask(material.opacityMap);
            // What it gives off, where that is a map. sRGB where the file
            // says so: a map of light is a colour like any other.
            ask(material.emissionMap);
            if (!options_->noDisplacement) {
                ask(material.displacementMap);
            }
        }
        if (ids_.empty()) {
            return ok();
        }
        auto loaded = textures_->commit();
        if (!loaded) return std::move(loaded).error();
        sampler_ = textures_->sampler(material::Wrap::Repeat, material::Wrap::Repeat);
        std::printf("mesh2splat: %zu of %zu textures decoded\n", *loaded, ids_.size());
        for (const auto& [file, id] : ids_) {
            const material::TextureInfo& info = textures_->info(id);
            if (!info.loaded) {
                std::fprintf(stderr, "mesh2splat: '%s' did not decode: %s\n", file.c_str(),
                             info.error.c_str());
            }
        }
        return ok();
    }

    /// One map as a picture, cached. Without `pack` it is one file in all four
    /// channels -- an albedo, a normal map. With it, the two files are packed
    /// as glTF packs them, metallic in blue and roughness in green, which is
    /// what the conversion reads, and either of them may be absent.
    [[nodiscard]] Result<image::ImagePtr> mapPicture(const std::string& metallic,
                                                     const std::string& roughness, bool pack = false) {
        const MapKey key{metallic, roughness, pack};
        if (const auto held = maps_.find(key); held != maps_.end()) {
            return held->second;
        }
        const auto sizeOf = [&](const std::string& file, uint32_t& width, uint32_t& height) {
            const auto id = ids_.find(file);
            if (id == ids_.end()) {
                return;
            }
            const material::TextureInfo& info = textures_->info(id->second);
            if (info.loaded) {
                width = std::max(width, info.width);
                height = std::max(height, info.height);
            }
        };
        uint32_t width = 0;
        uint32_t height = 0;
        sizeOf(metallic, width, height);
        sizeOf(roughness, width, height);
        if (width == 0 || height == 0) {
            return image::ImagePtr{};
        }
        if (options_->textureSize > 0) {
            width = std::min(width, options_->textureSize);
            height = std::min(height, options_->textureSize);
        }
        auto picture = image::Image::create({0, 0, static_cast<int32_t>(width), static_cast<int32_t>(height)});
        if (!picture) return std::move(picture).error();
        auto view = viewOf(*context_, *picture, "mesh2splat.map");
        if (!view) return std::move(view).error();

        gpu::CommandBatch batch(library_->device());
        const uint32_t stride = static_cast<uint32_t>((*picture)->stride());
        // `fallback` is what the picture holds where no texture wrote, and
        // the kernel multiplies it into the material's own value: a channel
        // with no map leaves that value as it is, so it is one. It was
        // (roughness 0.5, metallic 0), which made a metal with only a
        // roughness map a dielectric -- 1 x 0 -- and halved a roughness that
        // came with only a metallic map.
        const std::array<float, 4> defaults{1.0F, 1.0F, 1.0F, 1.0F};
        const auto write = [&](const std::string& file, uint32_t channels, bool clear) {
            const auto id = ids_.find(file);
            const uint32_t which = id != ids_.end() ? id->second : 0;
            rows_.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["rows"].setBinding(view->rhi());
                textures_->bind(cursor["table"]);
                cursor["params"]["id"].setData(which);
                cursor["params"]["sampler"].setData(sampler_);
                cursor["params"]["width"].setData(width);
                cursor["params"]["height"].setData(height);
                cursor["params"]["stride"].setData(stride);
                cursor["params"]["channels"].setData(channels);
                cursor["params"]["fromChannel"].setData(uint32_t{0});
                cursor["params"]["clear"].setData(clear ? 1U : 0U);
                cursor["params"]["fallback"].setData(defaults.data(), 16);
            });
        };
        if (!pack) {
            write(metallic, 15, true);   // a picture of its own: albedo, a normal map
        } else {
            // Blue is metallic, green is roughness: glTF's packing, and what
            // their fragment shader reads. The defaults go down first, so a
            // material that names one map and not the other keeps the
            // default for the one it does not name -- which is what it meant.
            write({}, 0, true);
            if (!metallic.empty()) {
                write(metallic, 4, false);
            }
            if (!roughness.empty()) {
                write(roughness, 2, false);
            }
        }
        ATHENEA_TRY(batch.submit(true));
        (*picture)->deviceWrote();
        maps_[key] = *picture;
        return *picture;
    }

    [[nodiscard]] Result<usd::DeviceSplatRecords> convert(aofx::Effect& effect,
                                                          std::vector<usd::StageMesh>& meshes) {
        usd::DeviceSplatRecords raw;
        raw.source = options_->stage;
        raw.encoding.x = 0; raw.encoding.y = 1; raw.encoding.z = 2; raw.encoding.opacity = 3;
        raw.encoding.scale0 = 4; raw.encoding.scale1 = 5; raw.encoding.scale2 = 6;
        raw.encoding.rotW = 7; raw.encoding.rotX = 8; raw.encoding.rotY = 9; raw.encoding.rotZ = 10;
        raw.encoding.dc0 = 11; raw.encoding.dc1 = 12; raw.encoding.dc2 = 13;
        raw.encoding.restBase = 23;
        raw.encoding.restPerColour = options_->bake ? kRestPerDegree[std::min(options_->bakeDegree, 3u)] : 0;
        raw.encoding.restColourOuter = 0;   // rgb per basis, which is how the bake writes them
        // What the gaussian reflects with, which is what lets a relit cloud
        // show the sheen its mesh had: two more floats a record.
        raw.encoding.metallic = 14;
        raw.encoding.roughness = 15;
        raw.encoding.transmission = 16;
        // AND WHICH WAY THE SURFACE FACED where the normal map turned it: the
        // shading normal, apart from the frame (whose short axis is the face's
        // own normal unless --normal-map-turns). Free -- the kernel worked it
        // out for every gaussian -- and what lets a relit cloud keep the
        // relief its mesh had.
        raw.encoding.normal = 17;
        // AND THE LIGHT IT GIVES OFF, where any material of the stage gives
        // off any: linear radiance, which a relit or transferred cloud adds
        // and a baked one already holds. The three floats are there either
        // way and say nothing when no material emits, so no file carries an
        // emission of zeros.
        // Every piece's material: a subset may be the lamp of its mesh.
        emits_ = std::any_of(pieces_.begin(), pieces_.end(),
                             [](const Piece& piece) { return piece.material->emits(); });
        raw.encoding.emission = emits_ ? 20u : io::SplatEncoding::kNoField;
        // AND WHAT THE MATERIALS LAYER OVER THEIR BASE -- a car's lacquer, a
        // specular's tint, a sheen -- where any of the stage's does: thirteen
        // floats more, last, after the harmonics the bake writes.
        layered_ = std::any_of(pieces_.begin(), pieces_.end(),
                               [](const Piece& piece) { return piece.material->layered(); });
        raw.encoding.floatsPerRecord = recordFloats();
        raw.encoding.lobes = layered_ ? raw.encoding.floatsPerRecord - 13 : io::SplatEncoding::kNoField;
        raw.encoding.opacity_ = io::SplatEncoding::Opacity::Linear;
        raw.encoding.scale_ = io::SplatEncoding::Scale::Linear;
        // LINEAR LIGHT, BAKED OR NOT. Not baked, the colours are a
        // material's albedo; baked, the harmonics are fitted to the light the
        // paths returned, already in the form a cloud keeps them (the kernel
        // shifted the constant term). Neither is a capture's sRGB, and the
        // file says so (`primvars:athenea:splat:linear`): every cloud is
        // blended in linear light, and this one's colours are drawn as they
        // are.
        raw.encoding.colour = options_->bake ? io::SplatEncoding::Colour::ShDc
                                             : io::SplatEncoding::Colour::LinearLight;
        raw.linear = true;
        raw.encoding.rest = io::SplatEncoding::Rest::Float;
        raw.encoding.rotation = io::SplatEncoding::Rotation::Float;

        uint64_t written = 0;
        uint64_t wanted = 0;
        uint64_t degenerate = 0;
        uint64_t capped = 0;   ///< cells the relief wanted split finer than --displace-refine
        uint64_t beyond = 0;   ///< cells a triangle had past --max-cells, left unwalked
        size_t   unconverted = 0;   ///< meshes the budget ran out before
        uint32_t reruns = 0;
        // THE CELL'S BOUNDS, DERIVED ONCE. Per mesh, no mesh is coarser than
        // the model's cell (nothing loses density against per-model) and none
        // finer than an eighth of it: a density ratio of 64 between the
        // finest part and the coarsest, which is about what a floor and the
        // car on it are apart, and which keeps a bolt from walking a grid
        // sixteen hundred times finer than the body's. Worked out on the
        // device over the boxes it folded (`mesh2splat_cells`), with the
        // cell each piece walks -- from a camera, where one is given.
        perMesh_ = options_->density == "per-mesh";
        const size_t pieces = pieces_.size();
        wantedBy_.assign(pieces, 0);
        shareOf_.assign(pieces, 0);
        ATHENEA_TRY(deriveCells());
        if (perMesh_ && !camera_) {
            std::printf("mesh2splat: density per mesh: %u cells across each mesh's longest side, the cell "
                        "held between %.4g and %.4g (the model's is %.4g)\n",
                        options_->resolution, cellMin_, cellMax_, modelCell_);
        }

        // WHAT EVERY PIECE WANTS, BEFORE ANY IS CONVERTED.
        //
        // The budget used to be spent in mesh order: a budget too small kept
        // the first meshes whole and the last ones not at all -- a car whose
        // wheels came after its body had none. So every piece is counted
        // first: the effect counts everything a run would write whatever its
        // budget, so a run with room for one gaussian is the count, and
        // costs a count and a scan. Then the budget is shared in proportion
        // to what each wants -- which is the same as one density floor for
        // all of them: every piece walks a cell sqrt(wanted / budget) times
        // coarser, so every mesh loses density alike and none is dropped.
        uint64_t total = 0;
        for (size_t k = 0; k < pieces; ++k) {
            if (triangles_[k] == 0 || coversNothing(*pieces_[k].material)) {
                continue;
            }
            auto counted = runOne(effect, meshes[pieces_[k].mesh], k, 1, 0);
            if (!counted) return std::move(counted).error();
            wantedBy_[k] = static_cast<uint32_t>(std::min<uint64_t>(counted->wanted, 0xFFFFFFFFu));
            total += wantedBy_[k];
        }
        if (total > options_->maxSplats) {
            // A slot for every piece that wants any, and the rest of the
            // budget in proportion: counts, not values.
            uint64_t wanting = 0;
            for (size_t k = 0; k < pieces; ++k) {
                wanting += wantedBy_[k] > 0 ? 1 : 0;
            }
            const uint64_t spread = options_->maxSplats > wanting ? options_->maxSplats - wanting : 0;
            for (size_t k = 0; k < pieces; ++k) {
                if (wantedBy_[k] > 0) {
                    const uint64_t part = spread * wantedBy_[k] / total;
                    shareOf_[k] = static_cast<uint32_t>(std::min<uint64_t>(part + 1, wantedBy_[k]));
                }
            }
            ATHENEA_TRY(deriveCells());
            std::fprintf(stderr,
                         "mesh2splat: warning: the meshes want %llu splats and --max-splats is %llu: the budget "
                         "is shared in proportion, every piece's cell coarsened alike\n",
                         static_cast<unsigned long long>(total),
                         static_cast<unsigned long long>(options_->maxSplats));
        } else {
            shareOf_ = wantedBy_;
        }
        for (size_t k = 0; k < pieces_.size(); ++k) {
            if (triangles_[k] == 0) {
                continue;
            }
            const Piece& piece = pieces_[k];
            const usd::StageMesh& owner = meshes[piece.mesh];
            const usd::StageMaterial& material = *piece.material;
            const uint64_t room = options_->maxSplats > written ? options_->maxSplats - written : 0;
            if (room == 0 || shareOf_[k] == 0) {
                if (room > 0) {
                    continue;   // it wanted nothing
                }
                // Said, not left to be noticed: every mesh from here on is
                // missing from the cloud.
                for (size_t rest = k; rest < pieces_.size(); ++rest) {
                    unconverted += triangles_[rest] > 0 ? 1 : 0;
                }
                break;
            }
            // THE MATTE NAMES PRIMS: a subset's gaussians are its mesh's,
            // as Hydra's ids are -- the matte picks the mesh, whatever its
            // faces are bound to.
            const uint32_t meshCrypto = athenea::core::cryptomatteId(owner.path);
            cryptoManifest_[owner.path] = meshCrypto;
            const usd::StageMaterial& what = material;
            std::printf("mesh2splat: %s uses %s (colour %.2f %.2f %.2f, albedo '%s', metallic %.2f, "
                        "roughness %.2f, transmission %.3f, opacity %.3f%s%s)\n",
                        piece.path.c_str(), what.path.empty() ? "no material" : what.path.c_str(),
                        static_cast<double>(what.baseColour[0]), static_cast<double>(what.baseColour[1]),
                        static_cast<double>(what.baseColour[2]), what.albedo.file.c_str(),
                        static_cast<double>(what.metallic), static_cast<double>(what.roughness),
                        static_cast<double>(what.transmission), static_cast<double>(what.opacity),
                        what.opacityMap.empty() ? ""
                        : what.opacityThreshold > 0.0F
                            ? (", cut-out at " + std::to_string(what.opacityThreshold)).c_str()
                            : ", coverage map",
                        what.emits() ? (", emission " + std::to_string(what.emission[0]) + " " +
                                        std::to_string(what.emission[1]) + " " + std::to_string(what.emission[2]) +
                                        (what.emissionMap.empty() ? "" : " x '" + what.emissionMap.file + "'"))
                                           .c_str()
                                     : "");
            if (what.layered()) {
                std::printf("mesh2splat: %s layers specular %.2f x (%.2f %.2f %.2f) at %.3f, coat %.2f rough %.2f "
                            "at %.3f, sheen (%.2f %.2f %.2f) rough %.2f\n",
                            piece.path.c_str(), static_cast<double>(what.specularWeight),
                            static_cast<double>(what.specularColour[0]), static_cast<double>(what.specularColour[1]),
                            static_cast<double>(what.specularColour[2]), static_cast<double>(what.ior),
                            static_cast<double>(what.coatWeight), static_cast<double>(what.coatRoughness),
                            static_cast<double>(what.coatIor), static_cast<double>(what.sheenColour[0]),
                            static_cast<double>(what.sheenColour[1]), static_cast<double>(what.sheenColour[2]),
                            static_cast<double>(what.sheenRoughness));
            }
            // A surface whose material says it is not there -- an opacity of
            // nothing, or a constant under its own threshold -- has no
            // gaussian worth writing.
            // (The count gave it no share.)
            if (coversNothing(what)) {
                std::printf("mesh2splat: %s covers nothing (opacity 0), skipped\n", piece.path.c_str());
                continue;
            }
            // A PICTURE FOR WHAT THIS MESH CAN WANT, NOT FOR THE WHOLE
            // BUDGET.
            //
            // The output used to be sized for everything still unconverted, a
            // mesh at a time. On the chess pawn, which has two, nobody
            // noticed; on a car of a hundred and sixty it asks the device for
            // a hundred and sixty pictures of a hundred and fifty megabytes,
            // the pool runs out around the thirteenth, and what it serves
            // instead is a heap image the effect cannot be handed.
            //
            // So the first run is sized by a guess, and the effect says how
            // many the mesh wanted whether or not they fit: a mesh that
            // overflows is run again at exactly that, and nothing else pays
            // for it.
            // A ceiling, because a picture is sixteen bytes an entry and six
            // entries a splat: two million is 192 MB, and the device pool
            // holds two gigabytes for every picture a conversion has open at
            // once. Asking for more than this does not fail, it dies --
            // measured on a car whose body wanted ten million at resolution
            // 1024 and took the process with it.
            // AND A MESH THAT WANTS MORE THAN THE CEILING IS CONVERTED IN
            // SLICES, each run starting at the triangle the last one's
            // budget cut into (the effect says which), in the mesh's own
            // order, so the output is the same array a single run of the
            // whole would have written. The sparrow's feathers wanted 6.7 M
            // and got the first 2.1 M: the cards later in the mesh -- half
            // the head and the breast -- were not in the cloud at all.
            constexpr uint64_t kRunCeiling = 2u << 20;
            uint64_t meshWritten = 0;
            uint64_t meshWanted = 0;
            bool     carried = false;
            uint32_t first = 0;
            uint32_t slices = 0;
            while (first < triangles_[k]) {
                // This piece's share, and never past the whole budget.
                const uint64_t global = options_->maxSplats > written ? options_->maxSplats - written : 0;
                const uint64_t mine = shareOf_[k] > meshWritten ? shareOf_[k] - meshWritten : 0;
                const uint64_t left = std::min(global, mine);
                if (left == 0) {
                    break;
                }
                // The count said how many: the first run is sized for them.
                const uint64_t guess = std::clamp<uint64_t>(left, 1, kRunCeiling);
                auto out = runOne(effect, owner, k, guess, first);
                if (!out) return std::move(out).error();
                if (out->wanted > out->written && out->written < std::min(left, kRunCeiling)) {
                    const uint64_t again = std::min({out->wanted, left, kRunCeiling});
                    if (again > guess) {
                        out = runOne(effect, owner, k, again, first);
                        if (!out) return std::move(out).error();
                        ++reruns;
                    }
                }
                if (slices == 0) {
                    meshWanted = out->wanted;   // the first run counts everything from here on
                    capped += out->capped;      // and so every cell the relief asked of
                    beyond += out->beyond;      // and every cell past the per-triangle bound
                }
                ++slices;
                written += out->written;
                meshWritten += out->written;
                degenerate += out->degenerate;
                // Into the cloud's own buffers, after everything before it,
                // on the device: the records, where the bake starts from,
                // and the joints.
                ATHENEA_TRY(keep(*out, written - out->written));
                displaced_ = displaced_ || (out->written > 0 && displaces(material));
                // WHICH PRIM THESE GAUSSIANS CAME FROM. The conversion knows
                // it -- this run is one mesh -- so the ancestry a matte needs
                // is inherited here and nowhere else (AtheneaSplatCryptomatteAPI).
                cryptoIds_.insert(cryptoIds_.end(), out->written, meshCrypto);
                // And whether it is a sheet: a thin wall's transmission is
                // its gaussians' own transparency (see `glassOpacity`).
                thinWalled_.insert(thinWalled_.end(), out->written,
                                   thinGlass(material) ? int32_t{1} : int32_t{0});
                // AND WHAT ITS GLASS BENDS BY. A transmitting gaussian
                // refracts only with an index (rt_shade: `ior > 1`), and a
                // cloud keeps one: without it the pawn's glass head was a
                // milky ball in every mode, relit, transferred or baked.
                if (out->written > 0 && material.transmission > 0.0F && !material.thinWalled) {
                    const float ior = material.ior;
                    if (glassIor_ > 0.0F && glassIor_ != ior) {
                        std::fprintf(stderr,
                                     "mesh2splat: %s bends by %.3f and an earlier glass by %.3f; a cloud keeps "
                                     "one index, the first\n",
                                     piece.path.c_str(), static_cast<double>(ior),
                                     static_cast<double>(glassIor_));
                    } else {
                        glassIor_ = ior;
                    }
                }
                // A MESH NOTHING CARRIES STILL TAKES ITS PLACE IN THE RIG.
                //
                // A stage's skinned meshes are rarely all of them -- the
                // sparrow comes with a cylinder and a plane beside the bird
                // -- and the influences have to stay one to one with the
                // gaussians or the cloud and its rig disagree about who is
                // who. Those gaussians get four joints of no weight, which
                // is what the skinner reads as "leave this one where the
                // bind pose put it".
                // (`keep` wrote them: the mesh's own, or four of no weight.)
                carried = carried || (out->carried && out->written > 0);
                if (out->wanted <= out->written || out->done <= first || out->done >= triangles_[k]) {
                    break;   // everything from here fit, or nothing more can
                }
                first = static_cast<uint32_t>(out->done);
            }
            wanted += meshWanted;
            // The cell this piece walked, as `mesh2splat_cells` worked it
            // out: the box it was measured over (or the camera), its longest
            // side over the resolution, held to the bounds, and coarsened to
            // fit its share. Said here so a log reads what a part got.
            const double cell = static_cast<double>(cellValues_[3 + k]);
            const double factor = static_cast<double>(cellValues_[3 + pieces + k]);
            std::printf("mesh2splat: %s -> %llu splats of %llu wanted (%u triangles%s, cell %.4g%s)%s\n",
                        piece.path.c_str(), static_cast<unsigned long long>(meshWritten),
                        static_cast<unsigned long long>(meshWanted), triangles_[k],
                        slices > 1 ? (", " + std::to_string(slices) + " slices").c_str() : "", cell,
                        factor > 1.0 ? (", " + std::to_string(factor).substr(0, 4) + "x coarser for the budget").c_str()
                                     : "",
                        carried ? (", carried by " + owner.skinning.skeleton).c_str() : "");
        }
        if (written == 0) {
            return Error(ErrorCode::InvalidArgument, "the conversion produced no splats");
        }
        if (wanted > written) {
            std::fprintf(stderr,
                         "mesh2splat: warning: the budget is exhausted: %llu splats did not fit --max-splats %llu; "
                         "raise --max-splats, lower --resolution, or with --density per-mesh raise --cell-min\n",
                         static_cast<unsigned long long>(wanted - written),
                         static_cast<unsigned long long>(options_->maxSplats));
        }
        if (unconverted > 0) {
            std::fprintf(stderr, "mesh2splat: warning: the budget ran out before %zu mesh(es), which are not in "
                                 "the cloud\n",
                         unconverted);
        }
        if (beyond > 0) {
            // A triangle walks at most --max-cells cells; the rest of a large
            // one is left bare, which reads as a hole in the cloud.
            std::fprintf(stderr,
                         "mesh2splat: warning: %llu cells lay past --max-cells %u on their triangles and were "
                         "not sampled; raise --max-cells or lower --resolution\n",
                         static_cast<unsigned long long>(beyond), options_->maxCells);
        }
        if (reruns > 0) {
            std::printf("mesh2splat: %u mesh run(s) wanted more than the first guess and ran again\n", reruns);
        }
        if (degenerate > 0) {
            std::printf("mesh2splat: %llu triangles had no frame to stand a gaussian on\n",
                        static_cast<unsigned long long>(degenerate));
        }
        if (capped > 0) {
            std::printf("mesh2splat: %llu cells of relief wanted more than %u gaussians along an axis and "
                        "were left thinner (--displace-refine)\n",
                        static_cast<unsigned long long>(capped), options_->displaceRefine);
        }
        raw.count = static_cast<uint32_t>(written);
        raw.records = records_;
        count_ = raw.count;
        return raw;
    }

private:
    static constexpr uint32_t kChunkEntries = 1024;
    static constexpr uint32_t kNoPrimvar = 0xFFFFFFFF;

    /// Floats a record: the canonical fourteen, the three the material
    /// reflects with, the shading normal, the emission, and the harmonics
    /// where a bake writes them.
    [[nodiscard]] uint32_t recordFloats() const {
        return 23 + (options_->bake ? kRestPerDegree[std::min(options_->bakeDegree, 3u)] : 0) * 3 +
               (layered_ ? 13u : 0u);
    }

    struct OneMesh {
        uint64_t           written = 0;
        uint64_t           wanted = 0;
        uint64_t           degenerate = 0;
        /// Cells the relief would have split finer than `--displace-refine`.
        uint64_t           capped = 0;
        /// Cells past `--max-cells` on their triangle, left unwalked (the
        /// effect's fourth counter).
        uint64_t           beyond = 0;
        /// The first triangle the budget cut into; the triangle count when
        /// everything fit. Where the next slice starts.
        uint64_t           done = 0;
        /// What the effect wrote, as it wrote it: `recordEntries` float4
        /// entries a gaussian, the record's own `ownEntries` and, where the
        /// material displaces, three more (`mesh2splat_gather` reads them).
        image::ImagePtr    picture;
        uint32_t           recordEntries = 0;
        uint32_t           ownEntries = 0;
        bool               displaced = false;
        /// The gaussians carry the joints that move them: (joint, weight)
        /// four times a gaussian, beside the record rather than in it --
        /// `io::SplatEncoding` has no field for a skeleton.
        bool               carried = false;
    };

    [[nodiscard]] Result<OneMesh> runOne(aofx::Effect& effect, const usd::StageMesh& mesh, size_t at,
                                         uint64_t room, uint32_t firstTriangle = 0) {
        const usd::StageMaterial& material = *pieces_[at].material;
        // A MAP THAT WILL NOT FIT IS A MAP THIS MATERIAL DOES NOT HAVE.
        //
        // The device pool is finite and a stage decides how many maps it
        // wants, so the two can disagree -- and when they do, the material's
        // own constants are a poorer answer than the map and a far better one
        // than no conversion at all. Said once a file, not once a mesh.
        const auto mapOrNone = [&](const std::string& first, const std::string& second, bool pack) {
            auto picture = mapPicture(first, second, pack);
            if (picture) {
                return *picture;
            }
            if (refused_.insert(first + "|" + second).second) {
                std::fprintf(stderr, "mesh2splat: '%s' is left out: %s\n",
                             (first.empty() ? second : first).c_str(), picture.error().toString().c_str());
            }
            return image::ImagePtr{};
        };
        const image::ImagePtr albedoMap =
            options_->noTextures || material.albedo.empty() ? image::ImagePtr{}
                                                            : mapOrNone(material.albedo.file, {}, false);
        const image::ImagePtr normalMap =
            options_->noTextures || material.normal.empty() ? image::ImagePtr{}
                                                            : mapOrNone(material.normal.file, {}, false);
        const bool anyMr = !options_->noTextures &&
                           (!material.metallicMap.empty() || !material.roughnessMap.empty());
        const image::ImagePtr mrMap =
            anyMr ? mapOrNone(material.metallicMap.file, material.roughnessMap.file, true)
                  : image::ImagePtr{};
        // THE CUT-OUT. A map on `opacity` is not a transmission: below the
        // cut the surface is not there at all. Loaded raw -- a mask is not
        // colour and must not be taken through sRGB.
        const image::ImagePtr cutMap =
            options_->noTextures || material.opacityMap.empty()
                ? image::ImagePtr{}
                : mapOrNone(material.opacityMap.file, {}, false);
        // What it gives off, where that is a map.
        const image::ImagePtr emissionMap =
            options_->noTextures || !emits_ || material.emissionMap.empty()
                ? image::ImagePtr{}
                : mapOrNone(material.emissionMap.file, {}, false);
        const image::ImagePtr* albedo = &albedoMap;
        const image::ImagePtr* normal = &normalMap;
        const image::ImagePtr* mr = &mrMap;

        // The records the effect writes go into a picture of their own, four
        // entries a splat: position and opacity, the three sizes, the rotation,
        // the colour.
        // Four entries a record, six with the PBR channels, nine when the
        // gaussian carries the joints that move it and how their weights
        // change across it.
        const bool carried = mesh.skinning.bound && skins_[at];
        const uint32_t kRecordEntries = carried ? 9U : 6U;
        const uint64_t budget = std::min<uint64_t>(room, 1ull << 23);
        // DISPLACED: three entries more a record -- the point of the flat
        // surface and its height, that surface's normal, the relief's.
        const bool displaced = !options_->noDisplacement && material.displaces();
        const image::ImagePtr heightMap =
            displaced && !options_->noTextures && !material.displacementMap.empty()
                ? mapOrNone(material.displacementMap.file, {}, false)
                : image::ImagePtr{};
        const uint32_t ownEntries = kRecordEntries;
        // AND ONE FOR WHAT IT GIVES OFF, the last, where the stage emits.
        // AND FOUR FOR WHAT THE MATERIAL LAYERS OVER ITS BASE, after it.
        const uint32_t recordEntries =
            kRecordEntries + (displaced ? 3U : 0U) + (emits_ ? 1U : 0U) + (layered_ ? 4U : 0U);
        const image::PixelRect bounds = pictureFor(budget * recordEntries);

        // The box the density is measured over: the model's, or this mesh's
        // own. An attachment replaces by id, so a mesh converted in slices
        // sees its box on every run.
        if (perMesh_ && at < meshBounds_.size()) {
            const std::array<float, 6>& own = meshBounds_[pieces_[at].mesh];
            streams_[at]->attach("bounds", {own[0], own[1], own[2], own[3], own[4], own[5]});
        } else {
            streams_[at]->attach("bounds", {boundsMin_[0], boundsMin_[1], boundsMin_[2], boundsMax_[0],
                                            boundsMax_[1], boundsMax_[2]});
        }
        aofx_host::EffectJob job;
        job.bounds = bounds;
        job.instance = "athenea/mesh2splat/" + std::to_string(at);
        job.inputs.push_back({"Mesh", streams_[at]});
        if (*albedo) job.inputs.push_back({"Albedo", *albedo});
        if (*normal) job.inputs.push_back({"Normal", *normal});
        if (*mr) job.inputs.push_back({"MetallicRoughness", *mr});
        if (cutMap) job.inputs.push_back({"Opacity", cutMap});
        if (carried) job.inputs.push_back({"Influences", skins_[at]});
        if (uv2s_[at]) job.inputs.push_back({"Texcoord2", uv2s_[at]});
        if (heightMap) job.inputs.push_back({"Displacement", heightMap});
        if (emissionMap) job.inputs.push_back({"Emission", emissionMap});

        const auto number = [&job](const char* name, double value) {
            job.params.push_back(aofx::ParamValue{name, {value}, {}});
        };
        number("triangles", static_cast<double>(triangles_[at]));
        // The piece's own: its share of the budget may coarsen it, and a
        // camera may decide its cell (`mesh2splat_cells`).
        number("resolution", static_cast<double>(cellValues_[3 + 2 * pieces_.size() + at]));
        number("maxSplats", static_cast<double>(budget));
        number("firstTriangle", static_cast<double>(firstTriangle));
        number("flatness", options_->flatness);
        // EVERY OPACITY HERE IS COVERAGE: how much of what stands behind the
        // surface it covers. The effect multiplies them -- `--opacity`, the
        // material's constant, the map's value, what a glass keeps -- and
        // only then gives each gaussian what one of the several over a point
        // needs for that (`m2sCoverageAlpha`).
        // Written straight into each gaussian, a mask of 0.5 covered 96% and
        // a glass kept at 0.6 covered 98%.
        number("coverage", 1.0);
        number("opacity", options_->opacity);
        number("materialOpacity", static_cast<double>(material.opacity));
        // A THIN WALL IS ITS OWN TRANSPARENCY: it covers what the sheet
        // reflects head on at its index, which the effect works out
        // (`m2sGlassCovers`). A solid covers `--glass-opacity`.
        number("glassOpacity", options_->minOpacity);
        number("thinWall", thinGlass(material) ? 1.0 : 0.0);
        number("ior", static_cast<double>(material.ior));
        number("maxCells", static_cast<double>(options_->maxCells));
        number("cellMin", static_cast<double>(cellValues_[3 + 3 * pieces_.size() + at]));
        number("cellMax", static_cast<double>(cellValues_[3 + 4 * pieces_.size() + at]));
        number("cellByLongest", perMesh_ || camera_ ? 1.0 : 0.0);
        number("useNormalMap", options_->normalMapTurns ? 1.0 : 0.0);
        number("simplify", options_->simplify);
        number("simplifyLevels", static_cast<double>(options_->simplifyLevels));
        if (cutMap) {
            const char channel = material.opacityMap.channel;
            const double which = channel == 'r'   ? 1.0
                                 : channel == 'g' ? 2.0
                                 : channel == 'b' ? 3.0
                                                  : 4.0;
            number("opacityChannel", which);
            // A threshold the material names is its own cut, and what it
            // keeps is whole; without one the map is coverage, cut where it
            // is too faint to be worth a gaussian.
            const bool threshold = material.opacityThreshold > 0.0F;
            number("opacityCut", threshold ? static_cast<double>(material.opacityThreshold) : options_->opacityCut);
            number("opacityBinary", threshold ? 1.0 : 0.0);
        }
        // Six entries a splat: the four a gaussian is, and the two that say
        // what it reflects with.
        number("writePbr", 1.0);
        number("writeInfluences", carried ? 1.0 : 0.0);
        if (displaced) {
            // The height in the world's units: authored in the mesh's own.
            const char channel = material.displacementMap.channel;
            number("displace", 1.0);
            number("displaceChannel", channel == 'g' ? 2.0 : channel == 'b' ? 3.0 : channel == 'a' ? 4.0 : 1.0);
            number("displaceScale", static_cast<double>(material.displacementScale) * mesh.displacementUnit);
            number("displaceBias", static_cast<double>(material.displacementBias) * mesh.displacementUnit);
            number("displaceRefine", static_cast<double>(options_->displaceRefine));
        }
        if (uv2s_[at]) {
            // Which maps the material reads by the second set of coordinates.
            const auto bySecond = [&mesh](const usd::StageTexture& texture) {
                return !texture.empty() && texture.uvSet == mesh.uv2 ? 1.0 : 0.0;
            };
            number("albedoUv2", bySecond(material.albedo));
            number("normalUv2", bySecond(material.normal));
            number("mrUv2", bySecond(material.metallicMap.empty() ? material.roughnessMap : material.metallicMap));
            number("opacityUv2", bySecond(material.opacityMap));
            number("displaceUv2", bySecond(material.displacementMap));
            number("emissionUv2", bySecond(material.emissionMap));
        }
        number("transmission", static_cast<double>(material.transmission));
        number("metallic", static_cast<double>(material.metallic));
        number("roughness", static_cast<double>(material.roughness));
        job.params.push_back(aofx::ParamValue{"sigma", {options_->sigma, options_->sigma}, {}});
        const auto colour = [&job](const char* name, const std::array<float, 3>& rgb) {
            job.params.push_back(aofx::ParamValue{name,
                                                 {static_cast<double>(rgb[0]), static_cast<double>(rgb[1]),
                                                  static_cast<double>(rgb[2])},
                                                 {}});
        };
        colour("materialColour", material.baseColour);
        colour("transmissionColour", material.transmissionColour);
        if (emits_) {
            number("writeEmission", 1.0);
            colour("emissionColour", material.emission);
            const char channel = material.emissionMap.channel;
            number("emissionChannel", channel == 'r'   ? 1.0
                                      : channel == 'g' ? 2.0
                                      : channel == 'b' ? 3.0
                                      : channel == 'a' ? 4.0
                                                       : 0.0);
        }

        if (layered_) {
            // Every piece writes them once one does, so the records stay one
            // layout; a material that names none writes the plain ones.
            number("writeLobes", 1.0);
            number("specularWeight", static_cast<double>(material.specularWeight));
            colour("specularColour", material.specularColour);
            number("specularIor", static_cast<double>(material.ior));
            number("coatWeight", static_cast<double>(material.coatWeight));
            number("coatRoughness", static_cast<double>(material.coatRoughness));
            number("coatIor", static_cast<double>(material.coatIor));
            colour("sheenColour", material.sheenColour);
            number("sheenRoughness", static_cast<double>(material.sheenRoughness));
            number("coatDarkening", static_cast<double>(material.coatDarkening));
        }

        auto rendered = aofx_host::renderEffect(*context_, effect, job);
        if (!rendered) return std::move(rendered).error();
        const image::Image& out = **rendered;
        const std::vector<float>* counted = out.attached("splats");
        if (counted == nullptr || counted->size() < 4) {
            return Error(ErrorCode::DeviceFailure, "the effect did not say how many splats it wrote");
        }
        OneMesh answer;
        answer.written = static_cast<uint64_t>(std::max((*counted)[0], 0.0F));
        answer.wanted = static_cast<uint64_t>(std::max((*counted)[1], 0.0F));
        answer.degenerate = static_cast<uint64_t>(std::max((*counted)[2], 0.0F));
        answer.beyond = static_cast<uint64_t>(std::max((*counted)[3], 0.0F));
        answer.done = counted->size() >= 6 ? static_cast<uint64_t>(std::max((*counted)[5], 0.0F))
                                           : uint64_t{triangles_[at]};
        answer.written = std::min(answer.written, budget);
        answer.capped = counted->size() >= 7 ? static_cast<uint64_t>(std::max((*counted)[6], 0.0F)) : 0;

        answer.picture = *rendered;
        answer.recordEntries = recordEntries;
        answer.ownEntries = ownEntries;
        answer.displaced = displaced;
        answer.carried = carried;
        return answer;
    }

    /// Room on the device for `splats` gaussians, what is there kept. The
    /// buffers grow by doubling, so a conversion of many runs copies each
    /// record a few times at most, on the device.
    [[nodiscard]] Result<void> reserve(uint64_t splats) {
        if (splats <= capacity_) {
            return ok();
        }
        gpu::Device& device = library_->device();
        const uint64_t room = std::max<uint64_t>({splats, capacity_ * 2, uint64_t{1} << 16});
        const uint32_t perRecord = recordFloats();
        const auto grown = [&](gpu::Buffer& held, uint64_t perSplat, uint32_t element,
                               const char* label) -> Result<void> {
            gpu::BufferDesc desc;
            desc.bytes = room * perSplat;
            desc.elementBytes = element;
            desc.label = label;
            auto made = gpu::Buffer::create(device, desc);
            if (!made) return std::move(made).error();
            if (held.valid() && used_ > 0) {
                gpu::CommandBatch batch(device);
                batch.encoder()->copyBuffer(made->rhi(), 0, held.rhi(), 0, used_ * perSplat);
                batch.markDirty();
                ATHENEA_TRY(batch.submit(true));
            }
            held = std::move(*made);
            return ok();
        };
        ATHENEA_TRY(grown(records_, uint64_t{perRecord} * 4, 4, "mesh2splat.records"));
        ATHENEA_TRY(grown(rays_, 48, 16, "mesh2splat.rays"));
        if (options_->skinned) {
            ATHENEA_TRY(grown(influences_, 32, 16, "mesh2splat.influences"));
            ATHENEA_TRY(grown(gradients_, 12, 4, "mesh2splat.weightGradients"));
        }
        capacity_ = room;
        return ok();
    }

    /// One run's gaussians into the cloud's buffers, from `destFirst` on: the
    /// records, where each one's bake starts, and the joints that carry it.
    [[nodiscard]] Result<void> keep(const OneMesh& run, uint64_t destFirst) {
        if (run.written == 0) {
            return ok();
        }
        ATHENEA_TRY(reserve(destFirst + run.written));
        auto view = viewOf(*context_, run.picture, "mesh2splat.records");
        if (!view) return std::move(view).error();
        const bool carried = run.carried && options_->skinned;
        gpu::CommandBatch batch(library_->device());
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["picture"].setBinding(view->rhi());
            cursor["records"].setBinding(records_.rhi());
            cursor["rays"].setBinding(rays_.rhi());
            // Bound whether or not there is a skeleton: a kernel's buffers
            // are all bound, and `carried` is what stops it being written.
            cursor["influences"].setBinding(influences_.valid() ? influences_.rhi() : rays_.rhi());
            cursor["gradients"].setBinding(gradients_.valid() ? gradients_.rhi() : rays_.rhi());
            cursor["gather"]["count"].setData(static_cast<uint32_t>(run.written));
            cursor["gather"]["destFirst"].setData(static_cast<uint32_t>(destFirst));
            cursor["gather"]["recordEntries"].setData(run.recordEntries);
            cursor["gather"]["ownEntries"].setData(run.ownEntries);
            cursor["gather"]["width"].setData(static_cast<uint32_t>(run.picture->bounds().width()));
            cursor["gather"]["stride"].setData(static_cast<uint32_t>(run.picture->stride()));
            cursor["gather"]["perRecord"].setData(recordFloats());
            cursor["gather"]["displaced"].setData(run.displaced ? 1u : 0u);
            cursor["gather"]["carried"].setData(carried ? 1u : 0u);
            cursor["gather"]["overArea"].setData(options_->simplify > 0.0 ? 1u : 0u);
            cursor["gather"]["emits"].setData(emits_ ? 1u : 0u);
            cursor["gather"]["lobes"].setData(layered_ ? 1u : 0u);
        };
        const uint32_t threads = static_cast<uint32_t>(run.written);
        gather_.dispatch(batch, {threads, 1, 1}, bind);
        // A MESH NOTHING CARRIES STILL TAKES ITS PLACE IN THE RIG. A stage's
        // skinned meshes are rarely all of them -- the sparrow comes with a
        // cylinder and a plane beside the bird -- and the influences have to
        // stay one to one with the gaussians or the cloud and its rig
        // disagree about who is who: four joints of no weight, which the
        // skinner reads as "leave this one where the bind pose put it".
        if (options_->skinned && !carried) {
            noInfluence_.dispatch(batch, {threads, 1, 1}, bind);
        }
        ATHENEA_TRY(batch.submit(true));
        used_ = std::max<uint64_t>(used_, destFirst + run.written);
        return ok();
    }

    /// THE RAYS' OFFSET, ON THE DEVICE: the cloud's box folded from its
    /// records (a low and a high corner a chunk, then the lot), and a
    /// ten-thousandth of its diagonal written into every ray
    /// (`athenea/usd/mesh2splat_span` says why that and not the scene's unit).
    [[nodiscard]] Result<void> spanRays() {
        if (count_ == 0) {
            return Error(ErrorCode::InternalError, "bake: no gaussians to start rays from");
        }
        gpu::Device& device = library_->device();
        const uint32_t chunks = (count_ + kChunkEntries - 1) / kChunkEntries;
        gpu::BufferDesc desc;
        desc.bytes = uint64_t{chunks} * 2 * 16;
        desc.elementBytes = 16;
        desc.label = "mesh2splat.recordExtents";
        auto extents = gpu::Buffer::create(device, desc);
        if (!extents) return std::move(extents).error();
        desc.bytes = 2 * 16;
        desc.label = "mesh2splat.cloudBox";
        auto box = gpu::Buffer::create(device, desc);
        if (!box) return std::move(box).error();
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["records"].setBinding(records_.rhi());
            cursor["extents"].setBinding(extents->rhi());
            cursor["box"].setBinding(box->rhi());
            cursor["rays"].setBinding(rays_.rhi());
            cursor["span"]["count"].setData(count_);
            cursor["span"]["perRecord"].setData(recordFloats());
            cursor["span"]["chunkSize"].setData(kChunkEntries);
            cursor["span"]["chunkCount"].setData(chunks);
        };
        gpu::CommandBatch batch(device);
        recordChunks_.dispatch(batch, {chunks, 1, 1}, bind);
        reduce_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["extents"].setBinding(extents->rhi());
            cursor["result"].setBinding(box->rhi());
            cursor["params"]["count"].setData(count_);
            cursor["params"]["chunkSize"].setData(kChunkEntries);
            cursor["params"]["chunkCount"].setData(chunks);
        });
        raySpan_.dispatch(batch, {count_, 1, 1}, bind);
        return batch.submit(true);
    }

    /// One counter, cleared.
    [[nodiscard]] Result<gpu::Buffer> counter() {
        const uint32_t zero[4] = {0, 0, 0, 0};
        gpu::BufferDesc desc;
        desc.bytes = sizeof(zero);
        desc.elementBytes = 4;
        desc.label = "mesh2splat.counter";
        return gpu::Buffer::create(library_->device(), desc, zero);
    }

public:
    /// The path tracer's answer at every gaussian, written into its colour
    /// (below, beside the free functions it replaced).
    [[nodiscard]] Result<void> bake(const std::string& stage, double time, const usd::BakeOptions& bake,
                                    bool defaultLights, aofx::Effect* filter);
    /// THE SPLAT BAKE FILTER over the indirect half (plugins/splatbakefilter):
    /// the half packed into the two pictures the effect reads, and its answer
    /// unpacked back over it (`athenea/usd/bake_filter_io`).
    [[nodiscard]] Result<void> filterIndirect(aofx::Effect& filter, usd::BakeSplit& split);
    /// How much of an environment reaches each gaussian, instead of the light.
    [[nodiscard]] Result<void> transfer(const std::string& stage, double time, uint32_t samples,
                                        uint32_t bounces, bool indirect, uint32_t cells, std::vector<float>& direct,
                                        std::vector<float>& bounced, std::vector<int32_t>& shadowBits);

private:
    /// Whether any material of the stage gives off light: the records then
    /// carry it (`io::SplatEncoding::emission`).
    bool                                     emits_ = false;
    /// Whether any material of the stage layers anything over its base
    /// (`usd::StageMaterial::layered`): the records then carry the thirteen
    /// floats of `io::SplatEncoding::lobes`, last.
    bool                                     layered_ = false;
    /// The Cryptomatte id of the prim each splat came from, in the same order,
    /// and what those ids are called.
    std::vector<uint32_t>                    cryptoIds_;
    /// 1 where the splat came from a thin-walled glass, in the same order.
    std::vector<int32_t>                     thinWalled_;
    /// The index the cloud's transmitting gaussians bend by: the first glass
    /// met's. 0 while there is none.
    float                                    glassIor_ = 0.0F;
    std::map<std::string, uint32_t>          cryptoManifest_;
    std::set<std::string>                    refused_;   ///< maps the device would not hold
    /// THE CLOUD, ON THE DEVICE, every run's gaussians at their place:
    /// `recordFloats()` floats a record, three float4 a ray, two float4 of
    /// joints where a skeleton carries it.
    gpu::Buffer                              records_;
    gpu::Buffer                              rays_;
    gpu::Buffer                              influences_;
    /// How those joints' weights change across each gaussian: three words of
    /// two halves a gaussian (`jointWeightGradients`), beside the joints.
    gpu::Buffer                              gradients_;
    uint64_t                                 capacity_ = 0;   ///< gaussians the buffers hold
    uint64_t                                 used_ = 0;       ///< gaussians written into them
    uint32_t                                 count_ = 0;      ///< gaussians the conversion kept
    bool                                     displaced_ = false;   ///< a ray faces as its relief does

public:
    [[nodiscard]] bool coversNothing(const usd::StageMaterial& material) const noexcept {
        return material.opacity <= 0.0F || options_->opacity <= 0.0;
    }
    [[nodiscard]] bool displaces(const usd::StageMaterial& material) const noexcept {
        return !options_->noDisplacement && material.displaces();
    }
    [[nodiscard]] const std::vector<uint32_t>& cryptoIds() const noexcept { return cryptoIds_; }
    [[nodiscard]] double modelCell() const noexcept { return modelCell_; }
    [[nodiscard]] const std::vector<int32_t>& thinWalled() const noexcept { return thinWalled_; }
    [[nodiscard]] float glassIor() const noexcept { return glassIor_; }
    [[nodiscard]] const std::map<std::string, uint32_t>& cryptoManifest() const noexcept {
        return cryptoManifest_;
    }
    /// (joint, weight) four times a gaussian, empty when nothing carries it:
    /// read back for the file, which wants them as an array.
    [[nodiscard]] Result<std::vector<float>> influences() const {
        if (!influences_.valid() || count_ == 0) {
            return std::vector<float>{};
        }
        std::vector<float> out(size_t{count_} * 8);
        ATHENEA_TRY(influences_.read(library_->device(), 0, out.size() * sizeof(float), out.data()));
        return out;
    }
    /// And how their weights change across each gaussian: three words of
    /// halves a gaussian, read back as the device packed them.
    [[nodiscard]] Result<std::vector<uint32_t>> weightGradients() const {
        if (!gradients_.valid() || count_ == 0) {
            return std::vector<uint32_t>{};
        }
        std::vector<uint32_t> out(size_t{count_} * 3);
        ATHENEA_TRY(gradients_.read(library_->device(), 0, out.size() * sizeof(uint32_t), out.data()));
        return out;
    }

private:
    gpu_host::Context*                       context_ = nullptr;
    gpu::ShaderLibrary*                      library_ = nullptr;
    const Options*                           options_ = nullptr;
    gpu::ComputeKernel                       pack_, chunks_, reduce_, reduceSlices_, rows_;
    gpu::ComputeKernel                       gather_, noInfluence_, recordChunks_, raySpan_, bakeInto_,
                                             transferInto_, subsetFlags_, subsetScatter_;
    gpu::PrefixSum                           prefix_;
    std::vector<Piece>                       pieces_;
    gpu::ComputeKernel                       cells_;
    gpu::Buffer                              boxes_;        ///< the model's box, then each mesh's
    std::vector<uint32_t>                    wantedBy_;     ///< what each piece wants, counted
    std::vector<uint32_t>                    shareOf_;      ///< what the budget gives it
    std::vector<float>                       cellValues_;   ///< mesh2splat_cells' answer
    std::optional<usd::StageCamera>          camera_;
    uint32_t                                 cameraPixels_ = 1920;
    std::unique_ptr<material::TextureStore>  textures_;
    std::map<std::string, uint32_t>          ids_;
    std::map<MapKey, image::ImagePtr>        maps_;
    std::vector<image::ImagePtr>             streams_;
    std::vector<image::ImagePtr>             skins_;
    std::vector<image::ImagePtr>             uv2s_;
    std::vector<uint32_t>                    triangles_;
    uint32_t                                 sampler_ = 0;
    std::array<float, 3>                     boundsMin_{0.0F, 0.0F, 0.0F};
    std::array<float, 3>                     boundsMax_{1.0F, 1.0F, 1.0F};
    /// Each mesh's own world-space box (min xyz, max xyz), folded on the device.
    std::vector<std::array<float, 6>>        meshBounds_;
    bool                                     perMesh_ = false;
    double                                   cellMin_ = 0.0;   ///< the cell's bounds sent to the effect (0: none)
    double                                   cellMax_ = 0.0;
    double                                   modelCell_ = 0.0;   ///< the model's longest side over the resolution
};

/// The path tracer's answer at every gaussian, written into its colour.
///
/// One ray a gaussian: from a little way along its normal, back down onto the
/// surface it came from. What the tracer finds there is the same surface the
/// mesh had -- the same material, the same texture, the same normal map --
/// and what it answers is the radiance leaving it along that normal, with
/// this stage's lights, its shadows and its bounces in it. That is the
/// conversion this engine can make and a relighting approximation cannot.
///
/// A direction had to be chosen, since one colour cannot be view-dependent,
/// and the surface's own normal is the one that needs no camera. What it
/// costs is the highlight that would only be seen from elsewhere.
///
/// ALL OF IT ON THE DEVICE. The rays are the cloud's own buffer, their offset
/// set from the cloud's box by a kernel; the renderer is opened on the
/// conversion's device, so it traces those rays where they are and answers
/// into a device buffer; and a kernel writes the answer into the records.
/// What crosses back is the count of gaussians the bake found a surface
/// under.
Result<void> Converter::bake(const std::string& stage, double time, const usd::BakeOptions& options,
                             bool defaultLights, aofx::Effect* filter) {
    const auto started = std::chrono::steady_clock::now();
    const uint32_t samples = options.samples;
    const uint32_t bounces = options.bounces;
    const uint32_t degree = options.degree;
    ATHENEA_TRY(spanRays());
    auto renderer = usd::StageRenderer::open(stage, context_->deviceShared());
    if (!renderer) return std::move(renderer).error();
    if (defaultLights) {
        ATHENEA_TRY((*renderer)->setDefaultLights(true));
    }
    // THE BAKE IN TWO HALVES (StageRenderer::bakeSplitOnDevice): direct and
    // indirect as sums, more paths where they are worth most, the indirect
    // half filtered between neighbours, then the two added and fitted.
    auto split = (*renderer)->bakeSplitOnDevice(rays_, count_, time, options);
    if (!split) return std::move(split).error();
    const double traced =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    double filtered = 0.0;
    if (filter != nullptr && options_->bakeFilter > 0) {
        const auto from = std::chrono::steady_clock::now();
        ATHENEA_TRY(filterIndirect(*filter, *split));
        filtered = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
    }
    auto baked = (*renderer)->combineBake(*split, rays_);
    if (!baked) return std::move(baked).error();
    auto lit = counter();
    if (!lit) return std::move(lit).error();
    const uint32_t coefficients = (degree + 1) * (degree + 1);
    gpu::Buffer& none = *lit;   // what the radiance bake has no use for, bound all the same
    gpu::CommandBatch batch(library_->device());
    bakeInto_.dispatch(batch, {count_, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["baked"].setBinding(baked->rhi());
        cursor["records"].setBinding(records_.rhi());
        cursor["direct"].setBinding(none.rhi());
        cursor["bounced"].setBinding(none.rhi());
        cursor["shadowBits"].setBinding(none.rhi());
        cursor["counts"].setBinding(lit->rhi());
        cursor["bake"]["count"].setData(count_);
        cursor["bake"]["coefficients"].setData(coefficients);
        cursor["bake"]["perRecord"].setData(recordFloats());
        cursor["bake"]["opacity"].setData(uint32_t{3});
        cursor["bake"]["dc0"].setData(uint32_t{11});
        cursor["bake"]["restBase"].setData(uint32_t{23});
        cursor["bake"]["indirect"].setData(uint32_t{0});
    });
    ATHENEA_TRY(batch.submit(true));
    uint32_t found = 0;
    ATHENEA_TRY(lit->read(library_->device(), 0, sizeof(found), &found));
    const double took =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    std::printf("mesh2splat: baked %u of %u gaussians (%u paths each, %u bounces, degree %u) in %.0f ms\n", found,
                count_, samples, bounces, degree, took);
    if (split->extraPasses > 0) {
        std::printf("mesh2splat: %llu more paths in %u adaptive passes of %u (%.1f a gaussian on average)\n",
                    static_cast<unsigned long long>(split->extraPoints) * options.passSamples, split->extraPasses,
                    options.passSamples,
                    static_cast<double>(split->extraPoints) * options.passSamples / std::max(count_, 1u));
    }
    std::printf("mesh2splat: traced in %.0f ms, filtered in %.0f ms (%u passes)\n", traced, filtered,
                filter != nullptr ? options_->bakeFilter : 0u);
    if (found * 2 < count_) {
        std::fprintf(stderr,
                     "mesh2splat: more than half the gaussians found no surface under them; the bake "
                     "is unlikely to be what you want\n");
    }
    return ok();
}

/// THE INDIRECT HALF, FILTERED BETWEEN NEIGHBOURS, as an AOFX effect: what
/// it is handed is pictures in the host's own storage, written and read back
/// by kernels through views of the same memory.
Result<void> Converter::filterIndirect(aofx::Effect& filter, usd::BakeSplit& split) {
    gpu::Device& device = library_->device();
    const uint32_t count = split.count;
    const uint32_t coefficients = split.coefficients;
    auto points = image::Image::create(pictureFor(uint64_t{count} * 3));
    if (!points) return std::move(points).error();
    auto light = image::Image::create(pictureFor(uint64_t{count} * coefficients));
    if (!light) return std::move(light).error();
    auto pointsView = viewOf(*context_, *points, "mesh2splat.filterPoints");
    if (!pointsView) return std::move(pointsView).error();
    auto lightView = viewOf(*context_, *light, "mesh2splat.filterIndirect");
    if (!lightView) return std::move(lightView).error();
    // The prim each gaussian came from: the matte's ids, which the filter
    // does not average across. A cloud whose ids do not line up with its
    // gaussians is filtered as one prim, and says so.
    std::vector<uint32_t> ids(count, 0);
    if (cryptoIds_.size() == count) {
        ids = cryptoIds_;
    } else {
        std::fprintf(stderr, "mesh2splat: %zu ids for %u gaussians; the filter takes them as one prim\n",
                     cryptoIds_.size(), count);
    }
    auto idBuffer = gpu::Buffer::fromSpan<uint32_t>(device, ids, "mesh2splat.filterIds");
    if (!idBuffer) return std::move(idBuffer).error();
    const auto kernel = [&](const char* entry) {
        return gpu::ComputeKernel::create(*library_, "athenea/usd/bake_filter_io", entry);
    };
    auto pack = kernel("bakeFilterPoints");
    if (!pack) return std::move(pack).error();
    auto in = kernel("bakeFilterIn");
    if (!in) return std::move(in).error();
    auto out = kernel("bakeFilterOut");
    if (!out) return std::move(out).error();
    const auto bind = [&](rhi::ShaderCursor cursor, const gpu::Buffer& picture, const image::ImagePtr& image) {
        cursor["rays"].setBinding(rays_.rhi());
        cursor["records"].setBinding(records_.rhi());
        cursor["io"]["perRecord"].setData(recordFloats());
        cursor["io"]["size"].setData(uint32_t{4});
        cursor["ids"].setBinding(idBuffer->rhi());
        cursor["guides"].setBinding(split.guides.rhi());
        cursor["indirect"].setBinding(split.indirect.rhi());
        cursor["direct"].setBinding(split.direct.rhi());
        cursor["io"]["withDirect"].setData(options_->bakeFilterIndirectOnly ? 0u : 1u);
        cursor["picture"].setBinding(picture.rhi());
        cursor["io"]["count"].setData(count);
        cursor["io"]["coefficients"].setData(coefficients);
        cursor["io"]["width"].setData(static_cast<uint32_t>(image->bounds().width()));
        cursor["io"]["stride"].setData(static_cast<uint32_t>(image->stride()));
    };
    const uint32_t entries = count * coefficients;
    {
        gpu::CommandBatch batch(device);
        pack->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) { bind(c, *pointsView, *points); });
        in->dispatch(batch, {entries, 1, 1}, [&](rhi::ShaderCursor c) { bind(c, *lightView, *light); });
        ATHENEA_TRY(batch.submit(true));
        // Written on the device: the host must not hand the effect the
        // copy it holds.
        (*points)->deviceWrote();
        (*light)->deviceWrote();
    }
    aofx_host::EffectJob job;
    job.bounds = pictureFor(uint64_t{count} * coefficients);
    job.instance = "athenea/mesh2splat/bakefilter";
    job.inputs.push_back({"Points", *points});
    job.inputs.push_back({"Indirect", *light});
    const auto number = [&job](const char* name, double value) {
        job.params.push_back(aofx::ParamValue{name, {value}, {}});
    };
    number("count", static_cast<double>(count));
    number("coefficients", static_cast<double>(coefficients));
    number("iterations", static_cast<double>(options_->bakeFilter));
    number("sigmaLuminance", options_->bakeFilterLuminance);
    auto rendered = aofx_host::renderEffect(*context_, filter, job);
    if (!rendered) return std::move(rendered).error();
    const std::vector<float>* said = (*rendered)->attached("filtered");
    if (said == nullptr || said->size() < 3) {
        return Error(ErrorCode::DeviceFailure, "the splat bake filter did not say what it filtered");
    }
    std::printf("mesh2splat: the bake filtered over a %.3g cell (%.0f gaussians left out of a full one)\n",
                static_cast<double>((*said)[2]), static_cast<double>((*said)[1]));
    auto answer = viewOf(*context_, *rendered, "mesh2splat.filtered");
    if (!answer) return std::move(answer).error();
    gpu::CommandBatch batch(device);
    out->dispatch(batch, {entries, 1, 1}, [&](rhi::ShaderCursor c) { bind(c, *answer, *rendered); });
    return batch.submit(true);
}

/// HOW MUCH OF AN ENVIRONMENT REACHES EACH GAUSSIAN, baked instead of the
/// light itself.
///
/// The radiance bake above keeps the light of the dome that was there, so
/// under another sky it is wrong. This keeps the geometry instead --
/// visibility times the cosine, the surface's own albedo taken as one --
/// which is the same rays, the same stratification and the same ray offset,
/// with the projection done where the path escapes rather than where it
/// gathers. The frame then reads `albedo * dot(transfer, sky)` under whatever
/// sky the cloud is put in. On the device as the bake is; the three arrays
/// the file keeps come back as bytes.
Result<void> Converter::transfer(const std::string& stage, double time, uint32_t samples, uint32_t bounces,
                                 bool indirect, uint32_t cells, std::vector<float>& direct, std::vector<float>& bounced,
                                 std::vector<int32_t>& shadowBits) {
    ATHENEA_TRY(spanRays());
    auto renderer = usd::StageRenderer::open(stage, context_->deviceShared());
    if (!renderer) return std::move(renderer).error();
    // Degree 2: nine coefficients hold the irradiance of any environment to
    // about a percent, and a transfer is exactly that shape.
    const uint32_t side = technique::transferCellSide(cells);
    const auto started = std::chrono::steady_clock::now();
    auto baked = (*renderer)->bakePointsOnDevice(rays_, count_, time, samples, bounces, 2, /*transfer=*/true,
                                                 /*batch=*/0, side);
    if (!baked) return std::move(baked).error();
    // The rays are the same rays whether the indirect half is kept or not, so
    // this number is what says the second half costs no bake: only the copy
    // below and the file differ.
    const double traced =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    gpu::Device& device = library_->device();
    const auto made = [&](uint64_t words, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(words, 1) * 4;
        desc.elementBytes = 4;
        desc.label = label;
        return gpu::Buffer::create(device, desc);
    };
    auto directs = made(uint64_t{count_} * 9, "mesh2splat.transferDirect");
    auto bounceds = made(indirect ? uint64_t{count_} * 27 : 1, "mesh2splat.transferIndirect");
    // Two words of open directions a gaussian, or eight or thirty-two with
    // the cells.
    const uint32_t words = technique::transferCellWords(side);
    auto bits = made(uint64_t{count_} * words, "mesh2splat.shadowBits");
    auto found = counter();
    if (!directs || !bounceds || !bits || !found) {
        return Error(ErrorCode::OutOfMemory, "transfer: cannot allocate what the file keeps");
    }
    gpu::CommandBatch batch(device);
    transferInto_.dispatch(batch, {count_, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["baked"].setBinding(baked->rhi());
        cursor["records"].setBinding(records_.rhi());
        cursor["direct"].setBinding(directs->rhi());
        cursor["bounced"].setBinding(bounceds->rhi());
        cursor["shadowBits"].setBinding(bits->rhi());
        cursor["counts"].setBinding(found->rhi());
        cursor["bake"]["count"].setData(count_);
        cursor["bake"]["coefficients"].setData(uint32_t{9});
        cursor["bake"]["perRecord"].setData(recordFloats());
        cursor["bake"]["opacity"].setData(uint32_t{3});
        cursor["bake"]["dc0"].setData(uint32_t{11});
        cursor["bake"]["restBase"].setData(uint32_t{23});
        cursor["bake"]["indirect"].setData(indirect ? 1u : 0u);
        cursor["bake"]["cells"].setData(side == 0 ? 0u : words);
    });
    ATHENEA_TRY(batch.submit(true));
    // What a USD array holds, as bytes.
    direct.resize(size_t{count_} * 9);
    shadowBits.resize(size_t{count_} * words);
    bounced.resize(indirect ? size_t{count_} * 27 : 0);
    ATHENEA_TRY(directs->read(device, 0, direct.size() * sizeof(float), direct.data()));
    ATHENEA_TRY(bits->read(device, 0, shadowBits.size() * sizeof(int32_t), shadowBits.data()));
    if (indirect) {
        ATHENEA_TRY(bounceds->read(device, 0, bounced.size() * sizeof(float), bounced.data()));
    }
    uint32_t reached = 0;
    ATHENEA_TRY(found->read(device, 0, sizeof(reached), &reached));
    std::printf("mesh2splat: transfer baked for %u of %u gaussians (%u paths each, %u bounces%s) in %.0f ms\n",
                reached, count_, samples, bounces, indirect ? ", with the indirect half" : "", traced);
    if (reached * 2 < count_) {
        std::fprintf(stderr,
                     "mesh2splat: more than half the gaussians found no surface under them; the "
                     "transfer is unlikely to be what you want\n");
    }
    return ok();
}

}   // namespace

void addMesh2Splat(CLI::App& app) {
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand(
        "mesh2splat", "convert a USD stage of meshes into gaussian splats (Electronic Arts' mesh2splat)");
    cmd->add_option("stage", o->stage, ".usd / .usda / .usdc holding meshes")->required();
    cmd->add_option("-o,--output", o->output,
                    "the ParticleField stage to write (.usda, .usdc, .usd), or a .athc with levels of detail");
    cmd->add_option("--prim", o->prim, "only meshes at or under this prim path");
    cmd->add_option("--hide", o->hidden, "a prim to leave out with all beneath it, as if invisible (repeatable)");
    cmd->add_option("--lod-levels", o->lodLevels,
                    "levels of detail: the conversion again at half the resolution each time, each level a "
                    "stage beside the output and the output one that draws them as one cloud (1: none)");
    cmd->add_option("--resolution", o->resolution,
                    "cells across the longest side of the box the density is measured over: the whole "
                    "model's, or each mesh's own with --density per-mesh");
    cmd->add_option("--density", o->density,
                    "per-model: one grid over the converted set, so a small part beside a large one is "
                    "sparse; per-mesh: each mesh's own box, the cell held between --cell-min and --cell-max");
    cmd->add_option("--cell-min", o->cellMin,
                    "no cell finer than this many world units (per-mesh: 0 is an eighth of the model's "
                    "cell; per-model: 0 is no bound)");
    cmd->add_option("--cell-max", o->cellMax,
                    "no cell coarser than this many world units (per-mesh: 0 is the model's cell, so "
                    "nothing is coarser than per-model; per-model: 0 is no bound). The same number "
                    "for both is one cell for the whole stage");
    cmd->add_option("--max-splats", o->maxSplats,
                    "the budget, over the whole stage, shared in proportion to what each mesh wants");
    cmd->add_option("--cell-from-camera", o->cellFromCamera,
                    "a camera prim: each mesh's cell is what one pixel of it covers where the mesh is "
                    "nearest to it, bounded by --cell-min and --cell-max (replaces --density)");
    cmd->add_option("--camera-pixels", o->cameraPixels, "--cell-from-camera: pixels across the camera's image")
        ->check(CLI::Range(1u, 65536u));
    cmd->add_option("--sigma", o->sigma,
                    "how wide a gaussian is against its cell; mesh2splat's own number is 0.65, which "
                    "leaves a traced surface 30% transparent");
    cmd->add_option("--flatness", o->flatness,
                    "the third size, as a fraction of the smaller of the other two");
    cmd->add_option("--opacity", o->opacity,
                    "how much of what stands behind it the converted surface covers, multiplied into "
                    "the material's own opacity");
    cmd->add_option("--glass-opacity", o->minOpacity,
                    "how much of what stands behind it a fully transmitting solid still covers. Low is "
                    "a window -- you see what stands behind it -- and translucency is not that: the "
                    "light comes through scattered, so the body stays mostly there");
    cmd->add_option("--opacity-cut", o->opacityCut,
                    "a material whose opacity is a map with no threshold of its own: below this the "
                    "surface is not there and no gaussian is written, so the budget goes where the "
                    "surface is; above it the surface covers what the map reads. It is what makes a "
                    "feather a feather and not the card it is drawn on");
    cmd->add_option("--max-cells", o->maxCells, "most cells one triangle may walk");
    cmd->add_option("--texture-size", o->textureSize,
                    "read maps no larger than this (0: their own size). A map is a float4 picture "
                    "on the device, so a 4k one is 268 MB and a stage with a few does not fit");
    cmd->add_flag("--no-textures", o->noTextures, "ignore the maps; materials keep their constant values");
    cmd->add_flag("--normal-map-turns", o->normalMapTurns,
                  "orient each gaussian by the normal map rather than the surface");
    cmd->add_flag("--no-displacement", o->noDisplacement,
                  "ignore the materials' displacement: every gaussian stands on the flat mesh");
    cmd->add_option("--displace-refine", o->displaceRefine,
                    "where the relief stretches a cell, split it into at most this many gaussians "
                    "along each axis")
        ->check(CLI::Range(1u, 64u));
    cmd->add_option("--simplify", o->simplify,
                    "where the surface is the same across a block of cells -- colour, metallic, roughness, "
                    "cut-out and normal each moving by no more than this -- make the block one gaussian of "
                    "its size (0: a gaussian a cell)")
        ->check(CLI::Range(0.0, 1.0));
    cmd->add_option("--simplify-levels", o->simplifyLevels,
                    "the largest block --simplify may merge is 2^this cells a side")
        ->check(CLI::Range(1u, 5u));
    cmd->add_flag("!--no-camera", o->addCamera, "do not add a camera framing the cloud");
    cmd->add_flag("!--no-bake", o->bake,
                  "do not bake: carry the material instead and let the scene's lights relight the "
                  "cloud every frame. Cheaper to convert, and the cloud can then be put under other "
                  "light; what it loses is the bounce, the shadows and the exactness");
    cmd->add_flag("--transfer", o->transfer,
                  "bake how much of an environment reaches each gaussian instead of the light itself, "
                  "so the cloud can be lit by any sky (excludes the radiance bake)");
    cmd->add_option("--transfer-cells", o->transferCells,
                    "--transfer: cells a side of the grid of open directions over the whole sphere, 16 or 32; "
                    "0 keeps the first transfer's 8 x 8 over the half a gaussian faces")
        ->check(CLI::IsMember({0u, 16u, 32u}));
    cmd->add_flag("!--no-indirect", o->indirect,
                  "--transfer: leave out the interreflection, which costs no bake time and 27 floats a "
                  "gaussian to keep");
    cmd->add_option("--bake-samples", o->bakeSamples, "paths a gaussian the bake traces");
    cmd->add_option("--bake-bounces", o->bakeBounces, "bounces after the first hit, in the bake");
    cmd->add_option("--bake-extra", o->bakeExtra,
                    "adaptive: paths a gaussian on average added after the first pass, shared out by "
                    "sqrt(relative variance / cost)");
    cmd->add_option("--bake-pass-samples", o->bakePassSamples, "paths a gaussian in each adaptive pass")
        ->check(CLI::Range(1u, 4096u));
    cmd->add_option("--bake-filter", o->bakeFilter,
                    "a-trous passes of the splat bake filter over the bake's indirect light (0: none); "
                    "the direct light, which holds the shadows, is never filtered")
        ->check(CLI::Range(0u, 8u));
    cmd->add_option("--bake-filter-luminance", o->bakeFilterLuminance,
                    "the filter's edge: a neighbour whose indirect light differs by this many standard "
                    "deviations counts e^-1 as much");
    cmd->add_flag("--bake-filter-indirect-only", o->bakeFilterIndirectOnly,
                  "filter the indirect light alone and leave the direct as traced (by default the whole "
                  "light is filtered, weighed by its own noise, which keeps a shadow's edge)");
    cmd->add_option("--bake-degree", o->bakeDegree,
                    "harmonics the bake fits, 0 to 3: 0 is one colour a gaussian and cannot hold a "
                    "reflection, and each degree costs a pass over the paths");
    cmd->add_flag("--skinned", o->skinned,
                  "carry the skeleton: the gaussians are built in the bind pose and each keeps the "
                  "four joints that move it, so the cloud deforms with the rig instead of being one "
                  "pose. Forces --no-bake: a baked radiance does not turn with a limb");
    cmd->add_option("--range", o->range,
                    "START:END[:STEP] in time codes: the instants a skinned cloud keeps its "
                    "skeleton's transforms at. The stage's own range by default, a code a step");
    cmd->add_flag("--default-lights", o->defaultLights,
                  "bake under a dome and a sun in the session layer, for a stage that brings no "
                  "lights of its own (what athenea view offers)");
    cmd->add_option("--time", o->time,
                    "the USD time code the stage is read at: the pose that becomes gaussians, "
                    "and the instant the bake traces. A skinned stage is posed for it");
    cmd->add_option("--path", o->paths, "extra AOFX bundle directories");
    cmd->callback([o] {
        if (o->density != "per-model" && o->density != "per-mesh") {
            std::fprintf(stderr, "--density wants per-model or per-mesh, not '%s'\n", o->density.c_str());
            throw CLI::RuntimeError(1);
        }
        // WHAT A .ATHC CAN CARRY: the gaussians -- position, opacity, sizes,
        // rotation, harmonics -- and the shading normal. It has no room for
        // a rig, a transfer, the material a relit cloud reflects with, the
        // matte's ids, a glass's index or the stage's up axis and unit, so a
        // conversion that needs one of the first two is refused rather than
        // written without it, and the rest is said.
        if (lod::isAthc(o->output)) {
            if (o->lodLevels > 1) {
                std::fprintf(stderr, "a .athc builds its own levels of detail: drop --lod-levels\n");
                throw CLI::RuntimeError(1);
            }
            if (o->skinned || o->transfer) {
                std::fprintf(stderr, "a .athc cannot carry %s: write a USD stage (.usda, .usdc, .usd)\n",
                             o->skinned ? "a skeleton (--skinned)" : "a transfer (--transfer)");
                throw CLI::RuntimeError(1);
            }
            std::printf("mesh2splat: a .athc keeps the gaussians and their shading normals; the metallic, "
                        "roughness and transmission a relit cloud reflects with, the Cryptomatte ids, the glass "
                        "index and the stage's up axis and unit stay out\n");
        }
        gpu_host::Context* context = gpu_host::installProcessContext();
        if (context == nullptr || context->compute() == nullptr) {
            std::fprintf(stderr, "no GPU compute device for AOFX kernels (gpe has no backend here)\n");
            throw CLI::RuntimeError(1);
        }
        gpu::ShaderLibrary library(context->deviceShared());

        aofx_host::EffectRegistry registry;
        for (const std::string& path : o->paths) {
            registry.addSearchPath(path);
        }
#ifdef ATHENEA_AOFX_BUNDLE_DIR
        registry.addSearchPath(ATHENEA_AOFX_BUNDLE_DIR);
#endif
        registry.scan(context);
        aofx::Effect* effect = registry.find("rt.sparrow.aofx.mesh2splat");
        if (effect == nullptr) {
            std::fprintf(stderr, "no Mesh2Splat bundle on the AOFX search path (try `athenea aofx list`)\n");
            throw CLI::RuntimeError(1);
        }
        // The filter a bake's indirect light goes through, where it is asked
        // for: missing, it is a conversion that cannot be what was asked.
        aofx::Effect* filter = nullptr;
        if (o->bake && !o->transfer && o->bakeFilter > 0) {
            filter = registry.find("rt.sparrow.aofx.splatbakefilter");
            if (filter == nullptr) {
                std::fprintf(stderr, "no SplatBakeFilter bundle on the AOFX search path (try `athenea aofx "
                                     "list`), and --bake-filter asks for it\n");
                throw CLI::RuntimeError(1);
            }
        }

        // Everything that touches the device happens on the host's own GPU
        // thread, because that is the thread the AOFX host renders on and a
        // device with two callers is the one bug this whole interface exists
        // to prevent. `run` is re-entrant from that thread, so `renderEffect`
        // asking for it again inside this costs nothing.
        // LEVELS OF DETAIL: the same conversion at the resolution asked for,
        // then at half of it, and so on, each level a stage of its own beside
        // the output, and the output a stage that draws them as one cloud
        // (usd::writeLodAssembly). A cloud a skeleton carries cannot be
        // merged into coarser cells -- a cell that took wing and body would
        // not know which to move with -- but it can be converted again.
        const uint32_t lodLevels = std::max(o->lodLevels, 1u);
        const std::string assemblyPath = o->output;
        const uint32_t baseResolution = o->resolution;
        std::vector<usd::LodLevelFile> levelFiles;
        for (uint32_t level = 0; level < lodLevels; ++level) {
            if (lodLevels > 1) {
                const std::filesystem::path out(assemblyPath);
                o->output = (out.parent_path() / (out.stem().string() + "_lod" + std::to_string(level) + ".usdc"))
                                .string();
                o->resolution = std::max(baseResolution >> level, 1u);
            }
            double levelCell = 0.0;
            uint32_t count = 0;
            Result<void> inside = ok();
            const auto work = [&]() -> Result<void> {
                auto builder = geom::MeshBuilder::create(library);
                if (!builder) return std::move(builder).error();
                auto stage = usd::MeshStage::open(o->stage);
                if (!stage) return std::move(stage).error();
                usd::MeshStageOptions read;
                read.prim = o->prim;
            // THE SAME INSTANT FOR BOTH. The gaussians come from the mesh at this
            // time and the bake traces the scene at this time, so the rays stand
            // on the surface they were built from. They did not: the conversion
            // read the stage at its default time whatever `--time` said, and a
            // bake at any other instant put its rays where the mesh used to be.
            read.time = o->time;
            read.skinned = o->skinned;
            read.hidden = o->hidden;
            if (o->transfer) {
                // The two are different answers to the same question and the file
                // has room for one: a transfer keeps the geometry, a radiance
                // bake keeps one sky's light.
                o->bake = false;
            }
            if (o->skinned && o->transfer) {
                // A transfer moves with the limb no better than a baked radiance
                // does: what it holds is the visibility of a pose.
                std::printf("mesh2splat: --skinned carries the material, not a transfer\n");
                o->transfer = false;
            }
            if (o->skinned && o->bake) {
                // A BAKED RADIANCE DOES NOT TURN WITH A LIMB. What the harmonics
                // hold is the environment and the bounce -- the ground under a
                // paw is in them -- and carrying that up with the leg is the
                // mistake of rotating a lightmap. A cloud a skeleton moves is
                // relit every frame instead, which is right by construction.
                std::printf("mesh2splat: --skinned carries the material, not a bake\n");
                o->bake = false;
            }
                auto meshes = stage->read(*builder, read);
                if (!meshes) return std::move(meshes).error();

                Converter converter(*context, library, *o);
                ATHENEA_TRY(converter.prepare());
                ATHENEA_TRY(converter.packMeshes(*meshes));
                if (!o->cellFromCamera.empty()) {
                    auto camera = (*stage).camera(o->cellFromCamera, o->time);
                    if (!camera) return std::move(camera).error();
                    converter.setCamera(*camera, o->cameraPixels);
                    std::printf("mesh2splat: the cell is a pixel of %s, %u across\n", o->cellFromCamera.c_str(),
                                o->cameraPixels);
                }
                ATHENEA_TRY(converter.loadTextures());
                auto raw = converter.convert(*effect, *meshes);
                if (!raw) return std::move(raw).error();
                count = raw->count;
                levelCell = converter.modelCell();

                // THE LIGHT THE MESH HAD, baked into the gaussians.
                //
                // A relit cloud carries the material and is lit again every
                // frame, which is an approximation: one sample a light, no
                // bounce, and a normal a splat never had. A bake asks the path
                // tracer instead -- the same stage, the same lights, the same
                // integrator -- and stores what it answers. What that costs is
                // the light: a baked cloud carries this scene's, and cannot be
                // put under another.
                std::vector<float> transferDirect;
                std::vector<float> transferIndirect;
                std::vector<int32_t> shadowBits;
                if (o->transfer) {
                    // WHAT AN ENVIRONMENT PUTS ON EACH GAUSSIAN, rather than what
                    // this one did. The colours stay the material's albedo and
                    // the frame lights them with whatever sky it has, so the same
                    // file is right under every HDRI rather than under one.
                    ATHENEA_TRY(converter.transfer(o->stage, o->time, o->bakeSamples + o->bakeExtra, o->bakeBounces, o->indirect,
                                                   o->transferCells, transferDirect, transferIndirect,
                                                   shadowBits));
                } else if (o->bake) {
                    usd::BakeOptions bake;
                    bake.samples = o->bakeSamples;
                    bake.bounces = o->bakeBounces;
                    bake.degree = std::min(o->bakeDegree, 3u);
                    bake.extraSamples = o->bakeExtra;
                    bake.passSamples = o->bakePassSamples;
                    ATHENEA_TRY(converter.bake(o->stage, o->time, bake, o->defaultLights, filter));
                }

                // THE RIG, WHEN THE CLOUD KEEPS ONE. Four joints a gaussian came
                // back with the records; what is gathered here is the joints'
                // own transforms at each instant of the range, which is the only
                // thing about an animated cloud that changes from frame to frame.
                usd::SplatSkinning rig;
                auto carriedBy = o->skinned ? converter.influences() : Result<std::vector<float>>(std::vector<float>{});
                if (!carriedBy) return std::move(carriedBy).error();
                if (o->skinned && !carriedBy->empty()) {
                    for (const usd::StageMesh& one : *meshes) {
                        if (one.skinning.bound) {
                            rig.skeleton = one.skinning.skeleton;
                            rig.jointNames = one.skinning.joints;
                            rig.geomBindTransform = one.skinning.geomBindTransform;
                            rig.joints = static_cast<uint32_t>(one.skinning.joints.size());
                            break;
                        }
                    }
                    rig.influences = std::move(*carriedBy);
                    auto slopes = converter.weightGradients();
                    if (!slopes) return std::move(slopes).error();
                    rig.weightGradients = std::move(*slopes);
                    const auto [begin, end] = (*stage).timeRange();
                    double from = begin;
                    double to = end;
                    double step = 1.0;
                    if (!o->range.empty()) {
                        if (std::sscanf(o->range.c_str(), "%lf:%lf:%lf", &from, &to, &step) < 2) {
                            return Error::make(ErrorCode::InvalidArgument,
                                               "'{}': --range wants START:END[:STEP]", o->range);
                        }
                    }
                    if (!(step > 0.0)) step = 1.0;
                    if (to < from) to = from;
                    for (double at = from; at <= to + 1e-9; at += step) {
                        rig.times.push_back(at);
                    }
                    auto moved = (*stage).skeletonTransforms(rig.skeleton, rig.times);
                    if (!moved) return std::move(moved).error();
                    rig.xforms = std::move(*moved);
                    rig.timeCodesPerSecond = (*stage).timeCodesPerSecond();
                    std::printf("mesh2splat: carried by %s, %u joints over %zu instants at %g fps\n",
                                rig.skeleton.c_str(), rig.joints, rig.times.size(),
                                rig.timeCodesPerSecond);
                }

                // A .ATHC: the cloud with its levels of detail, straight from
                // the device -- decoded into a cloud there (`CloudLoader`),
                // built into levels there (`LodBuilder`), and written as the
                // bytes they are. What it keeps is what a .athc has room
                // for: positions, shape, harmonics and shading normals.
                if (lod::isAthc(o->output)) {
                    auto loader = scene::CloudLoader::create(library);
                    if (!loader) return std::move(loader).error();
                    auto splats = loader->upload(raw->records, raw->count, raw->encoding, raw->source,
                                                 o->bake ? std::min(o->bakeDegree, 3u) : 0u);
                    if (!splats) return std::move(splats).error();
                    auto builder = lod::LodBuilder::create(library);
                    if (!builder) return std::move(builder).error();
                    auto built = builder->build(*splats);
                    if (!built) return std::move(built).error();
                    return platform::writeAtomically(o->output, [&](const std::filesystem::path& partial) {
                        return lod::writeAthc(library.device(), *built, partial);
                    });
                }

                usd::ExportOptions options;
                options.skinning = rig.valid() ? &rig : nullptr;
                options.maxDegree = o->bake ? std::min(o->bakeDegree, 3u) : 0;
                options.addCamera = o->addCamera;
                options.upAxis = (*stage).upAxis();
                options.metersPerUnit = (*stage).metersPerUnit();
                // Both baked and not, the cloud is relit -- what differs is what
                // its colours are. Baked, they are the light on the material's
                // body and the frame adds the polish; not baked, they are an
                // albedo and the frame lights them whole.
                // A cloud that carries harmonics carries the light whole, and a
                // frame adds nothing to it. One baked to a single colour has no
                // room for a reflection, so it keeps the material's body and the
                // frame puts the polish back. Not baked at all, the colours are an
                // albedo and the frame lights them.
                // What the cloud holds and what the frame adds. Baked, the cloud
                // carries the light on the material's body -- its harmonics say
                // how that light changes with the direction -- and the frame puts
                // the polish back, which is the one thing neither a colour nor
                // sixteen coefficients can hold: a reflection off a surface of
                // roughness 0.1 is far sharper than that. Not baked, the colours
                // are an albedo and the frame lights them whole.
                options.relight = true;
                // A transfer keeps the material's albedo and the geometry, so the
                // frame lights it whole; a radiance bake keeps the light on the
                // body and the frame adds only the polish.
                options.litBody = o->bake && !o->transfer;
                // Light, all of it: the albedo, the transfer's and the bake's.
                options.linear = true;
                // The matte's ancestry, gaussian by gaussian, as the conversion
                // inherited it from the prims it read.
                options.cryptoObject = converter.cryptoIds();
                options.cryptoManifest = converter.cryptoManifest();
                options.thinWalled = converter.thinWalled();
                options.ior = converter.glassIor();
                options.transferDirect = transferDirect;
                options.transferIndirect = transferIndirect;
                options.shadowBits = shadowBits;
                options.shadowWords = technique::transferCellWords(o->transferCells);
                // WHOLE OR NOT AT ALL: under another name beside it, and
                // under its own only once it is complete, so a conversion
                // that fails leaves no stage of half a cloud behind.
                return platform::writeAtomically(o->output, [&](const std::filesystem::path& partial) {
                    return usd::writeParticleFieldStage(library, *raw, partial, options);
                });
            };
            auto ran = context->run([&] { inside = work(); });
            if (!ran) {
                cli::fail(ran.error());
            }
            if (!inside) {
                cli::fail(inside.error());
            }
            std::printf("mesh2splat: wrote %s (%u splats)\n", o->output.c_str(), count);
            levelFiles.push_back({o->output, levelCell});
        }
        if (lodLevels > 1) {
            o->output = assemblyPath;
            o->resolution = baseResolution;
            if (auto made = platform::writeAtomically(assemblyPath,
                                                      [&](const std::filesystem::path& partial) {
                                                          return usd::writeLodAssembly(
                                                              partial, levelFiles,
                                                              std::filesystem::path(assemblyPath).stem().string());
                                                      });
                !made) {
                cli::fail(made.error());
            }
            std::printf("mesh2splat: wrote %s, %u levels of detail\n", assemblyPath.c_str(), lodLevels);
        }
    });
}

}   // namespace athenea::cli
