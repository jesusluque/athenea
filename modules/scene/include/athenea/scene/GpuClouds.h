// Copyright (c) 2026 jesus luque.
//
// Splat and point clouds as they live on the device, and how they get there.
//
// The layout is in shaders/athenea/common/packing.slang. Getting there is:
//
//   CPU  read the file, arrange float records (io::RawSplats / RawPoints)
//   GPU  validate -> prefix sum -> decode + compact -> bounds
//
// in slices of at most kSliceBytes of raw records, so a cloud larger than a
// single device buffer may be still loads. The only numbers read back are one
// count per slice and the six numbers of the bounding box.
#pragma once

#include <array>
#include <optional>
#include <cstddef>
#include <span>
#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/algo/PrefixSum.h"
#include "athenea/io/RawSplats.h"
#include "athenea/io/Sog.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::scene {

struct Bounds {
    std::array<float, 3> min{0, 0, 0};
    std::array<float, 3> max{0, 0, 0};
};

struct GpuSplats {
    std::string source;
    uint32_t    count = 0;          ///< splats kept
    uint32_t    declared = 0;       ///< records in the file
    uint32_t    restPerColour = 0;  ///< 0, 3, 8, 15 -> degree 0..3
    uint32_t    shWords = 0;
    gpu::Buffer positions;          ///< float4
    gpu::Buffer shape;              ///< uint * 4
    gpu::Buffer sh;                 ///< uint * shWords (one dummy word at degree 0)
    /// What each splat reflects with, where the cloud carries it: one uint a
    /// splat, metallic in the low byte and roughness in the next. Empty for a
    /// capture, which has neither -- `hasPbr` is how a kernel asks.
    gpu::Buffer pbr;
    /// WHICH PRIM EACH GAUSSIAN CAME FROM, one Cryptomatte id a splat, where
    /// the cloud was converted from meshes (`athenea mesh2splat` writes it) and so
    /// knows. Empty for a capture, which has no ancestry to inherit: a cloud
    /// without it is left out of the matte rather than named wrongly.
    gpu::Buffer crypto;
    /// HOW MUCH OF AN ENVIRONMENT REACHES EACH GAUSSIAN: `transferWords` uint
    /// a gaussian, f16 pairs, the nine scalars of the direct half first and
    /// then, where the cloud carries it, nine rgb triples of the indirect
    /// one. Or ten values, a ZONAL transfer (`isZonal`): two lobes, each an
    /// axis in the gaussian's own frame (the octahedral square's u, v) and
    /// three zonal coefficients, which turn with the gaussian -- what a cloud
    /// a skeleton carries keeps. Empty for a cloud baked the old way, whose
    /// colours are the light of the dome it was baked under.
    gpu::Buffer transfer;
    /// WHICH WAYS OUT ARE OPEN, two words a gaussian: sixty-four bits of an
    /// 8 x 8 octahedral grid over the sphere, set where a ray traced at the
    /// bake found nothing. What lets a sun cast a hard shadow. Empty for a
    /// cloud that was not baked with a transfer. Beside a zonal transfer the
    /// grid is over the gaussian's own frame rather than the world.
    gpu::Buffer shadowBits;
    /// WHICH RECORD EACH SPLAT CAME FROM, one uint a splat: validation drops
    /// what cannot be drawn, so anything else the file keeps a gaussian --
    /// a rig's influences, a visibility's parts -- is read through this
    /// (`CloudLoader::keptOnly`).
    gpu::Buffer origin;
    /// THE SHADING NORMAL, apart from the frame: one uint a splat, octahedral
    /// 2 x 16 bits (packing.slang's `packNormal`), in the cloud's own space.
    /// What a relit gaussian is lit with where the cloud carries it -- a
    /// conversion from a mesh does, so the relief its normal map drew is not
    /// lost to the disc's short axis -- while the frame keeps answering
    /// everything geometric (the footprint, a ray's hit). Empty for a capture;
    /// `hasNormals` is how a kernel asks. Whatever turns the frame (the
    /// skinner) turns this with it.
    gpu::Buffer normals;
    /// THE LIGHT THE SURFACE GAVE OFF BY ITSELF: one uint a splat, linear
    /// radiance in RGB9E5 (packing.slang's `packRgb9e5`). A relit gaussian
    /// adds it, unshadowed, to what it reflects -- a converted lamp shade, a
    /// screen -- and a cloud whose body was baked does not, since the bake
    /// already holds it. Empty for a capture and for a conversion of
    /// materials that give off nothing; `hasEmission` is how a kernel asks.
    /// Nothing turns it: it has no direction.
    gpu::Buffer emission;
    /// WHAT THE MATERIAL LAYERED OVER ITS BASE: three uint a splat, the
    /// specular's weight, colour and index, the coat's weight, roughness and
    /// index, the sheen's colour and roughness, a byte each (packing.slang's
    /// `packLobes`). What a relit gaussian reflects with on top of `pbr`'s
    /// base -- a car's lacquer over its metal flake, velvet's sheen. Empty
    /// for a capture and for a conversion whose materials name none of it,
    /// which reflects with the plain specular (`plainLobes`); `hasLobes` is
    /// how a kernel asks. Nothing turns it.
    gpu::Buffer lobes;
    /// Words a gaussian of `shadowBits`: 2 (8 x 8 cells over the sphere),
    /// or 8 (16 x 16, a TX transfer's).
    uint32_t    shadowWords = 0;
    uint32_t    transferWords = 0;
    /// Values a gaussian: 0, 10 (zonal lobes), or 9, 36, 84, 16, 64 or 112 --
    /// the count is the layout (athenea/common/transfer_layout.slang).
    uint32_t    transferCount = 0;
    /// THE SPACE ITS COLOURS ARE IN (`io::RawSplats::linear`): false for a
    /// capture, whose harmonics are sRGB and are decoded a splat at a time
    /// when they are evaluated; true for a cloud that holds light already.
    /// Either way the blend is in linear light. A property of the data, so
    /// whatever copies a cloud (the levels of detail, the cut, a pose, a
    /// decimation, `.athc`) copies it.
    bool        linear = false;
    /// WHAT THIS CLOUD CASTS ON THE SPACE AROUND IT, baked by part (a part is
    /// what one joint carries) and read as a product over parts. Empty for a
    /// cloud nothing baked; `hasVisibility` is how a kernel asks. The layout
    /// is splat_visibility.slang's: `visibilityParts` holds one
    /// `VisibilityPart` a part, `visibilityTexels` two f16 a word, and
    /// `visibilityPartOf` the part each gaussian belongs to.
    gpu::Buffer visibilityParts;
    gpu::Buffer visibilityTexels;
    gpu::Buffer visibilityPartOf;
    gpu::Buffer visibilityAmbient;   ///< a probe's mean over its directions, for domes
    uint32_t    visibilityPartCount = 0;
    Bounds      bounds;
    /// THE BOX IT WAS BOUND IN, where `positions` are a pose of another
    /// cloud's (a skinned cloud's posed copy): what does not breathe with
    /// the wings, which is what a shadow map is sized from. Empty for a cloud
    /// that is its own rest pose.
    std::optional<Bounds> restBounds;
    /// Counted up by whatever rewrites `positions` or `shape` in place -- the
    /// skinner, a frame -- so a structure built over them (the ray tracer's
    /// proxies) can tell a cloud posed anew from the one it built for. The
    /// buffers keep their handles across a pose, which is why a handle alone
    /// answered "the same cloud" to a bird that had flown off its bind pose.
    uint32_t    revision = 0;

    [[nodiscard]] bool hasPbr() const noexcept { return pbr.valid(); }
    [[nodiscard]] bool hasCrypto() const noexcept { return crypto.valid(); }
    [[nodiscard]] bool hasNormals() const noexcept { return normals.valid(); }
    [[nodiscard]] bool hasEmission() const noexcept { return emission.valid(); }
    [[nodiscard]] bool hasLobes() const noexcept { return lobes.valid(); }
    [[nodiscard]] bool hasTransfer() const noexcept { return transfer.valid() && transferCount >= 9; }
    /// Whether it carries which ways out are open (`shadowBits`).
    [[nodiscard]] bool hasShadowBits() const noexcept { return shadowBits.valid(); }
    /// Whether the indirect half is there as well as the direct one.
    [[nodiscard]] bool hasIndirect() const noexcept {
        return transferCount == 36 || transferCount == 84 || transferCount == 64 || transferCount == 112;
    }
    /// Whether the transfer is two zonal lobes in each gaussian's frame.
    [[nodiscard]] bool isZonal() const noexcept { return transferCount == kTransferZonalCount; }
    /// Values a gaussian of a zonal transfer (splat_relight.slang's
    /// `kTransferZonalCount`).
    static constexpr uint32_t kTransferZonalCount = 10;
    [[nodiscard]] bool hasVisibility() const noexcept {
        return visibilityPartCount > 0 && visibilityParts.valid() && visibilityTexels.valid() &&
               visibilityPartOf.valid();
    }

    [[nodiscard]] uint32_t degree() const noexcept {
        return restPerColour == 15 ? 3 : restPerColour == 8 ? 2 : restPerColour == 3 ? 1 : 0;
    }
};

struct GpuPoints {
    std::string source;
    uint32_t    count = 0;
    uint32_t    declared = 0;
    gpu::Buffer positions;   ///< float4 xyz, 1
    gpu::Buffer colours;     ///< uint * 2: f16 r|g, b|a (linear)
    Bounds      bounds;
};

/// A float array as it sits in memory -- float32 values, or float16 when
/// `half` -- uploaded as bytes and read on the device.
struct FloatStream {
    std::span<const std::byte> bytes;
    bool                       half = false;
    bool                       isDouble = false;   ///< float64 (matrix4d, point3d); `half` then false

    [[nodiscard]] bool   empty() const noexcept { return bytes.empty(); }
    [[nodiscard]] size_t values() const noexcept { return bytes.size() / (isDouble ? 8 : half ? 2 : 4); }
    /// The kind kernels read: 0 none, 1 float, 2 half, 3 double.
    [[nodiscard]] uint32_t kind() const noexcept { return empty() ? 0 : isDouble ? 3 : half ? 2 : 1; }
};

/// A splat cloud as separate arrays, the way USD's ParticleField stores one.
/// Empty streams take defaults: identity rotation, unit scale, opacity 1, DC 0.
struct SplatStreams {
    std::string source;
    uint32_t    count = 0;
    FloatStream positions;      ///< xyz
    FloatStream rotations;      ///< xyzw (GfQuat's layout)
    FloatStream scales;         ///< xyz, linear
    FloatStream opacities;      ///< linear
    uint32_t    coefficients = 0;   ///< SH coefficients per splat, DC first: (degree + 1)^2
    FloatStream sh;             ///< rgb per coefficient
    /// What the gaussian reflects with, one per splat, where the stage says
    /// so (`primvars:athenea:splat:metallic` and `:roughness`). Empty otherwise.
    FloatStream metallic;
    FloatStream roughness;
    FloatStream transmission;
    /// One Cryptomatte id a splat (`primvars:athenea:splat:cryptoObject`, int32),
    /// read as the bits they are and not as numbers to do arithmetic on.
    FloatStream cryptoObject;
    /// Nine floats a splat, and twenty-seven more where the cloud carries the
    /// indirect half (`primvars:athenea:splat:transferDirect` and
    /// `:transferIndirect`).
    FloatStream transferDirect;
    FloatStream transferIndirect;
    /// A TX transfer's reflected field (`primvars:athenea:splat:transferReflected`):
    /// 48 floats a splat, read only beside the indirect half. Sixteen and
    /// forty-eight in the two before it at degree 3; the counts say which
    /// (athenea/common/transfer_layout.slang).
    FloatStream transferReflected;
    /// Ten floats a splat, a zonal transfer (`primvars:athenea:splat:transferZonal`):
    /// two lobes, each an axis in the gaussian's frame and three coefficients.
    /// Where it is there it is what the cloud carries, in place of the nine
    /// harmonics.
    FloatStream transferZonal;
    /// Two int32 a splat (`primvars:athenea:splat:shadowBits`), read as bits;
    /// or eight, the 256 cells of a TX transfer, which the count says.
    FloatStream shadowBits;
    /// One int32 a splat, nonzero for a thin-walled glass
    /// (`primvars:athenea:splat:thinWalled`); read only beside the PBR arrays.
    FloatStream thinWalled;
    /// One int a splat, nonzero where its metal is a Schlick rather than a
    /// conductor (`primvars:athenea:splat:schlickMetal`); beside the PBR arrays.
    FloatStream schlickMetal;
    /// Three floats a splat, the shading normal (`primvars:athenea:splat:normal`).
    FloatStream normals;
    /// `primvars:athenea:splat:linear`: the colours are linear light
    /// (`GpuSplats::linear`). A field that does not say is a capture, sRGB.
    bool        linear = false;
    /// Three floats a splat, the radiance it gives off
    /// (`primvars:athenea:splat:emission`).
    FloatStream emission;
    /// The layers over the base, one array each, one value (or three, for a
    /// colour) a splat: `primvars:athenea:splat:specularWeight`,
    /// `:specularColor`, `:specularIor`, `:coatWeight`, `:coatRoughness`,
    /// `:coatIor`, `:sheenColor`, `:sheenRoughness`, `:coatDarkening`. The cloud carries the
    /// lobes where any of them is there; one that is missing takes its
    /// default (packing.slang's `plainLobes`).
    FloatStream specularWeight;
    FloatStream specularColour;
    FloatStream specularIor;
    FloatStream coatWeight;
    FloatStream coatRoughness;
    FloatStream coatIor;
    FloatStream sheenColour;
    FloatStream sheenRoughness;
    FloatStream coatDarkening;   ///< `:coatDarkening`, 0 where it is missing
    /// BLENDER'S LAYOUT. Four floats a splat, the DC coefficient's rgb and the
    /// opacity (a Gaussian-splat PointCloud's `radiance:base`); where present
    /// it stands for `opacities` and for the DC of `sh`.
    FloatStream radianceBase;
    /// One array a basis function after DC, rgb a splat (`radiance:sh_0`
    /// onwards), in place of `sh`: `coefficients` is then their count plus
    /// one. All float, or all half. Laid out on the device as `sh` is.
    std::vector<FloatStream> shPlanes;
};

/// A point cloud as separate arrays, the way UsdGeomPoints stores one.
struct PointStreams {
    std::string source;
    uint32_t    count = 0;
    FloatStream positions;   ///< xyz
    FloatStream colours;     ///< rgb, linear: one for every point, one per point, or empty (white)
};

class CloudLoader {
public:
    static constexpr uint64_t kSliceBytes = uint64_t{256} << 20;

    [[nodiscard]] static Result<CloudLoader> create(gpu::ShaderLibrary& library);

    /// `maxDegree` caps the harmonics kept (0..3).
    [[nodiscard]] Result<GpuSplats> upload(const io::RawSplats& raw, uint32_t maxDegree = 3);
    /// The same decode of records that are already on the device: `count`
    /// of them in `records`, laid out as `encoding` says. Nothing crosses to
    /// the processor (`athenea mesh2splat -o x.athc`).
    [[nodiscard]] Result<GpuSplats> upload(const gpu::Buffer& records, uint32_t count,
                                           const io::SplatEncoding& encoding, const std::string& source,
                                           uint32_t maxDegree = 3);
    /// A SOG's images, decoded on the device into records that then take the
    /// same validate and decode as any other format -- nothing crosses back.
    [[nodiscard]] Result<GpuSplats> upload(const io::RawSog& sog, uint32_t maxDegree = 3);
    /// The same decode, read back as float records, for what consumes records
    /// on the host side (the USD export).
    [[nodiscard]] Result<io::RawSplats> records(const io::RawSog& sog, uint32_t maxDegree = 3);
    /// `detail` keeps that fraction of the points, the same ones every time.
    [[nodiscard]] Result<GpuPoints> upload(const io::RawPoints& raw, float detail = 1.0F);
    /// Arrays uploaded as they are and interleaved into records on the device.
    [[nodiscard]] Result<GpuSplats> upload(const SplatStreams& streams, uint32_t maxDegree = 3);
    [[nodiscard]] Result<GpuPoints> upload(const PointStreams& streams, float detail = 1.0F);

    /// `byRecord`, `words` 32-bit words a record of the file `splats` came
    /// from, packed to the splats validation kept (`GpuSplats::origin`).
    /// Returned as it is when nothing was dropped.
    [[nodiscard]] Result<gpu::Buffer> keptOnly(const GpuSplats& splats, const gpu::Buffer& byRecord, uint32_t words,
                                               const char* label);
    /// The other way: `bySplat`, `words` words a kept splat, laid out a
    /// record each again, the dropped records holding `fill`.
    [[nodiscard]] Result<gpu::Buffer> toRecords(const GpuSplats& splats, const gpu::Buffer& bySplat, uint32_t words,
                                                uint32_t fill, const char* label);

    /// A cloud on the device back into float records -- position, linear
    /// opacity, log scales, rotation w x y z, the base colour (0.5 + SH0 * dc,
    /// as the shape keeps it) and the rest rgb per basis -- for what writes records (the USD
    /// export). Unpacked on the device; only the records cross.
    [[nodiscard]] Result<io::RawSplats> records(const GpuSplats& splats);
    /// The extent of `count` float4 positions, computed on the device.
    [[nodiscard]] Result<Bounds> boundsOf(const gpu::Buffer& positions, uint32_t count);

private:
    struct SogOnDevice;
    [[nodiscard]] Result<SogOnDevice> sogOnDevice(const io::RawSog& sog, uint32_t maxDegree);
    [[nodiscard]] Result<void> sogSlice(const SogOnDevice& on, uint32_t first, uint32_t n, const gpu::Buffer& into);
    [[nodiscard]] Result<GpuSplats> startSplats(const std::string& source, uint32_t declared, uint32_t keep,
                                                bool withPbr = false, bool withCrypto = false,
                                                uint32_t transferCount = 0, uint32_t shadowWords = 0,
                                                bool withNormals = false, bool withEmission = false,
                                                bool withLobes = false);
    /// Validates and decodes `n` records in `raw` into `splats` after `written`;
    /// `recordBase` is the first of them in the whole cloud, for `origin`.
    [[nodiscard]] Result<uint32_t> decodeSlice(const gpu::Buffer& raw, const io::SplatEncoding& e, uint32_t n,
                                               uint32_t written, uint32_t keep, GpuSplats& splats,
                                               uint32_t recordBase);
    [[nodiscard]] Result<void> finishSplats(GpuSplats& splats, uint32_t written);
    [[nodiscard]] Result<uint32_t> decodePoints(const gpu::Buffer& raw, uint32_t n, uint32_t first,
                                                uint32_t colourKind, float detail, uint32_t written,
                                                GpuPoints& points);
    [[nodiscard]] Result<gpu::Buffer> streamBuffer(const FloatStream& stream, const char* label);
    /// `SplatStreams::shPlanes` end to end in one buffer, `count` rgb a plane.
    [[nodiscard]] Result<gpu::Buffer> planeBuffer(const std::vector<FloatStream>& planes, uint32_t count,
                                                  const char* label);

    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel sogDecode_;
    gpu::PrefixSum     prefix_;
    gpu::ComputeKernel splatValidate_;
    gpu::ComputeKernel splatDecode_;
    gpu::ComputeKernel splatKept_;
    gpu::ComputeKernel splatKeptScatter_;
    gpu::ComputeKernel splatUnpack_;
    gpu::ComputeKernel pointsValidate_;
    gpu::ComputeKernel pointsDecode_;
    gpu::ComputeKernel splatStreams_;
    gpu::ComputeKernel pointStreams_;
    gpu::ComputeKernel boundsChunks_;
    gpu::ComputeKernel boundsReduce_;
};

/// Whether `path` names a SOG: a .sog bundle or an unbundled meta.json.
[[nodiscard]] bool isSog(const std::filesystem::path& path);

/// Any splat file onto the device: .ply, .splat, .spz through io::readSplats,
/// SOG through io::readSog and the device decode.
[[nodiscard]] Result<GpuSplats> loadSplatFile(CloudLoader& loader, const std::filesystem::path& path,
                                              uint32_t maxDegree = 3);

/// Any splat file as host records in the engine's float encoding, for what
/// consumes records (the USD export). SOG is decoded on the device and read back.
[[nodiscard]] Result<io::RawSplats> readSplatRecords(CloudLoader& loader, const std::filesystem::path& path,
                                                     uint32_t maxDegree = 3);

}   // namespace athenea::scene
