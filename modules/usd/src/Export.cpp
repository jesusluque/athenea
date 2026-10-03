// Copyright (c) 2026 jesus luque.
#include "athenea/usd/Export.h"

#include "athenea/scene/DecodeParams.h"

#include <algorithm>
#include <functional>
#include <cstring>
#include <set>
#include <type_traits>
#include <cmath>

#include <pxr/base/gf/vec3d.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdVol/particleField3DGaussianSplat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/layerUtils.h>
#include <pxr/usd/usdUtils/dependencies.h>
#include <pxr/base/gf/half.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>

#include "athenea/core/Hash.h"
#include "athenea/core/Log.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace athenea::usd {
namespace {

Result<gpu::Buffer> buffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                           const void* data = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, data);
}

}   // namespace

namespace {

/// `count` records, a slice of them at a time: `slice(first, n)` hands back a
/// buffer holding records `first` to `first + n`, from wherever they are.
using RecordSlice = std::function<Result<gpu::Buffer>(uint32_t first, uint32_t n)>;

Result<void> writeStage(gpu::ShaderLibrary& library, const io::SplatEncoding& e, uint32_t count,
                        const RecordSlice& slice, const std::filesystem::path& path,
                        const ExportOptions& options) {
    if (count == 0) {
        return Error(ErrorCode::InvalidArgument, "nothing to export");
    }
    static constexpr uint32_t kPerDegree[] = {0, 3, 8, 15};
    const uint32_t keep = std::min(e.restPerColour, kPerDegree[std::min(options.maxDegree, 3u)]);
    gpu::Device& device = library.device();
    auto kernel = gpu::ComputeKernel::create(library, "athenea/scene/splat_export", "splatExport");
    if (!kernel) {
        return std::move(kernel).error();
    }

    VtVec3fArray positions;
    VtQuatfArray orientations;
    VtVec3fArray scales;
    VtFloatArray opacities;
    VtVec3fArray coefficients;
    VtFloatArray metallics;
    VtFloatArray roughnesses;
    VtFloatArray transmissions;
    // The shading normal, as the kernel leaves it: unit length, and turned
    // as the positions are where the file is right-up-back.
    VtVec3fArray normals;
    const bool withNormals = e.normal != io::SplatEncoding::kNoField;
    const bool pbr = e.metallic != io::SplatEncoding::kNoField ||
                     e.roughness != io::SplatEncoding::kNoField ||
                     e.transmission != io::SplatEncoding::kNoField;
    positions.reserve(count);
    const uint32_t perRecord = 1 + keep;
    GfVec3d lo(1e30), hi(-1e30);
    uint32_t empty = 0;   ///< slots kept that hold no splat

    // Slices, as the loader does, so a very large cloud still fits a buffer.
    const uint64_t recordBytes = uint64_t{e.floatsPerRecord} * 4;
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, (uint64_t{256} << 20) / recordBytes));
    for (uint32_t first = 0; first < count; first += perSlice) {
        const uint32_t n = std::min(perSlice, count - first);
        auto rawBuffer = slice(first, n);
        if (!rawBuffer) return std::move(rawBuffer).error();
        auto posOpacity = buffer(device, n, 16, "export.posOpacity");
        auto rotation = buffer(device, n, 16, "export.rotation");
        auto scaleValid = buffer(device, n, 16, "export.scaleValid");
        auto coeff = buffer(device, uint64_t{n} * perRecord, 16, "export.coefficients");
        auto normal = buffer(device, withNormals ? n : 1, 16, "export.normal");
        auto material = buffer(device, pbr ? n : 1, 16, "export.pbr");
        if (!posOpacity || !rotation || !scaleValid || !coeff || !normal || !material) {
            return Error(ErrorCode::OutOfMemory, "cannot allocate export buffers");
        }
        gpu::CommandBatch batch(device);
        kernel->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["raw"].setBinding(rawBuffer->rhi());
            cursor["posOpacity"].setBinding(posOpacity->rhi());
            cursor["rotation"].setBinding(rotation->rhi());
            cursor["scaleValid"].setBinding(scaleValid->rhi());
            cursor["coefficients"].setBinding(coeff->rhi());
            cursor["normalOut"].setBinding(normal->rhi());
            cursor["pbrOut"].setBinding(material->rhi());
            scene::setDecodeParams(cursor, e, n, 0, keep, 1);
        });
        ATHENEA_TRY(batch.submit(true));
        auto po = posOpacity->readAll<float>(device);
        auto ro = rotation->readAll<float>(device);
        auto sv = scaleValid->readAll<float>(device);
        auto co = coeff->readAll<float>(device);
        auto no = normal->readAll<float>(device);
        auto pb = material->readAll<float>(device);
        if (!po || !ro || !sv || !co || !no || !pb) {
            return Error(ErrorCode::DeviceFailure, "cannot read export values back");
        }
        for (uint32_t i = 0; i < n; ++i) {
            // Every slot is written, whether or not there is a splat in it
            // (splat_export.slang says why). What an empty one does not do is
            // stretch the cloud's extent to wherever it stands.
            const bool there = (*sv)[size_t{i} * 4 + 3] >= 0.5F;
            empty += there ? 0u : 1u;
            const float* pp = po->data() + size_t{i} * 4;
            const float* rr = ro->data() + size_t{i} * 4;
            const float* ss = sv->data() + size_t{i} * 4;
            positions.push_back(GfVec3f(pp[0], pp[1], pp[2]));
            opacities.push_back(pp[3]);
            orientations.push_back(GfQuatf(rr[3], rr[0], rr[1], rr[2]));
            scales.push_back(GfVec3f(ss[0], ss[1], ss[2]));
            for (uint32_t k = 0; k < perRecord; ++k) {
                const float* cc = co->data() + (size_t{i} * perRecord + k) * 4;
                coefficients.push_back(GfVec3f(cc[0], cc[1], cc[2]));
            }
            if (pbr) {
                // Straight out of the record, as the kernel copied them: all
                // three are already 0 to 1, and a value is not a thing this
                // file decodes.
                const float* mm = pb->data() + size_t{i} * 4;
                metallics.push_back(mm[0]);
                roughnesses.push_back(mm[1]);
                transmissions.push_back(mm[2]);
            }
            if (withNormals) {
                const float* nn = no->data() + size_t{i} * 4;
                normals.push_back(GfVec3f(nn[0], nn[1], nn[2]));
            }
            if (there) {
                for (int axis = 0; axis < 3; ++axis) {
                    lo[axis] = std::min(lo[axis], static_cast<double>(pp[axis]));
                    hi[axis] = std::max(hi[axis], static_cast<double>(pp[axis]));
                }
            }
        }
    }
    if (positions.empty()) {
        return Error(ErrorCode::InvalidArgument, "no splat survived export");
    }
    if (empty == count) {
        return Error(ErrorCode::InvalidArgument, "no splat survived export");
    }
    if (empty > 0) {
        log::info("export: {} of {} slots hold no splat and are written empty", empty, count);
    }

    UsdStageRefPtr stage = UsdStage::CreateNew(path.string());
    if (!stage) {
        return Error::make(ErrorCode::IoFailure, "cannot create '{}'", path.string());
    }
    UsdGeomSetStageUpAxis(stage, options.upAxis == 'z' ? UsdGeomTokens->z : UsdGeomTokens->y);
    UsdGeomSetStageMetersPerUnit(stage, options.metersPerUnit);
    UsdGeomXform world = UsdGeomXform::Define(stage, SdfPath("/World"));
    stage->SetDefaultPrim(world.GetPrim());
    auto splats = UsdVolParticleField3DGaussianSplat::Define(stage, SdfPath("/World/Splats"));
    splats.CreatePositionsAttr(VtValue(positions));
    splats.CreateOrientationsAttr(VtValue(orientations));
    splats.CreateScalesAttr(VtValue(scales));
    splats.CreateOpacitiesAttr(VtValue(opacities));
    splats.CreateRadianceSphericalHarmonicsDegreeAttr(VtValue(keep == 15 ? 3 : keep == 8 ? 2 : keep == 3 ? 1 : 0));
    splats.CreateRadianceSphericalHarmonicsCoefficientsAttr(VtValue(coefficients));
    VtVec3fArray extent{GfVec3f(lo), GfVec3f(hi)};
    splats.CreateExtentAttr(VtValue(extent));
    if (options.rotateXDegrees != 0.0) {
        UsdGeomXformCommonAPI(splats.GetPrim()).SetRotate(
            GfVec3f(static_cast<float>(options.rotateXDegrees), 0.0F, 0.0F));
    }

    if (pbr) {
        // What a relit gaussian reflects with, beside the colours it reflects.
        // Primvars rather than attributes of the schema: the schema is USD's
        // and says nothing about a surface, while AtheneaSplatLightingAPI is ours.
        static const TfToken kMetallic("primvars:athenea:splat:metallic");
        static const TfToken kRoughness("primvars:athenea:splat:roughness");
        static const TfToken kTransmission("primvars:athenea:splat:transmission");
        UsdGeomPrimvarsAPI primvars(splats.GetPrim());
        primvars.CreatePrimvar(kMetallic, SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex)
            .Set(VtValue(metallics));
        primvars.CreatePrimvar(kRoughness, SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex)
            .Set(VtValue(roughnesses));
        primvars.CreatePrimvar(kTransmission, SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex)
            .Set(VtValue(transmissions));
    }

    // WHICH WAY THE SURFACE UNDER EACH GAUSSIAN FACED, apart from its frame
    // (AtheneaSplatLightingAPI): normal3f, so a host reads it as a normal and
    // a transform turns it as one. In the field's own space, as positions are.
    if (withNormals) {
        UsdGeomPrimvarsAPI(splats.GetPrim())
            .CreatePrimvar(TfToken("primvars:athenea:splat:normal"), SdfValueTypeNames->Normal3fArray,
                           UsdGeomTokens->vertex)
            .Set(VtValue(normals));
    }

    // WHICH PRIM EACH GAUSSIAN CAME FROM (AtheneaSplatCryptomatteAPI). One id a
    // record, empty slots included, so the primvar and the cloud's own arrays
    // stay index for index alike; the manifest says what the ids are called.
    if (options.cryptoObject.size() >= count && count > 0) {
        VtIntArray ids(count);
        for (uint32_t i = 0; i < count; ++i) {
            ids[i] = static_cast<int>(options.cryptoObject[i]);
        }
        UsdGeomPrimvarsAPI primvars(splats.GetPrim());
        primvars.CreatePrimvar(TfToken("primvars:athenea:splat:cryptoObject"), SdfValueTypeNames->IntArray,
                               UsdGeomTokens->vertex)
            .Set(VtValue(ids));
        primvars.CreatePrimvar(TfToken("primvars:athenea:splat:cryptoManifest"), SdfValueTypeNames->String,
                               UsdGeomTokens->constant)
            .Set(VtValue(core::cryptomatteManifest(options.cryptoManifest)));
        splats.GetPrim().ApplyAPI(TfToken("AtheneaSplatCryptomatteAPI"));
    }

    // WHICH GAUSSIANS ARE THIN WALLS: a glass sheet whose transmission is its
    // own transparency. Written only where there is one.
    if (options.thinWalled.size() >= count && count > 0 &&
        std::any_of(options.thinWalled.begin(), options.thinWalled.begin() + count,
                    [](int32_t v) { return v != 0; })) {
        VtIntArray thin(options.thinWalled.begin(), options.thinWalled.begin() + count);
        UsdGeomPrimvarsAPI(splats.GetPrim())
            .CreatePrimvar(TfToken("primvars:athenea:splat:thinWalled"), SdfValueTypeNames->IntArray,
                           UsdGeomTokens->vertex)
            .Set(VtValue(thin));
    }

    // HOW MUCH OF AN ENVIRONMENT REACHES EACH GAUSSIAN. Nine floats a record
    // of the direct half, and twenty-seven more of the indirect one where it
    // was baked, vertex-interpolated like everything else a gaussian carries.
    // A cloud with these is lit by whatever sky it is put under, which is the
    // whole reason they are here rather than a colour baked under one dome.
    if (options.transferDirect.size() >= size_t{count} * 9 && count > 0) {
        UsdGeomPrimvarsAPI primvars(splats.GetPrim());
        VtFloatArray direct(options.transferDirect.begin(),
                            options.transferDirect.begin() + size_t{count} * 9);
        UsdGeomPrimvar made = primvars.CreatePrimvar(TfToken("primvars:athenea:splat:transferDirect"),
                                                     SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex);
        made.Set(VtValue(direct));
        made.SetElementSize(9);
        if (options.transferIndirect.size() >= size_t{count} * 27) {
            VtFloatArray indirect(options.transferIndirect.begin(),
                                  options.transferIndirect.begin() + size_t{count} * 27);
            UsdGeomPrimvar bounced = primvars.CreatePrimvar(TfToken("primvars:athenea:splat:transferIndirect"),
                                                            SdfValueTypeNames->FloatArray,
                                                            UsdGeomTokens->vertex);
            bounced.Set(VtValue(indirect));
            bounced.SetElementSize(27);
        }
        if (options.shadowBits.size() >= size_t{count} * 2) {
            VtIntArray bits(options.shadowBits.begin(), options.shadowBits.begin() + size_t{count} * 2);
            UsdGeomPrimvar open = primvars.CreatePrimvar(TfToken("primvars:athenea:splat:shadowBits"),
                                                         SdfValueTypeNames->IntArray, UsdGeomTokens->vertex);
            open.Set(VtValue(bits));
            open.SetElementSize(2);
        }
    }

    if (options.relight) {
        // The cloud's colours are an albedo, not radiance: a mesh's material
        // said what the surface reflects, and nothing has lit it yet. The
        // scene's lights do that (AtheneaSplatLightingAPI), which is what makes a
        // converted mesh sit under the same lights the mesh would have.
        static const TfToken kRelight("primvars:athenea:splat:relight");
        splats.GetPrim().CreateAttribute(kRelight, SdfValueTypeNames->Bool, true).Set(true);
    }
    if (options.ior > 1.0F) {
        static const TfToken kIor("primvars:athenea:splat:ior");
        splats.GetPrim().CreateAttribute(kIor, SdfValueTypeNames->Float, false).Set(options.ior);
    }
    if (options.litBody) {
        // Its colours are light, not an albedo: what a frame adds is the
        // reflection, which is the part a single colour cannot hold.
        static const TfToken kLit("primvars:athenea:splat:litBody");
        splats.GetPrim().CreateAttribute(kLit, SdfValueTypeNames->Bool, true).Set(true);
    }

    // AND THE SCHEMA THOSE PRIMVARS ARE DECLARED BY. They were written and
    // the API never applied, so `relight` came out `custom` and a host that
    // asks the prim which APIs it has -- usdview, a validator -- saw none of
    // the lighting a cloud carries: its PBR, its transfer, its index.
    {
        static const TfToken kLighting[] = {
            TfToken("primvars:athenea:splat:relight"),      TfToken("primvars:athenea:splat:litBody"),
            TfToken("primvars:athenea:splat:ior"),          TfToken("primvars:athenea:splat:metallic"),
            TfToken("primvars:athenea:splat:roughness"),    TfToken("primvars:athenea:splat:transmission"),
            TfToken("primvars:athenea:splat:transferDirect"), TfToken("primvars:athenea:splat:transferIndirect"),
            TfToken("primvars:athenea:splat:thinWalled"),   TfToken("primvars:athenea:splat:shadowBits"),
            TfToken("primvars:athenea:splat:normal")};
        const UsdPrim prim = splats.GetPrim();
        if (std::any_of(std::begin(kLighting), std::end(kLighting),
                        [&](const TfToken& name) { return prim.GetAttribute(name).HasAuthoredValue(); })) {
            prim.ApplyAPI(TfToken("AtheneaSplatLightingAPI"));
        }
    }

    // THE RIG, WHEN THE CLOUD IS CARRIED BY ONE.
    //
    // Four joints a gaussian and their weights, the transform out of the
    // cloud's space, and the joints' own transforms as time samples -- which
    // is the only thing about an animated cloud that changes from frame to
    // frame. The splitting below is a rearrangement of values the device
    // computed, not arithmetic on them: the conversion writes `(joint,
    // weight)` pairs and USD wants two arrays.
    if (options.skinning != nullptr && options.skinning->valid()) {
        const SplatSkinning& rig = *options.skinning;
        const size_t carried = rig.influences.size() / 8;
        if (carried != positions.size()) {
            return Error::make(ErrorCode::InvalidArgument,
                               "the rig carries {} gaussians and the cloud has {}", carried,
                               positions.size());
        }
        VtIntArray   indices(carried * 4);
        VtFloatArray weights(carried * 4);
        for (size_t k = 0; k < carried * 4; ++k) {
            indices[k] = static_cast<int>(rig.influences[k * 2]);
            weights[k] = rig.influences[k * 2 + 1];
        }
        // THE BINDING IS UsdSkel'S. SkelBindingAPI on the cloud, as on a mesh:
        // the influences under `skel:jointIndices` and `skel:jointWeights`
        // (four a gaussian, the elementSize says), the bind transform, the
        // Skeleton by relationship and the joints by name. Any UsdSkel reader
        // sees a skinnable prim; this engine's own reader used to want the
        // same numbers under names of its own (AtheneaSplatSkinningAPI), and
        // still reads them from a file that has them.
        UsdSkelBindingAPI bound = UsdSkelBindingAPI::Apply(splats.GetPrim());
        bound.CreateJointIndicesPrimvar(false, 4).Set(indices);
        bound.CreateJointWeightsPrimvar(false, 4).Set(weights);
        GfMatrix4d bind;
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                bind[row][column] = rig.geomBindTransform[static_cast<size_t>(row) * 4 +
                                                          static_cast<size_t>(column)];
            }
        }
        bound.CreateGeomBindTransformAttr().Set(bind);
        if (!rig.skeleton.empty()) {
            bound.CreateSkeletonRel().SetTargets({SdfPath(rig.skeleton)});
        }
        if (!rig.jointNames.empty()) {
            VtTokenArray names(rig.jointNames.size());
            for (size_t j = 0; j < rig.jointNames.size(); ++j) {
                names[j] = TfToken(rig.jointNames[j]);
            }
            bound.CreateJointsAttr().Set(names);
        }

        // AND THE JOINTS' TRANSFORMS, WRITTEN DOWN. The Skeleton's animation
        // gives them at any instant, resolved at render time; this is the
        // same numbers at the instants the conversion read them, kept so a
        // stage that composes the cloud without its Skeleton still moves,
        // and one with it skips the resolving. AtheneaSplatSkinningAPI's one
        // remaining attribute.
        UsdGeomPrimvarsAPI rigged(splats.GetPrim());
        UsdGeomPrimvar moved = rigged.CreatePrimvar(TfToken("primvars:athenea:splat:skinningXforms"),
                                                    SdfValueTypeNames->Matrix4dArray,
                                                    UsdGeomTokens->constant);
        for (size_t frame = 0; frame < rig.times.size(); ++frame) {
            VtMatrix4dArray at(rig.joints);
            const float* held = rig.xforms.data() + frame * rig.joints * 16;
            for (uint32_t joint = 0; joint < rig.joints; ++joint) {
                for (int row = 0; row < 4; ++row) {
                    for (int column = 0; column < 4; ++column) {
                        at[joint][row][column] = held[joint * 16 + static_cast<size_t>(row) * 4 +
                                                     static_cast<size_t>(column)];
                    }
                }
            }
            moved.Set(at, UsdTimeCode(rig.times[frame]));
        }
        stage->SetStartTimeCode(rig.times.front());
        stage->SetEndTimeCode(rig.times.back());
        // WITHOUT THIS THE CLOUD PLAYS SLOW. A layer that does not say what a
        // time code is worth is read at 24, and USD scales every sample it
        // holds by the root layer's rate over that one -- so a cloud sampled
        // at 30 and composed under a stage at 30 was stretched by 30/24 and
        // drifted further from the mesh the further the clip ran.
        stage->SetTimeCodesPerSecond(rig.timeCodesPerSecond);
    }

    if (options.addCamera) {
        // The centre where the cloud is drawn: turned with it about x.
        GfVec3d centre = (lo + hi) * 0.5;
        if (options.rotateXDegrees != 0.0) {
            const double a = GfDegreesToRadians(options.rotateXDegrees);
            centre = GfVec3d(centre[0], centre[1] * std::cos(a) - centre[2] * std::sin(a),
                             centre[1] * std::sin(a) + centre[2] * std::cos(a));
        }
        const double size = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-3});
        UsdGeomCamera camera = UsdGeomCamera::Define(stage, SdfPath("/World/Camera"));
        camera.CreateFocalLengthAttr(VtValue(35.0F));
        camera.CreateHorizontalApertureAttr(VtValue(24.576F));
        camera.CreateVerticalApertureAttr(VtValue(13.824F));
        camera.CreateClippingRangeAttr(VtValue(GfVec2f(static_cast<float>(size * 0.001), static_cast<float>(size * 20.0))));
        // Looking down -Z at the centre from in front of it.
        UsdGeomXformCommonAPI(camera.GetPrim()).SetTranslate(
            GfVec3d(centre[0], centre[1], centre[2] + size * 1.2));
    }
    if (!stage->GetRootLayer()->Save()) {
        return Error::make(ErrorCode::IoFailure, "cannot save '{}'", path.string());
    }
    return ok();
}

}   // namespace

Result<void> writeParticleFieldStage(gpu::ShaderLibrary& library, const io::RawSplats& raw,
                                     const std::filesystem::path& path, const ExportOptions& options) {
    const io::SplatEncoding& e = raw.encoding;
    if (raw.records.size() < size_t{raw.count} * e.floatsPerRecord) {
        return Error(ErrorCode::InvalidArgument, "export: fewer records than the count says");
    }
    gpu::Device& device = library.device();
    return writeStage(
        library, e, raw.count,
        [&](uint32_t first, uint32_t n) {
            return buffer(device, uint64_t{n} * e.floatsPerRecord, 4, "export.raw",
                          raw.records.data() + size_t{first} * e.floatsPerRecord);
        },
        path, options);
}

Result<void> writeParticleFieldStage(gpu::ShaderLibrary& library, const DeviceSplatRecords& records,
                                     const std::filesystem::path& path, const ExportOptions& options) {
    const io::SplatEncoding& e = records.encoding;
    const uint64_t recordBytes = uint64_t{e.floatsPerRecord} * 4;
    if (!records.records.valid() || records.records.bytes() < uint64_t{records.count} * recordBytes) {
        return Error(ErrorCode::InvalidArgument, "export: fewer records on the device than the count says");
    }
    gpu::Device& device = library.device();
    return writeStage(
        library, e, records.count,
        [&](uint32_t first, uint32_t n) -> Result<gpu::Buffer> {
            // The whole cloud in one slice is the buffer itself; a slice of
            // a larger one is copied out of it on the device.
            if (first == 0 && n == records.count) {
                return records.records;
            }
            auto part = buffer(device, uint64_t{n} * e.floatsPerRecord, 4, "export.raw");
            if (!part) return std::move(part).error();
            gpu::CommandBatch batch(device);
            batch.encoder()->copyBuffer(part->rhi(), 0, records.records.rhi(), uint64_t{first} * recordBytes,
                                        uint64_t{n} * recordBytes);
            batch.markDirty();
            ATHENEA_TRY(batch.submit(true));
            return part;
        },
        path, options);
}

Result<void> writeVisibility(const std::filesystem::path& path, const std::string& prim,
                             std::span<const float> parts, std::span<const int32_t> texels,
                             std::span<const int32_t> partOf, std::span<const int32_t> ambient) {
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    if (!stage) {
        return Error::make(ErrorCode::IoFailure, "cannot open USD stage '{}'", path.string());
    }
    const UsdPrim held = stage->GetPrimAtPath(SdfPath(prim));
    if (!held) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no such prim on '{}'", prim, path.string());
    }
    UsdGeomPrimvarsAPI primvars(held);
    VtFloatArray partArray(parts.begin(), parts.end());
    VtIntArray texelArray(texels.begin(), texels.end());
    primvars.CreatePrimvar(TfToken("primvars:athenea:splat:visibilityParts"), SdfValueTypeNames->FloatArray,
                           UsdGeomTokens->constant)
        .Set(partArray);
    primvars.CreatePrimvar(TfToken("primvars:athenea:splat:visibilityTexels"), SdfValueTypeNames->IntArray,
                           UsdGeomTokens->constant)
        .Set(texelArray);
    VtIntArray ambientArray(ambient.begin(), ambient.end());
    primvars.CreatePrimvar(TfToken("primvars:athenea:splat:visibilityAmbient"), SdfValueTypeNames->IntArray,
                           UsdGeomTokens->constant)
        .Set(ambientArray);
    VtIntArray partOfArray(partOf.begin(), partOf.end());
    primvars.CreatePrimvar(TfToken("primvars:athenea:splat:visibilityPartOf"), SdfValueTypeNames->IntArray,
                           UsdGeomTokens->vertex)
        .Set(partOfArray);
    if (!stage->GetRootLayer()->Save()) {
        return Error::make(ErrorCode::IoFailure, "cannot save '{}'", path.string());
    }
    return ok();
}

Result<io::RawSplats> readParticleFieldRecords(const std::filesystem::path& path, const std::string& prim,
                                               std::string* where, bool* moved) {
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    if (!stage) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}'", path.string());
    }
    UsdVolParticleField3DGaussianSplat field;
    if (!prim.empty()) {
        field = UsdVolParticleField3DGaussianSplat(stage->GetPrimAtPath(SdfPath(prim)));
    } else {
        for (const UsdPrim& candidate : stage->Traverse()) {
            if (candidate.IsA<UsdVolParticleField3DGaussianSplat>()) {
                field = UsdVolParticleField3DGaussianSplat(candidate);
                break;
            }
        }
    }
    if (!field) {
        return Error::make(ErrorCode::NotFound, "'{}' has no gaussian ParticleField{}", path.string(),
                           prim.empty() ? std::string() : " at " + prim);
    }
    VtVec3fArray positions;
    VtQuatfArray orientations;
    VtQuathArray orientationsHalf;
    VtVec3fArray scales;
    VtFloatArray opacities;
    VtVec3fArray coefficients;
    int degree = 0;
    // The earliest value: a cloud whose arrays are sampled has no default.
    const UsdTimeCode at = UsdTimeCode::EarliestTime();
    field.GetPositionsAttr().Get(&positions, at);
    if (!field.GetOrientationsAttr().Get(&orientations, at) && field.GetOrientationsAttr().Get(&orientationsHalf, at)) {
        orientations.reserve(orientationsHalf.size());
        for (const GfQuath& q : orientationsHalf) {
            orientations.push_back(GfQuatf(q));
        }
    }
    field.GetScalesAttr().Get(&scales, at);
    field.GetOpacitiesAttr().Get(&opacities, at);
    field.GetRadianceSphericalHarmonicsDegreeAttr().Get(&degree, at);
    field.GetRadianceSphericalHarmonicsCoefficientsAttr().Get(&coefficients, at);
    const size_t n = positions.size();
    if (n == 0 || orientations.size() != n || scales.size() != n || opacities.size() != n) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': a ParticleField needs one position, orientation, scale "
                           "and opacity a gaussian", field.GetPath().GetString());
    }
    const uint32_t perSplat = static_cast<uint32_t>((degree + 1) * (degree + 1));
    if (coefficients.size() != n * perSplat) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': {} harmonic coefficients for {} gaussians at degree {}",
                           field.GetPath().GetString(), coefficients.size(), n, degree);
    }
    // The shading normal, where the field keeps one: three floats more a
    // record, after the harmonics.
    VtVec3fArray normals;
    {
        const UsdGeomPrimvar primvar = UsdGeomPrimvarsAPI(field.GetPrim()).GetPrimvar(TfToken("athenea:splat:normal"));
        VtValue value;
        if (primvar && primvar.Get(&value, at)) {
            if (value.IsHolding<VtVec3fArray>()) {
                normals = value.UncheckedGet<VtVec3fArray>();
            } else if (value.IsHolding<VtVec3hArray>()) {
                for (const GfVec3h& h : value.UncheckedGet<VtVec3hArray>()) {
                    normals.push_back(GfVec3f(h));
                }
            }
        }
        if (normals.size() != n) {
            normals.clear();   // a shorter array is no array
        }
    }
    io::RawSplats raw;
    raw.source = path.string();
    io::SplatEncoding& e = raw.encoding;
    const uint32_t keep = perSplat - 1;
    e.floatsPerRecord = 14 + 3 * keep + (normals.empty() ? 0 : 3);
    if (!normals.empty()) {
        e.normal = 14 + 3 * keep;
    }
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
    e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
    e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14;
    e.restPerColour = keep;
    e.restColourOuter = 0;
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Linear;
    raw.count = static_cast<uint32_t>(n);
    raw.records.resize(n * e.floatsPerRecord);
    for (size_t i = 0; i < n; ++i) {
        float* r = raw.records.data() + i * e.floatsPerRecord;
        r[0] = positions[i][0];
        r[1] = positions[i][1];
        r[2] = positions[i][2];
        r[3] = opacities[i];
        r[4] = scales[i][0];
        r[5] = scales[i][1];
        r[6] = scales[i][2];
        r[7] = orientations[i].GetReal();
        r[8] = orientations[i].GetImaginary()[0];
        r[9] = orientations[i].GetImaginary()[1];
        r[10] = orientations[i].GetImaginary()[2];
        for (uint32_t k = 0; k < perSplat; ++k) {
            const GfVec3f& c = coefficients[i * perSplat + k];
            r[11 + 3 * k] = c[0];
            r[12 + 3 * k] = c[1];
            r[13 + 3 * k] = c[2];
        }
        if (!normals.empty()) {
            r[e.normal] = normals[i][0];
            r[e.normal + 1] = normals[i][1];
            r[e.normal + 2] = normals[i][2];
        }
    }
    if (where != nullptr) {
        *where = field.GetPath().GetString();
    }
    if (moved != nullptr) {
        UsdGeomXformCache cache;
        *moved = cache.GetLocalToWorldTransform(field.GetPrim()) != GfMatrix4d(1.0);
    }
    return raw;
}

namespace {

/// The words of one array, however USD typed it: floats as floats (halves
/// and doubles widened -- a format's own types, not arithmetic on the data),
/// ints as ints, a quaternion real first. `false` for a type no gaussian
/// carries.
template <typename T>
float floatOf(const T& v) {
    return static_cast<float>(v);
}

template <typename Array, size_t Components>
void wordsOfVectors(const Array& array, std::vector<uint32_t>& words, bool integer) {
    words.reserve(array.size() * Components);
    for (const auto& element : array) {
        for (size_t c = 0; c < Components; ++c) {
            if (integer) {
                words.push_back(static_cast<uint32_t>(element[c]));
            } else {
                const float f = floatOf(element[c]);
                uint32_t w;
                std::memcpy(&w, &f, 4);
                words.push_back(w);
            }
        }
    }
}

template <typename Array>
void wordsOfScalars(const Array& array, std::vector<uint32_t>& words, bool integer) {
    words.reserve(array.size());
    for (const auto& element : array) {
        if (integer) {
            words.push_back(static_cast<uint32_t>(element));
        } else {
            const float f = floatOf(element);
            uint32_t w;
            std::memcpy(&w, &f, 4);
            words.push_back(w);
        }
    }
}

template <typename Quats>
void wordsOfQuats(const Quats& array, std::vector<uint32_t>& words) {
    words.reserve(array.size() * 4);
    for (const auto& q : array) {
        const float parts[4] = {floatOf(q.GetReal()), floatOf(q.GetImaginary()[0]), floatOf(q.GetImaginary()[1]),
                                floatOf(q.GetImaginary()[2])};
        for (const float f : parts) {
            uint32_t w;
            std::memcpy(&w, &f, 4);
            words.push_back(w);
        }
    }
}

/// Components an element of `value` has, and whether they are integers; 0
/// for a type this does not carry.
uint32_t componentsOf(const VtValue& value, bool& integer) {
    integer = false;
    if (value.IsHolding<VtFloatArray>() || value.IsHolding<VtHalfArray>() || value.IsHolding<VtDoubleArray>()) return 1;
    if (value.IsHolding<VtVec2fArray>() || value.IsHolding<VtVec2hArray>() || value.IsHolding<VtVec2dArray>()) return 2;
    if (value.IsHolding<VtVec3fArray>() || value.IsHolding<VtVec3hArray>() || value.IsHolding<VtVec3dArray>()) return 3;
    if (value.IsHolding<VtVec4fArray>() || value.IsHolding<VtVec4hArray>() || value.IsHolding<VtVec4dArray>() ||
        value.IsHolding<VtQuatfArray>() || value.IsHolding<VtQuathArray>()) {
        return 4;
    }
    integer = true;
    if (value.IsHolding<VtIntArray>() || value.IsHolding<VtUIntArray>()) return 1;
    if (value.IsHolding<VtVec2iArray>()) return 2;
    if (value.IsHolding<VtVec3iArray>()) return 3;
    if (value.IsHolding<VtVec4iArray>()) return 4;
    integer = false;
    return 0;
}

size_t lengthOf(const VtValue& value) {
    return value.GetArraySize();
}

std::vector<uint32_t> wordsOf(const VtValue& v) {
    std::vector<uint32_t> w;
    if (v.IsHolding<VtFloatArray>()) wordsOfScalars(v.UncheckedGet<VtFloatArray>(), w, false);
    else if (v.IsHolding<VtHalfArray>()) wordsOfScalars(v.UncheckedGet<VtHalfArray>(), w, false);
    else if (v.IsHolding<VtDoubleArray>()) wordsOfScalars(v.UncheckedGet<VtDoubleArray>(), w, false);
    else if (v.IsHolding<VtVec2fArray>()) wordsOfVectors<VtVec2fArray, 2>(v.UncheckedGet<VtVec2fArray>(), w, false);
    else if (v.IsHolding<VtVec2hArray>()) wordsOfVectors<VtVec2hArray, 2>(v.UncheckedGet<VtVec2hArray>(), w, false);
    else if (v.IsHolding<VtVec2dArray>()) wordsOfVectors<VtVec2dArray, 2>(v.UncheckedGet<VtVec2dArray>(), w, false);
    else if (v.IsHolding<VtVec3fArray>()) wordsOfVectors<VtVec3fArray, 3>(v.UncheckedGet<VtVec3fArray>(), w, false);
    else if (v.IsHolding<VtVec3hArray>()) wordsOfVectors<VtVec3hArray, 3>(v.UncheckedGet<VtVec3hArray>(), w, false);
    else if (v.IsHolding<VtVec3dArray>()) wordsOfVectors<VtVec3dArray, 3>(v.UncheckedGet<VtVec3dArray>(), w, false);
    else if (v.IsHolding<VtVec4fArray>()) wordsOfVectors<VtVec4fArray, 4>(v.UncheckedGet<VtVec4fArray>(), w, false);
    else if (v.IsHolding<VtVec4hArray>()) wordsOfVectors<VtVec4hArray, 4>(v.UncheckedGet<VtVec4hArray>(), w, false);
    else if (v.IsHolding<VtVec4dArray>()) wordsOfVectors<VtVec4dArray, 4>(v.UncheckedGet<VtVec4dArray>(), w, false);
    else if (v.IsHolding<VtQuatfArray>()) wordsOfQuats(v.UncheckedGet<VtQuatfArray>(), w);
    else if (v.IsHolding<VtQuathArray>()) wordsOfQuats(v.UncheckedGet<VtQuathArray>(), w);
    else if (v.IsHolding<VtIntArray>()) wordsOfScalars(v.UncheckedGet<VtIntArray>(), w, true);
    else if (v.IsHolding<VtUIntArray>()) wordsOfScalars(v.UncheckedGet<VtUIntArray>(), w, true);
    else if (v.IsHolding<VtVec2iArray>()) wordsOfVectors<VtVec2iArray, 2>(v.UncheckedGet<VtVec2iArray>(), w, true);
    else if (v.IsHolding<VtVec3iArray>()) wordsOfVectors<VtVec3iArray, 3>(v.UncheckedGet<VtVec3iArray>(), w, true);
    else if (v.IsHolding<VtVec4iArray>()) wordsOfVectors<VtVec4iArray, 4>(v.UncheckedGet<VtVec4iArray>(), w, true);
    return w;
}

float floatAt(const std::vector<uint32_t>& words, size_t k) {
    float f;
    std::memcpy(&f, &words[k], 4);
    return f;
}

template <typename Array, size_t Components>
VtValue vectorsOf(const std::vector<uint32_t>& words, bool integer) {
    Array out(words.size() / Components);
    for (size_t e = 0; e < out.size(); ++e) {
        for (size_t c = 0; c < Components; ++c) {
            using Scalar = std::decay_t<decltype(out[e][c])>;
            out[e][c] = integer ? static_cast<Scalar>(static_cast<int32_t>(words[e * Components + c]))
                                : static_cast<Scalar>(floatAt(words, e * Components + c));
        }
    }
    return VtValue(out);
}

template <typename Array>
VtValue scalarsOf(const std::vector<uint32_t>& words, bool integer) {
    Array out(words.size());
    for (size_t e = 0; e < out.size(); ++e) {
        using Scalar = std::decay_t<decltype(out[e])>;
        out[e] = integer ? static_cast<Scalar>(static_cast<int32_t>(words[e])) : static_cast<Scalar>(floatAt(words, e));
    }
    return VtValue(out);
}

template <typename Array, typename Quat, typename Scalar>
VtValue quatsOf(const std::vector<uint32_t>& words) {
    Array out(words.size() / 4);
    for (size_t e = 0; e < out.size(); ++e) {
        out[e] = Quat(Scalar(floatAt(words, e * 4)), Scalar(floatAt(words, e * 4 + 1)), Scalar(floatAt(words, e * 4 + 2)),
                      Scalar(floatAt(words, e * 4 + 3)));
    }
    return VtValue(out);
}

/// Words back into the array type `like` holds.
VtValue valueLike(const VtValue& like, const std::vector<uint32_t>& w) {
    if (like.IsHolding<VtFloatArray>()) return scalarsOf<VtFloatArray>(w, false);
    if (like.IsHolding<VtHalfArray>()) return scalarsOf<VtHalfArray>(w, false);
    if (like.IsHolding<VtDoubleArray>()) return scalarsOf<VtDoubleArray>(w, false);
    if (like.IsHolding<VtVec2fArray>()) return vectorsOf<VtVec2fArray, 2>(w, false);
    if (like.IsHolding<VtVec2hArray>()) return vectorsOf<VtVec2hArray, 2>(w, false);
    if (like.IsHolding<VtVec2dArray>()) return vectorsOf<VtVec2dArray, 2>(w, false);
    if (like.IsHolding<VtVec3fArray>()) return vectorsOf<VtVec3fArray, 3>(w, false);
    if (like.IsHolding<VtVec3hArray>()) return vectorsOf<VtVec3hArray, 3>(w, false);
    if (like.IsHolding<VtVec3dArray>()) return vectorsOf<VtVec3dArray, 3>(w, false);
    if (like.IsHolding<VtVec4fArray>()) return vectorsOf<VtVec4fArray, 4>(w, false);
    if (like.IsHolding<VtVec4hArray>()) return vectorsOf<VtVec4hArray, 4>(w, false);
    if (like.IsHolding<VtVec4dArray>()) return vectorsOf<VtVec4dArray, 4>(w, false);
    if (like.IsHolding<VtQuatfArray>()) return quatsOf<VtQuatfArray, GfQuatf, float>(w);
    if (like.IsHolding<VtQuathArray>()) return quatsOf<VtQuathArray, GfQuath, GfHalf>(w);
    if (like.IsHolding<VtIntArray>()) return scalarsOf<VtIntArray>(w, true);
    if (like.IsHolding<VtUIntArray>()) return scalarsOf<VtUIntArray>(w, true);
    if (like.IsHolding<VtVec2iArray>()) return vectorsOf<VtVec2iArray, 2>(w, true);
    if (like.IsHolding<VtVec3iArray>()) return vectorsOf<VtVec3iArray, 3>(w, true);
    if (like.IsHolding<VtVec4iArray>()) return vectorsOf<VtVec4iArray, 4>(w, true);
    return VtValue();
}

UsdVolParticleField3DGaussianSplat fieldOn(const UsdStageRefPtr& stage, const std::string& prim) {
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

/// The schema's own arrays, which a decimation writes from the gaussians.
bool isSchemaArray(const TfToken& name) {
    static const std::set<TfToken> schema{TfToken("positions"), TfToken("orientations"), TfToken("orientationsh"),
                                          TfToken("scales"), TfToken("opacities"),
                                          TfToken("radiance:sphericalHarmonicsCoefficients"), TfToken("extent")};
    return schema.contains(name);
}

}   // namespace

Result<std::vector<GaussianArray>> readGaussianArrays(const std::filesystem::path& path, const std::string& prim,
                                                     uint32_t count) {
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    if (!stage) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}'", path.string());
    }
    const UsdVolParticleField3DGaussianSplat field = fieldOn(stage, prim);
    if (!field || count == 0) {
        return Error::make(ErrorCode::NotFound, "'{}' has no gaussian ParticleField", path.string());
    }
    std::vector<GaussianArray> out;
    const UsdGeomPrimvarsAPI primvars(field.GetPrim());
    for (const UsdAttribute& attribute : field.GetPrim().GetAuthoredAttributes()) {
        const TfToken name = attribute.GetName();
        std::vector<double> times;
        attribute.GetTimeSamples(&times);
        // The schema's arrays are written from the gaussians -- unless they
        // move, when every sample is merged as any other array's.
        if (isSchemaArray(name) && times.empty()) {
            continue;
        }
        if (name == TfToken("extent")) {
            continue;
        }
        VtValue value;
        if (!attribute.Get(&value, UsdTimeCode::EarliestTime()) || !value.IsArrayValued()) {
            continue;
        }
        GaussianArray array;
        array.name = name.GetString();
        array.typeName = attribute.GetTypeName().GetAsToken().GetString();
        const uint32_t components = componentsOf(value, array.integer);
        if (components == 0) {
            continue;
        }
        const UsdGeomPrimvar primvar = primvars.GetPrimvar(UsdGeomPrimvar::StripPrimvarsName(name));
        if (UsdGeomPrimvar::IsPrimvar(attribute)) {
            const TfToken interpolation = primvar.GetInterpolation();
            if (interpolation != UsdGeomTokens->vertex && interpolation != UsdGeomTokens->varying) {
                continue;   // constant or uniform: one for the field, copied as it is
            }
            if (primvar.IsIndexed()) {
                athenea::log::info("decimate: '{}' is indexed, which a gaussian's own values are not; left as it is",
                               array.name);
                continue;
            }
            array.interpolation = interpolation.GetString();
            array.elementSize = static_cast<uint32_t>(std::max(primvar.GetElementSize(), 1));
        }
        const size_t length = lengthOf(value);
        if (length != size_t{count} * array.elementSize) {
            if (!array.interpolation.empty()) {
                athenea::log::info("decimate: '{}' has {} values for {} gaussians; left as it is", array.name, length,
                               count);
            }
            continue;   // not a gaussian's
        }
        array.width = components * array.elementSize;
        array.values.push_back(wordsOf(value));
        array.times = times;
        for (const double t : times) {
            VtValue sample;
            attribute.Get(&sample, UsdTimeCode(t));
            if (lengthOf(sample) != length) {
                return Error::make(ErrorCode::InvalidArgument,
                                   "'{}' changes length at time {}: a gaussian's array cannot be decimated", array.name,
                                   t);
            }
            array.values.push_back(wordsOf(sample));
        }
        out.push_back(std::move(array));
    }
    return out;
}

Result<void> writeDecimatedStage(const std::filesystem::path& input, const std::filesystem::path& output,
                                 const std::string& prim, const std::filesystem::path& gaussians,
                                 const std::vector<GaussianArray>& arrays) {
    SdfLayerRefPtr source = SdfLayer::FindOrOpen(input.string());
    if (!source) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}'", input.string());
    }
    if (!source->Export(output.string())) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}'", output.string());
    }
    SdfLayerRefPtr copy = SdfLayer::FindOrOpen(output.string());
    if (!copy) {
        return Error::make(ErrorCode::IoFailure, "cannot reopen '{}'", output.string());
    }
    copy->Reload(true);
    // Anchored where they came from: a copy elsewhere would lose every
    // relative path -- a texture, a sublayer, the skeleton's layer.
    if (std::filesystem::absolute(input).parent_path() != std::filesystem::absolute(output).parent_path()) {
        UsdUtilsModifyAssetPaths(copy, [&](const std::string& asset) {
            if (asset.empty() || !std::filesystem::path(asset).is_relative()) {
                return asset;
            }
            return SdfComputeAssetPathRelativeToLayer(source, asset);
        });
        copy->Save();
    }
    UsdStageRefPtr stage = UsdStage::Open(copy);
    UsdStageRefPtr kept = UsdStage::Open(gaussians.string());
    if (!stage || !kept) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}' or '{}'", output.string(), gaussians.string());
    }
    UsdVolParticleField3DGaussianSplat field = fieldOn(stage, prim);
    const UsdVolParticleField3DGaussianSplat fewer = fieldOn(kept, "/World/Splats");
    if (!field || !fewer) {
        return Error::make(ErrorCode::NotFound, "'{}': no gaussian ParticleField to write into", output.string());
    }
    stage->SetEditTarget(stage->GetRootLayer());
    // The schema's own arrays, as the gaussians are now.
    for (const char* name : {"positions", "orientations", "scales", "opacities",
                             "radiance:sphericalHarmonicsCoefficients", "radiance:sphericalHarmonicsDegree",
                             "extent"}) {
        const UsdAttribute from = fewer.GetPrim().GetAttribute(TfToken(name));
        VtValue value;
        if (!from || !from.Get(&value)) {
            continue;
        }
        UsdAttribute to = field.GetPrim().GetAttribute(TfToken(name));
        if (!to) {
            to = field.GetPrim().CreateAttribute(TfToken(name), from.GetTypeName());
        }
        // A half orientation where the file had one: the value, retyped.
        if (to.GetTypeName() != from.GetTypeName() && value.IsHolding<VtQuatfArray>() &&
            to.GetTypeName() == SdfValueTypeNames->QuathArray) {
            VtQuathArray halves;
            for (const GfQuatf& q : value.UncheckedGet<VtQuatfArray>()) {
                halves.push_back(GfQuath(q));
            }
            value = VtValue(halves);
        }
        to.Set(value);
    }
    if (field.GetPrim().HasAttribute(TfToken("orientationsh")) && fewer.GetOrientationsAttr()) {
        field.GetPrim().RemoveProperty(TfToken("orientationsh"));
    }
    // Everything else a gaussian carries, merged.
    for (const GaussianArray& array : arrays) {
        UsdAttribute to = field.GetPrim().GetAttribute(TfToken(array.name));
        if (!to) {
            continue;
        }
        VtValue like;
        to.Get(&like, UsdTimeCode::EarliestTime());
        if (!array.times.empty()) {
            to.Clear();   // the samples are the merged ones, every one of them
            for (size_t k = 0; k < array.times.size(); ++k) {
                to.Set(valueLike(like, array.values[k + 1]), UsdTimeCode(array.times[k]));
            }
        } else {
            to.Set(valueLike(like, array.values[0]));
        }
    }
    if (!stage->GetRootLayer()->Save()) {
        return Error::make(ErrorCode::IoFailure, "cannot save '{}'", output.string());
    }
    return ok();
}

Result<void> writeLodAssembly(const std::filesystem::path& path, const std::vector<LodLevelFile>& levels,
                              const std::string& group) {
    if (levels.empty()) {
        return Error(ErrorCode::InvalidArgument, "a level of detail assembly needs at least one level");
    }
    const UsdStageRefPtr finest = UsdStage::Open(levels.front().path.string());
    if (!finest) {
        return Error::make(ErrorCode::IoFailure, "'{}': cannot open the finest level", levels.front().path.string());
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    const UsdStageRefPtr stage = UsdStage::CreateNew(path.string());
    if (!stage) {
        return Error::make(ErrorCode::IoFailure, "cannot create '{}'", path.string());
    }
    UsdGeomSetStageUpAxis(stage, UsdGeomGetStageUpAxis(finest));
    UsdGeomSetStageMetersPerUnit(stage, UsdGeomGetStageMetersPerUnit(finest));
    if (finest->HasAuthoredTimeCodeRange()) {
        stage->SetStartTimeCode(finest->GetStartTimeCode());
        stage->SetEndTimeCode(finest->GetEndTimeCode());
    }
    stage->SetTimeCodesPerSecond(finest->GetTimeCodesPerSecond());
    UsdGeomXform::Define(stage, SdfPath("/World"));
    stage->SetDefaultPrim(stage->GetPrimAtPath(SdfPath("/World")));
    const std::filesystem::path directory = path.parent_path();
    for (size_t level = 0; level < levels.size(); ++level) {
        const SdfPath at(level == 0 ? "/World/Splats" : "/World/Splats_lod" + std::to_string(level));
        UsdPrim prim = stage->DefinePrim(at);
        // Relative where the level lies beside the assembly, as a conversion
        // writes it, so the files travel together.
        std::filesystem::path file = levels[level].path;
        if (file.parent_path() == directory || file.parent_path().empty()) {
            file = std::filesystem::path(".") / file.filename();
        }
        prim.GetReferences().AddReference(file.generic_string(), SdfPath("/World/Splats"));
        // The level names the Skeleton it was converted from, which is not on
        // this stage and lies outside what the reference brings: said empty
        // here, or every read of the binding warns. The joints travel in the
        // level's own cache, which is what a cloud is posed by.
        prim.CreateRelationship(TfToken("skel:skeleton")).SetTargets({});
        UsdGeomPrimvarsAPI primvars(prim);
        primvars.CreatePrimvar(TfToken("athenea:lod:group"), SdfValueTypeNames->String, UsdGeomTokens->constant)
            .Set(group);
        primvars.CreatePrimvar(TfToken("athenea:lod:cell"), SdfValueTypeNames->Float, UsdGeomTokens->constant)
            .Set(static_cast<float>(levels[level].cell));
    }
    if (!stage->GetRootLayer()->Save()) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}'", path.string());
    }
    return ok();
}

}   // namespace athenea::usd
