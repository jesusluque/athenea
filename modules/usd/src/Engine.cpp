// Copyright (c) 2026 jesus luque.
#include "Engine.h"
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/matrix3f.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <format>

#include <MaterialXFormat/File.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>

#include "athenea/core/Hash.h"
#include "athenea/core/Log.h"
#include "athenea/core/Platform.h"
#include "athenea/gpu/CommandBatch.h"

namespace athenea::usd {

std::unique_ptr<Engine> Engine::create(std::string& why) { return create(nullptr, why); }

std::unique_ptr<Engine> Engine::create(std::shared_ptr<gpu::Device> device, std::string& why) {
    if (device == nullptr) {
        auto opened = gpu::Device::create();
        if (!opened) {
            why = opened.error().toString();
            return nullptr;
        }
        device = *opened;
    }
    auto engine = std::unique_ptr<Engine>(new Engine());
    engine->device_ = std::move(device);
    engine->library_ = std::make_unique<gpu::ShaderLibrary>(engine->device_);
    auto loader = scene::CloudLoader::create(*engine->library_);
    if (!loader) {
        why = loader.error().toString();
        return nullptr;
    }
    engine->loader_.emplace(std::move(*loader));
    auto rasterizer = render::TileRasterizer::create(*engine->library_);
    if (!rasterizer) {
        why = rasterizer.error().toString();
        return nullptr;
    }
    engine->rasterizer_.emplace(std::move(*rasterizer));
    if (engine->device_->caps().rasterization) {
        if (auto points = render::PointRasterizer::create(*engine->library_)) {
            engine->pointRasterizer_.emplace(std::move(*points));
        } else {
            log::warn("hdAthenea: points will be drawn as discs: {}", points.error().toString());
        }
    }
    return engine;
}

namespace {

/// The most a splat of a streamed asset holds on the device: its position and
/// shape (two float4), degree-3 harmonics in f16 (23 words), a shading normal
/// and an emission word (lod/Athc.h).
constexpr uint64_t kStreamBytesPerSplat = 16 + 16 + 92 + 4 + 4;

/// What one gaussian's splat shadows hold on the hardware route: its proxy
/// (twelve float4 vertices and sixty indices, GaussianRayTracer::buildHardware),
/// and its twenty triangles' share of the BLAS, taken at 64 bytes a triangle
/// -- an estimate; the proxies alone are the measured part (~2.5 GB for
/// 5.9 M gaussians, docs/decisions.md).
constexpr uint64_t kShadowBytesPerGaussian = 12 * 16 + 60 * 4 + 20 * 64;

constexpr uint64_t kNoBudget = ~uint64_t{0};

uint64_t gaussiansOf(std::span<const render::SplatInstance> splats) {
    uint64_t total = 0;
    for (const render::SplatInstance& instance : splats) {
        total += instance.splats != nullptr ? instance.splats->count : 0;
    }
    return total;
}

/// A size for a message, in MiB, rounded up.
uint64_t mib(uint64_t bytes) {
    return (bytes + (uint64_t{1} << 20) - 1) >> 20;
}

}   // namespace

std::string Engine::relieveMemory() {
    // The failure being handled is the one already reported: drained, and
    // whatever else it took with it, so the retry starts clean.
    device_->releaseFreed();
    (void)device_->takeQueueError();
    const uint64_t before = device_->memoryInUse();
    // What is made again on demand.
    if (rasterizer_.has_value()) {
        rasterizer_->releaseScratch();
    }
    shadowTracer_.reset();
    splatShadowScene_.reset();
    shadowTracerReady_ = false;
    shadowGaussians_ = 0;
    rayTracer_.reset();
    denoiser_.reset();
    pathAuxValid_ = false;
    revision_.fetch_add(1);
    device_->releaseFreed();
    (void)device_->takeQueueError();
    std::string did = "the frame's scratch buffers given back";
    // And one thing more.
    constexpr uint32_t kMaxLodBias = 4;
    if (splatShadows_.load() && !memoryNoSplatShadows_.load()) {
        memoryNoSplatShadows_.store(true);
        did += "; splat shadows off";
    } else if (lodBias_ < kMaxLodBias) {
        ++lodBias_;
        did += std::format("; levels of detail {} coarser than asked", lodBias_);
        // Every streamed asset opened again, at the budget this level allows.
        const std::lock_guard<std::mutex> held(guard_);
        for (auto& [id, entry] : splats_) {
            if (!entry.asset.path.empty() && !entry.assetPending.has_value()) {
                entry.assetPending = entry.asset;
                entry.asset = StreamedAsset{};
                entry.lodCloud.reset();
                entry.pool.reset();
            }
        }
    } else {
        return {};
    }
    const uint64_t after = device_->memoryInUse();
    if (before != 0 || after != 0) {
        did += std::format(" ({} -> {} MiB in use)", mib(before), mib(after));
    }
    ++reliefs_;
    lastRelief_ = did;
    return did;
}

uint64_t Engine::streamBudget(const pxr::SdfPath& id, const StreamedAsset& asset) const {
    const uint64_t available = device_->memoryAvailable();
    uint64_t wanted = asset.budget;
    if (wanted == 0) {
        // Read whole, unless the file would take more than half of what the
        // budget has left: then streamed, which a cut can live within.
        std::error_code failed;
        const uint64_t fileBytes = std::filesystem::file_size(asset.path, failed);
        if (failed || ((available == kNoBudget || fileBytes <= available / 2) && lodBias_ == 0)) {
            return 0;
        }
        wanted = std::max<uint64_t>(fileBytes / kStreamBytesPerSplat, 1);
    }
    wanted = std::max<uint64_t>(wanted >> lodBias_, 1);
    if (available != kNoBudget) {
        const uint64_t fits = std::max<uint64_t>(available / 2 / kStreamBytesPerSplat, 1);
        wanted = std::min(wanted, fits);
    }
    if (wanted != asset.budget) {
        log::info("hdAthenea: {}: streaming budget {} splats (~{} MiB; asked {}, {} MiB of the GPU's budget left{})",
                  id.GetString(), wanted, mib(wanted * kStreamBytesPerSplat),
                  asset.budget == 0 ? std::string("the whole file") : std::to_string(asset.budget),
                  available == kNoBudget ? std::string("unknown") : std::to_string(mib(available)),
                  lodBias_ > 0 ? std::format(", {} levels coarser for memory", lodBias_) : std::string());
    }
    return wanted;
}

bool Engine::splatShadowsFit(uint64_t gaussians) {
    if (memoryNoSplatShadows_.load()) {
        return false;
    }
    if (gaussians == shadowGaussians_) {
        return true;   // built already, and counted in what is in use
    }
    uint64_t available = device_->memoryAvailable();
    if (available == kNoBudget) {
        shadowGaussians_ = gaussians;
        return true;
    }
    // What the tracer holds now goes when it is built again.
    available += shadowGaussians_ * kShadowBytesPerGaussian;
    const uint64_t need = gaussians * kShadowBytesPerGaussian;
    if (need <= available / 2) {
        shadowGaussians_ = gaussians;
        return true;
    }
    memoryNoSplatShadows_.store(true);
    shadowTracer_.reset();
    splatShadowScene_.reset();
    shadowTracerReady_ = false;
    shadowGaussians_ = 0;
    lastRelief_ = std::format("splat shadows skipped: the proxies of {} gaussians need ~{} MiB, more than half of "
                              "the {} MiB the GPU's {} MiB budget has left",
                              gaussians, mib(need), mib(available), mib(device_->memoryBudget()));
    ++reliefs_;
    log::warn("hdAthenea: {}", lastRelief_);
    return false;
}

void Engine::setSplats(const pxr::SdfPath& id, std::optional<ParticleFieldArrays> raw,
                       const render::Mat4* transform, std::optional<bool> visible,
                       std::optional<render::SplatEdit> edit, std::optional<StreamedAsset> asset,
                       std::optional<bool> relight,
                       std::optional<std::vector<pxr::TfToken>> categories, std::optional<bool> litBody,
                       std::optional<float> ior, std::optional<render::Mat4> transformStep,
                       std::optional<bool> catcher) {
    const std::lock_guard<std::mutex> held(guard_);
    SplatEntry& entry = splats_[id];
    if (transformStep) {
        entry.transformStep = transformStep->m;
    }
    if (asset.has_value()) {
        entry.assetPending = std::move(asset);
    }
    if (edit.has_value()) {
        entry.edit = *edit;
    }
    if (relight) {
        entry.relight = *relight;
    }
    if (litBody) {
        entry.litBody = *litBody;
    }
    if (catcher) {
        entry.catcher = *catcher;
    }
    if (ior) {
        entry.ior = *ior;
    }
    if (categories) {
        entry.categories = std::move(*categories);
    }
    if (raw.has_value()) {
        entry.pending = std::move(raw);
    }
    if (transform != nullptr) {
        entry.objectToWorld = *transform;
    }
    if (visible.has_value()) {
        entry.visible = *visible;
    }
}

void Engine::setPoints(const pxr::SdfPath& id, std::optional<PointsArrays> raw,
                       const render::Mat4* transform, std::optional<bool> visible,
                       std::optional<render::PointStyle> style) {
    const std::lock_guard<std::mutex> held(guard_);
    PointsEntry& entry = points_[id];
    if (raw.has_value()) {
        entry.pending = std::move(raw);
    }
    if (transform != nullptr) {
        entry.objectToWorld = *transform;
    }
    if (visible.has_value()) {
        entry.visible = *visible;
    }
    if (style.has_value()) {
        entry.style = *style;
    }
}

namespace {

/// An instancer chain composed on the device at the frame, and -- where an
/// instancer in it moves under the shutter -- at the samples about it: each
/// level from its own samples where it has them, the frame's arrays where it
/// does not. `start` and `end` are left empty when nothing in the chain moves;
/// the times are the first moving level's.
Result<void> composeChains(world::Instancing& instancing, const std::map<pxr::SdfPath, InstancerEntry>& instancers,
                           const std::vector<InstancerLink>& links, world::InstanceChain& chain,
                           world::InstanceChain& start, world::InstanceChain& end, double& timeStart,
                           double& timeEnd) {
    enum class At { Frame, Start, End };
    bool anyMoves = false;
    const auto compose = [&](At at) -> Result<world::InstanceChain> {
        std::vector<world::InstancerLevel> levels;
        for (const InstancerLink& link : links) {
            const InstancerArrays& a = instancers.at(link.instancer).arrays;
            const InstancerSample* sample = at == At::Start && a.start.has_value() ? &*a.start
                                            : at == At::End && a.end.has_value()   ? &*a.end
                                                                                    : nullptr;
            if (at == At::Frame && a.start.has_value() && a.end.has_value() && !anyMoves) {
                anyMoves = true;
                timeStart = a.timeStart;
                timeEnd = a.timeEnd;
            }
            world::InstancerLevel level;
            level.indices = std::span<const int32_t>(link.indices.cdata(), link.indices.size());
            level.translations = streamOf(sample != nullptr ? sample->translations : a.translations);
            level.rotations = streamOf(sample != nullptr ? sample->rotations : a.rotations);
            level.scales = streamOf(sample != nullptr ? sample->scales : a.scales);
            level.transforms = streamOf(sample != nullptr ? sample->transforms : a.transforms);
            level.instancerTransform = sample != nullptr ? sample->instancerTransform : a.instancerTransform;
            levels.push_back(level);
        }
        return instancing.compose(levels);
    };
    auto frame = compose(At::Frame);
    if (!frame) return std::move(frame).error();
    chain = std::move(*frame);
    start = {};
    end = {};
    if (anyMoves) {
        auto atStart = compose(At::Start);
        if (!atStart) return std::move(atStart).error();
        auto atEnd = compose(At::End);
        if (!atEnd) return std::move(atEnd).error();
        if (atStart->count == chain.count && atEnd->count == chain.count) {
            start = std::move(*atStart);
            end = std::move(*atEnd);
        }
    }
    return ok();
}

std::array<float, 16> matrixOf(const pxr::VtValue& value) {
    std::array<float, 16> out{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    if (value.IsHolding<pxr::GfMatrix4f>()) {
        const pxr::GfMatrix4f& m = value.UncheckedGet<pxr::GfMatrix4f>();
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out[static_cast<size_t>(r * 4 + c)] = m[r][c];
    } else if (value.IsHolding<pxr::GfMatrix4d>()) {
        const pxr::GfMatrix4d& m = value.UncheckedGet<pxr::GfMatrix4d>();
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out[static_cast<size_t>(r * 4 + c)] = static_cast<float>(m[r][c]);
    }
    return out;
}

template <typename Array>
std::span<const std::byte> bytesOf(const pxr::VtValue& value) {
    if (!value.IsHolding<Array>()) {
        return {};
    }
    const Array& array = value.UncheckedGet<Array>();
    return {reinterpret_cast<const std::byte*>(array.cdata()), array.size() * sizeof(typename Array::value_type)};
}

/// The skinner's input over Hydra's values: byte views, nothing converted
/// but the three matrices (and a double matrix to float, as Storm's kernel
/// casts them).
geom::SkinningInput skinningInputOf(const SkinningArrays& s, const scene::FloatStream& rest) {
    geom::SkinningInput in;
    in.points = static_cast<uint32_t>(rest.values() / 3);
    in.restPoints = rest.bytes;
    in.blendShapeOffsets = bytesOf<pxr::VtVec4fArray>(s.blendShapeOffsets);
    in.blendShapeOffsetRanges = bytesOf<pxr::VtVec2iArray>(s.blendShapeOffsetRanges);
    in.blendShapeWeights = bytesOf<pxr::VtFloatArray>(s.blendShapeWeights);
    in.influences = bytesOf<pxr::VtVec2fArray>(s.influences);
    in.numInfluencesPerPoint = static_cast<uint32_t>(std::max(s.numInfluencesPerComponent, 0));
    in.constantInfluences = s.hasConstantInfluences;
    in.method = s.dualQuaternion ? geom::SkinningMethod::DualQuaternion : geom::SkinningMethod::LinearBlend;
    in.skinningXforms = bytesOf<pxr::VtMatrix4fArray>(s.skinningXforms);
    in.skinningDualQuats = bytesOf<pxr::VtVec4fArray>(s.skinningDualQuats);
    if (in.skinningDualQuats.empty()) {
        in.skinningDualQuats = bytesOf<pxr::VtQuatfArray>(s.skinningDualQuats);
    }
    in.skinningScaleXforms = bytesOf<pxr::VtMatrix3fArray>(s.skinningScaleXforms);
    in.geomBindXform = matrixOf(s.geomBindXform);
    in.skelLocalToWorld = matrixOf(s.skelLocalToWorld);
    in.primWorldToLocal = matrixOf(s.primWorldToLocal);
    return in;
}

}   // namespace

void Engine::setCurves(const pxr::SdfPath& id, int32_t primId, const pxr::TfToken& renderTag,
                       std::optional<CurveArrays> arrays, const render::Mat4* transform, std::optional<bool> visible,
                       std::optional<MeshLook> look) {
    const std::lock_guard<std::mutex> held(guard_);
    MeshEntry& entry = meshes_[id];
    entry.primId = static_cast<uint32_t>(primId);
    entry.renderTag = renderTag;
    if (arrays.has_value()) {
        entry.pendingCurves = std::move(arrays);
    }
    if (transform != nullptr) {
        entry.objectToWorld = *transform;
    }
    if (visible.has_value()) {
        entry.visible = *visible;
    }
    if (look.has_value()) {
        entry.look = *look;
    }
    revision_.fetch_add(1);
}

void Engine::setMesh(const pxr::SdfPath& id, int32_t primId, const pxr::TfToken& renderTag,
                     std::optional<MeshArrays> arrays, const render::Mat4* transform, std::optional<bool> visible,
                     std::optional<MeshLook> look, std::optional<std::vector<InstancerLink>> instancing,
                     std::optional<MeshTransforms> shutter) {
    const std::lock_guard<std::mutex> held(guard_);
    MeshEntry& entry = meshes_[id];
    if (shutter.has_value()) {
        entry.shutter = *shutter;
    }
    if (arrays.has_value()) {
        entry.pendingStart = arrays->pointsStart;
        entry.pendingEnd = arrays->pointsEnd;
        entry.pointsTimeStart = arrays->pointsTimeStart;
        entry.pointsTimeEnd = arrays->pointsTimeEnd;
    }
    if (instancing.has_value()) {
        entry.instancing = std::move(*instancing);
        entry.chainDirty = true;
    }
    entry.primId = static_cast<uint32_t>(primId);
    entry.renderTag = renderTag;
    if (arrays.has_value()) {
        entry.subsetMaterials.clear();
        for (const MeshSubset& subset : arrays->subsets) {
            entry.subsetMaterials.push_back(subset.material);
        }
        entry.pending = std::move(arrays);
    }
    if (transform != nullptr) {
        entry.objectToWorld = *transform;
    }
    if (visible.has_value()) {
        entry.visible = *visible;
    }
    if (look.has_value()) {
        entry.look = *look;
    }
}

void Engine::setInstancer(const pxr::SdfPath& id, const pxr::SdfPath& parent, InstancerArrays arrays) {
    const std::lock_guard<std::mutex> held(guard_);
    InstancerEntry& entry = instancers_[id];
    entry.arrays = std::move(arrays);
    entry.parent = parent;
    entry.version = ++instancerVersion_;
}

void Engine::removeInstancer(const pxr::SdfPath& id) {
    const std::lock_guard<std::mutex> held(guard_);
    instancers_.erase(id);
}

void Engine::setMaterial(const pxr::SdfPath& id, std::shared_ptr<void> mtlxDocument) {
    const std::lock_guard<std::mutex> held(guard_);
    MaterialEntry& entry = materials_[id];
    entry.document = std::move(mtlxDocument);
    entry.cutout = material::MaterialCompiler::cutsOut(entry.document);
    entry.transparent = material::MaterialCompiler::transparentOpacity(entry.document);
    entry.volume = material::MaterialCompiler::volumeCoefficients(entry.document);
    entry.pending = true;
}

Result<void> Engine::updateExternalTexture(const std::string& name, const gpu::Buffer& rgba, uint32_t width,
                                           uint32_t height, uint32_t rowPixels) {
    if (!textures_) {
        auto made = material::TextureStore::create(*library_);
        if (!made) return std::move(made).error();
        textures_ = std::move(*made);
        textures_->setSearchPath(assetSearchPath_);
    }
    return textures_->updateExternal(name, rgba, width, height, rowPixels);
}

void Engine::setLight(const pxr::SdfPath& id, const light::Light& lamp, std::vector<InstancerLink> instancing) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    LightEntry& entry = lights_[id];
    entry.lamp = lamp;
    // Sync gives the chain whole each time; a change is any link differing.
    bool same = entry.instancing.size() == instancing.size();
    for (size_t i = 0; same && i < instancing.size(); ++i) {
        same = entry.instancing[i].instancer == instancing[i].instancer && entry.instancing[i].indices == instancing[i].indices;
    }
    if (!same) {
        entry.instancing = std::move(instancing);
        entry.chainDirty = true;
    }
}

uint32_t Engine::categoryBit(const std::string& name) {
    if (name.empty()) {
        return light::kLightUnlinked;
    }
    const auto found = categoryBits_.find(name);
    if (found != categoryBits_.end()) {
        return found->second;
    }
    if (categoryBits_.size() >= 64) {
        athenea::log::warn("hdAthenea: more than 64 light linking categories; '{}' links to nothing", name);
        return light::kLightUnlinked;
    }
    const uint32_t bit = static_cast<uint32_t>(categoryBits_.size());
    categoryBits_.emplace(name, bit);
    return bit;
}

uint64_t Engine::categoryMask(const std::vector<pxr::TfToken>& names) {
    uint64_t mask = 0;
    for (const pxr::TfToken& name : names) {
        const uint32_t bit = categoryBit(name.GetString());
        if (bit != light::kLightUnlinked) {
            mask |= uint64_t{1} << bit;
        }
    }
    return mask;
}

// These four change what a path finds, so a mean gathered under the old ones
// is not the same mean; each raises the revision and the accumulation starts
// again. setPathTotal does not: moving the finish line leaves what has been
// gathered still valid.
void Engine::setLightSamples(uint32_t samples) {
    if (lightSamples_.exchange(std::max(samples, 1u)) != std::max(samples, 1u)) {
        revision_.fetch_add(1);
    }
}

void Engine::setChooseLights(bool choose) {
    if (chooseLights_.exchange(choose) != choose) {
        revision_.fetch_add(1);
    }
}

void Engine::setSplatTransferIndirect(bool indirect) {
    if (transferIndirect_.exchange(indirect) != indirect) {
        revision_.fetch_add(1);
    }
}

/// THE TABLE, UPLOADED WHEN IT CHANGED AND NOT BEFORE.
///
/// Two float4 a row, as `athenea/common/splat_override.slang` reads them: the id
/// bit-cast into the first float, so the host writes the layout without a
/// struct for the two sides to agree on. Grown, never shrunk.
Result<void> Engine::commitSplatOverrides() {
    std::vector<render::SplatOverride> rows;
    {
        const std::lock_guard<std::mutex> held(guard_);
        if (!splatOverridesDirty_) {
            return ok();
        }
        rows = splatOverrides_;
        splatOverridesDirty_ = false;
    }
    splatOverrideRows_ = static_cast<uint32_t>(rows.size());
    if (rows.empty()) {
        return ok();
    }
    std::vector<float> packed(rows.size() * 8, 0.0F);
    for (size_t k = 0; k < rows.size(); ++k) {
        float id = 0.0F;
        std::memcpy(&id, &rows[k].id, sizeof(float));
        packed[k * 8 + 0] = id;
        packed[k * 8 + 1] = rows[k].metallic;
        packed[k * 8 + 2] = rows[k].roughness;
        packed[k * 8 + 3] = rows[k].transmission;
        packed[k * 8 + 4] = rows[k].tint[0];
        packed[k * 8 + 5] = rows[k].tint[1];
        packed[k * 8 + 6] = rows[k].tint[2];
        packed[k * 8 + 7] = rows[k].replaceColour ? 1.0F : 0.0F;
    }
    const uint64_t bytes = packed.size() * sizeof(float);
    if (!splatOverrideBuffer_.valid() || splatOverrideBuffer_.bytes() < bytes) {
        gpu::BufferDesc desc;
        desc.bytes = bytes;
        desc.elementBytes = 16;
        desc.label = "splat.overrides";
        auto made = gpu::Buffer::create(*device_, desc, packed.data());
        if (!made) return std::move(made).error();
        splatOverrideBuffer_ = std::move(*made);
        return ok();
    }
    ATHENEA_TRY(splatOverrideBuffer_.write(*device_, 0, bytes, packed.data()));
    return ok();
}

void Engine::setSplatOverride(const render::SplatOverride& said) {
    const std::lock_guard<std::mutex> held(guard_);
    for (render::SplatOverride& row : splatOverrides_) {
        if (row.id == said.id) {
            row = said;
            splatOverridesDirty_ = true;
            return;
        }
    }
    splatOverrides_.push_back(said);
    splatOverridesDirty_ = true;
}

void Engine::clearSplatOverride(uint32_t id) {
    const std::lock_guard<std::mutex> held(guard_);
    const auto was = splatOverrides_.size();
    splatOverrides_.erase(std::remove_if(splatOverrides_.begin(), splatOverrides_.end(),
                                         [id](const render::SplatOverride& row) { return row.id == id; }),
                          splatOverrides_.end());
    splatOverridesDirty_ = splatOverridesDirty_ || splatOverrides_.size() != was;
}

void Engine::clearSplatOverrides() {
    const std::lock_guard<std::mutex> held(guard_);
    if (!splatOverrides_.empty()) {
        splatOverrides_.clear();
        splatOverridesDirty_ = true;
    }
}

std::vector<render::SplatOverride> Engine::splatOverrides() const {
    const std::lock_guard<std::mutex> held(guard_);
    return splatOverrides_;
}

/// Over every cloud in the scene, since an id belongs to the frame and a prim
/// may have been converted into more than one of them.
Result<render::SplatIdReading> Engine::measureSplatId(uint32_t id) {
    std::vector<const scene::GpuSplats*> clouds;
    {
        const std::lock_guard<std::mutex> held(guard_);
        for (const auto& [path, entry] : splats_) {
            if (entry.gpu != nullptr) {
                clouds.push_back(entry.posed != nullptr ? entry.posed.get() : entry.gpu.get());
            }
        }
    }
    render::SplatIdReading total;
    std::array<double, 3> sum{};
    for (const scene::GpuSplats* cloud : clouds) {
        auto one = render::measureSplatId(*library_, *cloud, id);
        if (!one) return std::move(one).error();
        if (one->count == 0) {
            continue;
        }
        for (uint32_t c = 0; c < 3; ++c) {
            sum[c] += double(one->mean[c]) * double(one->count);
            total.low[c] = total.count == 0 ? one->low[c] : std::min(total.low[c], one->low[c]);
            total.high[c] = std::max(total.high[c], one->high[c]);
        }
        total.count += one->count;
        total.hasPbr = total.hasPbr || one->hasPbr;
    }
    if (total.count > 0) {
        for (uint32_t c = 0; c < 3; ++c) {
            total.mean[c] = static_cast<float>(sum[c] / double(total.count));
        }
    }
    return total;
}

void Engine::setSplatReflections(bool reflect) {
    if (splatReflections_.exchange(reflect) != reflect) {
        revision_.fetch_add(1);
    }
}

void Engine::setSplatShadows(bool shadows) {
    if (splatShadows_.exchange(shadows) != shadows) {
        revision_.fetch_add(1);
        // Asked for again: tried again, whatever memory said last time.
        memoryNoSplatShadows_.store(false);
    }
}

void Engine::setCloudShadows(bool shadows) {
    cloudShadows_.store(shadows);
}

void Engine::setDomePrefiltered(bool prefiltered) {
    domePrefiltered_.store(prefiltered);
}

void Engine::setCloudShadowResolution(uint32_t texels) {
    cloudShadowTexels_.store(std::clamp(texels, 64u, 8192u));
}

void Engine::setCloudShadowTerms(uint32_t terms) {
    cloudShadowTerms_.store(std::min(terms, 7u));
}

void Engine::setAssetSearchPath(const std::filesystem::path& directory) {
    assetSearchPath_ = directory;
    if (textures_) {
        textures_->setSearchPath(directory);
    }
}

void Engine::setAntialias(bool on) {
    antialias_.store(on);
}

void Engine::setCloudShadowDensity(float density) {
    cloudShadowDensity_.store(std::clamp(density, 0.0F, 16.0F));
}

void Engine::setPathSamples(uint32_t samples) {
    if (pathSamples_.exchange(std::max(samples, 1u)) != std::max(samples, 1u)) {
        revision_.fetch_add(1);
    }
}

void Engine::setPathBounces(uint32_t bounces) {
    if (pathBounces_.exchange(bounces) != bounces) {
        revision_.fetch_add(1);
    }
}

void Engine::setShutter(double open, double close) {
    const bool changed = shutterOpen_.exchange(open) != open || shutterClose_.exchange(close) != close;
    if (changed) {
        revision_.fetch_add(1);
        shutterSettle_.store(2);   // this frame, and the one that resamples
    }
}

void Engine::setMotionBuckets(uint32_t buckets) {
    const uint32_t clamped = std::min(std::max(buckets, 1u), 8u);
    if (motionBuckets_.exchange(clamped) != clamped) {
        revision_.fetch_add(1);
    }
}

void Engine::setPathTotal(uint32_t total) { pathTotal_.store(std::max(total, 1u)); }

void Engine::setDenoise(bool denoise) { denoise_.store(denoise); }

void Engine::setPathAdaptive(bool adaptive) { pathAdaptive_.store(adaptive); }
void Engine::setPathMis(bool mis) { pathMis_.store(mis); }

void Engine::setPathError(float error) { pathError_.store(std::max(error, 1.0e-4F)); }

uint32_t Engine::pathAccumulated() const noexcept {
    return pathTracer_.has_value() && pathState_.traced ? pathTracer_->accumulated() : 0;
}

bool Engine::pathConverged() const noexcept {
    if (shutterSettle_.load() > 0) {
        return false;   // the prims are still being resampled about it
    }
    if (!pathState_.traced) {
        return true;   // nothing being gathered
    }
    if (pathAccumulated() >= pathTotal_.load()) {
        return true;
    }
    return pathState_.adaptive && pathProgress_.covered > 0 && pathProgress_.converged == pathProgress_.covered;
}

std::vector<CoordSysBinding> Engine::coordSysOf(const pxr::SdfPath& id) const {
    const std::lock_guard<std::mutex> held(guard_);
    const auto found = meshes_.find(id);
    return found != meshes_.end() ? found->second.look.coordSys : std::vector<CoordSysBinding>{};
}

uint64_t Engine::meshGeneration() const noexcept {
    return scene_.has_value() ? scene_->generation() : 0;
}

uint64_t Engine::meshPositionsRevision() const noexcept {
    return scene_.has_value() ? scene_->positionsRevision() : 0;
}

void Engine::removeLight(const pxr::SdfPath& id) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    lights_.erase(id);
}

void Engine::removeMaterial(const pxr::SdfPath& id) {
    const std::lock_guard<std::mutex> held(guard_);
    materials_.erase(id);
    materialsChanged_ = true;
}

void Engine::setVolume(const pxr::SdfPath& id, VolumeArrays arrays) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    volumes_[id] = std::move(arrays);
    ++volumesVersion_;
}

void Engine::removeVolume(const pxr::SdfPath& id) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    volumes_.erase(id);
    ++volumesVersion_;
}

void Engine::setVolumeField(const pxr::SdfPath& id, VolumeFieldAsset asset) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    volumeFields_[id] = std::move(asset);
    ++volumesVersion_;
}

void Engine::removeVolumeField(const pxr::SdfPath& id) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    volumeFields_.erase(id);
    ++volumesVersion_;
}

void Engine::remove(const pxr::SdfPath& id) {
    const std::lock_guard<std::mutex> held(guard_);
    splats_.erase(id);
    points_.erase(id);
    meshes_.erase(id);
}

/// WHICH LEVEL OF EACH CLOUD THIS VIEW DRAWS. A cloud converted at several
/// cells is several prims of one `athenea:lod:group`; the one drawn is the
/// coarsest whose cell, at the nearest point of its bounds, spans no more than
/// its threshold in pixels -- the finest where none is that fine. The others
/// are not drawn, and cost the frame nothing but their memory.
namespace {

/// Device memory a cloud's arrays hold: every buffer it carries.
uint64_t bytesOf(const scene::GpuSplats& cloud) {
    uint64_t n = 0;
    for (const gpu::Buffer* b : {&cloud.positions, &cloud.shape, &cloud.sh, &cloud.pbr, &cloud.lobes, &cloud.crypto, &cloud.transfer,
                                 &cloud.shadowBits, &cloud.origin, &cloud.normals, &cloud.emission,
                                 &cloud.visibilityParts, &cloud.visibilityTexels, &cloud.visibilityPartOf,
                                 &cloud.visibilityAmbient}) {
        n += b->valid() ? b->bytes() : 0;
    }
    return n;
}

/// And a level of detail's: the store, its merged levels and its tables.
uint64_t bytesOf(const lod::LodCloud& cloud) {
    uint64_t n = bytesOf(cloud.splats);
    for (const gpu::Buffer* b : {&cloud.groups, &cloud.resident, &cloud.starts, &cloud.order}) {
        n += b->valid() ? b->bytes() : 0;
    }
    for (const lod::LodLevel& level : cloud.levels) {
        n += bytesOf(level.gaussians) + (level.cells.valid() ? level.cells.bytes() : 0);
    }
    return n;
}

}   // namespace

void Engine::noteGaussians(const std::string& route, std::span<const render::SplatInstance> splats,
                           std::span<const std::string> prims, std::span<const lod::CutStats> cutStats,
                           std::span<const std::string> cutPrims, const std::set<const SplatEntry*>& levels) {
    GaussianStats g;
    g.frame = frameSerial_;
    g.route = route;
    for (const render::SplatInstance& instance : splats) {
        g.submitted += instance.splats != nullptr ? instance.splats->count : 0u;
    }
    // A variant level's place among its group's, finest first, for the words.
    std::map<std::string, std::vector<std::pair<float, const SplatEntry*>>> groups;
    const std::lock_guard<std::mutex> held(guard_);
    for (const auto& [id, entry] : splats_) {
        if (!entry.lodGroup.empty() && entry.gpu != nullptr) {
            groups[entry.lodGroup].push_back({entry.lodCell, &entry});
        }
    }
    for (auto& [name, members] : groups) {
        std::sort(members.begin(), members.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    for (const auto& [id, entry] : splats_) {
        GaussianCloudStats c;
        c.prim = id.GetString();
        const lod::LodCloud* lodCloud = entry.pool != nullptr ? &entry.pool->cloud() : entry.lodCloud.get();
        const scene::GpuSplats* data = entry.gpu != nullptr ? entry.gpu.get()
                                       : lodCloud != nullptr ? &lodCloud->splats
                                                             : nullptr;
        if (data == nullptr) {
            continue;   // synced, not yet uploaded
        }
        c.gaussians = entry.gpu != nullptr ? entry.gpu->count : lodCloud->count;
        c.shDegree = data->degree();
        c.linear = data->linear;
        c.relit = entry.relight;
        c.litBody = entry.litBody;
        c.transfer = data->hasTransfer() ? data->transferCount : 0u;
        c.skinned = entry.posed != nullptr;
        c.normals = data->hasNormals();
        c.emission = data->hasEmission();
        c.pbr = data->hasPbr();
        c.crypto = data->hasCrypto();
        c.visibility = data->hasVisibility();
        c.ior = entry.ior;
        if (entry.gpu != nullptr) {
            c.bytes += bytesOf(*entry.gpu);
        }
        if (entry.posed != nullptr) {
            c.bytes += bytesOf(*entry.posed);
        }
        for (const gpu::Buffer* b : {&entry.influences, &entry.xforms, &entry.xformsEnd, &entry.motion}) {
            c.bytes += b->valid() ? b->bytes() : 0;
        }
        g.cloudBytes += c.bytes;
        if (lodCloud != nullptr) {
            g.poolBytes += bytesOf(*lodCloud);
        }
        if (!entry.lodGroup.empty()) {
            const auto& members = groups[entry.lodGroup];
            size_t at = 0;
            for (size_t k = 0; k < members.size(); ++k) {
                if (members[k].second == &entry) {
                    at = k;
                }
            }
            c.lod = "level " + std::to_string(at) + " of " + std::to_string(members.size()) + " in '" +
                    entry.lodGroup + "'";
        }
        c.drawn = entry.visible && (entry.lodGroup.empty() || levels.count(&entry) != 0);
        if (entry.pool != nullptr) {
            const lod::StreamingPool::Status status = entry.pool->status();
            c.streamed = true;
            c.chunks = lodCloud->chunks();
            c.chunksResident = status.resident;
            c.chunksMissing = status.missing;
            c.chunksInFlight = status.inFlight;
        }
        if (lodCloud != nullptr && entry.gpu == nullptr) {
            c.lod = "cut";
            for (size_t k = 0; k < cutPrims.size() && k < cutStats.size(); ++k) {
                if (cutPrims[k] == c.prim) {
                    c.lodOwn = cutStats[k].splats;
                    c.lodMerged = cutStats[k].merged;
                    c.chunksWanted = static_cast<uint32_t>(
                        std::count_if(cutStats[k].needs.begin(), cutStats[k].needs.end(),
                                      [](uint32_t need) { return need != 0; }));
                }
            }
        }
        for (size_t k = 0; k < prims.size() && k < splats.size(); ++k) {
            if (prims[k] == c.prim && splats[k].splats != nullptr) {
                c.submitted += splats[k].splats->count;
            }
        }
        g.inStage += c.gaussians;
        g.clouds.push_back(std::move(c));
    }
    gaussianStats_ = std::move(g);
    // The last counts the device finished, until newer ones arrive: they are
    // labelled with the frame they belong to.
    takeSplatCounters();
}

void Engine::takeSplatCounters() {
    if (rasterizer_.has_value()) {
        if (std::optional<render::SplatCounters> latest = rasterizer_->latestCounters();
            latest && (!lastCounted_ || latest->tag != lastCounted_->tag)) {
            const auto frame = std::find_if(countedFrames_.begin(), countedFrames_.end(),
                                            [&](const CountedFrame& f) { return f.tag == latest->tag; });
            lastCountedPrims_ = frame != countedFrames_.end() ? frame->prims : std::vector<std::string>();
            lastCounted_ = std::move(latest);
            // Frames older than the one counted will not be asked for again.
            countedFrames_.erase(countedFrames_.begin(), frame != countedFrames_.end() ? frame : countedFrames_.begin());
        }
    }
    if (!lastCounted_) {
        return;
    }
    GaussianStats& g = gaussianStats_;
    const render::SplatCounters& c = *lastCounted_;
    g.counted = true;
    g.countedFrame = c.tag;
    g.countedSlots = c.slots;
    g.visible = c.visible;
    g.pairs = c.pairs;
    g.maxTiles = c.maxTiles;
    for (size_t k = 0; k < g.culled.size() && k < c.culled.size(); ++k) {
        g.culled[k] = c.culled[k];
    }
    for (GaussianCloudStats& cloud : g.clouds) {
        cloud.counted = false;
        cloud.visible = 0;
        cloud.pairs = 0;
        for (size_t k = 0; k < c.clouds.size() && k < lastCountedPrims_.size(); ++k) {
            if (lastCountedPrims_[k] == cloud.prim) {
                cloud.counted = true;
                cloud.visible += c.clouds[k].visible;
                cloud.pairs += c.clouds[k].pairs;
            }
        }
    }
}

std::set<const SplatEntry*> Engine::lodLevelsFor(const render::Projection& projection) const {
    std::map<std::string, std::vector<const SplatEntry*>> groups;
    for (const auto& [id, entry] : splats_) {
        if (!entry.lodGroup.empty() && entry.visible && entry.gpu != nullptr && entry.lodCell > 0.0F) {
            groups[entry.lodGroup].push_back(&entry);
        }
    }
    std::set<const SplatEntry*> chosen;
    const double focal = std::max(std::abs(projection.focalX), 1e-9);
    for (auto& [name, members] : groups) {
        std::sort(members.begin(), members.end(),
                  [](const SplatEntry* a, const SplatEntry* b) { return a->lodCell < b->lodCell; });
        size_t pickAt = 0;
        for (size_t at = 0; at < members.size(); ++at) {
            const SplatEntry* level = members[at];
            const render::Mat4& m = level->objectToWorld;
            // Where it stands this frame: posed where a skeleton carries it.
            const scene::GpuSplats& cloud = level->posed != nullptr ? *level->posed : *level->gpu;
            // Where the cloud is nearest the eye, in its own space.
            const render::Vec3 eye = aofx::xform::inverseAffine(m).point(projection.eyeWorld);
            double near2 = 0.0;
            const double e[3] = {eye.x, eye.y, eye.z};
            for (int k = 0; k < 3; ++k) {
                const double below = double(cloud.bounds.min[k]) - e[k];
                const double above = e[k] - double(cloud.bounds.max[k]);
                const double out = std::max({below, above, 0.0});
                near2 += out * out;
            }
            const double scale = std::sqrt(m.at(0, 0) * m.at(0, 0) + m.at(1, 0) * m.at(1, 0) + m.at(2, 0) * m.at(2, 0));
            const double distance = std::max(std::sqrt(near2) * scale, projection.nearZ);
            const double pixels = double(level->lodCell) * scale * focal / distance;
            if (pixels <= double(level->lodThreshold)) {
                pickAt = at;   // coarser still fits: members go finest to coarsest
            }
        }
        // Short of memory (relieveMemory): as many levels coarser as given up.
        chosen.insert(members[std::min(pickAt + lodBias_, members.size() - 1)]);
    }
    return chosen;
}

Result<size_t> Engine::commit() {
    // How long the commit took, for ATHENEA_STAGES: every return passes here.
    struct Clock {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        double* into;
        ~Clock() {
            *into = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
    } clock{std::chrono::steady_clock::now(), &lastCommitMs_};
    const std::lock_guard<std::mutex> held(guard_);
    size_t uploaded = 0;
    for (auto& [id, entry] : materials_) {
        if (!entry.pending) {
            continue;
        }
        entry.pending = false;
        entry.compiled.reset();
        materialsChanged_ = true;
        ++uploaded;
        if (!entry.document) {
            continue;
        }
        if (!compiler_ && !compilerFailed_) {
            std::vector<std::filesystem::path> shaders;
            for (const std::string& path : device_->shaderSearchPaths()) {
                shaders.emplace_back(path);
            }
            // $ATHENEA_MATERIALX_ROOT: MaterialX's libraries of the
            // engine's own version, in place of the ones the host's USD
            // loaded -- a host whose MaterialX is older than the Slang
            // generator (Blender's 1.39.4) has no genslang implementations
            // and older node definitions. Read by this compiler alone, so
            // the host's own renderers keep theirs.
            const std::string ownRoot = platform::env("ATHENEA_MATERIALX_ROOT");
            Result<std::unique_ptr<material::MaterialCompiler>> made = Error(ErrorCode::NotFound, "");
            if (!ownRoot.empty()) {
                made = material::MaterialCompiler::create(std::vector<std::filesystem::path>{ownRoot}, shaders);
            } else {
                std::vector<std::filesystem::path> sources;
                for (const MaterialX::FilePath& path : pxr::HdMtlxSearchPaths()) {
                    sources.emplace_back(path.asString());
                    sources.emplace_back(std::filesystem::path(path.asString()).parent_path());
                }
                made = material::MaterialCompiler::create(pxr::HdMtlxStdLibraries(), sources, shaders);
            }
            if (made) {
                compiler_ = std::move(*made);
            } else {
                compilerFailed_ = true;
                log::warn("hdAthenea: materials show displayColor: {}", made.error().toString());
            }
        }
        if (!compiler_) {
            continue;
        }
        // A Material with a volume terminal and no surface: a Volume's
        // medium reads its coefficients (volumeCoefficients); there is no
        // surface in it to compile.
        if (entry.volume.has_value()) {
            entry.compiled.reset();
            continue;
        }
        auto compiled = compiler_->compileDocument(entry.document);
        if (compiled) {
            entry.compiled = std::move(*compiled);
        } else {
            log::warn("hdAthenea: material {}: {}", id.GetString(), compiled.error().toString());
        }
        // A cut-out is also asked by shadow rays, which want its opacity and
        // nothing else: compiled once more, the BSDF left out.
        entry.opacity.reset();
        if (entry.compiled && entry.cutout) {
            auto opacity = compiler_->compileDocument(entry.document, {}, material::ClosureVariant::Opacity);
            if (opacity) {
                entry.opacity = std::move(*opacity);
            } else {
                log::warn("hdAthenea: material {}: its opacity: {}", id.GetString(), opacity.error().toString());
            }
        }
    }
    for (auto& [id, entry] : meshes_) {
        if (entry.pendingCurves.has_value()) {
            const CurveArrays& c = *entry.pendingCurves;
            geom::CurveInput input;
            input.source = id.GetString();
            input.points = streamOf(c.points);
            input.curveVertexCounts = std::span<const int32_t>(c.curveVertexCounts.cdata(), c.curveVertexCounts.size());
            input.curveIndices = std::span<const int32_t>(c.curveIndices.cdata(), c.curveIndices.size());
            if (c.type == pxr::HdTokens->cubic) {
                input.basis = c.basis == pxr::HdTokens->bspline      ? geom::CurveBasis::BSpline
                              : c.basis == pxr::HdTokens->catmullRom ? geom::CurveBasis::CatmullRom
                                                                     : geom::CurveBasis::Bezier;
            } else {
                input.basis = geom::CurveBasis::Linear;
            }
            input.wrap = c.wrap == pxr::HdTokens->periodic ? geom::CurveWrap::Periodic : geom::CurveWrap::Nonperiodic;
            std::vector<float> widths;
            if (c.widths.IsHolding<pxr::VtFloatArray>()) {
                const auto& w = c.widths.UncheckedGet<pxr::VtFloatArray>();
                widths.assign(w.begin(), w.end());
            } else if (c.widths.IsHolding<float>()) {
                widths.push_back(c.widths.UncheckedGet<float>());
            }
            input.widths = widths;
            // HdInterpolation: constant 0, uniform 1, varying 2, vertex 3.
            input.widthInterpolation = c.widthsInterpolation == 1   ? geom::WidthInterpolation::Uniform
                                       : c.widthsInterpolation == 2 ? geom::WidthInterpolation::Varying
                                       : c.widthsInterpolation == 3 ? geom::WidthInterpolation::Vertex
                                                                    : geom::WidthInterpolation::Constant;
            if (widths.size() == 1) {
                input.widthInterpolation = geom::WidthInterpolation::Constant;
                input.width = widths.front();
            }
            std::vector<geom::PrimvarInput> primvars;
            for (const PrimvarArrays& p : c.primvars) {
                geom::PrimvarInput primvar;
                primvar.name = p.name;
                primvar.interpolation = static_cast<geom::Interpolation>(p.interpolation);
                primvar.values = primvarStreamOf(p.values, &primvar.components);
                primvars.push_back(std::move(primvar));
            }
            input.primvars = primvars;
            if (c.topologyChanged || entry.topologyKey == 0) {
                entry.topologyKey = ++nextTopologyKey_;
            }
            input.topology = entry.topologyKey;
            entry.gpu.reset();
            if (!curveBuilder_.has_value()) {
                auto made = geom::CurveBuilder::create(*library_);
                if (!made) return std::move(made).error();
                curveBuilder_.emplace(std::move(*made));
            }
            auto built = curveBuilder_->build(input);
            if (built) {
                entry.gpu = std::make_shared<const geom::GpuMesh>(std::move(built->mesh));
            } else {
                log::warn("hdAthenea: {}: {}", id.GetString(), built.error().toString());
            }
            entry.pendingCurves.reset();
            ++uploaded;
            continue;
        }
        if (!entry.pending.has_value()) {
            continue;
        }
        const MeshArrays& a = *entry.pending;
        geom::MeshInput input;
        input.source = id.GetString();
        input.points = streamOf(a.points);
        input.faceVertexCounts = std::span<const int32_t>(a.faceVertexCounts.cdata(), a.faceVertexCounts.size());
        input.faceVertexIndices = std::span<const int32_t>(a.faceVertexIndices.cdata(), a.faceVertexIndices.size());
        input.holeIndices = std::span<const int32_t>(a.holeIndices.cdata(), a.holeIndices.size());
        input.invisibleFaces = std::span<const int32_t>(a.invisibleFaces.cdata(), a.invisibleFaces.size());
        input.leftHanded = a.leftHanded;
        input.smoothNormals = a.smoothNormals;
        // The same key while Hydra says the topology stands, so the scene
        // takes the rebuilt mesh as the old one deformed and refits.
        if (a.topologyChanged || entry.topologyKey == 0) {
            entry.topologyKey = ++nextTopologyKey_;
        }
        input.topology = entry.topologyKey;
        std::vector<geom::PrimvarInput> primvars;
        for (const PrimvarArrays& p : a.primvars) {
            geom::PrimvarInput primvar;
            primvar.name = p.name;
            // HdInterpolation's order is geom::Interpolation's.
            primvar.interpolation = static_cast<geom::Interpolation>(p.interpolation);
            primvar.values = primvarStreamOf(p.values, &primvar.components);
            primvar.indices = std::span<const int32_t>(p.indices.cdata(), p.indices.size());
            primvars.push_back(std::move(primvar));
        }
        // The coordinate systems bound to the prim, as the material's
        // transform nodes read them: each system's transform to world, its
        // rows as three constant float4 primvars ("atheneaCoordSys_NAME_0" to
        // "_2"). Values handed over as Hydra gave them; any inverse is the
        // shader's.
        std::vector<std::array<float, 12>> coordSysRows;
        coordSysRows.reserve(entry.look.coordSys.size());
        for (const CoordSysBinding& binding : entry.look.coordSys) {
            coordSysRows.push_back(binding.toWorld.rows3x4());
            const std::array<float, 12>& rows = coordSysRows.back();
            for (uint32_t r = 0; r < 3; ++r) {
                geom::PrimvarInput primvar;
                primvar.name = "atheneaCoordSys_" + binding.name + "_" + std::to_string(r);
                primvar.interpolation = geom::Interpolation::Constant;
                primvar.components = 4;
                primvar.values = {std::as_bytes(std::span<const float>(rows.data() + r * 4, 4)), false};
                primvars.push_back(std::move(primvar));
            }
        }
        input.primvars = primvars;
        for (const MeshSubset& subset : a.subsets) {
            input.subsets.emplace_back(subset.faces.cdata(), subset.faces.size());
        }
        entry.gpu.reset();
        // Skinned: the rest points go through the skinner on the device, and
        // the mesh is built from what comes out, under the same topology key
        // -- a deformation of the rest mesh, refit and not rebuilt.
        gpu::Buffer skinned;
        if (a.skinning.has_value() && input.points.values() >= 3) {
            if (!skinner_.has_value()) {
                auto made = geom::Skinner::create(*library_);
                if (!made) return std::move(made).error();
                skinner_.emplace(std::move(*made));
            }
            auto result = skinner_->skin(skinningInputOf(*a.skinning, input.points));
            if (result) {
                skinned = std::move(*result);
                input.devicePositions = &skinned;
                input.devicePoints = static_cast<uint32_t>(input.points.values() / 3);
            } else {
                log::warn("hdAthenea: {}: skinning: {}", id.GetString(), result.error().toString());
            }
        }
        // A subdivision surface at a refine level above zero: refined on the
        // device (after the skinning, when there is any) and built from the
        // refined mesh. The refined topology is the mesh's own key.
        std::optional<geom::Refined> refined;
        std::optional<geom::Refined::AsInput> refinedInput;
        const bool subdivides = a.refineLevel > 0 && !a.scheme.IsEmpty() &&
                                a.scheme != pxr::PxOsdOpenSubdivTokens->none && input.points.values() >= 3 &&
                                !a.faceVertexCounts.empty();
        if (subdivides) {
            if (!subdivider_.has_value()) {
                auto made = geom::Subdivider::create(*library_);
                if (!made) return std::move(made).error();
                subdivider_.emplace(std::move(*made));
            }
            geom::SubdivisionInput sub;
            sub.source = input.source;
            sub.points = input.points;
            sub.devicePositions = input.devicePositions;
            sub.devicePoints = input.devicePoints;
            sub.faceVertexCounts = input.faceVertexCounts;
            sub.faceVertexIndices = input.faceVertexIndices;
            sub.holeIndices = input.holeIndices;
            sub.scheme = a.scheme == pxr::PxOsdOpenSubdivTokens->loop       ? geom::SubdivisionScheme::Loop
                         : a.scheme == pxr::PxOsdOpenSubdivTokens->bilinear ? geom::SubdivisionScheme::Bilinear
                                                                             : geom::SubdivisionScheme::CatmullClark;
            sub.levels = static_cast<uint32_t>(std::min(a.refineLevel, 5));
            sub.creaseIndices = std::span<const int32_t>(a.creaseIndices.cdata(), a.creaseIndices.size());
            sub.creaseLengths = std::span<const int32_t>(a.creaseLengths.cdata(), a.creaseLengths.size());
            sub.creaseSharpnesses = std::span<const float>(a.creaseSharpnesses.cdata(), a.creaseSharpnesses.size());
            sub.cornerIndices = std::span<const int32_t>(a.cornerIndices.cdata(), a.cornerIndices.size());
            sub.cornerSharpnesses = std::span<const float>(a.cornerSharpnesses.cdata(), a.cornerSharpnesses.size());
            sub.primvars = input.primvars;
            auto made = subdivider_->refine(sub);
            if (made) {
                refined.emplace(std::move(*made));
                refinedInput.emplace(refined->asMeshInput(input.source, input.topology));
                refinedInput->mesh.invisibleFaces = {};   // a coarse face's flag does not survive refinement yet
                refinedInput->mesh.leftHanded = input.leftHanded;
            } else {
                log::warn("hdAthenea: {}: subdivision: {}", id.GetString(), made.error().toString());
            }
        }
        const geom::MeshInput& built_input = refinedInput.has_value() ? refinedInput->mesh : input;
        if (input.points.values() >= 3 && !a.faceVertexCounts.empty()) {
            if (!meshBuilder_.has_value()) {
                auto made = geom::MeshBuilder::create(*library_);
                if (!made) return std::move(made).error();
                meshBuilder_.emplace(std::move(*made));
            }
            auto mesh = meshBuilder_->build(built_input);
            if (!mesh && mesh.error().code() == ErrorCode::OutOfMemory) {
                return Error::make(ErrorCode::OutOfMemory, "{}: {}", id.GetString(), mesh.error().message());   // kept pending, as a cloud is
            }
            if (mesh) {
                entry.gpu = std::make_shared<const geom::GpuMesh>(std::move(*mesh));
            } else {
                log::warn("hdAthenea: {}: {}", id.GetString(), mesh.error().toString());
            }
            // The shutter's meshes: the same topology, the points there.
            entry.gpuStart.reset();
            entry.gpuEnd.reset();
            for (auto [pending, into] : {std::pair{&entry.pendingStart, &entry.gpuStart},
                                         std::pair{&entry.pendingEnd, &entry.gpuEnd}}) {
                if (pending->IsEmpty() || !entry.gpu) {
                    continue;
                }
                geom::MeshInput sample = input;
                sample.points = streamOf(*pending);
                if (sample.points.values() != input.points.values()) {
                    continue;
                }
                auto built = meshBuilder_->build(sample);
                if (built) {
                    *into = std::make_shared<const geom::GpuMesh>(std::move(*built));
                }
            }
        }
        entry.pending.reset();
        entry.pendingStart = pxr::VtValue();
        entry.pendingEnd = pxr::VtValue();
        ++uploaded;
    }
    for (auto& [id, entry] : splats_) {
        if (entry.assetPending.has_value()) {
            StreamedAsset asset = std::move(*entry.assetPending);
            entry.assetPending.reset();
            if (asset.path != entry.asset.path || asset.budget != entry.asset.budget) {
                entry.lodCloud.reset();
                entry.pool.reset();
                if (!asset.path.empty()) {
                    // Held to the device's budget (streamBudget), which may
                    // stream a file asked to be read whole.
                    if (const uint64_t budget = streamBudget(id, asset); budget > 0) {
                        auto pool = lod::StreamingPool::open(*device_, asset.path, {budget, 2});
                        if (pool) {
                            entry.pool = std::move(*pool);
                        } else {
                            log::warn("hdAthenea: {}: {}", id.GetString(), pool.error().toString());
                        }
                    } else {
                        auto read = lod::readAthc(*device_, asset.path);
                        if (read) {
                            entry.lodCloud = std::make_unique<lod::LodCloud>(std::move(*read));
                        } else {
                            log::warn("hdAthenea: {}: {}", id.GetString(), read.error().toString());
                        }
                    }
                }
                ++uploaded;
            }
            entry.asset = std::move(asset);
        }
        if (!entry.asset.path.empty()) {
            // The asset stands in for the prim's own arrays.
            entry.pending.reset();
            entry.gpu.reset();
            continue;
        }
        if (!entry.pending.has_value()) {
            continue;
        }
        // A TIME CHANGE IS NOT A NEW CLOUD.
        //
        // usdVolImaging flags a ParticleField as time varying, so every step
        // of the timeline arrives here with `DirtyPoints | DirtyPrimvar` and
        // every array read again -- and this used to decode the whole cloud
        // again with it. `VtArray` is copy-on-write, so a `Get` at a new time
        // of an attribute that has no time samples hands back the same
        // buffer: comparing what the arrays point at is how a cloud whose
        // geometry did not change is not uploaded a second time. On a skinned
        // bird of 4 269 858 gaussians only `skinningXforms` has samples, and
        // that is 609 matrices.
        // What this cloud's ids are called, as its primvar holds them: read
        // with the arrays, since a manifest is the cloud's, not a frame's.
        entry.cryptoManifest = core::parseCryptomatteManifest(entry.pending->cryptoManifest);
        entry.lodGroup = entry.pending->lodGroup;
        entry.lodCell = entry.pending->lodCell;
        entry.lodThreshold = entry.pending->lodThreshold;
        const CloudIdentity identity = identityOf(*entry.pending);
        const bool sameCloud = entry.gpu != nullptr && identity == entry.uploaded;
        if (!sameCloud) {
            const scene::SplatStreams streams = splatStreams(*entry.pending, id.GetString());
            if (streams.count == 0) {
                entry.gpu.reset();
            } else {
                auto splats = loader_->upload(streams);
                cloudUploads_.fetch_add(1);
                if (!splats && splats.error().code() == ErrorCode::OutOfMemory) {
                    // Kept pending, and the commit stopped here: the render
                    // pass gives memory back and commits again, and a cloud
                    // that still does not fit is the frame's error rather
                    // than a warning over an image without it.
                    return Error::make(ErrorCode::OutOfMemory, "{}: {}", id.GetString(), splats.error().message());
                }
                if (!splats) {
                    log::warn("hdAthenea: {}: {}", id.GetString(), splats.error().toString());
                    entry.gpu.reset();
                } else {
                    entry.gpu = std::make_unique<scene::GpuSplats>(std::move(*splats));
                    // THE BAKED FIELDS, where the file carries them: two
                    // arrays that go up as they are, the same words the bake
                    // wrote (`athenea visibility`), and nothing computed here.
                    const auto* partsHeld = entry.pending->visibilityParts.IsHolding<pxr::VtFloatArray>()
                                                ? &entry.pending->visibilityParts.UncheckedGet<pxr::VtFloatArray>()
                                                : nullptr;
                    const auto* texelsHeld = entry.pending->visibilityTexels.IsHolding<pxr::VtIntArray>()
                                                 ? &entry.pending->visibilityTexels.UncheckedGet<pxr::VtIntArray>()
                                                 : nullptr;
                    const auto* partOfHeld = entry.pending->visibilityPartOf.IsHolding<pxr::VtIntArray>()
                                                 ? &entry.pending->visibilityPartOf.UncheckedGet<pxr::VtIntArray>()
                                                 : nullptr;
                    if (partsHeld != nullptr && texelsHeld != nullptr && partOfHeld != nullptr &&
                        partsHeld->size() % 12 == 0 && !partsHeld->empty() && !texelsHeld->empty() &&
                        partOfHeld->size() == entry.gpu->declared) {
                        auto parts = gpu::Buffer::fromSpan<float>(
                            *device_, std::span<const float>(partsHeld->cdata(), partsHeld->size()),
                            "splat.visibility.parts");
                        auto texels = gpu::Buffer::fromSpan<int32_t>(
                            *device_, std::span<const int32_t>(texelsHeld->cdata(), texelsHeld->size()),
                            "splat.visibility.texels");
                        auto partOf = gpu::Buffer::fromSpan<int32_t>(
                            *device_, std::span<const int32_t>(partOfHeld->cdata(), partOfHeld->size()),
                            "splat.visibility.partOf");
                        const auto* ambientHeld = entry.pending->visibilityAmbient.IsHolding<pxr::VtIntArray>()
                                                      ? &entry.pending->visibilityAmbient.UncheckedGet<pxr::VtIntArray>()
                                                      : nullptr;
                        if (ambientHeld != nullptr && !ambientHeld->empty()) {
                            auto ambient = gpu::Buffer::fromSpan<int32_t>(
                                *device_, std::span<const int32_t>(ambientHeld->cdata(), ambientHeld->size()),
                                "splat.visibility.ambient");
                            if (ambient) entry.gpu->visibilityAmbient = std::move(*ambient);
                        }
                        // A part a record: packed to the splats kept, as a
                        // rig's influences are.
                        if (partOf) {
                            partOf = loader_->keptOnly(*entry.gpu, *partOf, 1, "splat.visibility.partOf.kept");
                        }
                        if (parts && texels && partOf) {
                            entry.gpu->visibilityParts = std::move(*parts);
                            entry.gpu->visibilityTexels = std::move(*texels);
                            entry.gpu->visibilityPartOf = std::move(*partOf);
                            entry.gpu->visibilityPartCount = static_cast<uint32_t>(partsHeld->size() / 12);
                        } else {
                            log::warn("hdAthenea: {}: the baked visibility did not upload", id.GetString());
                        }
                    }
                }
            }
            entry.uploaded = identity;
        }
        // THE RIG, IF A SKELETON CARRIES THIS CLOUD.
        //
        // The cloud that was just uploaded is the bind pose, and it stays
        // that way: what a frame draws is `posed`, which the skinner writes
        // into buffers of its own. The joints and their weights are the same
        // at every instant and are uploaded with the cloud; only the
        // skeleton's transforms are taken again, and for sixty joints that is
        // four kilobytes.
        if (entry.gpu != nullptr && !entry.lodGroup.empty()) {
            // A level of detail: posed when a view draws it.
            entry.deferredReupload = entry.deferredReupload || !sameCloud;
            entry.deferredPose = std::move(entry.pending);
        } else if (entry.gpu != nullptr) {
            ATHENEA_TRY(carryCloud(id, entry, !sameCloud));
        }
        entry.pending.reset();
        if (!sameCloud) {
            ++uploaded;
        }
    }
    for (auto& [id, entry] : points_) {
        if (!entry.pending.has_value()) {
            continue;
        }
        const scene::PointStreams streams = pointStreams(*entry.pending, id.GetString());
        if (streams.count == 0) {
            entry.gpu.reset();
        } else {
            auto points = loader_->upload(streams);
            if (!points && points.error().code() == ErrorCode::OutOfMemory) {
                return Error::make(ErrorCode::OutOfMemory, "{}: {}", id.GetString(), points.error().message());   // kept pending, as a cloud is
            }
            if (!points) {
                log::warn("hdAthenea: {}: {}", id.GetString(), points.error().toString());
                entry.gpu.reset();
            } else {
                entry.gpu = std::make_unique<scene::GpuPoints>(std::move(*points));
            }
        }
        entry.pending.reset();
        ++uploaded;
    }
    // Anything uploaded is something a path could find: whatever a path traced
    // frame had accumulated was of a scene that no longer exists.
    if (uploaded > 0) {
        revision_.fetch_add(1);
    }
    return uploaded;
}

Result<std::optional<scene::Bounds>> Engine::bounds() {
    std::optional<scene::Bounds> all;
    const auto grow = [&](const scene::Bounds& box, const render::Mat4& toWorld) {
        // A box through an affine map, as scene_bounds.slang does per instance:
        // one prim's matrix and its cloud's box, not the cloud.
        scene::Bounds moved;
        for (int r = 0; r < 3; ++r) {
            double centre = toWorld.at(r, 3);
            double extent = 0.0;
            for (int c = 0; c < 3; ++c) {
                const double mid = 0.5 * (double(box.min[size_t(c)]) + double(box.max[size_t(c)]));
                const double half = 0.5 * (double(box.max[size_t(c)]) - double(box.min[size_t(c)]));
                centre += toWorld.at(r, c) * mid;
                extent += std::abs(toWorld.at(r, c)) * half;
            }
            moved.min[size_t(r)] = static_cast<float>(centre - extent);
            moved.max[size_t(r)] = static_cast<float>(centre + extent);
        }
        if (!all) {
            all = moved;
            return;
        }
        for (size_t k = 0; k < 3; ++k) {
            all->min[k] = std::min(all->min[k], moved.min[k]);
            all->max[k] = std::max(all->max[k], moved.max[k]);
        }
    };
    {
        const std::lock_guard<std::mutex> held(guard_);
        for (const auto& [id, entry] : splats_) {
            if (!entry.visible) {
                continue;
            }
            if (entry.gpu != nullptr) {
                // Where it is drawn, which for a cloud a skeleton carries is
                // not where its bind pose stands.
                grow(entry.posed != nullptr ? entry.posed->bounds : entry.gpu->bounds,
                     entry.objectToWorld);
            } else if (entry.lodCloud != nullptr) {
                grow(entry.lodCloud->splats.bounds, entry.objectToWorld);
            } else if (entry.pool != nullptr) {
                grow(entry.pool->cloud().splats.bounds, entry.objectToWorld);
            }
        }
        for (const auto& [id, entry] : points_) {
            if (entry.visible && entry.gpu != nullptr) {
                grow(entry.gpu->bounds, entry.objectToWorld);
            }
        }
    }
    if (scene_.has_value()) {
        auto meshes = scene_->worldBounds();
        if (!meshes) return std::move(meshes).error();
        if (meshes->has_value()) {
            grow(**meshes, render::Mat4::identity());
        }
    }
    return all;
}

namespace {
/// What in a generated material's source says it varies over the surface
/// with no slot to show for it: MaterialX's position and texture coordinate
/// nodes, and the noises and patterns made of them.
constexpr const char* kPlaceDependent[] = {"gAtheneaInputs.", "atheneaPrimvar", "_Pworld", "_Pobject", "_Pmodel",
                                           "texcoord", "geomprop_UV", "noise", "worley", "fractal", "u_time",
                                           "u_frame", "geomcolor", "geompropvalue"};
}   // namespace

Result<void> Engine::prepareMaterials(const std::vector<std::string>& aovPrimvars) {
    if (!scene_.has_value()) {
        auto scene = world::GpuScene::create(*library_);
        if (!scene) return std::move(scene).error();
        scene_.emplace(std::move(*scene));
    }
    if (!textures_) {
        auto made = material::TextureStore::create(*library_);
        if (!made) return std::move(made).error();
        textures_ = std::move(*made);
        textures_->setSearchPath(assetSearchPath_);
    }
    if (!materialPrograms_.has_value()) {
        auto made = technique::MaterialPrograms::create(*library_);
        if (!made) return std::move(made).error();
        materialPrograms_.emplace(std::move(*made));
    }
    if (!materialShading_.has_value()) {
        auto made = technique::MaterialShading::create(*library_);
        if (!made) return std::move(made).error();
        materialShading_.emplace(std::move(*made));
    }
    // Primvar slots: the AOVs' and every primvar a material reads.
    std::vector<std::string> names = aovPrimvars;
    const auto fixed = [](const std::string& name) {
        return name == "displayColor" || name == "displayOpacity" || name == "normals" || name == "st";
    };
    for (const auto& [id, entry] : materials_) {
        if (!entry.compiled) {
            continue;
        }
        for (const material::MaterialSlot& slot : entry.compiled->slots) {
            if (slot.kind == material::MaterialSlot::Kind::Primvar && !fixed(slot.name) &&
                std::find(names.begin(), names.end(), slot.name) == names.end()) {
                names.push_back(slot.name);
            }
        }
    }
    scene_->setExtraPrimvarSlots(names);
    if (!materialsChanged_ && names == materialSlotNames_ && materialRecords_.valid()) {
        return ok();
    }
    // One module per distinct structure; a row and blob words per material.
    std::vector<material::CompiledMaterial> modules;
    // Beside each module, its opacity where a material of it cuts: empty
    // where none does, and a shadow ray then takes the surface as opaque.
    std::vector<material::CompiledMaterial> opacities;
    std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}};
    std::vector<float> blob;
    materialRows_.clear();
    materialCutouts_ = false;
    for (const auto& [id, entry] : materials_) {
        if (!entry.compiled) {
            continue;
        }
        uint32_t function = 0;
        for (size_t k = 0; k < modules.size(); ++k) {
            if (modules[k].module == entry.compiled->module) {
                function = static_cast<uint32_t>(k + 1);
            }
        }
        if (function == 0) {
            modules.push_back(*entry.compiled);
            opacities.push_back(material::CompiledMaterial{});
            function = static_cast<uint32_t>(modules.size());
        }
        if (entry.opacity && opacities[function - 1].module.empty()) {
            opacities[function - 1] = *entry.opacity;
        }
        const std::vector<float> words = material::MaterialCompiler::parameters(
            *entry.compiled, *textures_, [&](const std::string& name) { return scene_->slotOf(name); });
        materialRows_[id] = static_cast<uint32_t>(rows.size());
        {
            std::string files;
            for (const material::MaterialSlot& slot : entry.compiled->slots) {
                if (slot.kind == material::MaterialSlot::Kind::Texture && !slot.name.empty()) {
                    files += (files.empty() ? "" : ", ") + std::filesystem::path(slot.name).filename().string() +
                             " " + (slot.space.empty() ? std::string("auto") : slot.space);
                }
            }
            log::debug("hdAthenea: material {} row {} module {} [{}]", id.GetString(), rows.size(),
                       entry.compiled->module, files);
        }
        // THE SAME EVERYWHERE: every input a value, and nothing in the graph
        // that reads where on the surface it is (a texture coordinate, a
        // position, a noise of either). What lets a frame lit by
        // prefiltered domes alone shade it from a table (kMaterialUniform).
        const bool uniform =
            std::all_of(entry.compiled->slots.begin(), entry.compiled->slots.end(),
                        [](const material::MaterialSlot& slot) {
                            return slot.kind == material::MaterialSlot::Kind::Value;
                        }) &&
            std::none_of(std::begin(kPlaceDependent), std::end(kPlaceDependent), [&](const char* token) {
                return entry.compiled->source.find(token) != std::string::npos;
            });
        const uint32_t flags = (entry.cutout ? technique::kMaterialCutout : 0u) |
                               (entry.transparent ? technique::kMaterialTransparent : 0u) |
                               (uniform && !entry.cutout && !entry.transparent ? technique::kMaterialUniform : 0u);
        materialCutouts_ = materialCutouts_ || entry.cutout;
        rows.push_back({function, static_cast<uint32_t>(blob.size()), flags, 0});
        blob.insert(blob.end(), words.begin(), words.end());
    }
    if (blob.empty()) {
        blob.push_back(0.0F);
    }
    if (auto loaded = textures_->commit(); !loaded) {
        return std::move(loaded).error();
    }
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*device_, rows, "materials.records");
    if (!records) return std::move(records).error();
    auto words = gpu::Buffer::fromSpan<float>(*device_, blob, "materials.blob");
    if (!words) return std::move(words).error();
    materialRecords_ = std::move(*records);
    materialBlob_ = std::move(*words);
    ATHENEA_TRY(materialPrograms_->setModules(modules, opacities));
    ATHENEA_TRY(materialShading_->setPrograms(*materialPrograms_));
    materialsChanged_ = false;
    materialSlotNames_ = std::move(names);
    return ok();
}

Result<void> Engine::prepareCryptoTable() {
    // The table is the frame's prims as they stand: rebuilt when a prim was
    // added, removed or resynced, and left alone otherwise.
    const uint64_t revision = revision_.load();
    if (cryptoOfPrim_.valid() && cryptoTableRevision_ == revision &&
        cryptoTableMeshes_ == meshes_.size() + splats_.size()) {
        return ok();
    }
    uint32_t highest = 0;
    for (const auto& [path, entry] : meshes_) {
        highest = std::max(highest, entry.primId);
    }
    std::vector<uint32_t> ids(size_t{highest} + 2, 0u);
    cryptoManifest_.clear();
    for (const auto& [path, entry] : meshes_) {
        const std::string name = path.GetString();
        const uint32_t id = core::cryptomatteId(name);
        ids[entry.primId] = id;
        cryptoManifest_[name] = id;
    }
    // Every cloud the frame can draw names itself too, from the manifest its
    // own file carries: a matte that named a number and nothing else would
    // leave a reader with no way back to the prim.
    for (const auto& [path, entry] : splats_) {
        for (const auto& [name, id] : entry.cryptoManifest) {
            cryptoManifest_[name] = id;
        }
    }
    auto made = gpu::Buffer::fromSpan<uint32_t>(*device_, ids, "crypto.ofPrim");
    if (!made) return std::move(made).error();
    cryptoOfPrim_ = std::move(*made);
    cryptoPrimCount_ = static_cast<uint32_t>(ids.size());
    cryptoTableRevision_ = revision;
    cryptoTableMeshes_ = meshes_.size() + splats_.size();
    return ok();
}

AovView Engine::aovView(const render::RenderTargets& targets, AovSource aov) const {
    AovView view;
    view.ids = aov.kind == AovKind::PrimId || aov.kind == AovKind::InstanceId || aov.kind == AovKind::ElementId;
    switch (aov.kind) {
    case AovKind::Colour: view.buffer = &targets.colour; return view;
    case AovKind::Depth: view.buffer = &targets.depth; view.source = 1; return view;
    // A Cryptomatte layer: the blend writes the three of them plane after
    // plane, so a layer is one whole image at an offset of its own, the way a
    // light group's plane is.
    case AovKind::Crypto:
        if (targets.crypto.valid() && aov.primvar < 3) {
            view.buffer = &targets.crypto;
            view.offset = static_cast<uint32_t>(aov.primvar * uint64_t{targets.width} * targets.height);
        }
        return view;
    // The path tracer's, written at its first hit: nothing where the frame
    // was not path traced.
    case AovKind::Albedo:
        if (pathAuxValid_ && pathAux_.width == targets.width && pathAux_.height == targets.height) {
            view.buffer = &pathAux_.planes;
        }
        return view;
    case AovKind::ShadingNormal:
        if (pathAuxValid_ && pathAux_.width == targets.width && pathAux_.height == targets.height) {
            view.buffer = &pathAux_.planes;
            view.offset = static_cast<uint32_t>(pathAux_.normalOffset());
        }
        return view;
    // A light group's plane, one of the frame's in one buffer: the offset
    // addresses it at a stride of one.
    case AovKind::LightGroup:
        if (aov.primvar >= lightGroupCount_ || lightGroupPixels_ != uint64_t{targets.width} * targets.height) {
            return view;
        }
        // The raster's shading writes the planes; a path traced frame's means
        // are copied there (gatherLightGroups) before the domes and the
        // exposure reach them.
        if (lightGroupColour_.valid()) {
            view.buffer = &lightGroupColour_;
            view.offset = static_cast<uint32_t>(aov.primvar * lightGroupPixels_);
        }
        return view;
    default: break;
    }
    if (!aovsValid_ || aovs_.width != targets.width || aovs_.height != targets.height ||
        (aov.kind == AovKind::Primvar && aov.primvar >= aovs_.primvarSlots)) {
        return view;   // nothing a mesh drew
    }
    switch (aov.kind) {
    case AovKind::PrimId: view.buffer = &aovs_.ids; view.source = 2; view.stride = 3; view.offset = 0; break;
    case AovKind::InstanceId: view.buffer = &aovs_.ids; view.source = 2; view.stride = 3; view.offset = 1; break;
    case AovKind::ElementId: view.buffer = &aovs_.ids; view.source = 2; view.stride = 3; view.offset = 2; break;
    case AovKind::EyeNormal: view.buffer = &aovs_.eyeNormals; break;
    case AovKind::WorldNormal: view.buffer = &aovs_.worldNormals; break;
    default: view.buffer = &aovs_.primvars; view.stride = aovs_.primvarSlots; view.offset = aov.primvar; break;
    }
    return view;
}

Result<void> Engine::writeAov(const render::RenderTargets& targets, AovSource aov, const AovLayout& layout,
                              const double* projection, std::span<uint8_t> into) {
    const uint64_t bytes = uint64_t{targets.width} * targets.height * layout.channels * layout.componentBytes;
    if (into.size() < bytes || layout.channels == 0 || layout.channels > 4 ||
        (layout.componentBytes != 1 && layout.componentBytes != 2 && layout.componentBytes != 4)) {
        return Error(ErrorCode::InvalidArgument, "a render buffer the engine cannot fill");
    }
    const AovView view = aovView(targets, aov);
    if (view.buffer == nullptr) {
        // Nothing a mesh drew: the clear value, -1 for ids and 0 otherwise.
        std::fill(into.begin(), into.begin() + static_cast<std::ptrdiff_t>(bytes), static_cast<uint8_t>(view.ids ? 0xFF : 0));
        return ok();
    }
    const gpu::Buffer* source = view.buffer;
    const uint32_t kind = view.source;
    const uint32_t stride = view.stride;
    const uint32_t offset = view.offset;
    if (!source->valid()) {
        return Error(ErrorCode::InvalidArgument, "nothing rendered to convert");
    }
    if (!aovConvert_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "athenea/usd/aov_convert", "aovConvert");
        if (!made) return std::move(made).error();
        aovConvert_.emplace(std::move(*made));
    }
    const uint32_t words = static_cast<uint32_t>((bytes + 3) / 4);
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{words} * 4;
    desc.elementBytes = 4;
    desc.label = "hydra.aov";
    auto out = gpu::Buffer::create(*device_, desc);
    if (!out) return std::move(out).error();
    gpu::BufferDesc one;
    one.bytes = 16;
    one.elementBytes = 16;
    one.label = "hydra.placeholder";
    auto placeholder = gpu::Buffer::create(*device_, one);
    if (!placeholder) return std::move(placeholder).error();
    gpu::CommandBatch batch(*device_);
    aovConvert_->dispatch(batch, {words, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(kind == 0 ? source->rhi() : placeholder->rhi());
        cursor["depth"].setBinding(kind == 1 ? source->rhi() : placeholder->rhi());
        cursor["idSource"].setBinding(kind == 2 ? source->rhi() : placeholder->rhi());
        cursor["words"].setBinding(out->rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["width"].setData(targets.width);
        p["height"].setData(targets.height);
        p["words"].setData(words);
        p["source"].setData(kind);
        p["stride"].setData(stride);
        p["offset"].setData(offset);
        p["channels"].setData(layout.channels);
        p["componentBytes"].setData(layout.componentBytes);
        p["componentKind"].setData(layout.componentKind);
        // Matrix terms, not data: the host's projection, as floats.
        p["p22"].setData(static_cast<float>(projection[10]));
        p["p32"].setData(static_cast<float>(projection[14]));
        p["p23"].setData(static_cast<float>(projection[11]));
        p["p33"].setData(static_cast<float>(projection[15]));
    });
    ATHENEA_TRY(batch.submit(true));
    // A readback for a host that maps the buffer: output IO, the only copy.
    return out->read(*device_, 0, bytes, into.data());
}

/// The lights a frame of relit splats needs, where nothing else built them.
///
/// A mesh layer builds the table for its own shading; a frame that is only
/// splats has no such layer, and both routes that draw one -- the tile
/// rasteriser and the ray tracer -- ask for the same table.
Result<void> Engine::prepareLightImages(std::vector<light::Light>& lamps) {
    // A dome's image goes through the same texture table the materials
    // sample, so it is requested here and committed before the frame's
    // records are uploaded.
    //
    // THIS USED TO SIT INSIDE THE MESH LAYER'S PREPARATION, and a frame with
    // no mesh in it never reached it: a cloud standing alone under a domeem
    // read the dome as the white one a light with no image is, so the
    // workshop an HDR held was painted as a blank white wall behind the car.
    if (lamps.empty()) {
        return ok();
    }
    if (!textures_) {
        auto made = material::TextureStore::create(*library_);
        if (!made) return std::move(made).error();
        textures_ = std::move(*made);
        textures_->setSearchPath(assetSearchPath_);
    }
    bool domeTextures = false;
    for (light::Light& lamp : lamps) {
        // An IES profile, read the first time its path is seen.
        if (!lamp.iesFile.empty() && !lamp.ies) {
            auto known = iesProfiles_.find(lamp.iesFile);
            if (known != iesProfiles_.end()) {
                lamp.ies = known->second;
            } else if (!iesFailed_.contains(lamp.iesFile)) {
                auto read = io::readIes(lamp.iesFile);
                if (read) {
                    lamp.ies = std::make_shared<const io::IesProfile>(std::move(*read));
                    iesProfiles_[lamp.iesFile] = lamp.ies;
                } else {
                    iesFailed_.insert(lamp.iesFile);
                    log::warn("hdAthenea: light without its IES profile: {}", read.error().toString());
                }
            }
        }
        if (lamp.texture.empty()) {
            continue;
        }
        lamp.textureId = textures_->request(lamp.texture, lamp.textureColourSpace);
        // Lat-long: around in u, clamped at the poles.
        lamp.sampler = textures_->sampler(material::Wrap::Repeat, material::Wrap::Clamp);
        domeTextures = true;
    }
    if (domeTextures) {
        if (auto loaded = textures_->commit(); !loaded) {
            return std::move(loaded).error();
        }
    }
    return ok();
}

Result<void> Engine::prepareSplatLights(std::vector<light::Light>& lamps,
                                        std::span<const render::SplatInstance> splats) {
    if (!lightTable_.has_value()) {
        auto made = light::LightTable::create(*library_);
        if (!made) return std::move(made).error();
        lightTable_.emplace(std::move(*made));
    }
    ATHENEA_TRY(prepareLightImages(lamps));
    if (!splats.empty()) {
        // A dome's or a sun's share of the power wants the scene's reach;
        // a cloud's own bounds are what there is.
        float radius = 0.0F;
        for (const render::SplatInstance& instance : splats) {
            if (instance.splats == nullptr) {
                continue;
            }
            if (!loader_.has_value()) {
                break;
            }
            auto bounds = loader_->boundsOf(instance.splats->positions, instance.splats->count);
            if (!bounds) return std::move(bounds).error();
            const scene::Bounds& b = *bounds;
            const float dx = b.max[0] - b.min[0], dy = b.max[1] - b.min[1], dz = b.max[2] - b.min[2];
            radius = std::max(radius, 0.5F * std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        lightSceneRadius_ = std::max(radius, 1.0e-3F);
    }
    ATHENEA_TRY(lightTable_->set(lamps, lightSceneRadius_));
    return prepareEnvironment(lamps);
}

/// THE DOMES' SKIES, PREPARED ONCE.
///
/// A dome's image goes through the material texture table, and the splat
/// projection sits below material in the module order: it cannot sample it.
/// So a cloud under an HDRI was lit by the dome's mean colour while the mesh
/// beside it reflected the sky. `technique::Environment` writes that image
/// into two things a buffer can hold, and this is what decides when.
///
/// The key is the domes' own records: a frame whose sky did not change builds
/// nothing, and a camera that moved is not a change. The shape is
/// `EmissiveKey`'s, for the same reason.
Result<void> Engine::prepareEnvironment(const std::vector<light::Light>& lamps) {
    std::vector<uint32_t> domeLights;
    std::vector<uint32_t> domeTextures;
    std::vector<light::LightRecord> keys;
    for (uint32_t k = 0; k < lamps.size(); ++k) {
        if (lamps[k].kind != light::LightKind::Dome) {
            continue;
        }
        // A DOME WITH NO IMAGE IS PREPARED TOO, and it was not.
        //
        // A constant sky is what the closed form in `splat_relight` answers
        // exactly -- for a point that sees all of it. A gaussian carrying a
        // transfer does not: what it knows is how much of the sky reaches it,
        // and that road is the harmonics'. Left out, a cloud under a plain
        // dome threw its own visibility away and lit every pocket as if it
        // were open, which is what the pawn's disc measured.
        if (domeLights.size() >= technique::kEnvironmentDomes) {
            // Past the fourth, a dome is its colour, as every dome was
            // before there was an environment at all.
            break;
        }
        domeLights.push_back(k);
        domeTextures.push_back(lamps[k].textureId);
        keys.push_back(light::LightTable::recordOf(lamps[k]));
    }
    if (domeLights.empty()) {
        environmentKeys_.clear();
        return ok();
    }
    const bool same = environment_.has_value() && environment_->ready() &&
                      keys.size() == environmentKeys_.size() &&
                      std::memcmp(keys.data(), environmentKeys_.data(),
                                  keys.size() * sizeof(light::LightRecord)) == 0;
    if (same) {
        return ok();
    }
    if (!environment_.has_value()) {
        auto made = technique::Environment::create(*library_);
        if (!made) {
            // A device that cannot compile it is a device that draws domes as
            // it always did, not a device that fails a frame.
            log::info("hdAthenea: no prepared environment on this device ({})", made.error().toString());
            environmentKeys_.clear();
            return ok();
        }
        environment_.emplace(std::move(*made));
    }
    if (!textures_) {
        return ok();   // no table, so no image to prepare from
    }
    ATHENEA_TRY(environment_->build(*lightTable_, *textures_, domeLights, domeTextures,
                                static_cast<uint32_t>(lamps.size())));
    environmentKeys_ = std::move(keys);
    return ok();
}

/// What a renderer is handed of the prepared sky: nothing at all where none
/// was prepared, and then a dome is its colour.
void Engine::bindEnvironment(render::SplatLights& lights) const {
    if (!environment_.has_value() || !environment_->ready()) {
        return;
    }
    lights.envTexels = &environment_->texels();
    lights.envSh = &environment_->sh();
    lights.envOfLight = &environment_->domeOfLight();
    lights.envLights = environment_->lightCount();
    lights.envBaseSide = environment_->baseSide();
    lights.envSun = &environment_->sun();
    // What a cloud may keep between frames is good while these stand: the
    // lights as the table last changed them (the sky is prepared from them).
    lights.revision = lightTable_.has_value() ? lightTable_->revision() : 0;
}

/// A bake is a frame whose camera is a list of rays. Everything the frame
/// needs -- the meshes on the device, their materials, the lights, the
/// acceleration structure -- is what `render` prepares, so the bake goes
/// through it rather than around it: a 1 by 1 frame, drawn nowhere, with the
/// request carried to the one place the path tracer is asked to trace.
Result<void> Engine::bakePoints(const BakeRequest& bake, const render::Projection& projection,
                                const render::RenderSettings& settings) {
    if (bake.rays == nullptr || !bake.rays->valid() || bake.count == 0 || bake.out == nullptr) {
        return Error(ErrorCode::InvalidArgument, "nothing to bake");
    }
    render::RenderSettings frame = settings;
    frame.width = 1;
    frame.height = 1;
    render::RenderTargets nowhere;
    return render(projection, frame, nowhere, Technique::RayTraced, true, nullptr, {},
                  MeshVisibility::Automatic, &bake);
}

Result<void> Engine::render(const render::Projection& base, const render::RenderSettings& settings,
                            render::RenderTargets& targets, Technique technique, bool settleStreams,
                            const pxr::TfTokenVector* renderTags, const AovRequest& aovRequest,
                            MeshVisibility visibility, const BakeRequest* bake) {
    // ANTIALIASING, WHERE A FRAME IS GATHERED OVER PASSES.
    //
    // Every ray of this engine goes through the middle of its pixel, and the
    // visibility buffer holds one hit a pixel, so an edge is a staircase
    // however many paths are cast at it. A path traced frame, though, is a
    // mean over passes, and each pass runs the visibility again: moving the
    // camera by a fraction of a pixel between them makes that mean an average
    // over the pixel's area, which is what antialiasing is. It costs nothing
    // -- the same passes, the same rays -- and it is why the first pass takes
    // no offset at all, so a single pass frame is exactly what it always was.
    //
    // The offsets are a Halton sequence in 2 and 3: they fill the pixel
    // evenly at any number of passes, which a random pair does not.
    render::Projection projection = base;
    const auto halton = [](uint32_t index, uint32_t prime) {
        double out = 0.0;
        double f = 1.0 / double(prime);
        for (uint32_t i = index; i > 0; i /= prime) {
            out += f * double(i % prime);
            f /= double(prime);
        }
        return out;
    };
    if (technique == Technique::RayTraced && pathSeed_ > 0 && bake == nullptr && antialias_.load()) {
        projection.centreX += halton(pathSeed_, 2) - 0.5;
        projection.centreY += halton(pathSeed_, 3) - 0.5;
    }
    // A frame builds the clouds' shadow proxies at most once, wherever it
    // first needs them: for a mesh's shadow rays, or for a relit cloud's own.
    shadowTracerReady_ = false;
    lastTargets_ = &targets;
    aovsValid_ = false;
    // A frame drawn since the shutter changed: this one draws the prims as
    // they were sampled, the next draws them resampled (docs: the shutter).
    if (const int settling = shutterSettle_.load(); settling > 0) {
        shutterSettle_.store(settling - 1);
    }
    {
        bool anyMesh = false;
        {
            const std::lock_guard<std::mutex> held(guard_);
            // A traced volume draws through the mesh layer, which wants the
            // scene and the material programs even when no mesh is in it.
            anyMesh = !meshes_.empty() || (technique == Technique::RayTraced && !volumes_.empty());
        }
        if (anyMesh) {
            ATHENEA_TRY(prepareMaterials(aovRequest.primvars));
        }
    }
    const auto rowOf = [&](const pxr::SdfPath& material) -> uint32_t {
        const auto found = materialRows_.find(material);
        return found != materialRows_.end() ? found->second : 0u;
    };
    const auto subsetRowsOf = [&](const MeshEntry& entry) {
        std::vector<uint32_t> rows;
        rows.reserve(entry.subsetMaterials.size());
        for (const pxr::SdfPath& material : entry.subsetMaterials) {
            rows.push_back(rowOf(material));
        }
        return rows;
    };
    std::vector<light::Light> lamps;
    std::vector<world::MeshInstance> meshInstances;
    std::vector<world::InstanceSet> meshSets;
    std::vector<render::SplatInstance> splats;
    std::vector<render::PointInstance> points;
    std::vector<lod::LodInstance> cuts;
    std::vector<lod::StreamingPool*> poolOf;   // per cut: its pool, if streamed
    // Which prim each instance and each cut came from, for the Gaussians
    // panel: the device counts by instance, and only this says whose it was.
    ++frameSerial_;
    std::vector<std::string> instancePrims;
    std::vector<std::string> cutPrims;
    std::vector<lod::CutStats> cutStats;
    std::set<const SplatEntry*> drawnLevels;
    const uint32_t motionBuckets = technique == Technique::RayTraced ? motionBuckets_.load() : 1u;
    // OBJECT TO VIEW, AND WHAT THE SHUTTER CHANGES OF IT: the camera's motion
    // and the prim's in one 3x4, which is what the rasteriser smears a
    // gaussian along (`SplatInstance::viewStep`). A camera that does not move
    // is `worldToView` at both ends.
    const bool cameraMoves = projection.viewToWorldStart.m != projection.viewToWorldEnd.m;
    const render::Mat4 viewOpen =
        cameraMoves ? aofx::xform::inverseAffine(projection.viewToWorldStart) : projection.worldToView;
    const render::Mat4 viewClose =
        cameraMoves ? aofx::xform::inverseAffine(projection.viewToWorldEnd) : projection.worldToView;
    const auto viewStepOf = [&](const render::Mat4& objectToWorld, const std::array<double, 16>& step) {
        std::array<float, 12> out{};
        const bool still = std::all_of(step.begin(), step.end(), [](double v) { return v == 0.0; });
        if (still && !cameraMoves) {
            return out;
        }
        render::Mat4 closing = objectToWorld;
        for (size_t k = 0; k < 16; ++k) {
            closing.m[k] += step[k];
        }
        const std::array<float, 12> a = (viewClose * closing).rows3x4();
        const std::array<float, 12> b = (viewOpen * objectToWorld).rows3x4();
        for (size_t k = 0; k < 12; ++k) {
            out[k] = a[k] - b[k];
        }
        return out;
    };
    const double shutterOpen = shutterOpen_.load();
    const double shutterClose = shutterClose_.load();
    {
        const std::lock_guard<std::mutex> held(guard_);
        lamps.reserve(lights_.size());
        for (auto& [id, entry] : lights_) {
            const light::Light& lamp = entry.lamp;
            lamps.push_back(lamp);
            lamps.back().lightCategory = categoryBit(lamp.lightLink);
            lamps.back().shadowCategory = categoryBit(lamp.shadowLink);
            // Its light group, numbered among the ones this frame asks for
            // (1 + the index; 0 for none, and for a group nobody asked for).
            lamps.back().groupIndex = 0;
            for (size_t g = 0; g < aovRequest.lightGroups.size() && !lamp.group.empty(); ++g) {
                if (aovRequest.lightGroups[g] == lamp.group) {
                    lamps.back().groupIndex = static_cast<uint32_t>(g + 1);
                    break;
                }
            }
            // Under an instancer: the chain composed on the device, as a
            // mesh's, once per change; its rows place the light's copies.
            if (!entry.instancing.empty()) {
                std::vector<uint64_t> versions;
                bool complete = true;
                for (const InstancerLink& link : entry.instancing) {
                    const auto found = instancers_.find(link.instancer);
                    complete = complete && found != instancers_.end();
                    versions.push_back(found != instancers_.end() ? found->second.version : 0);
                }
                if (!complete) {
                    // Its instancer has not arrived: as a mesh, not drawn yet.
                    lamps.pop_back();
                    continue;
                }
                {
                    if (entry.chainDirty || versions != entry.chainVersions) {
                        if (!instancing_.has_value()) {
                            auto made = world::Instancing::create(*library_);
                            if (!made) return std::move(made).error();
                            instancing_.emplace(std::move(*made));
                        }
                        ATHENEA_TRY(composeChains(*instancing_, instancers_, entry.instancing, entry.chain,
                                              entry.chainStart, entry.chainEnd, entry.chainTimeStart,
                                              entry.chainTimeEnd));
                        entry.chainVersions = std::move(versions);
                        entry.chainDirty = false;
                    }
                    lamps.back().instanceRows = &entry.chain.rows;
                    lamps.back().instanceCount = entry.chain.count;
                    if (motionBuckets > 1 && entry.chain.count > 0 && entry.chainStart.count == entry.chain.count &&
                        entry.chainStart.rows.valid() && entry.chainEnd.rows.valid()) {
                        lamps.back().instanceRowsStart = &entry.chainStart.rows;
                        lamps.back().instanceRowsEnd = &entry.chainEnd.rows;
                        lamps.back().instanceTimeStart = static_cast<float>(entry.chainTimeStart);
                        lamps.back().instanceTimeEnd = static_cast<float>(entry.chainTimeEnd);
                        if (entry.chainTimeEnd == entry.chainTimeStart) {
                            lamps.back().instanceTimeStart = static_cast<float>(shutterOpen);
                            lamps.back().instanceTimeEnd = static_cast<float>(shutterClose);
                        }
                    }
                }
            }
        }
        // WHAT A CLOUD WITH A BAKED VISIBILITY NEEDS MEASURED BEFORE THE FRAME:
        // its factors, one a splat a light, laid out one slot a splat in the
        // order the instances go to the renderers -- which is the order their
        // colour slots take, so the same base serves both routes.
        frameMeasured_.clear();
        frameClouds_.clear();
        frameSlots_ = 0;
        const std::set<const SplatEntry*> levels = lodLevelsFor(projection);
        drawnLevels = levels;
        // The levels this view draws are posed now, with the latest joints
        // their prims were given; the others keep theirs for when they are.
        for (auto& [id, entry] : splats_) {
            if (levels.count(&entry) != 0 && entry.deferredPose.has_value()) {
                entry.pending = std::move(entry.deferredPose);
                entry.deferredPose.reset();
                const bool reuploaded = entry.deferredReupload;
                entry.deferredReupload = false;
                const Result<void> posed = carryCloud(id, entry, reuploaded);
                entry.pending.reset();
                if (!posed) return posed;
            }
        }
        for (const auto& [id, entry] : splats_) {
            if (!entry.visible) {
                continue;
            }
            // One level of a cloud that has several: the one this view wants.
            if (!entry.lodGroup.empty() && levels.count(&entry) == 0) {
                continue;
            }
            if (entry.gpu != nullptr) {
                // The posed cloud where a skeleton carries it, and the cloud
                // itself where nothing does.
                const scene::GpuSplats* drawn = entry.posed != nullptr ? entry.posed.get() : entry.gpu.get();
                splats.push_back({drawn, entry.objectToWorld, entry.edit, entry.relight,
                                  entry.litBody, categoryMask(entry.categories)});
                instancePrims.push_back(id.GetString());
                splats.back().transferIndirect = transferIndirect_.load();
                splats.back().reflectCloud = splatReflections_.load();
                splats.back().ior = entry.ior;
                splats.back().catcher = entry.catcher;
                // What the shutter moved each gaussian, where a skeleton
                // carries the cloud and the camera's is open. The rasteriser
                // smears the splat along it; the tracer ignores it, since its
                // blur is the shutter's slices.
                if (entry.motion.valid() && entry.motionScale > 0.0F) {
                    splats.back().motion = &entry.motion;
                    splats.back().motionScale = entry.motionScale;
                }
                splats.back().viewStep = viewStepOf(entry.objectToWorld, entry.transformStep);
                if (entry.relight && entry.gpu->hasVisibility()) {
                    MeasuredCloud m;
                    m.drawn = drawn;
                    m.fields = entry.gpu.get();
                    m.xforms = entry.posed != nullptr ? &entry.xforms : nullptr;
                    m.slot = frameSlots_;
                    m.rows = entry.objectToWorld.rows3x4();
                    m.categories = categoryMask(entry.categories);
                    // The skinner's bind transform, rows 0..2 of the 4x4 as it holds it.
                    for (size_t r = 0; r < 3; ++r) {
                        for (size_t c = 0; c < 4; ++c) {
                            m.geomBind[r * 4 + c] = entry.geomBind[r * 4 + c];
                        }
                    }
                    frameMeasured_.push_back(m);
                }
                frameClouds_.push_back({drawn, frameSlots_, entry.relight,
                                        entry.relight && entry.gpu->hasVisibility(), entry.objectToWorld.rows3x4()});
                frameSlots_ += drawn->count;
            }
            const lod::LodCloud* cloud = entry.pool != nullptr ? &entry.pool->cloud() : entry.lodCloud.get();
            if (cloud == nullptr) {
                continue;
            }
            if (technique == Technique::RayTraced) {
                // A cut changes every frame, and the ray tracer would rebuild
                // every frame: it draws the whole cloud, when it is whole.
                if (entry.lodCloud != nullptr) {
                    splats.push_back({&entry.lodCloud->splats, entry.objectToWorld, entry.edit, entry.relight,
                                      entry.litBody, categoryMask(entry.categories)});
                    instancePrims.push_back(id.GetString());
                    splats.back().transferIndirect = transferIndirect_.load();
                    splats.back().reflectCloud = splatReflections_.load();
                    splats.back().ior = entry.ior;
                    splats.back().catcher = entry.catcher;
                splats.back().catcher = entry.catcher;
                    frameSlots_ += entry.lodCloud->splats.count;
                } else {
                    log::warn("hdAthenea: {}: a streamed asset is drawn by the rasteriser only", id.GetString());
                }
                continue;
            }
            // Short of memory, each level given up doubles the pixels a merged
            // cell may span: a coarser cut, fewer splats.
            cuts.push_back({cloud, entry.objectToWorld, entry.edit,
                            entry.asset.threshold * static_cast<float>(1u << lodBias_)});
            poolOf.push_back(entry.pool.get());
            cutPrims.push_back(id.GetString());
        }
        for (const auto& [id, entry] : meshes_) {
            if (!entry.visible || entry.gpu == nullptr) {
                continue;
            }
            if (renderTags != nullptr && !renderTags->empty() &&
                std::find(renderTags->begin(), renderTags->end(), entry.renderTag) == renderTags->end()) {
                continue;
            }
            if (!entry.instancing.empty()) {
                // Instanced: the chain, recomposed on the device when an
                // instancer in it changed.
                std::vector<uint64_t> versions;
                bool complete = true;
                for (const InstancerLink& link : entry.instancing) {
                    const auto found = instancers_.find(link.instancer);
                    complete = complete && found != instancers_.end();
                    versions.push_back(found != instancers_.end() ? found->second.version : 0);
                }
                if (!complete) {
                    continue;
                }
                auto& mutableEntry = const_cast<MeshEntry&>(entry);
                if (entry.chainDirty || versions != entry.chainVersions) {
                    if (!instancing_.has_value()) {
                        auto made = world::Instancing::create(*library_);
                        if (!made) return std::move(made).error();
                        instancing_.emplace(std::move(*made));
                    }
                    ATHENEA_TRY(composeChains(*instancing_, instancers_, entry.instancing, mutableEntry.chain,
                                          mutableEntry.chainStart, mutableEntry.chainEnd,
                                          mutableEntry.chainTimeStart, mutableEntry.chainTimeEnd));
                    mutableEntry.chainVersions = std::move(versions);
                    mutableEntry.chainDirty = false;
                }
                world::InstanceSet set;
                set.mesh = entry.gpu;
                set.chainRows = entry.chain.rows;
                set.count = entry.chain.count;
                if (motionBuckets > 1 && entry.chainStart.count == entry.chain.count && entry.chain.count > 0 &&
                    entry.chainStart.rows.valid() && entry.chainEnd.rows.valid()) {
                    set.chainRowsStart = entry.chainStart.rows;
                    set.chainRowsEnd = entry.chainEnd.rows;
                    set.timeStart = entry.chainTimeStart;
                    set.timeEnd = entry.chainTimeEnd;
                    if (set.timeEnd == set.timeStart) {
                        set.timeStart = shutterOpen;
                        set.timeEnd = shutterClose;
                    }
                }
                set.prototype = entry.objectToWorld;
                set.primId = entry.primId;
                set.displayColor = entry.look.displayColor;
                set.displayOpacity = entry.look.displayOpacity;
                set.doubleSided = entry.look.doubleSided;
                set.material = rowOf(entry.look.material);
                set.subsetMaterials = subsetRowsOf(entry);
                set.categories = categoryMask(entry.look.categories);
                meshSets.push_back(std::move(set));
                continue;
            }
            world::MeshInstance instance;
            instance.mesh = entry.gpu;
            instance.objectToWorld = entry.objectToWorld;
            if (motionBuckets > 1 && (entry.shutter.start.has_value() || entry.shutter.end.has_value() ||
                                      entry.gpuStart != nullptr || entry.gpuEnd != nullptr)) {
                // The samples' times: the transform's, or the points' where
                // only they move. Where both move at different times, the
                // transform's are taken and the points are placed at them --
                // a compromise, noted in the docs.
                world::MeshMotion motion;
                motion.objectToWorldStart = entry.shutter.start.value_or(entry.objectToWorld);
                motion.objectToWorldEnd = entry.shutter.end.value_or(entry.objectToWorld);
                motion.meshStart = entry.gpuStart;
                motion.meshEnd = entry.gpuEnd;
                const bool transformMoves = entry.shutter.start.has_value() || entry.shutter.end.has_value();
                motion.timeStart = transformMoves ? entry.shutter.timeStart : entry.pointsTimeStart;
                motion.timeEnd = transformMoves ? entry.shutter.timeEnd : entry.pointsTimeEnd;
                if (motion.timeEnd == motion.timeStart) {
                    motion.timeStart = shutterOpen;
                    motion.timeEnd = shutterClose;
                }
                instance.motion = motion;
            }
            instance.primId = entry.primId;
            instance.displayColor = entry.look.displayColor;
            instance.displayOpacity = entry.look.displayOpacity;
            instance.doubleSided = entry.look.doubleSided;
            instance.material = rowOf(entry.look.material);
            // Which material a mesh ended up with, for the days when a stage
            // arrives grey and the question is whether the material failed or
            // never reached the geometry.
            log::debug("hdAthenea: mesh {} binds {} (row {})", id.GetString(),
                       entry.look.material.IsEmpty() ? std::string("nothing") : entry.look.material.GetString(),
                       instance.material);
            instance.subsetMaterials = subsetRowsOf(entry);
            instance.categories = categoryMask(entry.look.categories);
            meshInstances.push_back(std::move(instance));
        }
        for (const auto& [id, entry] : points_) {
            if (entry.visible && entry.gpu != nullptr) {
                points.push_back({entry.gpu.get(), entry.objectToWorld, entry.style});
            }
        }
    }
    if (!cuts.empty()) {
        if (!cutter_.has_value()) {
            auto made = lod::CutSelector::create(*library_);
            if (!made) return std::move(made).error();
            cutter_.emplace(std::move(*made));
        }
        const bool streamed = std::any_of(poolOf.begin(), poolOf.end(), [](auto* p) { return p != nullptr; });
        std::vector<lod::CutStats>& stats = cutStats;
        const auto placeCuts = [&](const std::vector<render::SplatInstance>& selected) {
            splats.insert(splats.end(), selected.begin(), selected.end());
            for (size_t k = 0; k < selected.size(); ++k) {
                instancePrims.push_back(k < cutPrims.size() ? cutPrims[k] : std::string());
            }
        };
        for (int round = 0;; ++round) {
            auto selected = cutter_->select(projection, cuts, 0.0F, streamed ? &stats : nullptr);
            if (!selected) return std::move(selected).error();
            if (!streamed) {
                placeCuts(*selected);
                break;
            }
            for (size_t k = 0; k < poolOf.size(); ++k) {
                if (poolOf[k] != nullptr) {
                    poolOf[k]->want(stats[k].needs);
                }
            }
            uint32_t placed = 0;
            for (size_t k = 0; k < poolOf.size(); ++k) {
                const auto earlier = poolOf.begin() + static_cast<std::ptrdiff_t>(k);
                // A pool two prims share is updated once.
                if (poolOf[k] != nullptr && std::find(poolOf.begin(), earlier, poolOf[k]) == earlier) {
                    auto n = poolOf[k]->update(settleStreams);
                    if (!n) return std::move(n).error();
                    placed += *n;
                }
            }
            if (!settleStreams || placed == 0 || round >= 64) {
                placeCuts(*selected);
                break;
            }
        }
    }
    // WHAT A PICK SAID EACH PRIM IS MADE OF, handed to every cloud in the
    // frame. The ids in the table are the Cryptomatte's, which belong to the
    // frame and not to one cloud, so one table serves them all.
    ATHENEA_TRY(commitSplatOverrides());
    if (splatOverrideRows_ > 0) {
        for (render::SplatInstance& instance : splats) {
            instance.overrides = &splatOverrideBuffer_;
            instance.overrideCount = splatOverrideRows_;
        }
    }
    // Opaque layers first -- meshes, points -- then splats blended over them.
    const bool drawMeshes = !meshInstances.empty() || !meshSets.empty();
    // Volumes are path traced through the mesh layer, whose visibility then
    // finds nothing and every sample walks its camera ray: a traced frame
    // with volumes has that layer even when it has no mesh.
    bool volumesInFrame = false;
    if (technique == Technique::RayTraced) {
        const std::lock_guard<std::mutex> held(guard_);
        for (const auto& [id, volume] : volumes_) {
            volumesInFrame = volumesInFrame || volume.visible;
        }
    }
    const gpu::Caps& caps = device_->caps();
    if (countSplats_.load()) {
        const bool tracedAlone = technique == Technique::RayTraced && !drawMeshes && !volumesInFrame;
        noteGaussians(tracedAlone ? "rt" : technique == Technique::RayTraced ? "rt+raster" : "raster", splats,
                      instancePrims, cutStats, cutPrims, drawnLevels);
    }
    // A frame of nothing but splats is GaussianRayTracer's, and it writes the
    // whole image: there is no layer to compose under it, so it returns here.
    // With meshes in the frame the traced technique means something else --
    // the surfaces are path traced below and the splats composed over them by
    // the rasteriser, because the tracer takes no `under` layer. Splats inside
    // the rays is still to be written (docs/decisions.md, M6).
    if (technique == Technique::RayTraced && !drawMeshes && !volumesInFrame) {
        if (!points.empty()) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                log::warn("hdAthenea: a traced frame of splats alone leaves the points out");
            }
        }
        if (!rayTracer_.has_value()) {
            auto made = render::GaussianRayTracer::create(*library_);
            if (!made) return std::move(made).error();
            rayTracer_.emplace(std::move(*made));
        }
        // A cloud whose prim asked to be relit is relit here as it is in the
        // rasteriser: the flag used to mean nothing on this route, so a cloud
        // converted from a mesh came back as its own albedo, unlit.
        render::SplatLights tracedLights;
        const bool relitTraced = std::any_of(splats.begin(), splats.end(),
                                             [](const render::SplatInstance& s) { return s.relight; });
        if (relitTraced && !lamps.empty()) {
            ATHENEA_TRY(prepareSplatLights(lamps, splats));
            if (lightTable_->count() > 0) {
                tracedLights.records = &lightTable_->records();
                bindEnvironment(tracedLights);
                tracedLights.count = lightTable_->count();
            }
        }
        ATHENEA_TRY(measureVisibility(tracedLights));
        auto tracedStats = rayTracer_->render(projection, splats, settings, targets, &tracedLights);
        if (!tracedStats) return std::move(tracedStats).error();
        counters_ = {tracedStats->splats, 0u, 0u, 0u,
                     lightTable_.has_value() ? lightTable_->count() : 0u, false};
        if (countSplats_.load()) {
            GaussianStats& g = gaussianStats_;
            g.traced = true;
            g.rebuilt = tracedStats->rebuilt;
            g.buildMs = tracedStats->buildMs;
            g.traceMs = tracedStats->renderMs;
            g.tracedMs = tracedStats->totalMs;
            g.tracedSplats = tracedStats->splats;
            g.tracedChunks = tracedStats->chunks;
            g.traceRoute = tracedStats->route == render::RayTracingRoute::Hardware ? "hardware" : "compute BVH";
            // The counts a raster frame left are not this frame's.
            g.counted = false;
            for (GaussianCloudStats& cloud : g.clouds) {
                cloud.counted = false;
            }
        }
        // What every other route does when it has finished drawing, and what
        // this one used to return without: the sky behind the frame, and the
        // camera's exposure. A stage of nothing but splats came back over
        // black where the rasteriser drew it over its dome, which is the sort
        // of difference that is only ever noticed by putting the two side by
        // side.
        ATHENEA_TRY(paintDomes(projection, settings.width, settings.height, targets));
        ATHENEA_TRY(applyExposure(projection.exposure, settings.width, settings.height, targets));
        return ok();
    }
    if (visibility == MeshVisibility::Automatic) {
        // Rays first: a draw costs the host a few microseconds to record, and
        // Kitchen_set's 1800 of them outweigh a frame of rays at any size
        // measured (docs/decisions.md, M2).
        visibility = caps.rayQuery && caps.accelerationStructure ? MeshVisibility::Rays
                     : caps.rasterization                         ? MeshVisibility::Raster
                                                                  : MeshVisibility::Bvh;
    }
    if (drawMeshes && visibility == MeshVisibility::Raster && !caps.rasterization) {
        return Error(ErrorCode::Unsupported, "mesh visibility by raster: the device does not rasterise");
    }
    // Rays inline or rays in a pipeline: VisibilityTrace takes either, and on
    // CUDA the second is the only one there is (OptiX, no RayQuery).
    if (drawMeshes && visibility == MeshVisibility::Rays &&
        !(caps.accelerationStructure && (caps.rayQuery || caps.rayTracing))) {
        return Error(ErrorCode::Unsupported, "mesh visibility by rays: the device has no ray tracing");
    }
    const bool meshLayer = drawMeshes || volumesInFrame;
    // WHAT THE FRAME'S IDS ARE CALLED, before anything is drawn: the mesh
    // pass reads the table, and a frame of splats alone has no mesh pass at
    // all while its manifest is still what a reader needs.
    if (aovRequest.cryptomatte) {
        ATHENEA_TRY(prepareCryptoTable());
    }
    // A cloud asked to be relit needs the frame's lights as much as a mesh
    // does, and a frame of splats alone has no mesh layer to build them for:
    // before this, a stage of a relit capture under a light showed what it
    // was baked with, because the table was never made.
    const bool relitSplats = std::any_of(splats.begin(), splats.end(),
                                         [](const render::SplatInstance& s) { return s.relight; });
    const bool lightsWanted = meshLayer || relitSplats;
    const bool pathTracing = meshLayer && technique == Technique::RayTraced;
    if (!pathTracing) {
        pathState_.traced = false;
        pathAuxValid_ = false;
    }
    const auto meshLayerStart = std::chrono::steady_clock::now();
    if (meshLayer) {
        // A light that moves cuts the frame into shutter slices as geometry
        // that moves does. The mesh scene is the mesh layer's; a frame of
        // relit splats wants the lights and has no geometry to update.
        const bool lightsMove = motionBuckets > 1 && std::any_of(lamps.begin(), lamps.end(), [](const light::Light& l) {
                                    return l.movesUnderShutter();
                                });
        if (meshLayer) {
            ATHENEA_TRY(scene_->update(meshInstances, projection, meshSets, motionBuckets, shutterOpen, shutterClose,
                                   lightsMove));
        }
        // The frame's lights, and what a shadow ray traces against: rays
        // shadow whatever route found the visibility, so the structure is
        // built even where the rasteriser drew. All of this uploads and
        // builds -- each submitting a batch of its own -- so it happens
        // before the frame's batch opens: a submit inside a batch that has
        // already recorded work releases what that work still refers to.
        if (!lightTable_.has_value()) {
            auto made = light::LightTable::create(*library_);
            if (!made) return std::move(made).error();
            lightTable_.emplace(std::move(*made));
        }
        ATHENEA_TRY(prepareLightImages(lamps));
        // The scene's reach, for a dome's or a sun's share of the lights'
        // power: its bounds are a kernel's, read again when the mesh set or
        // its points change.
        if (scene_->generation() != lightRadiusGeneration_ || scene_->positionsRevision() != lightRadiusRevision_) {
            lightRadiusGeneration_ = scene_->generation();
            lightRadiusRevision_ = scene_->positionsRevision();
            auto bounds = scene_->worldBounds();
            if (!bounds) return std::move(bounds).error();
            lightSceneRadius_ = 1.0F;
            if (bounds->has_value()) {
                const scene::Bounds& b = **bounds;
                const float dx = b.max[0] - b.min[0], dy = b.max[1] - b.min[1], dz = b.max[2] - b.min[2];
                lightSceneRadius_ = std::max(0.5F * std::sqrt(dx * dx + dy * dy + dz * dz), 1.0e-3F);
            }
        }
        ATHENEA_TRY(lightTable_->set(lamps, lightSceneRadius_));
        ATHENEA_TRY(prepareEnvironment(lamps));
        // A path traced surface needs a structure whatever the lights do --
        // its bounce is a ray -- while shading needs one only where a light
        // casts a shadow. The same structure serves both, and a path traced
        // frame without it would trace against nothing and never know.
        bool raysWanted = false;
        if ((lightTable_->anyShadow() || pathTracing) && caps.rayQuery && caps.accelerationStructure) {
            if (!rayTracingScene_.has_value()) {
                auto accel = world::RayTracingScene::create(*library_);
                if (!accel) return std::move(accel).error();
                rayTracingScene_.emplace(std::move(*accel));
            }
            if (visibility != MeshVisibility::Rays) {
                ATHENEA_TRY(rayTracingScene_->build(*scene_));   // the rays route builds it in the pass below
            }
            raysWanted = true;
        }
        // What a material needs wherever it is evaluated: shading always,
        // visibility only where a material cuts its samples away.
        // Splats a mesh is shadowed by: the proxies of every cloud in the
        // frame, and their tables packed into one buffer. Only path traced,
        // only where the device traces inline, and only when the setting asks
        // -- the same setting a cloud's own shadow answers to.
        technique::SplatShadows* meshSplatShadows = nullptr;
        if (pathTracing && !splats.empty() && splatShadows_.load() && device_->caps().rayQuery &&
            device_->caps().accelerationStructure && splatShadowsFit(gaussiansOf(splats))) {
            if (!shadowTracer_.has_value()) {
                render::RayTracerSettings rtSettings;
                rtSettings.route = render::RayTracingRoute::Hardware;
                auto made = render::GaussianRayTracer::create(*library_, rtSettings);
                if (!made) return std::move(made).error();
                shadowTracer_.emplace(std::move(*made));
            }
            if (!splatShadowScene_.has_value()) {
                auto made = technique::SplatShadows::create(*library_);
                if (!made) return std::move(made).error();
                splatShadowScene_.emplace(std::move(*made));
            }
            auto prepared = shadowTracer_->prepare(projection, splats, settings.maxShDegree);
            if (!prepared) return std::move(prepared).error();
            shadowTracerReady_ = true;
            {
                gpu::CommandBatch packing(*device_);
                ATHENEA_TRY(splatShadowScene_->prepare(packing, shadowTracer_->shadowScene()));
                ATHENEA_TRY(packing.submit(true));
            }
            if (splatShadowScene_->valid()) {
                meshSplatShadows = &*splatShadowScene_;
            }
        }
        // WHAT THE CLOUDS OF THIS FRAME STOP, SEEN FROM THE LIGHTS.
        //
        // One pass a light over the gaussians, no ray and no sort, into a map
        // a mesh's shading reads. It is what shadows a floor under a bird in
        // the raster route -- which had no cloud shadow of any kind -- and it
        // is the only kind a device without ray tracing can have.
        meshShadowMapMs_ = 0.0;
        if (cloudShadows_.load() && !splats.empty() && lightTable_.has_value() && lightTable_->count() > 0) {
            if (!cloudShadowMap_.has_value()) {
                auto made = technique::SplatShadowMap::create(*library_);
                if (!made) return std::move(made).error();
                cloudShadowMap_.emplace(std::move(*made));
            }
            technique::ShadowMapJob job;
            job.lights = &lightTable_->records();
            job.lightCount = lightTable_->count();
            job.resolution = cloudShadowTexels_.load();
            // A gaussian standing inside the cloud cannot be answered with
            // the total -- it would read the whole cloud's shadow and the
            // cloud would go flat -- so a frame with a relit cloud that has
            // no baked field of its own gets the Fourier terms as well.
            const bool cloudReceives =
                std::any_of(frameClouds_.begin(), frameClouds_.end(),
                            [](const SlottedCloud& c) { return c.relight && !c.hasField; });
            const uint32_t asked = cloudShadowTerms_.load();
            job.coefficients = asked != 0 ? asked : (cloudReceives ? 5u : 1u);
            job.density = cloudShadowDensity_.load();
            // A dome among the lights casts too, along six of its directions
            // (its zenith and a ring forty degrees up), in the slots the
            // lights leave: the car on the ground under a sky (task TX).
            job.domeSlots = 6;
            for (const render::SplatInstance& instance : splats) {
                // A shadow catcher is a shadow already: cast into the map, it
                // laid its own patch's outline on the ground beneath it once
                // more along each of the dome's directions -- the straight-
                // edged blocks under goegap's sun, and the mesh car's frame a
                // third dark.
                if (instance.splats == nullptr || instance.splats->count == 0 || instance.catcher) {
                    continue;
                }
                technique::ShadowMapCaster caster;
                caster.cloud = instance.splats;
                caster.positions = &instance.splats->positions;
                caster.objectToWorld = instance.objectToWorld.rows3x4();
                caster.categories = instance.categories;
                caster.restBounds = instance.splats->restBounds.value_or(instance.splats->bounds);
                job.casters.push_back(caster);
            }
            // A MAP IS THE CASTERS' AND THE LIGHTS', NOT THE CAMERA'S: a frame
            // whose clouds, poses, transforms, lights and settings are what the
            // last map was built from reads that map again (task PLAY-G: seven
            // passes over the Corvette's 14.7 M gaussians were 77 ms a frame of
            // a car that does not move). The key is bookkeeping the host holds:
            // which clouds, their revision (a pose counts it up), where they
            // stand, the lights' records and the job. A frame with levels of
            // detail always builds -- its cut's clouds are the camera's.
            std::vector<uint64_t> key;
            const auto keyBytes = [&key](const void* data, size_t bytes) {
                const size_t at = key.size();
                key.resize(at + (bytes + 7) / 8, 0);
                std::memcpy(key.data() + at, data, bytes);
            };
            for (const technique::ShadowMapCaster& caster : job.casters) {
                const uint64_t ids[4] = {reinterpret_cast<uintptr_t>(caster.cloud),
                                         reinterpret_cast<uintptr_t>(caster.cloud->positions.rhi()),
                                         caster.cloud->count, caster.cloud->revision};
                keyBytes(ids, sizeof(ids));
                keyBytes(caster.objectToWorld.data(), sizeof(float) * 12);
                keyBytes(&caster.categories, sizeof(caster.categories));
                keyBytes(&caster.cloud->bounds, sizeof(caster.cloud->bounds));
                keyBytes(&*caster.restBounds, sizeof(scene::Bounds));
            }
            for (const light::Light& lamp : lamps) {
                const light::LightRecord record = light::LightTable::recordOf(lamp);
                keyBytes(&record, sizeof(record));
            }
            const uint32_t settingsKey[4] = {job.resolution, job.coefficients, job.domeSlots, job.lightCount};
            keyBytes(settingsKey, sizeof(settingsKey));
            keyBytes(&job.density, sizeof(job.density));
            const bool sameMap = cuts.empty() && cloudShadowMap_->valid() && key == cloudShadowKey_;
            if (!job.casters.empty() && !sameMap) {
                const auto castStart = std::chrono::steady_clock::now();
                gpu::CommandBatch casting(*device_);
                ATHENEA_TRY(cloudShadowMap_->build(casting, job));
                ATHENEA_TRY(casting.submit(true));
                meshShadowMapMs_ =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - castStart).count();
                cloudShadowKey_ = std::move(key);
            }
            if (cloudShadowMap_->valid()) {
                if (std::getenv("ATHENEA_SHADOW_DEBUG") != nullptr) {
                    auto mean = cloudShadowMap_->meanTransmittance(0);
                    if (mean) {
                        log::info("cloud shadow: {} casters, {} lights, {} texels, mean transmittance {:.4f}",
                                  job.casters.size(), job.lightCount, job.resolution, *mean);
                    }
                }
            }
        } else if (cloudShadowMap_.has_value()) {
            cloudShadowMap_.reset();   // nothing casts: the kernel goes back to the one without a map
            cloudShadowKey_.clear();
        }
        technique::MaterialFrame frame;
        frame.programs = &*materialPrograms_;
        frame.alphaDither = technique == Technique::RayTraced ? 0u : 1u;
        frame.cutouts = materialCutouts_ || scene_->anyHidden();
        frame.scene = &*scene_;
        frame.records = &materialRecords_;
        frame.blob = &materialBlob_;
        frame.textures = textures_.get();
        frame.lights = &*lightTable_;
        frame.samples = lightSamples_.load();
        frame.chooseLights = chooseLights_.load();
        // The light groups' planes, one buffer for the frame's groups.
        if (aovRequest.lightGroups.size() > technique::kMaxLightGroups) {
            return Error::make(ErrorCode::InvalidArgument, "{} light groups asked for; at most {}",
                               aovRequest.lightGroups.size(), technique::kMaxLightGroups);
        }
        lightGroupCount_ = static_cast<uint32_t>(aovRequest.lightGroups.size());
        if (lightGroupCount_ > 0) {
            const uint64_t pixels = uint64_t{settings.width} * settings.height;
            const uint64_t bytes = pixels * lightGroupCount_ * 16;
            if (!lightGroupColour_.valid() || lightGroupColour_.bytes() < bytes || lightGroupPixels_ != pixels) {
                gpu::BufferDesc desc;
                desc.bytes = bytes;
                desc.elementBytes = 16;
                desc.label = "lights.groups.colour";
                auto colour = gpu::Buffer::create(*device_, desc);
                if (!colour) return std::move(colour).error();
                lightGroupColour_ = std::move(*colour);
            }
            lightGroupPixels_ = pixels;
            frame.groups = {&lightGroupColour_, lightGroupCount_};
        }
        // The frame's volumes, laid out again whenever a volume or a field
        // changed: each grid read once per file and name.
        if (pathTracing) {
            std::vector<world::VolumeInput> inputs;
            std::vector<std::shared_ptr<const io::NanoGrid>> holding;
            bool rebuild = false;
            {
                const std::lock_guard<std::mutex> held(guard_);
                rebuild = volumesVersion_ != volumesBuilt_;
                if (rebuild) {
                    volumesBuilt_ = volumesVersion_;
                    for (const auto& [id, volume] : volumes_) {
                        if (!volume.visible || volume.field.IsEmpty()) {
                            continue;
                        }
                        const auto field = volumeFields_.find(volume.field);
                        if (field == volumeFields_.end() || field->second.path.empty()) {
                            continue;   // its field has not arrived
                        }
                        const auto key = std::make_pair(field->second.path, field->second.gridName);
                        auto cached = nanoGrids_.find(key);
                        if (cached == nanoGrids_.end()) {
                            auto read = io::readVdbGrid(field->second.path, field->second.gridName);
                            if (!read) {
                                log::warn("hdAthenea: volume {}: {}", id.GetString(), read.error().toString());
                            }
                            cached = nanoGrids_
                                         .emplace(key, read ? std::make_shared<const io::NanoGrid>(std::move(*read))
                                                            : std::shared_ptr<const io::NanoGrid>())
                                         .first;
                        }
                        if (cached->second == nullptr) {
                            continue;
                        }
                        world::VolumeInput input;
                        input.grid = cached->second.get();
                        input.objectToWorld = volume.objectToWorld;
                        input.densityScale = volume.densityScale;
                        input.albedo = volume.albedo;
                        input.g = volume.g;
                        // THE MATERIAL'S WORD, WHERE THE VOLUME BINDS ONE. The
                        // specification puts a medium's shading in the Material
                        // bound to the Volume (a `volume` terminal), not on the
                        // prim; the three primvars stay for a Volume without.
                        // MaterialX gives absorption and scattering per unit
                        // density; the medium here walks one extinction and
                        // scatters with an albedo, so extinction is their sum's
                        // mean over the channels and the albedo the ratio a
                        // channel: a coloured extinction is carried by the
                        // albedo, not by the free flight.
                        if (!volume.material.IsEmpty()) {
                            const auto bound = materials_.find(volume.material);
                            if (bound != materials_.end() && bound->second.volume.has_value()) {
                                const material::VolumeCoefficients& c = *bound->second.volume;
                                float extinction = 0.0F;
                                for (size_t k = 0; k < 3; ++k) {
                                    const float sum = c.absorption[k] + c.scattering[k];
                                    input.albedo[k] = sum > 0.0F ? c.scattering[k] / sum : 0.0F;
                                    extinction += sum / 3.0F;
                                }
                                input.densityScale = extinction;
                                input.g = c.anisotropy;
                                input.emission = c.emission;
                            }
                        }
                        inputs.push_back(input);
                        holding.push_back(cached->second);
                    }
                }
            }
            if (rebuild) {
                if (!volumeSet_.has_value()) {
                    auto made = world::VolumeSet::create(*library_);
                    if (!made) return std::move(made).error();
                    volumeSet_.emplace(std::move(*made));
                }
                gpu::CommandBatch volumeBatch(*device_);
                ATHENEA_TRY(volumeSet_->set(volumeBatch, inputs));
                ATHENEA_TRY(volumeBatch.submit(true));
                volumesDrawn_ = static_cast<uint32_t>(inputs.size());
            }
            if (volumesDrawn_ > 0) {
                frame.volumes = &volumeSet_->words();
                frame.volumeCount = volumesDrawn_;
            }
        } else if (!volumesUndrawnSaid_) {
            const std::lock_guard<std::mutex> held(guard_);
            if (!volumes_.empty()) {
                volumesUndrawnSaid_ = true;
                log::info("hdAthenea: volumes are drawn by the rt technique; the raster technique draws none");
            }
        }
        // The emitting triangles, for the path tracer's next event estimation:
        // probed and weighed on the device again whenever what they depend on
        // changed -- the scene, its positions, the materials, anything the
        // revision counts. Not through media, whose kernel does not sample them.
        if (pathTracing && frame.volumeCount == 0) {
            const EmissiveKey key{scene_->generation(), scene_->positionsRevision(), revision_.load(),
                                  materialRecords_.rhi(), scene_->instanceCount()};
            if (!emissiveTable_.has_value()) {
                auto made = technique::EmissiveTable::create(*library_);
                if (!made) return std::move(made).error();
                emissiveTable_.emplace(std::move(*made));
            }
            if (!(key == emissiveKey_)) {
                ATHENEA_TRY(emissiveTable_->build(frame, static_cast<uint32_t>(materialRecords_.count())));
                emissiveKey_ = key;
            }
            if (emissiveTable_->totalPower() > 0.0F) {
                frame.emissive = &emissiveTable_->table();
                frame.emissivePower = emissiveTable_->totalPower();
            }
        }
        const technique::MaterialFrame* cutouts = (materialCutouts_ || scene_->anyHidden()) ? &frame : nullptr;
            const auto meshStart = std::chrono::steady_clock::now();
            gpu::CommandBatch batch(*device_);
        switch (visibility) {
        case MeshVisibility::Automatic:
        case MeshVisibility::Raster:
            if (!visibilityRaster_.has_value()) {
                auto made = technique::VisibilityRaster::create(*library_);
                if (!made) return std::move(made).error();
                visibilityRaster_.emplace(std::move(*made));
            }
            ATHENEA_TRY(visibilityRaster_->render(batch, *scene_, projection, settings.width, settings.height,
                                              visibility_, cutouts));
            break;
        case MeshVisibility::Rays:
            if (!visibilityTrace_.has_value()) {
                auto accel = world::RayTracingScene::create(*library_);
                if (!accel) return std::move(accel).error();
                rayTracingScene_.emplace(std::move(*accel));
                auto made = technique::VisibilityTrace::create(*library_);
                if (!made) return std::move(made).error();
                visibilityTrace_.emplace(std::move(*made));
            }
            ATHENEA_TRY(rayTracingScene_->build(*scene_));
            ATHENEA_TRY(visibilityTrace_->render(batch, *rayTracingScene_, projection, settings.width,
                                             settings.height, visibility_, cutouts));
            break;
        case MeshVisibility::Bvh:
            if (!visibilityBvh_.has_value()) {
                auto bvh = world::BvhScene::create(*library_);
                if (!bvh) return std::move(bvh).error();
                bvhScene_.emplace(std::move(*bvh));
                auto made = technique::VisibilityBvh::create(*library_);
                if (!made) return std::move(made).error();
                visibilityBvh_.emplace(std::move(*made));
            }
            ATHENEA_TRY(bvhScene_->build(*scene_));
            ATHENEA_TRY(visibilityBvh_->render(batch, *scene_, *bvhScene_, projection, settings.width,
                                           settings.height, visibility_, cutouts));
            break;
        }
        // ATHENEA_STAGES: the visibility pass on its own, waited for.
        const bool meshStages = !platform::env("ATHENEA_STAGES").empty();
        double meshVisibilityMs = 0.0;
        if (meshStages) {
            ATHENEA_TRY(batch.submit(true));
            meshVisibilityMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - meshStart).count();
        }
        // Read after the visibility pass, never before it: the rays route
        // rebuilds this structure there, and the one it replaces is released
        // with it -- taking the pointer earlier left shading tracing against
        // a freed structure.
        frame.shadows = raysWanted ? rayTracingScene_->topLevel() : nullptr;
        frame.splatShadows = meshSplatShadows;
        frame.cloudShadow =
            cloudShadowMap_.has_value() && cloudShadowMap_->valid() ? cloudShadowMap_->textureView() : nullptr;
        frame.cloudShadowChain =
            cloudShadowMap_.has_value() && cloudShadowMap_->valid() ? cloudShadowMap_->chainView() : nullptr;
        // The domes read prefiltered on the raster route (task PLAY-G): the
        // sky prepared once (prepareEnvironment), no sample and no ray. The
        // path tracer samples them, as the ground truth must.
        frame.domeLighting = domePrefiltered_.load() && !pathTracing && environment_.has_value() &&
                                     environment_->ready()
                                 ? environment_->meshView()
                                 : nullptr;
        frame.domesOnly = frame.domeLighting != nullptr && !lamps.empty() &&
                          lamps.size() <= technique::kEnvironmentDomes &&
                          std::all_of(lamps.begin(), lamps.end(),
                                      [](const light::Light& l) { return l.kind == light::LightKind::Dome; });
        if (pathTracing) {
            if (!pathTracer_.has_value()) {
                auto made = technique::PathTracer::create(*library_);
                if (!made) return std::move(made).error();
                pathTracer_.emplace(std::move(*made));
            }
            ATHENEA_TRY(pathTracer_->setPrograms(*materialPrograms_));
            technique::PathSettings paths;
            paths.samples = pathSamples_.load();
            paths.bounces = pathBounces_.load();
            paths.adaptive = pathAdaptive_.load();
            paths.mis = pathMis_.load();
            paths.errorTarget = pathError_.load();
            paths.headlight = frame.lights == nullptr || frame.lights->count() == 0;
            PathState now;
            now.worldToView = projection.worldToView;
            now.focalX = projection.focalX;
            now.lensRadius = projection.lensRadius;
            now.focusDistance = projection.focusDistance;
            now.distortionK1 = projection.distortionK1;
            now.distortionK2 = projection.distortionK2;
            now.focalY = projection.focalY;
            // The camera the caller asked for, not this pass's offset: the
            // jitter is what the mean is made of and must not restart it.
            now.centreX = base.centreX;
            now.centreY = base.centreY;
            now.nearZ = projection.nearZ;
            now.farZ = projection.farZ;
            now.orthographic = projection.orthographic;
            now.width = settings.width;
            now.height = settings.height;
            now.samples = paths.samples;
            now.bounces = paths.bounces;
            now.adaptive = paths.adaptive;
            now.mis = paths.mis;
            now.cameraMoves = projection.cameraMoves;
            now.cameraStart = projection.viewToWorldStart;
            now.cameraEnd = projection.viewToWorldEnd;
            now.error = paths.errorTarget;
            now.revision = revision_.load();
            if (renderTags != nullptr) {
                // Which purposes are drawn is part of what a path finds.
                uint64_t hash = 1469598103934665603ull;
                for (const pxr::TfToken& tag : *renderTags) {
                    for (const char c : tag.GetString()) {
                        hash = (hash ^ static_cast<uint8_t>(c)) * 1099511628211ull;
                    }
                    hash = (hash ^ 0x2Cu) * 1099511628211ull;
                }
                // And which light groups are gathered: their planes accumulate
                // with the colour and stand only while the same ones do.
                for (const std::string& group : aovRequest.lightGroups) {
                    for (const char c : group) {
                        hash = (hash ^ static_cast<uint8_t>(c)) * 1099511628211ull;
                    }
                    hash = (hash ^ 0x3Bu) * 1099511628211ull;
                }
                now.tags = hash;
            }
            now.traced = true;
            // The same frame continued, or a new one: a camera that moved, a
            // scene that changed or a setting that did all start the mean
            // again, and only an identical frame adds to it. This is what
            // makes a viewport converge while it is left alone.
            const bool same = now == pathState_;
            paths.accumulate = same;
            if (!same) {
                pathTracer_->restart();
                pathSeed_ = 0;
                pathState_ = now;
            }
            paths.seed = pathSeed_++ * 7919u;
            // The denoiser's guides only where something asks for them: they
            // cost a buffer, and on Metal that buffer is the one a cloud's
            // shadow tables want (technique::PathTracer says which wins).
            const bool wantAux = denoise_.load() || aovRequest.aux;
            if (bake != nullptr) {
                // The frame's own work, at the caller's points instead of at
                // this frame's pixels. One pass, its own mean: a bake does not
                // converge over frames the way a viewport does.
                technique::BakePoints points;
                points.rays = bake->rays;
                points.count = bake->count;
                points.width = std::min(bake->count, 4096u);
                points.height = (bake->count + points.width - 1) / points.width;
                points.coefficients = bake->coefficients;
                points.transfer = bake->transfer;
                points.cellSide = bake->transfer ? technique::transferCellSide(bake->cellSide) : 0u;
                points.split = bake->split;
                paths.seed = bake->seed;
                paths.accumulate = false;
                paths.adaptive = false;
                // A bake has no camera, so it cannot have a headlight: what
                // would be baked in is a lamp standing wherever the frame's
                // one-pixel camera happened to be. A stage with no lights
                // bakes black, and says so.
                paths.headlight = false;
                paths.samples = std::max(bake->samples, 1u);
                paths.bounces = bake->bounces;
                pathTracer_->restart();
                ATHENEA_TRY(pathTracer_->bake(batch, visibility_, projection, frame, paths, points, *bake->out));
                ATHENEA_TRY(batch.submit(true));
                pathState_ = {};
                return ok();
            }
            ATHENEA_TRY(pathTracer_->trace(batch, visibility_, projection, frame, paths, meshLayer_,
                                       wantAux ? &pathAux_ : nullptr));
            pathAuxValid_ = wantAux;
        } else {
            materialShading_->timeStages(meshStages);
            ATHENEA_TRY(materialShading_->shade(batch, visibility_, projection, frame, meshLayer_));
        }
        // WHICH PRIM EACH PIXEL'S SURFACE IS, for the matte the splat blend
        // builds: this layer's ids travel with it as its `cryptoIds`, and the
        // blend names by them whatever coverage the layer closed the walk with.
        if (aovRequest.cryptomatte) {
            if (!cryptoShading_.has_value()) {
                auto made = technique::CryptoShading::create(*library_);
                if (!made) return std::move(made).error();
                cryptoShading_.emplace(std::move(*made));
            }
            ATHENEA_TRY(cryptoShading_->shade(batch, *scene_, visibility_, projection, cryptoOfPrim_,
                                          cryptoPrimCount_, meshLayer_.cryptoIds));
        }
        if (aovRequest.ids || aovRequest.normals || !aovRequest.primvars.empty()) {
            if (!aovShading_.has_value()) {
                auto made = technique::AovShading::create(*library_);
                if (!made) return std::move(made).error();
                aovShading_.emplace(std::move(*made));
            }
            std::vector<uint32_t> slots;
            for (const std::string& name : aovRequest.primvars) {
                slots.push_back(scene_->slotOf(name));
            }
            ATHENEA_TRY(aovShading_->shade(batch, *scene_, visibility_, projection, slots, aovs_));
            aovsValid_ = true;
        }
        ATHENEA_TRY(batch.submit(true));
        // ATHENEA_STAGES=1: what the mesh layer took, a line a frame -- the
        // cloud map from the lights, and the visibility and shading passes.
        if (meshStages) {
            const technique::MaterialShading::StageTimes& t = materialShading_->stageTimes();
            log::info("stages: mesh prepare {:.1f} ms (cloud shadow map {:.1f} of it), visibility {:.1f}, lobes {:.1f}, "
                      "shadow rays {:.1f}, shading {:.1f}, domes {:.1f}; mesh layer {:.1f} in all",
                      std::chrono::duration<double, std::milli>(meshStart - meshLayerStart).count(),
                      meshShadowMapMs_, meshVisibilityMs, t.lobes, t.shadows, t.shade, t.domes,
                      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - meshLayerStart)
                          .count());
        }
        // The adaptive gate's counters, after the pass: which covered pixels
        // have stopped. Its own dispatch and readback, so after the batch.
        if (pathTracing && pathState_.adaptive) {
            auto progress = pathTracer_->progress(visibility_);
            if (!progress) return std::move(progress).error();
            pathProgress_ = *progress;
        }
        // After the frame's batch, never inside it: the denoiser submits work
        // of its own and waits for OIDN. It runs on a path traced frame that
        // has gathered what it was asked for, in place over the mean -- the
        // input is copied to OIDN's staging before OIDN writes anything.
        if (pathTracing && denoise_.load() && !denoiserFailed_ && pathTracer_->accumulated() >= pathTotal_.load()) {
            if (!denoiser_.has_value()) {
                auto made = technique::Denoiser::create(*library_);
                if (!made) {
                    denoiserFailed_ = true;
                    log::warn("hdAthenea: no denoiser: {}", made.error().toString());
                } else {
                    denoiser_.emplace(std::move(*made));
                }
            }
            if (denoiser_.has_value()) {
                if (auto ran = denoiser_->denoise(meshLayer_.colour, &pathAux_.planes, 0, pathAux_.normalOffsetBytes(),
                                                 meshLayer_.colour, settings.width, settings.height);
                    !ran) {
                    denoiserFailed_ = true;
                    log::warn("hdAthenea: denoising stopped: {}", ran.error().toString());
                }
            }
        }
    }
    // A frame of relit splats and no mesh layer: the lights, and nothing else
    // of what the mesh layer needs. Before this the table was never built and
    // a relit cloud showed what it was baked with.
    // A relit cloud wants them to be lit by; a baked one wants the dome it
    // stands under painted behind it, which is the same table.
    if (!meshLayer && !lamps.empty()) {
        ATHENEA_TRY(prepareSplatLights(lamps, splats));
    }
    const bool pointLayer = !points.empty() && pointRasterizer_.has_value();
    if (pointLayer) {
        ATHENEA_TRY(pointRasterizer_->render(projection, points, settings, pointLayer_));
    }
    const render::RenderTargets* under = nullptr;
    if (meshLayer && pointLayer) {
        const uint64_t pixels = uint64_t{settings.width} * settings.height;
        if (opaqueLayer_.width != settings.width || opaqueLayer_.height != settings.height ||
            !opaqueLayer_.colour.valid()) {
            gpu::BufferDesc colour;
            colour.bytes = pixels * 16;
            colour.elementBytes = 16;
            colour.label = "engine.opaque.colour";
            auto madeColour = gpu::Buffer::create(*device_, colour);
            if (!madeColour) return std::move(madeColour).error();
            gpu::BufferDesc depth;
            depth.bytes = pixels * 4;
            depth.elementBytes = 4;
            depth.label = "engine.opaque.depth";
            auto madeDepth = gpu::Buffer::create(*device_, depth);
            if (!madeDepth) return std::move(madeDepth).error();
            opaqueLayer_ = {settings.width, settings.height, std::move(*madeColour), std::move(*madeDepth)};
        }
        // The ids of the nearer layer, where the frame keeps a matte: points
        // carry none, so where a point is nearest the pixel goes unnamed.
        const bool carryIds = aovRequest.cryptomatte && meshLayer_.cryptoIds.valid();
        if (carryIds && (!opaqueLayer_.cryptoIds.valid() || opaqueLayer_.cryptoIds.bytes() < pixels * 4)) {
            gpu::BufferDesc desc;
            desc.bytes = pixels * 4;
            desc.elementBytes = 4;
            desc.label = "engine.opaque.cryptoIds";
            auto made = gpu::Buffer::create(*device_, desc);
            if (!made) return std::move(made).error();
            opaqueLayer_.cryptoIds = std::move(*made);
        }
        if (!nearest_.has_value()) {
            auto made = gpu::ComputeKernel::create(*library_, "athenea/technique/layers_nearest", "layersNearest");
            if (!made) return std::move(made).error();
            nearest_.emplace(std::move(*made));
        }
        gpu::CommandBatch batch(*device_);
        nearest_->dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colourA"].setBinding(meshLayer_.colour.rhi());
            cursor["depthA"].setBinding(meshLayer_.depth.rhi());
            cursor["colourB"].setBinding(pointLayer_.colour.rhi());
            cursor["depthB"].setBinding(pointLayer_.depth.rhi());
            cursor["colour"].setBinding(opaqueLayer_.colour.rhi());
            cursor["depth"].setBinding(opaqueLayer_.depth.rhi());
            cursor["idsA"].setBinding(carryIds ? meshLayer_.cryptoIds.rhi() : meshLayer_.depth.rhi());
            cursor["idsOut"].setBinding(carryIds ? opaqueLayer_.cryptoIds.rhi() : opaqueLayer_.depth.rhi());
            cursor["params"]["hasIds"].setData(uint32_t{carryIds ? 1u : 0u});
            cursor["params"]["pixels"].setData(static_cast<uint32_t>(pixels));
        });
        ATHENEA_TRY(batch.submit(true));
        under = &opaqueLayer_;
    } else if (meshLayer) {
        under = &meshLayer_;
    } else if (pointLayer) {
        under = &pointLayer_;
    }
    // What a splat relights from, where its prim asked to be relit
    // (AtheneaSplatLightingAPI). Handed over as buffers and counts, since render
    // sits below light in the module order and cannot name its types.
    render::SplatLights splatLights;
    if (lightTable_.has_value() && lightTable_->count() > 0) {
        splatLights.records = &lightTable_->records();
        splatLights.count = lightTable_->count();
        bindEnvironment(splatLights);
    }
    // ATHENEA_STAGES=1: where a rasterised frame of splats goes, a line a frame.
    // Each stage waits for the device, so the frame is slower for it.
    static const bool stages = !platform::env("ATHENEA_STAGES").empty();
    const auto visibilityStart = std::chrono::steady_clock::now();
    ATHENEA_TRY(measureVisibility(splatLights));
    const double visibilityMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - visibilityStart).count();
    // And what it casts its shadow against: the proxies of every cloud in the
    // frame, built by a tracer of their own on the hardware route. Only where
    // a cloud is relit, the device traces inline, and the setting asks.
    const bool shadowedSplats = splatShadows_.load() && splatLights.any() && relitSplats;
    if (shadowedSplats && device_->caps().rayQuery && device_->caps().accelerationStructure &&
        splatShadowsFit(gaussiansOf(splats))) {
        if (!shadowTracer_.has_value()) {
            render::RayTracerSettings rtSettings;
            rtSettings.route = render::RayTracingRoute::Hardware;
            auto made = render::GaussianRayTracer::create(*library_, rtSettings);
            if (!made) return std::move(made).error();
            shadowTracer_.emplace(std::move(*made));
        }
        if (!shadowTracerReady_) {
            auto prepared = shadowTracer_->prepare(projection, splats, settings.maxShDegree);
            if (!prepared) return std::move(prepared).error();
        }
        const render::ShadowScene scene = shadowTracer_->shadowScene();
        splatLights.shadowTlas = scene.tlas;
        splatLights.shadowFrames = scene.frames;
        splatLights.shadowColours = scene.colours;
        splatLights.shadowInstanceData = scene.instanceData;
        splatLights.shadowInstanceIndices = scene.instanceIndices;
    }
    // The matte is the blend's to build, so the setting reaches it here: one
    // copy, since the frame's settings are the caller's.
    render::RenderSettings splatSettings = settings;
    splatSettings.cryptomatte = aovRequest.cryptomatte;
    splatSettings.timeStages = splatSettings.timeStages || stages || timeSplatStages_.load();
    splatSettings.countSplats = countSplats_.load();
    splatSettings.countersTag = frameSerial_;
    if (splatSettings.countSplats) {
        // The prims of this frame's instances, kept until its counts arrive.
        countedFrames_.push_back({frameSerial_, instancePrims});
        if (countedFrames_.size() > 8) {
            countedFrames_.erase(countedFrames_.begin());
        }
    }
    // The counts the frame already had, kept for whoever draws a panel: the
    // rasteriser hands them back and nothing here measures anything for it.
    const auto drawn = under != nullptr
                           ? rasterizer_->render(projection, splats, splatSettings, targets, {}, under, &splatLights)
                           : rasterizer_->render(projection, splats, splatSettings, targets, points, nullptr,
                                                 &splatLights);
    if (!drawn) return std::move(drawn).error();
    if (stages) {
        log::info("stages: commit {:.1f} ms, visibility {:.1f}, project {:.1f}, counts {:.1f}, depth sort {:.1f}, "
                  "emit {:.1f}, tile sort {:.1f}, blend {:.1f}; {} splats, {} visible, {} pairs",
                  lastCommitMs_, visibilityMs, drawn->projectMs, drawn->countsMs, drawn->depthSortMs,
                  drawn->emitMs, drawn->tileSortMs, drawn->blendMs, drawn->splats, drawn->visible, drawn->pairs);
    }
    if (splatSettings.countSplats) {
        GaussianStats& g = gaussianStats_;
        g.stagesTimed = splatSettings.timeStages;
        g.projectMs = drawn->projectMs;
        g.countsMs = drawn->countsMs;
        g.depthSortMs = drawn->depthSortMs;
        g.emitMs = drawn->emitMs;
        g.tileSortMs = drawn->tileSortMs;
        g.blendMs = drawn->blendMs;
        g.totalMs = drawn->totalMs;
        takeSplatCounters();
    }
    counters_ = {drawn->splats,
                 drawn->visible,
                 drawn->pairs,
                 scene_.has_value() ? scene_->instanceCount() : 0u,
                 lightTable_.has_value() ? lightTable_->count() : 0u,
                 aovRequest.cryptomatte};
    ATHENEA_TRY(gatherLightGroups(pathTracing, settings.width, settings.height));
    ATHENEA_TRY(paintDomes(projection, settings.width, settings.height, targets));
    ATHENEA_TRY(applyExposure(projection.exposure, settings.width, settings.height, targets));
    return ok();
}

Result<void> Engine::gatherLightGroups(bool traced, uint32_t width, uint32_t height) {
    const uint64_t pixels = uint64_t{width} * height;
    if (!traced || lightGroupCount_ == 0 || !lightGroupColour_.valid() || lightGroupPixels_ != pixels ||
        !pathTracer_.has_value() || pathTracer_->lightGroups() < lightGroupCount_) {
        return ok();
    }
    if (!groupsScaled_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "athenea/technique/exposure", "copyScaled");
        if (!made) return std::move(made).error();
        groupsScaled_.emplace(std::move(*made));
    }
    gpu::CommandBatch batch(*device_);
    const uint32_t entries = static_cast<uint32_t>(pixels * lightGroupCount_);
    groupsScaled_->dispatch(batch, {entries, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["source"].setBinding(pathTracer_->sum().rhi());
        cursor["sourceBase"].setData(static_cast<uint32_t>(pathTracer_->lightGroupMeanOffset(0)));
        cursor["colour"].setBinding(lightGroupColour_.rhi());
        cursor["params"]["scale"].setData(1.0F);
        cursor["params"]["pixels"].setData(entries);
    });
    return batch.submit(true);
}

Result<void> Engine::applyExposure(double stops, uint32_t width, uint32_t height, render::RenderTargets& targets) {
    if (stops == 0.0 || !targets.colour.valid()) {
        return ok();
    }
    if (!exposure_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "athenea/technique/exposure", "applyExposure");
        if (!made) return std::move(made).error();
        exposure_.emplace(std::move(*made));
    }
    const uint32_t pixels = width * height;
    gpu::CommandBatch batch(*device_);
    exposure_->dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(targets.colour.rhi());
        cursor["params"]["scale"].setData(static_cast<float>(std::exp2(stops)));
        cursor["params"]["pixels"].setData(pixels);
    });
    // The light groups are what the colour is made of, and take its exposure.
    if (lightGroupCount_ > 0 && lightGroupColour_.valid() && lightGroupPixels_ == pixels) {
        const uint32_t entries = pixels * lightGroupCount_;
        exposure_->dispatch(batch, {entries, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(lightGroupColour_.rhi());
            cursor["params"]["scale"].setData(static_cast<float>(std::exp2(stops)));
            cursor["params"]["pixels"].setData(entries);
        });
    }
    return batch.submit(true);
}

namespace {

/// Where an array USD handed over lives, and how much of it there is. A
/// `FloatStream` already points into the `VtArray`'s own storage, so this is
/// that storage's address -- not a hash, and nothing is read.
[[nodiscard]] std::pair<const void*, size_t> arrayIdentity(const pxr::VtValue& value) {
    const scene::FloatStream stream = streamOf(value);
    if (!stream.empty()) {
        return {static_cast<const void*>(stream.bytes.data()), stream.bytes.size()};
    }
    if (value.IsHolding<pxr::VtIntArray>()) {
        const pxr::VtIntArray& held = value.UncheckedGet<pxr::VtIntArray>();
        return {static_cast<const void*>(held.cdata()), held.size() * sizeof(int)};
    }
    return {nullptr, 0};
}

}   // namespace

CloudIdentity Engine::identityOf(const ParticleFieldArrays& arrays) {
    // Blender's planes come and go together: the first stands for them all.
    static const pxr::VtValue kNone;
    const pxr::VtValue* const held[17] = {
        &arrays.positions,   &arrays.orientations, &arrays.scales,       &arrays.opacities,
        &arrays.shCoefficients, &arrays.metallic,  &arrays.roughness,    &arrays.transmission,
        &arrays.jointIndices,   &arrays.jointWeights, &arrays.visibilityParts, &arrays.visibilityTexels,
        &arrays.visibilityPartOf, &arrays.visibilityAmbient, &arrays.radianceBase,
        arrays.shPlanes.empty() ? &kNone : &arrays.shPlanes.front(), &arrays.jointWeightGradients};
    CloudIdentity identity;
    for (size_t k = 0; k < 17; ++k) {
        const auto [data, bytes] = arrayIdentity(*held[k]);
        identity.data[k] = data;
        identity.bytes[k] = bytes;
    }
    identity.shDegree = arrays.shPlanes.empty() ? arrays.shDegree : 100 + static_cast<int>(arrays.shPlanes.size());
    return identity;
}

Result<Engine::BakedVisibility> Engine::bakeVisibility(const pxr::SdfPath& id,
                                                       const technique::VisibilityParts& parts,
                                                       const technique::VisibilityBakeOptions& options) {
    const std::lock_guard<std::mutex> held(guard_);
    auto found = splats_.find(id);
    if (found == splats_.end() || found->second.gpu == nullptr) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no committed cloud", id.GetString());
    }
    SplatEntry& entry = found->second;
    if (!entry.influences.valid() || entry.joints == 0) {
        return Error::make(ErrorCode::InvalidArgument,
                           "'{}': a visibility bake needs a cloud a skeleton carries", id.GetString());
    }
    if (parts.jointToPart.size() != entry.joints) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': {} joints partitioned, the cloud has {}",
                           id.GetString(), parts.jointToPart.size(), entry.joints);
    }
    if (!splatVisibility_.has_value()) {
        auto made = technique::SplatVisibility::create(*library_);
        if (!made) return std::move(made).error();
        splatVisibility_.emplace(std::move(*made));
    }
    ATHENEA_TRY(splatVisibility_->bake(*entry.gpu, entry.influences, entry.perSplat, parts, options));
    BakedVisibility out;
    out.partCount = entry.gpu->visibilityPartCount;
    out.parts.resize(size_t{out.partCount} * 12);
    ATHENEA_TRY(entry.gpu->visibilityParts.read(*device_, 0, out.parts.size() * sizeof(float), out.parts.data()));
    out.texels.resize(entry.gpu->visibilityTexels.count());
    ATHENEA_TRY(entry.gpu->visibilityTexels.read(*device_, 0, out.texels.size() * sizeof(int32_t), out.texels.data()));
    // A part a record of the file, as the file keeps it: the splats
    // validation dropped belong to no part (-1).
    auto byRecord = loader_->toRecords(*entry.gpu, entry.gpu->visibilityPartOf, 1, 0xFFFFFFFFu,
                                       "splat.visibility.partOf.records");
    if (!byRecord) return std::move(byRecord).error();
    out.partOf.resize(entry.gpu->declared);
    ATHENEA_TRY(byRecord->read(*device_, 0, out.partOf.size() * sizeof(int32_t), out.partOf.data()));
    out.ambient.resize(entry.gpu->visibilityAmbient.count());
    ATHENEA_TRY(entry.gpu->visibilityAmbient.read(*device_, 0, out.ambient.size() * sizeof(int32_t), out.ambient.data()));
    return out;
}

Result<void> Engine::measureVisibility(render::SplatLights& into) {
    // What fills the factors: a baked field where a cloud carries one, the
    // shadow map for every other relit cloud, and 1 -- nothing in the way --
    // wherever neither says anything.
    const bool fromMap = cloudShadowMap_.has_value() && cloudShadowMap_->valid();
    if ((frameMeasured_.empty() && !fromMap) || !into.any()) {
        return ok();
    }
    if (!splatVisibility_.has_value()) {
        auto made = technique::SplatVisibility::create(*library_);
        if (!made) return std::move(made).error();
        splatVisibility_.emplace(std::move(*made));
    }
    const uint32_t lightsMeasured = std::min(into.count, uint32_t{8});
    const uint64_t wanted = uint64_t{frameSlots_} * lightsMeasured;
    if (!visibilityFactors_.valid() || visibilityFactors_.count() < wanted) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(wanted, 1) * 4;
        desc.elementBytes = 4;
        desc.label = "splat.visibilityFactors";
        auto made = gpu::Buffer::create(*device_, desc);
        if (!made) return std::move(made).error();
        visibilityFactors_ = std::move(*made);
    }
    gpu::CommandBatch batch(*device_);
    // Cleared first, every frame: only the clouds that have something to say
    // write here, and what the last frame left in the rest is not an answer.
    if (fromMap) {
        ATHENEA_TRY(cloudShadowMap_->clearFactors(batch, visibilityFactors_, static_cast<uint32_t>(wanted)));
        // A relit cloud with no baked field reads the map at its own depth:
        // one cloud shadowing another, and -- with the Fourier terms -- a
        // cloud shadowing itself.
        for (const SlottedCloud& cloud : frameClouds_) {
            if (!cloud.relight || cloud.hasField || cloud.drawn == nullptr) {
                continue;
            }
            technique::ShadowMapCaster caster;
            caster.cloud = cloud.drawn;
            caster.positions = &cloud.drawn->positions;
            caster.objectToWorld = cloud.rows;
            ATHENEA_TRY(cloudShadowMap_->factors(batch, caster, cloud.slot, visibilityFactors_));
        }
    }
    for (const MeasuredCloud& m : frameMeasured_) {
        technique::VisibilityFactorsJob job;
        job.cloud = m.fields;
        job.positions = &m.drawn->positions;
        job.skinningXforms = m.xforms;
        job.lights = into.records;
        job.lightCount = lightsMeasured;
        job.base = m.slot;
        job.categories = m.categories;
        job.objectToWorld = m.rows;
        job.geomBind = m.geomBind;
        ATHENEA_TRY(splatVisibility_->factors(batch, job, visibilityFactors_));
    }
    ATHENEA_TRY(batch.submit(true));
    into.visibilityFactors = &visibilityFactors_;
    into.visibilityLights = lightsMeasured;
    if (std::getenv("ATHENEA_VISIBILITY_DEBUG") != nullptr) {
        auto share = splatVisibility_->shadowedShare(visibilityFactors_, frameSlots_ * lightsMeasured);
        if (share) {
            log::info("visibility: {} clouds, {} slots x {} lights, {:.1f}% of factors under a half",
                      frameMeasured_.size(), frameSlots_, lightsMeasured, 100.0 * *share);
        }
    }
    return ok();
}

Result<void> Engine::carryCloud(const pxr::SdfPath& id, SplatEntry& entry, bool reuploaded) {
    const ParticleFieldArrays& arrays = *entry.pending;
    const auto* indices = arrays.jointIndices.IsHolding<pxr::VtIntArray>()
                              ? &arrays.jointIndices.UncheckedGet<pxr::VtIntArray>()
                              : nullptr;
    const auto* weights = arrays.jointWeights.IsHolding<pxr::VtFloatArray>()
                              ? &arrays.jointWeights.UncheckedGet<pxr::VtFloatArray>()
                              : nullptr;
    // The joints, whichever precision the file kept them in, transposed as
    // geom::Skinner sends a mesh's: USD puts vectors on the left and the
    // kernel multiplies rows by a column.
    std::vector<float> joints;
    std::vector<float> jointsEnd;
    const auto readInto = [](std::vector<float>& into, const pxr::VtValue& value) {
        if (value.IsHolding<pxr::VtMatrix4dArray>()) {
            const auto& held = value.UncheckedGet<pxr::VtMatrix4dArray>();
            into.resize(held.size() * 16);
            for (size_t j = 0; j < held.size(); ++j) {
                for (int row = 0; row < 4; ++row) {
                    for (int column = 0; column < 4; ++column) {
                        into[j * 16 + static_cast<size_t>(row) * 4 + static_cast<size_t>(column)] =
                            static_cast<float>(held[j][column][row]);
                    }
                }
            }
        } else if (value.IsHolding<pxr::VtMatrix4fArray>()) {
            const auto& held = value.UncheckedGet<pxr::VtMatrix4fArray>();
            into.resize(held.size() * 16);
            for (size_t j = 0; j < held.size(); ++j) {
                for (int row = 0; row < 4; ++row) {
                    for (int column = 0; column < 4; ++column) {
                        into[j * 16 + static_cast<size_t>(row) * 4 + static_cast<size_t>(column)] =
                            static_cast<float>(held[j][column][row]);
                    }
                }
            }
        }
    };
    readInto(joints, arrays.skinningXforms);
    // WHERE THE SHUTTER LEAVES THE SKELETON. Two poses of 609 joints is 78 kB
    // a frame; what it buys is a wing that blurs rather than strobes.
    readInto(jointsEnd, arrays.skinningXformsEnd);
    // THE FILE'S GAUSSIANS, NOT THE ONES KEPT. Influences are a record each,
    // and validation drops what cannot be drawn: measured against the kept
    // count, a glass sparrow that lost its faintest feathers lost its whole
    // rig with them, and drew in its bind pose. They are packed to the kept
    // splats on the device below (CloudLoader::keptOnly).
    const uint32_t count = entry.gpu->declared;
    // How many joints carry a gaussian: SkelBindingAPI's elementSize, which
    // is what the arrays' length over the count says; a constant binding
    // (the same joints for every gaussian) has one set for all of them.
    const bool constantInfluences = indices != nullptr && weights != nullptr && count > 0 &&
                                    indices->size() < size_t{count} && !indices->empty();
    const size_t perSplat = indices == nullptr || count == 0 ? 0
                            : constantInfluences              ? indices->size()
                                                              : indices->size() / count;
    if (indices == nullptr || weights == nullptr || joints.empty() || perSplat == 0 ||
        weights->size() != indices->size() ||
        (!constantInfluences && indices->size() != size_t{count} * perSplat)) {
        entry.posed.reset();
        entry.influences = gpu::Buffer{};
        entry.weightGradients = gpu::Buffer{};
        entry.xforms = gpu::Buffer{};
        entry.joints = 0;
        return ok();
    }
    if (!splatSkinner_.has_value()) {
        auto made = scene::SplatSkinner::create(*library_);
        if (!made) return std::move(made).error();
        splatSkinner_.emplace(std::move(*made));
    }
    // The joints and their weights do not change over time, so they are
    // uploaded with the cloud; `(joint, weight)` pairs is the layout every
    // skinner here reads, and pairing them is a rearrangement of values the
    // file holds.
    //
    // ONLY WHEN THE CLOUD WAS UPLOADED. Interleaving them is a loop over four
    // values a gaussian on the CPU and a buffer of eight floats each -- 137 MB
    // for the sparrow -- and none of it changes from one instant to the next.
    // Every frame of the timeline used to pay for it.
    if (reuploaded || !entry.influences.valid() || entry.perSplat != perSplat) {
        std::vector<float> pairs(size_t{count} * perSplat * 2);
        const size_t held = indices->size();
        for (size_t k = 0; k < size_t{count} * perSplat; ++k) {
            pairs[k * 2] = static_cast<float>(std::max((*indices)[k % held], 0));
            pairs[k * 2 + 1] = (*weights)[k % held];
        }
        auto influences = gpu::Buffer::fromSpan<float>(*device_, pairs, "splat.influences");
        if (!influences) return std::move(influences).error();
        auto kept = loader_->keptOnly(*entry.gpu, *influences, static_cast<uint32_t>(perSplat * 2),
                                      "splat.influences.kept");
        if (!kept) return std::move(kept).error();
        entry.influences = std::move(*kept);
        entry.perSplat = static_cast<uint32_t>(perSplat);

        // AND HOW THE WEIGHTS CHANGE ACROSS EACH GAUSSIAN, where the
        // conversion kept it: two halves for every joint but the last, which
        // is a word each -- uploaded as the file holds them and packed to the
        // kept splats as the influences are. A file without them, or with a
        // count that does not match, is carried as before.
        entry.weightGradients = gpu::Buffer{};
        const scene::FloatStream slopes = streamOf(arrays.jointWeightGradients);
        const size_t words = perSplat > 1 ? perSplat - 1 : 0;
        if (words > 0 && !constantInfluences && slopes.half &&
            slopes.bytes.size() == size_t{count} * words * 4) {
            gpu::BufferDesc desc;
            desc.bytes = slopes.bytes.size();
            desc.elementBytes = 4;
            desc.label = "splat.weightGradients";
            auto raw = gpu::Buffer::create(*device_, desc, slopes.bytes.data());
            if (!raw) return std::move(raw).error();
            auto keptSlopes = loader_->keptOnly(*entry.gpu, *raw, static_cast<uint32_t>(words),
                                                "splat.weightGradients.kept");
            if (!keptSlopes) return std::move(keptSlopes).error();
            entry.weightGradients = std::move(*keptSlopes);
        }
    }
    auto xforms = gpu::Buffer::fromSpan<float>(*device_, joints, "splat.skinningXforms");
    if (!xforms) return std::move(xforms).error();
    entry.xforms = std::move(*xforms);
    entry.joints = static_cast<uint32_t>(joints.size() / 16);
    // The prim worked the scale out where it knew both the shutter and the
    // instants Hydra bracketed it with; the engine's own shutter is not set
    // until the pass runs, which is after this.
    const bool moves = jointsEnd.size() == joints.size() && !jointsEnd.empty() &&
                       arrays.motionScale > 0.0F;
    entry.motionScale = 0.0F;
    if (moves) {
        auto end = gpu::Buffer::fromSpan<float>(*device_, jointsEnd, "splat.skinningXformsEnd");
        if (!end) return std::move(end).error();
        entry.xformsEnd = std::move(*end);
        entry.motionScale = arrays.motionScale;
    } else {
        entry.xformsEnd = gpu::Buffer{};
        entry.motion = gpu::Buffer{};
    }
    if (arrays.geomBindTransform.IsHolding<pxr::GfMatrix4d>()) {
        const pxr::GfMatrix4d& bind = arrays.geomBindTransform.UncheckedGet<pxr::GfMatrix4d>();
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                entry.geomBind[static_cast<size_t>(row) * 4 + static_cast<size_t>(column)] =
                    static_cast<float>(bind[row][column]);
            }
        }
    }

    // What is posed is what was kept: the influences were packed to it.
    const uint32_t kept = entry.gpu->count;
    // The posed cloud shares everything the skinner does not write -- the
    // harmonics, the PBR channels -- and owns the two buffers it does.
    if (entry.posed == nullptr || entry.posed->count != kept ||
        entry.posed->hasNormals() != entry.gpu->hasNormals()) {
        scene::GpuSplats posed = *entry.gpu;
        gpu::BufferDesc desc;
        desc.bytes = uint64_t{kept} * 16;
        desc.elementBytes = 16;
        desc.label = "splat.posed.positions";
        auto positions = gpu::Buffer::create(*device_, desc);
        if (!positions) return std::move(positions).error();
        desc.bytes = uint64_t{kept} * 4 * 4;
        desc.elementBytes = 4;
        desc.label = "splat.posed.shape";
        auto shape = gpu::Buffer::create(*device_, desc);
        if (!shape) return std::move(shape).error();
        posed.positions = std::move(*positions);
        posed.shape = std::move(*shape);
        // The shading normal turns with the frame, so the posed cloud owns
        // its own: the rest one is the bind pose's light.
        if (entry.gpu->hasNormals()) {
            desc.bytes = uint64_t{kept} * 4;
            desc.elementBytes = 4;
            desc.label = "splat.posed.normals";
            auto normals = gpu::Buffer::create(*device_, desc);
            if (!normals) return std::move(normals).error();
            posed.normals = std::move(*normals);
        }
        posed.source = id.GetString() + " (posed)";
        entry.posed = std::make_unique<scene::GpuSplats>(std::move(posed));
    }
    // The space its colours are in is the cloud's, whichever pose it is in:
    // a posed cloud kept from before a reupload takes the new one's.
    entry.posed->linear = entry.gpu->linear;
    if (moves && (!entry.motion.valid() || entry.motion.count() < uint64_t{kept} * 2)) {
        gpu::BufferDesc desc;
        desc.bytes = uint64_t{kept} * 2 * 4;
        desc.elementBytes = 4;
        desc.label = "splat.motion";
        auto motion = gpu::Buffer::create(*device_, desc);
        if (!motion) return std::move(motion).error();
        entry.motion = std::move(*motion);
    }
    gpu::CommandBatch batch(*device_);
    scene::SplatSkinInput input;
    input.rest = entry.gpu.get();
    input.influences = &entry.influences;
    input.perSplat = entry.perSplat;
    input.weightGradients = entry.weightGradients.valid() ? &entry.weightGradients : nullptr;
    input.skinningXforms = &entry.xforms;
    input.skinningXformsEnd = moves ? &entry.xformsEnd : nullptr;
    input.geomBindTransform = entry.geomBind;
    ATHENEA_TRY(splatSkinner_->skin(batch, input, entry.posed->positions, entry.posed->shape,
                                moves ? &entry.motion : nullptr,
                                entry.posed->hasNormals() ? &entry.posed->normals : nullptr));
    ATHENEA_TRY(batch.submit(true));
    // The box the cloud now fills. It is not the bind pose's: a skeleton
    // moves a cloud out from under its own extent, and everything that culls,
    // frames or sorts by it would be looking in the wrong place.
    auto box = loader_->boundsOf(entry.posed->positions, kept);
    if (!box) return std::move(box).error();
    entry.posed->bounds = *box;
    // And the box it was bound in, which no pose changes: what the cloud's
    // shadow map is sized from, so the map does not breathe with the wings.
    entry.posed->restBounds = entry.gpu->bounds;
    // A new pose in the same buffers: whatever built proxies over the last
    // one -- the ray tracer's, for a traced frame or a mesh's shadow rays --
    // finds out here rather than tracing the bind pose for the rest of the
    // film, which is what it did.
    entry.posed->revision += 1;
    return ok();
}

Result<void> Engine::paintDomes(const render::Projection& projection, uint32_t width, uint32_t height,
                                render::RenderTargets& targets) {
    if (!lightTable_.has_value() || !lightTable_->anyDome() || !targets.colour.valid()) {
        return ok();
    }
    if (!domeBackground_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "athenea/technique/dome_background", "domeBackground");
        if (!made) return std::move(made).error();
        domeBackground_.emplace(std::move(*made));
    }
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    gpu::CommandBatch batch(*device_);
    // The light groups' planes first: a dome seen by the camera is its group's,
    // and each plane takes the sky under the same coverage the beauty will, so
    // that the groups still sum to it. They read that coverage out of the
    // beauty's own alpha, which the pass below writes to one, so this one goes
    // first.
    if (lightGroupCount_ > 0 && lightGroupColour_.valid() && lightGroupPixels_ == uint64_t{width} * height) {
        if (!domeGroups_.has_value()) {
            auto made = gpu::ComputeKernel::create(*library_, "athenea/technique/dome_background", "domeBackgroundGroups");
            if (!made) return std::move(made).error();
            domeGroups_.emplace(std::move(*made));
        }
        domeGroups_->dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
            lightTable_->bind(cursor);
            if (textures_) {
                textures_->bind(cursor["gTextures"]);
            }
            cursor["colour"].setBinding(targets.colour.rhi());
            cursor["groupPlanes"].setBinding(lightGroupColour_.rhi());
            cursor["groupBase"].setData(uint32_t{0});
            cursor["groupCount"].setData(lightGroupCount_);
            technique::setCamera(cursor["camera"], projection, width, height);
            static constexpr const char* kNames[12] = {"v00", "v01", "v02", "v03", "v10", "v11",
                                                       "v12", "v13", "v20", "v21", "v22", "v23"};
            for (size_t k = 0; k < 12; ++k) {
                cursor["background"][kNames[k]].setData(toWorld[k]);
            }
        });
    }
    domeBackground_->dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
        lightTable_->bind(cursor);
        // A dome reads its image through the same table the materials sample,
        // so the background pass binds it too: without it every dome is the
        // white a missing file falls back to.
        if (textures_) {
            textures_->bind(cursor["gTextures"]);
        }
        cursor["colour"].setBinding(targets.colour.rhi());
        technique::setCamera(cursor["camera"], projection, width, height);
        static constexpr const char* kNames[12] = {"v00", "v01", "v02", "v03", "v10", "v11",
                                                   "v12", "v13", "v20", "v21", "v22", "v23"};
        for (size_t k = 0; k < 12; ++k) {
            cursor["background"][kNames[k]].setData(toWorld[k]);
        }
    });
    ATHENEA_TRY(batch.submit(true));
    return ok();
}

}   // namespace athenea::usd
