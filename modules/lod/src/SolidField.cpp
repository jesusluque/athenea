// Copyright (c) 2026 jesus luque.
#include "athenea/lod/SolidField.h"

#include <algorithm>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::lod {
namespace {

constexpr const char* kModule = "athenea/lod/ssf_grid";
constexpr uint32_t    kInfoWords = 8;   // ssf_field.slang's kGridWords

Result<gpu::Buffer> words(gpu::Device& device, uint64_t count, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * 4;
    desc.elementBytes = 4;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

}   // namespace

Result<SolidField> SolidField::create(gpu::ShaderLibrary& library) {
    SolidField f;
    f.device_ = &library.device();
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    f.sort_ = std::move(*sort);
    const auto make = [&](gpu::ComputeKernel& into, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, kModule, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    ATHENEA_TRY(make(f.clear_, "ssfGridClear"));
    ATHENEA_TRY(make(f.bounds_, "ssfGridBounds"));
    ATHENEA_TRY(make(f.setup_, "ssfGridSetup"));
    ATHENEA_TRY(make(f.keys_, "ssfGridKeys"));
    ATHENEA_TRY(make(f.eval_, "ssfFieldEval"));
    auto bounds = words(*f.device_, 8, "ssf.gridBounds");
    if (!bounds) return std::move(bounds).error();
    f.bounds_words_ = std::move(*bounds);
    auto info = words(*f.device_, kInfoWords, "ssf.gridInfo");
    if (!info) return std::move(info).error();
    f.info_ = std::move(*info);
    return f;
}

Result<void> SolidField::build(gpu::CommandBatch& batch, const SolidLayer& layer, const gpu::Buffer& normals) {
    if (!layer.present() || layer.anchors == 0) {
        return Error(ErrorCode::InvalidArgument, "a solid field needs a layer with blobs and anchors");
    }
    if (!normals.valid()) {
        return Error(ErrorCode::InvalidArgument, "a solid field needs its blobs' normals");
    }
    layer_ = layer;
    normals_ = normals;
    const uint32_t n = layer.blobs;
    const auto assign = [&](gpu::Buffer& into, const char* label) -> Result<void> {
        auto made = words(*device_, n, label);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(assign(sorting_.keysLo, "ssf.keys"));
    ATHENEA_TRY(assign(sorting_.values, "ssf.order"));
    ATHENEA_TRY(assign(sorting_.scratchKeysLo, "ssf.keys2"));
    ATHENEA_TRY(assign(sorting_.scratchValues, "ssf.order2"));
    const auto bindLayer = [&](rhi::ShaderCursor c) {
        c["blobPositions"].setBinding(layer_.positions.rhi());
        c["blobShape"].setBinding(layer_.shape.rhi());
        c["blobNormals"].setBinding(normals_.rhi());
        c["anchors"].setBinding(layer_.anchorWords.rhi());
        c["gridBounds"].setBinding(bounds_words_.rhi());
        c["gridOut"].setBinding(info_.rhi());
        c["keys"].setBinding(sorting_.keysLo.rhi());
        c["values"].setBinding(sorting_.values.rhi());
        c["params"]["count"].setData(n);
    };
    clear_.dispatch(batch, {8, 1, 1}, bindLayer);
    bounds_.dispatch(batch, {n, 1, 1}, bindLayer);
    setup_.dispatch(batch, {1, 1, 1}, bindLayer);
    keys_.dispatch(batch, {n, 1, 1}, bindLayer);
    ATHENEA_TRY(sort_.sort(batch, sorting_, n, 30));
    batch.markDirty();
    return ok();
}

void SolidField::bind(rhi::ShaderCursor c) const {
    c["blobPositions"].setBinding(layer_.positions.rhi());
    c["blobShape"].setBinding(layer_.shape.rhi());
    c["blobNormals"].setBinding(normals_.rhi());
    c["anchors"].setBinding(layer_.anchorWords.rhi());
    c["gridKeys"].setBinding(sorting_.keysLo.rhi());
    c["gridOrder"].setBinding(sorting_.values.rhi());
    c["gridInfo"].setBinding(info_.rhi());
}

void SolidField::evaluate(gpu::CommandBatch& batch, const gpu::Buffer& points, uint32_t count, float tau,
                          const gpu::Buffer& out) const {
    eval_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) {
        bind(c);
        c["points"].setBinding(points.rhi());
        c["field"].setBinding(out.rhi());
        c["params"]["count"].setData(count);
        c["params"]["tau"].setData(tau);
    });
}

}   // namespace athenea::lod
