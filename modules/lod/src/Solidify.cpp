// Copyright (c) 2026 jesus luque.
#include "athenea/lod/Solidify.h"

#include <algorithm>

#include "athenea/core/Log.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::lod {
namespace {

constexpr const char* kSolidify = "athenea/lod/ssf_solidify";
constexpr const char* kSplat = "athenea/lod/ssf_splat";

Result<gpu::Buffer> deviceWords(gpu::Device& device, uint64_t count, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

/// Every buffer ssf_solidify.slang names, each bound to `dummy` unless set:
/// a kernel reads only its own, but every name is bound so that none is left
/// to whatever a backend does with an unbound one.
struct Binds {
    const gpu::Buffer* srcPositions = nullptr;
    const gpu::Buffer* srcShape = nullptr;
    const gpu::Buffer* srcNormals = nullptr;
    const gpu::Buffer* srcPbr = nullptr;
    const gpu::Buffer* flags = nullptr;
    const gpu::Buffer* offsets = nullptr;
    const gpu::Buffer* sourceIn = nullptr;
    const gpu::Buffer* outPositions = nullptr;
    const gpu::Buffer* outShape = nullptr;
    const gpu::Buffer* outNormals = nullptr;
    const gpu::Buffer* outFlags = nullptr;
    const gpu::Buffer* outAnchor = nullptr;
    const gpu::Buffer* solidSource = nullptr;
    const gpu::Buffer* residualSource = nullptr;
    const gpu::Buffer* stats = nullptr;
    const gpu::Buffer* pickPositions = nullptr;
    const gpu::Buffer* pickShape = nullptr;
    const gpu::Buffer* pickNormals = nullptr;
    const gpu::Buffer* pickIndex = nullptr;
    const gpu::Buffer* pickSource = nullptr;
};

void bindAll(rhi::ShaderCursor c, const Binds& b, const gpu::Buffer& dummy) {
    const auto set = [&](const char* name, const gpu::Buffer* buffer) {
        c[name].setBinding((buffer != nullptr ? *buffer : dummy).rhi());
    };
    set("srcPositions", b.srcPositions);
    set("srcShape", b.srcShape);
    set("srcNormals", b.srcNormals);
    set("srcPbr", b.srcPbr);
    set("flags", b.flags);
    set("offsets", b.offsets);
    set("sourceIn", b.sourceIn);
    set("outPositions", b.outPositions);
    set("outShape", b.outShape);
    set("outNormals", b.outNormals);
    set("outFlags", b.outFlags);
    set("outAnchor", b.outAnchor);
    set("solidSource", b.solidSource);
    set("residualSource", b.residualSource);
    set("stats", b.stats);
    set("pickPositions", b.pickPositions);
    set("pickShape", b.pickShape);
    set("pickNormals", b.pickNormals);
    set("pickIndex", b.pickIndex);
    set("pickSource", b.pickSource);
}

void setParams(rhi::ShaderCursor p, uint32_t count, const SolidifySettings& s, bool normals, bool pbrs) {
    p["count"].setData(count);
    p["normals"].setData(uint32_t{normals ? 1u : 0u});
    p["pbrs"].setData(uint32_t{pbrs ? 1u : 0u});
    p["opacityMin"].setData(s.opacityMin);
    p["flatMax"].setData(s.flatMax);
    p["coherenceMin"].setData(s.coherenceMin);
    p["radiusScale"].setData(s.radiusScale);
    p["thicknessRatio"].setData(s.thicknessRatio);
    p["tau"].setData(s.tau);
    p["level"].setData(s.level);
    p["mergeFraction"].setData(s.mergeFraction);
}

/// A layer's three blob arrays, `count` blobs.
struct Blobs {
    gpu::Buffer positions, shape, normals;
};

Result<Blobs> blobsOf(gpu::Device& device, uint64_t count) {
    Blobs b;
    auto p = deviceWords(device, count, 16, "ssf.blobs");
    if (!p) return std::move(p).error();
    auto s = deviceWords(device, count * 4, 4, "ssf.blobs");
    if (!s) return std::move(s).error();
    auto n = deviceWords(device, count, 4, "ssf.normals");
    if (!n) return std::move(n).error();
    b.positions = std::move(*p);
    b.shape = std::move(*s);
    b.normals = std::move(*n);
    return b;
}

Result<uint32_t> readCount(gpu::Device& device, const gpu::Buffer& total) {
    uint32_t n = 0;
    ATHENEA_TRY(total.read(device, 0, sizeof n, &n));
    return n;
}

}   // namespace

Result<Solidifier> Solidifier::create(gpu::ShaderLibrary& library) {
    Solidifier s;
    s.library_ = &library;
    s.device_ = &library.device();
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    s.prefix_ = std::move(*prefix);
    auto field = SolidField::create(library);
    if (!field) return std::move(field).error();
    s.field_ = std::move(*field);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    ATHENEA_TRY(make(s.convert_, kSolidify, "ssfConvert"));
    ATHENEA_TRY(make(s.anchor_, kSolidify, "ssfAnchor"));
    ATHENEA_TRY(make(s.compact_, kSolidify, "ssfCompact"));
    ATHENEA_TRY(make(s.coherence_, kSolidify, "ssfCoherence"));
    ATHENEA_TRY(make(s.weights_, kSolidify, "ssfWeights"));
    ATHENEA_TRY(make(s.ridge_, kSolidify, "ssfRidge"));
    ATHENEA_TRY(make(s.normals_, kSolidify, "ssfNormals"));
    ATHENEA_TRY(make(s.merge_, kSolidify, "ssfMerge"));
    ATHENEA_TRY(make(s.iota_, kSolidify, "ssfIota"));
    ATHENEA_TRY(make(s.pick_, kSolidify, "ssfPickBlobs"));
    ATHENEA_TRY(make(s.toSplat_, kSplat, "ssfToSplat"));
    ATHENEA_TRY(make(s.gather_, kSplat, "ssfGather"));
    return s;
}

Result<SolidifyResult> Solidifier::solidify(const scene::GpuSplats& cloud, const SolidifySettings& settings) {
    gpu::Device& device = *device_;
    const uint32_t n = cloud.count;
    if (n == 0) {
        return Error(ErrorCode::InvalidArgument, "an empty cloud has nothing to solidify");
    }
    const bool normals = cloud.hasNormals();
    const bool pbrs = cloud.hasPbr();
    auto dummyMade = deviceWords(device, 4, 4, "ssf.dummy");
    if (!dummyMade) return std::move(dummyMade).error();
    const gpu::Buffer dummy = std::move(*dummyMade);
    const auto words = [&](uint64_t count, const char* label) { return deviceWords(device, count, 4, label); };
    // Zeroed where a kernel adds to it.
    const auto zeroed = [&](uint64_t count, const char* label) -> Result<gpu::Buffer> {
        const std::vector<uint32_t> zeros(std::max<uint64_t>(count, 1), 0u);
        return gpu::Buffer::fromSpan(device, std::span<const uint32_t>(zeros), label);
    };

    SolidifyResult result;
    result.gaussians = n;
    // 1. Every gaussian a candidate blob, and the anchor.
    auto cand = blobsOf(device, n);
    if (!cand) return std::move(cand).error();
    auto flagsA = words(n, "ssf.flags");
    auto offsetsA = words(n, "ssf.offsets");
    auto total = words(1, "ssf.total");
    auto stats = zeroed(2, "ssf.stats");
    auto anchor = words(4, "ssf.anchor");
    auto identity = words(n, "ssf.identity");
    auto source1 = words(n, "ssf.source");
    auto residual1 = words(n, "ssf.residual");
    for (auto* r : {&flagsA, &offsetsA, &total, &stats, &anchor, &identity, &source1, &residual1}) {
        if (!*r) return std::move(*r).error();
    }
    {
        gpu::CommandBatch batch(device);
        Binds b;
        b.srcPositions = &cloud.positions;
        b.srcShape = &cloud.shape;
        b.srcNormals = normals ? &cloud.normals : nullptr;
        b.srcPbr = pbrs ? &cloud.pbr : nullptr;
        b.outPositions = &cand->positions;
        b.outShape = &cand->shape;
        b.outNormals = &cand->normals;
        b.outFlags = &*flagsA;
        b.outAnchor = &*anchor;
        b.stats = &*stats;
        b.solidSource = &*identity;
        convert_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor c) {
            bindAll(c, b, dummy);
            setParams(c["params"], n, settings, normals, pbrs);
        });
        anchor_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
            bindAll(c, b, dummy);
            setParams(c["params"], n, settings, normals, pbrs);
        });
        iota_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor c) {
            bindAll(c, b, dummy);
            setParams(c["params"], n, settings, normals, pbrs);
        });
        ATHENEA_TRY(prefix_.apply(batch, *flagsA, *offsetsA, *total, n));
        Binds k;
        k.flags = &*flagsA;
        k.offsets = &*offsetsA;
        k.sourceIn = &*identity;
        k.solidSource = &*source1;
        k.residualSource = &*residual1;
        compact_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor c) {
            bindAll(c, k, dummy);
            setParams(c["params"], n, settings, normals, pbrs);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    auto m1 = readCount(device, *total);
    if (!m1) return std::move(m1).error();
    result.candidates = *m1;
    const auto residualsOnly = [&]() -> Result<SolidifyResult> {
        result.residuals = n;
        result.residualSource = std::move(*identity);
        return std::move(result);
    };
    if (*m1 == 0) {
        return residualsOnly();
    }

    // Picks blobs `index` names out of `from` (sources `fromSource`) into a
    // dense layer of `count`, its sources into `outSource`.
    const auto pick = [&](gpu::CommandBatch& batch, const Blobs& from, const gpu::Buffer& index,
                          const gpu::Buffer& fromSource, uint32_t count, Blobs& into, const gpu::Buffer& outSource) {
        Binds p;
        p.pickPositions = &from.positions;
        p.pickShape = &from.shape;
        p.pickNormals = &from.normals;
        p.pickIndex = &index;
        p.pickSource = &fromSource;
        p.outPositions = &into.positions;
        p.outShape = &into.shape;
        p.outNormals = &into.normals;
        p.residualSource = &outSource;
        pick_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) {
            bindAll(c, p, dummy);
            setParams(c["params"], count, settings, normals, pbrs);
        });
    };
    const auto layerOf = [&](const Blobs& blobs, uint32_t count) {
        SolidLayer l;
        l.blobs = count;
        l.anchors = 1;
        l.positions = blobs.positions;
        l.shape = blobs.shape;
        l.anchorWords = *anchor;
        return l;
    };

    // 2. The candidates whose neighbours agree. Picked by gaussian index:
    // `source1` names gaussians, which are the candidates' own indices.
    auto layer1 = blobsOf(device, *m1);
    if (!layer1) return std::move(layer1).error();
    auto picked1 = words(*m1, "ssf.source");
    if (!picked1) return std::move(picked1).error();
    auto flagsB = words(*m1, "ssf.flags");
    auto offsetsB = words(*m1, "ssf.offsets");
    auto source2 = words(*m1, "ssf.source");
    auto residual2 = words(*m1, "ssf.residual");
    for (auto* r : {&flagsB, &offsetsB, &source2, &residual2}) {
        if (!*r) return std::move(*r).error();
    }
    {
        gpu::CommandBatch batch(device);
        pick(batch, *cand, *source1, *identity, *m1, *layer1, *picked1);
        ATHENEA_TRY(field_.build(batch, layerOf(*layer1, *m1), layer1->normals));
        Binds b;
        b.outFlags = &*flagsB;
        coherence_.dispatch(batch, {*m1, 1, 1}, [&](rhi::ShaderCursor c) {
            field_.bind(c);
            bindAll(c, b, dummy);
            field_.bind(c);
            setParams(c["params"], *m1, settings, normals, pbrs);
        });
        ATHENEA_TRY(prefix_.apply(batch, *flagsB, *offsetsB, *total, *m1));
        Binds k;
        k.flags = &*flagsB;
        k.offsets = &*offsetsB;
        k.sourceIn = &*source1;
        k.solidSource = &*source2;
        k.residualSource = &*residual2;
        compact_.dispatch(batch, {*m1, 1, 1}, [&](rhi::ShaderCursor c) {
            bindAll(c, k, dummy);
            setParams(c["params"], *m1, settings, normals, pbrs);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    auto m2 = readCount(device, *total);
    if (!m2) return std::move(m2).error();
    result.coherent = *m2;
    // The residuals: the non-candidates, then the incoherent.
    const uint32_t residuals = n - *m2;
    auto residualAll = words(residuals, "ssf.residualSource");
    if (!residualAll) return std::move(residualAll).error();
    {
        gpu::CommandBatch batch(device);
        if (n > *m1) {
            batch.encoder()->copyBuffer(residualAll->rhi(), 0, residual1->rhi(), 0, uint64_t{n - *m1} * 4);
        }
        if (*m1 > *m2) {
            batch.encoder()->copyBuffer(residualAll->rhi(), uint64_t{n - *m1} * 4, residual2->rhi(), 0,
                                        uint64_t{*m1 - *m2} * 4);
        }
        batch.markDirty();
        ATHENEA_TRY(batch.submit(true));
    }
    result.residuals = residuals;
    result.residualSource = std::move(*residualAll);
    if (*m2 == 0) {
        result.residualSource = std::move(*identity);
        result.residuals = n;
        return std::move(result);
    }

    // 3. The coherent blobs: weights, ridge, normals.
    auto layer2 = blobsOf(device, *m2);
    if (!layer2) return std::move(layer2).error();
    auto picked2 = words(*m2, "ssf.source");
    auto scratch = blobsOf(device, *m2);
    if (!picked2) return std::move(picked2).error();
    if (!scratch) return std::move(scratch).error();
    {
        gpu::CommandBatch batch(device);
        pick(batch, *cand, *source2, *identity, *m2, *layer2, *picked2);
        ATHENEA_TRY(field_.build(batch, layerOf(*layer2, *m2), layer2->normals));
        ATHENEA_TRY(batch.submit(true));
    }
    const auto pass = [&](gpu::ComputeKernel& kernel, bool positions, bool rebuild) -> Result<void> {
        gpu::CommandBatch batch(device);
        Binds b;
        b.outPositions = &scratch->positions;
        b.outNormals = &scratch->normals;
        kernel.dispatch(batch, {*m2, 1, 1}, [&](rhi::ShaderCursor c) {
            bindAll(c, b, dummy);
            field_.bind(c);
            setParams(c["params"], *m2, settings, normals, pbrs);
        });
        const gpu::Buffer& from = positions ? scratch->positions : scratch->normals;
        const gpu::Buffer& to = positions ? layer2->positions : layer2->normals;
        batch.encoder()->copyBuffer(to.rhi(), 0, from.rhi(), 0, uint64_t{*m2} * (positions ? 16 : 4));
        batch.markDirty();
        if (rebuild) {
            ATHENEA_TRY(field_.build(batch, layerOf(*layer2, *m2), layer2->normals));
        }
        return batch.submit(true);
    };
    for (uint32_t r = 0; r < settings.weightRounds; ++r) {
        ATHENEA_TRY(pass(weights_, true, false));
    }
    for (uint32_t r = 0; r < settings.ridgeRounds; ++r) {
        ATHENEA_TRY(pass(ridge_, true, true));
        // The weights again, where the centres moved.
        ATHENEA_TRY(pass(weights_, true, false));
    }
    ATHENEA_TRY(pass(normals_, false, true));

    // 4. Of blobs nearly on one another, one. The dropped stay absorbed: the
    // kept one stands for them.
    auto flagsC = words(*m2, "ssf.flags");
    auto offsetsC = words(*m2, "ssf.offsets");
    auto local = words(*m2, "ssf.local");
    auto localIdentity = words(*m2, "ssf.identity");
    auto droppedLocal = words(*m2, "ssf.dropped");
    for (auto* r : {&flagsC, &offsetsC, &local, &localIdentity, &droppedLocal}) {
        if (!*r) return std::move(*r).error();
    }
    {
        gpu::CommandBatch batch(device);
        Binds b;
        b.outFlags = &*flagsC;
        b.solidSource = &*localIdentity;
        merge_.dispatch(batch, {*m2, 1, 1}, [&](rhi::ShaderCursor c) {
            bindAll(c, b, dummy);
            field_.bind(c);
            setParams(c["params"], *m2, settings, normals, pbrs);
        });
        iota_.dispatch(batch, {*m2, 1, 1}, [&](rhi::ShaderCursor c) {
            bindAll(c, b, dummy);
            setParams(c["params"], *m2, settings, normals, pbrs);
        });
        ATHENEA_TRY(prefix_.apply(batch, *flagsC, *offsetsC, *total, *m2));
        Binds k;
        k.flags = &*flagsC;
        k.offsets = &*offsetsC;
        k.sourceIn = &*localIdentity;
        k.solidSource = &*local;
        k.residualSource = &*droppedLocal;
        compact_.dispatch(batch, {*m2, 1, 1}, [&](rhi::ShaderCursor c) {
            bindAll(c, k, dummy);
            setParams(c["params"], *m2, settings, normals, pbrs);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    auto m3 = readCount(device, *total);
    if (!m3) return std::move(m3).error();
    auto layer3 = blobsOf(device, *m3);
    if (!layer3) return std::move(layer3).error();
    auto source3 = words(*m3, "ssf.solidSource");
    if (!source3) return std::move(source3).error();
    {
        gpu::CommandBatch batch(device);
        pick(batch, *layer2, *local, *source2, *m3, *layer3, *source3);
        ATHENEA_TRY(batch.submit(true));
    }
    result.solid = layerOf(*layer3, *m3);
    result.solid.normals = layer3->normals;
    result.normals = std::move(layer3->normals);
    result.solidSource = std::move(*source3);
    log::info("solidify: {} gaussians, {} candidates, {} absorbed ({:.1f} %), {} blobs after merging, {} residuals", n,
              result.candidates, result.coherent, 100.0 * result.absorbed(), *m3, result.residuals);
    return std::move(result);
}

Result<scene::GpuSplats> Solidifier::toSplats(const SolidifyResult& result, const scene::GpuSplats& source,
                                              const SolidifySettings& settings) {
    gpu::Device& device = *device_;
    const uint32_t m = result.solid.blobs;
    const uint32_t total = m + result.residuals;
    scene::GpuSplats out;
    out.source = source.source + " (SSF mode A)";
    out.count = total;
    out.declared = total;
    out.restPerColour = source.restPerColour;
    out.shWords = source.shWords;
    out.linear = source.linear;
    out.bounds = source.bounds;
    auto p = deviceWords(device, total, 16, "ssf.splats");
    if (!p) return std::move(p).error();
    auto s = deviceWords(device, uint64_t{total} * 4, 4, "ssf.splats");
    if (!s) return std::move(s).error();
    auto sh = deviceWords(device, uint64_t{total} * source.shWords, 4, "ssf.splats");
    if (!sh) return std::move(sh).error();
    out.positions = std::move(*p);
    out.shape = std::move(*s);
    out.sh = std::move(*sh);
    const bool normals = source.hasNormals();
    if (normals) {
        auto nm = deviceWords(device, total, 4, "ssf.splats");
        if (!nm) return std::move(nm).error();
        out.normals = std::move(*nm);
    }
    auto dummyMade = deviceWords(device, 4, 4, "ssf.dummy");
    if (!dummyMade) return std::move(dummyMade).error();
    const gpu::Buffer dummy = std::move(*dummyMade);
    gpu::CommandBatch batch(device);
    const auto bindSplat = [&](rhi::ShaderCursor c, const gpu::Buffer& sources, uint32_t count, uint32_t first,
                               uint32_t geometry) {
        c["blobPositions"].setBinding((m > 0 ? result.solid.positions : dummy).rhi());
        c["blobShape"].setBinding((m > 0 ? result.solid.shape : dummy).rhi());
        c["blobNormals"].setBinding((m > 0 ? result.normals : dummy).rhi());
        c["anchors"].setBinding((m > 0 ? result.solid.anchorWords : dummy).rhi());
        c["gridKeys"].setBinding(dummy.rhi());
        c["gridOrder"].setBinding(dummy.rhi());
        c["gridInfo"].setBinding(dummy.rhi());
        c["srcPositions"].setBinding(source.positions.rhi());
        c["srcShape"].setBinding(source.shape.rhi());
        c["srcSh"].setBinding(source.sh.rhi());
        c["srcNormals"].setBinding((normals ? source.normals : dummy).rhi());
        c["sources"].setBinding(sources.rhi());
        c["dstPositions"].setBinding(out.positions.rhi());
        c["dstShape"].setBinding(out.shape.rhi());
        c["dstSh"].setBinding(out.sh.rhi());
        c["dstNormals"].setBinding((normals ? out.normals : dummy).rhi());
        c["params"]["count"].setData(count);
        c["params"]["first"].setData(first);
        c["params"]["shWords"].setData(source.shWords);
        c["params"]["normals"].setData(uint32_t{normals ? 1u : 0u});
        c["params"]["geometry"].setData(geometry);
        c["params"]["radiusScale"].setData(settings.radiusScale);
        c["params"]["thinRatio"].setData(settings.thinRatio);
    };
    if (m > 0) {
        toSplat_.dispatch(batch, {m, 1, 1},
                          [&](rhi::ShaderCursor c) { bindSplat(c, result.solidSource, m, 0, 0); });
        gather_.dispatch(batch, {m, 1, 1},
                         [&](rhi::ShaderCursor c) { bindSplat(c, result.solidSource, m, 0, 0); });
    }
    if (result.residuals > 0) {
        gather_.dispatch(batch, {result.residuals, 1, 1},
                         [&](rhi::ShaderCursor c) { bindSplat(c, result.residualSource, result.residuals, m, 1); });
    }
    ATHENEA_TRY(batch.submit(true));
    return out;
}

}   // namespace athenea::lod
