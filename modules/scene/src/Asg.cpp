// Copyright (c) 2026 jesus luque.
#include "athenea/scene/Asg.h"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <vector>

#include "athenea/core/Log.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::scene {
namespace {

// The header's flags and an anchor's format word (proposal 002).
constexpr uint32_t kTransfer = 1, kShadowBits = 2, kOpacityLog = 4, kMeansCpca = 16, kShadowGlobal = 32;
constexpr uint32_t kFloatsPerRecord = 20;

template <typename T>
T at(const std::vector<uint8_t>& bytes, size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

Error bad(const std::filesystem::path& path, const char* what) {
    return Error::make(ErrorCode::InvalidArgument, "{}: not an ASG v1 file this reads ({})", path.string(), what);
}

}   // namespace

bool isAsg(const std::filesystem::path& path) {
    return path.extension() == ".asg";
}

Result<AsgCloud> decodeAsg(gpu::ShaderLibrary& library, const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return Error::make(ErrorCode::NotFound, "{}: cannot be read", path.string());
    }
    const auto size = static_cast<size_t>(in.tellg());
    // Whole words, so the device reads it as one: the leaves' last word may be
    // read past the file's end by a field that ends in it.
    std::vector<uint8_t> bytes((size + 7) / 4 * 4, 0);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (size < 160 || std::memcmp(bytes.data(), "ASG1", 4) != 0 || at<uint16_t>(bytes, 4) != 1) {
        return bad(path, "magic or version");
    }

    // THE HEADER: the counts, the widths, where each section is.
    const uint32_t count = at<uint32_t>(bytes, 8), anchors = at<uint32_t>(bytes, 12), flags = at<uint32_t>(bytes, 16);
    const uint32_t S = at<uint16_t>(bytes, 20), blocks = bytes[22];
    const std::array<uint32_t, 3> D{at<uint16_t>(bytes, 24), at<uint16_t>(bytes, 26), at<uint16_t>(bytes, 28)};
    const std::array<uint32_t, 3> K{bytes[48], bytes[49], bytes[50]};
    const uint32_t meanK = bytes[51], C = at<uint16_t>(bytes, 52);
    const std::array<uint32_t, 3> Cm{at<uint16_t>(bytes, 54), at<uint16_t>(bytes, 56), at<uint16_t>(bytes, 58)};
    const uint32_t anchorSize = at<uint32_t>(bytes, 68);
    const uint64_t offMat = at<uint64_t>(bytes, 72), offShadowDict = at<uint64_t>(bytes, 80);
    const uint64_t offMeans = at<uint64_t>(bytes, 88), offBases = at<uint64_t>(bytes, 96);
    const uint64_t offAnchors = at<uint64_t>(bytes, 104), offLeaves = at<uint64_t>(bytes, 112);
    const bool transfer = (flags & kTransfer) != 0, shadows = (flags & kShadowBits) != 0;
    if (count == 0 || anchors == 0 || blocks > 3 || (shadows && S != 64 && S != 256) ||
        offAnchors + uint64_t{anchors} * anchorSize > size || offLeaves > size || (offLeaves % 4) != 0 ||
        bytes.size() > 0xFFFFFFF0ULL) {
        return bad(path, "sizes");
    }

    // THE ANCHOR RECORD: 84 bytes always, then -- in the order the writer
    // lays them -- each block's transfer mean (f16) where the means are not
    // CPCA, each block's residual scales, each block's mean cluster and
    // weights where they are, each block's residual basis, the majority
    // shadow pattern and the anchor's own dictionary.
    std::array<uint32_t, 3> meanAt{}, scaleAt{}, meanClusterAt{}, residualAt{};
    uint32_t o = 84;
    if (transfer) {
        for (uint32_t b = 0; b < blocks; ++b) {
            if ((flags & kMeansCpca) == 0) { meanAt[b] = o; o += D[b] * 2; }
        }
        for (uint32_t b = 0; b < blocks; ++b) {
            if (K[b] != 0) { scaleAt[b] = o; o += K[b] * 2; }
        }
        for (uint32_t b = 0; b < blocks; ++b) {
            if ((flags & kMeansCpca) != 0) { meanClusterAt[b] = o; o += 1 + meanK; }
        }
        for (uint32_t b = 0; b < blocks; ++b) {
            if (K[b] != 0) { residualAt[b] = o; o += 1; }
        }
    }
    uint32_t majorityAt = 0, dictionaryAt = 0;
    if (shadows) {
        majorityAt = o;
        o += S / 8;
        if ((flags & kShadowGlobal) == 0) { dictionaryAt = o; o += (1U << bytes[40]) * S / 8; }
    }
    if ((o + 7) / 8 * 8 != anchorSize) {
        return bad(path, "anchor record size");
    }
    // The tables: each block's CPCA of the means (centroid, axes, scales a
    // cluster, f16), then each block's residual bases (C x K x D, f16).
    std::array<uint32_t, 3> meansTableAt{}, basesAt{};
    uint64_t t = offMeans;
    for (uint32_t b = 0; b < blocks && transfer && (flags & kMeansCpca) != 0; ++b) {
        meansTableAt[b] = static_cast<uint32_t>(t);
        t += uint64_t{Cm[b]} * (D[b] + meanK * D[b] + meanK) * 2;
    }
    t = offBases;
    for (uint32_t b = 0; b < blocks && transfer; ++b) {
        if (K[b] != 0) { basesAt[b] = static_cast<uint32_t>(t); t += uint64_t{C} * K[b] * D[b] * 2; }
    }

    gpu::Device& device = library.device();
    AsgCloud cloud;
    cloud.count = count;
    cloud.anchors = anchors;
    cloud.directCount = transfer && blocks > 0 ? D[0] : 0;
    cloud.indirectCount = transfer && blocks > 1 ? D[1] : 0;
    cloud.shadowWords = shadows ? S / 32 : 0;
    io::SplatEncoding& e = cloud.encoding;
    e.floatsPerRecord = kFloatsPerRecord;
    e.x = 0, e.y = 1, e.z = 2, e.opacity = 3, e.scale0 = 4, e.scale1 = 5, e.scale2 = 6;
    e.rotW = 7, e.rotX = 8, e.rotY = 9, e.rotZ = 10, e.dc0 = 11, e.dc1 = 12, e.dc2 = 13;
    e.metallic = 14, e.roughness = 15, e.transmission = 16, e.normal = 17;
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Linear;
    e.colour = io::SplatEncoding::Colour::ShDc;

    auto make = [&](uint64_t words, uint32_t element, const char* label, const void* initial = nullptr) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(words, 1) * 4;
        desc.elementBytes = element;
        desc.label = label;
        return gpu::Buffer::create(device, desc, initial);
    };
    auto file = make(bytes.size() / 4, 0, "asg.file", bytes.data());
    auto records = make(uint64_t{count} * kFloatsPerRecord, 4, "asg.records");
    auto crypto = make(count, 4, "asg.crypto");
    auto thin = make(count, 4, "asg.thinWalled");
    auto direct = make(uint64_t{count} * std::max(cloud.directCount, 1U), 4, "asg.direct");
    auto indirect = make(uint64_t{count} * std::max(cloud.indirectCount, 1U), 4, "asg.indirect");
    auto bits = make(uint64_t{count} * std::max(cloud.shadowWords, 1U), 4, "asg.shadowBits");
    for (auto* r : {&file, &records, &crypto, &thin, &direct, &indirect, &bits}) {
        if (!*r) return std::move(*r).error();
    }
    auto kernel = gpu::ComputeKernel::create(library, "athenea/scene/asg_decode", "asgDecode");
    if (!kernel) return std::move(kernel).error();

    // Run once to build the pipeline, then timed: what a frame that decodes a
    // chunk pays.
    for (int pass = 0; pass < 2; ++pass) {
    const auto start = std::chrono::steady_clock::now();
    gpu::CommandBatch batch(device);
    kernel->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) {
        c["file"].setBinding(file->rhi());
        c["records"].setBinding(records->rhi());
        c["crypto"].setBinding(crypto->rhi());
        c["thinWalled"].setBinding(thin->rhi());
        c["direct"].setBinding(direct->rhi());
        c["indirect"].setBinding(indirect->rhi());
        c["shadowBits"].setBinding(bits->rhi());
        auto p = c["params"];
        p["count"].setData(count);
        p["anchors"].setData(anchors);
        p["flags"].setData(flags);
        p["anchorSize"].setData(anchorSize);
        p["offAnchors"].setData(static_cast<uint32_t>(offAnchors));
        p["offLeaves"].setData(static_cast<uint32_t>(offLeaves));
        p["offMaterials"].setData(static_cast<uint32_t>(offMat));
        p["offShadowDict"].setData(static_cast<uint32_t>(offShadowDict));
        // The widths: tilt, theta, ls, aniso, thick, op, col, rough, shadow, tw,
        // full_pos, full_rot, then tilt, theta, ls, aniso of a glossy anchor.
        const uint32_t widths[16] = {bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37],
                                     bytes[38], bytes[39], bytes[40], bytes[41], bytes[42], bytes[43],
                                     bytes[44], bytes[45], bytes[46], bytes[47]};
        p["widths0"].setData(widths, 16);
        p["widths1"].setData(widths + 4, 16);
        p["widths2"].setData(widths + 8, 16);
        p["widths3"].setData(widths + 12, 16);
        p["blocks"].setData(transfer ? blocks : 0U);
        p["shadowBitsPer"].setData(shadows ? S : 0U);
        p["meanK"].setData(meanK);
        const uint32_t dims[4] = {D[0], D[1], D[2], 0};
        const uint32_t ks[4] = {K[0], K[1], K[2], 0};
        const uint32_t clusterAt[4] = {meanClusterAt[0], meanClusterAt[1], meanClusterAt[2], 0};
        const uint32_t meanF16At[4] = {meanAt[0], meanAt[1], meanAt[2], 0};
        const uint32_t scales[4] = {scaleAt[0], scaleAt[1], scaleAt[2], 0};
        const uint32_t residual[4] = {residualAt[0], residualAt[1], residualAt[2], 0};
        const uint32_t tables[4] = {meansTableAt[0], meansTableAt[1], meansTableAt[2], 0};
        const uint32_t bases[4] = {basesAt[0], basesAt[1], basesAt[2], 0};
        p["dims"].setData(dims, 16);
        p["ks"].setData(ks, 16);
        p["meanClusterAt"].setData(clusterAt, 16);
        p["meanAt"].setData(meanF16At, 16);
        p["scaleAt"].setData(scales, 16);
        p["residualAt"].setData(residual, 16);
        p["meansTableAt"].setData(tables, 16);
        p["basesAt"].setData(bases, 16);
        p["majorityAt"].setData(majorityAt);
        p["dictionaryAt"].setData(dictionaryAt);
        p["stride"].setData(kFloatsPerRecord);
    });
    ATHENEA_TRY(batch.submit(true));
    if (pass == 1) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        log::info("asg: {} gaussians from {} anchors decoded in {:.2f} ms", count, anchors, ms);
    }
    }
    cloud.records = std::move(*records);
    cloud.crypto = std::move(*crypto);
    cloud.thinWalled = std::move(*thin);
    cloud.direct = std::move(*direct);
    cloud.indirect = std::move(*indirect);
    cloud.shadowBits = std::move(*bits);
    return cloud;
}

}   // namespace athenea::scene
