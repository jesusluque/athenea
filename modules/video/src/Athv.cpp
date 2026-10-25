// Copyright (c) 2026 jesus luque.
#include "athenea/video/Athv.h"

#include <algorithm>
#include <cstring>

#include <zlib.h>

#include "athenea/core/Platform.h"

namespace athenea::video {
namespace {

constexpr uint32_t kFrameMagic = 0x46565441;   // "ATVF"
constexpr uint64_t kHeaderBytes = 128;   // version 2; version 1's was 64
constexpr uint32_t kMostParameterSets = 16;
constexpr uint64_t kMostRecord = uint64_t{1} << 31;

struct Out {
    std::vector<uint8_t> bytes;
    void u16(uint16_t v) { put(&v, 2); }
    void u32(uint32_t v) { put(&v, 4); }
    void u64(uint64_t v) { put(&v, 8); }
    void f32(float v) { put(&v, 4); }
    void f64(double v) { put(&v, 8); }
    void put(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        bytes.insert(bytes.end(), b, b + n);
    }
    void align() {
        while (bytes.size() % 4 != 0) bytes.push_back(0);
    }
};

/// A bounds-checked cursor over a record.
struct In {
    const uint8_t* p = nullptr;
    size_t         n = 0;
    size_t         at = 0;
    bool           bad = false;
    bool take(void* into, size_t k) {
        if (bad || n - at < k) {
            bad = true;
            return false;
        }
        std::memcpy(into, p + at, k);
        at += k;
        return true;
    }
    uint16_t u16() {
        uint16_t v = 0;
        take(&v, 2);
        return v;
    }
    uint32_t u32() {
        uint32_t v = 0;
        take(&v, 4);
        return v;
    }
    float f32() {
        float v = 0;
        take(&v, 4);
        return v;
    }
    uint64_t u64() {
        uint64_t v = 0;
        take(&v, 8);
        return v;
    }
    double f64() {
        double v = 0;
        take(&v, 8);
        return v;
    }
    void align() { at = std::min(n, (at + 3) / 4 * 4); }
};

}   // namespace

Result<std::unique_ptr<AthvWriter>> AthvWriter::create(const std::filesystem::path& path, const AthvHeader& header) {
    std::unique_ptr<AthvWriter> w(new AthvWriter());
    // Written under a partial name, renamed by finish: a failure leaves no
    // half file under the name asked for.
    w->path_ = path;
    w->partial_ = platform::partialPathFor(path);
    w->header_ = header;
    w->file_.open(w->partial_, std::ios::binary | std::ios::trunc);
    if (!w->file_) {
        return Error::make(ErrorCode::IoFailure, "athv: cannot write {}", w->partial_.string());
    }
    ATHENEA_TRY(w->writeHeader());
    return w;
}

AthvWriter::~AthvWriter() {
    if (file_.is_open()) {
        file_.close();
        platform::removeFile(partial_);
    }
}

Result<void> AthvWriter::write(AthvFrame& f) {
    Out o;
    // The fixed prefix (AthvRecordPrefix): magic, type, rendition, frame,
    // PTP time, the record's bytes, version, flags.
    o.u32(kFrameMagic);
    o.u16(0);
    o.u16(static_cast<uint16_t>(header_.rendition));
    o.u32(f.frame);
    o.u64(f.ptpTime);
    o.u32(0);   // the record's bytes, below
    o.u16(static_cast<uint16_t>(AthvHeader::kVersion));
    o.u16(static_cast<uint16_t>((f.keyframe ? 1u : 0u) | (f.geometryKey ? 2u : 0u)));
    o.u32(f.index);
    o.u32(f.source);
    o.u32(f.slice);
    o.u32(f.slices);
    o.u32(f.firstSplat);
    o.u32(f.splats);
    o.u32(f.sourceSplats);
    o.f64(f.time);
    char label[16] = {};
    std::memcpy(label, f.timecode.data(), std::min<size_t>(f.timecode.size(), sizeof label - 1));
    o.put(label, sizeof label);
    std::vector<uint8_t> deflated;
    if (!f.high.empty()) {
        uLongf size = compressBound(static_cast<uLong>(f.high.size()));
        deflated.resize(size);
        if (compress2(deflated.data(), &size, reinterpret_cast<const Bytef*>(f.high.data()),
                      static_cast<uLong>(f.high.size()), 9) != Z_OK) {
            return Error::make(ErrorCode::IoFailure, "athv: frame {}: the high parts would not deflate", f.index);
        }
        deflated.resize(size);
    }
    f.highStored = deflated.size();
    std::vector<uint8_t> geometry;
    if (!f.geometry.empty()) {
        uLongf size = compressBound(static_cast<uLong>(f.geometry.size()));
        geometry.resize(size);
        if (compress2(geometry.data(), &size, reinterpret_cast<const Bytef*>(f.geometry.data()),
                      static_cast<uLong>(f.geometry.size()), 6) != Z_OK) {
            return Error::make(ErrorCode::IoFailure, "athv: frame {}: the geometry would not deflate", f.index);
        }
        geometry.resize(size);
    }
    f.geometryStored = geometry.size();
    o.u32(static_cast<uint32_t>(f.ranges.size()));
    o.u32(static_cast<uint32_t>(f.high.size()));
    o.u32(static_cast<uint32_t>(deflated.size()));
    o.u32(static_cast<uint32_t>(f.parameterSets.size()));
    o.u32(static_cast<uint32_t>(f.unit.size()));
    o.u32(static_cast<uint32_t>(geometry.size()));
    o.put(f.ranges.data(), f.ranges.size() * sizeof(float));
    o.put(deflated.data(), deflated.size());
    o.align();
    o.put(geometry.data(), geometry.size());
    o.align();
    for (const auto& ps : f.parameterSets) {
        o.u32(static_cast<uint32_t>(ps.size()));
        o.put(ps.data(), ps.size());
        o.align();
    }
    o.put(f.unit.data(), f.unit.size());
    o.align();
    const uint32_t total = static_cast<uint32_t>(o.bytes.size());
    std::memcpy(o.bytes.data() + AthvRecordPrefix::kBytesAt, &total, 4);
    file_.write(reinterpret_cast<const char*>(o.bytes.data()), static_cast<std::streamsize>(o.bytes.size()));
    if (!file_) {
        return Error::make(ErrorCode::IoFailure, "athv: cannot write picture {} to {}", f.index, partial_.string());
    }
    f.recordBytes = o.bytes.size();
    header_.maxSplats = std::max(header_.maxSplats, f.splats);
    return ok();
}

Result<void> AthvWriter::writeHeader() {
    Out o;
    o.u32(AthvHeader::kMagic);
    o.u32(AthvHeader::kVersion);
    o.u32(header_.codec);
    o.u32(header_.bits);
    o.u32(header_.restPerColour);
    o.u32(header_.chunkSplats);
    o.u32(header_.columns);
    o.u32(header_.maxSplats);
    o.u32((header_.hiInVideo ? 1u : 0u) | (header_.antialiased ? 2u : 0u) | (header_.chroma ? 4u : 0u) |
          (header_.stableOrder ? 8u : 0u) |
          (header_.geometry ? 16u : 0u) | (header_.geometryStagger ? 32u : 0u));
    o.u32(header_.rendition);
    o.f64(header_.rate);
    o.u32(header_.rateNumerator);
    o.u32(header_.rateDenominator);
    o.u32(header_.sourceFrames);
    o.f32(header_.quality);
    o.u32(header_.positionBlock);
    o.u32(header_.positionHighBits);
    o.f32(header_.positionTarget);
    o.u32(header_.geometryInterval);
    o.u32(header_.geometryPositionBits | (header_.geometryScaleBits << 8u) | (header_.geometryOrder << 16u) |
          (header_.geometryRotationBits << 24u));
    o.bytes.resize(kHeaderBytes, 0);
    const auto end = file_.tellp();
    file_.seekp(0);
    file_.write(reinterpret_cast<const char*>(o.bytes.data()), static_cast<std::streamsize>(o.bytes.size()));
    if (end > static_cast<std::streamoff>(kHeaderBytes)) {
        file_.seekp(end);
    }
    if (!file_) {
        return Error::make(ErrorCode::IoFailure, "athv: cannot write the header of {}", partial_.string());
    }
    return ok();
}

Result<void> AthvWriter::finish(uint32_t maxSplats, uint32_t sourceFrames) {
    header_.maxSplats = std::max(header_.maxSplats, maxSplats);
    header_.sourceFrames = std::max(header_.sourceFrames, sourceFrames);
    ATHENEA_TRY(writeHeader());
    file_.close();
    if (!file_) {
        return Error::make(ErrorCode::IoFailure, "athv: cannot write {}", partial_.string());
    }
    return platform::replaceFile(partial_, path_);
}

AthvReader::~AthvReader() = default;

Result<std::unique_ptr<AthvReader>> AthvReader::open(const std::filesystem::path& path) {
    std::unique_ptr<AthvReader> r(new AthvReader());
    r->name_ = path.string();
    r->file_.open(path, std::ios::binary);
    if (!r->file_) {
        return Error::make(ErrorCode::NotFound, "athv: cannot read {}", r->name_);
    }
    r->file_.seekg(0, std::ios::end);
    r->size_ = static_cast<uint64_t>(r->file_.tellg());
    r->file_.seekg(0);
    std::vector<uint8_t> h(kHeaderBytes);
    if (r->size_ < 64 ||
        !r->file_.read(reinterpret_cast<char*>(h.data()),
                       static_cast<std::streamsize>(std::min(kHeaderBytes, r->size_)))) {
        return Error::make(ErrorCode::InvalidArgument, "athv: {} is shorter than a header", r->name_);
    }
    In in{h.data(), h.size()};
    const uint32_t magic = in.u32(), version = in.u32();
    if (magic != AthvHeader::kMagic || version < 1 || version > AthvHeader::kVersion) {
        return Error::make(ErrorCode::InvalidArgument, "athv: {} is not a version 1 or {} .athv", r->name_,
                           AthvHeader::kVersion);
    }
    AthvHeader& hd = r->header_;
    hd.version = version;
    hd.codec = in.u32();
    hd.bits = in.u32();
    hd.restPerColour = in.u32();
    hd.chunkSplats = in.u32();
    hd.columns = in.u32();
    hd.maxSplats = in.u32();
    const uint32_t flags = in.u32();
    hd.rendition = version >= 2 ? in.u32() : (in.u32(), 0u);
    hd.rate = in.f64();
    hd.hiInVideo = (flags & 1u) != 0;
    hd.antialiased = (flags & 2u) != 0;
    hd.chroma = (flags & 4u) != 0;
    hd.stableOrder = (flags & 8u) != 0;
    hd.geometry = (flags & 16u) != 0;
    hd.geometryStagger = (flags & 32u) != 0;
    if (version >= 2) {
        hd.rateNumerator = in.u32();
        hd.rateDenominator = in.u32();
        hd.sourceFrames = in.u32();
        hd.quality = in.f32();
        hd.positionBlock = in.u32();
        hd.positionHighBits = in.u32();
        hd.positionTarget = in.f32();
        hd.geometryInterval = in.u32();
        const uint32_t geometryBits = in.u32();
        if (geometryBits != 0) {
            hd.geometryPositionBits = geometryBits & 0xFFu;
            hd.geometryScaleBits = (geometryBits >> 8u) & 0xFFu;
            hd.geometryOrder = (geometryBits >> 16u) & 0xFFu;
            hd.geometryRotationBits = (geometryBits >> 24u) != 0 ? geometryBits >> 24u : 10u;
        }
    } else {
        hd.positionHighBits = hd.bits;
    }
    // The records, walked by their lengths.
    for (uint64_t at = hd.bytes(); at < r->size_;) {
        // Version 2: the fixed prefix, index, source, slice, slices, first
        // splat, splats, source splats, time code. Version 1: magic, bytes,
        // index, splats, flags, source, first splat, source splats, PTP time,
        // time code.
        uint8_t head[64] = {};
        const size_t headBytes = version >= 2 ? 64 : 48;
        r->file_.seekg(static_cast<std::streamoff>(at));
        if (r->size_ - at < headBytes || !r->file_.read(reinterpret_cast<char*>(head), headBytes)) {
            return Error::make(ErrorCode::InvalidArgument, "athv: {}: a broken record at byte {}", r->name_, at);
        }
        In rec{head, headBytes};
        uint32_t recordMagic = 0, bytes = 0, source = 0, sourceSplats = 0;
        double time = 0.0;
        if (version >= 2) {
            recordMagic = rec.u32();
            rec.u16();
            rec.u16();
            rec.u32();
            rec.u64();
            bytes = rec.u32();
            rec.u32();
            rec.u32();
            source = rec.u32();
            rec.u32();
            rec.u32();
            rec.u32();
            rec.u32();
            sourceSplats = rec.u32();
            time = rec.f64();
        } else {
            recordMagic = rec.u32();
            bytes = rec.u32();
            rec.u32();
            rec.u32();
            rec.u32();
            source = rec.u32();
            rec.u32();
            sourceSplats = rec.u32();
            rec.u64();
            time = rec.f64();
        }
        // A slice of the frame before, or the next frame: a picture never
        // names a frame far past the last.
        if (recordMagic != kFrameMagic || bytes < headBytes || bytes > r->size_ - at || source > r->slices_.size()) {
            return Error::make(ErrorCode::InvalidArgument, "athv: {}: a broken record at byte {}", r->name_, at);
        }
        if (source == r->slices_.size()) {
            r->slices_.emplace_back();
        }
        r->slices_[source].push_back(static_cast<uint32_t>(r->offsets_.size()));
        r->mostSource_ = std::max(r->mostSource_, sourceSplats);
        r->offsets_.push_back(at);
        r->times_.push_back(time);
        at += bytes;
    }
    return r;
}

Result<AthvFrame> AthvReader::read(uint32_t k) {
    if (k >= offsets_.size()) {
        return Error::make(ErrorCode::InvalidArgument, "athv: {} has no frame {}", name_, k);
    }
    const uint64_t at = offsets_[k];
    const uint64_t end = k + 1 < offsets_.size() ? offsets_[k + 1] : size_;
    if (end - at > kMostRecord) {
        return Error::make(ErrorCode::InvalidArgument, "athv: {}: frame {} of {} bytes", name_, k, end - at);
    }
    std::vector<uint8_t> rec(end - at);
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(at));
    if (!file_.read(reinterpret_cast<char*>(rec.data()), static_cast<std::streamsize>(rec.size()))) {
        return Error::make(ErrorCode::IoFailure, "athv: cannot read frame {} of {}", k, name_);
    }
    In in{rec.data(), rec.size()};
    AthvFrame f;
    if (header_.version >= 2) {
        in.u32();
        in.u16();
        f.rendition = in.u16();
        f.frame = in.u32();
        f.ptpTime = in.u64();
        f.recordBytes = in.u32();
        in.u16();
        const uint16_t recordFlags = in.u16();
        f.keyframe = (recordFlags & 1u) != 0;
        f.geometryKey = (recordFlags & 2u) != 0;
        f.index = in.u32();
        f.source = in.u32();
        f.slice = in.u32();
        f.slices = in.u32();
        f.firstSplat = in.u32();
        f.splats = in.u32();
        f.sourceSplats = in.u32();
        f.time = in.f64();
    } else {
        in.u32();
        f.recordBytes = in.u32();
        f.index = in.u32();
        f.splats = in.u32();
        f.keyframe = (in.u32() & 1u) != 0;
        f.source = in.u32();
        f.firstSplat = in.u32();
        f.sourceSplats = in.u32();
        f.ptpTime = in.u64();
        f.time = in.f64();
        f.frame = f.source;
        f.slices = f.splats != 0 ? (f.sourceSplats + f.splats - 1) / f.splats : 1;
        f.slice = f.splats != 0 ? f.firstSplat / f.splats : 0;
    }
    char label[16] = {};
    in.take(label, sizeof label);
    label[15] = 0;
    f.timecode = label;
    const uint32_t rangeCount = in.u32(), highRaw = in.u32(), highStored = in.u32(), sets = in.u32(),
                   unitBytes = in.u32();
    const uint32_t geometryStored = header_.version >= 2 ? in.u32() : (in.u32(), 0u);
    if (in.bad || rangeCount > 4096 || sets > kMostParameterSets) {
        return Error::make(ErrorCode::InvalidArgument, "athv: {}: frame {} has a broken header", name_, k);
    }
    f.ranges.resize(rangeCount);
    in.take(f.ranges.data(), f.ranges.size() * sizeof(float));
    std::vector<uint8_t> deflated(highStored);
    in.take(deflated.data(), deflated.size());
    in.align();
    f.highStored = highStored;
    std::vector<uint8_t> geometry(geometryStored);
    in.take(geometry.data(), geometry.size());
    in.align();
    f.geometryStored = geometryStored;
    for (uint32_t s = 0; s < sets && !in.bad; ++s) {
        const uint32_t n = in.u32();
        if (n > in.n - in.at) {
            in.bad = true;
            break;
        }
        f.parameterSets.emplace_back(n);
        in.take(f.parameterSets.back().data(), n);
        in.align();
    }
    if (!in.bad && unitBytes <= in.n - in.at) {
        f.unit.resize(unitBytes);
        in.take(f.unit.data(), unitBytes);
    } else {
        in.bad = true;
    }
    if (in.bad) {
        return Error::make(ErrorCode::InvalidArgument, "athv: {}: frame {} is cut short", name_, k);
    }
    if (geometryStored != 0) {
        const uint64_t chunk = header_.chunkSplats;
        const uint64_t raw = chunk == 0 ? 0 : (f.splats + chunk - 1) / chunk * 16 * chunk;
        f.geometry.resize(raw);
        uLongf size = static_cast<uLongf>(raw);
        if (uncompress(reinterpret_cast<Bytef*>(f.geometry.data()), &size, geometry.data(),
                       static_cast<uLong>(geometry.size())) != Z_OK ||
            size != raw) {
            return Error::make(ErrorCode::InvalidArgument, "athv: {}: frame {}'s geometry does not inflate", name_, k);
        }
    }
    if (highRaw != 0) {
        f.high.resize(highRaw);
        uLongf size = highRaw;
        if (uncompress(reinterpret_cast<Bytef*>(f.high.data()), &size, deflated.data(),
                       static_cast<uLong>(deflated.size())) != Z_OK ||
            size != highRaw) {
            return Error::make(ErrorCode::InvalidArgument, "athv: {}: frame {}'s high parts do not inflate", name_,
                               k);
        }
    }
    return f;
}

std::vector<uint8_t> annexB(const AthvFrame& frame) {
    std::vector<uint8_t> out;
    static constexpr uint8_t kStart[4] = {0, 0, 0, 1};
    for (const auto& ps : frame.parameterSets) {
        out.insert(out.end(), kStart, kStart + 4);
        out.insert(out.end(), ps.begin(), ps.end());
    }
    // Length-prefixed NAL units: each length read as the bytes it is.
    for (size_t at = 0; at + 4 <= frame.unit.size();) {
        const uint32_t n = (uint32_t{frame.unit[at]} << 24) | (uint32_t{frame.unit[at + 1]} << 16) |
                           (uint32_t{frame.unit[at + 2]} << 8) | uint32_t{frame.unit[at + 3]};
        at += 4;
        if (n > frame.unit.size() - at) {
            break;
        }
        out.insert(out.end(), kStart, kStart + 4);
        out.insert(out.end(), frame.unit.begin() + static_cast<std::ptrdiff_t>(at),
                   frame.unit.begin() + static_cast<std::ptrdiff_t>(at + n));
        at += n;
    }
    return out;
}

}   // namespace athenea::video
