// Copyright (c) 2026 jesus luque.
#include "athenea/video/Athv.h"

#include <algorithm>
#include <cstring>

#include <zlib.h>

#include "athenea/core/Platform.h"

namespace athenea::video {
namespace {

constexpr uint32_t kFrameMagic = 0x46565441;   // "ATVF"
constexpr uint64_t kHeaderBytes = 64;
constexpr uint32_t kMostParameterSets = 16;
constexpr uint64_t kMostRecord = uint64_t{1} << 31;

struct Out {
    std::vector<uint8_t> bytes;
    void u32(uint32_t v) { put(&v, 4); }
    void u64(uint64_t v) { put(&v, 8); }
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
    uint32_t u32() {
        uint32_t v = 0;
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
    o.u32(kFrameMagic);
    o.u32(0);   // the record's bytes, below
    o.u32(f.index);
    o.u32(f.splats);
    o.u32(f.keyframe ? 1u : 0u);
    o.u32(f.source);
    o.u32(f.firstSplat);
    o.u32(f.sourceSplats);
    o.u64(f.ptpTime);
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
    o.u32(static_cast<uint32_t>(f.ranges.size()));
    o.u32(static_cast<uint32_t>(f.high.size()));
    o.u32(static_cast<uint32_t>(deflated.size()));
    o.u32(static_cast<uint32_t>(f.parameterSets.size()));
    o.u32(static_cast<uint32_t>(f.unit.size()));
    o.u32(0);
    o.put(f.ranges.data(), f.ranges.size() * sizeof(float));
    o.put(deflated.data(), deflated.size());
    o.align();
    for (const auto& ps : f.parameterSets) {
        o.u32(static_cast<uint32_t>(ps.size()));
        o.put(ps.data(), ps.size());
        o.align();
    }
    o.put(f.unit.data(), f.unit.size());
    o.align();
    const uint32_t total = static_cast<uint32_t>(o.bytes.size());
    std::memcpy(o.bytes.data() + 4, &total, 4);
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
    o.u32((header_.hiInVideo ? 1u : 0u) | (header_.antialiased ? 2u : 0u));
    o.u32(0);
    o.f64(header_.rate);
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

Result<void> AthvWriter::finish(uint32_t maxSplats) {
    header_.maxSplats = std::max(header_.maxSplats, maxSplats);
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
    if (r->size_ < kHeaderBytes || !r->file_.read(reinterpret_cast<char*>(h.data()), kHeaderBytes)) {
        return Error::make(ErrorCode::InvalidArgument, "athv: {} is shorter than a header", r->name_);
    }
    In in{h.data(), h.size()};
    const uint32_t magic = in.u32(), version = in.u32();
    if (magic != AthvHeader::kMagic || version != AthvHeader::kVersion) {
        return Error::make(ErrorCode::InvalidArgument, "athv: {} is not a version {} .athv", r->name_,
                           AthvHeader::kVersion);
    }
    AthvHeader& hd = r->header_;
    hd.codec = in.u32();
    hd.bits = in.u32();
    hd.restPerColour = in.u32();
    hd.chunkSplats = in.u32();
    hd.columns = in.u32();
    hd.maxSplats = in.u32();
    const uint32_t flags = in.u32();
    in.u32();
    hd.rate = in.f64();
    hd.hiInVideo = (flags & 1u) != 0;
    hd.antialiased = (flags & 2u) != 0;
    // The records, walked by their lengths.
    for (uint64_t at = kHeaderBytes; at < r->size_;) {
        // magic, bytes, index, splats, flags, source, first splat, source splats, PTP time, time code
        uint8_t head[48] = {};
        r->file_.seekg(static_cast<std::streamoff>(at));
        if (r->size_ - at < sizeof head || !r->file_.read(reinterpret_cast<char*>(head), sizeof head)) {
            return Error::make(ErrorCode::InvalidArgument, "athv: {}: a broken record at byte {}", r->name_, at);
        }
        In rec{head, sizeof head};
        const uint32_t recordMagic = rec.u32(), bytes = rec.u32();
        rec.u32();
        rec.u32();
        rec.u32();
        const uint32_t source = rec.u32();
        rec.u32();
        const uint32_t sourceSplats = rec.u32();
        rec.u64();
        const double time = rec.f64();
        // A slice of the frame before, or the next frame: a picture never
        // names a frame far past the last.
        if (recordMagic != kFrameMagic || bytes < sizeof head || bytes > r->size_ - at || source > r->slices_.size()) {
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
    char label[16] = {};
    in.take(label, sizeof label);
    label[15] = 0;
    f.timecode = label;
    const uint32_t rangeCount = in.u32(), highRaw = in.u32(), highStored = in.u32(), sets = in.u32(),
                   unitBytes = in.u32();
    in.u32();
    if (in.bad || rangeCount > 4096 || sets > kMostParameterSets) {
        return Error::make(ErrorCode::InvalidArgument, "athv: {}: frame {} has a broken header", name_, k);
    }
    f.ranges.resize(rangeCount);
    in.take(f.ranges.data(), f.ranges.size() * sizeof(float));
    std::vector<uint8_t> deflated(highStored);
    in.take(deflated.data(), deflated.size());
    in.align();
    f.highStored = highStored;
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
