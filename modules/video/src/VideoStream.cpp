// Copyright (c) 2026 jesus luque.
#include "athenea/video/VideoStream.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/lod/SplatStream.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::video {
namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kKeepDropped = 16;   // frames newer than one never placed whole before it goes

double msSince(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

}   // namespace

Result<std::shared_ptr<VideoStream>> VideoStream::open(const std::filesystem::path& path,
                                                       gpu::ShaderLibrary& library) {
    std::shared_ptr<VideoStream> s(new VideoStream());
    auto reader = AthvReader::open(path);
    if (!reader) return std::move(reader).error();
    s->reader_ = std::move(*reader);
    if (s->reader_->frames() == 0) {
        return Error::make(ErrorCode::InvalidArgument, "video: {} holds no frames", path.string());
    }
    const AthvHeader& h = s->reader_->header();
    auto decoder = createHardwareDecoder(h.codec, h.bits);
    if (!decoder) return std::move(decoder).error();
    s->decoder_ = std::move(*decoder);
    auto textures = createLumaTextures(library.device());
    if (!textures) return std::move(textures).error();
    s->textures_ = std::move(*textures);
    auto codec = PlanesCodec::create(library);
    if (!codec) return std::move(codec).error();
    s->codec_ = std::make_unique<PlanesCodec>(std::move(*codec));
    s->device_ = &library.device();
    s->key_ = DevicePages::publish(s);
    return s;
}

VideoStream::~VideoStream() = default;

void VideoStream::close() {
    if (key_ != 0) {
        DevicePages::withdraw(key_);
        key_ = 0;
    }
}

uint32_t VideoStream::chunks() const noexcept {
    const AthvHeader& h = reader_->header();
    return h.chunkSplats == 0 ? 0 : (reader_->mostSourceSplats() + h.chunkSplats - 1) / h.chunkSplats;
}

double VideoStream::time(uint32_t k) const {
    return reader_->time(reader_->slicesOf(k).front());
}

uint64_t VideoStream::frameBytes(uint32_t k) const {
    uint64_t bytes = 0;
    for (const uint32_t picture : reader_->slicesOf(k)) bytes += reader_->recordBytes(picture);
    return bytes;
}

Result<void> VideoStream::push(lod::PageQueue& queue, uint32_t stream, uint32_t k, uint64_t epoch, uint64_t ptpTime,
                               uint32_t pages) {
    const AthvHeader& h = reader_->header();
    if (k >= reader_->sourceFrames()) {
        return Error::make(ErrorCode::InvalidArgument, "video: no frame {}", k);
    }
    // Every picture of the frame read and decoded first: the epoch's pages
    // go out together.
    struct Picture {
        AthvFrame   frame;
        PlaneLayout layout;
        uint64_t    ticket = 0;
    };
    std::vector<Picture> pictures;
    double readMs = 0.0, decodeMs = 0.0;
    for (const uint32_t index : reader_->slicesOf(k)) {
        Clock::time_point t = Clock::now();
        auto frame = reader_->read(index);
        if (!frame) return std::move(frame).error();
        readMs += msSince(t);
        t = Clock::now();
        auto decoded = decoder_->decode(frame->unit, frame->parameterSets, frame->keyframe);
        if (!decoded) return std::move(decoded).error();
        decodeMs += msSince(t);
        auto layout = PlaneLayout::make(frame->splats, h.bits, h.restPerColour, h.hiInVideo, h.chunkSplats, h.columns,
                                        h.maxSplats, 0, 0, h.chroma, h.geometry);
        if (!layout) return std::move(layout).error();
        layout->positionBlock = h.positionBlock;
        layout->positionHighBits = h.positionHighBits;
        layout->geometryPositionBits = h.geometryPositionBits;
        layout->geometryScaleBits = h.geometryScaleBits;
        layout->geometryRotationBits = h.geometryRotationBits;
        layout->positionTarget = h.positionTarget;
        if ((*decoded)->width() != layout->width || (*decoded)->height() != layout->height ||
            frame->ranges.size() != size_t{layout->attributes()} * 2 ||
            frame->high.size() != uint64_t{layout->chunks()} * layout->highChunkBytes() ||
            frame->firstSplat % h.chunkSplats != 0) {
            return Error::make(ErrorCode::InvalidArgument,
                               "video: picture {} decoded {} x {} with {} ranges and {} high bytes, not its layout's",
                               index, (*decoded)->width(), (*decoded)->height(), frame->ranges.size(),
                               frame->high.size());
        }
        Picture p;
        p.layout = *layout;
        {
            std::lock_guard lock(mutex_);
            p.ticket = nextTicket_++;
            Held& held = held_[p.ticket];
            held.frame = std::move(*decoded);
            held.layout = *layout;
            held.geometryAt = GeometryFrame{frame->source, h.geometryInterval, h.geometryStagger, h.geometryOrder};
            held.frameFirst = frame->firstSplat;
            held.pagesLeft = layout->chunks();
        }
        p.frame = std::move(*frame);
        pictures.push_back(std::move(p));
    }
    {
        std::lock_guard lock(mutex_);
        stats_.readMs.push_back(readMs);
        stats_.decodeMs.push_back(decodeMs);
        evictLocked();
    }
    uint32_t chunks = 0;
    for (const Picture& p : pictures) {
        chunks = std::max(chunks, p.frame.firstSplat / h.chunkSplats + p.layout.chunks());
    }
    pages = std::max(pages, chunks);
    std::vector<uint8_t> covered(pages, 0);
    for (const Picture& p : pictures) {
        const PlaneLayout& layout = p.layout;
        scene::Bounds bounds;
        for (int axis = 0; axis < 3; ++axis) {
            bounds.min[axis] = p.frame.ranges[axis * 2];
            bounds.max[axis] = p.frame.ranges[axis * 2 + 1];
        }
        const uint64_t rangeBytes = layout.rangeBytes(), highBytes = layout.highChunkBytes(),
                       geometryBytes = layout.geometryChunkBytes();
        const uint32_t firstChunk = p.frame.firstSplat / h.chunkSplats;
        for (uint32_t c = 0; c < layout.chunks(); ++c) {
            lod::PageUpdate u;
            u.stream = stream;
            u.epoch = epoch;
            u.chunk = firstChunk + c;
            u.pages = pages;
            u.ptpTime = ptpTime;
            lod::DevicePageHeader ph;
            ph.source = key_;
            ph.frame = p.ticket;
            ph.chunk = c;
            ph.first = c * layout.chunkSplats;
            ph.count = std::min(layout.chunkSplats, layout.splats - ph.first);
            ph.antialiased = h.antialiased ? 1u : 0u;
            ph.payloadBytes = static_cast<uint32_t>(rangeBytes + highBytes + geometryBytes);
            u.op = lod::PageUpdate::Op::Replace;
            u.bounds = bounds;
            u.bytes.resize(sizeof ph + ph.payloadBytes);
            std::memcpy(u.bytes.data(), &ph, sizeof ph);
            std::memcpy(u.bytes.data() + sizeof ph, p.frame.ranges.data(), rangeBytes);
            if (highBytes != 0) {
                std::memcpy(u.bytes.data() + sizeof ph + rangeBytes, p.frame.high.data() + c * highBytes, highBytes);
            }
            if (geometryBytes != 0 && p.frame.geometry.size() >= (c + 1) * geometryBytes) {
                std::memcpy(u.bytes.data() + sizeof ph + rangeBytes + highBytes,
                            p.frame.geometry.data() + c * geometryBytes, geometryBytes);
            }
            if (u.chunk < pages) covered[u.chunk] = 1;
            queue.push(std::move(u));
        }
    }
    // Chunks past the frame's own: removed (a frame of fewer than the most).
    for (uint32_t c = 0; c < pages; ++c) {
        if (covered[c] != 0) continue;
        lod::PageUpdate u;
        u.stream = stream;
        u.epoch = epoch;
        u.chunk = c;
        u.pages = pages;
        u.ptpTime = ptpTime;
        u.op = lod::PageUpdate::Op::Remove;
        queue.push(std::move(u));
    }
    return ok();
}

Result<void> VideoStream::place(gpu::CommandBatch& batch, const lod::DevicePageHeader& page,
                                const gpu::Buffer& staging, uint64_t payloadOffset, scene::GpuSplats& store,
                                uint64_t at, uint32_t keepRest, const void* pool, uint64_t fence) {
    std::lock_guard lock(mutex_);
    const auto it = held_.find(page.frame);
    if (it == held_.end()) {
        return Error::make(ErrorCode::NotFound, "video: chunk {} of a frame no longer held", page.chunk);
    }
    Held& held = it->second;
    if (!held.luma.texture) {
        auto luma = textures_->wrap(*held.frame);
        if (!luma) return std::move(luma).error();
        held.luma = std::move(*luma);
    }
    if (page.first + page.count > held.layout.splats || page.payloadBytes < held.layout.rangeBytes()) {
        return Error::make(ErrorCode::InvalidArgument, "video: chunk {} of {} splats past its frame's {}",
                           page.chunk, page.count, held.layout.splats);
    }
    if (held.layout.geometry) {
        const uint64_t want = uint64_t{reader_->mostSourceSplats()} * 32;   // this frame's words and the one before's
        if (!geometryState_.valid() || geometryState_.bytes() < want) {
            gpu::BufferDesc desc;
            desc.bytes = std::max<uint64_t>(want, 16);
            desc.elementBytes = 4;
            desc.label = "video.geometryState";
            auto made = gpu::Buffer::create(*device_, desc);
            if (!made) return std::move(made).error();
            geometryState_ = std::move(*made);
        }
        ATHENEA_TRY(codec_->geometryApply(batch, held.layout, staging.rhi(),
                                          payloadOffset + held.layout.rangeBytes() + held.layout.highChunkBytes(),
                                          held.geometryAt, held.frameFirst, page.first, page.count, geometryState_));
    }
    ATHENEA_TRY(codec_->decode(batch, held.layout, held.luma.texture.get(), held.luma.chroma.get(), staging.rhi(),
                               payloadOffset, payloadOffset + held.layout.rangeBytes(), page.first, page.count, store,
                               at, keepRest, held.layout.geometry ? &geometryState_ : nullptr, held.frameFirst));
    held.uses.emplace_back(pool, fence);
    held.pagesLeft = held.pagesLeft > 0 ? held.pagesLeft - 1 : 0;
    ++stats_.placed;
    return ok();
}

void VideoStream::retired(const void* pool, uint64_t completed) {
    std::lock_guard lock(mutex_);
    for (auto& [ticket, held] : held_) {
        auto& uses = held.uses;
        uses.erase(std::remove_if(uses.begin(), uses.end(),
                                  [&](const auto& u) { return u.first == pool && u.second <= completed; }),
                   uses.end());
    }
    evictLocked();
}

void VideoStream::evictLocked() {
    const uint64_t newest = held_.empty() ? 0 : held_.rbegin()->first;
    for (auto it = held_.begin(); it != held_.end();) {
        const Held& h = it->second;
        const bool done = h.pagesLeft == 0 && h.uses.empty();
        const bool abandoned = h.uses.empty() && it->first + kKeepDropped < newest;
        it = (done || abandoned) && it->first != newest ? held_.erase(it) : std::next(it);
    }
    stats_.held = held_.size();
}

VideoStream::Stats VideoStream::stats() const {
    std::lock_guard lock(mutex_);
    Stats out = stats_;
    out.hardware = decoder_->hardware();
    return out;
}

}   // namespace athenea::video
