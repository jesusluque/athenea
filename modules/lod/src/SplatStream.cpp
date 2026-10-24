// Copyright (c) 2026 jesus luque.
#include "athenea/lod/SplatStream.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <map>

#include "athenea/core/Log.h"
#include "athenea/io/Readers.h"
#include "SpzStaging.h"

namespace athenea::lod {

void PageQueue::push(PageUpdate page) {
    std::lock_guard lock(mutex_);
    pending_[page.stream].push_back(std::move(page));
}

void PageQueue::drain(uint32_t stream, std::vector<PageUpdate>& out) {
    std::vector<PageUpdate> taken;
    {
        std::lock_guard lock(mutex_);
        const auto it = pending_.find(stream);
        if (it == pending_.end()) {
            return;
        }
        taken.swap(it->second);
    }
    for (PageUpdate& p : taken) {
        out.push_back(std::move(p));
    }
}

std::shared_ptr<PageQueue> PageQueue::named(const std::string& name) {
    static std::mutex                                                 mutex;
    static std::unordered_map<std::string, std::shared_ptr<PageQueue>> queues;
    std::lock_guard lock(mutex);
    std::shared_ptr<PageQueue>& queue = queues[name];
    if (queue == nullptr) {
        queue = std::make_shared<PageQueue>();
    }
    return queue;
}

namespace {

/// An epoch's pages as they come: by chunk, a later page of a chunk in the
/// same epoch replacing the earlier.
struct Gathering {
    uint32_t                         pages = 0;      // the epoch's, as its first page said
    uint32_t                         received = 0;
    uint64_t                         ptpTime = 0;
    std::map<uint32_t, PageUpdate>   byChunk;
};

}   // namespace

struct LiveSource::Impl {
    std::shared_ptr<PageQueue> queue;
    LiveSettings               settings;

    // Frame thread (poll).
    std::map<uint64_t, Gathering> gathering;
    uint64_t                      handed = 0;   // the last epoch handed over or dropped
    uint64_t                      nextTicket = 1;

    // Shared with the loaders and status readers.
    mutable std::mutex                                          mutex;
    std::unordered_map<uint64_t, std::vector<std::byte>>        staged;   // ticket -> bytes
    std::vector<scene::Bounds>                                  bounds;   // per chunk, latest handed out
    Stats                                                       stats;
};

LiveSource::LiveSource() : impl_(std::make_unique<Impl>()) {}
LiveSource::~LiveSource() = default;

Result<std::unique_ptr<LiveSource>> LiveSource::create(std::shared_ptr<PageQueue> queue,
                                                       const LiveSettings& settings) {
    if (queue == nullptr) {
        return Error(ErrorCode::InvalidArgument, "a live source needs a page queue");
    }
    const ChunkFormat& f = settings.format;
    if (settings.chunks == 0 || f.chunkSplats == 0 || f.shWords == 0 ||
        uint64_t{settings.chunks} * f.chunkSplats > UINT32_MAX) {
        return Error::make(ErrorCode::InvalidArgument,
                           "{}: a live stream needs chunks, splats a chunk and SH words, and at most 2^32 splats",
                           settings.name);
    }
    std::unique_ptr<LiveSource> source(new LiveSource());
    source->impl_->queue = std::move(queue);
    source->impl_->settings = settings;
    source->impl_->bounds.assign(settings.chunks, settings.bounds);
    return source;
}

Result<void> LiveSource::check(const PageUpdate& page, const LiveSettings& settings) {
    const ChunkFormat& f = settings.format;
    if (page.stream != settings.stream) {
        return Error::make(ErrorCode::InvalidArgument, "{}: a page of stream {}, not {}", settings.name, page.stream,
                           settings.stream);
    }
    if (page.chunk >= settings.chunks) {
        return Error::make(ErrorCode::InvalidArgument, "{}: chunk {}, past its {}", settings.name, page.chunk,
                           settings.chunks);
    }
    if (page.epoch == 0 || page.pages == 0) {
        return Error::make(ErrorCode::InvalidArgument, "{}: a page needs an epoch from 1 and its epoch's page count",
                           settings.name);
    }
    if (page.op == PageUpdate::Op::Remove) {
        if (!page.bytes.empty()) {
            return Error::make(ErrorCode::InvalidArgument, "{}: a remove carries no bytes", settings.name);
        }
        return ok();
    }
    if (f.payload == ChunkFormat::Payload::Spz) {
        // An SPZ file: how many splats it holds is known once a loader has
        // decompressed it (fetch says so then).
        if (page.bytes.empty()) {
            return Error::make(ErrorCode::InvalidArgument, "{}: chunk {} page is empty", settings.name, page.chunk);
        }
        return ok();
    }
    const uint64_t per = f.elementBytes();
    const uint64_t n = page.bytes.size() / per;
    if (page.bytes.size() % per != 0 || n == 0 || n > f.chunkSplats) {
        return Error::make(ErrorCode::InvalidArgument,
                           "{}: chunk {} page of {} bytes is not 1 to {} splats of {} bytes", settings.name,
                           page.chunk, page.bytes.size(), f.chunkSplats, per);
    }
    return ok();
}

const ChunkFormat& LiveSource::format() const noexcept {
    return impl_->settings.format;
}

uint32_t LiveSource::chunkCount() const noexcept {
    return impl_->settings.chunks;
}

scene::Bounds LiveSource::bounds(uint32_t chunk) const {
    std::lock_guard lock(impl_->mutex);
    return chunk < impl_->bounds.size() ? impl_->bounds[chunk] : impl_->settings.bounds;
}

std::string LiveSource::name() const {
    return impl_->settings.name;
}

Result<LodCloud> LiveSource::openCloud(gpu::Device& /*device*/) {
    const LiveSettings& s = impl_->settings;
    LodCloud lod;
    lod.chunkSplats = s.format.chunkSplats;
    lod.count = s.chunks * s.format.chunkSplats;
    lod.splats.bounds = s.bounds;
    return lod;
}

ChunkRequest LiveSource::request(uint32_t chunk) const {
    return {chunk, 0, 0, 0};
}

Result<void> LiveSource::fetch(const ChunkRequest& request, std::span<std::byte> staging) {
    std::vector<std::byte> bytes;
    {
        std::lock_guard lock(impl_->mutex);
        const auto it = impl_->staged.find(request.ticket);
        if (it == impl_->staged.end()) {
            return Error::make(ErrorCode::NotFound, "{}: no page {} to fetch", name(), request.ticket);
        }
        bytes = std::move(it->second);
        impl_->staged.erase(it);
    }
    if (impl_->settings.format.payload == ChunkFormat::Payload::Spz) {
        // Decompressed here, on the loader's thread, and laid in staging as
        // a header and the six streams; the device does the rest.
        auto raw = io::readSpz(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()),
                               name() + " chunk " + std::to_string(request.chunk));
        if (!raw) return std::move(raw).error();
        if (raw->count > impl_->settings.format.chunkSplats ||
            ChunkFormat::kSpzHeaderBytes + raw->packed.size() > staging.size()) {
            return Error::make(ErrorCode::InvalidArgument, "{}: chunk {} page holds {} splats, past a chunk's {}",
                               name(), request.chunk, raw->count, impl_->settings.format.chunkSplats);
        }
        SpzStaged h;
        h.count = raw->count;
        h.shDim = raw->spz.shDim;
        h.fractionalBits = raw->spz.fractionalBits;
        h.smallestThree = raw->spz.smallestThree ? 1u : 0u;
        h.flipYZ = raw->encoding.flipYZ ? 1u : 0u;
        h.antialiased = raw->antialiased.value_or(false) ? 1u : 0u;
        const io::SpzStreams& at = raw->spz;
        const uint64_t offsets[6] = {at.positions, at.alphas, at.colours, at.scales, at.rotations, at.sh};
        std::copy(offsets, offsets + 6, h.offsets);
        std::memcpy(staging.data(), &h, sizeof h);
        std::memcpy(staging.data() + ChunkFormat::kSpzHeaderBytes, raw->packed.data(), raw->packed.size());
        return ok();
    }
    if (bytes.size() != staging.size()) {
        return Error::make(ErrorCode::InvalidArgument, "{}: page {} is {} bytes, staged into {}", name(),
                           request.ticket, bytes.size(), staging.size());
    }
    std::memcpy(staging.data(), bytes.data(), bytes.size());
    return ok();
}

void LiveSource::discard(const ChunkRequest& request) {
    std::lock_guard lock(impl_->mutex);
    impl_->staged.erase(request.ticket);
}

void LiveSource::poll(std::vector<ChunkChange>& out) {
    Impl& s = *impl_;
    std::vector<PageUpdate> pages;
    s.queue->drain(s.settings.stream, pages);
    uint64_t accepted = 0, refused = 0, late = 0, dropped = 0, epochs = 0;
    for (PageUpdate& page : pages) {
        if (const auto checked = check(page, s.settings); !checked) {
            log::warn("{}", checked.error().message());
            ++refused;
            continue;
        }
        if (page.epoch <= s.handed) {
            ++late;
            continue;
        }
        Gathering& g = s.gathering[page.epoch];
        if (g.received == 0) {
            g.pages = page.pages;
            g.ptpTime = page.ptpTime;
        } else if (g.pages != page.pages) {
            log::warn("{}: epoch {} said {} pages, and now {}", s.settings.name, page.epoch, g.pages, page.pages);
            ++refused;
            continue;
        }
        ++g.received;
        ++accepted;
        const uint32_t chunk = page.chunk;
        g.byChunk.insert_or_assign(chunk, std::move(page));
    }
    // The latest complete epoch; those before it are handed over in order
    // when complete, and dropped when not.
    uint64_t complete = 0;
    for (const auto& [epoch, g] : s.gathering) {
        if (g.received >= g.pages) {
            complete = epoch;
        }
    }
    std::vector<std::pair<uint64_t, std::vector<std::byte>>> toStage;
    std::vector<std::pair<uint32_t, scene::Bounds>> newBounds;
    const uint64_t per = s.settings.format.elementBytes();
    for (auto it = s.gathering.begin(); it != s.gathering.end() && it->first <= complete;) {
        Gathering& g = it->second;
        if (g.received < g.pages) {
            log::warn("{}: epoch {} dropped with {} of its {} pages: epoch {} came whole first", s.settings.name,
                      it->first, g.received, g.pages, complete);
            ++dropped;
        } else {
            for (auto& [chunk, page] : g.byChunk) {
                ChunkChange c;
                c.epoch = it->first;
                c.op = page.op;
                c.bounds = page.bounds;
                c.ptpTime = g.ptpTime;
                c.request.chunk = chunk;
                if (page.op != PageUpdate::Op::Remove) {
                    c.request.ticket = s.nextTicket++;
                    const bool spz = s.settings.format.payload == ChunkFormat::Payload::Spz;
                    // An SPZ page: staging for the most a chunk takes, and
                    // its count known when a loader has decompressed it.
                    c.request.bytes = spz ? s.settings.format.chunkBytes() : page.bytes.size();
                    c.request.splats = spz ? 0u : static_cast<uint32_t>(page.bytes.size() / per);
                    toStage.emplace_back(c.request.ticket, std::move(page.bytes));
                    newBounds.emplace_back(chunk, page.bounds);
                }
                out.push_back(c);
            }
            ++epochs;
        }
        it = s.gathering.erase(it);
    }
    s.handed = std::max(s.handed, complete);
    std::lock_guard lock(s.mutex);
    for (auto& [ticket, bytes] : toStage) {
        s.staged.emplace(ticket, std::move(bytes));
    }
    for (const auto& [chunk, b] : newBounds) {
        s.bounds[chunk] = b;
    }
    s.stats.pages += accepted;
    s.stats.refused += refused;
    s.stats.late += late;
    s.stats.dropped += dropped;
    s.stats.epochs += epochs;
}

LiveSource::Stats LiveSource::stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->stats;
}

}   // namespace athenea::lod
