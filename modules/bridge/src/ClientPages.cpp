// Copyright (c) 2026 jesus luque.
#include "athenea/bridge/ClientPages.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "athenea/core/Log.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"

namespace athenea::bridge {
namespace {

uint64_t roundUp(uint64_t value, uint64_t to) {
    return (value + to - 1) / to * to;
}

uint32_t restFor(uint32_t degree) {
    return (degree + 1) * (degree + 1) - 1;
}

}   // namespace

io::SplatEncoding clientRecordEncoding(uint32_t degree) {
    io::SplatEncoding e;
    e.x = 0;
    e.y = 1;
    e.z = 2;
    e.opacity = 3;
    e.scale0 = 4;
    e.scale1 = 5;
    e.scale2 = 6;
    e.rotW = 7;
    e.rotX = 8;
    e.rotY = 9;
    e.rotZ = 10;
    e.dc0 = 11;
    e.dc1 = 12;
    e.dc2 = 13;
    e.restBase = 14;
    e.restPerColour = restFor(degree);
    e.restColourOuter = 1;
    e.floatsPerRecord = 14 + 3 * restFor(degree);
    return e;
}

namespace {
io::SplatEncoding recordEncoding(uint32_t degree) {
    return clientRecordEncoding(degree);
}
}   // namespace

ClientPages::~ClientPages() {
    if (server_ != nullptr) server_->withdraw(info_.token);
    if (key_ != 0) lod::DevicePages::withdraw(key_);
}

Result<std::shared_ptr<ClientPages>> ClientPages::create(gpu::Device& device, gpu::ShaderLibrary& library,
                                                         platform::HandleServer& server, uint64_t records,
                                                         uint32_t degree) {
    if (device.backend() != gpu::Backend::Metal) {
        return Error::make(ErrorCode::Unsupported, "device pages from another process on {}: not done (Metal)",
                           gpu::toString(device.backend()));
    }
    if (degree > 3 || records == 0 || records > (uint64_t{1} << 26)) {
        return Error(ErrorCode::InvalidArgument, "device pages: 1 to 2^26 records of degree 0 to 3");
    }
    auto out = std::shared_ptr<ClientPages>(new ClientPages());
    out->device_ = &device;
    out->server_ = &server;
    Info& info = out->info_;
    info.degree = degree;
    info.records = records;
    info.floatsPerRecord = 14 + 3 * restFor(degree);
    const uint64_t page = platform::pageSize();
    const uint64_t recordBytes = roundUp(records * info.floatsPerRecord * 4, page);
    info.bytes = page + recordBytes;
    char name[32];
    std::snprintf(name, sizeof name, "ap-%x-%08x", platform::processId(),
                  static_cast<uint32_t>(platform::randomToken()));
    info.memory = name;
    auto memory = platform::SharedMemory::create(info.memory, info.bytes);
    if (!memory) return std::move(memory).error();
    out->memory_ = std::move(*memory);
    out->header_ = new (out->memory_.bytes().data()) PagesHeader();
    out->header_->floatsPerRecord = info.floatsPerRecord;
    out->header_->degree = degree;
    out->header_->records = records;
    auto buffer = gpu::Buffer::overSharedPages(device, out->memory_.bytes().data() + page, recordBytes, 4,
                                               "bridge.pages");
    if (!buffer) return std::move(buffer).error();
    out->records_ = std::move(*buffer);
    auto written = gpu::SharedFence::create(device, 0);
    if (!written) return std::move(written).error();
    out->written_ = std::move(*written);
    auto exported = out->written_.exportHandle();
    if (!exported) return std::move(exported).error();
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return std::move(loader).error();
    out->loader_ = std::make_unique<scene::CloudLoader>(std::move(*loader));
    info.server = server.name();
    info.token = platform::randomToken();
    server.offer(info.token, {*exported});
    out->key_ = lod::DevicePages::publish(out);
    return out;
}

Result<std::vector<std::byte>> ClientPages::page(uint64_t ticket, uint32_t chunk, uint64_t first, uint32_t count,
                                                 uint64_t fenceValue, bool antialiased) {
    if (count == 0 || first + count > info_.records || first > UINT32_MAX) {
        return Error::make(ErrorCode::InvalidArgument, "records {}..{} outside the region's {}", first,
                           first + count, info_.records);
    }
    lod::DevicePageHeader h;
    h.source = key_;
    h.frame = ticket;
    h.chunk = chunk;
    h.first = static_cast<uint32_t>(first);
    h.count = count;
    h.antialiased = antialiased ? 1 : 0;
    h.payloadBytes = sizeof(ClientPagePayload);
    ClientPagePayload payload;
    payload.fenceValue = fenceValue;
    {
        std::lock_guard lock(mutex_);
        uint64_t& kept = fences_[ticket];
        kept = std::max(kept, fenceValue);
    }
    std::vector<std::byte> bytes(sizeof h + sizeof payload);
    std::memcpy(bytes.data(), &h, sizeof h);
    std::memcpy(bytes.data() + sizeof h, &payload, sizeof payload);
    return bytes;
}

void ClientPages::committed(uint64_t ticket, uint32_t pages) {
    std::lock_guard lock(mutex_);
    tickets_[ticket].first += pages;
}

Result<void> ClientPages::place(gpu::CommandBatch& batch, const lod::DevicePageHeader& page,
                                const gpu::Buffer& staging, uint64_t payloadOffset, scene::GpuSplats& store,
                                uint64_t at, uint32_t keepRest, const void* pool, uint64_t fence) {
    (void)staging;
    (void)payloadOffset;
    std::lock_guard lock(mutex_);
    if (uint64_t{page.first} + page.count > info_.records || page.count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "a device page of chunk {} outside its region", page.chunk);
    }
    // THE DEVICE WAITS for the producer's device to have written the records.
    const auto value = fences_.find(page.frame);
    if (value != fences_.end() && value->second > 0) {
        ATHENEA_TRY(written_.waitBefore(batch, value->second));
        waits_.fetch_add(1);
    }
    size_t k = 0;
    while (k < scratch_.size() && (scratchBusy_[k] || scratchCapacity_[k] < page.count)) ++k;
    const uint64_t recordBytes = uint64_t{info_.floatsPerRecord} * 4;
    if (k == scratch_.size()) {
        const uint32_t capacity = std::max(page.count, 4096u);
        auto made = loader_->decodeScratch(capacity);
        if (!made) return std::move(made).error();
        gpu::BufferDesc desc;
        desc.bytes = capacity * recordBytes;
        desc.elementBytes = 4;
        desc.label = "bridge.pages.records";
        auto records = gpu::Buffer::create(*device_, desc);
        if (!records) return std::move(records).error();
        scratch_.push_back(std::move(*made));
        scratchRecords_.push_back(std::move(*records));
        scratchBusy_.push_back(false);
        scratchCapacity_.push_back(capacity);
    }
    // The page's records, copied on the device to where the decode starts
    // reading (it reads a buffer from its first record).
    batch.encoder()->copyBuffer(scratchRecords_[k].rhi(), 0, records_.rhi(), uint64_t{page.first} * recordBytes,
                                uint64_t{page.count} * recordBytes);
    batch.markDirty();
    const io::SplatEncoding encoding = recordEncoding(info_.degree);
    const uint32_t keep = std::min(encoding.restPerColour, keepRest);
    ATHENEA_TRY(loader_->decodeInto(batch, scratchRecords_[k], encoding, page.count, static_cast<uint32_t>(at), keep,
                                    store, scratch_[k], page.first));
    scratchBusy_[k] = true;
    uses_.push_back({pool, fence, page.frame, k});
    placed_.fetch_add(1);
    std::atomic_ref<uint64_t>(header_->placed).fetch_add(1, std::memory_order_release);
    return ok();
}

void ClientPages::retired(const void* pool, uint64_t completed) {
    std::lock_guard lock(mutex_);
    for (auto it = uses_.begin(); it != uses_.end();) {
        if (it->pool == pool && it->fence <= completed) {
            scratchBusy_[it->scratch] = false;
            tickets_[it->ticket].second += 1;
            it = uses_.erase(it);
        } else {
            ++it;
        }
    }
    // The newest ticket whose committed pages are all retired, where no
    // page placed and still running is of it or older.
    uint64_t oldestRunning = UINT64_MAX;
    for (const Use& u : uses_) oldestRunning = std::min(oldestRunning, u.ticket);
    uint64_t consumed = header_->consumed;
    for (const auto& [ticket, counts] : tickets_) {
        if (ticket >= oldestRunning) break;
        if (counts.first > 0 && counts.second >= counts.first) consumed = std::max(consumed, ticket);
    }
    std::atomic_ref<uint64_t>(header_->consumed).store(consumed, std::memory_order_release);
    const uint64_t forget = std::min(consumed, oldestRunning == UINT64_MAX ? consumed : oldestRunning - 1);
    tickets_.erase(tickets_.begin(), tickets_.upper_bound(forget));
    fences_.erase(fences_.begin(), fences_.upper_bound(forget));
}

ClientPages::Stats ClientPages::stats() const {
    Stats s;
    s.placed = placed_.load();
    s.waits = waits_.load();
    s.consumed = std::atomic_ref<uint64_t>(header_->consumed).load(std::memory_order_acquire);
    return s;
}

Result<ClientPagesView> ClientPagesView::open(gpu::Device& device, const ClientPages::Info& info) {
    ClientPagesView view;
    auto memory = platform::SharedMemory::open(info.memory, info.bytes);
    if (!memory) return std::move(memory).error();
    view.memory_ = std::move(*memory);
    const auto* header = reinterpret_cast<const PagesHeader*>(view.memory_.bytes().data());
    if (header->magic != kPagesMagic || header->floatsPerRecord != info.floatsPerRecord) {
        return Error(ErrorCode::InvalidArgument, "device pages that do not say what the stream said");
    }
    const uint64_t page = platform::pageSize();
    auto records = gpu::Buffer::overSharedPages(device, view.memory_.bytes().data() + page, info.bytes - page, 4,
                                                "producer.pages");
    if (!records) return std::move(records).error();
    view.records_ = std::move(*records);
    auto handles = platform::fetchHandles(info.server, info.token);
    if (!handles) return std::move(handles).error();
    if (handles->empty()) return Error(ErrorCode::InvalidArgument, "device pages handed over no fence");
    auto written = gpu::SharedFence::import(device, handles->front());
    for (const platform::OsHandle& h : *handles) platform::closeOsHandle(h);
    if (!written) return std::move(written).error();
    view.written_ = std::move(*written);
    return view;
}

uint64_t ClientPagesView::consumed() const {
    auto* header = reinterpret_cast<PagesHeader*>(memory_.bytes().data());
    return std::atomic_ref<uint64_t>(header->consumed).load(std::memory_order_acquire);
}

}   // namespace athenea::bridge
