// Copyright (c) 2026 jesus luque.
#include "athenea/bridge/Frames.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>

#include <nlohmann/json.hpp>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/technique/DisplayTransform.h"

namespace athenea::bridge {
namespace {

uint64_t roundUp(uint64_t value, uint64_t to) {
    return (value + to - 1) / to * to;
}

uint64_t loadAcquire(const uint64_t& word) {
    return std::atomic_ref<uint64_t>(const_cast<uint64_t&>(word)).load(std::memory_order_acquire);
}

void storeRelease(uint64_t& word, uint64_t value) {
    std::atomic_ref<uint64_t>(word).store(value, std::memory_order_release);
}

/// A shared memory name of at most 30 characters: what, this process, random.
std::string regionName(const char* what) {
    char name[32];
    std::snprintf(name, sizeof name, "%s-%x-%08x", what, platform::processId(),
                  static_cast<uint32_t>(platform::randomToken()));
    return name;
}

const char* backendName(gpu::Backend backend) {
    return gpu::toString(backend);
}

}   // namespace

const char* formatName(Format format) noexcept {
    switch (format) {
        case Format::R32F: return "r32f";
        case Format::R16F: return "r16f";
        case Format::RGBA32F: return "rgba32f";
        case Format::RGBA16F: return "rgba16f";
        case Format::R32U: return "r32u";
        case Format::RGBA8: return "rgba8";
    }
    return "?";
}

std::optional<Format> parseFormat(const std::string& name) noexcept {
    for (Format f : {Format::R32F, Format::R16F, Format::RGBA32F, Format::RGBA16F, Format::R32U, Format::RGBA8}) {
        if (name == formatName(f)) return f;
    }
    return std::nullopt;
}

uint32_t bytesPerPixel(Format format) noexcept {
    switch (format) {
        case Format::R32F: return 4;
        case Format::R16F: return 2;
        case Format::RGBA32F: return 16;
        case Format::RGBA16F: return 8;
        case Format::R32U: return 4;
        case Format::RGBA8: return 4;
    }
    return 0;
}

Result<Format> formatFor(const std::string& aov, const std::string& format) {
    struct Rule {
        const char*         aov;
        std::vector<Format> formats;   // the first is the default
    };
    static const Rule rules[] = {
        {"color", {Format::RGBA32F, Format::RGBA16F, Format::RGBA8}},
        {"alpha", {Format::R32F, Format::R16F}},
        {"depth", {Format::R32F, Format::R16F}},
        {"normal", {Format::RGBA32F, Format::RGBA16F}},
        {"Neye", {Format::RGBA32F, Format::RGBA16F}},
        {"primId", {Format::R32U}},
        {"instanceId", {Format::R32U}},
        {"elementId", {Format::R32U}},
        {"CryptoObject00", {Format::RGBA32F}},
        {"CryptoObject01", {Format::RGBA32F}},
        {"CryptoObject02", {Format::RGBA32F}},
    };
    for (const Rule& rule : rules) {
        if (aov != rule.aov) continue;
        if (format.empty()) return rule.formats.front();
        const std::optional<Format> asked = parseFormat(format);
        if (asked && std::find(rule.formats.begin(), rule.formats.end(), *asked) != rule.formats.end()) {
            return *asked;
        }
        std::string allowed;
        for (Format f : rule.formats) allowed += std::string(allowed.empty() ? "" : ", ") + formatName(f);
        return Error::make(ErrorCode::InvalidArgument, "AOV '{}' comes as {}, not '{}'", aov, allowed, format);
    }
    return Error::make(ErrorCode::InvalidArgument,
                       "no AOV '{}': color, alpha, depth, normal, Neye, primId, instanceId, elementId, "
                       "CryptoObject00..02",
                       aov);
}

// --- the answer ------------------------------------------------------------------

std::string SubscriptionInfo::toJson() const {
    nlohmann::json aovList = nlohmann::json::array();
    for (const Aov& a : aovs) {
        aovList.push_back({{"name", a.name}, {"format", formatName(a.format)}, {"offset", a.offset}, {"bytes", a.bytes}});
    }
    char token16[17];
    std::snprintf(token16, sizeof token16, "%016llx", static_cast<unsigned long long>(token));
    const nlohmann::json out{{"ring", ring},         {"ringBytes", ringBytes}, {"data", data},
                             {"dataBytes", dataBytes}, {"server", server},     {"token", token16},
                             {"backend", backend},   {"slots", slots},         {"width", width},
                             {"height", height},     {"sync", fenceSync ? "fence" : "poll"},
                             {"slotBytes", slotBytes}, {"stateOffset", stateOffset},
                             {"aovs", aovList},       {"rows", "bottom first; rgba8 top first"}};
    return out.dump();
}

Result<SubscriptionInfo> SubscriptionInfo::fromJson(const std::string& text) {
    try {
        const nlohmann::json in = nlohmann::json::parse(text);
        SubscriptionInfo info;
        info.ring = in.at("ring").get<std::string>();
        info.ringBytes = in.at("ringBytes").get<uint64_t>();
        info.data = in.at("data").get<std::string>();
        info.dataBytes = in.at("dataBytes").get<uint64_t>();
        info.server = in.at("server").get<std::string>();
        info.token = std::stoull(in.at("token").get<std::string>(), nullptr, 16);
        info.backend = in.at("backend").get<std::string>();
        info.slots = in.at("slots").get<uint32_t>();
        info.width = in.at("width").get<uint32_t>();
        info.height = in.at("height").get<uint32_t>();
        info.fenceSync = in.at("sync").get<std::string>() == "fence";
        info.slotBytes = in.at("slotBytes").get<uint64_t>();
        info.stateOffset = in.value("stateOffset", uint64_t{0});
        for (const nlohmann::json& a : in.at("aovs")) {
            Aov aov;
            aov.name = a.at("name").get<std::string>();
            const auto format = parseFormat(a.at("format").get<std::string>());
            if (!format) return Error(ErrorCode::InvalidArgument, "a subscription's AOV of no format this knows");
            aov.format = *format;
            aov.offset = a.at("offset").get<uint64_t>();
            aov.bytes = a.at("bytes").get<uint64_t>();
            info.aovs.push_back(std::move(aov));
        }
        return info;
    } catch (const std::exception& error) {
        return Error::make(ErrorCode::InvalidArgument, "not a subscription: {}", error.what());
    }
}

// --- the engine's side -------------------------------------------------------------

struct FramePublisher::Impl {
    FrameThread             thread;   // bound where create ran
    gpu::Device*            device = nullptr;
    platform::HandleServer* server = nullptr;
    PublisherDesc           desc;
    platform::SharedMemory  ring, data;
    RingHeader*             header = nullptr;
    std::vector<std::vector<gpu::Buffer>> slotBuffers;   // Metal: [slot][aov]
    std::vector<gpu::Buffer> stateBuffers;               // Metal: [slot]
    gpu::Buffer             whole;                       // elsewhere: every slot
    gpu::SharedFence        frames, release;
    gpu::ComputeKernel      pack, shown;
    std::unique_ptr<technique::DisplayTransform> display;
    std::optional<gpu::Texture> shownTexture;
    std::atomic<uint64_t>   published{0}, skipped{0}, completed{0}, unblocked{0};
};

FramePublisher::~FramePublisher() {
    if (impl_ != nullptr && impl_->server != nullptr) {
        impl_->server->withdraw(info_.token);
    }
}

Result<std::unique_ptr<FramePublisher>> FramePublisher::create(gpu::Device& device, gpu::ShaderLibrary& library,
                                                               platform::HandleServer& server,
                                                               const PublisherDesc& desc) {
    if (desc.width == 0 || desc.height == 0 || desc.width > 16384 || desc.height > 16384) {
        return Error::make(ErrorCode::InvalidArgument, "a subscription of {} x {} pixels (1 to 16384 a side)",
                           desc.width, desc.height);
    }
    if (desc.slots < 2 || desc.slots > 8) {
        return Error::make(ErrorCode::InvalidArgument, "a subscription of {} slots (2 to 8)", desc.slots);
    }
    if (desc.aovs.empty() || desc.aovs.size() > kMaxAovs) {
        return Error::make(ErrorCode::InvalidArgument, "a subscription of {} AOVs (1 to {})", desc.aovs.size(),
                           kMaxAovs);
    }
    const bool metal = device.backend() == gpu::Backend::Metal;
    if (!metal && device.backend() != gpu::Backend::Vulkan) {
        return Error::make(ErrorCode::Unsupported, "frames to another process on {}: not done (Metal; Vulkan "
                                                   "untested)", gpu::toString(device.backend()));
    }
    auto out = std::unique_ptr<FramePublisher>(new FramePublisher());
    out->impl_ = std::make_unique<Impl>();
    Impl& impl = *out->impl_;
    impl.device = &device;
    impl.server = &server;
    impl.desc = desc;

    // THE LAYOUT. Each AOV on whole pages on Metal, so either side wraps it
    // as a buffer of its own; words elsewhere, at offsets in one buffer.
    const uint64_t align = metal ? platform::pageSize() : 256;
    SubscriptionInfo& info = out->info_;
    uint64_t at = 0;
    for (const AovSpec& spec : desc.aovs) {
        SubscriptionInfo::Aov aov;
        aov.name = spec.aov;
        aov.format = spec.format;
        aov.offset = at;
        aov.bytes = roundUp(uint64_t{desc.width} * desc.height * bytesPerPixel(spec.format), 4);
        at += roundUp(aov.bytes, align);
        info.aovs.push_back(aov);
    }
    // The frame's state: its own page (Metal), so it is a buffer of its own too.
    info.stateOffset = at;
    at += roundUp(sizeof(FrameState), align);
    info.slotBytes = at;
    info.slots = desc.slots;
    info.width = desc.width;
    info.height = desc.height;
    info.fenceSync = desc.fenceSync;
    info.backend = backendName(device.backend());
    info.dataBytes = info.slotBytes * desc.slots;
    info.ringBytes = roundUp(sizeof(RingHeader), platform::pageSize());

    // THE RING: the header, the layout, the records.
    info.ring = regionName("ab");
    auto ring = platform::SharedMemory::create(info.ring, info.ringBytes);
    if (!ring) return std::move(ring).error();
    impl.ring = std::move(*ring);
    impl.header = new (impl.ring.bytes().data()) RingHeader();
    RingHeader& h = *impl.header;
    h.backend = static_cast<uint32_t>(device.backend());
    h.slots = desc.slots;
    h.width = desc.width;
    h.height = desc.height;
    h.aovCount = static_cast<uint32_t>(desc.aovs.size());
    h.fenceSync = desc.fenceSync ? 1 : 0;
    h.slotBytes = info.slotBytes;
    h.dataBytes = info.dataBytes;
    h.stateOffset = info.stateOffset;
    for (size_t k = 0; k < info.aovs.size(); ++k) {
        std::snprintf(h.aovs[k].name, sizeof h.aovs[k].name, "%s", info.aovs[k].name.c_str());
        h.aovs[k].format = static_cast<uint32_t>(info.aovs[k].format);
        h.aovs[k].bytesPerPixel = bytesPerPixel(info.aovs[k].format);
        h.aovs[k].offset = info.aovs[k].offset;
        h.aovs[k].bytes = info.aovs[k].bytes;
    }

    // THE SLOTS.
    std::vector<platform::OsHandle> handles;
    if (metal) {
        info.data = regionName("ad");
        auto data = platform::SharedMemory::create(info.data, info.dataBytes);
        if (!data) return std::move(data).error();
        impl.data = std::move(*data);
        impl.slotBuffers.resize(desc.slots);
        for (uint32_t s = 0; s < desc.slots; ++s) {
            for (const SubscriptionInfo::Aov& aov : info.aovs) {
                std::byte* pages = impl.data.bytes().data() + uint64_t{s} * info.slotBytes + aov.offset;
                auto buffer = gpu::Buffer::overSharedPages(device, pages, roundUp(aov.bytes, align), 4,
                                                           "bridge.slot." + aov.name);
                if (!buffer) return std::move(buffer).error();
                impl.slotBuffers[s].push_back(std::move(*buffer));
            }
            std::byte* pages = impl.data.bytes().data() + uint64_t{s} * info.slotBytes + info.stateOffset;
            auto state = gpu::Buffer::overSharedPages(device, pages, align, 4, "bridge.slot.state");
            if (!state) return std::move(state).error();
            impl.stateBuffers.push_back(std::move(*state));
        }
    } else {
        gpu::BufferDesc whole;
        whole.bytes = info.dataBytes;
        whole.elementBytes = 4;
        whole.shared = true;
        whole.label = "bridge.slots";
        auto buffer = gpu::Buffer::create(device, whole);
        if (!buffer) return std::move(buffer).error();
        impl.whole = std::move(*buffer);
    }

    // THE FENCES, and what the subscriber fetches.
    auto frames = gpu::SharedFence::create(device, 0);
    if (!frames) return std::move(frames).error();
    impl.frames = std::move(*frames);
    auto release = gpu::SharedFence::create(device, 0);
    if (!release) return std::move(release).error();
    impl.release = std::move(*release);
    for (gpu::SharedFence* fence : {&impl.frames, &impl.release}) {
        auto exported = fence->exportHandle();
        if (!exported) {
            for (const platform::OsHandle& made : handles) platform::closeOsHandle(made);
            return std::move(exported).error();
        }
        handles.push_back(*exported);
    }
    if (!metal) {
        auto shared = impl.whole.sharedHandle();
        if (!shared || shared->type != rhi::NativeHandleType::FileDescriptor) {
            for (const platform::OsHandle& made : handles) platform::closeOsHandle(made);
            return Error(ErrorCode::Unsupported, "the slots' buffer exports no descriptor");
        }
        auto copy = platform::duplicateOsHandle({platform::OsHandle::Kind::Fd, shared->value});
        if (!copy) return std::move(copy).error();
        handles.push_back(*copy);
    }
    info.server = server.name();
    info.token = platform::randomToken();
    server.offer(info.token, std::move(handles));

    // THE KERNELS.
    auto pack = gpu::ComputeKernel::create(library, "athenea/bridge/bridge_pack", "bridgePack");
    if (!pack) return std::move(pack).error();
    impl.pack = std::move(*pack);
    for (const AovSpec& spec : desc.aovs) {
        if (spec.format != Format::RGBA8 || impl.display != nullptr) continue;
        auto shown = gpu::ComputeKernel::create(library, "athenea/bridge/bridge_shown", "bridgePackShown");
        if (!shown) return std::move(shown).error();
        impl.shown = std::move(*shown);
        auto display = technique::DisplayTransform::create(library);
        if (!display) return std::move(display).error();
        impl.display = std::make_unique<technique::DisplayTransform>(std::move(*display));
        gpu::TextureDesc tex;
        tex.width = desc.width;
        tex.height = desc.height;
        tex.format = rhi::Format::RGBA8Unorm;
        tex.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
        tex.label = "bridge.shown";
        auto texture = gpu::Texture::create(device, tex);
        if (!texture) return std::move(texture).error();
        impl.shownTexture = std::move(*texture);
    }
    return out;
}

FramePublisher::Stats FramePublisher::stats() const {
    Stats s;
    s.published = impl_->published.load();
    s.skipped = impl_->skipped.load();
    s.completed = impl_->completed.load();
    s.unblocked = impl_->unblocked.load();
    return s;
}

Result<bool> FramePublisher::publish(uint64_t frame, std::span<const AovSource> sources, const RingRecord& record,
                                     const FrameTotals& totals) {
    Impl& impl = *impl_;
    ATHENEA_TRY(impl.thread.require("FramePublisher::publish"));
    if (frame == 0 || frame <= loadAcquire(impl.header->published)) {
        return Error::make(ErrorCode::InvalidArgument, "frame {} after frame {}: frames rise from 1", frame,
                           loadAcquire(impl.header->published));
    }
    if (sources.size() != impl.desc.aovs.size()) {
        return Error::make(ErrorCode::InvalidArgument, "{} AOV sources for {} AOVs", sources.size(),
                           impl.desc.aovs.size());
    }
    const uint32_t slots = impl.desc.slots;
    const uint32_t slot = static_cast<uint32_t>((frame - 1) % slots);
    const uint64_t previous = frame > slots ? frame - slots : 0;   // the frame the slot last held
    if (!impl.desc.fenceSync && previous > 0 && loadAcquire(impl.header->released) < previous) {
        impl.skipped.fetch_add(1);
        return false;
    }
    const uint32_t width = impl.desc.width;
    const uint32_t height = impl.desc.height;
    const uint32_t pixels = width * height;
    gpu::CommandBatch batch(*impl.device);
    if (impl.desc.fenceSync && previous > 0) {
        ATHENEA_TRY(impl.release.waitBefore(batch, previous));
    }
    for (size_t k = 0; k < sources.size(); ++k) {
        const AovSpec& spec = impl.desc.aovs[k];
        const SubscriptionInfo::Aov& aov = info_.aovs[k];
        const gpu::Buffer& dst = impl.slotBuffers.empty() ? impl.whole : impl.slotBuffers[slot][k];
        const uint64_t dstByte = impl.slotBuffers.empty() ? uint64_t{slot} * info_.slotBytes + aov.offset : 0;
        const uint32_t dstWord = static_cast<uint32_t>(dstByte / 4);
        if (spec.format == Format::RGBA8) {
            if (sources[k].shown == nullptr) {
                return Error::make(ErrorCode::InvalidArgument, "AOV {} as rgba8 wants the colour's display source",
                                   spec.aov);
            }
            technique::DisplaySettings settings;
            ATHENEA_TRY(impl.display->run(batch, *sources[k].shown, settings, impl.shownTexture->rhi(), width, height));
            impl.shown.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["shown"].setBinding(impl.shownTexture->rhi());
                cursor["dst"].setBinding(dst.rhi());
                cursor["params"]["width"].setData(width);
                cursor["params"]["height"].setData(height);
                cursor["params"]["dstWord"].setData(dstWord);
            });
            continue;
        }
        const AovSource& src = sources[k];
        if (src.buffer == nullptr || !src.buffer->valid()) {
            // An AOV this frame did not draw (no mesh wrote an id): the slot
            // holds zeros.
            batch.encoder()->clearBuffer(dst.rhi(), dstByte, roundUp(aov.bytes, 4));
            batch.markDirty();
            continue;
        }
        const uint32_t components =
            spec.format == Format::RGBA32F || spec.format == Format::RGBA16F ? 4u : 1u;
        const bool halves = spec.format == Format::R16F || spec.format == Format::RGBA16F;
        const uint32_t values = pixels * components;
        const uint32_t words = halves ? (values + 1) / 2 : values;
        if (uint64_t{src.stride} * pixels > src.buffer->bytes() / 4) {
            return Error::make(ErrorCode::InvalidArgument, "AOV {}: its buffer holds fewer than {} x {} pixels",
                               spec.aov, width, height);
        }
        impl.pack.dispatch(batch, {words, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["src"].setBinding(src.buffer->rhi());
            cursor["dst"].setBinding(dst.rhi());
            cursor["params"]["pixels"].setData(pixels);
            cursor["params"]["stride"].setData(src.stride);
            cursor["params"]["offset"].setData(src.offset);
            cursor["params"]["components"].setData(components);
            cursor["params"]["format"].setData(uint32_t{halves ? 1u : 0u});
            cursor["params"]["dstWord"].setData(dstWord);
            cursor["params"]["words"].setData(words);
        });
    }
    // THE FRAME'S STATE, beside its AOVs: the rasteriser's totals as the
    // device left them, and the pairs its buffers held.
    {
        const gpu::Buffer& state = impl.stateBuffers.empty() ? impl.whole : impl.stateBuffers[slot];
        const uint64_t at = impl.stateBuffers.empty() ? uint64_t{slot} * info_.slotBytes + info_.stateOffset : 0;
        if (totals.totals != nullptr && totals.totals->valid()) {
            batch.encoder()->copyBuffer(state.rhi(), at, totals.totals->rhi(), 0, 8);
        } else {
            batch.encoder()->clearBuffer(state.rhi(), at, 8);
        }
        const uint32_t tail[2] = {totals.pairCapacity, 1};
        batch.encoder()->uploadBufferData(state.rhi(), at + 8, sizeof tail, tail);
        batch.markDirty();
    }
    // The record, before the frame goes: the subscriber reads it once the
    // fence says the frame is there.
    RingRecord& kept = impl.header->records[frame % kRecords];
    storeRelease(kept.frame, 0);
    RingRecord copy = record;
    copy.frame = 0;
    copy.slot = slot;
    std::memcpy(reinterpret_cast<std::byte*>(&kept) + sizeof(uint64_t),
                reinterpret_cast<const std::byte*>(&copy) + sizeof(uint64_t), sizeof(RingRecord) - sizeof(uint64_t));
    storeRelease(kept.frame, frame);
    ATHENEA_TRY(impl.frames.signalAfter(batch, frame));
    storeRelease(impl.header->published, frame);
    impl.published.fetch_add(1);
    return true;
}

uint64_t FramePublisher::observe() {
    Impl& impl = *impl_;
    const uint64_t done = impl.frames.value();
    if (done > loadAcquire(impl.header->completed)) {
        storeRelease(impl.header->completed, done);
        impl.completed.store(done);
    }
    const uint64_t published = loadAcquire(impl.header->published);
    return published > done ? published - done : 0;
}

void FramePublisher::unblock() {
    Impl& impl = *impl_;
    const uint64_t published = loadAcquire(impl.header->published);
    if (impl.release.value() < published) {
        impl.release.setValue(published);
        impl.unblocked.fetch_add(1);
    }
}

// --- the subscriber's side -----------------------------------------------------------

struct FrameSubscriber::Impl {
    gpu::Device*           device = nullptr;
    platform::SharedMemory ring, data;
    RingHeader*            header = nullptr;
    std::vector<std::vector<gpu::Buffer>> slotBuffers;
    gpu::SharedFence       frames, release;
};

FrameSubscriber::~FrameSubscriber() = default;

Result<std::unique_ptr<FrameSubscriber>> FrameSubscriber::open(gpu::Device& device, const SubscriptionInfo& info) {
    if (device.backend() != gpu::Backend::Metal) {
        return Error::make(ErrorCode::Unsupported, "a subscriber on {}: not done (Metal)", gpu::toString(device.backend()));
    }
    auto out = std::unique_ptr<FrameSubscriber>(new FrameSubscriber());
    out->impl_ = std::make_unique<Impl>();
    out->info_ = info;
    Impl& impl = *out->impl_;
    impl.device = &device;
    auto ring = platform::SharedMemory::open(info.ring, info.ringBytes);
    if (!ring) return std::move(ring).error();
    impl.ring = std::move(*ring);
    if (impl.ring.bytes().size() < sizeof(RingHeader)) {
        return Error(ErrorCode::InvalidArgument, "a frame ring smaller than its header");
    }
    impl.header = reinterpret_cast<RingHeader*>(impl.ring.bytes().data());
    if (impl.header->magic != kRingMagic || impl.header->version != 1 || impl.header->slots != info.slots ||
        impl.header->slotBytes != info.slotBytes) {
        return Error(ErrorCode::InvalidArgument, "a frame ring that does not say what the subscription said");
    }
    auto data = platform::SharedMemory::open(info.data, info.dataBytes);
    if (!data) return std::move(data).error();
    impl.data = std::move(*data);
    const uint64_t page = platform::pageSize();
    impl.slotBuffers.resize(info.slots);
    for (uint32_t s = 0; s < info.slots; ++s) {
        for (const SubscriptionInfo::Aov& aov : info.aovs) {
            if (uint64_t{s} * info.slotBytes + aov.offset + roundUp(aov.bytes, page) > info.dataBytes) {
                return Error(ErrorCode::InvalidArgument, "a subscription's AOV outside its slots");
            }
            std::byte* pages = impl.data.bytes().data() + uint64_t{s} * info.slotBytes + aov.offset;
            auto buffer = gpu::Buffer::overSharedPages(device, pages, roundUp(aov.bytes, page), 4,
                                                       "subscriber.slot." + aov.name);
            if (!buffer) return std::move(buffer).error();
            impl.slotBuffers[s].push_back(std::move(*buffer));
        }
    }
    if (info.fenceSync) {
        auto handles = platform::fetchHandles(info.server, info.token);
        if (!handles) return std::move(handles).error();
        if (handles->size() < 2) {
            for (const platform::OsHandle& h : *handles) platform::closeOsHandle(h);
            return Error(ErrorCode::InvalidArgument, "a subscription that handed over fewer than two fences");
        }
        auto frames = gpu::SharedFence::import(device, (*handles)[0]);
        auto release = frames ? gpu::SharedFence::import(device, (*handles)[1]) : Result<gpu::SharedFence>(frames.error());
        for (const platform::OsHandle& h : *handles) platform::closeOsHandle(h);
        if (!frames) return std::move(frames).error();
        if (!release) return std::move(release).error();
        impl.frames = std::move(*frames);
        impl.release = std::move(*release);
    }
    return out;
}

const RingHeader& FrameSubscriber::header() const noexcept {
    return *impl_->header;
}

FrameState FrameSubscriber::state(uint32_t slot) const {
    FrameState out;
    if (slot >= info_.slots) return out;
    std::memcpy(&out, impl_->data.bytes().data() + uint64_t{slot} * info_.slotBytes + info_.stateOffset, sizeof out);
    return out;
}

uint64_t FrameSubscriber::completed() const {
    return info_.fenceSync ? impl_->frames.value() : loadAcquire(impl_->header->completed);
}

std::optional<RingRecord> FrameSubscriber::record(uint64_t frame) const {
    const RingRecord& kept = impl_->header->records[frame % kRecords];
    if (loadAcquire(kept.frame) != frame) return std::nullopt;
    RingRecord copy;
    std::memcpy(&copy, &kept, sizeof copy);
    if (loadAcquire(kept.frame) != frame) return std::nullopt;
    copy.frame = frame;
    return copy;
}

Result<uint64_t> FrameSubscriber::waitFrame(uint64_t atLeast, std::chrono::milliseconds timeout) const {
    const auto until = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        const uint64_t done = completed();
        if (done >= atLeast) return done;
        if (std::chrono::steady_clock::now() >= until) {
            return Error::make(ErrorCode::IoFailure, "no frame {} within {} ms (the newest is {})", atLeast,
                               timeout.count(), done);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

const gpu::Buffer& FrameSubscriber::aov(uint32_t slot, uint32_t index) const {
    return impl_->slotBuffers.at(slot).at(index);
}

uint64_t FrameSubscriber::aovOffset(uint32_t, uint32_t) const {
    return 0;
}

Result<void> FrameSubscriber::release(gpu::CommandBatch& batch, uint64_t frame) {
    if (info_.fenceSync) {
        return impl_->release.signalAfter(batch, frame);
    }
    ATHENEA_TRY(batch.submit(true));
    uint64_t& word = impl_->header->released;
    if (frame > loadAcquire(word)) storeRelease(word, frame);
    return ok();
}

}   // namespace athenea::bridge
