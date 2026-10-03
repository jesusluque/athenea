// Copyright (c) 2026 jesus luque.
#include "athenea/gpu/FrameProfiler.h"

#include <algorithm>

#include "athenea/core/Platform.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"

namespace athenea::gpu {
namespace {

constexpr uint32_t kPools = 4;
constexpr size_t   kKeptFrames = 8;

}   // namespace

FrameProfiler::FrameProfiler(Device& device) : device_(&device) {
    // Metal says it has no timestamp queries, and slang-rhi's Metal
    // writeTimestamp is a no-op besides: command buffers are what it times.
    timestamps_ = device.backend() != Backend::Metal && device.caps().timestampQuery;
}

FrameProfiler::~FrameProfiler() {
    if (open_.has_value()) {
        release(*open_);
    }
    for (Frame& frame : finished_) {
        release(frame);
    }
}

void FrameProfiler::release(Frame& frame) {
    for (Segment& segment : frame.segments) {
        if (segment.commandBuffer != 0) {
            platform::releaseMetalObject(reinterpret_cast<void*>(segment.commandBuffer));
            segment.commandBuffer = 0;
        }
    }
}

int FrameProfiler::top() const noexcept {
    return open_.has_value() && !open_->stack.empty() ? open_->stack.back() : -1;
}

Result<void> FrameProfiler::boundary(CommandBatch& batch) {
    Frame& frame = *open_;
    if (timestamps_) {
        if (frame.stamps >= kStamps) {
            return ok();   // a frame of more boundaries than a pool holds keeps what fitted
        }
        batch.encoder()->writeTimestamp(pools_[frame.pool].get(), frame.first + frame.stamps);
        batch.markDirty();
        frame.stamps += 1;
        return ok();
    }
    // The segment open until now ends with what the batch holds.
    auto handle = batch.submitNative();
    if (!handle) return std::move(handle).error();
    if (!frame.segments.empty() && *handle != 0) {
        platform::retainMetalObject(reinterpret_cast<void*>(*handle));
        frame.segments.back().commandBuffer = *handle;
    }
    return ok();
}

Result<void> FrameProfiler::beginFrame(CommandBatch& batch, uint64_t tag) {
    if (!enabled_) {
        return ok();
    }
    if (open_.has_value()) {
        ATHENEA_TRY(endFrame(batch));
    }
    if (timestamps_ && pools_.empty()) {
        for (uint32_t k = 0; k < kPools; ++k) {
            rhi::QueryPoolDesc desc;
            desc.type = rhi::QueryType::Timestamp;
            desc.count = kStamps;
            desc.label = "frame.phases";
            rhi::ComPtr<rhi::IQueryPool> pool;
            if (SLANG_FAILED(device_->rhi()->createQueryPool(desc, pool.writeRef()))) {
                timestamps_ = false;   // and command buffers stand in, as on Metal
                pools_.clear();
                break;
            }
            pools_.push_back(pool);
        }
    }
    Frame frame;
    frame.tag = tag;
    if (timestamps_) {
        frame.pool = nextPool_;
        nextPool_ = (nextPool_ + 1) % kPools;
        // A pool still held by a frame not yet read is taken back: that
        // frame's times are lost rather than mixed with this one's.
        for (auto it = finished_.begin(); it != finished_.end();) {
            if (it->pool == frame.pool) {
                release(*it);
                it = finished_.erase(it);
            } else {
                ++it;
            }
        }
        (void)pools_[frame.pool]->reset();
    }
    open_ = std::move(frame);
    // Whatever the batch held before the frame is not the frame's.
    if (!timestamps_) {
        auto flushed = batch.submitNative();
        if (!flushed) return std::move(flushed).error();
    } else {
        ATHENEA_TRY(boundary(batch));
    }
    open_->segments.push_back({-1, 0});
    return ok();
}

Result<void> FrameProfiler::push(CommandBatch& batch, const char* name) {
    if (!open_.has_value()) {
        return ok();
    }
    ATHENEA_TRY(boundary(batch));
    Frame& frame = *open_;
    int index = -1;
    for (size_t k = 0; k < frame.names.size(); ++k) {
        if (frame.names[k] == name) {
            index = static_cast<int>(k);
        }
    }
    if (index < 0) {
        frame.names.emplace_back(name);
        index = static_cast<int>(frame.names.size() - 1);
    }
    frame.stack.push_back(index);
    frame.segments.push_back({index, 0});
    return ok();
}

Result<void> FrameProfiler::pop(CommandBatch& batch) {
    if (!open_.has_value()) {
        return ok();
    }
    ATHENEA_TRY(boundary(batch));
    Frame& frame = *open_;
    if (!frame.stack.empty()) {
        frame.stack.pop_back();
    }
    frame.segments.push_back({top(), 0});
    return ok();
}

Result<void> FrameProfiler::endFrame(CommandBatch& batch) {
    if (!open_.has_value()) {
        return ok();
    }
    ATHENEA_TRY(boundary(batch));
    finished_.push_back(std::move(*open_));
    open_.reset();
    while (finished_.size() > kKeptFrames) {
        release(finished_.front());
        finished_.pop_front();
    }
    return ok();
}

std::optional<FrameTimes> FrameProfiler::resolve(Frame& frame, bool& pending) {
    pending = false;
    std::vector<double> segmentMs(frame.segments.size(), 0.0);
    FrameTimes out;
    out.tag = frame.tag;
    if (timestamps_) {
        out.source = "timestamps";
        if (frame.stamps < 2) {
            return std::nullopt;
        }
        rhi::QueryResultState state = rhi::QueryResultState::Pending;
        if (SLANG_FAILED(pools_[frame.pool]->getResultState(frame.first, frame.stamps, &state))) {
            return std::nullopt;
        }
        // Reset: the batch holding them has not been submitted yet.
        if (state != rhi::QueryResultState::Resolved) {
            pending = true;
            return std::nullopt;
        }
        std::vector<uint64_t> ticks(frame.stamps);
        if (SLANG_FAILED(pools_[frame.pool]->getResult(frame.first, frame.stamps, ticks.data()))) {
            return std::nullopt;
        }
        const double frequency = static_cast<double>(device_->rhi()->getInfo().timestampFrequency);
        if (frequency <= 0.0) {
            return std::nullopt;
        }
        const auto ms = [&](uint64_t a, uint64_t b) { return b > a ? double(b - a) * 1000.0 / frequency : 0.0; };
        for (size_t k = 0; k < segmentMs.size() && k + 1 < ticks.size(); ++k) {
            segmentMs[k] = ms(ticks[k], ticks[k + 1]);
        }
        out.spanMs = ms(ticks.front(), ticks.back());
    } else {
        out.source = "command buffers";
        double previousEnd = -1.0;
        double firstStart = -1.0;
        for (size_t k = 0; k < frame.segments.size(); ++k) {
            const uint64_t cb = frame.segments[k].commandBuffer;
            if (cb == 0) {
                continue;
            }
            double start = 0.0;
            double end = 0.0;
            const int said = platform::metalCommandBufferTimes(reinterpret_cast<void*>(cb), start, end);
            if (said == 0) {
                pending = true;
                return std::nullopt;
            }
            if (said < 0) {
                continue;
            }
            const double from = previousEnd >= 0.0 ? std::max(start, previousEnd) : start;
            segmentMs[k] = std::max(0.0, end - from) * 1000.0;
            if (firstStart < 0.0) {
                firstStart = start;
            }
            previousEnd = std::max(previousEnd, end);
        }
        out.spanMs = firstStart >= 0.0 ? (previousEnd - firstStart) * 1000.0 : 0.0;
    }
    for (const std::string& name : frame.names) {
        out.phases.push_back({name, 0.0});
    }
    double outside = 0.0;
    for (size_t k = 0; k < frame.segments.size(); ++k) {
        const int owner = frame.segments[k].owner;
        if (owner >= 0) {
            out.phases[static_cast<size_t>(owner)].ms += segmentMs[k];
        } else {
            outside += segmentMs[k];
        }
    }
    if (outside > 0.0) {
        out.phases.push_back({"other", outside});
    }
    return out;
}

std::optional<FrameTimes> FrameProfiler::latest() {
    std::optional<FrameTimes> newest;
    while (!finished_.empty()) {
        bool pending = false;
        std::optional<FrameTimes> times = resolve(finished_.front(), pending);
        if (pending) {
            break;
        }
        if (times.has_value()) {
            newest = std::move(times);
        }
        release(finished_.front());
        finished_.pop_front();
    }
    return newest;
}

Phase::Phase(CommandBatch& batch, const char* name) : batch_(batch) {
    // The boundary first: on Metal it ends a command buffer, and the group
    // belongs in the one that follows.
    FrameProfiler& profiler = batch_.device().profiler();
    if (profiler.enabled() && profiler.inFrame()) {
        timed_ = static_cast<bool>(profiler.push(batch_, name));
    }
    batch_.pushGroup(name);
}

Phase::~Phase() {
    batch_.popGroup();
    if (timed_) {
        (void)batch_.device().profiler().pop(batch_);
    }
}

}   // namespace athenea::gpu
