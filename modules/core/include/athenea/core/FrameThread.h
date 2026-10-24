// Copyright (c) 2026 jesus luque.
//
// THE ONE THREAD THAT CALLS THE DEVICE. An object that records commands --
// uploads, copies, dispatches, submits -- binds itself to the thread that
// opened it, and every call that would touch the device asks this first.
// Other threads may read files, decompress and fill memory the frame thread
// handed them, or queue work for it; they never reach the device. The check
// is one comparison of thread ids, on in every build: a call from the wrong
// thread is refused with an error, not left to corrupt a command encoder.
#pragma once

#include <string_view>
#include <thread>

#include "athenea/core/Result.h"

namespace athenea {

class FrameThread {
public:
    /// The calling thread becomes the frame thread. Any thread, once, before
    /// the object is shared.
    void bind() noexcept { owner_ = std::this_thread::get_id(); }

    /// Whether the calling thread is the frame thread. Any thread.
    [[nodiscard]] bool current() const noexcept { return owner_ == std::this_thread::get_id(); }

    /// Ok on the frame thread; on any other, an error naming `what`. Any thread.
    [[nodiscard]] Result<void> require(std::string_view what) const {
        if (current()) {
            return ok();
        }
        return Error::make(ErrorCode::InvalidArgument,
                           "{} called off the frame thread: only the thread that opened it may call the device",
                           what);
    }

private:
    std::thread::id owner_ = std::this_thread::get_id();
};

}   // namespace athenea
