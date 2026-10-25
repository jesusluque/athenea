// Copyright (c) 2026 jesus luque.
//
// CAMERAS WITH THEIR INSTANTS (camera_set). A tracked camera reports where it
// was at a PTP instant; a frame stands for an instant of its own (the frame
// clock's, the one that also picks a live stream's epoch), so the frame is
// drawn from the pose at that instant: between the two reports that bracket
// it, the position along the line between them and the turn along the arc
// (quaternions, slerp), the lens's numbers in proportion. Before the first
// report the first is held, after the last the last: never extrapolated.
//
// What is checked before a camera is queued (`check`): every number finite;
// a matrix whose upper 3x3 is a rotation (orthonormal, right-handed, to 1e-3)
// and whose last row is 0 0 0 1; a quaternion of length 1 (to 1e-3); a focal
// length, apertures and clipping planes that make a frustum; square pixels
// (fx = fy to 1e-3) for intrinsics in pixels. A camera that fails is refused
// and counted, never queued.
//
// THREADS. push and check: any thread (the MCP server's). at: the frame
// thread, which drops what it no longer needs. size, stats: any thread.
#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

#include "athenea/core/FrameThread.h"
#include "athenea/core/Result.h"
#include "athenea/render/Camera.h"

namespace athenea::bridge {

struct CameraSample {
    uint64_t       ptpNs = 0;
    render::Camera camera;
};

/// What a frame was drawn from.
struct CameraAt {
    render::Camera camera;
    uint32_t       samples = 0;   ///< 1 held (before the first, after the last), 2 interpolated
    uint64_t       fromNs = 0, toNs = 0;
};

/// Whether `camera` makes a frustum (above); InvalidArgument naming what does not.
[[nodiscard]] Result<void> checkCamera(const render::Camera& camera);

/// The camera a fraction `t` (0..1) of the way from `a` to `b`.
[[nodiscard]] render::Camera interpolateCamera(const render::Camera& a, const render::Camera& b, double t);

/// A rotation matrix's quaternion (x, y, z, w) and back, both unit.
struct Quat {
    double x = 0, y = 0, z = 0, w = 1;
};
[[nodiscard]] Quat quatFromMatrix(const render::Mat4& m);
[[nodiscard]] render::Mat4 matrixFromQuat(const Quat& q, const render::Vec3& position);

class CameraQueue {
public:
    static constexpr size_t kCapacity = 1024;

    struct Stats {
        uint64_t queued = 0;     ///< accepted
        uint64_t refused = 0;    ///< failed the check
        uint64_t replaced = 0;   ///< a report at an instant already queued
        uint64_t dropped = 0;    ///< pushed out past the capacity
    };

    /// ANY THREAD: checks and queues; refused cameras are counted.
    Result<void> push(const CameraSample& sample);
    /// FRAME THREAD -- the first thread to call it; any other is refused
    /// (InvalidArgument): the pose at `instantNs`, nothing when no camera
    /// was ever queued. Reports older than the one before the instant go.
    [[nodiscard]] Result<std::optional<CameraAt>> at(uint64_t instantNs);
    /// Counts a refusal made before push could see the camera (a malformed
    /// argument). Any thread.
    void countRefused();

    [[nodiscard]] size_t size() const;
    [[nodiscard]] Stats stats() const;

private:
    mutable std::mutex       mutex_;
    std::optional<FrameThread> frame_;   // bound by the first `at`
    std::deque<CameraSample> samples_;   // by instant
    Stats                    stats_;
};

}   // namespace athenea::bridge
