// Copyright (c) 2026 jesus luque.
#include "athenea/bridge/CameraQueue.h"

#include <algorithm>
#include <cmath>

namespace athenea::bridge {
namespace {

bool finite(double v) {
    return std::isfinite(v);
}

}   // namespace

Result<void> checkCamera(const render::Camera& camera) {
    for (double v : camera.cameraToWorld.m) {
        if (!finite(v)) return Error(ErrorCode::InvalidArgument, "a camera matrix with a number that is not finite");
    }
    const render::Mat4& m = camera.cameraToWorld;
    if (std::abs(m.at(3, 0)) > 1e-9 || std::abs(m.at(3, 1)) > 1e-9 || std::abs(m.at(3, 2)) > 1e-9 ||
        std::abs(m.at(3, 3) - 1.0) > 1e-9) {
        return Error(ErrorCode::InvalidArgument, "a camera matrix whose last row is not 0 0 0 1");
    }
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            const double d = m.at(0, a) * m.at(0, b) + m.at(1, a) * m.at(1, b) + m.at(2, a) * m.at(2, b);
            if (std::abs(d - (a == b ? 1.0 : 0.0)) > 1e-3) {
                return Error(ErrorCode::InvalidArgument,
                             "a camera matrix whose turn is not a rotation (its axes are not unit and square to "
                             "each other within 1e-3; no scale, no shear)");
            }
        }
    }
    if (aofx::xform::determinant3(m) < 0.0) {
        return Error(ErrorCode::InvalidArgument, "a camera matrix that mirrors (determinant below 0)");
    }
    const render::Lens& lens = camera.lens;
    for (double v : {lens.focal, lens.haperture, lens.vaperture, lens.nearZ, lens.farZ, lens.windowTranslate[0],
                     lens.windowTranslate[1]}) {
        if (!finite(v)) return Error(ErrorCode::InvalidArgument, "a lens with a number that is not finite");
    }
    if (lens.focal <= 0.0 || lens.haperture <= 0.0 || lens.vaperture <= 0.0) {
        return Error(ErrorCode::InvalidArgument, "a lens's focal length and apertures are above 0");
    }
    if (lens.nearZ <= 0.0 || lens.farZ <= lens.nearZ) {
        return Error(ErrorCode::InvalidArgument, "a lens's clipping planes: 0 < near < far");
    }
    return ok();
}

Quat quatFromMatrix(const render::Mat4& m) {
    // Shepperd's method on the upper 3x3.
    const double r00 = m.at(0, 0), r11 = m.at(1, 1), r22 = m.at(2, 2);
    const double trace = r00 + r11 + r22;
    Quat q;
    if (trace > 0.0) {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        q.w = 0.25 * s;
        q.x = (m.at(2, 1) - m.at(1, 2)) / s;
        q.y = (m.at(0, 2) - m.at(2, 0)) / s;
        q.z = (m.at(1, 0) - m.at(0, 1)) / s;
    } else if (r00 > r11 && r00 > r22) {
        const double s = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
        q.w = (m.at(2, 1) - m.at(1, 2)) / s;
        q.x = 0.25 * s;
        q.y = (m.at(0, 1) + m.at(1, 0)) / s;
        q.z = (m.at(0, 2) + m.at(2, 0)) / s;
    } else if (r11 > r22) {
        const double s = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
        q.w = (m.at(0, 2) - m.at(2, 0)) / s;
        q.x = (m.at(0, 1) + m.at(1, 0)) / s;
        q.y = 0.25 * s;
        q.z = (m.at(1, 2) + m.at(2, 1)) / s;
    } else {
        const double s = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
        q.w = (m.at(1, 0) - m.at(0, 1)) / s;
        q.x = (m.at(0, 2) + m.at(2, 0)) / s;
        q.y = (m.at(1, 2) + m.at(2, 1)) / s;
        q.z = 0.25 * s;
    }
    const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return {q.x / n, q.y / n, q.z / n, q.w / n};
}

render::Mat4 matrixFromQuat(const Quat& q, const render::Vec3& p) {
    const double x = q.x, y = q.y, z = q.z, w = q.w;
    render::Mat4 m;
    m.at(0, 0) = 1 - 2 * (y * y + z * z);
    m.at(0, 1) = 2 * (x * y - z * w);
    m.at(0, 2) = 2 * (x * z + y * w);
    m.at(1, 0) = 2 * (x * y + z * w);
    m.at(1, 1) = 1 - 2 * (x * x + z * z);
    m.at(1, 2) = 2 * (y * z - x * w);
    m.at(2, 0) = 2 * (x * z - y * w);
    m.at(2, 1) = 2 * (y * z + x * w);
    m.at(2, 2) = 1 - 2 * (x * x + y * y);
    m.at(0, 3) = p.x;
    m.at(1, 3) = p.y;
    m.at(2, 3) = p.z;
    return m;
}

render::Camera interpolateCamera(const render::Camera& a, const render::Camera& b, double t) {
    t = std::clamp(t, 0.0, 1.0);
    const auto mix = [t](double u, double v) { return u + (v - u) * t; };
    Quat qa = quatFromMatrix(a.cameraToWorld);
    const Quat qb = quatFromMatrix(b.cameraToWorld);
    double d = qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w;
    if (d < 0.0) {   // the shorter arc
        qa = {-qa.x, -qa.y, -qa.z, -qa.w};
        d = -d;
    }
    double wa = 1.0 - t;
    double wb = t;
    if (d < 0.9995) {
        const double angle = std::acos(std::clamp(d, -1.0, 1.0));
        const double s = std::sin(angle);
        wa = std::sin((1.0 - t) * angle) / s;
        wb = std::sin(t * angle) / s;
    }
    Quat q{wa * qa.x + wb * qb.x, wa * qa.y + wb * qb.y, wa * qa.z + wb * qb.z, wa * qa.w + wb * qb.w};
    const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    q = {q.x / n, q.y / n, q.z / n, q.w / n};
    const render::Vec3 pa = a.cameraToWorld.translation();
    const render::Vec3 pb = b.cameraToWorld.translation();
    render::Camera out = a;
    out.cameraToWorld = matrixFromQuat(q, {mix(pa.x, pb.x), mix(pa.y, pb.y), mix(pa.z, pb.z)});
    out.lens.focal = mix(a.lens.focal, b.lens.focal);
    out.lens.haperture = mix(a.lens.haperture, b.lens.haperture);
    out.lens.vaperture = mix(a.lens.vaperture, b.lens.vaperture);
    out.lens.nearZ = mix(a.lens.nearZ, b.lens.nearZ);
    out.lens.farZ = mix(a.lens.farZ, b.lens.farZ);
    out.lens.windowTranslate[0] = mix(a.lens.windowTranslate[0], b.lens.windowTranslate[0]);
    out.lens.windowTranslate[1] = mix(a.lens.windowTranslate[1], b.lens.windowTranslate[1]);
    out.lens.focusDistance = mix(a.lens.focusDistance, b.lens.focusDistance);
    return out;
}

Result<void> CameraQueue::push(const CameraSample& sample) {
    if (Result<void> checked = checkCamera(sample.camera); !checked) {
        std::lock_guard lock(mutex_);
        ++stats_.refused;
        return checked;
    }
    std::lock_guard lock(mutex_);
    auto it = std::lower_bound(samples_.begin(), samples_.end(), sample.ptpNs,
                               [](const CameraSample& s, uint64_t t) { return s.ptpNs < t; });
    if (it != samples_.end() && it->ptpNs == sample.ptpNs) {
        *it = sample;
        ++stats_.replaced;
    } else {
        samples_.insert(it, sample);
    }
    ++stats_.queued;
    while (samples_.size() > kCapacity) {
        samples_.pop_front();
        ++stats_.dropped;
    }
    return ok();
}

Result<std::optional<CameraAt>> CameraQueue::at(uint64_t instantNs) {
    std::lock_guard lock(mutex_);
    if (!frame_.has_value()) {
        frame_.emplace();   // binds the calling thread
    }
    ATHENEA_TRY(frame_->require("CameraQueue::at"));
    if (samples_.empty()) return std::optional<CameraAt>{};
    // Keep the last report at or before the instant, and everything after.
    while (samples_.size() > 1 && samples_[1].ptpNs <= instantNs) {
        samples_.pop_front();
    }
    const CameraSample& first = samples_.front();
    CameraAt out;
    if (instantNs <= first.ptpNs || samples_.size() == 1) {
        out.camera = first.camera;
        out.samples = 1;
        out.fromNs = out.toNs = first.ptpNs;
        return std::optional<CameraAt>(out);
    }
    const CameraSample& next = samples_[1];
    const double t = static_cast<double>(instantNs - first.ptpNs) / static_cast<double>(next.ptpNs - first.ptpNs);
    out.camera = interpolateCamera(first.camera, next.camera, t);
    out.samples = 2;
    out.fromNs = first.ptpNs;
    out.toNs = next.ptpNs;
    return std::optional<CameraAt>(out);
}

void CameraQueue::countRefused() {
    std::lock_guard lock(mutex_);
    ++stats_.refused;
}

size_t CameraQueue::size() const {
    std::lock_guard lock(mutex_);
    return samples_.size();
}

CameraQueue::Stats CameraQueue::stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}

}   // namespace athenea::bridge
