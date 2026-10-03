// Copyright (c) 2026 jesus luque.
//
// Setting shaders/athenea/splat/frame.slang's FrameParams by name, shared by every
// renderer that speaks it (the tile rasteriser and the reference).
#pragma once

#include <array>
#include <cstdint>

#include <slang-rhi/shader-cursor.h>

#include "athenea/render/Camera.h"
#include "athenea/render/Points.h"
#include "athenea/render/TileRasterizer.h"

namespace athenea::render {

inline uint32_t bitsFor(uint32_t values) {
    uint32_t bits = 1;
    while (bits < 32 && (uint64_t{1} << bits) < values) {
        ++bits;
    }
    return bits;
}

struct FrameCommon {
    uint32_t width, height, tilesX, tilesY;
    const Projection* projection;
    const RenderSettings* settings;
};

inline void setFrame(rhi::ShaderCursor cursor, const FrameCommon& f) {
    rhi::ShaderCursor p = cursor["params"];
    p["width"].setData(f.width);
    p["height"].setData(f.height);
    p["tilesX"].setData(f.tilesX);
    p["tilesY"].setData(f.tilesY);
    p["focalX"].setData(static_cast<float>(f.projection->focalX));
    p["focalY"].setData(static_cast<float>(f.projection->focalY));
    p["centreX"].setData(static_cast<float>(f.projection->centreX));
    p["centreY"].setData(static_cast<float>(f.projection->centreY));
    p["nearZ"].setData(static_cast<float>(f.projection->nearZ));
    p["farZ"].setData(static_cast<float>(f.projection->farZ));
    p["orthographic"].setData(uint32_t{f.projection->orthographic ? 1u : 0u});
    p["antialias"].setData(uint32_t{f.settings->antialias ? 1u : 0u});
    // The lens, where the camera authored one: the projection carries it for
    // the path tracer, and the raster spreads the covariance by it instead of
    // sampling a disk.
    p["lensRadius"].setData(static_cast<float>(f.projection->orthographic ? 0.0 : f.projection->lensRadius));
    p["focusDistance"].setData(static_cast<float>(f.projection->focusDistance));
    p["shLimit"].setData(f.settings->maxShDegree);
    p["depthMode"].setData(uint32_t{f.settings->depth == RenderSettings::Depth::Mean ? 0u : 1u});
    p["depthThreshold"].setData(f.settings->depthThreshold);
    p["bgR"].setData(f.settings->background[0]);
    p["bgG"].setData(f.settings->background[1]);
    p["bgB"].setData(f.settings->background[2]);
    p["bgA"].setData(f.settings->background[3]);
    // Every splat is blended in linear light; what space a cloud's colours
    // are in is the cloud's (setCloudSpace), set at its own projection.
    p["linearCloud"].setData(uint32_t{0});
    p["hasUnder"].setData(uint32_t{0});
    // No sky prepared unless the dispatch that has one says so after this.
    p["envLights"].setData(uint32_t{0});
    p["envBaseSide"].setData(uint32_t{1});
    p["overrideCount"].setData(uint32_t{0});
    // No transfer unless the cloud being projected carries one.
    p["transferWords"].setData(uint32_t{0});
    p["transferCount"].setData(uint32_t{0});
    p["transferIndirect"].setData(uint32_t{0});
    // Nothing bends unless the cloud being projected says what by.
    p["ior"].setData(0.0F);
    // No matte unless the dispatch that wants one says so after this.
    p["cryptoMode"].setData(uint32_t{0});
    p["hasUnderCrypto"].setData(uint32_t{0});
    p["tileBits"].setData(bitsFor(f.tilesX * f.tilesY));
}

inline void setObject(rhi::ShaderCursor cursor, const Mat4& objectToView, const Vec3& eyeObject) {
    rhi::ShaderCursor p = cursor["params"];
    static constexpr const char* kNames[12] = {"m00", "m01", "m02", "m03", "m10", "m11",
                                              "m12", "m13", "m20", "m21", "m22", "m23"};
    const std::array<float, 12> rows = objectToView.rows3x4();
    for (size_t k = 0; k < 12; ++k) {
        p[kNames[k]].setData(rows[k]);
    }
    p["eyeX"].setData(static_cast<float>(eyeObject.x));
    p["eyeY"].setData(static_cast<float>(eyeObject.y));
    p["eyeZ"].setData(static_cast<float>(eyeObject.z));
}


/// The space a cloud's colours are in (`scene::GpuSplats::linear`), set at
/// every projection dispatch: the projection makes a capture's sRGB light a
/// splat at a time, and takes a cloud that holds light as it is.
inline void setCloudSpace(rhi::ShaderCursor cursor, const scene::GpuSplats& splats) {
    cursor["params"]["linearCloud"].setData(uint32_t{splats.linear ? 1u : 0u});
}

/// What moves during the shutter, per instance. Called at every projection
/// dispatch, including a still one: a constant buffer reused between
/// dispatches would otherwise keep the last instance's flag.
inline void setMotion(rhi::ShaderCursor cursor, const SplatInstance& instance, float motionMax) {
    rhi::ShaderCursor p = cursor["params"];
    uint32_t flags = 0;
    if (instance.motion != nullptr && instance.motionScale != 0.0F) {
        flags |= 1u;
    }
    for (const float v : instance.viewStep) {
        if (v != 0.0F) {
            flags |= 2u;
            break;
        }
    }
    p["motion"].setData(flags);
    p["motionScale"].setData(instance.motionScale);
    p["motionMax"].setData(motionMax);
    static constexpr const char* kNames[12] = {"s00", "s01", "s02", "s03", "s10", "s11",
                                              "s12", "s13", "s20", "s21", "s22", "s23"};
    for (size_t k = 0; k < 12; ++k) {
        p[kNames[k]].setData(instance.viewStep[k]);
    }
}

/// EditParams (shaders/athenea/common/edit.slang) under `e`: the struct nested as
/// `edit` in FrameParams and RtParams.
inline void setEdit(rhi::ShaderCursor e, const SplatEdit& edit) {
    e["active"].setData(uint32_t{edit.active ? 1u : 0u});
    e["shape"].setData(static_cast<uint32_t>(edit.shape));
    e["mode"].setData(static_cast<uint32_t>(edit.mode));
    e["invert"].setData(uint32_t{edit.invert ? 1u : 0u});
    e["centreX"].setData(edit.centre[0]);
    e["centreY"].setData(edit.centre[1]);
    e["centreZ"].setData(edit.centre[2]);
    e["sizeX"].setData(edit.size[0]);
    e["sizeY"].setData(edit.size[1]);
    e["sizeZ"].setData(edit.size[2]);
    e["tintR"].setData(edit.tint[0]);
    e["tintG"].setData(edit.tint[1]);
    e["tintB"].setData(edit.tint[2]);
    e["saturation"].setData(edit.saturation);
    e["brightness"].setData(edit.brightness);
    e["opacity"].setData(edit.opacity);
    e["minOpacity"].setData(edit.minOpacity);
    e["maxScale"].setData(edit.maxScale);
}

/// PointParams (shaders/athenea/points/point_frame.slang) under `cursor`, which is
/// the root's "params" for the raster and EDL passes and "points" for the
/// disc projection.
inline void setPointParams(rhi::ShaderCursor p, const Projection& projection, uint32_t width,
                           uint32_t height, const Mat4& objectToView, const PointStyle& style,
                           uint32_t count) {
    p["width"].setData(width);
    p["height"].setData(height);
    p["focalX"].setData(static_cast<float>(projection.focalX));
    p["focalY"].setData(static_cast<float>(projection.focalY));
    p["centreX"].setData(static_cast<float>(projection.centreX));
    p["centreY"].setData(static_cast<float>(projection.centreY));
    p["nearZ"].setData(static_cast<float>(projection.nearZ));
    p["farZ"].setData(static_cast<float>(projection.farZ));
    p["orthographic"].setData(uint32_t{projection.orthographic ? 1u : 0u});
    p["sizeMode"].setData(uint32_t{style.sizeMode == PointStyle::Size::Pixels ? 1u : 0u});
    p["size"].setData(style.size);
    p["count"].setData(count);
    static constexpr const char* kNames[12] = {"m00", "m01", "m02", "m03", "m10", "m11",
                                              "m12", "m13", "m20", "m21", "m22", "m23"};
    const std::array<float, 12> rows = objectToView.rows3x4();
    for (size_t k = 0; k < 12; ++k) {
        p[kNames[k]].setData(rows[k]);
    }
    p["colourMode"].setData(uint32_t{style.constantColour ? 1u : 0u});
    p["colourR"].setData(style.colour[0]);
    p["colourG"].setData(style.colour[1]);
    p["colourB"].setData(style.colour[2]);
    p["edlStrength"].setData(style.edlStrength);
    p["edlRadius"].setData(style.edlRadius);
    p["surface"].setData(uint32_t{style.surfaceDepthOffset > 0.0F ? 1u : 0u});
    p["depthOffset"].setData(style.surfaceDepthOffset);
}

}   // namespace athenea::render
