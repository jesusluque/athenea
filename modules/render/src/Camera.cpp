// Copyright (c) 2026 jesus luque.
#include "athenea/render/Camera.h"

#include <algorithm>
#include <cmath>

namespace athenea::render {

Camera Camera::lookingAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    Camera camera;
    camera.cameraToWorld = aofx::xform::lookAt(eye, target, up);
    // THE NEAR PLANE IS THE SUBJECT'S, NOT A NUMBER.
    //
    // A tenth of a unit is a near plane for a scene measured in metres and a
    // guillotine for anything smaller: a chess pawn 66 mm tall, looked at
    // from 130 mm, came back with the top of its glass ball sliced off and
    // the collar showing through the cut. What a camera built from an eye and
    // a target knows is how far the two are, so the plane is a thousandth of
    // that -- the same rule the framing camera already used -- and a caller
    // who wants another sets `lens.nearZ` after this.
    const Vec3 away{eye.x - target.x, eye.y - target.y, eye.z - target.z};
    const double distance = std::sqrt(away.x * away.x + away.y * away.y + away.z * away.z);
    camera.lens.nearZ = std::max(distance * 1e-3, 1e-5);
    return camera;
}

Mat4 viewFromCamera(const Mat4& cameraToWorld) {
    return aofx::xform::scaling(Vec3{1.0, 1.0, -1.0}) *
           aofx::xform::viewFromCameraWorld(cameraToWorld);
}

Projection projectionFor(const Camera& camera, uint32_t width, uint32_t height) {
    Projection out;
    out.worldToView = viewFromCamera(camera.cameraToWorld);
    if (std::abs(camera.lens.windowRoll) > 1e-12) {
        // A turn of the picture about the view's own z, after the flip.
        out.worldToView = aofx::xform::rotationZ(camera.lens.windowRoll) * out.worldToView;
    }
    const double w = static_cast<double>(width);
    const double h = static_cast<double>(height);
    // W * focal / haperture: pixels per unit of x/z, and pixels per world unit
    // for an orthographic camera -- Nuke's rule, why focal is an ortho zoom.
    const double focal = w * std::max(camera.lens.focal, 1e-9) /
                         std::max(camera.lens.haperture, 1e-9);
    out.focalX = focal * camera.lens.windowScale[0];
    out.focalY = focal * camera.lens.windowScale[1];
    out.centreX = w * 0.5 + camera.lens.windowTranslate[0] * w * 0.5;
    out.centreY = h * 0.5 + camera.lens.windowTranslate[1] * w * 0.5;
    out.nearZ = camera.lens.nearZ;
    out.farZ = camera.lens.farZ;
    out.orthographic = camera.lens.projection == Lens::Projection::Orthographic;
    out.eyeWorld = camera.cameraToWorld.translation();
    out.exposure = camera.lens.exposure;
    // The focal length is in millimetres and the lens radius in scene units,
    // so what a unit is decides the size of the diaphragm: 0.01 m a unit --
    // USD's default, and GfCamera's hard-coded FOCAL_LENGTH_UNIT -- makes a
    // 50 mm lens 5 units, and a stage in metres makes it 0.05.
    const double perUnit = std::max(camera.lens.metersPerUnit, 1e-9);
    out.lensRadius = camera.lens.fStop > 0.0 ? camera.lens.focal * 0.001 / perUnit / (2.0 * camera.lens.fStop) : 0.0;
    out.focusDistance = camera.lens.focusDistance;
    out.distortionK1 = camera.lens.distortionK1;
    out.distortionK2 = camera.lens.distortionK2;
    return out;
}

}   // namespace athenea::render
