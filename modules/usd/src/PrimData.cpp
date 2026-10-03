// Copyright (c) 2026 jesus luque.
#include "athenea/usd/PrimData.h"

#include <algorithm>
#include <span>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec2h.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/gf/vec4h.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>

namespace athenea::usd {
namespace {

template <typename Element>
scene::FloatStream bytesOf(const pxr::VtArray<Element>& array, bool half, size_t* elements) {
    if (elements != nullptr) {
        *elements = array.size();
    }
    return {std::as_bytes(std::span<const Element>(array.cdata(), array.size())), half};
}

}   // namespace

scene::FloatStream streamOf(const pxr::VtValue& value, size_t* elements) {
    using namespace pxr;
    if (elements != nullptr) {
        *elements = 0;
    }
    if (value.IsHolding<VtVec3fArray>()) return bytesOf(value.UncheckedGet<VtVec3fArray>(), false, elements);
    if (value.IsHolding<VtVec3hArray>()) return bytesOf(value.UncheckedGet<VtVec3hArray>(), true, elements);
    if (value.IsHolding<VtQuatfArray>()) return bytesOf(value.UncheckedGet<VtQuatfArray>(), false, elements);
    if (value.IsHolding<VtQuathArray>()) return bytesOf(value.UncheckedGet<VtQuathArray>(), true, elements);
    if (value.IsHolding<VtFloatArray>()) return bytesOf(value.UncheckedGet<VtFloatArray>(), false, elements);
    if (value.IsHolding<VtHalfArray>()) return bytesOf(value.UncheckedGet<VtHalfArray>(), true, elements);
    if (value.IsHolding<VtVec4fArray>()) return bytesOf(value.UncheckedGet<VtVec4fArray>(), false, elements);
    if (value.IsHolding<VtVec4hArray>()) return bytesOf(value.UncheckedGet<VtVec4hArray>(), true, elements);
    const auto doubles = [&](scene::FloatStream stream) {
        stream.half = false;
        stream.isDouble = true;
        return stream;
    };
    if (value.IsHolding<VtVec3dArray>()) return doubles(bytesOf(value.UncheckedGet<VtVec3dArray>(), false, elements));
    if (value.IsHolding<VtQuatdArray>()) return doubles(bytesOf(value.UncheckedGet<VtQuatdArray>(), false, elements));
    if (value.IsHolding<VtMatrix4dArray>()) {
        return doubles(bytesOf(value.UncheckedGet<VtMatrix4dArray>(), false, elements));
    }
    if (value.IsHolding<VtMatrix4fArray>()) return bytesOf(value.UncheckedGet<VtMatrix4fArray>(), false, elements);
    if (value.IsHolding<GfVec3f>()) {
        if (elements != nullptr) {
            *elements = 1;
        }
        const GfVec3f& v = value.UncheckedGet<GfVec3f>();
        return {std::as_bytes(std::span<const float>(v.data(), 3)), false};
    }
    return {};
}

scene::FloatStream primvarStreamOf(const pxr::VtValue& value, uint32_t* components) {
    using namespace pxr;
    *components = 0;
    const auto array = [&](uint32_t n, scene::FloatStream s) {
        *components = s.empty() ? 0 : n;
        return s;
    };
    const auto scalar = [&](uint32_t n, const auto& v, bool isDouble) {
        *components = n;
        scene::FloatStream s{std::as_bytes(std::span(&v, 1)), false, isDouble};
        return s;
    };
    const auto doubles = [](scene::FloatStream s) {
        s.isDouble = true;
        return s;
    };
    if (value.IsHolding<VtFloatArray>()) return array(1, bytesOf(value.UncheckedGet<VtFloatArray>(), false, nullptr));
    if (value.IsHolding<VtVec2fArray>()) return array(2, bytesOf(value.UncheckedGet<VtVec2fArray>(), false, nullptr));
    if (value.IsHolding<VtVec3fArray>()) return array(3, bytesOf(value.UncheckedGet<VtVec3fArray>(), false, nullptr));
    if (value.IsHolding<VtVec4fArray>()) return array(4, bytesOf(value.UncheckedGet<VtVec4fArray>(), false, nullptr));
    if (value.IsHolding<VtHalfArray>()) return array(1, bytesOf(value.UncheckedGet<VtHalfArray>(), true, nullptr));
    if (value.IsHolding<VtVec2hArray>()) return array(2, bytesOf(value.UncheckedGet<VtVec2hArray>(), true, nullptr));
    if (value.IsHolding<VtVec3hArray>()) return array(3, bytesOf(value.UncheckedGet<VtVec3hArray>(), true, nullptr));
    if (value.IsHolding<VtVec4hArray>()) return array(4, bytesOf(value.UncheckedGet<VtVec4hArray>(), true, nullptr));
    if (value.IsHolding<VtDoubleArray>()) {
        return array(1, doubles(bytesOf(value.UncheckedGet<VtDoubleArray>(), false, nullptr)));
    }
    if (value.IsHolding<VtVec2dArray>()) {
        return array(2, doubles(bytesOf(value.UncheckedGet<VtVec2dArray>(), false, nullptr)));
    }
    if (value.IsHolding<VtVec3dArray>()) {
        return array(3, doubles(bytesOf(value.UncheckedGet<VtVec3dArray>(), false, nullptr)));
    }
    if (value.IsHolding<VtVec4dArray>()) {
        return array(4, doubles(bytesOf(value.UncheckedGet<VtVec4dArray>(), false, nullptr)));
    }
    if (value.IsHolding<float>()) return scalar(1, value.UncheckedGet<float>(), false);
    if (value.IsHolding<double>()) return scalar(1, value.UncheckedGet<double>(), true);
    if (value.IsHolding<GfVec2f>()) return scalar(2, value.UncheckedGet<GfVec2f>(), false);
    if (value.IsHolding<GfVec3f>()) return scalar(3, value.UncheckedGet<GfVec3f>(), false);
    if (value.IsHolding<GfVec4f>()) return scalar(4, value.UncheckedGet<GfVec4f>(), false);
    if (value.IsHolding<GfVec3d>()) return scalar(3, value.UncheckedGet<GfVec3d>(), true);
    return {};
}

scene::SplatStreams splatStreams(const ParticleFieldArrays& a, std::string source) {
    scene::SplatStreams s;
    s.source = std::move(source);
    size_t count = 0;
    s.positions = streamOf(a.positions, &count);
    s.count = static_cast<uint32_t>(count);
    s.rotations = streamOf(a.orientations);
    s.scales = streamOf(a.scales);
    s.opacities = streamOf(a.opacities);
    const int degree = std::clamp(a.shDegree, 0, 3);
    s.coefficients = static_cast<uint32_t>((degree + 1) * (degree + 1));
    s.sh = streamOf(a.shCoefficients);
    if (s.sh.values() < uint64_t{s.count} * s.coefficients * 3) {
        s.sh = {};   // fewer coefficients than the degree says: none, as before
    }
    // BLENDER'S LAYOUT: DC and opacity in one float4, and a plane a basis
    // function. As many planes as a whole degree holds (3, 8 or 15), each
    // the cloud's length; anything else is no harmonics, as above.
    s.radianceBase = streamOf(a.radianceBase);
    if (s.radianceBase.values() < uint64_t{s.count} * 4) {
        s.radianceBase = {};
    }
    if (!a.shPlanes.empty()) {
        std::vector<scene::FloatStream> planes;
        for (const pxr::VtValue& plane : a.shPlanes) {
            planes.push_back(streamOf(plane));
            if (planes.back().values() < uint64_t{s.count} * 3) {
                planes.pop_back();
                break;
            }
        }
        const size_t whole = planes.size() >= 15 ? 15 : planes.size() >= 8 ? 8 : planes.size() >= 3 ? 3 : 0;
        planes.resize(whole);
        s.sh = {};
        s.coefficients = static_cast<uint32_t>(whole + 1);
        s.shPlanes = std::move(planes);
    }
    // One value a particle, or nothing: a shorter array is no array, as the
    // harmonics above are.
    s.metallic = streamOf(a.metallic);
    s.roughness = streamOf(a.roughness);
    s.transmission = streamOf(a.transmission);
    if (s.metallic.values() < s.count) {
        s.metallic = {};
    }
    if (s.roughness.values() < s.count) {
        s.roughness = {};
    }
    if (s.transmission.values() < s.count) {
        s.transmission = {};
    }
    // The Cryptomatte ids: int32 words read as the bits they are, never as
    // numbers, so they travel as a stream of 32-bit values like any other.
    if (a.cryptoObject.IsHolding<pxr::VtIntArray>()) {
        const pxr::VtIntArray& ids = a.cryptoObject.UncheckedGet<pxr::VtIntArray>();
        if (ids.size() >= s.count && s.count > 0) {
            s.cryptoObject = {std::as_bytes(std::span<const int>(ids.cdata(), ids.size())), false, false};
        }
    }
    // The transfer, as floats: nine a gaussian and twenty-seven more where
    // the indirect half is there. A shorter array is no array, as everywhere
    // else here.
    s.transferDirect = streamOf(a.transferDirect);
    s.transferIndirect = streamOf(a.transferIndirect);
    if (s.transferDirect.values() < uint64_t{s.count} * 9) {
        s.transferDirect = {};
        s.transferIndirect = {};
    }
    if (s.transferIndirect.values() < uint64_t{s.count} * 27) {
        s.transferIndirect = {};
    }
    // Which gaussians are thin walls: ints, like the ids.
    if (a.thinWalled.IsHolding<pxr::VtIntArray>()) {
        const pxr::VtIntArray& thin = a.thinWalled.UncheckedGet<pxr::VtIntArray>();
        if (thin.size() >= s.count && s.count > 0) {
            s.thinWalled = {std::as_bytes(std::span<const int>(thin.cdata(), thin.size())), false, false};
        }
    }
    s.linear = a.linear;
    // The shading normal: three floats a gaussian, or nothing.
    s.normals = streamOf(a.normals);
    if (s.normals.values() < uint64_t{s.count} * 3) {
        s.normals = {};
    }
    // The radiance it gives off: three floats a gaussian, or nothing.
    s.emission = streamOf(a.emission);
    if (s.emission.values() < uint64_t{s.count} * 3) {
        s.emission = {};
    }
    // The layers over the base: one value or one colour a gaussian each, or
    // nothing.
    const auto wholeOr = [&s](const pxr::VtValue& value, uint32_t per) {
        scene::FloatStream stream = streamOf(value);
        return stream.values() < uint64_t{s.count} * per ? scene::FloatStream{} : stream;
    };
    s.specularWeight = wholeOr(a.specularWeight, 1);
    s.specularColour = wholeOr(a.specularColour, 3);
    s.specularIor = wholeOr(a.specularIor, 1);
    s.coatWeight = wholeOr(a.coatWeight, 1);
    s.coatRoughness = wholeOr(a.coatRoughness, 1);
    s.coatIor = wholeOr(a.coatIor, 1);
    s.sheenColour = wholeOr(a.sheenColour, 3);
    s.sheenRoughness = wholeOr(a.sheenRoughness, 1);
    // Which ways out are open: bits, like the ids, and only beside a transfer.
    if (!s.transferDirect.empty() && a.shadowBits.IsHolding<pxr::VtIntArray>()) {
        const pxr::VtIntArray& bits = a.shadowBits.UncheckedGet<pxr::VtIntArray>();
        if (bits.size() >= size_t{s.count} * 2 && s.count > 0) {
            s.shadowBits = {std::as_bytes(std::span<const int>(bits.cdata(), bits.size())), false, false};
        }
    }
    return s;
}

scene::PointStreams pointStreams(const PointsArrays& a, std::string source) {
    scene::PointStreams s;
    s.source = std::move(source);
    size_t count = 0;
    s.positions = streamOf(a.positions, &count);
    s.count = static_cast<uint32_t>(count);
    s.colours = streamOf(a.colours);
    return s;
}

}   // namespace athenea::usd
