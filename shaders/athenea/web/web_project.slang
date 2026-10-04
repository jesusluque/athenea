// Copyright (c) 2026 jesus luque.
//
// THE WEB'S PROJECTION, PRESET T1 (proposal 072, stage E1): a captured cloud
// with its harmonics, nothing relit, under the specification's default limits
// (8 storage buffers a stage, 16 KiB of group memory). The same arithmetic as
// splat_project.slang's projectSplat<0> with every feature 072 lists as off
// left out -- the edit, the shutter, the lens, the light, the matte -- since
// a WGSL binding a kernel names counts against the limit whether a flag reads
// it or not.
//
// The arenas of 072 (section 2.1b) for this preset:
//
//   B0 geometry   positions, shape, sh       read, a splat's row each
//   W0 records    proj                       written, a slot each
//   W1 frame      visible | tilesTouched | depthKeys | tileRects
//                 one buffer, each stream at its word offset in `arena`
//
// Five storage bindings and two uniforms. W1 is one allocation so the kernels
// after this one (compact, gather, emit), which read its streams under their
// own names, bind ranges of it: offsets are multiples of 64 words, the web's
// 256-byte storage offset alignment.
//
// Until the arena change reaches splat_project.slang (after the TX transfer
// settles there), this file repeats its geometry; then the two become one
// kernel with the features as specialisation constants (docs/decisions.md,
// "WebGPU").
module web_project;
import athenea.common.packing;
import athenea.common.sh;
import athenea.common.color;
import athenea.splat.frame;

/// The highest degree of harmonics this pipeline evaluates: a WGSL `override`
/// (a Metal function constant), set when the pipeline is made, so a device
/// that is short of time drops the view-dependent colour without a second
/// module. Lowered further per frame by `params.shLimit`.
[SpecializationConstant]
const uint kShDegree = 3;

/// Where each stream of W1 starts, in words.
public struct WebArena {
    public uint visibleAt;
    public uint touchedAt;
    public uint keysAt;
    public uint rectsAt;
};

StructuredBuffer<float4>       positions;
StructuredBuffer<uint>         shape;
StructuredBuffer<uint>         sh;
RWStructuredBuffer<ProjRecord> proj;
RWStructuredBuffer<uint>       frameWords;
ConstantBuffer<FrameParams>    params;
ConstantBuffer<WebArena>       arena;

void culled(uint slot, uint reason) {
    frameWords[arena.touchedAt + slot] = 0;
    frameWords[arena.visibleAt + slot] = 0;
    frameWords[arena.keysAt + slot] = kCulledKey | reason;
    const uint r = arena.rectsAt + slot * 4;
    frameWords[r] = 1;
    frameWords[r + 1] = 1;
    frameWords[r + 2] = 0;
    frameWords[r + 3] = 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void webProject(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    const float4 posOpacity = positions[i];
    const uint row = i * 4;
    const float3 scale = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));

    const float3 t = toView(params, posOpacity.xyz);
    if (!(t.z > params.nearZ) || t.z > params.farZ) {
        culled(slot, kCullDepth);
        return;
    }

    // 3D covariance in the cloud's space: R S S^T R^T.
    const float4 q = decodeQuaternion(shape[row]);
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float3x3 r = float3x3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
    const float3x3 m = float3x3(r[0] * scale, r[1] * scale, r[2] * scale);

    // T = J W, linearised at a point clamped to 1.3x the frustum (projectSplat).
    const bool ortho = params.orthographic != 0;
    float3 jx, jy;
    float2 centre;
    if (ortho) {
        jx = float3(params.focalX, 0.0, 0.0);
        jy = float3(0.0, params.focalY, 0.0);
        centre = float2(params.focalX * t.x + params.centreX, params.focalY * t.y + params.centreY);
    } else {
        const float invZ = 1.0 / t.z;
        centre = float2(params.focalX * t.x * invZ + params.centreX, params.focalY * t.y * invZ + params.centreY);
        const float limLoX = (-0.15 * float(params.width) - params.centreX) / params.focalX;
        const float limHiX = (1.15 * float(params.width) - params.centreX) / params.focalX;
        const float limLoY = (-0.15 * float(params.height) - params.centreY) / params.focalY;
        const float limHiY = (1.15 * float(params.height) - params.centreY) / params.focalY;
        const float tx = clamp(t.x * invZ, limLoX, limHiX) * t.z;
        const float ty = clamp(t.y * invZ, limLoY, limHiY) * t.z;
        jx = float3(params.focalX * invZ, 0.0, -params.focalX * tx * invZ * invZ);
        jy = float3(0.0, params.focalY * invZ, -params.focalY * ty * invZ * invZ);
    }
    const float3x3 wv = viewLinear(params);
    const float3 ax = mul(mul(jx, wv), m);
    const float3 ay = mul(mul(jy, wv), m);
    float a = dot(ax, ax);
    const float b = dot(ax, ay);
    float c = dot(ay, ay);
    const float detBefore = a * c - b * b;

    // A third of a pixel of dilation, and the Mip-Splatting filter paying its
    // energy back as opacity where `antialias` says so.
    a += 0.3;
    c += 0.3;
    const float det = a * c - b * b;
    if (!(det > 1e-12)) {
        culled(slot, kCullDegenerate);
        return;
    }
    const float paid = params.antialias != 0 ? sqrt(max(detBefore, 0.0) / det) : 1.0;
    const float alpha = paidOpacity(posOpacity.w, paid);
    if (!(alpha >= 1.0 / 255.0)) {
        culled(slot, kCullFaint);
        return;
    }
    const float3 conic = float3(c / det, -b / det, a / det);
    const float cutoff = cutoffPower(alpha);
    const float2 extent = footprintExtent(conic, cutoff);
    uint4 rect;
    tileRect(params, centre, extent, rect);
    if (rect.x > rect.z || rect.y > rect.w) {
        culled(slot, kCullOffscreen);
        return;
    }
    uint touched = 0;
    for (uint ty = rect.y; ty <= rect.w; ++ty) {
        for (uint tx = rect.x; tx <= rect.z; ++tx) {
            if (tileTouched(params, centre, conic, cutoff, tx, ty)) {
                touched += 1;
            }
        }
    }
    if (touched == 0) {
        culled(slot, kCullNoTile);
        return;
    }

    // Colour from this eye, in the cloud's space, made linear here.
    const float3 base = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]), highHalf(shape[row + 3]));
    const uint degree = min(min(degreeOf(params.restPerColour), params.shLimit), kShDegree);
    const float3 dir = normalize(posOpacity.xyz - float3(params.eyeX, params.eyeY, params.eyeZ));
    const float3 colour = cloudLight(base + evaluateRest(sh, i * params.shWords, degree, dir), params.linearCloud != 0);

    ProjRecord record;
    record.screen = float4(centre, t.z, cutoff);
    record.conic = float4(conic, alpha);
    record.colour = float4(colour, 0.0);
    proj[slot] = record;
    const uint at = arena.rectsAt + slot * 4;
    frameWords[at] = rect.x;
    frameWords[at + 1] = rect.y;
    frameWords[at + 2] = rect.z;
    frameWords[at + 3] = rect.w;
    frameWords[arena.touchedAt + slot] = touched;
    frameWords[arena.visibleAt + slot] = 1;
    frameWords[arena.keysAt + slot] = bit_cast<uint>(t.z) >> 7;
}
