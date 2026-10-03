// Copyright (c) 2026 jesus luque.
//
// A ZONAL TRANSFER, FITTED AND TURNED (plugins/splattransferzonal, and
// `splatTransferFrame` in splat/splat_relight.slang, which a frame reads it
// through).
//
// tzcSynth writes, a gaussian a thread, the picture the fit reads: a frame
// (a random rotation, or none for one gaussian in eight), the gaussian's
// transfer as nine harmonics in the world, and sixty-four random bits.
// Even gaussians carry a transfer that IS two zonal lobes, which the fit must
// give back to the halves' rounding; odd ones carry what a bake gives -- the
// clamped cosine about the frame's normal with a cap of the sky taken away by
// an occluder -- which two lobes can only approximate, and how well is the
// number this test states.
//
// tzcPack lays the fit's answer out as a frame reads a cloud's transfer: ten
// halves a gaussian, f16 pairs (scene/splat_decode.slang's layout).
//
// tzcCheck reads it back through `splatTransferFrame`:
//   - against the harmonics it was fitted to;
//   - turned: a gaussian whose frame is turned by M, read along M w, gives
//     what the unturned one gives along w -- whether M is the gaussian's own
//     frame (a skeleton's pose) or the instance's rows (a prim's transform);
//   - the sun's share through the bits laid out in the frame, turned the same
//     way, is the same share;
//   - a gaussian with no frame to turn keeps its bits exactly.
module transfer_zonal_check;
import athenea.splat.splat_relight;
import athenea.common.packing;

struct SynthParams {
    uint count;
    uint inWidth;     // the fit's input picture: entries a row, pixels a row
    uint inStride;
    uint outWidth;    // its answer
    uint outStride;
    uint lobes;
    uint pad0;
    uint pad1;
};

RWStructuredBuffer<float4> picture;   // the fit's input
StructuredBuffer<float4>   fitted;    // its answer
RWStructuredBuffer<uint>   packed;    // tzcPack's: five words a gaussian
StructuredBuffer<uint>     halves;    // the same, read as a frame reads it
RWStructuredBuffer<float>  truth;     // twelve a gaussian: nine harmonics, two words of bits, the case
RWStructuredBuffer<uint>   results;   // see tzcCheck
ConstantBuffer<SynthParams> synth;

static const float kPi = 3.14159265358979;

uint at(uint entry, uint width, uint stride) {
    return (entry / width) * stride + entry % width;
}

uint hashOf(uint k) {
    uint h = k * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

float uniformOf(inout uint state) {
    state = hashOf(state);
    return float(state >> 8) / 16777216.0;
}

float3 directionOf(inout uint state) {
    const float z = uniformOf(state) * 2.0 - 1.0;
    const float phi = uniformOf(state) * 2.0 * kPi;
    const float r = sqrt(max(1.0 - z * z, 0.0));
    return float3(r * cos(phi), r * sin(phi), z);
}

float4 rotationOf(inout uint state) {
    // A uniform rotation (Shoemake).
    const float u1 = uniformOf(state), u2 = uniformOf(state), u3 = uniformOf(state);
    const float a = sqrt(1.0 - u1), b = sqrt(u1);
    return float4(a * sin(2.0 * kPi * u2), a * cos(2.0 * kPi * u2), b * sin(2.0 * kPi * u3), b * cos(2.0 * kPi * u3));
}

float evaluate(float f[9], float3 d) {
    float sum = 0.0;
    [unroll] for (uint k = 0; k < 9; ++k) {
        sum += f[k] * envBasisValue(k, d);
    }
    return sum;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void tzcSynth(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= synth.count) {
        return;
    }
    uint state = g * 9781u + 17u;
    const float4 q = (g % 8) == 0 ? float4(0.0, 0.0, 0.0, 1.0) : rotationOf(state);
    const float3x3 frame = axesOfQuaternion(q);
    const float3 normal = float3(frame[0][2], frame[1][2], frame[2][2]);
    float f[9];
    [unroll] for (uint k = 0; k < 9; ++k) {
        f[k] = 0.0;
    }
    if ((g & 1) == 0) {
        // Two zonal lobes, exactly.
        [unroll] for (uint lobe = 0; lobe < 2; ++lobe) {
            const float3 axis = directionOf(state);
            const float3 z = float3(0.1 + 0.4 * uniformOf(state), 0.6 * uniformOf(state) - 0.3,
                                    0.6 * uniformOf(state) - 0.3) * (lobe == 0 ? 1.0 : 0.5);
            [unroll] for (uint k = 0; k < 9; ++k) {
                f[k] += z[zonalBandOf(k)] * zonalBandScale(k) * envBasisValue(k, axis);
            }
        }
    } else {
        // A BAKE'S SHAPE: the clamped cosine over pi about the normal, with
        // the directions within `cap` of an occluder's centre taken away; the
        // occluder leans 30 to 70 degrees off the normal. Projected over a
        // thousand directions.
        const float tilt = (30.0 + 40.0 * uniformOf(state)) * kPi / 180.0;
        const float cap = (20.0 + 30.0 * uniformOf(state)) * kPi / 180.0;
        const float3 helper = abs(normal.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
        const float3 side = normalize(cross(helper, normal));
        const float spin = uniformOf(state) * 2.0 * kPi;
        const float3 across = cos(spin) * side + sin(spin) * cross(normal, side);
        const float3 centre = normalize(cos(tilt) * normal + sin(tilt) * across);
        const uint directions = 1024;
        for (uint i = 0; i < directions; ++i) {
            const float z = 1.0 - (float(i) + 0.5) / float(directions) * 2.0;
            const float r = sqrt(max(1.0 - z * z, 0.0));
            const float phi = float(i) * 2.39996322972865332;
            const float3 w = float3(r * cos(phi), r * sin(phi), z);
            const float cosine = dot(w, normal);
            if (cosine <= 0.0 || dot(w, centre) > cos(cap)) {
                continue;
            }
            const float value = cosine / kPi * (4.0 * kPi / float(directions));
            [unroll] for (uint k = 0; k < 9; ++k) {
                f[k] += value * envBasisValue(k, w);
            }
        }
    }
    const uint bitsLo = hashOf(state + 1u);
    const uint bitsHi = hashOf(state + 2u);
    picture[at(g * 4, synth.inWidth, synth.inStride)] = float4(f[0], f[1], f[2], f[3]);
    picture[at(g * 4 + 1, synth.inWidth, synth.inStride)] = float4(f[4], f[5], f[6], f[7]);
    picture[at(g * 4 + 2, synth.inWidth, synth.inStride)] = float4(f[8], asfloat(bitsLo), asfloat(bitsHi), 1.0);
    picture[at(g * 4 + 3, synth.inWidth, synth.inStride)] = q;
    [unroll] for (uint k = 0; k < 9; ++k) {
        truth[g * 12 + k] = f[k];
    }
    truth[g * 12 + 9] = asfloat(bitsLo);
    truth[g * 12 + 10] = asfloat(bitsHi);
    truth[g * 12 + 11] = asfloat(q.x == 0.0 && q.y == 0.0 && q.z == 0.0 ? 1u : 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void tzcPack(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= synth.count) {
        return;
    }
    const float4 e0 = fitted[at(g * 3, synth.outWidth, synth.outStride)];
    const float4 e1 = fitted[at(g * 3 + 1, synth.outWidth, synth.outStride)];
    const float4 e2 = fitted[at(g * 3 + 2, synth.outWidth, synth.outStride)];
    const float values[10] = { e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w, e2.x, e2.y };
    [unroll] for (uint w = 0; w < 5; ++w) {
        packed[g * 5 + w] = (f32tof16(values[w * 2 + 1]) << 16) | f32tof16(values[w * 2]);
    }
}

/// The gaussian `g`'s transfer, read as a frame reads it: under `frame` (the
/// gaussian's own axes, columns) and the instance rows.
SplatSurface surfaceOf(uint g, float3x3 frame, float4 row0, float4 row1, float4 row2) {
    SplatSurface s;
    s.transferCount = kTransferZonalCount;
    s.transferBase = g * 10;
    s.thinWalled = false;
    s.schlickMetal = false;
    s.hasShadowBits = false;
    splatTransferFrame(s, halves, frame, row0, row1, row2);
    return s;
}

float3x3 matrixOf(float4 r0, float4 r1, float4 r2) {
    return float3x3(r0.xyz, r1.xyz, r2.xyz);
}

/// results[0] lobes-exact gaussians whose fit is off by more than 0.5%
/// results[1] the bake-shaped ones' relative error, summed, in 1e-5
/// results[2] the bake-shaped ones counted
/// results[3] gaussians a turn of their frame did not carry
/// results[4] gaussians a turn of the instance did not carry
/// results[5] gaussians whose sun share changed under a turn
/// results[6] unturned gaussians whose bits changed
/// results[7] the lobes-exact gaussians' worst relative error, in 1e-6
/// results[8] the bake-shaped ones under 10% relative error
[shader("compute")]
[numthreads(64, 1, 1)]
void tzcCheck(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= synth.count) {
        return;
    }
    uint state = g * 9781u + 17u;
    const float4 q = (g % 8) == 0 ? float4(0.0, 0.0, 0.0, 1.0) : rotationOf(state);
    const float3x3 frame = axesOfQuaternion(q);
    float f[9];
    float whole = 0.0;
    [unroll] for (uint k = 0; k < 9; ++k) {
        f[k] = truth[g * 12 + k];
        whole += f[k] * f[k];
    }
    const float4 i0 = float4(1.0, 0.0, 0.0, 0.0);
    const float4 i1 = float4(0.0, 1.0, 0.0, 0.0);
    const float4 i2 = float4(0.0, 0.0, 1.0, 0.0);
    const SplatSurface s = surfaceOf(g, frame, i0, i1, i2);
    float wrong = 0.0;
    [unroll] for (uint k = 0; k < 9; ++k) {
        const float d = s.transferSh[k] - f[k];
        wrong += d * d;
    }
    const float relative = sqrt(wrong / max(whole, 1.0e-20));
    if ((g & 1) == 0) {
        if (relative > 5.0e-3) {
            InterlockedAdd(results[0], 1u);
        }
        InterlockedMax(results[7], uint(min(relative, 1000.0) * 1.0e6));
    } else {
        InterlockedAdd(results[1], uint(min(relative, 10.0) * 1.0e5));
        InterlockedAdd(results[2], 1u);
        if (relative < 0.1) {
            InterlockedAdd(results[8], 1u);
        }
    }

    // TURNED. M a random rotation; the frame turned by it, and separately
    // the instance's rows, must each read along M w what the unturned
    // gaussian reads along w.
    const float3x3 turn = axesOfQuaternion(rotationOf(state));
    const SplatSurface byFrame = surfaceOf(g, mul(turn, frame), i0, i1, i2);
    const SplatSurface byRows = surfaceOf(g, frame, float4(turn[0], 0.0), float4(turn[1], 0.0), float4(turn[2], 0.0));
    float scale = 0.0;
    [unroll] for (uint k = 0; k < 9; ++k) {
        scale = max(scale, abs(s.transferSh[k]));
    }
    bool frameOff = false;
    bool rowsOff = false;
    for (uint i = 0; i < 16; ++i) {
        const float3 w = directionOf(state);
        const float3 turned = mul(turn, w);
        const float want = evaluate(s.transferSh, w);
        if (abs(evaluate(byFrame.transferSh, turned) - want) > 1.0e-4 * max(scale, 1.0e-3)) {
            frameOff = true;
        }
        if (abs(evaluate(byRows.transferSh, turned) - want) > 1.0e-4 * max(scale, 1.0e-3)) {
            rowsOff = true;
        }
    }
    if (frameOff) {
        InterlockedAdd(results[3], 1u);
    }
    if (rowsOff) {
        InterlockedAdd(results[4], 1u);
    }

    // THE SUN THROUGH THE BITS, laid out in the frame: turned with the frame,
    // the same sun turned with it gets through by the same share.
    const uint2 local = uint2(asuint(fitted[at(g * 3 + 2, synth.outWidth, synth.outStride)].z),
                              asuint(fitted[at(g * 3 + 2, synth.outWidth, synth.outStride)].w));
    SplatSurface lit = s;
    lit.hasShadowBits = true;
    lit.shadowBits = local;
    lit.normal = float3(frame[0][2], frame[1][2], frame[2][2]);
    SplatSurface litTurned = byFrame;
    litTurned.hasShadowBits = true;
    litTurned.shadowBits = local;
    litTurned.normal = mul(turn, lit.normal);
    bool sunOff = false;
    for (uint i = 0; i < 8; ++i) {
        const float3 sun = directionOf(state);
        if (abs(splatSunOpen(lit, halves, sun) - splatSunOpen(litTurned, halves, mul(turn, sun))) > 1.0e-3) {
            sunOff = true;
        }
    }
    if (sunOff) {
        InterlockedAdd(results[5], 1u);
    }
    // A frame that is the world's: the bits laid out over it are the bits.
    if ((g % 8) == 0 && (local.x != asuint(truth[g * 12 + 9]) || local.y != asuint(truth[g * 12 + 10]))) {
        InterlockedAdd(results[6], 1u);
    }
}
