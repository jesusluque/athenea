// Copyright (c) 2026 jesus luque.
//
// THE GEOMETRY LAYER (video_planes_common.slang, "THE GEOMETRY LAYER"):
// position, scale and rotation sent exact beside the video, absolute on a
// key or as a zigzagged difference from the same gaussian a frame before.
//   videoGeometryQuantise  encode: a picture's gaussians' four words, at
//                          their index in the frame (a stable order's)
//   videoGeometryHold      encode: conditional replenishment, a field that
//                          moved too little keeps the value the decoder holds
//   videoGeometryPack      encode: those words, absolute or as differences
//                          from the frame before's, into 16 byte planes a
//                          chunk; a thread a word of four bytes
//   videoGeometryApply     decode: the planes onto the state the frame before
//                          left -- the key's words, or the state plus the
//                          difference; a thread a gaussian
module video_geometry;
import athenea.common.packing;
import video_planes_common;

struct GeometryParams {
    uint count;         // the picture's gaussians, or the page's
    uint first;         // encode: the picture's first Morton rank (= its first splat in the frame); decode: the page's
    uint chunkSplats;
    uint frame;         // the source frame, from 0
    uint interval;      // frames from one key to the next (0: every frame a key)
    uint stagger;       // 1: a key refreshes blocks of 256 in turn, not the whole frame
    uint planesOffset;  // decode: bytes into `page` where the page's planes start
    uint frameFirst;    // decode: the picture's first splat in its frame
    uint pageFirst;     // decode: the page's first splat in its picture
    uint positionBits;  // 10-16
    uint scaleBits;     // 6-10
    uint order;         // the prediction: 1 the frame before, 2 its motion carried on
    uint state2At;      // decode: the word where the frame before's state starts in `state` (its second half)
    uint rotationBits;  // 5-10: a rotation component's bits (the word keeps 10, the low ones zero)
    uint holdPosition;  // videoGeometryHold: steps a position field may drift from what the decoder holds
    uint holdScale;     // ... a log scale field
    uint holdRotation;  // ... a rotation component (in `rotationBits` steps); the index never holds
    uint rangesOffset;  // decode, order 3: bytes into `page` where the record's ranges (2 floats an attribute) start
    uint chunkBytes;    // a chunk's geometry bytes: 16 planes, then (order 3) its clusters' motion
    uint pad1;
    uint pad2;
};

StructuredBuffer<float4>         positions;
StructuredBuffer<uint>           shape;
StructuredBuffer<uint>           order;      // Morton rank -> source splat
StructuredBuffer<float>          ranges;     // attributes x 2, the sequence's
RWStructuredBuffer<uint>         now;        // frame splat j: four words
StructuredBuffer<uint>           before;     // the frame before's
StructuredBuffer<uint>           before2;    // the one before that (order 2)
RWStructuredBuffer<uint>         planes;     // encode: 16 byte planes a chunk, a chunk after another
StructuredBuffer<uint>           page;       // decode: the side data holding the planes
StructuredBuffer<uint>           rangesPage; // decode, order 3: where the ranges are (the page; the encoder's own)
RWStructuredBuffer<uint>         code;       // encode, order 3: what is sent for frame splat j, four words
RWStructuredBuffer<uint>         clusters;   // encode, order 3: kMotionWords a cluster of the frame
RWStructuredBuffer<uint>         state;      // decode: frame splat j's four words, kept from frame to frame; from
                                             // `state2At` on, the frame before's (order 2)
ConstantBuffer<GeometryParams>   params;

// --- MOTION (order 3) -------------------------------------------------------------
// A frame's gaussians in clusters of kMotionCluster consecutive indices (a run
// of the stable order: a small region). Each cluster's rigid motion from the
// state the decoder holds to this frame is fitted on the encoder (Kabsch:
// centroids, cross-covariance, Horn's quaternion) and sent after the chunk's
// planes, eight words a cluster:
//   0-2  c   the held centroid, int32, quarter position steps
//   3-5  t   the frame's centroid, int32, quarter position steps
//   6    qx | qy << 16, 7  qz | qw << 16   the rotation, int16 each, 8192 = 1, w >= 0
// A gaussian between keys is predicted from its held value through its
// cluster's motion and sends what the prediction missed:
//   position  p^ = (t + (R (4 p - c) + 4096) >> 13 + 2) >> 2, integer only, R
//             the rotation's matrix in 1/8192 (motionPosition); held more
//             than 2^14 steps from c, the translation alone
//   log scale the held value (no prediction)
//   rotation  q^ = q_R * q_held; sent as n, three integers: the turn from q^
//             to the gaussian in its own frame (log map), in steps of
//             motionSteps -- the 'rotationBits' component step as an angle,
//             widened on an axis to what moves the gaussian's extent half a
//             position step (|sigma_j - sigma_k| an axis's lever), none past
//             pi; q = q^ * exp(n * step), packed in the store's 10-bit word
// The fields, zigzagged: px py pz (16 bits), lsx lsy lsz (16), and n0 | n1 <<
// 10 | n2 << 20 (10 bits each) in the rotation's word. A key sends the four
// words absolute -- what the decoder would have held, not the source again,
// so a key does not jump.
public static const uint kMotionOrder = 3;
public static const uint kMotionCluster = kGeometryKeyBlock;   // a staggered key's block
public static const uint kMotionWords = 8;
public static const int kMotionOne = 8192;

struct Motion {
    int3 c;
    int3 t;
    int4 q;
};
Motion motionOf(uint w[kMotionWords]) {
    Motion m;
    m.c = int3(int(w[0]), int(w[1]), int(w[2]));
    m.t = int3(int(w[3]), int(w[4]), int(w[5]));
    m.q = int4(int(w[6] << 16) >> 16, int(w[6]) >> 16, int(w[7] << 16) >> 16, int(w[7]) >> 16);
    return m;
}
struct MotionRanges {
    float r[12];   // position x, y, z then log scale x, y, z: lo, hi each
}

uint3 motionPosition(uint3 held, Motion m, uint bits) {
    const int3 d = int3(held) * 4 - m.c;
    int3 p4;
    if (any(abs(d) >= int3(1 << 16))) {
        p4 = int3(held) * 4 + (m.t - m.c);
    } else {
        const int x = m.q.x, y = m.q.y, z = m.q.z, w = m.q.w;
        const int3 r0 = int3(kMotionOne - ((y * y + z * z) >> 12), (x * y - z * w) >> 12, (x * z + y * w) >> 12);
        const int3 r1 = int3((x * y + z * w) >> 12, kMotionOne - ((x * x + z * z) >> 12), (y * z - x * w) >> 12);
        const int3 r2 = int3((x * z - y * w) >> 12, (y * z + x * w) >> 12, kMotionOne - ((x * x + y * y) >> 12));
        const int3 rd = int3(r0.x * d.x + r0.y * d.y + r0.z * d.z, r1.x * d.x + r1.y * d.y + r1.z * d.z,
                             r2.x * d.x + r2.y * d.y + r2.z * d.z);
        p4 = m.t + ((rd + 4096) >> 13);
    }
    return uint3(clamp((p4 + 2) >> 2, int3(0), int3((1 << bits) - 1)));
}
float4 quatMul(float4 a, float4 b) {
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
float4 quatConj(float4 q) { return float4(-q.xyz, q.w); }
float3 quatLog(float4 q) {
    q = q.w < 0.0 ? -q : q;
    const float s = length(q.xyz);
    return s < 1.0e-12 ? float3(0.0) : q.xyz / s * (2.0 * atan2(s, q.w));
}
float4 quatExp(float3 v) {
    const float a = length(v);
    return a < 1.0e-12 ? float4(0.0, 0.0, 0.0, 1.0) : float4(v / a * sin(0.5 * a), cos(0.5 * a));
}
float4 motionRotation(uint heldWord, Motion m) {
    const float4 qr = normalize(float4(m.q) / float(kMotionOne));
    return normalize(quatMul(qr, decodeQuaternion(heldWord)));
}
/// The rotation's step on each of the gaussian's axes (0: not sent).
float3 motionSteps(uint3 logScale, MotionRanges r, uint positionBits, uint scaleBits, uint rotationBits) {
    const float levels = float((1u << scaleBits) - 1u), pl = float((1u << positionBits) - 1u);
    float3 sigma;
    for (uint a = 0; a < 3; ++a) {
        sigma[a] = exp(dequantise(logScale[a], r.r[6 + 2 * a], r.r[7 + 2 * a], (1u << scaleBits) - 1u));
    }
    const float positionStep = ((r.r[1] - r.r[0]) + (r.r[3] - r.r[2]) + (r.r[5] - r.r[4])) / (3.0 * pl);
    const float base = 4.0 * kSqrtHalf / float((1u << min(rotationBits, 10u)) - 1u);
    float3 step;
    for (uint a = 0; a < 3; ++a) {
        const float lever = abs(sigma[(a + 1) % 3] - sigma[(a + 2) % 3]);
        const float s = max(base, 0.5 * positionStep / max(lever, 1.0e-12));
        step[a] = s > 3.14159265 ? 0.0 : s;
    }
    return step;
}
uint zigzagSigned(int v, uint bits) { return (uint(v << 1) ^ uint(v >> 31)) & ((1u << bits) - 1u); }
int unzigzagSigned(uint z) { return int(z >> 1) ^ -int(z & 1u); }
/// The state after a frame between keys: `held` moved by its cluster, plus what
/// was sent (`sent`: the zigzagged fields).
uint4 motionApply(uint4 held, Motion m, uint4 sent, MotionRanges r, uint positionBits, uint scaleBits,
                  uint rotationBits) {
    const uint3 p = motionPosition(uint3(held.x & 0xFFFFu, held.x >> 16, held.y & 0xFFFFu), m, positionBits);
    uint f[6];
    for (uint a = 0; a < 3; ++a) {
        f[a] = (p[a] + unzigzag16(geometryField(sent, a))) & 0xFFFFu;
        f[3 + a] = (geometryField(held, 3 + a) + unzigzag16(geometryField(sent, 3 + a))) & 0xFFFFu;
    }
    const float3 step = motionSteps(uint3(f[3], f[4], f[5]), r, positionBits, scaleBits, rotationBits);
    const int3 n = int3(unzigzagSigned(sent.w & 0x3FFu), unzigzagSigned((sent.w >> 10) & 0x3FFu),
                        unzigzagSigned((sent.w >> 20) & 0x3FFu));
    const float4 q = quatMul(motionRotation(held.w, m), quatExp(float3(n) * step));
    return uint4(f[0] | (f[1] << 16u), f[2] | (f[3] << 16u), f[4] | (f[5] << 16u), encodeQuaternion(q));
}
MotionRanges encodeRanges() {
    MotionRanges r;
    for (uint k = 0; k < 6; ++k) {
        r.r[k] = ranges[k];
        r.r[6 + k] = ranges[8 + k];
    }
    return r;
}

/// Encode, order 3: each cluster's motion from the held state (`before`) to
/// the frame's source positions; a workgroup a cluster.
groupshared float gsSum[256][9];
[shader("compute")]
[numthreads(256, 1, 1)]
void videoGeometryFit(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID) {
    const uint cluster = params.first / kMotionCluster + gid.x, t = lid.x;   // `first` a multiple of a chunk
    const uint j = cluster * kMotionCluster + t;            // frame index
    const bool valid = j >= params.first && j < params.first + params.count;
    float3 x = float3(0.0), y = float3(0.0);
    if (valid) {
        x = float3(float(before[j * 4] & 0xFFFFu), float(before[j * 4] >> 16), float(before[j * 4 + 1] & 0xFFFFu));
        const float4 p = positions[order[j]];
        const float levels = float((1u << params.positionBits) - 1u);
        for (uint a = 0; a < 3; ++a) {
            const float lo = ranges[a * 2], hi = ranges[a * 2 + 1];
            y[a] = hi > lo ? saturate((p[a] - lo) / (hi - lo)) * levels : 0.0;
        }
    }
    // Centroids first, then the cross-covariance about them.
    gsSum[t][0] = x.x; gsSum[t][1] = x.y; gsSum[t][2] = x.z;
    gsSum[t][3] = y.x; gsSum[t][4] = y.y; gsSum[t][5] = y.z;
    gsSum[t][6] = valid ? 1.0 : 0.0;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) {
            for (uint k = 0; k < 7; ++k) gsSum[t][k] += gsSum[t + s][k];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float count = gsSum[0][6];
    const float3 cx = count > 0.0 ? float3(gsSum[0][0], gsSum[0][1], gsSum[0][2]) / count : float3(0.0);
    const float3 cy = count > 0.0 ? float3(gsSum[0][3], gsSum[0][4], gsSum[0][5]) / count : float3(0.0);
    GroupMemoryBarrierWithGroupSync();
    const float3 dx = valid ? x - cx : float3(0.0), dy = valid ? y - cy : float3(0.0);
    for (uint a = 0; a < 3; ++a) {
        for (uint b = 0; b < 3; ++b) gsSum[t][a * 3 + b] = dx[a] * dy[b];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) {
            for (uint k = 0; k < 9; ++k) gsSum[t][k] += gsSum[t + s][k];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t != 0 || count == 0.0) {
        return;
    }
    const float xx = gsSum[0][0], xy = gsSum[0][1], xz = gsSum[0][2], yx = gsSum[0][3], yy = gsSum[0][4],
                yz = gsSum[0][5], zx = gsSum[0][6], zy = gsSum[0][7], zz = gsSum[0][8];
    // Horn: the rotation is the eigenvector of the largest eigenvalue, (w, x, y, z).
    const float4x4 n = float4x4(xx + yy + zz, yz - zy, zx - xz, xy - yx,
                                yz - zy, xx - yy - zz, xy + yx, zx + xz,
                                zx - xz, xy + yx, -xx + yy - zz, yz + zy,
                                xy - yx, zx + xz, yz + zy, -xx - yy + zz);
    float shift = 0.0;
    for (uint r = 0; r < 4; ++r) {
        for (uint c = 0; c < 4; ++c) shift += abs(n[r][c]);
    }
    float4 v = float4(1.0, 0.0, 0.0, 0.0);
    if (count >= 4.0 && shift > 1.0e-6) {
        for (uint it = 0; it < 64; ++it) {
            v = normalize(mul(n, v) + shift * v);
        }
    }
    float4 q = float4(v.y, v.z, v.w, v.x);   // (x, y, z, w)
    q = q.w < 0.0 ? -q : q;
    const int4 qi = int4(round(q * float(kMotionOne)));
    const int3 c4 = int3(round(cx * 4.0)), t4 = int3(round(cy * 4.0));
    const uint o = cluster * kMotionWords;
    clusters[o + 0] = uint(c4.x); clusters[o + 1] = uint(c4.y); clusters[o + 2] = uint(c4.z);
    clusters[o + 3] = uint(t4.x); clusters[o + 4] = uint(t4.y); clusters[o + 5] = uint(t4.z);
    clusters[o + 6] = (uint(qi.x) & 0xFFFFu) | (uint(qi.y) << 16u);
    clusters[o + 7] = (uint(qi.z) & 0xFFFFu) | (uint(qi.w) << 16u);
}

/// Encode, order 3: each gaussian of the picture against its cluster's
/// prediction -- what is sent (`code`); `now` takes the held words, and the
/// encoder then runs videoGeometryApply, the decoder's own kernel, on the
/// packed planes, so what it holds is what a decoder holds bit for bit (the
/// rotation's float steps rounded by the same compiled code).
/// Holds (conditional replenishment against the prediction): a position
/// axis or log scale within its threshold, or a rotation whose three steps
/// all are, sends 0 and keeps the prediction.
[shader("compute")]
[numthreads(256, 1, 1)]
void videoGeometryMotion(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint j = params.first + i;
    const uint s = order[j];
    const MotionRanges r = encodeRanges();
    const uint4 b = uint4(before[j * 4], before[j * 4 + 1], before[j * 4 + 2], before[j * 4 + 3]);
    uint w[kMotionWords];
    for (uint k = 0; k < kMotionWords; ++k) {
        w[k] = clusters[(j / kMotionCluster) * kMotionWords + k];
    }
    const Motion m = motionOf(w);
    // The source, quantised as a key would send it.
    const float4 p = positions[s];
    const uint w1 = shape[s * 4 + 1], w2 = shape[s * 4 + 2];
    const float3 ls =
        clamp(float3(lowHalf(w1), highHalf(w1), lowHalf(w2)), float3(kLogScaleFloor), float3(kLogScaleTop));
    const uint3 pred = motionPosition(uint3(b.x & 0xFFFFu, b.x >> 16, b.y & 0xFFFFu), m, params.positionBits);
    uint sent[6];
    uint3 lsNow;
    for (uint a = 0; a < 3; ++a) {
        const int qp = int(quantise(p[a], ranges[a * 2], ranges[a * 2 + 1], (1u << params.positionBits) - 1u));
        int d = qp - int(pred[a]);
        d = uint(abs(d)) <= params.holdPosition ? 0 : d;
        sent[a] = zigzagSigned(d, 16u);
        const int ql = int(quantise(ls[a], ranges[(4 + a) * 2], ranges[(4 + a) * 2 + 1], (1u << params.scaleBits) - 1u));
        int e = ql - int(geometryField(b, 3 + a));
        e = uint(abs(e)) <= params.holdScale ? 0 : e;
        sent[3 + a] = zigzagSigned(e, 16u);
        lsNow[a] = uint(int(geometryField(b, 3 + a)) + e);
    }
    const float3 step = motionSteps(lsNow, r, params.positionBits, params.scaleBits, params.rotationBits);
    const float3 turn = quatLog(quatMul(quatConj(motionRotation(b.w, m)), normalize(decodeQuaternion(shape[s * 4]))));
    int3 n;
    for (uint a = 0; a < 3; ++a) {
        n[a] = step[a] > 0.0 ? clamp(int(round(turn[a] / step[a])), -511, 511) : 0;
    }
    if (all(uint3(abs(n)) <= uint3(params.holdRotation))) {
        n = int3(0);
    }
    const uint4 sentWords = uint4(sent[0] | (sent[1] << 16u), sent[2] | (sent[3] << 16u), sent[4] | (sent[5] << 16u),
                                  zigzagSigned(n.x, 10u) | (zigzagSigned(n.y, 10u) << 10u) |
                                      (zigzagSigned(n.z, 10u) << 20u));
    const uint4 held = motionApply(b, m, sentWords, r, params.positionBits, params.scaleBits, params.rotationBits);
    const bool key = geometryKey(j, params.frame, params.interval, params.stagger);
    for (uint k = 0; k < 4; ++k) {
        now[j * 4 + k] = b[k];
        code[j * 4 + k] = key ? held[k] : sentWords[k];
    }
}


[shader("compute")]
[numthreads(256, 1, 1)]
void videoGeometryQuantise(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint j = params.first + i;
    const uint s = order[j];
    const float4 p = positions[s];
    uint q[6];
    for (uint a = 0; a < 3; ++a) {
        q[a] = quantise(p[a], ranges[a * 2], ranges[a * 2 + 1], (1u << params.positionBits) - 1u);
    }
    const uint w1 = shape[s * 4 + 1], w2 = shape[s * 4 + 2];
    const float3 ls =
        clamp(float3(lowHalf(w1), highHalf(w1), lowHalf(w2)), float3(kLogScaleFloor), float3(kLogScaleTop));
    for (uint a = 0; a < 3; ++a) {
        q[3 + a] = quantise(ls[a], ranges[(4 + a) * 2], ranges[(4 + a) * 2 + 1], (1u << params.scaleBits) - 1u);
    }
    now[j * 4 + 0] = q[0] | (q[1] << 16u);
    now[j * 4 + 1] = q[2] | (q[3] << 16u);
    now[j * 4 + 2] = q[4] | (q[5] << 16u);
    // The store's own rotation word, its three components rounded to
    // `rotationBits` (the low bits zero): what the layer sends exactly.
    uint w = shape[s * 4];
    const uint drop = 10u - min(params.rotationBits, 10u);
    if (drop != 0) {
        uint rounded = w & 3u;
        for (uint c = 0; c < 3; ++c) {
            const uint shift = 22u - 10u * c;
            const uint v = min((((w >> shift) & 0x3FFu) + (1u << (drop - 1u))) >> drop, (1u << (10u - drop)) - 1u);
            rounded |= (v << drop) << shift;
        }
        w = rounded;
    }
    now[j * 4 + 3] = w;
}

/// CONDITIONAL REPLENISHMENT (`--geometry-hold`): a field that moved no more
/// than its threshold from what the decoder holds (`before`, the frame
/// before as the decoder rebuilt it) is not sent -- `now` takes the held value,
/// so its residual is 0 and the decoder's state stays exactly the encoder's.
/// Values hold still until they have moved enough: no flicker by construction.
/// Only between keys (a key sends what is held) and with order 1.
[shader("compute")]
[numthreads(256, 1, 1)]
void videoGeometryHold(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint j = params.first + i;
    if (geometryKey(j, params.frame, params.interval, params.stagger)) {
        return;
    }
    uint4 g = uint4(now[j * 4], now[j * 4 + 1], now[j * 4 + 2], now[j * 4 + 3]);
    const uint4 b = uint4(before[j * 4], before[j * 4 + 1], before[j * 4 + 2], before[j * 4 + 3]);
    uint f[6];
    for (uint k = 0; k < 6; ++k) {
        const uint v = geometryField(g, k), w = geometryField(b, k);
        const int d = int((v - w) << 16) >> 16;
        f[k] = uint(abs(d)) <= (k < 3 ? params.holdPosition : params.holdScale) ? w : v;
    }
    uint rot = g.w;
    if ((g.w & 3u) == (b.w & 3u)) {
        const uint n = min(params.rotationBits, 10u), drop = 10u - n;
        bool held = true;
        for (uint c = 0; c < 3; ++c) {
            const uint shift = 22u - 10u * c + drop, mask = (1u << n) - 1u;
            const int d = int((g.w >> shift) & mask) - int((b.w >> shift) & mask);
            held = held && uint(abs(d)) <= params.holdRotation;
        }
        rot = held ? b.w : g.w;
    }
    now[j * 4 + 0] = f[0] | (f[1] << 16u);
    now[j * 4 + 1] = f[2] | (f[3] << 16u);
    now[j * 4 + 2] = f[4] | (f[5] << 16u);
    now[j * 4 + 3] = rot;
}

/// Byte `plane` of gaussian j's value: its field absolute, or zigzagged from its prediction.
uint geometryByte(uint j, uint plane) {
    const uint f = plane < 12 ? plane / 2u : 6u;
    const uint shift = plane < 12 ? (plane & 1u) * 8u : (plane - 12u) * 8u;
    if (params.order == kMotionOrder) {
        const uint4 c = uint4(code[j * 4], code[j * 4 + 1], code[j * 4 + 2], code[j * 4 + 3]);
        return (geometryField(c, f) >> shift) & 0xFFu;
    }
    const uint4 g = uint4(now[j * 4], now[j * 4 + 1], now[j * 4 + 2], now[j * 4 + 3]);
    uint v = geometryField(g, f);
    const uint order = geometryOrder(j, params.frame, params.interval, params.stagger, params.order);
    if (order != 0) {
        const uint4 b = uint4(before[j * 4], before[j * 4 + 1], before[j * 4 + 2], before[j * 4 + 3]);
        const uint4 c = uint4(before2[j * 4], before2[j * 4 + 1], before2[j * 4 + 2], before2[j * 4 + 3]);
        v = f == 6 ? rotationResidualBits(v, predictRotation(b.w, c.w, order), params.rotationBits)
                   : zigzag16(v - predictField(geometryField(b, f), geometryField(c, f), order, 0xFFFFu));
    }
    return (v >> shift) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoGeometryPack(uint3 tid: SV_DispatchThreadID) {
    const uint perPlane = params.chunkSplats / 4u;
    const uint chunks = (params.count + params.chunkSplats - 1) / params.chunkSplats;
    const uint chunkWords = params.chunkBytes / 4u;
    const uint chunk = tid.x / chunkWords, inChunk = tid.x % chunkWords;
    if (chunk >= chunks) {
        return;
    }
    if (inChunk >= perPlane * kGeometryPlanes) {
        // Order 3: the chunk's clusters' motion after its planes.
        const uint k = inChunk - perPlane * kGeometryPlanes;
        const uint cluster = (params.first + chunk * params.chunkSplats) / kMotionCluster + k / kMotionWords;
        planes[tid.x] = clusters[cluster * kMotionWords + k % kMotionWords];
        return;
    }
    const uint w = inChunk % perPlane, plane = inChunk / perPlane;
    uint word = 0;
    for (uint m = 0; m < 4; ++m) {
        const uint i = chunk * params.chunkSplats + w * 4u + m;
        if (i < params.count) {
            word |= geometryByte(params.first + i, plane) << (m * 8u);
        }
    }
    planes[tid.x] = word;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoGeometryApply(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    const uint i = params.pageFirst + tid.x;              // in the picture
    const uint j = params.frameFirst + i;                 // in the frame
    const uint chunk = tid.x / params.chunkSplats;        // in the page
    const uint at = tid.x % params.chunkSplats;
    uint bytes[kGeometryPlanes];
    for (uint plane = 0; plane < kGeometryPlanes; ++plane) {
        const uint e = params.planesOffset + chunk * params.chunkBytes + plane * params.chunkSplats + at;
        bytes[plane] = (page[e >> 2] >> ((e & 3u) * 8u)) & 0xFFu;
    }
    uint v[7];
    for (uint f = 0; f < 6; ++f) {
        v[f] = bytes[2 * f] | (bytes[2 * f + 1] << 8u);
    }
    v[6] = bytes[12] | (bytes[13] << 8u) | (bytes[14] << 16u) | (bytes[15] << 24u);
    const uint4 b = uint4(state[j * 4], state[j * 4 + 1], state[j * 4 + 2], state[j * 4 + 3]);
    if (params.order == kMotionOrder) {
        uint4 g = uint4(v[0] | (v[1] << 16u), v[2] | (v[3] << 16u), v[4] | (v[5] << 16u), v[6]);
        if (!geometryKey(j, params.frame, params.interval, params.stagger)) {
            // The cluster's motion, after the chunk's planes.
            const uint m = params.planesOffset + chunk * params.chunkBytes + kGeometryPlanes * params.chunkSplats +
                           (at / kMotionCluster) * kMotionWords * 4u;
            uint words[kMotionWords];
            for (uint k = 0; k < kMotionWords; ++k) {
                words[k] = page[(m >> 2) + k];
            }
            float r[12];
            for (uint k = 0; k < 12; ++k) {
                r[k] = asfloat(rangesPage[(params.rangesOffset >> 2) + (k < 6 ? k : k + 2u)]);
            }
            g = motionApply(b, motionOf(words), g, MotionRanges(r), params.positionBits, params.scaleBits,
                            params.rotationBits);
        }
        state[j * 4 + 0] = g.x;
        state[j * 4 + 1] = g.y;
        state[j * 4 + 2] = g.z;
        state[j * 4 + 3] = g.w;
        return;
    }
    const uint order = geometryOrder(j, params.frame, params.interval, params.stagger, params.order);
    if (order != 0) {
        const uint at2 = params.state2At + j * 4;
        const uint4 c = uint4(state[at2], state[at2 + 1], state[at2 + 2], state[at2 + 3]);
        for (uint f = 0; f < 6; ++f) {
            v[f] = (predictField(geometryField(b, f), geometryField(c, f), order, 0xFFFFu) + unzigzag16(v[f])) & 0xFFFFu;
        }
        v[6] = rotationFromBits(v[6], predictRotation(b.w, c.w, order), params.rotationBits);
    }
    // The frame before moves down a place: what the next frame's order 2 takes.
    state[params.state2At + j * 4 + 0] = b.x;
    state[params.state2At + j * 4 + 1] = b.y;
    state[params.state2At + j * 4 + 2] = b.z;
    state[params.state2At + j * 4 + 3] = b.w;
    state[j * 4 + 0] = v[0] | (v[1] << 16u);
    state[j * 4 + 1] = v[2] | (v[3] << 16u);
    state[j * 4 + 2] = v[4] | (v[5] << 16u);
    state[j * 4 + 3] = v[6];
}
