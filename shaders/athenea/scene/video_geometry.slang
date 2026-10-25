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
    uint pad1;
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
RWStructuredBuffer<uint>         state;      // decode: frame splat j's four words, kept from frame to frame; from
                                             // `state2At` on, the frame before's (order 2)
ConstantBuffer<GeometryParams>   params;


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
    const uint4 g = uint4(now[j * 4], now[j * 4 + 1], now[j * 4 + 2], now[j * 4 + 3]);
    const uint f = plane < 12 ? plane / 2u : 6u;
    uint v = geometryField(g, f);
    const uint order = geometryOrder(j, params.frame, params.interval, params.stagger, params.order);
    if (order != 0) {
        const uint4 b = uint4(before[j * 4], before[j * 4 + 1], before[j * 4 + 2], before[j * 4 + 3]);
        const uint4 c = uint4(before2[j * 4], before2[j * 4 + 1], before2[j * 4 + 2], before2[j * 4 + 3]);
        v = f == 6 ? rotationResidualBits(v, predictRotation(b.w, c.w, order), params.rotationBits)
                   : zigzag16(v - predictField(geometryField(b, f), geometryField(c, f), order, 0xFFFFu));
    }
    const uint shift = plane < 12 ? (plane & 1u) * 8u : (plane - 12u) * 8u;
    return (v >> shift) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoGeometryPack(uint3 tid: SV_DispatchThreadID) {
    const uint perPlane = params.chunkSplats / 4u;
    const uint chunks = (params.count + params.chunkSplats - 1) / params.chunkSplats;
    const uint w = tid.x % perPlane, plane = (tid.x / perPlane) % kGeometryPlanes;
    const uint chunk = tid.x / (perPlane * kGeometryPlanes);
    if (chunk >= chunks) {
        return;
    }
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
        const uint e = (params.planesOffset) + (chunk * kGeometryPlanes + plane) * params.chunkSplats + at;
        bytes[plane] = (page[e >> 2] >> ((e & 3u) * 8u)) & 0xFFu;
    }
    uint v[7];
    for (uint f = 0; f < 6; ++f) {
        v[f] = bytes[2 * f] | (bytes[2 * f + 1] << 8u);
    }
    v[6] = bytes[12] | (bytes[13] << 8u) | (bytes[14] << 16u) | (bytes[15] << 24u);
    const uint4 b = uint4(state[j * 4], state[j * 4 + 1], state[j * 4 + 2], state[j * 4 + 3]);
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
