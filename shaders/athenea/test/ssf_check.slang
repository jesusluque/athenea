// Copyright (c) 2026 jesus luque.
//
// THE SOLID LAYER'S WORDS (SSF, common/packing.slang's packSolidBlob and
// packAnchor), made and checked on the device:
//   ssfGenerate   a layer of blobs and anchors from a hash of their index --
//                 every field varied, so a word written to the wrong place or
//                 not at all reads back different
//   ssfPackCheck  each blob and anchor unpacked and packed again: how many
//                 words change (none may: packing is idempotent), and how many
//                 blobs come back further from what was generated than the
//                 packing's step
// What a test reads back are these counters, never the layer.
module ssf_check;
import athenea.common.packing;

struct SsfCheckParams {
    uint blobs;
    uint anchors;
    uint seed;
    uint pad0;
};

RWStructuredBuffer<float4> blobPositions;
RWStructuredBuffer<uint>   blobShape;   ///< 4 words a blob
RWStructuredBuffer<uint>   anchors;     ///< 4 words an anchor
RWStructuredBuffer<uint>   blobNormals; ///< packNormal a blob
RWStructuredBuffer<uint>   wrong;       ///< [0] words changed by a repack, [1] blob fields off, [2] anchor fields off, [3] read
ConstantBuffer<SsfCheckParams> params;

uint pcg(uint v) {
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float unit(uint i, uint field) {
    return float(pcg(pcg(i * 16u + field) ^ params.seed) >> 8) * (1.0 / 16777216.0);
}

SolidBlob generatedBlob(uint i) {
    SolidBlob b;
    b.position = float3(unit(i, 0), unit(i, 1), unit(i, 2)) * 4.0 - 2.0;
    b.weight = 0.5 + 1.5 * unit(i, 3);
    b.radii = exp(lerp(log(1.0e-3), log(1.0e-1), float2(unit(i, 4), unit(i, 5))));
    b.theta = 3.14159265 * unit(i, 6);
    b.colour = float3(unit(i, 7), unit(i, 8), unit(i, 9)) * 1.2;
    b.anchor = params.anchors > 0 ? i % params.anchors : 0;
    return b;
}

SolidAnchor generatedAnchor(uint i) {
    SolidAnchor a;
    a.thickness = exp(lerp(log(1.0e-4), log(1.0e-2), unit(i, 10)));
    a.smoothK = (i & 1u) != 0 ? 0.0 : 0.05 * unit(i, 11);
    a.appearance = (i % 3u) == 0 ? kNoAppearance : pcg(i);
    a.flags = 0;
    return a;
}

uint4 shapeAt(uint i) {
    return uint4(blobShape[i * 4], blobShape[i * 4 + 1], blobShape[i * 4 + 2], blobShape[i * 4 + 3]);
}

uint4 anchorAt(uint i) {
    return uint4(anchors[i * 4], anchors[i * 4 + 1], anchors[i * 4 + 2], anchors[i * 4 + 3]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfGenerate(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i < params.blobs) {
        float4 position;
        uint4  shape;
        packSolidBlob(generatedBlob(i), position, shape);
        blobPositions[i] = position;
        [unroll] for (uint k = 0; k < 4; ++k) {
            blobShape[i * 4 + k] = shape[k];
        }
        blobNormals[i] = packNormal(normalize(float3(unit(i, 12), unit(i, 13), unit(i, 14)) * 2.0 - 1.0 + 1.0e-3));
    }
    if (i < params.anchors) {
        const uint4 w = packAnchor(generatedAnchor(i));
        [unroll] for (uint k = 0; k < 4; ++k) {
            anchors[i * 4 + k] = w[k];
        }
    }
}

uint changed(uint4 a, uint4 b) {
    return uint(a.x != b.x) + uint(a.y != b.y) + uint(a.z != b.z) + uint(a.w != b.w);
}

bool near(float a, float b, float relative) {
    return abs(a - b) <= relative * max(abs(b), 1.0e-6);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfPackCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i < params.blobs) {
        const float4 position = blobPositions[i];
        const uint4  shape = shapeAt(i);
        const SolidBlob b = unpackSolidBlob(position, shape);
        float4 position2;
        uint4  shape2;
        packSolidBlob(b, position2, shape2);
        const uint off = changed(asuint(position), asuint(position2)) + changed(shape, shape2);
        if (off != 0) {
            InterlockedAdd(wrong[0], off);
        }
        // Against what was generated: f16 of a logarithm is a 0.1% step of
        // the radius, f16 of a colour 0.05%, theta's 2^-16 of pi.
        const SolidBlob g = generatedBlob(i);
        const float dTheta = abs(b.theta - g.theta);
        const bool fine = all(b.position == g.position) && b.weight == g.weight && near(b.radii.x, g.radii.x, 2.0e-3) &&
                          near(b.radii.y, g.radii.y, 2.0e-3) && all(abs(b.colour - g.colour) <= 1.0e-3) &&
                          min(dTheta, 3.14159265 - dTheta) <= 1.0e-4 && b.anchor == g.anchor;
        if (!fine) {
            InterlockedAdd(wrong[1], 1u);
        }
        InterlockedAdd(wrong[3], 1u);
    }
    if (i < params.anchors) {
        const uint4 w = anchorAt(i);
        const uint off = changed(w, packAnchor(unpackAnchor(w)));
        if (off != 0) {
            InterlockedAdd(wrong[0], off);
        }
        if (changed(w, packAnchor(generatedAnchor(i))) != 0) {
            InterlockedAdd(wrong[2], 1u);
        }
    }
}
