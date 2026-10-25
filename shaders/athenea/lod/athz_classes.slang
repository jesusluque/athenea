// Copyright (c) 2026 jesus luque.
//
// AN IMAGE AGAINST ITS REFERENCE, CLASS BY CLASS (`athenea athz-compare`):
// the pixels a class map gives to each of three classes -- the map's
// dominant channel, red, green or blue, where it is at least three times
// either other and above a floor -- measured as glass is measured here: the
// mean luminance of each against the reference's, and the relMSE of the two
// smoothed (a 9 x 9 gaussian, sigma 2) after the image's mean is matched to
// the reference's, so a lens image shifted by a pixel and the reference's own
// noise do not reward a darker image.
//
//   athzClassSums     per workgroup: each class's pixels, sum of image and
//                     reference luminance
//   athzClassReduce   the workgroups' partial sums, one thread a sum
//   athzClassError    per workgroup: each class's sum of the smoothed,
//                     matched squared relative error
module athz_classes;

struct ClassParams {
    uint  width;
    uint  height;
    uint  groups;     ///< workgroups of the pass reduced
    uint  sums;       ///< sums a workgroup writes
    float scale1;     ///< athzClassError: each class's reference-to-image mean ratio
    float scale2;
    float scale3;
    float pad;
};
ConstantBuffer<ClassParams> params;

StructuredBuffer<float4>  image;
StructuredBuffer<float4>  reference;
StructuredBuffer<float4>  classes;
RWStructuredBuffer<float> partial;   ///< groups * sums
RWStructuredBuffer<float> totals;    ///< sums

static const uint kClasses = 4;      // 0: none
groupshared float gSum[256 * 4];

float luminance(float4 c) {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

uint classAt(uint i) {
    const float3 c = classes[i].xyz;
    const float m = max(c.x, max(c.y, c.z));
    if (m < 0.02) {
        return 0u;
    }
    if (c.x == m && c.x > 3.0 * max(c.y, c.z)) return 1u;
    if (c.y == m && c.y > 3.0 * max(c.x, c.z)) return 2u;
    if (c.z == m && c.z > 3.0 * max(c.x, c.y)) return 3u;
    return 0u;
}

/// The groupshared tree over 256 threads of `v` into `partial[at]`.
void reduce(uint local, uint group, uint at, float v) {
    gSum[local] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (local < stride) {
            gSum[local] += gSum[local + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (local == 0u) {
        partial[group * params.sums + at] = gSum[0];
    }
    GroupMemoryBarrierWithGroupSync();
}

[shader("compute")]
[numthreads(256, 1, 1)]
void athzClassSums(uint3 tid: SV_DispatchThreadID, uint3 local: SV_GroupThreadID, uint3 group: SV_GroupID) {
    const uint i = tid.x;
    const bool inside = i < params.width * params.height;
    const uint c = inside ? classAt(i) : 0u;
    for (uint k = 1; k < kClasses; ++k) {
        const bool mine = inside && c == k;
        reduce(local.x, group.x, (k - 1u) * 3u + 0u, mine ? 1.0 : 0.0);
        reduce(local.x, group.x, (k - 1u) * 3u + 1u, mine ? luminance(image[i]) : 0.0);
        reduce(local.x, group.x, (k - 1u) * 3u + 2u, mine ? luminance(reference[i]) : 0.0);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void athzClassReduce(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.sums) {
        return;
    }
    float s = 0.0;
    for (uint g = 0; g < params.groups; ++g) {
        s += partial[g * params.sums + tid.x];
    }
    totals[tid.x] = s;
}

float smoothed(StructuredBuffer<float4> from, int x, int y) {
    float s = 0.0, w = 0.0;
    for (int dy = -4; dy <= 4; ++dy) {
        for (int dx = -4; dx <= 4; ++dx) {
            const int px = clamp(x + dx, 0, int(params.width) - 1);
            const int py = clamp(y + dy, 0, int(params.height) - 1);
            const float k = exp(-float(dx * dx + dy * dy) / 8.0);   // sigma 2
            s += k * luminance(from[uint(py) * params.width + uint(px)]);
            w += k;
        }
    }
    return s / w;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void athzClassError(uint3 tid: SV_DispatchThreadID, uint3 local: SV_GroupThreadID, uint3 group: SV_GroupID) {
    const uint i = tid.x;
    const bool inside = i < params.width * params.height;
    const uint c = inside ? classAt(i) : 0u;
    float e = 0.0;
    if (inside && c != 0u) {
        const int x = int(i % params.width), y = int(i / params.width);
        const float scale = c == 1u ? params.scale1 : (c == 2u ? params.scale2 : params.scale3);
        const float a = smoothed(image, x, y) * scale;
        const float b = smoothed(reference, x, y);
        e = (a - b) * (a - b) / (b * b + 1.0e-2);
    }
    for (uint k = 1; k < kClasses; ++k) {
        reduce(local.x, group.x, k - 1u, c == k ? e : 0.0);
    }
}
