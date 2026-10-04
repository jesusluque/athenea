// Copyright (c) 2026 jesus luque.
//
// What the path tracer answered at every gaussian, written where the cloud
// keeps it, on the device (`StageRenderer::bakePointsOnDevice` leaves it
// there, a gaussian's entries together).
//
// m2sBakeInto: a radiance bake, `coefficients` float4 a gaussian. The
// constant term is the DC the cloud keeps its colour in -- the bake already
// shifted it to where 3DGS keeps its own -- and the rest are the harmonics,
// rgb per basis, the layout the record declares. A gaussian the bake found
// nothing under is not a black gaussian: its coefficients come back zero, and
// a zero constant term decodes as black, so a disc seen edge on at a
// silhouette would be a black splinter (the pawn's gold ring had forty-four).
// It stands for nothing, so it draws nothing: its opacity goes to zero.
//
// m2sTransferInto: how much of an environment reaches each gaussian, eleven
// float4 a gaussian -- nine coefficients whose rgb is the indirect half and
// whose alpha the direct one, the coverage, and the sixty-four bits of which
// ways out are open carried as the floats they are the bits of. Laid out as
// the file keeps them: nine floats of the direct half, twenty-seven of the
// indirect, two ints of bits.
//
// With `cells` (task TX) the bake kept 256 or 1024 bits over the whole
// sphere, a float4 of four words for every 128 after the coverage, and the
// file keeps them as eight or thirty-two ints; after them sixteen float4 of
// the reflected field, kept as forty-eight floats; and with `coefficients` 16
// the two halves are degree 3, sixteen and forty-eight.
//
// `counts[0]` counts the gaussians the bake found a surface under: the one
// number that crosses back, for the log.
module mesh2splat_bake;

import athenea.usd.transfer_field_fill;

struct BakeParams {
    uint count;          ///< gaussians
    uint coefficients;   ///< float4 a gaussian the radiance bake answered
    uint perRecord;      ///< floats a record
    uint opacity;        ///< the opacity's field
    uint dc0;            ///< the constant term's first field
    uint restBase;       ///< the harmonics' first field
    uint indirect;       ///< the transfer keeps its indirect half
    uint cells;          ///< words a gaussian of a TX grid, 8 or 32, in entries of four after the coverage; 0 the first two
    uint first;          ///< m2sTransferInto: the slice's first gaussian in `records` (the answer and arrays are the slice's)
};

StructuredBuffer<float4>     baked;
RWStructuredBuffer<float>    records;
RWStructuredBuffer<float>    direct;     // `coefficients` a gaussian
RWStructuredBuffer<float>    bounced;    // 3 x `coefficients` a gaussian, where `indirect`
RWStructuredBuffer<int>      shadowBits; // 2 a gaussian, or `cells`
RWStructuredBuffer<float>    reflected;  // 48 a gaussian, with `cells` and `indirect`: the reflected field
RWStructuredBuffer<uint>     counts;
ConstantBuffer<BakeParams>   bake;

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sBakeInto(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= bake.count) {
        return;
    }
    const uint first = k * bake.coefficients;
    const uint at = k * bake.perRecord;
    const float4 constant = baked[first];
    if (!(constant.w > 0.0)) {
        records[at + bake.opacity] = 0.0;
        return;
    }
    records[at + bake.dc0] = constant.x;
    records[at + bake.dc0 + 1] = constant.y;
    records[at + bake.dc0 + 2] = constant.z;
    for (uint c = 1; c < bake.coefficients; ++c) {
        const float4 value = baked[first + c];
        const uint rest = at + bake.restBase + (c - 1) * 3;
        records[rest] = value.x;
        records[rest + 1] = value.y;
        records[rest + 2] = value.z;
    }
    InterlockedAdd(counts[0], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sTransferInto(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= bake.count) {
        return;
    }
    const uint coefficients = bake.coefficients;
    // The coverage, the bits, and with the cells the reflected field's
    // sixteen (technique::transferPlanes).
    const uint entries = coefficients + 1u + (bake.cells != 0 ? bake.cells / 4u + 16u : 1u);
    const uint first = k * entries;
    const bool found = baked[first + coefficients].w > 0.0;
    if (!found) {
        // A transfer of nothing is black under every sky: transparent
        // instead, as the radiance bake does, and its arrays zero.
        records[(bake.first + k) * bake.perRecord + bake.opacity] = 0.0;
    }
    if (bake.cells != 0) {
        for (uint plane = 0; plane < bake.cells / 4u; ++plane) {
            const uint4 words = asuint(baked[first + coefficients + 1 + plane]);
            [unroll] for (uint w = 0; w < 4; ++w) {
                shadowBits[k * bake.cells + plane * 4 + w] = found ? asint(words[w]) : 0;
            }
        }
    } else {
        const float4 bits = baked[first + coefficients + 1];
        shadowBits[k * 2 + 0] = found ? asint(bits.x) : 0;
        shadowBits[k * 2 + 1] = found ? asint(bits.y) : 0;
    }
    // THE REFLECTED FIELD (task TX): sixteen rgb coefficients of what
    // arrives by direction after meeting the scene, under a white sky. Kept
    // only beside the indirect half, which is what couples it to a sky.
    if (bake.cells != 0 && bake.indirect != 0) {
        const uint field = first + coefficients + 1u + bake.cells / 4u;
        float3 radiance[16];
        float  closed[16];
        for (uint c = 0; c < 16u; ++c) {
            const float4 value = baked[field + c];
            radiance[c] = value.xyz;
            closed[c] = value.w;
        }
        if (found) {
            m2sFieldOverClosed(radiance, closed);
        }
        for (uint c = 0; c < 16u; ++c) {
            reflected[k * 48 + c * 3] = found ? radiance[c].x : 0.0;
            reflected[k * 48 + c * 3 + 1] = found ? radiance[c].y : 0.0;
            reflected[k * 48 + c * 3 + 2] = found ? radiance[c].z : 0.0;
        }
    }
    for (uint c = 0; c < coefficients; ++c) {
        const float4 value = baked[first + c];
        direct[k * coefficients + c] = found ? value.w : 0.0;
        if (bake.indirect != 0) {
            const uint into = k * coefficients * 3 + c * 3;
            bounced[into] = found ? value.x : 0.0;
            bounced[into + 1] = found ? value.y : 0.0;
            bounced[into + 2] = found ? value.z : 0.0;
        }
    }
    if (found) {
        InterlockedAdd(counts[0], 1u);
    }
}
