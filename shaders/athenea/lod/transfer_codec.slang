// Copyright (c) 2026 jesus luque.
//
// THE TRANSFER THROUGH ITS CODEC AND BACK (`athenea decimate --transfer-basis`):
// clustered PCA (Sloan et al. 2003), the basis fitted by athenea-research
// (Q97) on a sample of the Corvette's transfer. A gaussian's three blocks --
// direct 16, indirect 48 and reflected 48 coefficients, each divided by its
// block's energy norm and laid end to end -- choose the nearest of `C`
// centroids; each block is then `K` weights on that cluster's axes,
// quantised to `levels` steps a side of the cluster's scale, and decoded
// again as the mean plus the axes times the weights. What a file would hold
// is one cluster id and 3 K weights; this writes back what a reader decodes
// from them, so a frame of the cloud shows what the codec costs.
//
// The basis is one array of floats: the centroids [C][112], the three energy
// norms, then for each block its means [C][dim], axes [C][dim][K] and scales
// [C][K].
module transfer_codec;

struct CodecParams {
    uint  count;
    uint  clusters;    ///< C
    uint  components;  ///< K
    float levels;      ///< 127 for 8 bit weights, 7 for 4

    uint  meanAt0;     ///< where each block's means start, in floats
    uint  meanAt1;
    uint  meanAt2;
    uint  normAt;      ///< the three energy norms
};

StructuredBuffer<float>    basis;
RWStructuredBuffer<float>  direct;      // 16 a gaussian
RWStructuredBuffer<float>  indirect;    // 48
RWStructuredBuffer<float>  reflected;   // 48
RWStructuredBuffer<uint>   counters;    // [clipped weights]
ConstantBuffer<CodecParams> params;

static const uint kDims[3] = { 16u, 48u, 48u };
static const uint kJoint = 112u;

float coefficient(uint block, uint g, uint d) {
    return block == 0u ? direct[g * 16u + d] : (block == 1u ? indirect[g * 48u + d] : reflected[g * 48u + d]);
}

void setCoefficient(uint block, uint g, uint d, float v) {
    if (block == 0u) {
        direct[g * 16u + d] = v;
    } else if (block == 1u) {
        indirect[g * 48u + d] = v;
    } else {
        reflected[g * 48u + d] = v;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transferCodec(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    // The nearest centroid in the joint, normalised space.
    float best = 3.4e38;
    uint chosen = 0u;
    for (uint c = 0; c < params.clusters; ++c) {
        float distance = 0.0;
        uint j = 0u;
        for (uint b = 0; b < 3u; ++b) {
            const float norm = max(basis[params.normAt + b], 1.0e-30);
            for (uint d = 0; d < kDims[b]; ++d, ++j) {
                const float e = coefficient(b, g, d) / norm - basis[c * kJoint + j];
                distance += e * e;
            }
        }
        if (distance < best) {
            best = distance;
            chosen = c;
        }
    }
    const uint meanAt[3] = { params.meanAt0, params.meanAt1, params.meanAt2 };
    const uint k = params.components;
    for (uint b = 0; b < 3u; ++b) {
        const uint dims = kDims[b];
        const uint means = meanAt[b] + chosen * dims;
        const uint axes = meanAt[b] + params.clusters * dims + chosen * dims * k;
        const uint scales = meanAt[b] + params.clusters * dims + params.clusters * dims * k + chosen * k;
        float weights[16];
        for (uint a = 0; a < k && a < 16u; ++a) {
            float w = 0.0;
            for (uint d = 0; d < dims; ++d) {
                w += basis[axes + d * k + a] * (coefficient(b, g, d) - basis[means + d]);
            }
            const float scale = max(basis[scales + a], 1.0e-30);
            float q = round(w / scale * params.levels);
            if (abs(q) > params.levels) {
                InterlockedAdd(counters[0], 1u);
                q = clamp(q, -params.levels, params.levels);
            }
            weights[a] = q * scale / params.levels;
        }
        for (uint d = 0; d < dims; ++d) {
            float v = basis[means + d];
            for (uint a = 0; a < k && a < 16u; ++a) {
                v += basis[axes + d * k + a] * weights[a];
            }
            setCoefficient(b, g, d, v);
        }
    }
}
