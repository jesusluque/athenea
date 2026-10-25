// Copyright (c) 2026 jesus luque.
//
// THE TRANSFER AS CLUSTERED PCA WEIGHTS, MEASURED (`athenea athz-census
// --transfer-basis`; E1b): what transfer_codec.slang would keep of a TX
// gaussian -- the nearest of C centroids and, for each of the three blocks
// (direct 16, indirect 48, field 48), K weights quantised to `levels` steps a
// side -- and what those weights cost at order 0, as they are and as a
// residual from the parent group's (the finest merged level's, which is the
// weighted mean of its children's transfer).
//
//   athzCpca      each element's id and 3 K weights (q + levels), into `codes`
//   athzCpcaHist  the splats' histograms: ids, and ids where the parent's
//                 differs; whether it differs; each weight as it is; each
//                 weight as zigzag(q - parent's q) where the ids agree, and as
//                 it is where they do not
//   athzCpcaBits  each group of 256 counts at its entropy, in bytes
module athz_cpca;

struct CpcaParams {
    uint  count;
    uint  clusters;
    uint  components;
    float levels;
    uint  meanAt0;
    uint  meanAt1;
    uint  meanAt2;
    uint  normAt;
    uint  transferWords;
    uint  groups;          ///< athzCpcaBits
    uint  codeWords;       ///< 1 + 3 K
    uint  parentBase;      ///< athzCpcaHist: where the parents' codes start
};
ConstantBuffer<CpcaParams> params;

StructuredBuffer<float>      basis;
ByteAddressBuffer            transfer;    ///< f16 pairs, transferWords an element
ByteAddressBuffer            groupOf;     ///< athzCpcaHist: a splat's finest group
RWStructuredBuffer<uint>     codes;       ///< codeWords an element
RWStructuredBuffer<uint>     histogram;
RWStructuredBuffer<uint>     bytes;

static const uint kDims[3] = { 16u, 48u, 48u };
static const uint kJoint = 112u;

float coefficient(uint g, uint j) {
    const uint word = transfer.Load((g * params.transferWords + j / 2u) * 4u);
    return f16tof32((j & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void athzCpca(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    float best = 3.4e38;
    uint chosen = 0u;
    for (uint c = 0; c < params.clusters; ++c) {
        float distance = 0.0;
        uint j = 0u;
        for (uint b = 0; b < 3u; ++b) {
            const float norm = max(basis[params.normAt + b], 1.0e-30);
            for (uint d = 0; d < kDims[b]; ++d, ++j) {
                const float e = coefficient(g, j) / norm - basis[c * kJoint + j];
                distance += e * e;
            }
        }
        if (distance < best) {
            best = distance;
            chosen = c;
        }
    }
    codes[g * params.codeWords] = chosen;
    const uint meanAt[3] = { params.meanAt0, params.meanAt1, params.meanAt2 };
    const uint k = params.components;
    uint at = 0u;
    for (uint b = 0; b < 3u; ++b) {
        const uint dims = kDims[b];
        const uint means = meanAt[b] + chosen * dims;
        const uint axes = meanAt[b] + params.clusters * dims + chosen * dims * k;
        const uint scales = meanAt[b] + params.clusters * dims + params.clusters * dims * k + chosen * k;
        for (uint a = 0; a < k; ++a) {
            float w = 0.0;
            for (uint d = 0; d < dims; ++d) {
                w += basis[axes + d * k + a] * (coefficient(g, at + d) - basis[means + d]);
            }
            const float scale = max(basis[scales + a], 1.0e-30);
            const float q = clamp(round(w / scale * params.levels), -params.levels, params.levels);
            codes[g * params.codeWords + 1u + b * k + a] = uint(q + params.levels);
        }
        at += dims;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void athzCpcaHist(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint weights = params.codeWords - 1u;
    const uint parent = params.parentBase + groupOf.Load(g * 4u);
    const uint id = codes[g * params.codeWords];
    const uint parentId = codes[parent * params.codeWords];
    const bool same = id == parentId;
    InterlockedAdd(histogram[id], 1u);                         // group 0: ids
    if (!same) {
        InterlockedAdd(histogram[256u + id], 1u);              // group 1: ids where the parent's differs
    }
    InterlockedAdd(histogram[512u + (same ? 0u : 1u)], 1u);    // group 2: whether it differs
    for (uint a = 0; a < weights; ++a) {
        const uint q = codes[g * params.codeWords + 1u + a];
        InterlockedAdd(histogram[(3u + a) * 256u + q], 1u);    // groups 3..: each weight
        uint r = q;
        if (same) {
            const int d = int(q) - int(codes[parent * params.codeWords + 1u + a]);
            r = uint(d >= 0 ? 2 * d : -2 * d - 1);
            InterlockedAdd(histogram[(3u + weights + a) * 256u + r], 1u);         // residuals, ids agreeing
        } else {
            InterlockedAdd(histogram[(3u + 2u * weights + a) * 256u + r], 1u);   // as they are, ids not
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void athzCpcaBits(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.groups) {
        return;
    }
    const uint at = tid.x * 256u;
    uint total = 0u;
    for (uint s = 0; s < 256u; ++s) {
        total += histogram[at + s];
    }
    float bits = 0.0;
    for (uint s = 0; s < 256u; ++s) {
        const uint c = histogram[at + s];
        if (c != 0u) {
            bits += float(c) * log2(float(total) / float(c));
        }
    }
    bytes[tid.x] = uint(ceil(bits / 8.0));
}
