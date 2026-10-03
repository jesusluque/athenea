// Copyright (c) 2026 jesus luque.
//
// THE TWO SIDES OF ONE BASIS, TERM FOR TERM.
//
// `sh.slang` holds the harmonics twice: `evaluateRest`, which a renderer uses
// to add the view dependent part of a cloud's colour, and `shBasisValue`,
// which a bake and a transfer use to project onto the same functions. The
// file says they must agree, and until this nothing checked that they do --
// a sign or an order wrong in one of them would bake a cloud that decodes to
// something else, and no test of either side alone can see it.
//
// The check is the only one there is: set one coefficient to one and every
// other to nothing, and what `evaluateRest` reads along a direction has to be
// what `shBasisValue` answers for that term.
module sh_basis_check;
import athenea.common.sh;

struct ShBasisParams {
    uint  terms;        ///< rest coefficients checked, 15 at degree 3
    uint  rowWords;     ///< words a row: 15 coefficients, rgb, two halves a word
    uint  directions;   ///< directions each term is checked along
    float tolerance;    ///< absolute, on a basis of order one
};

RWStructuredBuffer<uint>   rows;    ///< written by the first entry, read by the second
StructuredBuffer<uint>     sh;      ///< the same buffer, bound to read
RWStructuredBuffer<uint>   stats;   ///< 8 wide
RWStructuredBuffer<float>  worst;   ///< 8 wide
ConstantBuffer<ShBasisParams> params;

/// Directions with no pole to crowd, as the environment check uses.
float3 spiralDirection(uint i, uint n) {
    const float golden = 2.39996322972865332;
    const float z = 1.0 - 2.0 * (float(i) + 0.5) / float(max(n, 1u));
    const float r = sqrt(max(1.0 - z * z, 0.0));
    const float phi = golden * float(i);
    return float3(r * cos(phi), r * sin(phi), z);
}

/// One row a term: every half zero but the three of that term, which are one.
/// A thread owns its whole row, so no two threads share a word.
[shader("compute")]
[numthreads(16, 1, 1)]
void shBasisWrite(uint3 tid: SV_DispatchThreadID) {
    const uint term = tid.x;
    if (term >= params.terms) {
        return;
    }
    const uint row = term * params.rowWords;
    for (uint w = 0; w < params.rowWords; ++w) {
        rows[row + w] = 0u;
    }
    // The halves this term occupies: rgb, at `term * 3`.
    for (uint c = 0; c < 3; ++c) {
        const uint index = term * 3 + c;
        const uint word = row + index / 2;
        const uint one = f32tof16(1.0);
        rows[word] = (index & 1u) == 0u ? (rows[word] & 0xFFFF0000u) | one
                                        : (rows[word] & 0x0000FFFFu) | (one << 16);
    }
}

/// stats[0] readings beyond the tolerance
/// stats[1] readings taken
/// stats[2] the worst deviation, scaled by 1e6
/// worst[0] that deviation; worst[1] the first term's first reading
[shader("compute")]
[numthreads(16, 1, 1)]
void shBasisCheck(uint3 tid: SV_DispatchThreadID) {
    const uint term = tid.x;
    if (term >= params.terms) {
        return;
    }
    const uint row = term * params.rowWords;
    float worstHere = 0.0;
    for (uint i = 0; i < params.directions; ++i) {
        const float3 dir = spiralDirection(i, params.directions);
        // Degree 3 reads every term; a coefficient that is zero adds nothing.
        const float3 read = evaluateRest(sh, row, 3, dir);
        const float want = shBasisValue(term + 1, dir);
        const float got = (read.r + read.g + read.b) / 3.0;
        const float off = abs(got - want);
        worstHere = max(worstHere, off);
        if (off > params.tolerance) {
            InterlockedAdd(stats[0], 1u);
        }
        // What the largest reading was worth, so a run that compared nothing
        // but zeros cannot pass for a run that compared the basis.
        InterlockedMax(stats[3], uint(abs(want) * 1.0e6));
        InterlockedAdd(stats[1], 1u);
        if (term == 0 && i == 0) {
            worst[1] = got;
            worst[2] = want;
        }
    }
    InterlockedMax(stats[2], uint(worstHere * 1.0e6));
}
