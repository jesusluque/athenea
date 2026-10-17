// Copyright (c) 2026 jesus luque.
//
// WHERE A NODE CANNOT BE SEEN FROM, for a player to leave it out
// (`athenea decimate --occlusion-mask`; research, 11 Oct). A TX transfer
// keeps 256 bits a gaussian, one a cell of a 16 x 16 octahedral grid over
// the world's sphere, set where a ray from it reached the sky: a direction
// whose cells are all closed is one an eye outside the asset (beyond every
// occluder: outside its box) cannot see the gaussian from. The bits a
// shader reads for shadows will not do for that: a merge holds its leaves'
// bits by majority, and the phone's codebook (shadow_vq) is a majority too
// -- both would close a direction some leaf is seen from.
//
// So each node gets an occlusion mask of its own, open wherever
//   - any leaf under it is open (an OR up the tree), and
//   - a cell next to an open one is (one cell of margin: at 16 x 16 a cell
//     is ~11 degrees, and the edge of an occluder closed directions a
//     grille's chrome was still seen from), and
//   - everywhere on glass (transmission over a half: what stands behind it
//     is seen through it, though its rays stopped on it), and on a leaf
//     with no bits.
// A player culls a node when the four cells it reads a direction from
// (splatCellsOpen's, clamped) are all closed in its mask.
//
// The masks then go into a codebook a phone can keep: `clusters` entries of
// 256 bits and an index a node. It must never close what a node has open:
// an entry is the OR of its members, and after the first round (nearest by
// Hamming distance) a node takes only an entry that holds its own open set,
// the one with the fewest bits open -- the codebook only gets tighter.
module occlusion_mask;
import athenea.common.octahedral;

struct OcclusionParams {
    uint count;      ///< the file's gaussians (a tree's indices are the file's)
    uint words;      ///< 32-bit words of a gaussian's shadow bits (8: 16 x 16 cells)
    uint side;       ///< cells a side (16)
    uint clusters;   ///< codebook entries
    uint hasBits;    ///< 0: no shadow bits at all -- every mask open
    uint hasGlass;   ///< 1: `transmission` is bound
    uint subset;     ///< 1: an assignment must hold the node's open set (after round one)
    uint minBin;     ///< a node whose worth falls in a lower bin (log2) takes entry 0, all open, and is not fitted
    float4 up;       ///< the stage's up axis (the bits' world sphere); w the lowest dot an eye is looked for at
};

StructuredBuffer<uint>   bits;           // file gaussian -> its shadow bits, `words` each
StructuredBuffer<float>  transmission;   // file gaussian -> its transmission
StructuredBuffer<uint>   parent;         // file gaussian -> its parent (itself at a root)
StructuredBuffer<uint>   leaves;         // file gaussian -> the leaves it stands for
RWStructuredBuffer<uint> mask;           // file gaussian -> its occlusion mask, 8 words (open = 1)
RWStructuredBuffer<uint> changed;        // [masks raised this round]
RWStructuredBuffer<uint> centroids;      // the codebook, 8 words an entry
RWStructuredBuffer<uint> next;           // the entries being gathered: the OR of their members
RWStructuredBuffer<uint> members;        // an entry's
RWStructuredBuffer<uint> assigned;       // file gaussian -> its entry
RWStructuredBuffer<uint> lost;           // [cells reopened by the entries; closed; closed where an eye looks from]
RWStructuredBuffer<uint> worthBins;      // 64 bins of log2(worth): how many nodes fall in each
ConstantBuffer<OcclusionParams> params;

static const uint kWords = 8;
static const uint kRoot = 0xFFFFFFFFu;

bool cellOpen(uint g, int2 c) {
    const uint index = uint(c.y) * params.side + uint(c.x);
    return ((bits[g * params.words + (index >> 5u)] >> (index & 31u)) & 1u) != 0u;
}

/// A neighbour across the octahedral square's edge is the mirrored cell
/// along it (the fold of the lower hemisphere): either way more of it is
/// open, never less.
int2 across(int2 c) {
    const int side = int(params.side);
    if (c.x < 0) { c = int2(0, side - 1 - c.y); }
    if (c.x >= side) { c = int2(side - 1, side - 1 - c.y); }
    if (c.y < 0) { c = int2(side - 1 - c.x, 0); }
    if (c.y >= side) { c = int2(side - 1 - c.x, side - 1); }
    return clamp(c, int2(0), int2(side - 1));
}

/// A leaf's mask: its open cells grown by one; all open on glass or with no
/// bits. A merge starts closed and is opened by the climb.
[shader("compute")]
[numthreads(64, 1, 1)]
void occlusionLeaf(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const bool leaf = leaves[g] <= 1u;
    const bool glass = params.hasGlass != 0 && transmission[g] > 0.5;
    uint out[kWords];
    for (uint w = 0; w < kWords; ++w) {
        out[w] = (glass || (leaf && params.hasBits == 0)) ? 0xFFFFFFFFu : 0u;
    }
    if (leaf && !glass && params.hasBits != 0) {
        const int side = int(params.side);
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                bool open = false;
                for (int dy = -1; dy <= 1 && !open; ++dy) {
                    for (int dx = -1; dx <= 1 && !open; ++dx) {
                        open = cellOpen(g, across(int2(x + dx, y + dy)));
                    }
                }
                if (open) {
                    const uint index = uint(y * side + x);
                    out[index >> 5u] |= 1u << (index & 31u);
                }
            }
        }
    }
    for (uint w = 0; w < kWords; ++w) {
        mask[g * kWords + w] = out[w];
    }
}

/// One level of the climb: each node opens its parent where it is open.
/// Run until nothing changes.
[shader("compute")]
[numthreads(256, 1, 1)]
void occlusionUp(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint up = parent[g];
    if (up == g || up == kRoot || up >= params.count) {
        return;
    }
    bool raised = false;
    for (uint w = 0; w < kWords; ++w) {
        const uint own = mask[g * kWords + w];
        uint was;
        InterlockedOr(mask[up * kWords + w], own, was);
        raised = raised || (own & ~was) != 0u;
    }
    if (raised) {
        InterlockedAdd(changed[0], 1u);
    }
}

// WHAT A NODE'S CULLING IS WORTH: the cells an eye looks from it is closed
// in. A hood's nodes, closed toward the ground
// an eye never stands in, are many and worth little; a grille's are few and
// worth half the culling -- an entry spent on each node alike gave the
// grille 64 % of what its own masks cull (research). Entry 0 is all open:
// the nodes in the bins worth least, together under a small share of the
// whole, take it and leave the rest of the codebook to the others.
// WHERE AN EYE STANDS: above the ground, so a cell well below the horizon
// (a hood closed toward the engine under it) is one no camera looks from.
// Such cells are still kept conservatively -- an entry holds a node's open
// set over every cell -- but they buy nothing, so the worth, the first
// round's distance and the choice of the tightest entry count only the
// cells an eye may look from.
uint relevantWord(uint w) {
    uint word = 0u;
    for (uint b = 0; b < 32u; ++b) {
        const uint index = w * 32u + b;
        const uint x = index % params.side, y = index / params.side;
        const float3 d = octDecode((float2(float(x), float(y)) + 0.5) / float(params.side));
        if (dot(d, params.up.xyz) > params.up.w) {
            word |= 1u << b;
        }
    }
    return word;
}

uint closedCells(uint g) {
    uint closed = 0u;
    for (uint w = 0; w < kWords; ++w) {
        closed += countbits(~mask[g * kWords + w] & relevantWord(w));
    }
    return closed;
}

uint worthBin(uint g) {
    // A node in a cut is one gaussian drawn however many leaves it stands
    // for: the leaves weighed a fender's large merges over a grille's many
    // small bars (research: the grille at 69 % of its culling).
    const float worth = float(closedCells(g));
    return worth >= 1.0 ? min(uint(log2(worth)) + 1u, 63u) : 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void occlusionWorth(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g < params.count) {
        InterlockedAdd(worthBins[worthBin(g)], 1u);
    }
}

bool fitted(uint g) {
    return worthBin(g) >= params.minBin;
}

/// The codebook's start: entry 0 all open, the rest masks of fitted nodes
/// spread through the file.
[shader("compute")]
[numthreads(64, 1, 1)]
void occlusionSeed(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.clusters) {
        return;
    }
    uint g = uint((uint64_t(c) * params.count) / max(params.clusters, 1u));
    for (uint step = 0; step < 4096u && g + 1u < params.count && !fitted(g); ++step) {
        ++g;
    }
    for (uint w = 0; w < kWords; ++w) {
        centroids[c * kWords + w] = c == 0u ? 0xFFFFFFFFu : mask[g * kWords + w];
        next[c * kWords + w] = 0u;
    }
    members[c] = 0u;
}

/// Each node's entry: the nearest (round one), or the tightest that holds
/// its open set; its mask ORed into that entry's next value.
[shader("compute")]
[numthreads(64, 1, 1)]
void occlusionAssign(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    uint own[kWords];
    uint seen[kWords];
    for (uint w = 0; w < kWords; ++w) {
        own[w] = mask[g * kWords + w];
        seen[w] = relevantWord(w);
    }
    if (!fitted(g)) {
        assigned[g] = 0u;
        return;
    }
    uint best = 0xFFFFFFFFu;
    uint chosen = params.subset != 0 ? assigned[g] : 0u;
    for (uint c = 1; c < params.clusters; ++c) {
        uint d = 0u;
        bool holds = true;
        for (uint w = 0; w < kWords; ++w) {
            const uint e = centroids[c * kWords + w];
            if (params.subset != 0) {
                holds = holds && (own[w] & ~e) == 0u;
                d += countbits(e & seen[w]);
            } else {
                d += countbits((own[w] ^ e) & seen[w]);
            }
        }
        if (holds && d < best) {
            best = d;
            chosen = c;
        }
    }
    assigned[g] = chosen;
    InterlockedAdd(members[chosen], 1u);
    for (uint w = 0; w < kWords; ++w) {
        InterlockedOr(next[chosen * kWords + w], own[w]);
    }
}

/// Each entry becomes the OR of its members (one nobody chose keeps what it
/// had); the gathering cleared for the next round.
[shader("compute")]
[numthreads(64, 1, 1)]
void occlusionUpdate(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.clusters) {
        return;
    }
    if (members[c] > 0u && c != 0u) {
        for (uint w = 0; w < kWords; ++w) {
            centroids[c * kWords + w] = next[c * kWords + w];
        }
    }
    for (uint w = 0; w < kWords; ++w) {
        next[c * kWords + w] = 0u;
    }
    members[c] = 0u;
}

/// What the codebook costs: the cells a node's own mask closes and its entry
/// opens, summed over the nodes that close any.
[shader("compute")]
[numthreads(64, 1, 1)]
void occlusionLost(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint c = assigned[g];
    uint gone = 0u, closed = 0u, seen = 0u;
    for (uint w = 0; w < kWords; ++w) {
        const uint own = mask[g * kWords + w];
        gone += countbits(centroids[c * kWords + w] & ~own);
        closed += countbits(~own);
        seen += countbits(~own & relevantWord(w));
    }
    if (closed > 0u) {
        InterlockedAdd(lost[0], gone);
        InterlockedAdd(lost[1], closed);
    }
    InterlockedAdd(lost[2], seen);
}
