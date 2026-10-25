// Copyright (c) 2026 jesus luque.
//
// THE athz ENCODER (athz_common.slang says what is coded and how). Five
// passes over a list of pages -- each page `pageGaussians` elements of one
// kind, a merged level's groups or the cloud's own splats, from `base`:
//
//   athzHistogram   per plane, how often each byte value occurs: a workgroup
//                   a (page, plane), its counts summed in workgroup memory
//                   and added once. Sums: the same whatever order the pages
//                   come in.
//   athzNormalize   per plane: its range (the raw width) and, where rANS
//                   pays, its frequencies scaled to 2^tableLog, every symbol
//                   seen kept at one at least.
//   athzEncodeRuns  a workgroup of 32 lanes a run and a layer: the run's
//                   symbols of that layer encoded backwards, the stream of
//                   words written from its end; the raw planes packed.
//   athzPlace       each run's stream at its place in its page (the offsets a
//                   prefix sum gave), the page's table of run ends first.
//   athzPageInfo    per page: the box of its positions in f16 rounded
//                   outwards, the error of its level, the page it depends on.
module athz_encode;
import athz_common;

struct AthzParams {
    uint  shWords;
    uint  normals;
    uint  emission;
    uint  pbrWords;
    uint  lobesWords;
    uint  transferWords;
    uint  shadowWords;
    uint  words;          ///< an element's words
    uint  pages;          ///< in the list
    uint  pageRuns;
    uint  runGaussians;
    uint  tableLog;
    uint  layer;          ///< athzEncodeRuns, athzPlace: the layer coded
    uint  hasRans;        ///< whether that layer has a plane of encoding 1
    uint  rawWords;       ///< its raw words a run
    uint  capWords;       ///< its rANS symbols a run: the most words a run's stream can take
    uint  planes;         ///< 4 * words
    uint  allowRans;      ///< 0: every plane raw
    uint  levels;         ///< athzPageInfo: entries of `levelInfo`
    uint  leaves;         ///< athzPageInfo: 1 when the pages are the cloud's own splats
    float extent;         ///< the cloud's largest edge
    uint  transform;      ///< athzHistogram: the one every word takes (0..6), or 0xFF each plane's own
    uint  tailWord;       ///< the tail word's index in an element
    uint  allowed;        ///< athzChoose: a bit a transform it may take
    uint  elements;       ///< athzParents: elements of the kind
    uint  groups;         ///< athzEntropy: groups of 256 counts
    uint  curvatureWords;
    uint  pad1;
    uint  pad2;
};
ConstantBuffer<AthzParams> params;

/// Per page: x = first element, y = elements, z = page index in the file, w = band.
StructuredBuffer<uint4> pageList;
StructuredBuffer<uint4> planes;


/// The kind's elements interleaved, `words` an element (athz_interleave), and
/// the merged levels' end to end, coarsest first: every element's parent is
/// one of those.
ByteAddressBuffer records;
ByteAddressBuffer parentRecords;
/// Per element of the kind: its parent among the merged arrays, or ~0.
RWStructuredBuffer<uint> parentOf;

RWStructuredBuffer<uint> histogram;   ///< planes * 256 (athzContext: planes * 2 * 4 * 256)
RWStructuredBuffer<uint> planesOut;   ///< athzNormalize: planes * 4 (a uint4 each)
RWStructuredBuffer<uint> freqs;       ///< planes * 256: freq | cum << 16
RWStructuredBuffer<uint> scratch;     ///< runs * capWords: a 16-bit word in each
RWStructuredBuffer<uint> rawScratch;  ///< runs * rawWords
RWStructuredBuffer<uint> states;      ///< runs * 32
RWStructuredBuffer<uint> runWords;    ///< runs: words of the stream
RWStructuredBuffer<uint> runBytes;    ///< runs: bytes the run takes, its page's table on its first
StructuredBuffer<uint>   runOffsets;  ///< runs: where each starts in the layer's output (prefix of runBytes)
RWByteAddressBuffer      output;
RWStructuredBuffer<uint> pageInfo;    ///< pages * 8
/// athzPageInfo: per level of the list, x = first element, y = groups,
/// z = first page, w = octree level.
StructuredBuffer<uint4>  levelInfo;

AthzLayout layout() {
    AthzLayout l;
    l.shWords = params.shWords;
    l.normals = params.normals;
    l.emission = params.emission;
    l.pbrWords = params.pbrWords;
    l.lobesWords = params.lobesWords;
    l.transferWords = params.transferWords;
    l.shadowWords = params.shadowWords;
    l.curvatureWords = params.curvatureWords;
    return l;
}

uint loadWord(uint e, uint w) {
    return records.Load((e * params.words + w) * 4u);
}

uint loadParent(uint e, uint w) {
    return parentRecords.Load((e * params.words + w) * 4u);
}

/// Word `w` of element `e` -- the `g`th of its run -- as transform `t` codes it.
uint residualWord(uint e, uint g, uint w, uint t) {
    const uint v = loadWord(e, w);
    if (t == kXfNone) {
        return v;
    }
    if (t == kXfWithin || t == kXfParentWithin) {
        const uint pw = athzPreviousWord(w, params.tailWord);
        uint prev = loadWord(e, pw);
        uint here = v;
        if (t == kXfParentWithin) {
            const uint parent = parentOf[e];
            if (parent != 0xFFFFFFFFu) {
                here = sub16x2(v, loadParent(parent, w));
                prev = sub16x2(prev, loadParent(parent, pw));
            }
        }
        return athzWithin(here, prev);
    }
    uint p = 0u;
    if (t <= kXfXor) {
        p = g > 0u ? loadWord(e - 1u, w) : 0u;
    } else {
        const uint parent = parentOf[e];
        p = parent != 0xFFFFFFFFu ? loadParent(parent, w) : 0u;
    }
    return athzForward(t, v, p);
}

uint transformOf(uint b) {
    return params.transform != 0xFFu ? params.transform : athzTransform(planes[b]);
}

uint loadByte(uint e, uint g, uint b) {
    return (residualWord(e, g, b >> 2, transformOf(b)) >> ((b & 3u) * 8u)) & 0xFFu;
}

// --- histograms ---------------------------------------------------------------

groupshared uint gBins[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void athzHistogram(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    const uint page = group.x;
    const uint plane = group.y;
    gBins[local.x] = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (page < params.pages && plane < params.planes) {
        const uint4 p = pageList[page];
        for (uint g = local.x; g < p.y; g += 256u) {
            InterlockedAdd(gBins[loadByte(p.x + g, g % params.runGaussians, plane)], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (plane < params.planes && gBins[local.x] != 0u) {
        InterlockedAdd(histogram[plane * 256u + local.x], gBins[local.x]);
    }
}

// --- parents, choices, contexts (E1b) ------------------------------------------

/// Each element's parent: a splat's is its finest merged group (the tail
/// word, from the finest level's start); a merged group's the group at the
/// level above whose cell holds its own (`levelInfo` the levels, `pageList`'s
/// w its level). The coarsest level's: none.
[shader("compute")]
[numthreads(64, 1, 1)]
void athzParents(uint3 tid: SV_DispatchThreadID) {
    const uint page = tid.y;
    const uint g = tid.x;
    if (page >= params.pages) {
        return;
    }
    const uint4 p = pageList[page];
    if (g >= p.y) {
        return;
    }
    const uint e = p.x + g;
    uint parent = 0xFFFFFFFFu;
    if (params.leaves != 0u) {
        const uint4 finest = levelInfo[params.levels - 1u];
        parent = finest.x + min(loadWord(e, params.tailWord), finest.y - 1u);
    } else if (p.w > 0u) {
        const uint4 up = levelInfo[p.w - 1u];
        const uint cell = loadWord(e, params.tailWord) >> 3;
        uint a = 0u, b = up.y;
        while (a < b) {
            const uint m = (a + b) / 2u;
            if (loadWord(up.x + m, params.tailWord) < cell) {
                a = m + 1u;
            } else {
                b = m;
            }
        }
        parent = up.x + min(a, up.y - 1u);
    }
    parentOf[e] = parent;
}

/// Bits a group of 256 counts takes at its own entropy.
float entropyBits(uint at) {
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
    return bits;
}

/// What each group of 256 counts takes at its entropy, in bytes (`freqs`).
[shader("compute")]
[numthreads(64, 1, 1)]
void athzEntropy(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.groups) {
        return;
    }
    freqs[tid.x] = uint(ceil(entropyBits(tid.x * 256u) / 8.0));
}

/// Each word's transform: of those `allowed`, the one whose four planes'
/// histograms (`histogram`: kXfCount sets of planes * 256, a transform each)
/// take least, each plane at the lesser of its entropy and its raw width.
/// Written into `planesOut`'s w of the word's planes.
[shader("compute")]
[numthreads(64, 1, 1)]
void athzChoose(uint3 tid: SV_DispatchThreadID) {
    const uint w = tid.x;
    if (w >= params.words) {
        return;
    }
    uint best = kXfNone;
    float least = 3.4e38;
    for (uint t = 0; t < kXfCount; ++t) {
        // What the word may take: the asset's choice, and what keeps a byte
        // from depending on a layer a reader may leave out (the word's first
        // plane's z, set by the writer).
        const uint mayTake = params.allowed & planes[w * 4u].z;
        if (((mayTake >> t) & 1u) == 0u || (w == params.tailWord && t >= kXfParent)) {
            continue;
        }
        float bits = 0.0;
        for (uint b = w * 4u; b < w * 4u + 4u; ++b) {
            const uint at = (t * params.planes + b) * 256u;
            uint total = 0u, lo = 255u, hi = 0u;
            for (uint s = 0; s < 256u; ++s) {
                const uint c = histogram[at + s];
                total += c;
                if (c != 0u) {
                    lo = min(lo, s);
                    hi = max(hi, s);
                }
            }
            uint width = 0u;
            while (hi > lo && ((hi - lo) >> width) != 0u) {
                ++width;
            }
            bits += min(entropyBits(at), float(total) * float(width));
        }
        if (bits < least * 0.999) {   // a transform must earn its place
            least = bits;
            best = t;
        }
    }
    for (uint b = w * 4u; b < w * 4u + 4u; ++b) {
        planesOut[b * 4u + 3u] = best;
    }
}

/// ORDER 1, MEASURED: each plane's residual symbols counted under two
/// contexts, each the top two bits of a symbol the decoder has already --
/// the same plane's previous element in the run (set 0), the word's previous
/// byte of the same element (set 1; a word's first byte: context 0).
[shader("compute")]
[numthreads(256, 1, 1)]
void athzContext(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    const uint page = group.x;
    const uint plane = group.y;
    if (page >= params.pages || plane >= params.planes) {
        return;
    }
    const uint4 p = pageList[page];
    for (uint g = local.x; g < p.y; g += 256u) {
        const uint inRun = g % params.runGaussians;
        const uint s = loadByte(p.x + g, inRun, plane);
        const uint prevElem = inRun > 0u ? loadByte(p.x + g - 1u, inRun - 1u, plane) >> 6 : 0u;
        const uint prevByte = (plane & 3u) != 0u ? loadByte(p.x + g, inRun, plane - 1u) >> 6 : 0u;
        InterlockedAdd(histogram[((plane * 2u + 0u) * 4u + prevElem) * 256u + s], 1u);
        InterlockedAdd(histogram[((plane * 2u + 1u) * 4u + prevByte) * 256u + s], 1u);
    }
}

// --- tables ------------------------------------------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void athzNormalize(uint3 tid: SV_DispatchThreadID) {
    const uint plane = tid.x;
    if (plane >= params.planes) {
        return;
    }
    const uint at = plane * 256u;
    uint total = 0u, symbols = 0u, lo = 255u, hi = 0u, most = 0u;
    for (uint s = 0; s < 256u; ++s) {
        const uint c = histogram[at + s];
        total += c;
        if (c != 0u) {
            symbols += 1u;
            lo = min(lo, s);
            hi = max(hi, s);
            most = c > histogram[at + most] ? s : most;
        }
    }
    uint width = 0u;
    if (symbols > 1u) {
        const uint span = hi - lo;
        while ((span >> width) != 0u) {
            ++width;
        }
    }
    const uint base = symbols > 0u ? lo : 0u;
    const uint layer = athzLayer(planes[plane]);
    uint encoding = 0u;
    uint estimate = 0u;
    for (uint s = 0; s < 256u; ++s) {
        freqs[at + s] = 0u;
    }
    if (params.allowRans != 0u && symbols > 1u) {
        const uint scale = 1u << params.tableLog;
        uint sum = 0u;
        uint f[256];
        for (uint s = 0; s < 256u; ++s) {
            const uint c = histogram[at + s];
            f[s] = c == 0u ? 0u : max(1u, uint((uint64_t(c) * uint64_t(scale)) / uint64_t(total)));
            sum += f[s];
        }
        if (sum < scale) {
            f[most] += scale - sum;
            sum = scale;
        }
        while (sum > scale) {
            uint top = 0u;
            for (uint s = 1; s < 256u; ++s) {
                top = f[s] > f[top] ? s : top;
            }
            const uint cut = min(f[top] - 1u, sum - scale);
            f[top] -= cut;
            sum -= cut;
        }
        // What it costs, against the raw width: the frequencies' entropy.
        float bits = 0.0;
        uint cum = 0u;
        for (uint s = 0; s < 256u; ++s) {
            const uint c = histogram[at + s];
            if (c != 0u) {
                bits += float(c) * (float(params.tableLog) - log2(float(f[s])));
            }
            freqs[at + s] = f[s] | (cum << 16);
            cum += f[s];
        }
        estimate = uint(ceil(bits / 8.0));
        const float raw = float(total) * float(width) / 8.0;
        encoding = bits / 8.0 < raw * 0.97 ? 1u : 0u;
    }
    planesOut[plane * 4u + 0u] = encoding | (width << 8) | (base << 16) | (layer << 24);
    planesOut[plane * 4u + 1u] = kAthzNoTable;
    planesOut[plane * 4u + 2u] = 0u;
    planesOut[plane * 4u + 3u] = estimate;
}

// --- runs --------------------------------------------------------------------

groupshared uint gMask[3];

[shader("compute")]
[numthreads(32, 1, 1)]
void athzEncodeRuns(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    const uint run = group.x;
    const uint lane = local.x;
    const uint page = run / params.pageRuns;
    const uint inPage = run % params.pageRuns;
    if (lane < 3u) {
        gMask[lane] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (page >= params.pages) {
        return;   // uniform across the workgroup
    }
    const uint4 p = pageList[page];
    const uint first = inPage * params.runGaussians;
    const uint here = p.y > first ? min(p.y - first, params.runGaussians) : 0u;
    const uint steps = params.runGaussians / kAthzLanes;
    const uint n = params.tableLog;

    uint x = kAthzLow;
    uint top = params.capWords;   // the stream is written down from here
    uint step = 0u;
    if (params.hasRans != 0u) {
        // Backwards through the planes' coding order, and each plane's steps.
        for (uint i = params.planes; i-- > 0u;) {
            const uint b = athzPlaneAt(i, params.tailWord);
            const uint4 plane = planes[b];
            if (athzLayer(plane) != params.layer || athzEncoding(plane) != 1u) {
                continue;
            }
            for (uint k = steps; k-- > 0u;) {
                const uint g = lane * steps + k;   // a lane's symbols are a block of the run
                const bool live = g < here;
                uint f = 1u, c = 0u;
                if (live) {
                    const uint fc = freqs[b * 256u + loadByte(p.x + first + g, g, b)];
                    f = fc & 0xFFFFu;
                    c = fc >> 16;
                }
                const bool emit = live && x >= (f << (32u - n));
                if (emit) {
                    InterlockedOr(gMask[step % 3u], 1u << lane);
                }
                if (lane == 0u) {
                    gMask[(step + 1u) % 3u] = 0u;
                }
                GroupMemoryBarrierWithGroupSync();
                const uint mask = gMask[step % 3u];
                const uint total = countbits(mask);
                if (emit) {
                    const uint rank = countbits(mask & ((1u << lane) - 1u));
                    scratch[run * params.capWords + top - total + rank] = x & 0xFFFFu;
                    x >>= 16;
                }
                top -= total;
                if (live) {
                    x = ((x / f) << n) + (x % f) + c;
                }
                step += 1u;
            }
        }
        states[run * kAthzLanes + lane] = x;
        if (lane == 0u) {
            runWords[run] = params.capWords - top;
        }
    }
    // The raw planes: lane by lane over each plane's words.
    for (uint b = 0; b < params.planes; ++b) {
        const uint4 plane = planes[b];
        if (athzLayer(plane) != params.layer || athzEncoding(plane) != 0u) {
            continue;
        }
        const uint width = athzWidth(plane);
        if (width == 0u) {
            continue;
        }
        const uint words = athzRawWords(params.runGaussians, width);
        const uint mask = (1u << width) - 1u;
        for (uint w = lane; w < words; w += kAthzLanes) {
            uint word = 0u;
            const uint bit0 = w * 32u;
            for (uint g = bit0 / width; g * width < bit0 + 32u && g < params.runGaussians; ++g) {
                if (g >= here) {
                    break;
                }
                const uint v = (loadByte(p.x + first + g, g, b) - athzBase(plane)) & mask;
                const int shift = int(g * width) - int(bit0);
                word |= shift >= 0 ? (v << uint(shift)) : (v >> uint(-shift));
            }
            rawScratch[run * params.rawWords + plane.z + w] = word;
        }
    }
    if (lane == 0u) {
        const uint words = params.hasRans != 0u ? runWords[run] : 0u;
        uint bytes = 4u * params.rawWords;
        if (params.hasRans != 0u) {
            bytes += 4u + 4u * kAthzLanes + 4u * ((words + 1u) / 2u);
        }
        if (inPage == 0u) {
            bytes += 4u * params.pageRuns;
        }
        runBytes[run] = bytes;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void athzPlace(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    const uint run = group.x;
    const uint lane = local.x;
    const uint page = run / params.pageRuns;
    const uint inPage = run % params.pageRuns;
    if (page >= params.pages) {
        return;
    }
    const uint pageRun = page * params.pageRuns;
    const uint pageStart = runOffsets[pageRun];
    if (lane == 0u) {
        output.Store(pageStart + 4u * inPage, runOffsets[run] + runBytes[run] - pageStart);
    }
    uint at = runOffsets[run] + (inPage == 0u ? 4u * params.pageRuns : 0u);
    if (params.hasRans != 0u) {
        const uint words = runWords[run];
        if (lane == 0u) {
            output.Store(at, words);
        }
        output.Store(at + 4u + 4u * lane, states[run * kAthzLanes + lane]);
        at += 4u + 4u * kAthzLanes;
        const uint from = run * params.capWords + params.capWords - words;
        const uint pairs = (words + 1u) / 2u;
        for (uint i = lane; i < pairs; i += kAthzLanes) {
            const uint lo = scratch[from + 2u * i];
            const uint hi = 2u * i + 1u < words ? scratch[from + 2u * i + 1u] : 0u;
            output.Store(at + 4u * i, lo | (hi << 16));
        }
        at += 4u * pairs;
    }
    for (uint i = lane; i < params.rawWords; i += kAthzLanes) {
        output.Store(at + 4u * i, rawScratch[run * params.rawWords + i]);
    }
}

// --- pages -------------------------------------------------------------------

/// v as f16, rounded down (`up` false) or up.
uint halfOutward(float v, bool up) {
    uint h = f32tof16(v);
    const float back = f16tof32(h);
    if (up ? back < v : back > v) {
        const bool negative = (h & 0x8000u) != 0u;
        const bool zero = (h & 0x7FFFu) == 0u;
        if (up) {
            h = zero ? 0x0001u : (negative ? h - 1u : h + 1u);
        } else {
            h = zero ? 0x8001u : (negative ? h + 1u : h - 1u);
        }
    }
    return h;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void athzPageInfo(uint3 tid: SV_DispatchThreadID) {
    const uint page = tid.x;
    if (page >= params.pages) {
        return;
    }
    const uint4 p = pageList[page];
    float3 lo = float3(3.0e38, 3.0e38, 3.0e38);
    float3 hi = -lo;
    for (uint g = 0; g < p.y; ++g) {
        const float3 x = asfloat(uint3(loadWord(p.x + g, 0u), loadWord(p.x + g, 1u), loadWord(p.x + g, 2u)));
        lo = min(lo, x);
        hi = max(hi, x);
    }
    // The page's level, its error (a cell's edge; a splat's own: none) and
    // the page of the coarser level its first element hangs from.
    uint level = 0xFFu;
    float eps = 0.0;
    uint parent = 0xFFFFFFFFu;
    if (params.leaves == 0u) {
        const uint4 me = levelInfo[p.w];
        level = me.w;
        eps = params.extent / float(1u << min(me.w, 30u));
        if (p.w > 0u) {
            const uint4 up = levelInfo[p.w - 1u];
            const uint cell = loadWord(p.x, params.tailWord) >> 3;
            uint a = 0u, b = up.y;   // first of the coarser level's cells not below `cell`
            while (a < b) {
                const uint m = (a + b) / 2u;
                if (loadWord(up.x + m, params.tailWord) < cell) {
                    a = m + 1u;
                } else {
                    b = m;
                }
            }
            parent = up.z + min(a, up.y - 1u) / (params.pageRuns * params.runGaussians);
        }
    } else if (params.levels > 0u) {
        const uint4 finest = levelInfo[params.levels - 1u];
        const uint group = loadWord(p.x, params.tailWord);
        parent = finest.z + min(group, finest.y - 1u) / (params.pageRuns * params.runGaussians);
    }
    const uint at = page * 8u;
    pageInfo[at + 0u] = p.y | (p.w << 16) | (level << 24);
    pageInfo[at + 1u] = parent;
    pageInfo[at + 2u] = halfOutward(lo.x, false) | (halfOutward(lo.y, false) << 16);
    pageInfo[at + 3u] = halfOutward(lo.z, false) | (halfOutward(hi.x, true) << 16);
    pageInfo[at + 4u] = halfOutward(hi.y, true) | (halfOutward(hi.z, true) << 16);
    pageInfo[at + 5u] = halfOutward(eps, true) | (halfOutward(0.0, false) << 16);
    pageInfo[at + 6u] = 0u;
    pageInfo[at + 7u] = 0u;
}
