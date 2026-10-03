// Copyright (c) 2026 jesus luque.
//
// Per pixel: its tile's splats, nearest first, until the pixel is opaque.
module splat_blend;
import athenea.common.color;
import frame;

StructuredBuffer<uint>      ranges;
StructuredBuffer<uint>      pairSplats;
StructuredBuffer<ProjRecord> proj;
// What is under the splats: a rasterised point layer (or nothing: bound to a
// placeholder with params.hasUnder == 0). Linear, premultiplied, bottom row first.
StructuredBuffer<float4>    underColour;
StructuredBuffer<float>     underDepth;
// CRYPTOMATTE. `cryptoIds` is the id a projected slot carries (splat_project
// wrote it), `underCrypto` the id of the opaque layer under the splats, and
// `crypto` the matte itself: kCryptoSlots ids and coverages a pixel, three
// float4 (id, cov, id, cov) in descending coverage. All bound whether the
// frame keeps a matte or not: a name a shader declares must be bound.
StructuredBuffer<uint>      cryptoIds;
StructuredBuffer<uint>      underCrypto;
RWStructuredBuffer<float4>  crypto;
RWStructuredBuffer<float4>  colour;   // width * height, premultiplied, linear
RWStructuredBuffer<float>   depth;    // width * height
ConstantBuffer<FrameParams> params;

/// How many ids a pixel can name. Six is the Cryptomatte specification's
/// three layers of two ranks: past that the lightest is dropped, which is
/// what every renderer that writes the format does.
static const uint kCryptoSlots = 6;

/// The matte a pixel builds as the walk goes: an id's coverage is the sum of
/// the weights (transmittance * alpha) of everything with that id, which is
/// exactly the alpha that id contributed to the pixel. Held in registers, so
/// the loop costs a search over six and no memory.
struct CryptoList {
    uint  ids[kCryptoSlots];
    float weights[kCryptoSlots];

    [[mutating]] void add(uint id, float weight) {
        // Id 0 is nothing named: a capture with no ancestry, a point, the
        // background. `cryptomatteId` never returns it (the fixup moves a
        // hash of zero off it), so it is free to mean absence, and what it
        // covers is simply missing from the matte rather than called
        // something it is not.
        if (weight <= 0.0 || id == 0) {
            return;
        }
        uint lightest = 0;
        for (uint k = 0; k < kCryptoSlots; ++k) {
            if (ids[k] == id && weights[k] > 0.0) {
                weights[k] += weight;
                return;
            }
            if (weights[k] < weights[lightest]) {
                lightest = k;
            }
        }
        // A free slot is one whose weight is still zero, so the same test
        // finds it; a full list gives up its lightest rank, and only if this
        // one outweighs it.
        if (weight > weights[lightest]) {
            ids[lightest] = id;
            weights[lightest] = weight;
        }
    }

    /// Ranks in descending coverage, written as (id, cov) pairs.
    void write(uint at, uint pixels) {
        uint order[kCryptoSlots];
        for (uint k = 0; k < kCryptoSlots; ++k) {
            order[k] = k;
        }
        for (uint i = 1; i < kCryptoSlots; ++i) {
            const uint pick = order[i];
            uint j = i;
            while (j > 0 && weights[order[j - 1]] < weights[pick]) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = pick;
        }
        for (uint layer = 0; layer < kCryptoSlots / 2; ++layer) {
            const uint a = order[layer * 2];
            const uint b = order[layer * 2 + 1];
            crypto[layer * pixels + at] =
                float4(weights[a] > 0.0 ? bit_cast<float>(ids[a]) : 0.0, max(weights[a], 0.0),
                       weights[b] > 0.0 ? bit_cast<float>(ids[b]) : 0.0, max(weights[b], 0.0));
        }
    }
};

/// The walk, with `composite` a constant at each call site: the splat-only
/// entry point folds the disc and under-layer branches away. Both cost the
/// blend -- the hottest loop in the engine -- half as much again when they
/// are live in a frame that has neither.
// A TILE'S SPLATS, READ ONCE FOR ITS 256 PIXELS. The group is the tile
// (16 x 16 threads): it loads the next 256 of the tile's records into group
// memory together, and every pixel walks them from there, rather than each of
// the 256 reading every record through `pairSplats` on its own. A pixel that
// is done still loads for the others; the group stops loading when all are.
static const uint kBatch = kTile * kTile;
groupshared ProjRecord sRecords[kBatch];
groupshared uint       sSplats[kBatch];
groupshared uint       sDone;

void blendPixel(uint3 tid, uint3 groupThread, bool composite, bool matte) {
    const uint px = tid.x;
    const uint py = tid.y;
    const uint lane = groupThread.y * kTile + groupThread.x;
    // Outside the image: no pixel, but a pair of hands for the loads.
    const bool inside = px < params.width && py < params.height;
    const uint tile = (py / kTile) * params.tilesX + px / kTile;
    const uint start = ranges[tile * 2];
    const uint end = ranges[tile * 2 + 1];
    const float2 pixel = float2(float(px) + 0.5, float(py) + 0.5);

    const uint at = inside ? py * params.width + px : 0u;
    const bool under = inside && composite && params.hasUnder != 0;
    const float underZ = under ? underDepth[at] : 0.0;
    const bool underHit = under && underZ > 0.0;

    float3 rgb = float3(0.0);
    float transmittance = 1.0;
    float depthSum = 0.0;
    float depthAt = -1.0;
    // What ends the walk from behind: the raster under-layer, or the first
    // opaque disc. Either is a linear colour composited under the splats'
    // (linearised) contribution -- the same arithmetic whichever route drew
    // the points, which is what lets the two routes agree.
    bool opaqueHit = false;
    float4 opaqueColour = float4(0.0);
    float opaqueZ = 0.0;
    uint opaqueId = 0;
    CryptoList list;
    if (matte) {
        for (uint k = 0; k < kCryptoSlots; ++k) {
            list.ids[k] = 0;
            list.weights[k] = 0.0;
        }
    }
    bool done = !inside;
    for (uint first = start; first < end; first += kBatch) {
        // Stop loading once every pixel of the tile has stopped walking.
        if (lane == 0) {
            sDone = 0;
        }
        GroupMemoryBarrierWithGroupSync();
        if (done) {
            InterlockedAdd(sDone, 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (sDone == kBatch) {
            break;
        }
        const uint mine = first + lane;
        if (mine < end) {
            const uint splat = pairSplats[mine];
            sSplats[lane] = splat;
            sRecords[lane] = proj[splat];
        }
        GroupMemoryBarrierWithGroupSync();
        const uint count = min(kBatch, end - first);
        for (uint j = 0; j < count && !done; ++j) {
            const ProjRecord record = sRecords[j];
            const float4 screen = record.screen;
            if (underHit && screen.z >= underZ) {
                done = true;
                break;
            }
            const float2 d = pixel - screen.xy;
            const float4 conic = record.conic;
            const float power = conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y;
            if (power > screen.w) {
                continue;
            }
            if (composite && record.colour.w > 0.5) {
                opaqueHit = true;
                opaqueColour = float4(record.colour.rgb, 1.0);
                opaqueZ = screen.z;
                if (matte) {
                    opaqueId = cryptoIds[sSplats[j]];
                }
                done = true;
                break;
            }
            const float alpha = min(0.99, conic.w * exp(-0.5 * power));
            if (alpha < 1.0 / 255.0) {
                continue;
            }
            const float weight = transmittance * alpha;
            if (matte) {
                list.add(cryptoIds[sSplats[j]], weight);
            }
            rgb += weight * record.colour.xyz;
            depthSum += weight * screen.z;
            transmittance *= 1.0 - alpha;
            if (depthAt < 0.0 && 1.0 - transmittance >= params.depthThreshold) {
                depthAt = screen.z;
            }
            if (transmittance < 1e-4) {
                done = true;
                break;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (!inside) {
        return;
    }
    if (!opaqueHit && underHit) {
        opaqueHit = true;
        opaqueColour = underColour[at];
        opaqueZ = underZ;
        if (matte && params.hasUnderCrypto != 0) {
            opaqueId = underCrypto[at];
        }
    }
    // What closed the walk takes the rest of the pixel: whatever light got
    // through the splats is its coverage, which is what makes the layer's
    // coverages sum to the pixel's alpha.
    if (matte) {
        list.add(opaqueId, opaqueHit ? transmittance : 0.0);
        list.write(at, params.width * params.height);
    }

    const float coverage = 1.0 - transmittance;
    // Blended in the encoded space the cloud was trained in; the finished
    // splat contribution is made linear. Linearising each splat instead turned
    // letters red and skies electric in openFXplayer.
    float3 linearRgb = rgb;
    if (params.linearise != 0 && coverage > 1e-6) {
        linearRgb = srgbToLinear(rgb / coverage) * coverage;
    }
    float4 behind = float4(params.bgR, params.bgG, params.bgB, params.bgA);
    if (opaqueHit) {
        behind = opaqueColour;
        depthSum += transmittance * opaqueZ;
        if (depthAt < 0.0) {
            depthAt = opaqueZ;
        }
    }
    const float total = opaqueHit ? 1.0 : coverage;
    colour[at] = float4(linearRgb + transmittance * behind.rgb, coverage + transmittance * behind.w);
    depth[at] = params.depthMode == 0 ? (total > 1e-6 ? depthSum / total : 0.0) : max(depthAt, 0.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatBlend(uint3 tid: SV_DispatchThreadID, uint3 groupThread: SV_GroupThreadID) {
    blendPixel(tid, groupThread, false, false);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatBlendComposite(uint3 tid: SV_DispatchThreadID, uint3 groupThread: SV_GroupThreadID) {
    blendPixel(tid, groupThread, true, false);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatBlendCrypto(uint3 tid: SV_DispatchThreadID, uint3 groupThread: SV_GroupThreadID) {
    blendPixel(tid, groupThread, false, true);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatBlendCompositeCrypto(uint3 tid: SV_DispatchThreadID, uint3 groupThread: SV_GroupThreadID) {
    blendPixel(tid, groupThread, true, true);
}
