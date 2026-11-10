// Copyright (c) 2026 jesus luque.
//
// THE SPLATS BY THE DEVICE'S RASTERISER. After the projection and the depth
// sort a frame of plain gaussians needs no tiles: one quad a visible splat,
// nearest first, the device deciding which pixels each covers and blending
// them under what is already there (front to back: what a pixel holds hides
// what comes after). It stands for the counts, the pairs, the tile sort and
// the blend of the tile route, which on a town were half a frame.
//
// What it draws is the tile route's arithmetic for a plain gaussian -- the
// same record, the same falloff, the same cut at 1/255 -- so the two agree
// to the blend's rounding. A surfel, a catcher, a medium, a slope or a sharp
// lobe are the tile route's.
module splat_quads;
import athenea.common.color;
import frame;

ConstantBuffer<FrameParams> params;   // count: visible splats

StructuredBuffer<uint>       order;   // visible splats, nearest first
StructuredBuffer<ProjRecord> proj;

struct QuadVertex {
    float4 position : SV_Position;
    nointerpolation float4 screen : SCREEN;
    nointerpolation float4 conic : CONIC;
    nointerpolation float3 colour : COLOUR;
};

[shader("vertex")]
QuadVertex splatQuadVertex(uint vertex: SV_VertexID, uint instance: SV_InstanceID, uint start: SV_StartInstanceLocation) {
    // The batch's first splat: an instance id counts from it, or from 0.
    const ProjRecord record = proj[order[params.base != 0 ? instance : instance + start]];
    const float4 conic = record.conic;
    // The ellipse a x^2 + 2 b x y + c y^2 = cutoff by its own axes: a quad
    // along them is the smallest that holds it (an upright box round a long
    // gaussian lying across the pixels is several times its area).
    const float mid = 0.5 * (conic.x + conic.z);
    const float spread = sqrt(max(0.25 * (conic.x - conic.z) * (conic.x - conic.z) + conic.y * conic.y, 0.0));
    const float steep = mid + spread;                  // across the short axis
    const float gentle = max(mid - spread, 1.0e-12);   // along the long one
    float2 across = abs(conic.y) > 1.0e-12 * steep ? float2(conic.y, steep - conic.x)
                                                    : (conic.x >= conic.z ? float2(1.0, 0.0) : float2(0.0, 1.0));
    across = normalize(across);
    const float2 along = float2(-across.y, across.x);
    const float cutoff = max(record.screen.w, 0.0);
    const float2 reach = sqrt(cutoff / float2(steep, gentle)) + 0.5;
    // Two triangles: 0 1 2, 2 1 3.
    const uint c = vertex == 3 ? 2u : vertex == 4 ? 1u : vertex == 5 ? 3u : vertex;
    const float2 side = float2((c & 1u) != 0 ? 1.0 : -1.0, (c & 2u) != 0 ? 1.0 : -1.0);
    const float2 pixel = record.screen.xy + side.x * reach.x * across + side.y * reach.y * along;
    QuadVertex o;
    // Pixels from the top left, as the frame's buffers are.
    o.position = float4(pixel.x / float(params.width) * 2.0 - 1.0, 1.0 - pixel.y / float(params.height) * 2.0, 0.5, 1.0);
    o.screen = record.screen;
    o.conic = conic;
    o.colour = record.colour.xyz;
    return o;
}

struct QuadOut {
    float4 colour : SV_Target0;   // premultiplied, and the coverage
    float4 depth : SV_Target1;    // the weight times view z
};

[shader("fragment")]
QuadOut splatQuadFragment(QuadVertex v) {
    const float2 d = v.position.xy - v.screen.xy;
    const float power = v.conic.x * d.x * d.x + 2.0 * v.conic.y * d.x * d.y + v.conic.z * d.y * d.y;
    if (power > v.screen.w) {
        discard;
    }
    const float alpha = min(0.99, falloffAlpha(abs(v.conic.w), exp(-0.5 * power)));
    if (alpha < 1.0 / 255.0) {
        discard;
    }
    QuadOut o;
    o.colour = float4(v.colour * alpha, alpha);
    o.depth = float4(v.screen.z * alpha, 0.0, 0.0, alpha);
    return o;
}

Texture2D<float4>            quadColour;
Texture2D<float4>            quadDepth;

/// WHERE A PIXEL IS FULL. The quads are drawn in batches, each twice the one
/// before, and between two batches a triangle over the frame writes the
/// nearest depth wherever the coverage has reached kQuadFull: the quads lie
/// behind it, so the device's depth test drops theirs there before a
/// fragment is shaded. Drawn all at once a town's six million quads cost 48
/// ms, more than the tiles; what a pixel holds hides most of them.
static const float kQuadFull = 0.996;

[shader("vertex")]
float4 splatQuadsFullVertex(uint vertex: SV_VertexID) : SV_Position {
    return float4(vertex == 1 ? 3.0 : -1.0, vertex == 2 ? 3.0 : -1.0, 0.0, 1.0);
}

[shader("fragment")]
void splatQuadsFullFragment(float4 position: SV_Position) {
    if (quadColour.Load(int3(int2(position.xy), 0)).w < kQuadFull) {
        discard;
    }
}
StructuredBuffer<float4>     underColour;
StructuredBuffer<float>      underDepth;
RWStructuredBuffer<float4>   colour;   // width * height, premultiplied, linear
RWStructuredBuffer<float>    depth;    // width * height

/// What the quads drew, into the frame's buffers over what lies under the
/// splats: the tile route's last lines.
[shader("compute")]
[numthreads(8, 8, 1)]
void splatQuadsResolve(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    const float4 drawn = quadColour.Load(int3(tid.xy, 0));
    float depthSum = quadDepth.Load(int3(tid.xy, 0)).x;
    const float coverage = saturate(drawn.w);
    const float transmittance = 1.0 - coverage;
    const float underZ = params.hasUnder != 0 ? underDepth[at] : 0.0;
    const bool opaqueHit = underZ > 0.0;
    float4 behind = float4(params.bgR, params.bgG, params.bgB, params.bgA);
    if (opaqueHit) {
        behind = underColour[at];
        if (params.displayBlend != 0) {
            behind.rgb = linearToSrgb(behind.rgb);
        }
        depthSum += transmittance * underZ;
    }
    const float total = opaqueHit ? 1.0 : coverage;
    float4 pixelColour = float4(drawn.rgb + transmittance * behind.rgb, coverage + transmittance * behind.w);
    if (params.displayBlend != 0) {
        if (!opaqueHit) {
            pixelColour.rgb = drawn.rgb + transmittance * linearToSrgb(behind.rgb);
        }
        pixelColour.rgb = pixelColour.w > 1.0e-6 ? srgbToLinear(pixelColour.rgb / pixelColour.w) * pixelColour.w
                                                 : float3(0.0);
    }
    colour[at] = pixelColour;
    depth[at] = total > 1e-6 ? depthSum / total : 0.0;
}
