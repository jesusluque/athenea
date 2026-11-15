// Copyright (c) 2026 jesus luque.
//
// A SCENE'S LIGHT, TAKEN AGAIN FROM A FRAME.
//
// A found scene's gaussian keeps where a surface is and what it is made of, and the light a look saw leave
// it. When the stage's light changes and nothing else does, the first two still hold: only the light is
// taken again. A frame path traced under the new light is read at each standing gaussian's own middle --
// where the frame shows a surface at the gaussian's depth, which is that gaussian's surface -- and the
// gaussian's colour becomes the frame's there. Nothing is taken out and nothing is made.
//
// What the kernel leaves beside the cloud's colours is a layer: a float4 a gaussian, the colour as a record
// keeps one (rgb) and whether this frame said it (a). A scene keeps a layer a frame that took its light
// again, beside the piece, and whoever draws the scene at a time lays over each piece the layers up to it.
//
// Not asked: a gaussian that lets light through (its picture is made of what is behind it, not read off
// the frame), one the frame does not show (behind something, out of the frame), one that is gone. One a
// finer look was laid over is asked where the piece's record of what is gone is handed (`gone`): it is
// there still, and is what a level of detail draws.
module scene_light;
import athenea.common.packing;
import athenea.common.color;
import athenea.scene.splat_encoding;

struct RelightParams {
    uint  count;
    uint  width;
    uint  height;
    uint  linearCloud;
    float4 viewX;
    float4 viewY;
    float4 viewZ;
    float lensX; float lensY; float middleX; float middleY;
    float nearness;     // the share of its depth a surface may stand off the gaussian and be the gaussian's
    uint  hasPbr;
    uint  hasGone;      // `gone` says what stands; without it, a gaussian's opacity does
    uint  keep;         // sceneRelight: 1 leaves the cloud's colours as they are and hands the frame's light as it is (rgb, linear)
    float gainR; float gainG; float gainB; uint pad2;   // sceneAddLight: what the layer is multiplied by
    uint  floats;       // recordsAddLight: floats a record
    float own;          // recordsAddLight: what the record's own light is multiplied by
    uint  pad3; uint pad4;
};

StructuredBuffer<float4>   positions;    // xyz, opacity (0 and under: not standing)
RWStructuredBuffer<uint>   shape;        // four words a gaussian; the colour is written
StructuredBuffer<uint>     pbr;          // metallic, roughness, transmission
StructuredBuffer<uint>     gone;         // the look a gaussian went at (0: it is there, standing or laid over)
StructuredBuffer<float4>   frameColour;  // the frame's light a pixel
StructuredBuffer<float>    frameDepth;   // and its depth along the view (0: nothing)
RWStructuredBuffer<float4> layer;        // the colour as a record keeps it, and whether it was said
StructuredBuffer<float4>   added;        // sceneAddLight: light a gaussian (rgb, linear) and whether there is any (a)
RWStructuredBuffer<float>  records;      // recordsAddLight: a render's records, their colours written
RWStructuredBuffer<uint>   counters;     // [0]: gaussians whose light was taken again
ConstantBuffer<RelightParams> params;

float encodedLight(float linear) {
    const float v = max(linear, 0.0);
    return v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sceneRelight(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    layer[i] = float4(0.0);
    // With the record of what is gone, every gaussian that is still there is asked: the finest, and the
    // coarser ones finer looks were laid over, which are what the scene is drawn with from far.
    if (params.hasGone != 0u ? gone[i] != 0u : positions[i].w <= 0.0) {
        return;
    }
    // What lets light through is not said anew (its picture is made of what is behind it); for a group's
    // layer it is asked as the rest: what it shows under that group is the group's light it sends.
    if (params.keep == 0u && params.hasPbr != 0u && pbrTransmission(pbr[i]) > 0.02) {
        return;
    }
    const float3 p = positions[i].xyz;
    const float z = dot(params.viewZ.xyz, p) + params.viewZ.w;
    if (!(z > 0.0)) {
        return;
    }
    const float x = params.middleX + params.lensX * (dot(params.viewX.xyz, p) + params.viewX.w) / z;
    const float y = params.middleY + params.lensY * (dot(params.viewY.xyz, p) + params.viewY.w) / z;
    if (x < 0.0 || y < 0.0 || x >= float(params.width) || y >= float(params.height)) {
        return;
    }
    const uint at = uint(y) * params.width + uint(x);
    const float seenAt = frameDepth[at];
    if (!(seenAt > 0.0) || abs(seenAt - z) > params.nearness * z) {
        return;
    }
    const float4 pixel = frameColour[at];
    if (!(pixel.a > 0.5)) {
        return;
    }
    const float3 light = max(pixel.rgb, float3(0.0));
    const float3 shown = float3(encodedLight(light.r), encodedLight(light.g), encodedLight(light.b));
    InterlockedAdd(counters[0], 1u);
    if (params.keep != 0u) {
        // A GROUP OF LIGHTS' OWN LAYER: the frame was path traced under that group alone, and what each
        // gaussian sends then is kept as light, to be added to the scene's by a level (sceneAddLight).
        // And how far the eye was (a): of the views that show a gaussian, the nearest is the one that sees
        // it largest -- as the look that made it did -- and its word is the one kept.
        layer[i] = float4(light, z);
        return;
    }
    const float3 put = params.linearCloud != 0u ? light : shown;
    const uint row = i * 4u;
    shape[row + 2u] = packHalves(lowHalf(shape[row + 2u]), put.r);
    shape[row + 3u] = packHalves(put.g, put.b);
    layer[i] = float4((shown - 0.5) / 0.28209479, 1.0);
}

// LIGHT ADDED TO A CLOUD'S OWN, A GAUSSIAN: a group of lights' layer times its level and its tint. Light is
// linear: the scene with a lamp at half is the scene without it and half the lamp's layer.
[shader("compute")]
[numthreads(64, 1, 1)]
void sceneAddLight(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 more = added[i];
    if (!(more.a > 0.0)) {
        return;
    }
    const uint row = i * 4u;
    const float3 base = float3(highHalf(shape[row + 2u]), lowHalf(shape[row + 3u]), highHalf(shape[row + 3u]));
    const float3 lit = max(cloudLight(base, params.linearCloud != 0u) +
                           more.rgb * float3(params.gainR, params.gainG, params.gainB), float3(0.0));
    const float3 put = params.linearCloud != 0u ? lit : float3(encodedLight(lit.r), encodedLight(lit.g), encodedLight(lit.b));
    shape[row + 2u] = packHalves(lowHalf(shape[row + 2u]), put.r);
    shape[row + 3u] = packHalves(put.g, put.b);
}

float lightOfEncoded(float shown) {
    const float v = max(shown, 0.0);
    return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4);
}

// THE SAME OVER A RENDER'S RECORDS, as a file keeps them (scene_render.slang: the colour at 11, the first
// harmonic of the light as shown): for whoever hands the scene on with its groups at their levels. `own`
// multiplies the light the record has -- 0 hands on a group's light alone.
[shader("compute")]
[numthreads(64, 1, 1)]
void recordsAddLight(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint at = i * params.floats + 11u;
    const float3 shown = float3(records[at], records[at + 1u], records[at + 2u]) * 0.28209479 + 0.5;
    float3 lit = params.own * float3(lightOfEncoded(shown.r), lightOfEncoded(shown.g), lightOfEncoded(shown.b));
    const float4 more = added[i];
    if (more.a > 0.0) {
        lit += more.rgb * float3(params.gainR, params.gainG, params.gainB);
    }
    lit = max(lit, float3(0.0));
    records[at] = (encodedLight(lit.r) - 0.5) / 0.28209479;
    records[at + 1u] = (encodedLight(lit.g) - 0.5) / 0.28209479;
    records[at + 2u] = (encodedLight(lit.b) - 0.5) / 0.28209479;
}
