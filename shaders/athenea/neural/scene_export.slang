// Copyright (c) 2026 jesus luque.
//
// A FOUND SCENE HANDED ON AS A STANDARD SPLAT FILE. A render's record
// (scene_render.slang) laid out as the engine's flattened record is
// (splat/flatten_basis.slang: what flatten_pack.slang turns into a PLY, an
// SPZ or a glTF): the same position, shape and first harmonic, an opacity in
// place of its logit, and nothing above the first harmonic -- a gaussian a
// look made has one colour. The light may be scaled on the way (`gain`,
// linear): a scene is as bright as its lights were, and a viewer on the web
// shows what the file says.
module scene_export;
import athenea.splat.flatten_basis;

struct ExportParams {
    uint  count;
    uint  stride;      // floats a record in `records` (26, 27 or 30)
    float gain;        // on the light, before it is encoded
    uint  pad0;
};

StructuredBuffer<float>    records;
RWStructuredBuffer<float>  flat;
ConstantBuffer<ExportParams> params;

float decoded(float encoded) {
    const float v = max(encoded, 0.0);
    return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4);
}

float encoded(float linear) {
    const float v = max(linear, 0.0);
    return v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void exportFlat(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint from = i * params.stride;
    const uint to = i * kFlatStride;
    for (uint k = 0; k < kFlatStride; ++k) {
        flat[to + k] = 0.0;
    }
    for (uint a = 0; a < 3; ++a) {
        flat[to + kFlatPosition + a] = records[from + a];
        flat[to + kFlatLogScale + a] = records[from + 4 + a];
        flat[to + kFlatNormal + a] = records[from + 14 + a];
        // The first harmonic as a capture keeps a colour: display-encoded, about a half, over Y00.
        const float shown = records[from + 11 + a] * 0.28209479 + 0.5;
        flat[to + kFlatSh + a] = (encoded(decoded(shown) * params.gain) - 0.5) / 0.28209479;
    }
    flat[to + kFlatOpacity] = 1.0 / (1.0 + exp(-records[from + 3]));
    for (uint q = 0; q < 4; ++q) {
        flat[to + kFlatRotation + q] = records[from + 7 + q];
    }
    flat[to + kFlatShrink] = 1.0;
}
