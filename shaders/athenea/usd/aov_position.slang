// Copyright (c) 2026 jesus luque.
//
// Where each pixel's surface is in the world: the frame's view z put back
// through the camera that drew it. What a compositor relights with (Nuke's P
// pass): the frame, bottom row first, a float4 a pixel -- (x, y, z, 1) where
// something drew, zeros where nothing did, so the alpha says which.
//
// The pixel's centre goes back the way the rasteriser projected it
// (splat_project: pixel = focal * v.xy / v.z + centre, x right, y up, the
// rows bottom first), and view space goes to world through the rows of the
// view-to-world matrix.
module aov_position;

struct PositionParams {
    uint   width;
    uint   height;
    uint   orthographic;   // 1: pixels are focal per world unit, not per x/z
    uint   stride;         // depth entries a pixel, and which is the view z
    uint   offset;
    uint   pad1; uint pad2; uint pad3;
    float  focalX;
    float  focalY;
    float  centreX;
    float  centreY;
    float4 row0;           // view to world, row by row (+z forward)
    float4 row1;
    float4 row2;
};

StructuredBuffer<float>     depth;   // view z, 0 where nothing drew
RWStructuredBuffer<float4>  position;
ConstantBuffer<PositionParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void aovPosition(uint3 tid: SV_DispatchThreadID) {
    const uint pixel = tid.x;
    if (pixel >= params.width * params.height) {
        return;
    }
    const float z = depth[pixel * params.stride + params.offset];
    if (!(z > 0.0)) {
        position[pixel] = float4(0.0);
        return;
    }
    const float px = float(pixel % params.width) + 0.5;
    const float py = float(pixel / params.width) + 0.5;
    const float scale = params.orthographic != 0 ? 1.0 : z;
    const float4 v = float4((px - params.centreX) / params.focalX * scale,
                            (py - params.centreY) / params.focalY * scale, z, 1.0);
    position[pixel] = float4(dot(params.row0, v), dot(params.row1, v), dot(params.row2, v), 1.0);
}
