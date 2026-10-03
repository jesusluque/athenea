// Copyright (c) 2026 jesus luque.
//
// One gaussian as the decimation reads it: where it is, how opaque, its
// three sizes and axes, and its colour. A module of its own and of no
// buffers, so that the kernels sharing it do not share their bindings: Metal
// binds 31 buffers a kernel, and Slang numbers a module's globals together.
module lod_blob;
import athenea.common.packing;

public struct Blob {
    public float3   centre;
    public float    opacity;
    public float3   scale;
    public float3x3 axes;     // rows: the Gaussian's own axes
    public float3   colour;
};

public Blob blobOf(float4 p, uint row0, uint row1, uint row2, uint row3) {
    Blob b;
    b.centre = p.xyz;
    b.opacity = p.w;
    b.scale = exp(float3(lowHalf(row1), highHalf(row1), lowHalf(row2)));
    b.axes = transpose(axesOfQuaternion(decodeQuaternion(row0)));   // its columns are the axes
    // The shape keeps the base colour itself, 0.5 + SH0 * dc (splat_decode).
    b.colour = saturate(float3(highHalf(row2), lowHalf(row3), highHalf(row3)));
    return b;
}

