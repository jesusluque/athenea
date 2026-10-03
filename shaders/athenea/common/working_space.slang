// Copyright (c) 2026 jesus luque.
//
// THE WORKING SPACE, AS A RECORD EVERY KERNEL READS.
//
// Linear light here is whatever the working space is, and what a kernel
// needs of it -- a luminance, a colour given in CIE XYZ, a capture's sRGB --
// it asks this module rather than writing Rec.709's numbers in. The record
// is gpu::WorkingSpace (modules/gpu/include/athenea/gpu/WorkingSpace.h),
// field for field: four 3x3s as float4 rows and the luminance weights.
//
// `gWorkingSpace` is a uniform, so it lands in the kernel's global constant
// block beside the kernel's own uniforms and takes no buffer of its own; the
// kernel types bind it by name at every dispatch (gpu::bindWorkingSpace).
// For lin_rec709_scene the record holds the very constants the kernels held
// before it, so a frame is the same to the bit.
module working_space;

public struct WorkingSpace {
    public float4 toXyz0;      // working RGB to XYZ (D65), rows; w unused
    public float4 toXyz1;
    public float4 toXyz2;
    public float4 fromXyz0;    // XYZ (D65) to working RGB, rows
    public float4 fromXyz1;
    public float4 fromXyz2;
    public float4 fromSrgb0;   // sRGB decoded (linear Rec.709) to working RGB, rows
    public float4 fromSrgb1;
    public float4 fromSrgb2;
    public float4 toSrgb0;     // working RGB to linear Rec.709, rows
    public float4 toSrgb1;
    public float4 toSrgb2;
    public float4 luminance;   // xyz: the weights of a luminance
    public uint4  flags;       // x: kWorkingSpaceRec709Primaries
};

/// The working space's primaries are Rec.709's: decoded sRGB is already in
/// it, and `fromSrgb` is not applied.
public static const uint kWorkingSpaceRec709Primaries = 1u;

public uniform WorkingSpace gWorkingSpace;

/// A colour's luminance in the working space.
public float workingLuminance(float3 c) {
    return dot(c, gWorkingSpace.luminance.xyz);
}

/// The weights themselves, for a caller that dots them later.
public float3 workingLuminanceWeights() {
    return gWorkingSpace.luminance.xyz;
}

/// A colour given in CIE XYZ (D65), in the working space.
public float3 workingFromXyz(float3 xyz) {
    return float3(dot(gWorkingSpace.fromXyz0.xyz, xyz), dot(gWorkingSpace.fromXyz1.xyz, xyz),
                  dot(gWorkingSpace.fromXyz2.xyz, xyz));
}

/// A working-space colour in CIE XYZ (D65).
public float3 workingToXyz(float3 c) {
    return float3(dot(gWorkingSpace.toXyz0.xyz, c), dot(gWorkingSpace.toXyz1.xyz, c),
                  dot(gWorkingSpace.toXyz2.xyz, c));
}

/// Decoded sRGB -- linear light with Rec.709's primaries -- in the working
/// space. Where those are the working space's own, the colour as it is.
public float3 workingFromSrgbLinear(float3 c) {
    if ((gWorkingSpace.flags.x & kWorkingSpaceRec709Primaries) != 0u) {
        return c;
    }
    return float3(dot(gWorkingSpace.fromSrgb0.xyz, c), dot(gWorkingSpace.fromSrgb1.xyz, c),
                  dot(gWorkingSpace.fromSrgb2.xyz, c));
}

/// A working-space colour in linear Rec.709: what the display's own views
/// take. Where the primaries are Rec.709's, the colour as it is.
public float3 workingToSrgbLinear(float3 c) {
    if ((gWorkingSpace.flags.x & kWorkingSpaceRec709Primaries) != 0u) {
        return c;
    }
    return float3(dot(gWorkingSpace.toSrgb0.xyz, c), dot(gWorkingSpace.toSrgb1.xyz, c),
                  dot(gWorkingSpace.toSrgb2.xyz, c));
}
