// Copyright (c) 2026 jesus luque.
//
// THE RECORD OF A WORKING SPACE THAT IS NOT LINEAR REC.709, BUILT ON THE
// DEVICE (colour::workingSpaceRecord).
//
// Two matrices OCIO compiled, taken at the basis (colour_matrix.slang):
// linear Rec.709 to the working space, and back. Everything else follows from
// them and from Rec.709's own record (gpu::linRec709Scene, the table the
// kernels held), so the XYZ rows a blackbody is brought through are
// Rec.709's composed with the config's own matrix, and the luminance is the
// Y row that results:
//
//   toXyz     = rec709.toXyz   * toRec709
//   fromXyz   = fromRec709     * rec709.fromXyz
//   fromSrgb  = fromRec709,  toSrgb = toRec709
//   luminance = toXyz's second row
//
// `nonLinear` counts the probes that found a function no 3x3 is (a working
// space has to be linear and scene-referred: colour_matrix.slang), and
// `roundTrip` the entries of fromRec709 * toRec709 more than 1e-4 from the
// identity -- two functions that are not each other's inverse. The host
// reads the record and the two counts back once, when the working space is
// chosen: the record is a uniform every kernel binds, and a uniform is set
// from the host.
module working_space_build;
import athenea.common.working_space;

StructuredBuffer<float4>         fromRec709;   // colour_matrix rows: lin_rec709 -> working
StructuredBuffer<float4>         toRec709;     // working -> lin_rec709
RWStructuredBuffer<WorkingSpace> record;
RWStructuredBuffer<uint>         counts;       // [nonLinear, roundTrip]
uniform WorkingSpace             rec709;       // the table of lin_rec709_scene
uniform float                    linearBound;  // the departure a probe may show

float3x3 rowsOf(float4 a, float4 b, float4 c) {
    return float3x3(a.xyz, b.xyz, c.xyz);
}

void store(float3x3 m, out float4 a, out float4 b, out float4 c) {
    a = float4(m[0], 0.0);
    b = float4(m[1], 0.0);
    c = float4(m[2], 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void workingSpaceBuild(uint3 tid: SV_DispatchThreadID) {
    const float3x3 from709 = rowsOf(fromRec709[0], fromRec709[1], fromRec709[2]);
    const float3x3 to709 = rowsOf(toRec709[0], toRec709[1], toRec709[2]);
    const float3x3 toXyz = mul(rowsOf(rec709.toXyz0, rec709.toXyz1, rec709.toXyz2), to709);
    const float3x3 fromXyz = mul(from709, rowsOf(rec709.fromXyz0, rec709.fromXyz1, rec709.fromXyz2));
    WorkingSpace s = (WorkingSpace)0;
    store(toXyz, s.toXyz0, s.toXyz1, s.toXyz2);
    store(fromXyz, s.fromXyz0, s.fromXyz1, s.fromXyz2);
    store(from709, s.fromSrgb0, s.fromSrgb1, s.fromSrgb2);
    store(to709, s.toSrgb0, s.toSrgb1, s.toSrgb2);
    s.luminance = float4(toXyz[1], 0.0);
    s.flags = uint4(0u);
    record[0] = s;
    counts[0] = (fromRec709[3].x > linearBound ? 1u : 0u) + (toRec709[3].x > linearBound ? 1u : 0u);
    const float3x3 identity = mul(from709, to709);
    uint off = 0;
    for (uint r = 0; r < 3; ++r) {
        for (uint c = 0; c < 3; ++c) {
            off += abs(identity[r][c] - (r == c ? 1.0 : 0.0)) > 1e-4 ? 1u : 0u;
        }
    }
    counts[1] = off;
}
