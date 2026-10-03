// Copyright (c) 2026 jesus luque.
//
// A COMPILED COLOUR FUNCTION AS A 3x3, TAKEN ON THE DEVICE.
//
// Every linear RGB space is a 3x3 from every other. A kernel that needs one
// -- the working space's record, a linear cloud's colours brought into the
// working space -- does not import the function: a generated module
// (colour::ColourCompiler::matrix, `athenea_cm_<hash>`) evaluates the
// compiled function at the basis and hands the three values to
// `colourMatrixWrite`, which lays them out as rows. Two more evaluations,
// at e_r + e_g and 2 e_g, say whether the function is linear at all: their
// distance from what a matrix would give goes in the fourth row, which a
// consumer counts against a bound (a log or a curve is not a 3x3).
module colour_matrix;

/// Rows 0..2: the matrix, w 0. Row 3: x, the worst departure from linearity
/// over the two probes, relative to their size.
public RWStructuredBuffer<float4> matrixRows;

public void colourMatrixWrite(float3 r, float3 g, float3 b, float3 rg, float3 g2) {
    matrixRows[0] = float4(r.x, g.x, b.x, 0.0);
    matrixRows[1] = float4(r.y, g.y, b.y, 0.0);
    matrixRows[2] = float4(r.z, g.z, b.z, 0.0);
    const float3 additive = abs(rg - (r + g));
    const float3 homogeneous = abs(g2 - 2.0 * g);
    const float scale = max(max(max(abs(r.x), abs(r.y)), abs(r.z)), max(max(abs(g.x), abs(g.y)), abs(g.z)));
    const float worst = max(max(max(additive.x, additive.y), additive.z), max(max(homogeneous.x, homogeneous.y),
                                                                               homogeneous.z));
    matrixRows[3] = float4(worst / max(scale, 1e-12), 0.0, 0.0, 0.0);
}
