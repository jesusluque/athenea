// Copyright (c) 2026 jesus luque.
//
// The cell every piece of a conversion walks, worked out on the device.
//
// `athenea mesh2splat` hands the effect a resolution and a cell's bounds, and the
// effect works out each triangle's cell from them. What the command needs
// besides -- the model's cell (a level of detail is written with it), the
// bounds it derives when the user gave none, the cell each piece will walk
// (for the log, and as the cell itself where a camera decides it), and how
// much coarser a piece must walk to fit its share of the budget -- used to be
// three operations on six numbers read back, on the processor. They are
// here, over the boxes the device folded (`mesh_pack`, `bounds_reduce`,
// `bounds_reduce_slices`), and what crosses back is the answer.
//
//   cells[0]           the model's cell: its longest side over the resolution
//   cells[1], cells[2] the cell's bounds sent to the effect (0: none). Per
//                      mesh, derived where not given: no coarser than the
//                      model's cell, no finer than an eighth of it
//   cells[3 + p]       the cell piece p walks, coarsened by its factor
//   cells[3 + P + p]   that factor: sqrt(wanted / share) where the budget
//                      gives the piece less than it wants, 1 otherwise. A
//                      cell `f` times coarser covers the same surface with
//                      `f^2` fewer gaussians, so the piece then wants about
//                      its share
//   cells[3 + 2P + p]  what the effect is given for the piece: the
//   cells[3 + 3P + p]  resolution, and the cell's two bounds, all three
//   cells[3 + 4P + p]  coarsened by the factor -- or, from a camera, the
//                      piece's cell as both bounds, so every triangle walks
//                      exactly it
//
// FROM A CAMERA (Mesh2GS, arXiv 2606.21898): a piece's cell is what one
// pixel of the camera covers where the piece is nearest to it, `z * (aperture
// / pixels) / focal`, with `z` the distance from the camera to the piece's
// mesh's box (0 inside it, held to the near plane). One cell, one pixel, in
// the camera that will look -- the same rule for every mesh, which is what
// makes the density fair. The user's `--cell-min` and `--cell-max` bound it.
module mesh2splat_cells;

struct CellParams {
    uint   pieces;
    uint   resolution;
    float  cellMin;        ///< the user's, 0: derived
    float  cellMax;

    uint   perMesh;        ///< each mesh's own box
    uint   camera;         ///< the cell from the camera
    float  aperture;       ///< the camera's horizontal aperture
    float  focal;          ///< its focal length, in the aperture's units

    float  pixels;         ///< pixels across its image
    float  near;           ///< the nearest it sees
    float  pad0;
    float  pad1;

    float4 toWorld0;       ///< the camera's transform, three rows of four
    float4 toWorld1;
    float4 toWorld2;
};

StructuredBuffer<float4>   bounds;      // the model's low and high, then each mesh's
StructuredBuffer<uint>     pieceMesh;   // the mesh each piece is of
StructuredBuffer<uint>     wanted;      // what each piece wants; 0 before it is counted
StructuredBuffer<uint>     share;       // what the budget gives it; 0: all it wants
RWStructuredBuffer<float>  cells;
ConstantBuffer<CellParams> params;

float longestOf(float3 low, float3 high) {
    const float3 extent = high - low;
    return max(max(extent.x, extent.y), max(extent.z, 1.0e-20));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sCells(uint3 tid: SV_DispatchThreadID) {
    const uint p = tid.x;
    const float resolution = float(max(params.resolution, 1u));
    const float modelCell = longestOf(bounds[0].xyz, bounds[1].xyz) / resolution;
    const bool perMesh = params.perMesh != 0 && params.camera == 0;
    const float cellMax = params.cellMax > 0.0 ? params.cellMax : (perMesh ? modelCell : 0.0);
    const float cellMin = params.cellMin > 0.0 ? params.cellMin : (perMesh ? cellMax / 8.0 : 0.0);
    if (p == 0) {
        cells[0] = modelCell;
        cells[1] = cellMin;
        cells[2] = cellMax;
    }
    if (p >= params.pieces) {
        return;
    }
    const uint mesh = pieceMesh[p];
    const float3 low = bounds[2 + mesh * 2].xyz;
    const float3 high = bounds[3 + mesh * 2].xyz;
    float cell;
    if (params.camera != 0) {
        const float3 eye = float3(params.toWorld0.w, params.toWorld1.w, params.toWorld2.w);
        const float3 nearest = clamp(eye, low, high);
        const float z = max(length(nearest - eye), max(params.near, 1.0e-6));
        cell = z * (params.aperture / max(params.pixels, 1.0)) / max(params.focal, 1.0e-6);
    } else {
        const float longest = perMesh ? longestOf(low, high) : longestOf(bounds[0].xyz, bounds[1].xyz);
        cell = longest / resolution;
    }
    if (cellMin > 0.0) {
        cell = max(cell, cellMin);
    }
    if (cellMax > 0.0) {
        cell = min(cell, cellMax);
    }
    const uint want = wanted[p];
    const uint given = share[p];
    const float factor = given > 0 && want > given ? sqrt(float(want) / float(given)) : 1.0;
    const uint n = params.pieces;
    cells[3 + p] = cell * factor;
    cells[3 + n + p] = factor;
    if (params.camera != 0) {
        cells[3 + 2 * n + p] = resolution;
        cells[3 + 3 * n + p] = cell * factor;
        cells[3 + 4 * n + p] = cell * factor;
    } else {
        cells[3 + 2 * n + p] = resolution / factor;
        cells[3 + 3 * n + p] = cellMin * factor;
        cells[3 + 4 * n + p] = cellMax * factor;
    }
}
