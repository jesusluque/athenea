// Copyright (c) 2026 jesus luque.
//
// A split bake into the pictures the splat bake filter reads, and its answer
// back (`athenea mesh2splat`, plugins/splatbakefilter).
//
// The filter is an AOFX effect, so what it is handed is pictures: entries in
// rows of `width`, padded to `stride` pixels. Points, three entries a
// gaussian -- where it is and how wide (the bake's own ray's point, and the
// footprint its normal carries in w or else the record's size), which way it
// faces and the Cryptomatte id of the prim it came from (as the float its bits
// are), and the luminance of what is filtered with the variance of its mean
// (the split bake's guides) -- and the light's harmonics, `coefficients`
// entries a gaussian. What comes back is the second picture's shape, unpacked
// into the split bake's indirect half.
//
// THE DIRECT HALF MAY GO WITH IT (`withDirect`). On a model under a sky the
// grain is the direct light's, not the indirect's -- sixteen harmonics fitted
// from a few hundred paths, each looking from another direction at a sky with
// a sun in it -- so the whole light is filtered, weighed by the whole light's
// noise: a shadow's edge is a step many times that noise and stays
// (docs/decisions.md, "The bake's grain"). The direct half is then left empty
// and the other holds both.
module bake_filter_io;

struct FilterIoParams {
    uint count;
    uint coefficients;
    uint width;     ///< entries a row of the picture
    uint stride;    ///< pixels a row of it, padded
    uint perRecord; ///< floats a record
    uint size;      ///< the record's first size (its width along the surface)
    uint withDirect; ///< 1: the direct half goes too, weighed by the whole light's noise
    uint pad1;
};

StructuredBuffer<float4>       rays;       // 3 a gaussian (bakePointsOnDevice's layout)
StructuredBuffer<float>        records;    // the cloud's own, `perRecord` floats a gaussian
StructuredBuffer<uint>         ids;        // a gaussian: its prim's Cryptomatte id
StructuredBuffer<float4>       guides;     // a gaussian: (indirect luminance, its variance, the whole light's, its variance)
RWStructuredBuffer<float4>     indirect;   // coefficients a gaussian
RWStructuredBuffer<float4>     direct;     // the same of the direct half
RWStructuredBuffer<float4>     picture;
ConstantBuffer<FilterIoParams> io;

uint at(uint entry) {
    return (entry / io.width) * io.stride + entry % io.width;
}

/// Which way the gaussian faces: its own where it said one, else the surface's.
float3 facingOf(uint k) {
    const float4 own = rays[k * 3 + 2];
    return own.w > 0.5 ? normalize(own.xyz) : normalize(rays[k * 3 + 1].xyz);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bakeFilterPoints(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= io.count) {
        return;
    }
    const float4 point = rays[k * 3];
    // How wide it is: the footprint the bake spread its paths over where
    // there is one (a simplified block), else the record's own size.
    const float spread = rays[k * 3 + 1].w;
    const float footprint = spread > 0.0 ? spread : records[k * io.perRecord + io.size];
    picture[at(k * 3)] = float4(point.xyz, footprint);
    picture[at(k * 3 + 1)] = float4(facingOf(k), asfloat(ids[k]));
    // The noise the filter weighs by is the noise of what it filters: the
    // indirect half's, or the whole light's when the direct goes too.
    const float4 g = guides[k];
    picture[at(k * 3 + 2)] = io.withDirect != 0 ? float4(g.z, g.w, 0.0, 0.0) : float4(g.x, g.y, 0.0, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bakeFilterIn(uint3 tid: SV_DispatchThreadID) {
    const uint e = tid.x;
    if (e >= io.count * io.coefficients) {
        return;
    }
    picture[at(e)] = indirect[e] + (io.withDirect != 0 ? direct[e] : float4(0.0));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bakeFilterOut(uint3 tid: SV_DispatchThreadID) {
    const uint e = tid.x;
    if (e >= io.count * io.coefficients) {
        return;
    }
    indirect[e] = picture[at(e)];
    if (io.withDirect != 0) {
        direct[e] = float4(0.0);   // now in the other half, filtered with it
    }
}
