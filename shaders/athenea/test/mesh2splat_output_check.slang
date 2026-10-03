// Copyright (c) 2026 jesus luque.
//
// What a converted cloud carries, counted by colour: a file `athenea mesh2splat`
// wrote, read back as records (`usd::readParticleFieldRecords`: the constant
// term as 3DGS keeps it, in the space a cloud is blended in) and counted here.
//
// counts[0] red, [1] blue, [2] green, [3] any other colour, [4] every
// gaussian that is there at all (opacity above zero). A colour is a primary
// when its channel reads above 0.6 and the other two below 0.3.
module mesh2splat_output_check;

struct OutputParams {
    uint  count;
    uint  stride;    ///< floats a record
    uint  opacity;   ///< the opacity's field
    uint  dc0;       ///< the constant term's first field
};

StructuredBuffer<float>      records;
RWStructuredBuffer<uint>     counts;
ConstantBuffer<OutputParams> params;

static const float kSH0 = 0.28209479177387814;

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sColourClasses(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const uint at = k * params.stride;
    if (!(records[at + params.opacity] > 0.0)) {
        return;
    }
    InterlockedAdd(counts[4], 1u);
    const float3 c = 0.5 + kSH0 * float3(records[at + params.dc0], records[at + params.dc0 + 1],
                                         records[at + params.dc0 + 2]);
    const bool red = c.r > 0.6 && c.g < 0.3 && c.b < 0.3;
    const bool blue = c.b > 0.6 && c.r < 0.3 && c.g < 0.3;
    const bool green = c.g > 0.6 && c.r < 0.3 && c.b < 0.3;
    InterlockedAdd(counts[red ? 0 : blue ? 1 : green ? 2 : 3], 1u);
}
