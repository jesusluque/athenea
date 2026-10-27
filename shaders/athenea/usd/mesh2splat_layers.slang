// Copyright (c) 2026 jesus luque.
//
// THE EMISSION GROUPS' LAYERS OF A CONVERSION (`mesh2splat --light-groups`):
// the answer `StageRenderer::bakeGroupsOnDevice` laid out, a gaussian's groups
// together as float4, written as a layer a group of three floats a gaussian,
// the rgb a file keeps (`primvars:athenea:splat:lightLayer:emission:<group>`).
module mesh2splat_layers;

struct LayersParams {
    uint count;    ///< gaussians
    uint groups;   ///< float4 a gaussian in `answer`
};

StructuredBuffer<float4>     answer;
RWStructuredBuffer<float>    layers;   // groups x count x 3
ConstantBuffer<LayersParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sLayersInto(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    for (uint g = 0u; g < params.groups; ++g) {
        const float3 light = answer[i * params.groups + g].rgb;
        const uint at = (g * params.count + i) * 3u;
        layers[at] = light.r;
        layers[at + 1u] = light.g;
        layers[at + 2u] = light.b;
    }
}
