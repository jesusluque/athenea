// Copyright (c) 2026 jesus luque.
//
// A VOLUMETRIC CLOUD'S MEDIUM (GpuSplats::volume, proposal 105): what each
// gaussian of a `primvars:athenea:splat:volume` field carries besides its
// shape -- the extinction's peak sigma (density units), the temperature and
// the albedo -- packed two words a record as the stage holds them:
//
//   word 0   sigma | temperature << 16      halves
//   word 1   albedo rgb, 8 bits each (the fit's own precision), 255 in the top byte
//
// `volumePack` reads the three primvars (floats or halves, the temperature
// optional) and writes a record's words; validation's own packing to the
// kept splats is CloudLoader::keptOnly's.
module volume_pack;
import athenea.common.packing;

/// Value `index` of a float array: float32 words, or float16 halves packed two
/// to a little-endian word, first in the low bits (streams.slang's).
float streamValue(StructuredBuffer<uint> words, bool half, uint index) {
    if (half) {
        const uint word = words[index / 2];
        return (index & 1u) == 0 ? lowHalf(word) : highHalf(word);
    }
    return bit_cast<float>(words[index]);
}

struct VolumePackParams {
    uint count;          ///< records
    uint sigmaHalf;      ///< 1: the sigma stream is float16
    uint albedoKind;     ///< 0: none (white), 1: float, 2: half
    uint temperatureKind;///< 0: none (0), 1: float, 2: half
};

StructuredBuffer<uint>            sigmaStream;         // one a record
StructuredBuffer<uint>            albedoStream;        // rgb a record
StructuredBuffer<uint>            temperatureStream;   // one a record
RWStructuredBuffer<uint>          packedVolume;        // two words a record
ConstantBuffer<VolumePackParams>  volumePackParams;

[shader("compute")]
[numthreads(256, 1, 1)]
void volumePack(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    const VolumePackParams p = volumePackParams;
    if (i >= p.count) {
        return;
    }
    const float sigma = max(streamValue(sigmaStream, p.sigmaHalf != 0, i), 0.0);
    const float temperature =
        p.temperatureKind == 0 ? 0.0 : max(streamValue(temperatureStream, p.temperatureKind == 2, i), 0.0);
    float3 albedo = float3(1.0);
    if (p.albedoKind != 0) {
        const bool half = p.albedoKind == 2;
        albedo = float3(streamValue(albedoStream, half, i * 3), streamValue(albedoStream, half, i * 3 + 1),
                        streamValue(albedoStream, half, i * 3 + 2));
    }
    albedo = saturate(albedo);
    const uint3 bytes = uint3(albedo * 255.0 + 0.5);
    packedVolume[i * 2] = (f32tof16(sigma) & 0xFFFFu) | (f32tof16(temperature) << 16);
    packedVolume[i * 2 + 1] = bytes.x | (bytes.y << 8) | (bytes.z << 16) | (255u << 24);
}

// WHERE A MEDIUM REACHES: every gaussian's centre plus and minus 3.5 times its
// widest deviation, two points a gaussian, for CloudLoader::boundsOf. A
// medium's gaussians are wide, and the box of their centres alone fell short
// of the smoke: a shadow map sized from it cut the shadow's far edge.
struct VolumeReachParams {
    uint count;
};
StructuredBuffer<float4>          reachPositions;
StructuredBuffer<uint>            reachShape;       // 4 words a gaussian, as GpuSplats packs them
RWStructuredBuffer<float4>        reachCorners;     // two a gaussian
ConstantBuffer<VolumeReachParams> reachParams;

[shader("compute")]
[numthreads(256, 1, 1)]
void volumeReach(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= reachParams.count) {
        return;
    }
    const float3 centre = reachPositions[i].xyz;
    const uint row = i * 4;
    const float3 scale = exp(float3(lowHalf(reachShape[row + 1]), highHalf(reachShape[row + 1]),
                                    lowHalf(reachShape[row + 2])));
    const float reach = 3.5 * max(scale.x, max(scale.y, scale.z));
    reachCorners[i * 2] = float4(centre - reach, 1.0);
    reachCorners[i * 2 + 1] = float4(centre + reach, 1.0);
}
