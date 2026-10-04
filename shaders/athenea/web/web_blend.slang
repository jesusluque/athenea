// Copyright (c) 2026 jesus luque.
//
// THE WEB'S BLEND, PRESET T1: per pixel, its tile's splats nearest first until
// the pixel is opaque, over the frame's background -- splat_blend.slang's
// splatBlend without what the preset leaves out (the slope of a TX
// reflection, the layer under the splats, the matte, the depth), so that it
// binds four storage buffers and holds 12296 bytes of group memory: under the
// web's default 16 KiB, where splatBlend's 17416 are not (a `uint3` takes 16
// bytes in a WGSL array).
//
// The walk is splatBlend's, barrier for barrier: the group is the tile, it
// loads 256 records into group memory at a time, and decides to stop through
// workgroupUniformLoad, the one read Tint's uniformity analysis accepts
// (docs/decisions.md, "WebGPU"). It becomes splatBlend's specialisation once
// the arena change reaches the native kernels.
module web_blend;
import athenea.splat.frame;

StructuredBuffer<uint>       ranges;
StructuredBuffer<uint>       pairSplats;
StructuredBuffer<ProjRecord> proj;
RWStructuredBuffer<float4>   colour;   // width * height, linear, premultiplied, bottom row first
ConstantBuffer<FrameParams>  params;

static const uint kBatch = kTile * kTile;
groupshared ProjRecord   sRecords[kBatch];
groupshared Atomic<uint> sDone;
groupshared uint         sDoneAll;

[shader("compute")]
[numthreads(16, 16, 1)]
void webBlend(uint3 tid: SV_DispatchThreadID, uint3 groupThread: SV_GroupThreadID, uint3 group: SV_GroupID) {
    const uint px = tid.x;
    const uint py = tid.y;
    const uint lane = groupThread.y * kTile + groupThread.x;
    const bool inside = px < params.width && py < params.height;
    const uint tile = group.y * params.tilesX + group.x;
    const uint start = ranges[tile * 2];
    const uint end = ranges[tile * 2 + 1];
    const float2 pixel = float2(float(px) + 0.5, float(py) + 0.5);

    float3 rgb = float3(0.0);
    float transmittance = 1.0;
    bool done = !inside;
    for (uint first = start; first < end; first += kBatch) {
        if (lane == 0) {
            sDone.store(0u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (done) {
            sDone.add(1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            sDoneAll = sDone.load();
        }
        GroupMemoryBarrierWithGroupSync();
        if (workgroupUniformLoad(sDoneAll) == kBatch) {
            break;
        }
        const uint mine = first + lane;
        if (mine < end) {
            sRecords[lane] = proj[pairSplats[mine]];
        }
        GroupMemoryBarrierWithGroupSync();
        const uint count = min(kBatch, end - first);
        for (uint j = 0; j < count && !done; ++j) {
            const ProjRecord record = sRecords[j];
            const float2 d = pixel - record.screen.xy;
            const float4 conic = record.conic;
            const float power = conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y;
            if (power > record.screen.w) {
                continue;
            }
            const float alpha = min(0.99, conic.w * exp(-0.5 * power));
            if (alpha < 1.0 / 255.0) {
                continue;
            }
            rgb += transmittance * alpha * record.colour.xyz;
            transmittance *= 1.0 - alpha;
            if (transmittance < 1e-4) {
                done = true;
                break;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (!inside) {
        return;
    }
    const float4 behind = float4(params.bgR, params.bgG, params.bgB, params.bgA);
    colour[py * params.width + px] = float4(rgb + transmittance * behind.rgb, 1.0 - transmittance + transmittance * behind.w);
}
