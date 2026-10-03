// Copyright (c) 2026 jesus luque.
//
// What one run of the mesh2splat effect wrote, into the cloud's own buffers,
// on the device.
//
// The effect writes its records into a picture: `recordEntries` float4
// entries a gaussian, in rows of `width` entries padded to `stride` pixels --
// (position, opacity), the three sizes, the rotation, the colour, then
// (shading normal, metallic) and (roughness, transmission, u, v), two more
// with the joints that carry it, and three more where the material displaces
// (the flat surface under it, that surface's normal, the relief's). This
// lays them out as the cloud keeps them, at `destFirst` in buffers that hold
// every run of the conversion:
//
//   records     `perRecord` floats a gaussian, `io::RawSplats`' layout as
//               `athenea mesh2splat` declares it (the canonical fourteen, then
//               metallic, roughness, transmission and the shading normal;
//               the harmonics after it are zeros until the bake writes them)
//   rays        three float4 a gaussian, the bake's own layout
//               (`technique::BakePoints`): the point of the surface under it,
//               w left for the ray's offset (mesh2splat_span), that
//               surface's normal with the gaussian's footprint in w where
//               `overArea`, and the way the relief turned it (w 1) or zeros
//   influences  two float4 a gaussian, (joint, weight) four times, where a
//               skeleton carries it
//
// It used to be done a splat at a time on the processor, out of a picture
// read back for the purpose; nothing here computes, it only moves.
module mesh2splat_gather;

struct GatherParams {
    uint count;           ///< gaussians this run wrote
    uint destFirst;       ///< the first of them in the whole cloud
    uint recordEntries;   ///< float4 entries a gaussian in the picture
    uint ownEntries;      ///< the record's own; the displaced three follow them
    uint width;           ///< entries a row of the picture
    uint stride;          ///< pixels a row of it, padded
    uint perRecord;       ///< floats a record in `records`
    uint displaced;       ///< the picture carries the flat surface and the relief
    uint carried;         ///< and the joints
    uint overArea;        ///< the footprint goes to the bake (a simplified block)
    uint pad0;
    uint pad1;
};

StructuredBuffer<float4>     picture;
RWStructuredBuffer<float>    records;
RWStructuredBuffer<float4>   rays;
RWStructuredBuffer<float4>   influences;
ConstantBuffer<GatherParams> gather;

float4 entryOf(uint splat, uint component) {
    const uint index = splat * gather.recordEntries + component;
    return picture[(index / gather.width) * gather.stride + index % gather.width];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sGather(uint3 tid: SV_DispatchThreadID) {
    const uint splat = tid.x;
    if (splat >= gather.count) {
        return;
    }
    const float4 position = entryOf(splat, 0);
    const float4 sizes = entryOf(splat, 1);
    const float4 rotation = entryOf(splat, 2);
    const float4 rgb = entryOf(splat, 3);
    const float4 shading = entryOf(splat, 4);
    const float4 surface = entryOf(splat, 5);

    const uint out = gather.destFirst + splat;
    const uint at = out * gather.perRecord;
    records[at + 0] = position.x;
    records[at + 1] = position.y;
    records[at + 2] = position.z;
    records[at + 3] = position.w;
    records[at + 4] = sizes.x;
    records[at + 5] = sizes.y;
    records[at + 6] = sizes.z;
    records[at + 7] = rotation.x;
    records[at + 8] = rotation.y;
    records[at + 9] = rotation.z;
    records[at + 10] = rotation.w;
    records[at + 11] = rgb.x;
    records[at + 12] = rgb.y;
    records[at + 13] = rgb.z;
    records[at + 14] = shading.w;   // metallic
    records[at + 15] = surface.x;   // roughness
    records[at + 16] = surface.y;   // transmission
    records[at + 17] = shading.x;   // the shading normal
    records[at + 18] = shading.y;
    records[at + 19] = shading.z;
    // The harmonics' room, zero until a bake writes it: a cloud that is not
    // baked declares none, and one whose bake found nothing keeps these.
    for (uint k = 20; k < gather.perRecord; ++k) {
        records[at + k] = 0.0;
    }

    // WHERE THE BAKE STANDS AND WHICH WAY IT LOOKS. A displaced gaussian is
    // baked from the flat surface under it -- the only one the tracer holds
    // -- and faces as the relief does.
    const float footprint = gather.overArea != 0 ? sizes.x : 0.0;
    if (gather.displaced != 0) {
        const float4 flat = entryOf(splat, gather.ownEntries);
        const float4 flatNormal = entryOf(splat, gather.ownEntries + 1);
        const float4 relief = entryOf(splat, gather.ownEntries + 2);
        rays[out * 3 + 0] = float4(flat.xyz, 0.0);
        rays[out * 3 + 1] = float4(flatNormal.xyz, footprint);
        rays[out * 3 + 2] = float4(relief.xyz, 1.0);
    } else {
        rays[out * 3 + 0] = float4(position.xyz, 0.0);
        rays[out * 3 + 1] = float4(shading.xyz, footprint);
        rays[out * 3 + 2] = float4(0.0);
    }

    if (gather.carried != 0) {
        influences[out * 2 + 0] = entryOf(splat, 6);
        influences[out * 2 + 1] = entryOf(splat, 7);
    }
}

/// A gaussian nothing carries, in a cloud a skeleton does: four joints of no
/// weight, which the skinner reads as "leave it where the bind pose put it".
/// The influences stay one to one with the gaussians.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sNoInfluence(uint3 tid: SV_DispatchThreadID) {
    const uint splat = tid.x;
    if (splat >= gather.count) {
        return;
    }
    const uint out = gather.destFirst + splat;
    influences[out * 2 + 0] = float4(0.0);
    influences[out * 2 + 1] = float4(0.0);
}
