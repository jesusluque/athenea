// Copyright (c) 2026 jesus luque.
//
// Which prim each pixel's surface belongs to, as Cryptomatte names one: the
// visibility buffer says the instance, the instance says its prim id, and a
// table the frame built says what that prim's path hashes to. Zero where
// nothing was drawn, which the matte reads as absence.
module crypto_ids;
import camera;
import athenea.world.scene_types;

struct CryptoIdParams {
    uint primCount;   // rows in `cryptoOfPrim`
    uint pad0; uint pad1; uint pad2;
};

Texture2D<uint4>              visibility;
StructuredBuffer<InstanceRecord> instances;
StructuredBuffer<uint>        cryptoOfPrim;   // the id of prim id p, or 0
RWStructuredBuffer<uint>      cryptoIds;      // one a pixel, bottom row first
ConstantBuffer<CameraParams>  camera;
ConstantBuffer<CryptoIdParams> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void cryptoIdsPass(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    if (seen.x == 0) {
        cryptoIds[at] = 0;
        return;
    }
    // The instance record carries the prim id Hydra gave the prim, composed
    // for an instanced one as well, so one table serves both.
    const uint prim = instances[seen.x - 1].primId;
    cryptoIds[at] = prim < params.primCount ? cryptoOfPrim[prim] : 0u;
}
