# How gaussians move

A cloud in athenea moves in one of five ways. Which one depends on where the
motion comes from (a skeleton, a USD transform, a fluid's velocity) and on
where the cloud is drawn (the desktop engine, or a browser at the end of a
stream). This page names each one, says where it is chosen and what it costs,
and ends with motion blur, which they all share.

| Motion | Where it comes from | Where it is chosen | Drawn by |
|---|---|---|---|
| [Skinning on the device](#skinning-on-the-device) | a UsdSkel skeleton | `mesh2splat --skinned`, `athenea visibility` | the engine, every frame |
| [A skeleton in the browser](#a-skeleton-in-the-browser) | the same skeleton, sent once | the Spark WebGPU route | the browser |
| [A stream's geometry](#a-streams-geometry) | frames already posed or fitted | `video-encode --geometry-predict` | the `.athv` player |
| [A volume carried by its flow](#a-volume-carried-by-its-flow) | the simulation's `vel` grid | `vdb2splat --sequence` | the engine, or the `.athv` player |
| [USD transforms](#usd-transforms) | `xformOp`s, time-sampled | the stage | the engine |

## Skinning on the device

`athenea mesh2splat --skinned` builds the gaussians in the bind pose and gives
each one the joints that carry it, so the cloud is deformed at render time by
the Skeleton it is bound to, on the GPU, like the mesh it came from. Each
gaussian also keeps how its weights change across it
(`jointWeightGradients`, twelve bytes a gaussian): that is what makes it
stretch across a bend as its triangle does, instead of moving rigidly with the
blend of its joints.

```sh
athenea mesh2splat bird.usda --skinned --resolution 1100 -o bird_gs.usdc
athenea visibility bird_gs.usdc --skeleton-stage bird.usda --skeleton-prim /World/Skel
```

A light bake is refused with `--skinned`, because light baked in one pose is
wrong in every other. A transfer bake is kept instead, as two zonal lobes in
each gaussian's own frame, which turn with it. `athenea visibility` bakes what
the cloud casts on itself, by part, so a wing shadows the body at every pose
without tracing a ray (`SplatVisibility`). The sparrow (5 to 9 million
gaussians, 609 joints) moves this way.

## A skeleton in the browser

The other way to show a skinned cloud on the web is to send it once, with its
skeleton and its transfer, and animate and relight it in the browser. That is
the [Spark WebGPU](https://sparkwebgpu.lucab.co.uk/sparrow) route: the
gaussians travel in a `.athc` container, are skinned in the browser, and are
lit by the HDR the viewer picks. Once the cloud has loaded, it costs nothing
more to send; the browser does the work of posing it.

## A stream's geometry

A cloud that has been posed or fitted frame by frame can be sent as video
(`athenea video-encode`, the `.athv` format). Colour and opacity travel in
the video's planes; position, scale and rotation travel beside it, exactly,
in a geometry layer (`--geometry`). Between geometry keys
(`--geometry-key N`), each gaussian sends only its difference from what the
decoder predicts, and the prediction is the motion:

| `--geometry-predict` | Prediction | Suits |
|---|---|---|
| `1` | the frame before | anything with stable slots |
| `2` | the frame before, its motion carried on (2b − c) | smooth, steady motion |
| `3` | each run of 256 gaussians' rigid motion, fitted from what the decoder holds (32 bytes a run), the rotation sent as a turn in the gaussian's own frame | a skinned cloud: its runs move as rigid pieces |

The prediction needs `--order stable`: gaussian *k* must be the same gaussian
in every frame. `--geometry-stagger` refreshes blocks of 256 gaussians in turn
instead of whole key frames, which keeps the bit rate flat.
`--geometry-hold P,S,R` keeps a value the decoder already holds when the new
one is within that many steps of it.

On the sparrow, order 3 sends a third fewer bytes than order 1 at the same
picture (PSNR 34.15 against 34.17 dB at 150,000 gaussians, flicker
unchanged):

| Rung | Order 1, Mbit/s at 30 fps | Order 3 |
|---|---|---|
| 150k | 148.9 | 100.5 |
| 75k | 76.6 | 57.4 |
| 40k | 44.9 | 34.4 |
| 20k | 25.9 | 20.0 |
| 10k | 21.7 | 16.9 |

The [web player](https://atheneastream.lucab.co.uk/) decodes all three
orders. Simulated volumes stay at order 1: their gaussians do not move in
rigid runs, and order 3 saves nothing on them.

## A volume carried by its flow

A sequence of `.vdb` frames that holds a velocity grid is fitted with stable
slots: each frame starts from the frame before's gaussians, carried by the
flow. The centre moves with the velocity, and the covariance is deformed by
the flow's gradient (F = I + Δt ∇u, its volume kept; where the flow shears too
hard, only the centre is carried). Then a few optimisation steps correct the
frame.

```sh
athenea vdb2splat "smoke_####.vdb" --sequence --frames 1 120 -n 20000 \
    --velocity velocity --velocity-units mantaflow --fps 24 -o "smoke/f_####.usdc"
```

`--velocity` names the grid (`vel` by default). A staggered (MAC) grid, such
as Mantaflow's, is averaged to the voxel centres. `--velocity-units
mantaflow` reads it in Mantaflow's units. Because the slots are stable, the
stream can predict them (`--geometry-predict 1`, with `--slots` fixing the
count). A file without a velocity, such as EmberGen's, is fitted frame by
frame instead (`--independent`). See [Volumes](volumes.md).

## USD transforms

A cloud is a USD prim, so its `xformOp`s move it as they move a mesh,
time-sampled or not. Nothing is
converted: the gaussians stay where they are in the cloud's own space, and the
transform is applied when the frame is drawn.

## Motion blur

Under a shutter (the camera's shutter, `athenea:shutter` on the render
settings, or `athenea stage --shutter OPEN:CLOSE` for a camera made by `--eye`), the raster blurs each gaussian in one pass. A gaussian smeared
along a segment d over the shutter is that gaussian convolved with the
segment, so its 2D screen covariance gains one rank-one term,
S' = S + d dᵀ / 12, and its peak comes down to keep its energy. The
displacement comes from the cloud's transform and the camera over the shutter
(one difference, `V_close M_close − V_open M_open`) and, for a skinned cloud,
from what the skeleton moved each gaussian. The sparrow at 9.5 million
gaussians draws in 70 ms still and 164 ms under a 180-degree shutter
(900 × 675).

The path tracer blurs meshes by cutting the shutter into buckets (1 to 8,
`--motion-buckets`); it still draws a cloud at the frame's instant.

## Where the details are

- [`docs/operations.md` §2.8](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#28-athenea-mesh2splat--a-model-into-gaussians) (`--skinned`, `--range`, `--keep-order`), §2.9 (`athenea visibility`), [§2.7.2](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#272-athenea-video-encode--splat-frames-as-hardware-video) (the geometry layer), [§2.8.3](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#283-athenea-vdb2splat--a-volume-into-volumetric-gaussians-and-back) (sequences and velocity).
- `docs/decisions.md`: "The whole Jacobian of a skinned gaussian", "A splat under a shutter: one rank-one term, not five passes", "Motion blur: the shutter in buckets"; `docs/operations.md` §8.2, the `.athv` format.
