# Volumes: from a VDB to volumetric gaussians

athenea turns an OpenVDB fog volume (smoke, fire, an explosion, a cloud) into
a few tens of thousands of 3D gaussians that stand for the medium itself, not
for a surface. Its extinction is a sum of gaussians, μ_t(x) = Σ σ_i G_i(x),
each G_i of peak 1; the temperature and the colour grids, where the file has
them, ride on each gaussian as density-weighted means. The same gaussians are
drawn by the raster as a participating medium, given back to a grid for the
path tracer, and streamed to a browser as video.

This page follows one public simulation through the whole route: the
**gasoline explosion** from JangaFX's free EmberGen VDBs (CC0, 131 frames,
`density`, `temperature` and `flames` in half floats, up to 3.2 million active
voxels a frame), downloadable from
[jangafx.com](https://jangafx.com/software/embergen/download/free-vdb-animations).

## What the conversion does

`athenea vdb2splat` fits the mixture on the GPU, every step a compute kernel:

1. **Cut.** The density's leaves are cut into blocks (at most `--max-block`
   voxels a side, 64 by default), one gaussian each, placed at the block's
   moments. The first cut holds `--init-fraction` of the budget (0.6).
2. **EM.** `--em-iterations` steps (40) move each gaussian to the moments of
   the density it is responsible for.
3. **Adam.** `--iterations` steps (100) on the centre, the factor of the
   precision and the peak, against the grid's density at its active voxels.
   Every `--densify-every` steps (20) the budget's free slots are given where
   the residual is largest: half of them split the gaussians answerable for
   the most error along their longest axis, the rest are born at the worst
   voxels; a gaussian under `--prune-mass` of the total goes.
4. **Attributes.** With the geometry fixed, `--attribute-iterations` Jacobi
   steps (20) fit the temperature and the colour as density-weighted means.

A grid larger than the device is fitted a brick at a time (`--tile-dir`): the
plan is made from the tree's structure alone, each brick read with OpenVDB's
delayed loading, and a second pass (`--tile-refine`) closes the seams.

## What each gaussian holds

| | Bytes | What |
|---|---|---|
| On the stage | — | a `UsdVolParticleField3DGaussianSplat` with `primvars:athenea:splat:volume = true` and, a gaussian, `primvars:athenea:volume:sigma` (half), `…:albedo` (color3h, from Cd) and `…:temperature` (half) beside the schema's positions, orientations and scales |
| `--quantise half` | 24 | centre in 16 bits a component over the cloud's box, ln σ in 16 bits, the precision's factor in halves, Cd in rgb8, temperature in 8 bits |
| `--quantise byte` | 20 | the same with the factor in bytes over its ranges |

`radiance:base` carries an opacity for a standard viewer, 1 − exp(−τ) through
the gaussian's middle; athenea's own raster ignores it and draws the medium.

## One frame

```sh
GAS=embergen_gasoline_explosion/Gasoline_Explosion_01
athenea vdb2splat $GAS/embergen_gasoline_explosion_a_60.vdb -n 50000 \
    --density-scale 0.03 -o gas60_gs.usdc --back gas60_back.vdb --measure
```

`--back` writes the mixture evaluated at the input's own active voxels: a
`.vdb` the path tracer draws as any volume. `--measure` prints the relative L2
error of that grid against the input, its largest difference, and the
closed-form transmittance of the gaussians against ratio tracking through the
original along random rays.

On an Apple M5 Pro the gasoline's frames fit at:

| Gaussians | Mean density L2 error | Seconds a frame |
|---|---|---|
| 50,000 | 0.096 | 1.8 |
| 20,000 | 0.132 | 1.6 |
| 10,000 | 0.159 | 1.2 |

The heaviest public simulation tried, EmberGen's aerial explosion at up to
24 million active voxels a frame, fits 50,000 gaussians in 12 s a frame at a
mean error of 0.161.

## Drawing the medium

The raster draws a medium of gaussians as a participating medium, sorted and
blended with every other splat: each gaussian's alpha at a pixel is
1 − exp(−τ), τ the exact optical depth of the pixel's ray through it. Its
colour is the radiance a frame bakes for it:

- **emission**: a black body at the gaussian's temperature, in kelvin
  `temperatureScale × T + temperatureOffset`, times `emissionScale`;
- **single scattering** of each light, its transmittance read from the cloud
  shadow map at the gaussian (the medium casts into the map and reads it, so
  it shadows meshes and clouds and they shadow it);
- **the dome**, times the mean transmittance of the map's dome directions;
- **multiple scattering** as `octaves` (four by default).

The look is a few primvars on the field:

```usda
def "Fire" ( prepend references = @gas60_gs.usdc@</World/Splats> )
{
    float primvars:athenea:densityScale = 0.03
    color3f primvars:athenea:albedo = (0.8, 0.8, 0.8)
    float primvars:athenea:volume:emissionScale = 40
    float primvars:athenea:volume:temperatureScale = 2400
    float primvars:athenea:volume:temperatureOffset = 600
}
```

with a `DistantLight` and a `DomeLight` beside it, and the stage drawn with
`athenea stage`. Against a path trace of the original grid under the same key
light and dome (1,024 paths, four bounces, fire off), the raster of the
50,000-gaussian fit of frame 60 is at relMSE 0.0171 (29.6 dB, SSIM 0.977), and
draws with the fire in 9.3 ms a frame.

## A sequence

```sh
athenea vdb2splat "$GAS/embergen_gasoline_explosion_a_#.vdb" --sequence \
    --independent --frames 4 99 -n 50000 --device-mb 8192 \
    --density-scale 0.03 -o "gas/f_####.usdc"
```

A run of `#` names the frames. There are two ways to fit them:

- **`--independent`**: every frame a full fit of its own. This is what an
  EmberGen file needs: it has no velocity, and a fireball outruns gaussians
  carried from the frame before.
- **Warm frames** (the default): each frame starts from the frame before's
  gaussians in their own slots, carried by the file's `vel` grid (the
  centre moved by the flow, the covariance deformed by F = I + Δt ∇u with
  its volume kept; where the flow shears too hard, the centre alone), then
  `--warm-iterations` steps. Slots are stable, so gaussian *k* is the same gaussian in every
  frame, which is what lets the stream predict its motion. Mantaflow's
  staggered `velocity` grid is read with `--velocity velocity
  --velocity-units mantaflow --fps 24`.

A sequence prints, a frame, the gaussians alive, the density error, the
fit's time and the flicker: how wrong the frame-to-frame change of the grid
given back is against the original's.

## To a browser

Three steps take the fitted frames to a stream the
[web player](https://atheneastream.lucab.co.uk/) draws:

```sh
# 1. Bake each frame's light into a 3DGS PLY (radiance, ln σ, temperature)
ATHENEA_VOLUME_PLY=gas/b_0060.ply athenea stage gas/s.usda \
    --eye 0 0 500 --target 0 0 0 --size 32x32 -o bake.exr
# 2. Encode the baked frames as one rung of a ladder
athenea video-encode gas/b_*.ply --each-a-frame -o gas.athv --rendition 0 \
    --keyint 30 --bits 8 --rate 30 --sh-degree 1 --quality 0.7 --volume 0.03 \
    --order frame --range-group 30 --geometry --geometry-key 1 \
    --geometry-position-bits 11 --geometry-scale-bits 6 --geometry-rotation-bits 6
# 3. Decode it and judge it against the original grids
athenea vdb2splat "$GAS/embergen_gasoline_explosion_a_#.vdb" --sequence \
    --frames 4 99 --from-athv gas.athv
```

`ATHENEA_VOLUME_PLY` writes each medium a frame draws as a PLY with the light
that frame baked for it (step 1 runs once a frame, over `s.usda`, the look
above referencing `f_####.usdc`). `--volume` marks the `.athv` as a medium and
stores the density scale, so the player draws each splat with the alpha
1 − exp(−τ) of its own optical depth. Position, scale and rotation travel
exactly beside the video (`--geometry`), and opacity and colour in it.

The published explosions, each a ladder of three rungs:

| Simulation | Frames | Rungs | Mbit/s at 30 fps | Decoded L2 error, 50k |
|---|---|---|---|---|
| Gasoline explosion (EmberGen) | 96 | 50k · 20k · 10k | 118 · 48 · 25 | 0.129 |
| Mid-air explosion (EmberGen) | 96 | 50k · 20k · 10k | 121 · 50 · 26 | 0.129 |
| Aerial explosion (EmberGen) | 60 | 50k · 20k · 10k | 116 · 48 · 25 | 0.178 |
| Ground explosion (EmberGen) | 96 | 50k · 20k · 10k | 113 · 47 · 25 | 0.156 |
| Smoke column (Blender Mantaflow, warm frames) | 90 | 20k · 10k | 67 · 33 | 0.036 (20k) |
| Fireball (Blender Mantaflow, warm frames) | 60 | 50k · 20k · 10k | 169 · 63 · 30 | 0.050 |

A loop dissolves into its first frame when the player is given `fade=N`: the
last N frames' extinction goes linearly to none.

## Where the details are

- [`docs/operations.md` §2.8.3](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#283-athenea-vdb2splat--a-volume-into-volumetric-gaussians-and-back): every option of `vdb2splat`, the bricks and the primvars.
- [`docs/operations.md` §2.7.2](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#272-athenea-video-encode--splat-frames-as-hardware-video): `video-encode`, with `--volume`, `--slots`, `--cells` and the geometry layer.
- `docs/operations.md` §4 (authoring): a medium of gaussians and its primvars; §8.2: the `.athv` format and its volume flag.
- `docs/decisions.md`, sections "A volume into volumetric gaussians, and back", "A volume sequence as video planes" and "A medium on the web player": the design and every measurement.
