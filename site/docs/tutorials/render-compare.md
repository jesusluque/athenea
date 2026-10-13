# Rendering and comparing with the path trace

The cloud from the [previous tutorial](usd-to-cloud.md) is drawn under the stage's dome and sun, and the source meshes are path traced from the same camera under the same light. `athenea compare` then measures the first image against the second on the GPU. The numbers on this page come from one run on an Apple M5 Pro (Metal) with the public `main` branch.

## The relit cloud

```sh
athenea stage gs.usda --camera /World/Camera -o gs.exr
```

`athenea stage` draws through the same Hydra delegate a DCC loads. With the default `--technique raster` the cloud's splats are rasterised; the transfer of every Gaussian is combined with the sky the frame provides (the `DomeLight`) and with the sun. The frame is 1920 × 1080 by default and took **2 s**, shader compilation included. The output is a half-float RGBA EXR with a `Z` channel.

## The reference

```sh
athenea stage mesh.usda --camera /World/Camera --technique rt \
    --path-total 1024 --denoise -o pt.exr
```

`--technique rt` path traces the meshes with their materials; `--path-total 1024` draws until every pixel holds 1024 paths, and `--denoise` runs Open Image Denoise once it does. This frame took **155 s**.

<figure markdown>
  ![The pawn as relit Gaussians (left) and path traced as meshes (right)](../assets/pawn-gs-pt.png)
  <figcaption>Left: the cloud, relit at draw time. Right: the meshes, path traced with 1024 paths a pixel. Same camera, dome and sun.</figcaption>
</figure>

## The measurement

```sh
athenea compare gs.exr pt.exr
```

```text
image     mean 0.736384 0.794351 0.921122 1   max 12.2266 8.60938 3.81445 1   (2073600 pixels)
reference mean 0.732241 0.790268 0.916906 1   max 11.2812 10.9688 10.1328 1   (2073600 pixels)
hdr       relMSE 0.043379   p99 relative 2.181   max relative 279.2
8-bit     p99 60   max 225   over 2: 164522 of 2073600 pixels
```

| Line | What it says |
|---|---|
| `image`, `reference` | the mean and the largest value of each channel of the cloud's frame and of the path trace |
| `hdr` | the relative HDR error of the first against the second: `relMSE`, its 99th percentile and its largest value |
| `8-bit` | the difference in 8-bit sRGB code values: the 99th percentile, the largest, and the pixels that differ by more than 2 |

Most of this frame is the sky behind the pawn, so the means over the whole frame are close. `--window X0 Y0 X1 Y1` restricts the means to a region (rows counted from the bottom); the differences are always over the whole image. `--heatmap` writes the Measure effect's picture of where the two differ:

```sh
athenea compare gs.exr pt.exr --window 700 140 1220 680        # the body
athenea compare gs.exr pt.exr --heatmap rel.exr --show relative --gain 4
```

## Reading the comparison

The marble body, its veins, the gold ring and the highlights stand where the path trace has them. The glass head is the visible difference. Drawn as Gaussians the glass head reads lighter and less saturated than the path-traced solid glass. A fully transmitting solid keeps the coverage `--glass-opacity` gives it (0.6 by default); its Gaussians look up the environment along the refracted direction, but the scene behind them is seen straight through that coverage, not bent as through the traced glass.

The same four steps apply to any asset: a layer with the camera and the lights, the meshes and the cloud each referenced under it, a conversion with the camera's cell, and the two frames measured against each other region by region.

## Reference

[operations §2.5 `athenea stage`](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#25-athenea-stage--a-usd-stage-through-the-hydra-delegate) · [§2.11 `athenea compare`](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#211-athenea-compare--what-an-image-holds-measured-on-the-gpu) · [§3.2 rendering a stage](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#32-rendering-a-stage)
