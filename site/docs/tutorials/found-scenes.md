# A scene found by rendering

!!! example "Live"
    Scenes made this way, shown as they grew frame by frame: [neural.lucab.co.uk](https://neural.lucab.co.uk).

A path tracer usually leaves pictures. `athenea neural-scene shoot` leaves the scene: every frame it renders adds the gaussians that frame found, each with its material, its normal and the light it was found under, laid over the coarser gaussians a farther look had left. What stands at the end is drawn from any view, cut to a level of detail, relit by groups of lights and exported for a browser.

## The walk

A file of views, one a line: a name, the eye and what it looks at, and optionally the up vector.

```text
k00 150.0 0.0 175.0 1150.0 0.0 80.0 0 0 1
k01 172.7 4.3 174.7 1150.0 0.0 80.0 0 0 1
k02 195.4 8.5 174.3 1150.0 0.0 80.0 0 0 1
```

What a frame already holds is not found again: a later view adds only what it sees anew or sees finer.

## The shoot

```sh
athenea neural-scene shoot sponza_knight.usda knight.aths --views walk.txt \
    --focal 24 --size 960x540 --samples 32 --pass-samples 8 --bounces 3 \
    --denoise --ray-margin 2e-5 --inside-turned 0.1
```

`knight.aths` is a directory: a manifest (`scene.json`) and a piece for every look, its gaussians in an `.athg`. Intel's Sponza with the ASWF Open Chess Set's knight in glass, 60 views at 960 × 540 on an Apple M5 Pro, is **2 362 750 gaussians standing after 414.5 s** of path tracing. `--inside-turned 0.1` makes a gaussian inside a surface as long as its pixel lies on that surface, which is what keeps a floor seen at a grazing angle from being ruled in dashes at a coarse level of detail; it is off by default because views held out of the walk read 0.03 to 0.36 dB lower with it.

A shoot over a scene that already holds pieces adds to it. `--light-group NAME` with `--lights-on` shoots only those lights' share over the same gaussians, so the scene can be relit by groups afterwards.

## Drawing it

```sh
athenea neural-scene render knight.aths --eye 834 66 162 --target 1150 0 80 \
    --focal 24 --size 1920x1080 --glass -o view.exr
```

The view need not be one of the walk's. `--glass` shades what was found as glass or as a mirror by the eye that looks now, not as the picture it was found with. `--budget 1000000` draws the level of detail that fits in that many gaussians. Four views held out of the walk measure 31.7 to 34.2 dB against the path traced frames of the same views.

## For the web

```sh
athenea neural-scene export knight.aths --by-frame --history \
    --levels 1000000,500000,250000 -o web/scenes/sponza_knight
```

This writes SPZ parts, one a rendered frame, and an index `sponza_knight.json`; the levels came out at 913 434, 495 348 and 248 286 gaussians. The player is one page, [`scripts/examples/neural-scene-viewer/index.html`](https://github.com/jesusluque/athenea/tree/main/scripts/examples/neural-scene-viewer), served beside the `scenes/` directory and told what to show by its address: `?scene=sponza_knight`, `&detail=auto`, `&ride=1` to ride the render camera. It draws the coarsest level first and the detail behind it, and the level a device can hold: a million gaussians on a phone.

## Where to read more

- Every option of `shoot`, `render` and `export`, with its default and unit: [operations §2.3.4](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#234-athenea-neural-scene--a-scene-kept-in-pieces).
- What the player's address says: [operations §2.3.5](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#235-the-web-player--what-its-address-says).
- The `.aths` directory byte by byte: [formats §6](https://github.com/jesusluque/athenea/blob/main/docs/formats.md#6-aths-a-neural-scene).
- Why it is made the way it is, with the measurements: [`docs/decisions.md`](https://github.com/jesusluque/athenea/blob/main/docs/decisions.md).
