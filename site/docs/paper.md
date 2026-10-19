# Paper

*athenea: A Relight Baker for Gaussian Splatting from Production USD Assets* describes the whole development in two parts with
different goals:

- **Part I, the bake**: from production USD to a relightable master USDC. The engine, the conversion of meshes into
  Gaussians, the light transport baked into each Gaussian, animated assets and levels of detail.
- **Part II, the web format**: a delivery format derived from the master, compressed and streamed to a WebGPU
  player.

It closes with the evaluation against the path trace, the limits and the future work.

| Download | Language |
|---|---|
| [paper.en.pdf](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/paper.en.pdf) | English |
| [paper.es.pdf](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/paper.es.pdf) | Spanish |

!!! example "Live"
    Part II's web format, live: [sparkwebgpu.lucab.co.uk](https://sparkwebgpu.lucab.co.uk).
