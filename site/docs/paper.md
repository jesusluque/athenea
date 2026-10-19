# Paper

*athenea: A Relight Baker for Gaussian Splatting from Production USD Assets* describes the whole development in two parts with
different goals:

- **Part I, the bake**: from production USD to a relightable master USDC. The engine, the conversion of meshes into
  Gaussians, the light transport baked into each Gaussian, animated assets and levels of detail.
- **Part II, the web format**: a delivery format derived from the master, compressed and streamed to a WebGPU
  player.

It closes with the evaluation against the path trace, the limits and the future work.

This is the **annotated edition**: after each paragraph, a note explains the terms and the mathematics it rests on,
cites the bibliography and points to the code that implements it. It ends with a glossary of about 200 terms. Read it
here, section by section (from [Abstract](paper/00-front.md) to [References and glossary](paper/10-references.md)),
where a button on each page hides or shows the notes, or download it as a PDF.

| Download | Language |
|---|---|
| [gaussian-splat-relight-transfer-baking.en.pdf](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/gaussian-splat-relight-transfer-baking.en.pdf) | English |
| [gaussian-splat-relight-transfer-baking.es.pdf](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/gaussian-splat-relight-transfer-baking.es.pdf) | Spanish |

!!! example "Live"
    Part II's web format, live: [sparkwebgpu.lucab.co.uk](https://sparkwebgpu.lucab.co.uk).
