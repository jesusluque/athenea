# Paper

*athenea: A Relight Baker for Gaussian Splatting from Production USD Assets* is the technical paper on the baker, in English and Spanish. It describes the two processes of the work: Part I, the bake from a production USD asset to a relightable master USDC, judged by fidelity to a path trace; and Part II, the web format derived from that master, judged by size, memory and frame time.

## Downloads

The built paper is published as assets of the `docs-2026-10-19` release.

| File | Format | Language |
|---|---|---|
| [paper.en.pdf](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/paper.en.pdf) | PDF | English |
| [paper.es.pdf](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/paper.es.pdf) | PDF | Spanish |
| [paper.html](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/paper.html) | HTML, one file | English |
| [paper.es.html](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/paper.es.html) | HTML, one file | Spanish |

## Chapters

The sources are in [`docs/paper`](https://github.com/jesusluque/athenea/tree/main/docs/paper), one Markdown file per chapter and language (`<chapter>.md` and `<chapter>.es.md`).

| File | Chapter |
|---|---|
| `00-front` | Title, abstract and contents |
| `01-introduction` | 1. Introduction: the problem, goals and constraints, contributions, organisation, related work, conventions |
| `02-engine` | Part I. The bake: from production USD to a relightable master USDC. 2. The athenea engine and renderer |
| `03-conversion` | 3. From USD meshes to Gaussians (mesh2splat) |
| `04-transfer` | 4. Light transport baked into Gaussians (transfer) and relighting |
| `05-animation` | 5. Animated assets: skeletons, skinning and per-part visibility (the pawn and the sparrow) |
| `06-lod` | 6. Levels of detail: merge trees, cuts by error, pages and occlusion |
| `07-compression-player` | Part II. The web format: a derived delivery format. 7. The web format: compression, streaming and a WebGPU player |
| `08-evaluation` | 8. Evaluation |
| `09-limitations` | 9. Limitations and future work |
| `10-references` | 10. References, and Appendix A. Repositories and commands |

## Building it

`docs/paper/build.sh` joins the chapters into `paper.md` and `paper.es.md`, and builds from each an HTML file with every resource embedded and a PDF:

```sh
docs/paper/build.sh
```

| Tool | Used for |
|---|---|
| bash and Python 3 | joining the chapters, and moving the table of contents after the title page |
| pandoc | the HTML (`--mathml --embed-resources`, styled by `paper.css`) |
| Google Chrome, headless | the PDF, printed from the HTML; `CHROME` names the binary, by default the macOS application's |

The outputs are written to `docs/paper`: `paper.md`, `paper.es.md`, `paper.html`, `paper.es.html`, `paper.en.pdf` and `paper.es.pdf`.
