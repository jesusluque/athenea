# Paper

*athenea: un baker de relight para Gaussian splatting a partir de assets USD de producción* es el paper técnico sobre el baker, en inglés y en español. Describe los dos procesos del trabajo: la Parte I, el bake de un asset USD de producción a un master USDC reiluminable, que se juzga por su fidelidad a un path trace; y la Parte II, el formato web derivado de ese master, que se juzga por tamaño, memoria y tiempo de frame.

## Descargas

El paper compilado se publica como assets de la release `docs-2026-10-19`.

| Fichero | Formato | Idioma |
|---|---|---|
| [paper.en.pdf](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/paper.en.pdf) | PDF | inglés |
| [paper.es.pdf](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/paper.es.pdf) | PDF | español |
| [paper.html](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/paper.html) | HTML, un solo fichero | inglés |
| [paper.es.html](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/paper.es.html) | HTML, un solo fichero | español |

## Capítulos

Las fuentes están en [`docs/paper`](https://github.com/jesusluque/athenea/tree/main/docs/paper), un fichero Markdown por capítulo e idioma (`<capítulo>.md` y `<capítulo>.es.md`).

| Fichero | Capítulo |
|---|---|
| `00-front` | Título, resumen e índice |
| `01-introduction` | 1. Introducción: el problema, objetivos y restricciones, contribuciones, organización, trabajo relacionado, convenciones |
| `02-engine` | Parte I. El bake: de un USD de producción a un master USDC reiluminable. 2. El motor y el renderer athenea |
| `03-conversion` | 3. De meshes USD a Gaussians (mesh2splat) |
| `04-transfer` | 4. Transporte de luz en el bake de las Gaussians (transfer) y relight |
| `05-animation` | 5. Assets animados: skeletons, skinning y visibilidad por partes (el peón y el gorrión) |
| `06-lod` | 6. LOD: merge trees, cuts por error, pages y oclusión |
| `07-compression-player` | Parte II. El formato web: un formato de entrega derivado. 7. El formato web: compresión, streaming y un player WebGPU |
| `08-evaluation` | 8. Evaluación |
| `09-limitations` | 9. Limitaciones y trabajo futuro |
| `10-references` | 10. Referencias, y Apéndice A. Repositorios y comandos |

## Compilarlo

`docs/paper/build.sh` une los capítulos en `paper.md` y `paper.es.md`, y compila a partir de cada uno un fichero HTML con todos los recursos incrustados y un PDF:

```sh
docs/paper/build.sh
```

| Herramienta | Para qué se usa |
|---|---|
| bash y Python 3 | unir los capítulos, y mover el índice detrás de la portada |
| pandoc | el HTML (`--mathml --embed-resources`, con el estilo de `paper.css`) |
| Google Chrome, headless | el PDF, impreso a partir del HTML; `CHROME` nombra el binario, por defecto el de la aplicación de macOS |

Las salidas se escriben en `docs/paper`: `paper.md`, `paper.es.md`, `paper.html`, `paper.es.html`, `paper.en.pdf` y `paper.es.pdf`.
