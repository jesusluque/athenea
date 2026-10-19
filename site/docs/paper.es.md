# Paper

*athenea: A Relight Baker for Gaussian Splatting from Production USD Assets* describe todo el desarrollo en dos partes con objetivos
distintos:

- **Parte I, el bake**: del USD de producción a un master USDC reiluminable. El motor, la conversión de mallas en
  Gaussians, el transporte de luz horneado en cada Gaussian, los assets animados y los niveles de detalle.
- **Parte II, el formato web**: un formato de entrega derivado del master, comprimido y en streaming hacia un player
  WebGPU.

Cierra con la evaluación frente al path trace, los límites y el trabajo futuro.

Es la **edición anotada**: tras cada párrafo, una acotación explica los términos y las matemáticas en que se apoya,
cita la bibliografía y señala el código que lo implementa. Termina con un glosario de unos 200 términos. Se puede leer
aquí, sección a sección (del [Resumen](paper/00-front.md) a [Referencias y glosario](paper/10-references.md)), con un
botón en cada página para ocultar o mostrar las acotaciones, o descargar en PDF.

| Descarga | Idioma |
|---|---|
| [gaussian-splat-relight-transfer-baking.en.pdf](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/gaussian-splat-relight-transfer-baking.en.pdf) | inglés |
| [gaussian-splat-relight-transfer-baking.es.pdf](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/gaussian-splat-relight-transfer-baking.es.pdf) | español |

!!! example "En vivo"
    El formato web de la Parte II, en vivo: [sparkwebgpu.lucab.co.uk](https://sparkwebgpu.lucab.co.uk).
