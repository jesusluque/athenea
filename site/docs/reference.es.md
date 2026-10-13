# Referencia

La documentación del motor está en el repositorio y se mantiene en inglés y castellano en el mismo commit que el código que describe.

| Documento | Responde |
|---|---|
| [`README.md`](https://github.com/jesusluque/athenea/blob/main/README.md) | por qué existe el motor, cómo se compila y una invocación de cada cosa |
| [`docs/operations.es.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md) | cómo se ejecuta: cada subcomando y opción, las tareas, cómo se prepara una escena USD, el visor, el servidor MCP y los efectos AOFX |
| [`docs/development.es.md`](https://github.com/jesusluque/athenea/blob/main/docs/development.es.md) | cómo está hecho y cómo se cambia |
| [`docs/decisions.md`](https://github.com/jesusluque/athenea/blob/main/docs/decisions.md) | por qué es así, frente a qué, y qué se midió |

## Los subcomandos

| Subcomando | Sección (en inglés) |
|---|---|
| `athenea info` | [§2.1](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#21-athenea-info--the-device-and-what-it-can-do) |
| `athenea render`, `athenea bench` | [§2.2](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#22-athenea-render--splat-files-to-an-exr), [§2.3](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#23-athenea-bench--the-same-frame-timed) |
| `athenea convert`, `athenea decimate` | [§2.4](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#24-athenea-convert--a-splat-file-into-usd-or-into-a-athc), [§2.4.1](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#241-athenea-decimate--fewer-gaussians-kept-where-they-matter) |
| `athenea stage`, `athenea view`, `athenea live` | [§2.5](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#25-athenea-stage--a-usd-stage-through-the-hydra-delegate), [§2.6](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#26-athenea-view--a-window-on-a-stage), [§2.7](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#27-athenea-live--a-stage-on-a-clock) |
| `athenea mesh2splat` | [§2.8](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#28-athenea-mesh2splat--a-model-into-gaussians) |
| `athenea visibility` | [§2.9](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#29-athenea-visibility--what-a-skinned-cloud-casts-baked-by-part) |
| `athenea aofx`, `athenea compare` | [§2.10](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#210-athenea-aofx--the-effect-plugins), [§2.11](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#211-athenea-compare--what-an-image-holds-measured-on-the-gpu) |
