# Reference

The engine's documentation lives in the repository and is kept in English and Spanish in the same commit as the code it describes.

| Document | Answers |
|---|---|
| [`README.md`](https://github.com/jesusluque/athenea/blob/main/README.md) | why the engine exists, how it is built, and one invocation of each thing |
| [`docs/operations.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.md) | how it is run: every subcommand and option, the tasks, how a USD stage is authored for it, the viewer, the MCP server and the AOFX effects |
| [`docs/development.md`](https://github.com/jesusluque/athenea/blob/main/docs/development.md) | how it is made and how it is changed |
| [`docs/decisions.md`](https://github.com/jesusluque/athenea/blob/main/docs/decisions.md) | why it is so, against what, and what was measured |

## The subcommands

| Subcommand | Section |
|---|---|
| `athenea info` | [§2.1](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#21-athenea-info--the-device-and-what-it-can-do) |
| `athenea render`, `athenea bench` | [§2.2](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#22-athenea-render--splat-files-to-an-exr), [§2.3](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#23-athenea-bench--the-same-frame-timed) |
| `athenea convert`, `athenea decimate` | [§2.4](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#24-athenea-convert--a-splat-file-into-usd-or-into-a-athc), [§2.4.1](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#241-athenea-decimate--fewer-gaussians-kept-where-they-matter) |
| `athenea stage`, `athenea view`, `athenea live` | [§2.5](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#25-athenea-stage--a-usd-stage-through-the-hydra-delegate), [§2.6](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#26-athenea-view--a-window-on-a-stage), [§2.7](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#27-athenea-live--a-stage-on-a-clock) |
| `athenea mesh2splat` | [§2.8](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#28-athenea-mesh2splat--a-model-into-gaussians) |
| `athenea visibility` | [§2.9](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#29-athenea-visibility--what-a-skinned-cloud-casts-baked-by-part) |
| `athenea aofx`, `athenea compare` | [§2.10](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#210-athenea-aofx--the-effect-plugins), [§2.11](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#211-athenea-compare--what-an-image-holds-measured-on-the-gpu) |
