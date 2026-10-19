# Contributing to athenea

## Where work is tracked

Issues are grouped by area in the [athenea project](https://github.com/users/jesusluque/projects/3):
Engine, Blender, OFX/Nuke, Web, Platforms, Docs/Paper, Assets. Labels name the area (`area:*`), the platform
(`platform:*`) and the kind (`type:bug`, `type:feature`, `type:docs`, `type:triage`).

## Branches

`main` is the trunk and the only long-lived branch. Work happens on a short branch named after what it does, merged
into `main` when its tests pass. A finished branch is deleted; one kept for reference is tagged `archive/<name>`
first.

## Building and testing

The [installation page](https://jesusluque.github.io/athenea/install/) covers the toolchain on each system.

```sh
cmake --preset macos-arm64-debug && cmake --build --preset macos-arm64-debug
ctest --preset macos-arm64-debug                     # one test at a time: the tests share the GPU
ctest --test-dir build/macos-arm64-debug -R lod      # tests matching a name
```

GPU tests skip on a machine without a device, and a skip is not a pass.

## Rules the code holds to

- **No CPU arithmetic on scene data.** The CPU reads files, parses headers, decompresses and keeps the books.
  Decoding, sorting, merging, culling and image comparison are compute kernels. There is no CPU reference renderer
  and no CPU fallback.
- **Tests check on the GPU.** A test generates its input and checks its result with kernels, and reads back only
  counters or `compareImages` metrics (p99, max, over2) against a GPU reference.
- **Shader parameters are bound by name** through reflection, never by slot.
- **Errors** are `Result<T>` with `ATHENEA_TRY`, not exceptions. OS calls live only in `modules/core` (Platform).
- **New computation** is a Slang kernel. A data-processing step that a host can run on its own is an AOFX bundle
  under `plugins/`.
- **aofx compatibility is mandatory**: the SDK headers change only additively and `aofx_sdk_manifest` must pass.
- **Every first-party file** starts with `Copyright (c) 2026 jesus luque.` A contribution is made under the
  Apache License 2.0.

## Documentation changes with the code

- `docs/decisions.md` records each subsystem's design, what was measured and what is not done.
- `docs/operations.md` changes when a subcommand, option, environment variable, `athenea:*` setting, schema, format
  or printed error changes; `docs/development.md` when a module, rule or way of adding something changes.
- The Spanish manuals (`*.es.md`) are translations, kept to the same headings and tables, and change in the same
  commit. Technical terms stay in English.
- The site (`site/docs`) has every page in English and Spanish.

## Submodules

gpe and aopenfx are pinned submodules. A change to either is committed and pushed in its own repository before the
pin moves here, so a clone finds the pinned commit. genlock is private and optional (`ATHENEA_WITH_GENLOCK`).

## Commits and pull requests

A commit says what changed and why, in the present tense. A pull request lists the tests run and on which device,
and the manuals it changed.
