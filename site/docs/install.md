# Desktop installation

The reference platform is **macOS on Apple Silicon** (Metal), where every measurement in the repository is taken. **Linux on x86_64 with CUDA** is the second platform; without a CUDA device, Linux opens Vulkan. This page summarises the README's *Building* section and the first section of [`docs/operations.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#1-getting-it-running).

## Prerequisites

CMake ≥ 3.24, Ninja and a C++20 compiler, plus:

| Dependency | Version | Prefix | How it is built |
|---|---|---|---|
| Slang | 2026.14.1 | `~/tools/slang` | one Slang for gpe's blobs and for slang-rhi |
| OpenUSD with MaterialX 1.39.5 and OpenVDB | 26.08 | `~/tools/usd-26.08-mx` | `scripts/build-usd.sh`, once per machine |
| Open Image Denoise (GPU devices only) | 2.5.1 | `~/tools/oidn-2.5.1` | `scripts/build-oidn.sh`; without it the engine has no denoiser |
| OpenColorIO | 2.5.2 | `~/tools/ocio-2.5.2` | `scripts/build-ocio.sh` |
| libwebp | Homebrew `webp` | | for `.sog`; without it `.sog` is refused |

The submodules (gpe, genlock) come with `git submodule update --init --recursive`. slang-rhi, tinyexr, CLI11, nlohmann_json, Catch2 and SPZ arrive through FetchContent or `third_party/`.

## Building

```sh
git clone https://github.com/jesusluque/athenea.git && cd athenea
git submodule update --init --recursive
scripts/build-usd.sh && scripts/build-oidn.sh && scripts/build-ocio.sh
cmake --preset macos-arm64-release          # or macos-arm64-debug, linux-x86_64-*
cmake --build --preset macos-arm64-release
ctest --preset macos-arm64-debug            # GPU tests skip on a machine without a device
```

The binary is `build/macos-arm64-release/bin/athenea`. The Hydra plugin and the codeless schemas are in `build/macos-arm64-release/plugin/usd`.

## The runtime environment

| Variable | What it does |
|---|---|
| `ATHENEA_SHADER_DIR` | where the Slang shaders are read from; `athenea info` prints the directory in use |
| `PXR_PLUGINPATH_NAME` | points a USD application other than `athenea` (usdview, Blender) at `<build>/plugin/usd` |
| `ATHENEA_MATERIALX_ROOT` | a MaterialX `libraries/` directory read instead of the host's, for a host whose MaterialX predates the Slang generator |
| `AOFX_PLUGIN_PATH` | extra directories of AOFX bundles |
| `ATHENEA_BACKEND` | device order, e.g. `metal,cuda,vulkan,d3d12` |
| `ATHENEA_GPU_BUDGET` | device memory this run may hold, in MiB |
| `ATHENEA_SHADER_CACHE` | where compiled shaders are cached between runs |

## Checking the device

`athenea info` prints the device, what it can do (ray tracing by pipeline or inline queries, half precision), the shader directory and the optional dependencies the build found. On Metal there are inline ray queries and acceleration structures but no ray-tracing pipelines; CUDA has both, given the OptiX headers.

```sh
build/macos-arm64-release/bin/athenea info
```

A command prints its errors to standard error and exits with `1`; a GPU out of memory exits with `3`; success exits with `0`.

## What a missing optional dependency costs

| Missing | What is lost |
|---|---|
| OpenColorIO | the OCIO view transform; AgX and ACES 2.0 still work |
| Open Image Denoise | `--denoise` does nothing |
| libwebp | `.sog` clouds are refused |
| OpenVDB | `.vdb` volume fields are refused |
| OptiX headers (CUDA) | ray tracing by pipeline on CUDA; inline ray queries still work |
| zstd | `.spz` clouds are refused |
