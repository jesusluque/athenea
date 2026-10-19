# Desktop installation

The reference platform is **macOS on Apple Silicon** (Metal), where every measurement in the repository is taken. **Linux on x86_64 with CUDA** is the second platform; without a CUDA device, Linux opens Vulkan. **Windows x64 with CUDA** has its own presets and toolchain scripts. This page summarises the README's *Building* section and the first section of [`docs/operations.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#1-getting-it-running).

## Installers

A release on GitHub has an installer and an archive for each system ([releases](https://github.com/jesusluque/athenea/releases)). Each holds the programs with every library they load; the GPU driver is the machine's own.

| System | Installer | Installs to | Needs |
|---|---|---|---|
| macOS (Apple silicon) | `athenea-<v>-macos-arm64.pkg` | `/usr/local/athenea`, programs linked into `/usr/local/bin` | macOS 15 or later. Not signed with a Developer ID: open it with right click, Open. |
| Linux (x86_64) | `athenea_<v>_amd64.deb` | `/opt/athenea`, programs linked into `/usr/bin` | glibc 2.39 (Ubuntu 24.04 or later) and an NVIDIA driver with OptiX for baking (570 or later). |
| Windows (x64) | `athenea-<v>-windows-x64-setup.exe` | `Program Files\athenea`, `bin\` added to the machine's `PATH` | an NVIDIA driver with OptiX (570 or later). Not signed: SmartScreen asks once. |

The rest of this page builds from source.

## Prerequisites

CMake ≥ 3.24, Ninja and a C++20 compiler, plus:

| Dependency | Version | Prefix | How it is built |
|---|---|---|---|
| Slang | 2026.14.1 | `~/tools/slang` | one Slang for gpe's blobs and for slang-rhi |
| OpenUSD with MaterialX 1.39.5 and OpenVDB | 26.08 | `~/tools/usd-26.08-mx` | `scripts/build-usd.sh`, once per machine |
| Open Image Denoise (GPU devices only) | 2.5.1 | `~/tools/oidn-2.5.1` | `scripts/build-oidn.sh`; without it the engine has no denoiser |
| OpenColorIO | 2.5.2 | `~/tools/ocio-2.5.2` | `scripts/build-ocio.sh` |
| libwebp | Homebrew `webp` | | for `.sog`; without it `.sog` is refused |

On Windows the prefixes are under `D:\tools` (`D:\tools\slang`, `D:\tools\usd-26.08-mx`, `D:\tools\oidn-2.5.1`, `D:\tools\ocio-2.5.2`), as the `windows-base` preset names them, and the scripts are their PowerShell counterparts, `scripts\build-*.ps1`.

slang-rhi, tinyexr, CLI11, nlohmann_json, Catch2 and SPZ arrive through FetchContent or `third_party/`.

## The source and its submodules

```sh
git clone --recursive https://github.com/jesusluque/athenea.git && cd athenea
```

A clone made without `--recursive` takes the submodules with `git submodule update --init --recursive`.

| Submodule | What it is |
|---|---|
| `third_party/gpe` | the GPU engine that runs the AOFX effects on the engine's own device |
| `third_party/aopenfx` | the AOFX SDK and host |
| `third_party/openfx` | Natron's OpenFX, for the OpenFX plugin |
| `third_party/genlock` | PTP and ST 2059-1 timing; a private repository |

genlock's entry in `.gitmodules` has `update = none`, so a clone skips it. `ATHENEA_WITH_GENLOCK` is on only when `third_party/genlock` is checked out; without it everything builds except the `sched` module and `athenea live` (see the development manual, *Dependencies and toolchain*). Where the repository is available, `git submodule update --init --checkout third_party/genlock` brings it back.

## Building

### macOS

```sh
scripts/build-usd.sh && scripts/build-oidn.sh && scripts/build-ocio.sh
cmake --preset macos-arm64-release          # or macos-arm64-debug
cmake --build --preset macos-arm64-release
ctest --preset macos-arm64-debug            # GPU tests skip on a machine without a device
```

The binary is `build/macos-arm64-release/bin/athenea`. The Hydra plugin and the codeless schemas are in `build/macos-arm64-release/plugin/usd`.

The device is Metal; `scripts/build-oidn.sh` builds OIDN's Metal device.

### Linux

```sh
scripts/build-usd.sh && scripts/build-oidn.sh && scripts/build-ocio.sh
cmake --preset linux-x86_64-release         # or linux-x86_64-debug
cmake --build --preset linux-x86_64-release
ctest --preset linux-x86_64-debug
```

The device is CUDA, or Vulkan without a CUDA device. One CUDA toolkit serves gpe's kernels, slang-rhi's CUDA device and OIDN, whose CUDA device wants 12.8 or later: `ATHENEA_CUDA_ROOT` names it, and without it the configure takes the newest `/usr/local/cuda-*` that has `nvcc`. The OptiX headers that ray tracing by pipeline on CUDA needs are fetched by slang-rhi.

### Windows

From PowerShell:

```powershell
scripts\build-usd.ps1
scripts\build-oidn.ps1
scripts\build-ocio.ps1
cmake --preset windows-x64-release          # or windows-x64-debug
cmake --build --preset windows-x64-release
```

The scripts enter the MSVC x64 environment of the newest Visual Studio with the C++ tools (`scripts\msvc-env.ps1`, found by vswhere), unless `cl` is already on the path. Their prefixes are `$env:ATHENEA_USD_ROOT`, `$env:ATHENEA_OIDN_ROOT` and `$env:ATHENEA_OCIO_ROOT`, or the `D:\tools` defaults above. The presets compile with `cl`. The device is CUDA, and OIDN is built for CUDA, as on Linux. The OpenFX plugin is not built on Windows.

### The presets

| Configure preset | Platform |
|---|---|
| `macos-arm64-debug`, `macos-arm64-release` | macOS arm64 (Metal) |
| `linux-x86_64-debug`, `linux-x86_64-release` | Linux x86_64 (CUDA/Vulkan) |
| `windows-x64-debug`, `windows-x64-release` | Windows x64 (CUDA) |
| `ios-arm64-debug`, `ios-arm64-release` | iOS arm64 (Metal): the engine and its Hydra delegate as static libraries, without the CLI, the AOFX host, the MCP server or the tests |
| `macos-arm64-blender` | macOS arm64: hdAthenea for Blender (Metal) |

Debug presets build `Debug`; release presets build `RelWithDebInfo`. Each desktop preset has a build preset and a test preset of the same name; the tests run one at a time, since they share the GPU.

### The Blender add-on

The Blender add-on is built on macOS today, with the `macos-arm64-blender` preset, against headers that match Blender's own OpenUSD (`scripts/build-usd-blender.sh`). Other platforms are in progress. The steps are in [Gaussian splats in Blender](tutorials/blender.md) and in the development manual's *Dependencies and toolchain*.

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
| genlock | `athenea live` |
