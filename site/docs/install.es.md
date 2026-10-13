# Instalación de escritorio

La plataforma de referencia es **macOS en Apple Silicon** (Metal), donde se toma cada medida del repositorio. **Linux en x86_64 con CUDA** es la segunda plataforma; sin un dispositivo CUDA, Linux abre Vulkan. Esta página resume la sección *Building* del README y la primera sección de [`docs/operations.es.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md#1-ponerlo-en-marcha).

## Requisitos

CMake ≥ 3.24, Ninja y un compilador de C++20, además de:

| Dependencia | Versión | Prefijo | Cómo se compila |
|---|---|---|---|
| Slang | 2026.14.1 | `~/tools/slang` | un único Slang para los blobs de gpe y para slang-rhi |
| OpenUSD con MaterialX 1.39.5 y OpenVDB | 26.08 | `~/tools/usd-26.08-mx` | `scripts/build-usd.sh`, una vez por máquina |
| Open Image Denoise (solo dispositivos GPU) | 2.5.1 | `~/tools/oidn-2.5.1` | `scripts/build-oidn.sh`; sin él el motor no tiene denoiser |
| OpenColorIO | 2.5.2 | `~/tools/ocio-2.5.2` | `scripts/build-ocio.sh` |
| libwebp | `webp` de Homebrew | | para `.sog`; sin ella se rechaza `.sog` |

Los submódulos (gpe, genlock) llegan con `git submodule update --init --recursive`. slang-rhi, tinyexr, CLI11, nlohmann_json, Catch2 y SPZ llegan por FetchContent o `third_party/`.

## Compilación

```sh
git clone https://github.com/jesusluque/athenea.git && cd athenea
git submodule update --init --recursive
scripts/build-usd.sh && scripts/build-oidn.sh && scripts/build-ocio.sh
cmake --preset macos-arm64-release          # o macos-arm64-debug, linux-x86_64-*
cmake --build --preset macos-arm64-release
ctest --preset macos-arm64-debug            # las pruebas de GPU se saltan en una máquina sin dispositivo
```

El binario es `build/macos-arm64-release/bin/athenea`. El plugin de Hydra y los schemas codeless están en `build/macos-arm64-release/plugin/usd`.

## El entorno de ejecución

| Variable | Qué hace |
|---|---|
| `ATHENEA_SHADER_DIR` | de dónde se leen los shaders de Slang; `athenea info` imprime el directorio en uso |
| `PXR_PLUGINPATH_NAME` | apunta una aplicación USD que no sea `athenea` (usdview, Blender) a `<build>/plugin/usd` |
| `ATHENEA_MATERIALX_ROOT` | un directorio `libraries/` de MaterialX que se lee en lugar del del host, para un host cuyo MaterialX es anterior al generador de Slang |
| `AOFX_PLUGIN_PATH` | directorios extra de bundles AOFX |
| `ATHENEA_BACKEND` | orden de dispositivos, p. ej. `metal,cuda,vulkan,d3d12` |
| `ATHENEA_GPU_BUDGET` | la memoria del dispositivo que puede ocupar esta ejecución, en MiB |
| `ATHENEA_SHADER_CACHE` | dónde se cachean los shaders compilados entre ejecuciones |

## Comprobar el dispositivo

`athenea info` imprime el dispositivo, lo que puede hacer (ray tracing por pipeline o por consultas inline, media precisión), el directorio de shaders y las dependencias opcionales que encontró la compilación. En Metal hay consultas de rayos inline y estructuras de aceleración, pero no pipelines de ray tracing; CUDA tiene ambos, con las cabeceras de OptiX.

```sh
build/macos-arm64-release/bin/athenea info
```

Un comando imprime sus errores por la salida de error y termina con `1`; una GPU sin memoria termina con `3`; el éxito, con `0`.

## Qué se pierde con cada dependencia opcional que falte

| Falta | Qué se pierde |
|---|---|
| OpenColorIO | la transformación de vista de OCIO; AgX y ACES 2.0 siguen funcionando |
| Open Image Denoise | `--denoise` no hace nada |
| libwebp | se rechazan las nubes `.sog` |
| OpenVDB | se rechazan los campos de volumen `.vdb` |
| Cabeceras de OptiX (CUDA) | el ray tracing por pipeline en CUDA; las consultas inline siguen funcionando |
| zstd | se rechazan las nubes `.spz` |
