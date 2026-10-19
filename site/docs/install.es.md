# Instalación de escritorio

La plataforma de referencia es **macOS en Apple Silicon** (Metal), donde se toma cada medida del repositorio. **Linux en x86_64 con CUDA** es la segunda plataforma; sin un dispositivo CUDA, Linux abre Vulkan. **Windows x64 con CUDA** tiene sus propios presets y scripts de toolchain. Esta página resume la sección *Building* del README y la primera sección de [`docs/operations.es.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md#1-ponerlo-en-marcha).

## Instaladores

Una release de GitHub tiene un instalador y un archivo comprimido para cada sistema ([releases](https://github.com/jesusluque/athenea/releases)). Cada uno contiene los programas con todas las librerías que cargan; el driver de la GPU es el de la máquina.

| Sistema | Instalador | Se instala en | Necesita |
|---|---|---|---|
| macOS (Apple silicon) | `athenea-<v>-macos-arm64.pkg` | `/usr/local/athenea`, con los programas enlazados en `/usr/local/bin` | macOS 15 o posterior. No está firmado con un Developer ID: se abre con clic derecho, Abrir. |
| Linux (x86_64) | `athenea_<v>_amd64.deb` | `/opt/athenea`, con los programas enlazados en `/usr/bin` | glibc 2.39 (Ubuntu 24.04 o posterior) y un driver de NVIDIA con OptiX para el bake (570 o posterior). |
| Windows (x64) | `athenea-<v>-windows-x64-setup.exe` | `Program Files\athenea`, con `bin\` añadido al `PATH` de la máquina | un driver de NVIDIA con OptiX (570 o posterior). No está firmado: SmartScreen pregunta una vez. |

El resto de esta página compila desde el código fuente.

## Requisitos

CMake ≥ 3.24, Ninja y un compilador de C++20, además de:

| Dependencia | Versión | Prefijo | Cómo se compila |
|---|---|---|---|
| Slang | 2026.14.1 | `~/tools/slang` | un único Slang para los blobs de gpe y para slang-rhi |
| OpenUSD con MaterialX 1.39.5 y OpenVDB | 26.08 | `~/tools/usd-26.08-mx` | `scripts/build-usd.sh`, una vez por máquina |
| Open Image Denoise (solo dispositivos GPU) | 2.5.1 | `~/tools/oidn-2.5.1` | `scripts/build-oidn.sh`; sin él el motor no tiene denoiser |
| OpenColorIO | 2.5.2 | `~/tools/ocio-2.5.2` | `scripts/build-ocio.sh` |
| libwebp | `webp` de Homebrew | | para `.sog`; sin ella se rechaza `.sog` |

En Windows los prefijos están bajo `D:\tools` (`D:\tools\slang`, `D:\tools\usd-26.08-mx`, `D:\tools\oidn-2.5.1`, `D:\tools\ocio-2.5.2`), como los nombra el preset `windows-base`, y los scripts son sus equivalentes de PowerShell, `scripts\build-*.ps1`.

slang-rhi, tinyexr, CLI11, nlohmann_json, Catch2 y SPZ llegan por FetchContent o `third_party/`.

## El código fuente y sus submódulos

```sh
git clone --recursive https://github.com/jesusluque/athenea.git && cd athenea
```

Un clon hecho sin `--recursive` trae los submódulos con `git submodule update --init --recursive`.

| Submódulo | Qué es |
|---|---|
| `third_party/gpe` | el motor de GPU que ejecuta los efectos AOFX en el dispositivo del propio motor |
| `third_party/aopenfx` | el SDK y el host de AOFX |
| `third_party/openfx` | el OpenFX de Natron, para el plugin OpenFX |
| `third_party/genlock` | sincronización PTP y ST 2059-1; un repositorio privado |

La entrada de genlock en `.gitmodules` tiene `update = none`, así que un clon se la salta. `ATHENEA_WITH_GENLOCK` solo está activado cuando `third_party/genlock` está descargado; sin él se compila todo salvo el módulo `sched` y `athenea live` (ver el manual de desarrollo, *Dependencias y toolchain*). Donde el repositorio está disponible, `git submodule update --init --checkout third_party/genlock` lo trae de vuelta.

## Compilación

### macOS

```sh
scripts/build-usd.sh && scripts/build-oidn.sh && scripts/build-ocio.sh
cmake --preset macos-arm64-release          # o macos-arm64-debug
cmake --build --preset macos-arm64-release
ctest --preset macos-arm64-debug            # las pruebas de GPU se saltan en una máquina sin dispositivo
```

El binario es `build/macos-arm64-release/bin/athenea`. El plugin de Hydra y los schemas codeless están en `build/macos-arm64-release/plugin/usd`.

El dispositivo es Metal; `scripts/build-oidn.sh` compila el dispositivo Metal de OIDN.

### Linux

```sh
scripts/build-usd.sh && scripts/build-oidn.sh && scripts/build-ocio.sh
cmake --preset linux-x86_64-release         # o linux-x86_64-debug
cmake --build --preset linux-x86_64-release
ctest --preset linux-x86_64-debug
```

El dispositivo es CUDA, o Vulkan sin un dispositivo CUDA. Un único toolkit de CUDA sirve para los kernels de gpe, el dispositivo CUDA de slang-rhi y OIDN, cuyo dispositivo CUDA pide la 12.8 o posterior: `ATHENEA_CUDA_ROOT` lo nombra, y sin él el configure toma el `/usr/local/cuda-*` más reciente que tenga `nvcc`. Las cabeceras de OptiX que necesita el ray tracing por pipeline en CUDA las descarga slang-rhi.

### Windows

Desde PowerShell:

```powershell
scripts\build-usd.ps1
scripts\build-oidn.ps1
scripts\build-ocio.ps1
cmake --preset windows-x64-release          # o windows-x64-debug
cmake --build --preset windows-x64-release
```

Los scripts entran en el entorno MSVC x64 del Visual Studio más reciente con las herramientas de C++ (`scripts\msvc-env.ps1`, que lo encuentra con vswhere), salvo que `cl` ya esté en el path. Sus prefijos son `$env:ATHENEA_USD_ROOT`, `$env:ATHENEA_OIDN_ROOT` y `$env:ATHENEA_OCIO_ROOT`, o los predeterminados de `D:\tools` de arriba. Los presets compilan con `cl`. El dispositivo es CUDA, y OIDN se compila para CUDA, como en Linux. El plugin OpenFX no se compila en Windows.

### Los presets

| Preset de configuración | Plataforma |
|---|---|
| `macos-arm64-debug`, `macos-arm64-release` | macOS arm64 (Metal) |
| `linux-x86_64-debug`, `linux-x86_64-release` | Linux x86_64 (CUDA/Vulkan) |
| `windows-x64-debug`, `windows-x64-release` | Windows x64 (CUDA) |
| `ios-arm64-debug`, `ios-arm64-release` | iOS arm64 (Metal): el motor y su delegate de Hydra como librerías estáticas, sin la CLI, el host AOFX, el servidor MCP ni las pruebas |
| `macos-arm64-blender` | macOS arm64: hdAthenea para Blender (Metal) |
| `linux-x86_64-blender` | Linux x86_64: hdAthenea para Blender (CUDA/Vulkan) |
| `windows-x64-blender` | Windows x64: hdAthenea para Blender (CUDA/Vulkan) |

Los presets debug compilan `Debug`; los release, `RelWithDebInfo`. Cada preset de escritorio tiene un preset de build y uno de pruebas con el mismo nombre; las pruebas se ejecutan de una en una, porque comparten la GPU.

### El add-on de Blender

El add-on de Blender se compila contra cabeceras que coinciden con el OpenUSD propio de Blender: `scripts/build-usd-blender.sh` en macOS y Linux (`BLENDER_LIB` nombra la carpeta `lib` de Blender en Linux), `scripts\build-usd-blender.ps1 -BlenderDir <carpeta>` en Windows, y después el preset `macos-arm64-blender`, `linux-x86_64-blender` o `windows-x64-blender`. Bajo WSL un bake se rechaza: allí no hay OptiX. Los pasos están en [Gaussian splats en Blender](tutorials/blender.md) y en la sección *Dependencias y toolchain* del manual de desarrollo.

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
| genlock | `athenea live` |
