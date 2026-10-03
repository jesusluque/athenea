<!-- Copyright (c) 2026 jesus luque. -->
# The add-on's third-party licences

The `athenea_hydra` add-on is athenea's, under the Apache License, Version 2.0
(`LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` beside this file, which
cover what athenea's own binaries compile in: slang-rhi, metal-cpp, tinyexr,
miniz, CLI11, JSON for Modern C++, aopenfx, spz, gpe, genlock, mesh2splat's
port, the ACES and SOG ports). This file lists the libraries the add-on
package carries or compiles against beyond those, and where each one's
licence files are in `licenses/`. `scripts/package-blender-addon.sh` copies
every file named here from the toolchain's source and install trees, and
stops if one is missing.

## Carried by the package

| Component | Version | Licence | What the package holds of it | Licence files |
|---|---|---|---|---|
| [Slang](https://github.com/shader-slang/slang) | 2026.14.1 | Apache-2.0 WITH LLVM-exception; its third parties under the licences in `LICENSES/` | `libslang-compiler` and `libslang-glsl-module` beside `hdAthenea` | `licenses/slang/` (`LICENSE`, `LICENSES/*`) |
| [MaterialX](https://github.com/AcademySoftwareFoundation/MaterialX) | 1.39.5 | Apache-2.0 | `MaterialXGenSlang` and the hardware nodes compiled into `hdAthenea`; the `libraries/` data in `plugin/materialx` | `licenses/materialx/` (`LICENSE`, `THIRD-PARTY.md`) |
| [OpenVDB / NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb) | 10.1.0 | MPL-2.0 | `PNanoVDB.h` in `shaders/nanovdb`, only when the build had OpenVDB (the Blender build does not) | `licenses/openvdb/LICENSE` |

## Blender's own, not redistributed

hdAthenea for Blender links the libraries Blender ships in
`Blender.app/Contents/Resources/lib` and loads the ones Blender has already
loaded, so the package carries none of them. It is compiled against their
headers, and code inlined from a header is in `hdAthenea`, so their licence
files travel with it all the same.

| Component | Headers built against | Blender's library | Licence | Licence files |
|---|---|---|---|---|
| [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) | 26.03 | `libusd_ms` (26.03) | Apache-2.0 (Modified, as OpenUSD's `LICENSE.txt` states) | `licenses/openusd/` (`LICENSE.txt`, `NOTICE.txt`) |
| [oneTBB](https://github.com/uxlfoundation/oneTBB) | 2022.3.0 | `libtbb` | Apache-2.0 | `licenses/onetbb/` (`LICENSE.txt`, `third-party-programs.txt`) |
| [MaterialX](https://github.com/AcademySoftwareFoundation/MaterialX) | 1.39.4 configs | `libMaterialX*` (1.39.4) | Apache-2.0 | `licenses/materialx/` (above) |
| [OpenColorIO](https://github.com/AcademySoftwareFoundation/OpenColorIO) | 2.5.2 | `libOpenColorIO` (2.5.0) | BSD 3-Clause | `licenses/opencolorio/` (`LICENSE`, `THIRD-PARTY.md`) |
| [Open Image Denoise](https://github.com/RenderKit/oidn) | 2.5.1 | `libOpenImageDenoise*` (2.5.0) | Apache-2.0 | `licenses/oidn/` (`LICENSE.txt`, `third-party-programs.txt`, `third-party-programs-oneTBB.txt`, `third-party-programs-DPCPP.txt`) |

## Linked from the system, not carried

| Component | Where it is loaded from | Licence |
|---|---|---|
| [zstd](https://github.com/facebook/zstd) | `/opt/homebrew/opt/zstd/lib/libzstd.1.dylib` | BSD 3-Clause (or GPL-2.0) |
| [libwebp](https://chromium.googlesource.com/webm/libwebp) | `/opt/homebrew/opt/webp/lib/libwebp.7.dylib` | BSD 3-Clause |

`hdAthenea` names these by their Homebrew paths, so the package runs only on a
machine that has them there. A package that carries them changes those names
to `@loader_path` and carries their `LICENSE` / `COPYING` beside them.
