<!-- Copyright (c) 2026 jesus luque. -->
# Third-party notices

athenea is licensed under the Apache License, Version 2.0 (`LICENSE`, with
`NOTICE`). It carries, compiles in or links code that others wrote under
their own licences, listed below. Each licence text is reproduced verbatim
from the component's own licence file at the version the build pins. Where a
component's licence is the Apache License, Version 2.0, its text is the one in
`LICENSE` beside this file and is not repeated; the component's copyright line
is given here.

This file is installed with the binaries (`share/doc/athenea/`), inside every
AOFX bundle (`Contents/Resources/`) and beside the Hydra plugin
(`hdAthenea/resources/`), together with `LICENSE` and `NOTICE`.

## 1. mesh2splat (Electronic Arts)

- **Source:** https://github.com/electronicarts/mesh2splat (`main`, v1.2.0 /
  `c3d3d52`).
- **What athenea holds of it:** `plugins/mesh2splat/mesh2splat.slang` is
  derived from mesh2splat's conversion (`converterGS.glsl`,
  `converterFS.glsl`) and ported to a Slang compute kernel. It is compiled
  into the `Mesh2Splat.aofx.bundle` plugin, which `athenea mesh2splat` runs.
- **Licence:** BSD 3-Clause with a fourth clause on EA's marks and logos.

How it sits under athenea's Apache-2.0 licence:

- The ported file keeps EA's copyright notice, the four conditions and the
  disclaimer verbatim in its header (clause 1), and this file reproduces them
  for binary distributions (clause 2).
- The BSD terms are permissive and impose no condition that Apache-2.0
  contradicts. athenea's own modifications to the port are offered under
  Apache-2.0; EA's original code in it remains under EA's licence below, and
  Apache-2.0 grants no rights in it beyond what that licence grants.
- athenea names Electronic Arts only to attribute the origin of the
  algorithm; it does not use EA's name to endorse or promote anything
  (clause 3).
- athenea distributes none of EA's or SEED's marks or logos (clause 4).

```text
Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
  
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  
1.  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3.  Neither the name of Electronic Arts, Inc. ("EA") nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
4.  EA’s marks or logos, including SEED logos, are distributed with this software solely for demonstration purposes and may not be displayed or shared other than as part of a redistribution of this software, provided they are redistributed without modification.  No other rights are provided for the use of these marks and logos, other than for their intended demonstration purposes.
  
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 2. Other code athenea's sources derive from

| Component | Licence | Copyright | Where in athenea |
|---|---|---|---|
| ACES 2.0 output transform, [aces-core](https://github.com/ampas/aces-core) | Apache-2.0 (text in `LICENSE`) | Copyright Contributors to the ACES Project. | `shaders/athenea/technique/aces2.slang`, `aces2_prepare.slang`: ported function for function from `Lib.Academy.OutputTransform.ctl` and `Lib.Academy.Tonescale.ctl` |
| SOG format, [PlayCanvas engine](https://github.com/playcanvas/engine) and [splat-transform](https://github.com/playcanvas/splat-transform) | MIT (below) | Copyright (c) 2011-2026 PlayCanvas Ltd. | `modules/io/src/Sog.cpp`, `shaders/athenea/scene/sog_decode.slang`: the meaning of SOG's bytes, after `gsplat-sog-data.js` |
| [MaterialX](https://github.com/AcademySoftwareFoundation/MaterialX) Slang generator | Apache-2.0 (text in `LICENSE`) | Copyright Contributors to the MaterialX Project | `modules/material` (`MaterialCompiler`) builds on `MaterialXGenSlang` and emits its node implementations |
| [spz](https://github.com/nianticlabs/spz) (vendored in `third_party/spz`) | MIT (below) | Copyright (c) 2024 Niantic Labs; the source files carry Copyright (c) 2025 Niantic Labs and Copyright (c) 2025 Adobe Inc. | `io::readSpz`, compiled into the binaries |

PlayCanvas (MIT):

```text
Copyright (c) 2011-2026 PlayCanvas Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

spz (MIT, `third_party/spz/LICENSE`):

```text
MIT License

Copyright (c) 2024 Niantic Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. Compiled into athenea's binaries

These are fetched and pinned by `cmake/Dependencies.cmake` or are submodules,
and are linked statically into `athenea`, `athenea-mcp`, `hdAthenea` and the
AOFX bundles.

| Component | Version | Licence |
|---|---|---|
| [slang-rhi](https://github.com/shader-slang/slang-rhi) | `e17f6d7`, patched (`cmake/patches/`) | Apache-2.0 WITH LLVM-exception (below) |
| [metal-cpp](https://developer.apple.com/metal/cpp/) (fetched by slang-rhi; macOS) | as slang-rhi pins it | Apache-2.0 (text in `LICENSE`), Copyright © 2024 Apple Inc. |
| [tinyexr](https://github.com/syoyo/tinyexr) | v1.0.12 | BSD 3-Clause, with OpenEXR portions under BSD 3-Clause (below) |
| [miniz](https://github.com/richgel999/miniz) (bundled with tinyexr) | as tinyexr bundles it | MIT (below) |
| [CLI11](https://github.com/CLIUtils/CLI11) | v2.5.0 | BSD 3-Clause (below) |
| [JSON for Modern C++](https://github.com/nlohmann/json) | v3.12.0 | MIT (below) |
| [GLFW](https://github.com/glfw/glfw) | 3.4 | zlib (below) |
| [Dear ImGui](https://github.com/ocornut/imgui) | v1.92.9 | MIT (below) |
| [aopenfx](https://github.com/jesusluque/aopenfx) (submodule) | `sparrow` branch | BSD 3-Clause (below) |
| [spz](https://github.com/nianticlabs/spz) | vendored | MIT (section 2) |

`gpe` and `genlock` (submodules) are by athenea's author and carry no
third-party licence of their own; gpe's copy of metal-cpp is the Apple
component listed above.

slang-rhi (`LICENSE`; its full Apache-2.0 text is the one in `LICENSE`):

```text
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

LLVM Exceptions to the Apache 2.0 License

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
```

tinyexr (`tinyexr.h`):

```text
/*
Copyright (c) 2014 - 2021, Syoyo Fujita and many contributors.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Syoyo Fujita nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

tinyexr's OpenEXR portions (`tinyexr.h`):

```text
// TinyEXR contains some OpenEXR code, which is licensed under ------------

///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

// End of OpenEXR license -------------------------------------------------
```

miniz (`deps/miniz/LICENSE`):

```text
Copyright 2013-2014 RAD Game Tools and Valve Software
Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC

All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

CLI11 (`LICENSE`):

```text
CLI11 2.2 Copyright (c) 2017-2025 University of Cincinnati, developed by Henry
Schreiner under NSF AWARD 1414736. All rights reserved.

Redistribution and use in source and binary forms of CLI11, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

JSON for Modern C++ (`LICENSE.MIT`):

```text
MIT License 

Copyright (c) 2013-2025 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

GLFW (`LICENSE.md`):

```text
Copyright (c) 2002-2006 Marcus Geelnard

Copyright (c) 2006-2019 Camilla Löwy

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would
   be appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not
   be misrepresented as being the original software.

3. This notice may not be removed or altered from any source
   distribution.
```

Dear ImGui (`LICENSE.txt`):

```text
The MIT License (MIT)

Copyright (c) 2014-2026 Omar Cornut

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

aopenfx (`LICENSE`):

```text
BSD 3-Clause License

Copyright (c) 2026, aopenfx contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 4. Shared libraries athenea links at run time

athenea's build does not copy these libraries; it links the installs the
toolchain scripts make (`scripts/build-usd.sh`, `build-oidn.sh`,
`build-ocio.sh`) and the system's. A package that carries any of them -- a
plugin package with `libslang` beside `hdAthenea`, say -- carries that
library's own licence files with it as well.

| Component | Version | Licence | Copyright |
|---|---|---|---|
| [Slang](https://github.com/shader-slang/slang) | 2026.14.1 | Apache-2.0 WITH LLVM-exception (below) | see the Slang release's `LICENSE` and `LICENSES/` |
| [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) | 26.08 | Tomorrow Open Source Technology License 1.0 (`LICENSE.txt`, with the third-party licences it lists) | Copyright 2016 Pixar (`NOTICE.txt`) |
| [MaterialX](https://github.com/AcademySoftwareFoundation/MaterialX) | 1.39.5 | Apache-2.0 (text in `LICENSE`; see its `THIRD-PARTY.md`) | Copyright Contributors to the MaterialX Project |
| [OpenSubdiv](https://github.com/PixarAnimationStudios/OpenSubdiv) | 3.6.1 | Modified Apache 2.0 License (`LICENSE.txt`) | Pixar |
| [oneTBB](https://github.com/uxlfoundation/oneTBB) | 2021.12.0 | Apache-2.0 (text in `LICENSE`) | Intel Corporation |
| [OpenVDB / NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb) | 10.1.0 | MPL-2.0 | Copyright Contributors to the OpenVDB Project |
| [OpenEXR and Imath](https://github.com/AcademySoftwareFoundation/openexr) (in the OpenUSD install) | 3.1.13 | BSD 3-Clause | Copyright Contributors to the OpenEXR Project |
| [c-blosc](https://github.com/Blosc/c-blosc) | 1.21.6 | BSD 3-Clause | Copyright (c) 2009-2018 Francesc Alted; Copyright (c) 2019-present Blosc Development Team |
| [Boost](https://www.boost.org/) (through OpenVDB) | 1.82.0 | Boost Software License 1.0 | the Boost authors |
| [Open Image Denoise](https://github.com/RenderKit/oidn) | 2.5.1 | Apache-2.0 (text in `LICENSE`; see its `third-party-programs.txt`) | Intel Corporation |
| [OpenColorIO](https://github.com/AcademySoftwareFoundation/OpenColorIO) | 2.5.2 | BSD 3-Clause (below; see its `THIRD-PARTY.md`) | Copyright Contributors to the OpenColorIO Project. |
| [libwebp](https://chromium.googlesource.com/webm/libwebp) | the system's | BSD 3-Clause (below) | Copyright (c) 2010, Google Inc. |
| [Zstandard](https://github.com/facebook/zstd) | the system's | BSD 3-Clause (below) | Copyright (c) Meta Platforms, Inc. and affiliates. |
| [zlib](https://zlib.net/) | the system's | zlib | Jean-loup Gailly and Mark Adler |

NanoVDB's `PNanoVDB.h` is copied by the build into its shader directory and
compiled at run time; it is used unmodified and keeps its own MPL-2.0 header.
OpenVDB's source is at the link above (MPL-2.0, section 3.2).

Slang (`LICENSE`; its full Apache-2.0 text is the one in `LICENSE`):

```text
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

LLVM Exceptions to the Apache 2.0 License

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
```

OpenColorIO (`LICENSE`):

```text
Copyright Contributors to the OpenColorIO Project.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

libwebp (`COPYING`):

```text
Copyright (c) 2010, Google Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * Neither the name of Google nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

Zstandard (`LICENSE`):

```text
BSD License

For Zstandard software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook, nor Meta, nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 5. Used to build or test only, not distributed

- [Catch2](https://github.com/catchorg/Catch2) v3.8.1 (Boost Software
  License 1.0): the test binaries, which are not installed.
- The OptiX SDK headers (NVIDIA), fetched by slang-rhi on Linux so nvrtc can
  compile tracing kernels at run time; they stay in the build tree.
