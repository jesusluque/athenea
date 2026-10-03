cmake_minimum_required(VERSION 3.24)
project(athenea VERSION 0.1.0 LANGUAGES C CXX)
if(APPLE)
    # gpe's Metal backend is one .mm. A language enabled only inside a
    # subdirectory leaves CMake without its compile rule at generate time, so
    # it is enabled here, at the top, where every directory can see it.
    enable_language(OBJC OBJCXX)
endif()

# athenea: a Slang renderer for Gaussian splats and point clouds, and the
# engine openFXplayer will stand on. Linux and macOS first; Windows later, which
# is why nothing here names a POSIX call outside modules/core/Platform.
#
# The design, the decisions and why each was taken live in docs/decisions.md.
# The short version a build needs:
#
#   - slang-rhi is the render device (Metal on Apple, CUDA+OptiX or Vulkan on
#     Linux), with Slang compiled at run time and bound by reflection.
#   - gpe (submodule) is the compute/video engine openFXplayer already uses. It
#     does not make a second device: it adopts slang-rhi's, so a buffer written
#     by one is read by the other with nothing in between.
#   - aofx is the AOFX plugin SDK and its host, from aopenfx (third_party/aopenfx,
#     branch sparrow): the same headers and the same host every program that loads
#     bundles uses, so the bundles the compositor builds load here unchanged.
#   - OpenUSD is where entities live; a Hydra 2.0 render delegate feeds the
#     engine's flat RenderScene.
#   - genlock (submodule) is time: PTP and SMPTE ST 2059-1 alignment.
#
# Nothing numeric runs on the CPU, reference renders and test oracles
# included: the CPU reads files and parses headers, and the GPU does the rest.

# Where this engine's own files are and where its build lands, whoever the
# top-level project is. Built inside the compositor's tree, CMAKE_SOURCE_DIR and
# CMAKE_BINARY_DIR are the compositor's, and everything below that names a path
# of this engine's names it through these two.
set(ATHENEA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(ATHENEA_BUILD "${CMAKE_CURRENT_BINARY_DIR}")
# A parent that provides gpe, genlock, aopenfx and spz itself sets this and
# asks for the engine's apps, plugins and tests only by name. Standalone, all
# three are built; gpe, genlock and aopenfx are looked for by target either
# way, so the same file serves both shapes.
option(ATHENEA_EMBEDDED "Built inside another project's tree" OFF)
option(ATHENEA_BUILD_APPS "athenea and athenea-mcp" ON)
option(ATHENEA_BUILD_PLUGINS "the aofx bundles under plugins/" ON)

# iOS: the phone build. What it leaves out is what a phone has no use for --
# gpe (a video engine with a CUDA backend and a Qt viewer), genlock (PTP on a
# render farm), the CLI, the AOFX host (an app may not dlopen a bundle it did
# not ship), the MCP server and the tests. What is left is the engine and its
# Hydra delegate, as static libraries the app links.
# Configured by IOLucabRTrender (the preset ios-arm64-release).
set(ATHENEA_IOS OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(ATHENEA_IOS ON)
    set(ATHENEA_BUILD_APPS OFF CACHE BOOL "" FORCE)
    set(ATHENEA_BUILD_PLUGINS OFF CACHE BOOL "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

if(NOT ATHENEA_EMBEDDED)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${ATHENEA_BUILD}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${ATHENEA_BUILD}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${ATHENEA_BUILD}/lib")
endif()

# First in the module path, and this engine's own files included by path: a
# parent may well have a cmake/Warnings.cmake and a cmake/Dependencies.cmake of
# its own, and `include(Warnings)` found the parent's before this one was
# looked at.
list(PREPEND CMAKE_MODULE_PATH "${ATHENEA_ROOT}/cmake")

if(ATHENEA_IOS)
    # Cross-compiling, CMake looks for packages and libraries inside the
    # sysroot only. The prefixes built for the phone are roots of the same
    # kind, and have to be named as such or nothing in them is found.
    list(APPEND CMAKE_FIND_ROOT_PATH "${ATHENEA_USD_ROOT}")
endif()

include("${ATHENEA_ROOT}/cmake/Warnings.cmake")
# Before Dependencies: it fetches Catch2 only when BUILD_TESTING is on, and in a
# fresh build directory nothing defines BUILD_TESTING until CTest does. A parent
# decides BUILD_TESTING for itself.
if(ATHENEA_IOS)
    # ctest cannot run a binary on a phone, and Catch2's main is a console.
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
endif()
if(NOT ATHENEA_EMBEDDED)
    include(CTest)
endif()
include("${ATHENEA_ROOT}/cmake/Dependencies.cmake")
include("${ATHENEA_ROOT}/cmake/Shaders.cmake")
athenea_shader_copy_target()

# --- submodules -----------------------------------------------------------------
#
# Pinned, like openFXplayer pins them, and overridable for the same reason: a
# person working on gpe and this engine at once points ATHENEA_GPE_DIR at their
# own checkout and edits in place.

# One CUDA toolkit for everything that compiles or loads CUDA: gpe's kernels,
# slang-rhi's CUDA device and OIDN, whose CUDA device wants 12.8 or later
# (scripts/build-oidn.sh picks the same one). Distributions put an older
# toolkit's nvcc first on PATH, so ATHENEA_CUDA_ROOT, else the newest NVIDIA
# toolkit in /usr/local, is chosen here before gpe enables the language.
if(NOT APPLE AND NOT WIN32 AND NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    set(ATHENEA_CUDA_ROOT "$ENV{ATHENEA_CUDA_ROOT}" CACHE PATH "CUDA toolkit (default: the newest /usr/local/cuda-*)")
    if(NOT ATHENEA_CUDA_ROOT)
        file(GLOB _athenea_cuda_roots LIST_DIRECTORIES true "/usr/local/cuda-*")
        list(SORT _athenea_cuda_roots COMPARE NATURAL ORDER DESCENDING)
        foreach(_root IN LISTS _athenea_cuda_roots)
            if(EXISTS "${_root}/bin/nvcc")
                set(ATHENEA_CUDA_ROOT "${_root}" CACHE PATH "" FORCE)
                break()
            endif()
        endforeach()
    endif()
    if(ATHENEA_CUDA_ROOT AND EXISTS "${ATHENEA_CUDA_ROOT}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "${ATHENEA_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "" FORCE)
        set(CUDAToolkit_ROOT "${ATHENEA_CUDA_ROOT}" CACHE PATH "" FORCE)
        message(STATUS "CUDA toolkit: ${ATHENEA_CUDA_ROOT}")
    endif()
endif()

set(ATHENEA_GPE_DIR "${ATHENEA_ROOT}/third_party/gpe"
    CACHE PATH "Path to the gpe compute engine")
set(ATHENEA_GENLOCK_DIR "${ATHENEA_ROOT}/third_party/genlock"
    CACHE PATH "Path to the genlock timing library")

if(ATHENEA_IOS)
    message(STATUS "gpe, genlock: not built for the phone")
elseif(NOT TARGET gpe)
    if(NOT EXISTS "${ATHENEA_GPE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "gpe not found at ${ATHENEA_GPE_DIR}: "
                            "git submodule update --init --recursive")
    endif()
    # gpe's own tests are built only when asked for: they are gpe's to run, and
    # a consumer that builds them has two test suites fighting over one GPU.
    set(ATHENEA_BUILD_GPE_TESTS OFF CACHE BOOL "Build and register gpe's own tests")
    # gpe's QRhi viewer is a windowed presenter for gpe's own player. This
    # engine is headless and has no use for Qt.
    set(CMAKE_DISABLE_FIND_PACKAGE_Qt6 ON)
    # slang-rhi already defines metal-cpp's selector tables; one definition per
    # program. A parent that adds gpe itself decides this before it does.
    set(GPE_METAL_CPP_IMPLEMENTATION OFF CACHE BOOL "" FORCE)
    set(_athenea_testing ${BUILD_TESTING})
    set(BUILD_TESTING ${ATHENEA_BUILD_GPE_TESTS})
    add_subdirectory(${ATHENEA_GPE_DIR} ${ATHENEA_BUILD}/gpe)
    set(BUILD_TESTING ${_athenea_testing})
    unset(CMAKE_DISABLE_FIND_PACKAGE_Qt6)
    message(STATUS "gpe: ${ATHENEA_GPE_DIR} (backend ${GPE_BACKEND})")
else()
    message(STATUS "gpe: the parent project's (backend ${GPE_BACKEND})")
endif()

if(NOT ATHENEA_IOS AND NOT TARGET genlock::genlock)
    if(NOT EXISTS "${ATHENEA_GENLOCK_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "genlock not found at ${ATHENEA_GENLOCK_DIR}: "
                            "git submodule update --init --recursive")
    endif()
    set(GENLOCK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GENLOCK_BUILD_TOOLS ON CACHE BOOL "" FORCE)
    add_subdirectory(${ATHENEA_GENLOCK_DIR} ${ATHENEA_BUILD}/genlock EXCLUDE_FROM_ALL)
    if(TARGET genlock-cli)
        # Out of EXCLUDE_FROM_ALL and into bin/: `genlock-cli master` is what
        # a render node's --ptp points at, and a machine with one computer
        # needs to be able to serve the time it follows.
        set_target_properties(genlock-cli PROPERTIES EXCLUDE_FROM_ALL FALSE)
    endif()
endif()

# --- the engine -----------------------------------------------------------------

# aopenfx: the AOFX SDK and the reference host, one for every program that
# loads bundles. Its own repository, pinned as a submodule on its `sparrow`
# branch. The targets are skipped when a parent project already provides them
# (the compositor builds this engine inside its tree and adds aopenfx first),
# which is also why gpe and genlock above are looked for by target.
set(ATHENEA_AOFX_DIR "${ATHENEA_ROOT}/third_party/aopenfx"
    CACHE PATH "Path to aopenfx: the AOFX SDK and its host")
if(NOT EXISTS "${ATHENEA_AOFX_DIR}/sdk/include/aofx/Effect.h")
    message(FATAL_ERROR "aopenfx not found at ${ATHENEA_AOFX_DIR}: "
                        "git submodule update --init third_party/aopenfx")
endif()
if(NOT TARGET aofx::aofx)
    add_subdirectory(${ATHENEA_AOFX_DIR}/sdk ${ATHENEA_BUILD}/aofx-sdk)
endif()
if(NOT ATHENEA_IOS AND NOT TARGET aofx::host)
    add_subdirectory(${ATHENEA_AOFX_DIR}/host ${ATHENEA_BUILD}/aofx-host)
endif()

if(TARGET ofxp::spz)
    # The parent's copy of the same seven files: one set of symbols in the
    # process, not two.
    add_library(athenea_spz ALIAS ofxp_spz)
    set(ATHENEA_HAVE_SPZ TRUE CACHE INTERNAL "")
else()
    add_subdirectory(third_party/spz)
endif()
# zlib: the SOG bundle reads through it, and the tests write their fixtures
# with it. Found here, since an imported target belongs to the directory that
# found it and both modules/ and tests/ name it.
find_package(ZLIB REQUIRED)

# OpenUSD and MaterialX, seen by the modules that read them, the apps and tests.
include("${ATHENEA_ROOT}/cmake/Usd.cmake")
add_subdirectory(modules)
if(ATHENEA_BUILD_PLUGINS)
    add_subdirectory(plugins)
endif()
if(ATHENEA_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
