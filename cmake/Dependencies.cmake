# Third-party dependencies, fetched at configure time and pinned.
#
# Two are not fetched, on purpose:
#   - Slang itself: one version for the whole process, the one gpe's slangc
#     compiles blobs with (~/tools/slang, 2026.14.1) -- slang-rhi links that
#     same install. Two Slang runtimes in one process is not a thing to debug.
#   - OpenUSD: tens of minutes to build and its own oneTBB, so it is built once
#     per machine by scripts/build-usd.sh and found here (cmake/Usd.cmake).

include(FetchContent)
include(CMakeDependentOption)
set(FETCHCONTENT_QUIET ON)

# --- Slang --------------------------------------------------------------------

set(SLANG_ROOT "$ENV{SLANG_ROOT}" CACHE PATH "Slang release install (bin/, lib/, include/)")
if(NOT SLANG_ROOT)
    if(ATHENEA_IOS)
        # Slang is not a tool here but a library the app carries: the phone
        # compiles the shaders it opens. Built static, for one signed binary,
        # by IOLucabRTrender/scripts/build-slang-ios.sh.
        set(SLANG_ROOT "$ENV{HOME}/tools/ios/slang-2026.14.1" CACHE PATH "" FORCE)
    else()
        set(SLANG_ROOT "$ENV{HOME}/tools/slang" CACHE PATH "" FORCE)
    endif()
endif()
if(NOT EXISTS "${SLANG_ROOT}/include/slang.h")
    message(FATAL_ERROR
        "Slang not found at ${SLANG_ROOT}. Unpack the 2026.14.1 release from "
        "github.com/shader-slang/slang there, or set SLANG_ROOT.")
endif()
# gpe searches for slangc itself; pointing its search at the same install keeps
# build-time blobs and run-time compilation on one compiler.
set(ENV{SLANG_ROOT} "${SLANG_ROOT}")
if(ATHENEA_IOS)
    list(APPEND CMAKE_FIND_ROOT_PATH "${SLANG_ROOT}")
endif()
if(NOT ATHENEA_IOS)
    # gpe, which compiles blobs at build time. The iOS build has no gpe and no
    # blobs: every shader it runs is compiled on the device.
    find_program(GPE_SLANGC slangc HINTS "${SLANG_ROOT}/bin" NO_DEFAULT_PATH REQUIRED)
endif()

# --- OpenUSD ------------------------------------------------------------------
#
# Declared here, beside Slang, so a build outside the presets can be told
# where it is; found in cmake/Usd.cmake once the submodules are in.
set(ATHENEA_USD_ROOT "$ENV{HOME}/tools/usd-26.08-mx"
    CACHE PATH "OpenUSD built with MaterialX's Slang generator (scripts/build-usd.sh)")
if(NOT ATHENEA_BLENDER_LIB AND NOT EXISTS "${ATHENEA_USD_ROOT}/pxrConfig.cmake")
    message(FATAL_ERROR
        "OpenUSD not found at ${ATHENEA_USD_ROOT} (no pxrConfig.cmake). Build it with "
        "scripts/build-usd.sh, or set ATHENEA_USD_ROOT.")
endif()

# --- slang-rhi -----------------------------------------------------------------
#
# Pinned to a commit rather than a tag: slang-rhi does not cut releases, and
# its API moves. e17f6d7 is 2026-09-03 ("Add opacity micromap support").

set(SLANG_RHI_FETCH_SLANG OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_SLANG_INCLUDE_DIR "${SLANG_ROOT}/include" CACHE STRING "" FORCE)
set(SLANG_RHI_SLANG_BINARY_DIR "${SLANG_ROOT}" CACHE STRING "" FORCE)
set(SLANG_RHI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_BUILD_TESTS_WITH_GLFW OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_INSTALL OFF CACHE BOOL "" FORCE)
# No CPU backend: nothing in this engine may run its numbers on the CPU, and a
# backend that exists gets used as a fallback by someone. No WebGPU: Dawn is a
# large download for a target this engine does not have.
set(SLANG_RHI_ENABLE_CPU OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_ENABLE_WGPU OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_ENABLE_D3D11 OFF CACHE BOOL "" FORCE)
if(APPLE)
    # Metal only. Vulkan on macOS is MoltenVK over Metal, which is the device
    # we already have minus a translation layer.
    set(SLANG_RHI_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
endif()
if(ATHENEA_IOS)
    # CUDA's headers are fetched by slang-rhi whenever its CUDA backend is on,
    # and a phone has no such device to fetch them for.
    set(SLANG_RHI_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    set(SLANG_RHI_ENABLE_OPTIX OFF CACHE BOOL "" FORCE)
    set(SLANG_RHI_ENABLE_D3D12 OFF CACHE BOOL "" FORCE)
endif()

# Patched: see cmake/patches/. Each patch is a bug found here, small, and
# worth sending upstream; the list should shrink when the pin moves.
#   - slang-rhi-metal-render-target-array-length.patch: every Metal render
#     pass with more than one colour target was invalid.
#   - slang-rhi-metal-texture-view-format.patch: a view of a whole texture in
#     another format (sRGB over linear) came back as the texture itself, in
#     the texture's format.
#   - slang-rhi-cuda-driver-symbols.patch: slang-rhi loads the CUDA driver by
#     dlopen and holds its entry points in variables carrying the driver's own
#     names. At global scope those are the definitions the rest of the program
#     binds to, so gpe and OIDN called through slang-rhi's pointers instead of
#     libcuda and crashed on a null one. A namespace keeps them to slang-rhi.
#   - slang-rhi-metal-acceleration-structures.patch: once any
#     acceleration structure had been freed, the next build threw inside
#     Metal (a nil in the device's structure array) and aborted the process;
#     and an indexed triangle build took max(vertices, indices) / 3 triangles,
#     reading past its index window.
#   - slang-rhi-metal-command-buffer-errors.patch: a command buffer that
#     failed on the device (out of memory, under another job's load) hit an
#     assertion in its completion handler and aborted the process. The error
#     is now kept on the queue and returned by the next submit() or
#     waitOnHost() -- SLANG_E_OUT_OF_MEMORY for MTLCommandBufferErrorOutOfMemory
#     -- and waitOnHost no longer waits on a tracking event a failed buffer
#     never signals. A buffer or acceleration structure Metal will not make
#     is SLANG_E_OUT_OF_MEMORY rather than SLANG_FAIL.
#   - slang-rhi-ios.patch (iOS only): two things the backend assumes a Mac
#     for. Slang is linked statically, as everything in an app bundle is, and
#     the imported target had no iOS branch to say so. And the Metal target
#     was a metallib, which Xcode's `metal` compiler makes -- a downstream
#     tool no phone carries; on iOS the target is Metal source and Metal
#     compiles it itself.
set(ATHENEA_RHI_PATCHES
    "${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-metal-render-target-array-length.patch"
    "${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-metal-acceleration-structures.patch"
    "${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-metal-texture-view-format.patch"
    "${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-cuda-driver-symbols.patch"
    "${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-metal-command-buffer-errors.patch")
if(ATHENEA_IOS)
    list(APPEND ATHENEA_RHI_PATCHES "${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-ios.patch")
endif()
string(JOIN "$<SEMICOLON>" ATHENEA_RHI_PATCHES ${ATHENEA_RHI_PATCHES})

if(NOT TARGET slang-rhi)
FetchContent_Declare(slang_rhi
    GIT_REPOSITORY https://github.com/shader-slang/slang-rhi.git
    GIT_TAG        e17f6d75f858f9b7cb91bc102a7b8c6fda0435dc
    GIT_SHALLOW    FALSE
    GIT_SUBMODULES ""
    PATCH_COMMAND  ${CMAKE_COMMAND}
                   "-DPATCHES=${ATHENEA_RHI_PATCHES}"
                   -P "${CMAKE_CURRENT_LIST_DIR}/patches/apply.cmake"
    UPDATE_DISCONNECTED TRUE
    SYSTEM)
FetchContent_MakeAvailable(slang_rhi)
endif()

if(ATHENEA_IOS AND TARGET slang-rhi)
    # Xcode compiles every iOS target with -Wshorten-64-to-32, and slang-rhi
    # compiles its own sources with -Werror: `size()` into a uint32_t count is
    # how its headers are written, and it is not this build's warning to fix.
    target_compile_options(slang-rhi PRIVATE -Wno-shorten-64-to-32)
endif()

# Where slang-rhi put the OptiX headers it fetches itself. Slang's CUDA path
# compiles through nvrtc at run time and includes <optix.h> for any kernel that
# traces, and it looks for an installed SDK -- which there is none of here, so
# the build tells the engine where the fetched headers are and the device hands
# nvrtc the include path (modules/gpu/src/Device.cpp).
# slang-rhi fetches them into its own scope, so the directory is what says
# where they landed rather than a variable of ours.
foreach(version 9_0 8_1 8_0)
    set(candidate "${CMAKE_BINARY_DIR}/_deps/optix_${version}-src/include")
    if(EXISTS "${candidate}/optix.h")
        set(ATHENEA_OPTIX_INCLUDE_DIR "${candidate}" CACHE PATH "OptiX headers for nvrtc" FORCE)
        break()
    endif()
endforeach()
if(ATHENEA_OPTIX_INCLUDE_DIR)
    message(STATUS "gpu: OptiX headers for nvrtc at ${ATHENEA_OPTIX_INCLUDE_DIR}")
endif()

# --- small libraries ------------------------------------------------------------

if(ATHENEA_BUILD_APPS)
    FetchContent_Declare(cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG        v2.5.0
        GIT_SHALLOW    TRUE
        SYSTEM)
    set(CLI11_PRECOMPILED OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(cli11)
endif()

# A parent that found nlohmann_json already has the imported target; one copy.
if(NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        SYSTEM)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# tinyexr writes the renders. Its own miniz, compiled here, so EXR's zip
# compression needs nothing from the system.
FetchContent_Declare(tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG        v1.0.12
    GIT_SHALLOW    TRUE
    SYSTEM)
FetchContent_GetProperties(tinyexr)
if(NOT tinyexr_POPULATED)
    FetchContent_Populate(tinyexr)
endif()
add_library(athenea_tinyexr STATIC
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c)
target_include_directories(athenea_tinyexr SYSTEM PUBLIC
    ${tinyexr_SOURCE_DIR} ${tinyexr_SOURCE_DIR}/deps/miniz)
set_target_properties(athenea_tinyexr PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- GLFW and Dear ImGui: athenea view ---------------------------------------------
#
# The viewer's window and its panels. GLFW with no client API: slang-rhi makes
# the surface. Dear ImGui tessellates its panels on the CPU -- chrome, not
# scene data -- and draws through the engine's own slang-rhi backend
# (modules/view/src/ImGuiRenderer.cpp); only its GLFW input backend is used.

# GLFW opens a desktop window; on the phone the window is the app's own, and
# the panels are its native controls.
cmake_dependent_option(ATHENEA_BUILD_VIEW "athenea view: a window onto a stage (GLFW, Dear ImGui)" ON
                       "NOT ATHENEA_IOS" OFF)
if(ATHENEA_BUILD_VIEW)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
        GIT_SHALLOW    TRUE
        SYSTEM)
    FetchContent_MakeAvailable(glfw)

    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.92.9
        GIT_SHALLOW    TRUE
        SYSTEM)
    FetchContent_GetProperties(imgui)
    if(NOT imgui_POPULATED)
        FetchContent_Populate(imgui)
    endif()
    add_library(athenea_imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp)
    target_include_directories(athenea_imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
    # 32-bit indices: the renderer pulls vertices through them from a buffer.
    target_compile_definitions(athenea_imgui PUBLIC "ImDrawIdx=unsigned int" IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
    target_link_libraries(athenea_imgui PUBLIC glfw)
    set_target_properties(athenea_imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_TESTING AND NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.1
        GIT_SHALLOW    TRUE
        SYSTEM)
    FetchContent_MakeAvailable(catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()
