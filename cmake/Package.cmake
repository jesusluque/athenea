# Copyright (c) 2026 jesus luque.
#
# The package: `cpack` over `cmake --install`, one ZIP, from a release build
# tree.
#
#   cmake --build --preset windows-x64-release
#   cpack --config build/windows-x64-release/CPackConfig.cmake -B <out>
#
# Only the `athenea` component is packaged: the rules this project declares
# (CMAKE_INSTALL_DEFAULT_COMPONENT_NAME, set at the top before the first of
# them), not the ones its FetchContent dependencies carry (expat's headers,
# CMake configs). `cmake --install <build> --component athenea --prefix <p>`
# lays out the same folder without the archive.
#
# On Windows the install also carries what the programs load, since a DLL is
# looked for beside the program before the PATH (cmake/InstallRuntime.cmake):
# the toolchain's DLLs into bin/, OpenUSD's plugInfo tree into bin/usd/, its
# plugins into plugin/usd/, MaterialX's data into libraries/, CUDA's
# redistributable NVRTC and nvJitLink, the MSVC runtime, and then a check that
# every DLL the programs, hdAthenea and the bundles name is in bin/ or the
# system's. With them come the per-user install and uninstall scripts and the
# Blender add-on, when ATHENEA_PACKAGE_BLENDER_ADDON names its zip (a
# windows-x64-blender build, scripts/package-blender-addon.ps1).

if(ATHENEA_IOS OR ATHENEA_EMBEDDED)
    return()
endif()

set(ATHENEA_PACKAGE_BLENDER_ADDON "" CACHE FILEPATH
    "The Blender add-on's zip to carry in the package's blender/ folder (empty: none)")

# Documentation beside the licence files: the operations manual in both languages.
install(FILES "${ATHENEA_ROOT}/docs/operations.md" "${ATHENEA_ROOT}/docs/operations.es.md"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}")
if(EXISTS "${ATHENEA_ROOT}/CHANGELOG.md")
    install(FILES "${ATHENEA_ROOT}/CHANGELOG.md" DESTINATION "${CMAKE_INSTALL_DOCDIR}")
endif()

if(WIN32)
    set(_package_system "windows-x64")

    # The per-user installer at the package's root, with its notes.
    install(FILES "${ATHENEA_ROOT}/packaging/windows/install.ps1"
                  "${ATHENEA_ROOT}/packaging/windows/uninstall.ps1"
                  "${ATHENEA_ROOT}/packaging/windows/INSTALL.md"
            DESTINATION ".")
    if(ATHENEA_PACKAGE_BLENDER_ADDON)
        if(NOT EXISTS "${ATHENEA_PACKAGE_BLENDER_ADDON}")
            message(FATAL_ERROR "ATHENEA_PACKAGE_BLENDER_ADDON: no ${ATHENEA_PACKAGE_BLENDER_ADDON}")
        endif()
        install(FILES "${ATHENEA_PACKAGE_BLENDER_ADDON}" DESTINATION "blender")
    endif()

    # What the runtime step reads: the binaries to check, where the toolchain is.
    set(_runtime_executables "")
    foreach(_program athenea athenea-mcp)
        if(TARGET ${_program})
            list(APPEND _runtime_executables "$<TARGET_FILE_NAME:${_program}>")
        endif()
    endforeach()
    set(_runtime_modules "")
    if(TARGET hdAthenea)
        list(APPEND _runtime_modules "${ATHENEA_PLUGIN_INSTALL_DIR}/hdAthenea/$<TARGET_FILE_NAME:hdAthenea>")
    endif()
    set(_cuda_bin "")
    if(CMAKE_CUDA_COMPILER)
        get_filename_component(_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    elseif(DEFINED ENV{CUDA_PATH})
        set(_cuda_bin "$ENV{CUDA_PATH}/bin")
    endif()
    get_filename_component(_linker_dir "${CMAKE_LINKER}" DIRECTORY)
    install(CODE "
set(ATHENEA_RT_EXECUTABLES \"${_runtime_executables}\")
set(ATHENEA_RT_MODULES \"${_runtime_modules}\")
set(ATHENEA_RT_BUILD_BIN \"${ATHENEA_BUILD}/bin\")
set(ATHENEA_RT_SLANG \"${SLANG_ROOT}\")
set(ATHENEA_RT_USD \"${ATHENEA_USD_ROOT}\")
set(ATHENEA_RT_OCIO \"${ATHENEA_OCIO_ROOT}\")
set(ATHENEA_RT_OIDN \"${ATHENEA_OIDN_ROOT}\")
set(ATHENEA_RT_CUDA_BIN \"${_cuda_bin}\")
set(ATHENEA_RT_DUMPBIN \"${_linker_dir}/dumpbin.exe\")
set(ATHENEA_RT_AOFX_DIR \"${AOFX_BUNDLE_INSTALL_DIR}\")
set(ATHENEA_RT_OFX_DIR \"${ATHENEA_OFX_INSTALL_DIR}\")
set(ATHENEA_RT_DOCDIR \"${CMAKE_INSTALL_DOCDIR}\")
")
    install(SCRIPT "${ATHENEA_ROOT}/cmake/InstallRuntime.cmake")

    # The MSVC runtime of the compiler that built the programs, after the
    # toolchain's copies, so the newer one is what bin/ holds.
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    set(CMAKE_INSTALL_UCRT_LIBRARIES FALSE)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_COMPONENT athenea)
    include(InstallRequiredSystemLibraries)
elseif(APPLE)
    set(_package_system "macos-${CMAKE_SYSTEM_PROCESSOR}")
else()
    set(_package_system "linux-${CMAKE_SYSTEM_PROCESSOR}")
endif()

# --- CPack ------------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "athenea")
set(CPACK_PACKAGE_VENDOR "jesus luque")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "athenea: a GPU renderer for gaussian splats and USD stages")
set(CPACK_PACKAGE_FILE_NAME "athenea-${PROJECT_VERSION}-${_package_system}")
set(CPACK_RESOURCE_FILE_LICENSE "${ATHENEA_ROOT}/LICENSE")
set(CPACK_GENERATOR "ZIP")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
# A component archive asks its own variable: the folder athenea-<v>-<system> at the root.
set(CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY ON)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL athenea)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
set(CPACK_ARCHIVE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}")
set(CPACK_SOURCE_GENERATOR "")
include(CPack)
