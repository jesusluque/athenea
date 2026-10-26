# Copyright (c) 2026 jesus luque.
#
# Run by `cmake --install` on Windows (cmake/Package.cmake sets the
# ATHENEA_RT_* variables before it): what the installed programs load, beside
# them, and a check that nothing they name is missing.
#
#   bin/            the toolchain's DLLs: OpenUSD (with MaterialX, OpenVDB,
#                   TBB, Boost, OpenEXR), OpenColorIO, Slang (loaded at run
#                   time by slang-rhi), OIDN with its device DLLs, the DLLs
#                   the build copied beside its programs (DXC for D3D12), and
#                   CUDA's NVRTC and nvJitLink, which slang-rhi loads by name
#   bin/usd/        OpenUSD's plugInfo tree, which it reads beside its own DLLs
#   plugin/usd/     OpenUSD's own plugins (beside hdAthenea's)
#   libraries/      MaterialX's data libraries, found from the MaterialX DLLs' parent
#
# The NVIDIA driver (nvcuda.dll) and everything else in System32 is the
# machine's. The check reads each installed binary's imports (dumpbin, through
# file(GET_RUNTIME_DEPENDENCIES)) and fails the install on a DLL that is
# neither in bin/ nor in the system's folders.

# Paths are matched with forward slashes (the prefix may come with backslashes).
if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)
endif()
file(TO_CMAKE_PATH "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}" _prefix)
set(_bin "${_prefix}/bin")

function(_rt_copy_glob destination)
    foreach(_pattern IN LISTS ARGN)
        file(GLOB _found LIST_DIRECTORIES false "${_pattern}")
        # NVRTC's .alt.dll is the variant for programs that link cudart statically; nothing here loads it.
        list(FILTER _found EXCLUDE REGEX "\\.alt\\.dll$")
        if(_found)
            file(INSTALL ${_found} DESTINATION "${destination}")
        endif()
    endforeach()
endfunction()

# --- the toolchain's DLLs ----------------------------------------------------------
_rt_copy_glob("${_bin}"
    "${ATHENEA_RT_USD}/bin/*.dll" "${ATHENEA_RT_USD}/lib/*.dll"
    "${ATHENEA_RT_SLANG}/bin/*.dll"
    "${ATHENEA_RT_BUILD_BIN}/*.dll")
if(ATHENEA_RT_OCIO)
    _rt_copy_glob("${_bin}" "${ATHENEA_RT_OCIO}/bin/*.dll")
endif()
if(ATHENEA_RT_OIDN AND EXISTS "${ATHENEA_RT_OIDN}/bin")
    _rt_copy_glob("${_bin}" "${ATHENEA_RT_OIDN}/bin/*.dll")
endif()
if(ATHENEA_RT_CUDA_BIN)
    _rt_copy_glob("${_bin}"
        "${ATHENEA_RT_CUDA_BIN}/nvrtc64_*.dll" "${ATHENEA_RT_CUDA_BIN}/nvrtc-builtins64_*.dll"
        "${ATHENEA_RT_CUDA_BIN}/nvJitLink_*.dll" "${ATHENEA_RT_CUDA_BIN}/cudart64_*.dll")
    get_filename_component(_cuda_root "${ATHENEA_RT_CUDA_BIN}" DIRECTORY)
    if(EXISTS "${_cuda_root}/EULA.txt")
        file(INSTALL "${_cuda_root}/EULA.txt" DESTINATION "${_prefix}/${ATHENEA_RT_DOCDIR}/third-party"
             RENAME "CUDA-EULA.txt")
    endif()
endif()

# --- OpenUSD's data ------------------------------------------------------------------
if(EXISTS "${ATHENEA_RT_USD}/lib/usd")
    file(INSTALL "${ATHENEA_RT_USD}/lib/usd" DESTINATION "${_bin}")
endif()
if(EXISTS "${ATHENEA_RT_USD}/plugin/usd")
    file(GLOB _usd_plugins LIST_DIRECTORIES true "${ATHENEA_RT_USD}/plugin/usd/*")
    # plugin/usd/plugInfo.json is the same Includes line as hdAthenea's.
    list(FILTER _usd_plugins EXCLUDE REGEX "/plugInfo\\.json$")
    if(_usd_plugins)
        file(INSTALL ${_usd_plugins} DESTINATION "${_prefix}/plugin/usd")
    endif()
endif()
if(EXISTS "${ATHENEA_RT_USD}/libraries")
    file(INSTALL "${ATHENEA_RT_USD}/libraries" DESTINATION "${_prefix}")
endif()

# --- the third parties' licence files ----------------------------------------------
set(_licences "${_prefix}/${ATHENEA_RT_DOCDIR}/third-party")
function(_rt_licence name)
    foreach(_file IN LISTS ARGN)
        if(EXISTS "${_file}")
            file(INSTALL "${_file}" DESTINATION "${_licences}/${name}")
        endif()
    endforeach()
endfunction()
_rt_licence(openusd "${ATHENEA_RT_USD}/LICENSE.txt" "${ATHENEA_RT_USD}/LICENSE" "${ATHENEA_RT_USD}/THIRD-PARTY.md"
            "${ATHENEA_RT_USD}/NOTICE.txt")
_rt_licence(slang "${ATHENEA_RT_SLANG}/LICENSE" "${ATHENEA_RT_SLANG}/LICENSES")
if(ATHENEA_RT_OCIO)
    _rt_licence(opencolorio "${ATHENEA_RT_OCIO}/share/doc/OpenColorIO/LICENSE" "${ATHENEA_RT_OCIO}/LICENSE")
endif()
if(ATHENEA_RT_OIDN)
    _rt_licence(oidn "${ATHENEA_RT_OIDN}/share/doc/OpenImageDenoise/LICENSE.txt"
                "${ATHENEA_RT_OIDN}/share/doc/OpenImageDenoise/third-party-programs.txt")
endif()

# --- the check ------------------------------------------------------------------------
if(NOT EXISTS "${ATHENEA_RT_DUMPBIN}")
    find_program(ATHENEA_RT_DUMPBIN dumpbin)
endif()
if(NOT ATHENEA_RT_DUMPBIN OR NOT EXISTS "${ATHENEA_RT_DUMPBIN}")
    message(WARNING "athenea install: no dumpbin, so the DLLs the programs import were not checked")
    return()
endif()
set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM "windows+pe")
set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL "dumpbin")
set(CMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND "${ATHENEA_RT_DUMPBIN}")

set(_executables "")
foreach(_name IN LISTS ATHENEA_RT_EXECUTABLES)
    list(APPEND _executables "${_bin}/${_name}")
endforeach()
set(_modules "")
foreach(_relative IN LISTS ATHENEA_RT_MODULES)
    list(APPEND _modules "${_prefix}/${_relative}")
endforeach()
file(GLOB_RECURSE _bundles "${_prefix}/${ATHENEA_RT_AOFX_DIR}/*.aofx")
list(APPEND _modules ${_bundles})

file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES ${_executables}
    MODULES ${_modules}
    RESOLVED_DEPENDENCIES_VAR _resolved
    UNRESOLVED_DEPENDENCIES_VAR _unresolved
    CONFLICTING_DEPENDENCIES_PREFIX _conflict
    DIRECTORIES "${_bin}"
    PRE_EXCLUDE_REGEXES "^api-ms-.*" "^ext-ms-.*"
    POST_EXCLUDE_REGEXES "^[A-Za-z]:[/\\\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\].*")
set(_outside "")
string(TOLOWER "${_bin}/" _bin_lower)
foreach(_dll IN LISTS _resolved)
    file(TO_CMAKE_PATH "${_dll}" _dll)
    string(TOLOWER "${_dll}" _lower)
    string(FIND "${_lower}" "${_bin_lower}" _at)
    if(NOT _at EQUAL 0)
        list(APPEND _outside "${_dll}")
    endif()
endforeach()
list(LENGTH _resolved _count)
message(STATUS "athenea install: ${_count} DLLs the binaries import resolve in bin/ or the system's folders")
if(_unresolved OR _outside)
    message(FATAL_ERROR "athenea install: DLLs not in the package -- unresolved: ${_unresolved}; "
                        "found outside bin/: ${_outside}")
endif()
