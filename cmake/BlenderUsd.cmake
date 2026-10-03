# Copyright (c) 2026 jesus luque.
#
# OpenUSD, oneTBB, MaterialX, OpenColorIO and OIDN as Blender has them: the
# headers from ~/tools/usd-<version>-blender (scripts/build-usd-blender.sh),
# the libraries from Blender's own Contents/Resources/lib. hdAthenea loaded
# by Blender's HydraRenderEngine then binds to the libraries Blender already
# loaded -- one USD, one TBB, one MaterialX, one OCIO, one OIDN in the
# process -- and is built for nothing else (no tests, no CLI: they would
# need Blender's libraries to run).
#
# Included by cmake/Usd.cmake in place of find_package(pxr) when
# ATHENEA_BLENDER_LIB is set (the macos-arm64-blender preset).

set(ATHENEA_BLENDER_LIB "${ATHENEA_BLENDER_LIB}" CACHE PATH "Blender's Contents/Resources/lib")
set(ATHENEA_MATERIALX_SLANG_SRC "$ENV{HOME}/tools/src/MaterialX-1.39.5"
    CACHE PATH "MaterialX 1.39.5 sources, for MaterialXGenSlang backported onto Blender's 1.39.4")
if(NOT EXISTS "${ATHENEA_BLENDER_LIB}/libusd_ms.dylib")
    message(FATAL_ERROR "Blender's libusd_ms.dylib is not in ${ATHENEA_BLENDER_LIB}")
endif()
if(NOT EXISTS "${ATHENEA_USD_ROOT}/include/pxr/pxr.h")
    message(FATAL_ERROR "No USD headers at ${ATHENEA_USD_ROOT}: run scripts/build-usd-blender.sh")
endif()

# A library target whose code is Blender's. The link records the file's own
# install name (@rpath/libX.dylib), which resolves inside Blender through
# hdAthenea's rpath to the image Blender loaded.
function(_athenea_blender_lib target file)
    if(TARGET ${target})
        set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${ATHENEA_BLENDER_LIB}/${file}")
        get_target_property(_configs ${target} IMPORTED_CONFIGURATIONS)
        if(_configs)
            foreach(_config ${_configs})
                set_target_properties(${target} PROPERTIES IMPORTED_LOCATION_${_config} "${ATHENEA_BLENDER_LIB}/${file}")
            endforeach()
        endif()
    else()
        add_library(${target} SHARED IMPORTED GLOBAL)
        set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${ATHENEA_BLENDER_LIB}/${file}")
    endif()
endfunction()

# oneTBB 2022.3: Blender's libtbb under its headers.
_athenea_blender_lib(athenea_blender_tbb libtbb.dylib)
set_target_properties(athenea_blender_tbb PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ATHENEA_USD_ROOT}/include")
add_library(TBB::tbb ALIAS athenea_blender_tbb)

# Python's headers: VtValue and TfAnyWeakPtr are laid out for a USD built
# with Python, and the headers say so only by including Python.h.
file(GLOB _python_include LIST_DIRECTORIES true "$ENV{HOME}/.local/share/uv/python/cpython-3.13*/include/python3.13")
list(GET _python_include 0 _python_include)
set(ATHENEA_BLENDER_PYTHON_INCLUDE "${_python_include}" CACHE PATH "Python 3.13 headers (uv python install 3.13)")

# OpenUSD: one monolithic library, every per-library target a name for it.
_athenea_blender_lib(athenea_blender_usd libusd_ms.dylib)
set_target_properties(athenea_blender_usd PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ATHENEA_USD_ROOT}/include;${ATHENEA_BLENDER_PYTHON_INCLUDE}"
    INTERFACE_LINK_LIBRARIES "athenea_blender_tbb"
    # Python's symbols are the Blender executable's (its Python is linked
    # statically), so USD's headers that name them resolve at load time.
    INTERFACE_LINK_OPTIONS "LINKER:-undefined,dynamic_lookup")
foreach(_pxr arch tf gf pegtl js trace work plug vt ts ar kind sdf sdr pcp usd usdGeom usdVol usdMedia usdShade
        usdLod usdLux usdProc usdRender usdHydra usdRi usdSemantics usdSkel usdUI usdUtils usdPhysics usdMtlx
        garch hf hio cameraUtil pxOsd geomUtil glf hgi hgiGL hgiMetal hgiInterop hd hdar hdGp hdsi hdMtlx hdSt hdx
        usdImaging usdImagingGL usdProcImaging usdSkelImaging usdVolImaging usdAppUtils)
    add_library(${_pxr} INTERFACE IMPORTED GLOBAL)
    set_target_properties(${_pxr} PROPERTIES INTERFACE_LINK_LIBRARIES athenea_blender_usd)
endforeach()
set(pxr_DIR "${ATHENEA_USD_ROOT} (headers) + ${ATHENEA_BLENDER_LIB}/libusd_ms.dylib")

# MaterialX 1.39.4: the configs from the header build, the libraries
# Blender's.
find_package(MaterialX 1.39.4 EXACT CONFIG REQUIRED HINTS "${ATHENEA_USD_ROOT}" NO_DEFAULT_PATH)
foreach(_mx MaterialXCore MaterialXFormat MaterialXGenShader MaterialXGenGlsl MaterialXGenMsl MaterialXRender)
    _athenea_blender_lib(${_mx} lib${_mx}.dylib)
    set_target_properties(${_mx} PROPERTIES IMPORTED_GLOBAL TRUE)
endforeach()

# The Slang generator Blender's MaterialX does not have: 1.39.5's
# MaterialXGenSlang and the hardware nodes 1.39.5 moved into MaterialXGenHw,
# compiled into MaterialX_v1_39_4 over integrations/blender/materialx's
# headers. Two lines of 1.39.5 name API 1.39.4 lacks: a virtual
# requiresLighting (ours is called only by the Slang generator itself) and
# GenOptions::hwAiryFresnelIterations (1.39.5's default, 2).
set(_mx_src "${ATHENEA_MATERIALX_SLANG_SRC}/source")
if(NOT EXISTS "${_mx_src}/MaterialXGenSlang/SlangShaderGenerator.cpp")
    message(FATAL_ERROR "MaterialX 1.39.5 sources not at ${ATHENEA_MATERIALX_SLANG_SRC} (scripts/build-usd-blender.sh)")
endif()
set(_mx_out "${CMAKE_BINARY_DIR}/materialx-slang")
set(_mx_compat "${ATHENEA_ROOT}/integrations/blender/materialx")
file(MAKE_DIRECTORY "${_mx_out}/include/MaterialXGenSlang" "${_mx_out}/include/MaterialXGenHw/Nodes" "${_mx_out}/src")
foreach(_file SlangShaderGenerator.h SlangSyntax.h Export.h)
    file(READ "${_mx_src}/MaterialXGenSlang/${_file}" _text)
    string(REPLACE "requiresLighting(const ShaderGraph& graph) const override;"
                   "requiresLighting(const ShaderGraph& graph) const;" _text "${_text}")
    file(WRITE "${_mx_out}/include/MaterialXGenSlang/${_file}" "${_text}")
endforeach()
foreach(_file SlangShaderGenerator.cpp SlangSyntax.cpp)
    file(READ "${_mx_src}/MaterialXGenSlang/${_file}" _text)
    string(REPLACE "context.getOptions().hwAiryFresnelIterations" "2u" _text "${_text}")
    # The token 1.39.5's HwShaderGenerator substitutes and 1.39.4's does not
    # (pbrlib/genglsl/lib/mx_closure_type.glsl returns it).
    string(REPLACE "HwShaderGenerator(typeSystem, SlangSyntax::create(typeSystem))\n{\n"
                   "HwShaderGenerator(typeSystem, SlangSyntax::create(typeSystem))\n{\n    _tokenSubstitutions[\"$closureDataConstructor\"] = \"ClosureData(closureType, L, V, N, P, occlusion)\";\n"
                   _text "${_text}")
    file(WRITE "${_mx_out}/src/${_file}" "${_text}")
endforeach()
set(_mx_sources "${_mx_out}/src/SlangShaderGenerator.cpp" "${_mx_out}/src/SlangSyntax.cpp" "${_mx_compat}/src/HwCompat.cpp")
foreach(_node LightCompound Light LightSampler LightShader NumLights Surface MaterialCompound)
    configure_file("${_mx_src}/MaterialXGenHw/Nodes/Hw${_node}Node.h" "${_mx_out}/include/MaterialXGenHw/Nodes/Hw${_node}Node.h" COPYONLY)
    list(APPEND _mx_sources "${_mx_src}/MaterialXGenHw/Nodes/Hw${_node}Node.cpp")
endforeach()
add_library(athenea_materialx_slang STATIC ${_mx_sources})
target_include_directories(athenea_materialx_slang PUBLIC "${_mx_out}/include" "${_mx_compat}/include")
target_compile_definitions(athenea_materialx_slang PRIVATE MATERIALX_GENSLANG_EXPORTS)
target_link_libraries(athenea_materialx_slang PUBLIC MaterialXGenShader MaterialXFormat MaterialXCore)
set_target_properties(athenea_materialx_slang PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET default)
target_compile_options(athenea_materialx_slang PRIVATE -w)
add_library(MaterialXGenSlang ALIAS athenea_materialx_slang)
add_library(MaterialXGenHw INTERFACE IMPORTED GLOBAL)
set_target_properties(MaterialXGenHw PROPERTIES INTERFACE_LINK_LIBRARIES athenea_materialx_slang)

# OpenColorIO 2.5 and OIDN 2.5: the headers of the engine's own builds (2.5.2
# and 2.5.1, the same ABI), the libraries Blender's (2.5.0 both).
set(ATHENEA_OCIO_ROOT "$ENV{HOME}/tools/ocio-2.5.2" CACHE PATH "OpenColorIO install")
find_package(OpenColorIO 2.5 CONFIG QUIET HINTS "${ATHENEA_OCIO_ROOT}" NO_DEFAULT_PATH)
if(TARGET OpenColorIO::OpenColorIO)
    _athenea_blender_lib(OpenColorIO::OpenColorIO libOpenColorIO.dylib)
    set_target_properties(OpenColorIO::OpenColorIO PROPERTIES IMPORTED_GLOBAL TRUE)
endif()
find_package(OpenImageDenoise 2.5 CONFIG QUIET HINTS "${ATHENEA_OIDN_ROOT}" NO_DEFAULT_PATH)
if(TARGET OpenImageDenoise)
    _athenea_blender_lib(OpenImageDenoise libOpenImageDenoise.dylib)
    _athenea_blender_lib(OpenImageDenoise_core libOpenImageDenoise_core.dylib)
endif()

message(STATUS "OpenUSD: Blender's (${ATHENEA_BLENDER_LIB}), headers ${ATHENEA_USD_ROOT}")

# MaterialX 1.39.5's libraries beside the plugin, for the material compiler
# alone ($ATHENEA_MATERIALX_ROOT, which the add-on sets): Blender's 1.39.4
# libraries have no genslang implementations and older node definitions
# than the engine's closures. Blender's own Storm keeps its libraries.
set(ATHENEA_BLENDER_MTLX_DIR "${ATHENEA_BUILD}/plugin/materialx" CACHE INTERNAL "MaterialX 1.39.5 libraries for hdAthenea in Blender")
file(COPY "${ATHENEA_MATERIALX_SLANG_SRC}/libraries" DESTINATION "${ATHENEA_BLENDER_MTLX_DIR}")
