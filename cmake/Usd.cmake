# OpenUSD: where entities live. Built once per machine by scripts/build-usd.sh.
# pxrConfig finds its own dependencies (OpenSubdiv, TBB) through the prefix
# path -- first in it, so it finds the TBB it was built with and not the one a
# parent's prefix path (Homebrew's, say) lists before it: two TBBs in one
# process is the thing single_tbb exists to catch.
list(PREPEND CMAKE_PREFIX_PATH "${ATHENEA_USD_ROOT}")
if(NOT ATHENEA_IOS)
    # Imaging targets name OpenGL::GL in their link interface (garch) even
    # though nothing here draws with GL. iOS has no OpenGL, and the iOS build
    # of OpenUSD (an "embedded" target) names Metal instead.
    find_package(OpenGL REQUIRED)
endif()
find_package(pxr CONFIG REQUIRED HINTS "${ATHENEA_USD_ROOT}")
message(STATUS "OpenUSD: ${pxr_DIR}")
# MaterialX, which this OpenUSD is built with (scripts/build-usd.sh): the
# engine's materials are generated from MaterialX documents by its Slang
# generator.
if(NOT TARGET MaterialXCore)
    find_package(MaterialX 1.39.5 CONFIG REQUIRED HINTS "${ATHENEA_USD_ROOT}")
endif()
if(NOT TARGET MaterialXGenSlang)
    message(FATAL_ERROR "OpenUSD at ${ATHENEA_USD_ROOT} carries MaterialX without its Slang generator; "
                        "rebuild with scripts/build-usd.sh")
endif()
# Imported targets are directory-scoped; the engine's other modules and apps
# link MaterialX too.
foreach(_mx MaterialXCore MaterialXFormat MaterialXGenShader MaterialXGenHw MaterialXGenSlang hgi)
    if(TARGET ${_mx})
        # A monolithic MaterialX -- what the iOS build is -- exports each
        # library as an alias onto the one it merged them into. An alias
        # carries no properties of its own, and is global already.
        get_target_property(_aliased ${_mx} ALIASED_TARGET)
        if(_aliased)
            continue()
        endif()
        get_target_property(_global ${_mx} IMPORTED_GLOBAL)
        if(NOT _global)
            set_target_properties(${_mx} PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
    endif()
endforeach()

# OpenVDB and NanoVDB's headers come with the same prefix: what a volume is
# read from and laid out as. PNanoVDB.h travels with the shaders, since
# athenea/volume/nanovdb.slang includes it.
find_library(ATHENEA_OPENVDB_LIB openvdb HINTS "${ATHENEA_USD_ROOT}/lib" NO_DEFAULT_PATH)
if(ATHENEA_OPENVDB_LIB AND EXISTS "${ATHENEA_USD_ROOT}/include/nanovdb/PNanoVDB.h")
    set(ATHENEA_HAVE_OPENVDB ON)
    add_custom_command(OUTPUT "${ATHENEA_SHADER_OUTPUT_DIR}/nanovdb/PNanoVDB.h"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ATHENEA_USD_ROOT}/include/nanovdb/PNanoVDB.h"
                "${ATHENEA_SHADER_OUTPUT_DIR}/nanovdb/PNanoVDB.h"
        DEPENDS "${ATHENEA_USD_ROOT}/include/nanovdb/PNanoVDB.h" VERBATIM)
    add_custom_target(athenea_nanovdb_header ALL DEPENDS "${ATHENEA_SHADER_OUTPUT_DIR}/nanovdb/PNanoVDB.h")
    if(TARGET athenea_shaders_copy)
        add_dependencies(athenea_shaders_copy athenea_nanovdb_header)
    endif()
    message(STATUS "OpenVDB: ${ATHENEA_OPENVDB_LIB} (volumes read and laid out as NanoVDB)")
else()
    set(ATHENEA_HAVE_OPENVDB OFF)
    message(STATUS "OpenVDB: not found in ${ATHENEA_USD_ROOT}; .vdb is refused with a message")
endif()
