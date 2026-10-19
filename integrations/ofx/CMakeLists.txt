# Copyright (c) 2026 jesus luque.
#
# Athenea.ofx.bundle: athenea's gaussian rasteriser as an OpenFX generator for
# a compositor (Nuke). The plugin links nothing of the engine -- only the
# OpenFX support library (Natron's) and a JSON parser -- and drives athenea-mcp in a
# process of its own, which the bundle carries in Contents/Resources/bin.

set(ATHENEA_OPENFX_DIR "${ATHENEA_ROOT}/third_party/openfx" CACHE PATH "OpenFX (headers and support library)")
if(NOT EXISTS "${ATHENEA_OPENFX_DIR}/include/ofxImageEffect.h")
    message(WARNING "ofx: no OpenFX at ${ATHENEA_OPENFX_DIR}; git submodule update --init third_party/openfx")
    return()
endif()

file(GLOB _ofx_support_sources "${ATHENEA_OPENFX_DIR}/Support/Library/*.cpp")
if(NOT WIN32)
    list(FILTER _ofx_support_sources EXCLUDE REGEX "ofxsHWNDInteract")
endif()
add_library(athenea_ofx_support STATIC ${_ofx_support_sources})
target_include_directories(athenea_ofx_support PUBLIC "${ATHENEA_OPENFX_DIR}/include"
                                                     "${ATHENEA_OPENFX_DIR}/Support/include")
# Natron's OpenFX (the one openfx-misc and openfx-io build on): its support
# library carries the multi-plane extension -- Foundry's planes for Nuke,
# Natron's custom ones -- which is how a node hands a compositor its layers.
target_compile_definitions(athenea_ofx_support PUBLIC OFX_EXTENSIONS_NUKE OFX_EXTENSIONS_NATRON)
set_target_properties(athenea_ofx_support PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
# The support library is OpenFX's, not this project's: its warnings are not ours to fix.
target_compile_options(athenea_ofx_support PRIVATE -w)

add_library(athenea_ofx MODULE src/AtheneaOfx.cpp src/RenderServer.cpp)
target_link_libraries(athenea_ofx PRIVATE athenea_ofx_support nlohmann_json::nlohmann_json)
target_compile_features(athenea_ofx PRIVATE cxx_std_20)
# Only OfxGetNumberOfPlugins and OfxGetPlugin are the host's business.
set_target_properties(athenea_ofx PROPERTIES
    OUTPUT_NAME "Athenea"
    PREFIX ""
    SUFFIX ".ofx"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# THE BUNDLE, as OpenFX lays one out: <name>.ofx.bundle/Contents/<arch>/<name>.ofx.
if(APPLE)
    set(_ofx_arch "MacOS")
elseif(WIN32)
    set(_ofx_arch "Win64")
else()
    set(_ofx_arch "Linux-x86-64")
endif()
set(ATHENEA_OFX_BUNDLE "${ATHENEA_BUILD}/ofx/Athenea.ofx.bundle" CACHE INTERNAL "")
set_target_properties(athenea_ofx PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${ATHENEA_OFX_BUNDLE}/Contents/${_ofx_arch}")
if(APPLE)
    configure_file(Info.plist.in "${ATHENEA_OFX_BUNDLE}/Contents/Info.plist" @ONLY)
endif()
# The server beside the plugin, so a bundle copied into a host's plugin
# directory works as it is.
if(TARGET athenea-mcp)
    add_dependencies(athenea_ofx athenea-mcp)
    add_custom_command(TARGET athenea_ofx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${ATHENEA_OFX_BUNDLE}/Contents/Resources/bin"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:athenea-mcp>
                "${ATHENEA_OFX_BUNDLE}/Contents/Resources/bin/athenea-mcp"
        VERBATIM)
endif()
