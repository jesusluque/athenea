set(ATHENEA_USD_SOURCES
    src/HydraCamera.cpp
    src/Camera.cpp
    src/PrimData.cpp
    src/Engine.cpp
    src/RenderBuffer.cpp
    src/ParticleField.cpp
    src/Mesh.cpp
    src/Instancer.cpp
    src/Light.cpp
    src/Material.cpp
    src/Points.cpp
    src/Volume.cpp
    src/Curves.cpp
    src/RenderPass.cpp
    src/Resample.cpp
    src/SkelSplat.cpp
    src/RenderDelegate.cpp)
set(ATHENEA_USD_LIBS usd usdGeom usdVol usdShade usdLux usdRender usdHydra usdSkel usdImaging usdSkelImaging usdUtils
    hd hdx hdsi hdMtlx hio cameraUtil gf tf vt sdf plug athenea::gpu)
if(TARGET usd_m)
    # A monolithic OpenUSD -- what the iOS build is -- exports every library
    # above as an INTERFACE target carrying its include directories and its
    # dependencies, and puts the code in one usd_m. Named alone, they link
    # nothing.
    list(APPEND ATHENEA_USD_LIBS usd_m)
endif()

# The engine side of Hydra, linked into the CLI and tests.
athenea_add_module(usd
    SOURCES ${ATHENEA_USD_SOURCES} src/StageRenderer.cpp src/BindingPurposes.cpp src/Export.cpp src/Decimate.cpp src/MeshStage.cpp src/Migrate.cpp
    PUBLIC_LIBS athenea::ui athenea::render athenea::io athenea::scene athenea::lod athenea::geom athenea::world athenea::technique ${ATHENEA_USD_LIBS})
# USD's headers predate the warnings this engine asks for.
target_compile_options(athenea_usd PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-cast-align
                       -Wno-deprecated-declarations -Wno-double-promotion)
target_compile_definitions(athenea_usd PUBLIC PXR_STATIC=0 TBB_SUPPRESS_DEPRECATED_MESSAGES=1)
# Default visibility, unlike every other engine module. USD identifies value
# types by std::type_info (TfType), and a VtArray<GfVec3f> instantiated here
# with hidden visibility has its own type_info the USD libraries do not
# recognise: saving a stage failed with "Attempted to pack unsupported type
# 'GfVec3f'" on an array of GfVec3f.
set_target_properties(athenea_usd PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)

# The same, as a Hydra renderer plugin any USD application can load
# (usdview, Houdini, Nuke): PXR_PLUGINPATH_NAME=<build>/plugin/usd.
set(ATHENEA_HYDRA_PLUGIN_DIR "${ATHENEA_BUILD}/plugin/usd" CACHE INTERNAL "Where hdAthenea and its plugInfo land")
# RendererPlugin.cpp only here: it registers a TfType, and a process holding
# both the static library and the loaded plugin must not register it twice.
#
# Not on iOS: an app may not load a plugin bundle of its own, and it has no
# need to -- StageRenderer makes the delegate itself, and the only USD
# application on the phone is this one. The codeless schemas below are built
# all the same: they are data, and the stages name them.
if(NOT ATHENEA_IOS)
add_library(hdAthenea MODULE ${ATHENEA_USD_SOURCES} src/RendererPlugin.cpp)
target_include_directories(hdAthenea PRIVATE include src)
target_link_libraries(hdAthenea PRIVATE athenea::ui athenea::render athenea::io athenea::scene athenea::lod athenea::geom athenea::world athenea::technique
                      ${ATHENEA_USD_LIBS})
target_compile_options(hdAthenea PRIVATE -Wno-deprecated-declarations)
# The plugin's own directory first: a plugin shipped with its libraries
# beside it (libslang, in a package another program loads) finds them
# there, wherever the package was unpacked.
if(APPLE)
    set(ATHENEA_PLUGIN_ORIGIN "@loader_path")
else()
    set(ATHENEA_PLUGIN_ORIGIN "$ORIGIN")
endif()
set_target_properties(hdAthenea PROPERTIES
    PREFIX "" OUTPUT_NAME "hdAthenea"
    CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF
    LIBRARY_OUTPUT_DIRECTORY "${ATHENEA_HYDRA_PLUGIN_DIR}/hdAthenea"
    BUILD_RPATH "${ATHENEA_PLUGIN_ORIGIN};${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib"
    INSTALL_RPATH "${ATHENEA_PLUGIN_ORIGIN}")
file(MAKE_DIRECTORY "${ATHENEA_HYDRA_PLUGIN_DIR}/hdAthenea/resources")
if(ATHENEA_BLENDER_LIB)
    # Inside Blender: its libraries, which it has loaded already, through
    # the executable's own path to them -- and not the header build's
    # prefix, whose libtbb and libMaterialX* would be second copies.
    set_target_properties(hdAthenea PROPERTIES
        BUILD_RPATH "${ATHENEA_PLUGIN_ORIGIN};@executable_path/../Resources/lib;${SLANG_ROOT}/lib"
        INSTALL_RPATH "${ATHENEA_PLUGIN_ORIGIN};@executable_path/../Resources/lib")
endif()
configure_file(plugin/plugInfo.json.in "${ATHENEA_HYDRA_PLUGIN_DIR}/hdAthenea/resources/plugInfo.json" @ONLY)
# The plugin is what a package of it carries, so it carries what it owes.
foreach(legal IN LISTS ATHENEA_LEGAL_FILES)
    get_filename_component(legal_name "${legal}" NAME)
    configure_file("${legal}" "${ATHENEA_HYDRA_PLUGIN_DIR}/hdAthenea/resources/${legal_name}" COPYONLY)
endforeach()
else()
    # A tree configured for the desktop and then for the phone would leave
    # hdAthenea's plugInfo behind, and USD would go looking for a library that is
    # not there.
    file(REMOVE_RECURSE "${ATHENEA_HYDRA_PLUGIN_DIR}/hdAthenea")
endif()
file(WRITE "${ATHENEA_HYDRA_PLUGIN_DIR}/plugInfo.json" "{ \"Includes\": [ \"*/resources/\" ] }\n")

# The codeless schemas (AtheneaSplatEditAPI, AtheneaPointStyleAPI), beside hdAthenea so
# the same PXR_PLUGINPATH_NAME finds both.
file(MAKE_DIRECTORY "${ATHENEA_HYDRA_PLUGIN_DIR}/atheneaSchemas/resources")
configure_file(schemas/plugInfo.json "${ATHENEA_HYDRA_PLUGIN_DIR}/atheneaSchemas/resources/plugInfo.json" COPYONLY)
configure_file(schemas/generatedSchema.usda "${ATHENEA_HYDRA_PLUGIN_DIR}/atheneaSchemas/resources/generatedSchema.usda" COPYONLY)
