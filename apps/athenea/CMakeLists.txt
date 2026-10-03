add_executable(athenea
    src/main.cpp
    src/CmdInfo.cpp
    src/CmdCompare.cpp
    src/CmdRender.cpp
    src/CmdStage.cpp
    src/CmdDecimate.cpp
    src/CmdAofx.cpp
    src/CmdMesh2Splat.cpp
    src/CmdVisibility.cpp
    src/CmdLive.cpp)
target_link_libraries(athenea PRIVATE athenea::usd athenea::lod athenea::sched athenea::technique athenea::aofx_host CLI11::CLI11)
if(ATHENEA_BUILD_VIEW)
    target_sources(athenea PRIVATE src/CmdView.cpp)
    target_link_libraries(athenea PRIVATE athenea::view)
    target_compile_definitions(athenea PRIVATE ATHENEA_HAVE_VIEW=1)
else()
    target_compile_definitions(athenea PRIVATE ATHENEA_HAVE_VIEW=0)
endif()
if(TARGET MaterialXCore)
    target_link_libraries(athenea PRIVATE MaterialXCore)
    target_compile_definitions(athenea PRIVATE ATHENEA_HAVE_MATERIALX=1)
endif()
# Bundles built in this tree are found without a --path.
target_compile_definitions(athenea PRIVATE ATHENEA_AOFX_BUNDLE_DIR="${AOFX_BUNDLE_DIR}")
if(TARGET athenea_aofx_invert)
    add_dependencies(athenea athenea_aofx_invert athenea_aofx_mesh2splat)
endif()
target_include_directories(athenea PRIVATE src)
athenea_set_target_defaults(athenea)
set_target_properties(athenea PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
athenea_copy_shaders(athenea)

# slang-rhi loads libslang-compiler at run time from beside the binary on some
# platforms and through the rpath on others; both are pointed at SLANG_ROOT.
set_target_properties(athenea PROPERTIES
    BUILD_RPATH "${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib"
    INSTALL_RPATH "@loader_path/../lib;${SLANG_ROOT}/lib")
