add_executable(athenea
    src/main.cpp
    src/CmdInfo.cpp
    src/CmdCompare.cpp
    src/CmdRender.cpp
    src/CmdStage.cpp
    src/CmdDecimate.cpp
    src/CmdMigrate.cpp
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
# Bundles built in this tree, or installed beside it, are found without a
# --path: <exe>/../${AOFX_BUNDLE_INSTALL_DIR} first, else the build's own.
target_compile_definitions(athenea PRIVATE ATHENEA_AOFX_BUNDLE_DIR="${AOFX_BUNDLE_DIR}"
                           ATHENEA_AOFX_INSTALL_DIR="${AOFX_BUNDLE_INSTALL_DIR}")
# The one version, the project's (the top-level project() line): what
# `athenea --version` and `athenea info` print and what a release is tagged.
target_compile_definitions(athenea PRIVATE ATHENEA_VERSION="${athenea_VERSION}")
if(TARGET athenea_aofx_invert)
    add_dependencies(athenea athenea_aofx_invert athenea_aofx_mesh2splat athenea_aofx_measure
                     athenea_aofx_splatbakefilter)
endif()
target_include_directories(athenea PRIVATE src)
athenea_set_target_defaults(athenea)
set_target_properties(athenea PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
athenea_copy_shaders(athenea)

# slang-rhi loads libslang-compiler at run time from beside the binary on some
# platforms and through the rpath on others; both are pointed at SLANG_ROOT.
# Installed, the prefix's lib/ comes first (a package that carries the
# libraries), then SLANG_ROOT, then -- CMAKE_INSTALL_RPATH_USE_LINK_PATH --
# the toolchain directories it was linked against (USD, OIDN, OCIO). The
# build rpath starts the same way: on macOS the install rewrites only the
# entries that differ, and appends what is new at the end.
set_target_properties(athenea PROPERTIES
    BUILD_RPATH "${ATHENEA_ORIGIN}/../${CMAKE_INSTALL_LIBDIR};${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib"
    INSTALL_RPATH "${ATHENEA_ORIGIN}/../${CMAKE_INSTALL_LIBDIR};${SLANG_ROOT}/lib")
install(TARGETS athenea RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
