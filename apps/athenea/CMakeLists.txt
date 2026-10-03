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
    src/CmdLive.cpp
    src/Output.cpp)
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
set_target_properties(athenea PROPERTIES
    BUILD_RPATH "${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib"
    INSTALL_RPATH "@loader_path/../lib;${SLANG_ROOT}/lib")

# `athenea mesh2splat` inside hdAthenea, as a C entry point (src/Embedded.cpp),
# for a host that cannot load a second USD and would convert with the one it
# has: Blender's add-on calls it through ctypes. On for the Blender build.
if(ATHENEA_BLENDER_LIB)
    set(_hydra_commands_default ON)
else()
    set(_hydra_commands_default OFF)
endif()
option(ATHENEA_HYDRA_COMMANDS "athenea mesh2splat built into hdAthenea, called by a host in process"
       ${_hydra_commands_default})
if(ATHENEA_HYDRA_COMMANDS AND TARGET hdAthenea)
    # The stage reading and writing the command uses, which athenea_usd has
    # beside the delegate's sources and hdAthenea compiles itself.
    set(_usd_src "${ATHENEA_ROOT}/modules/usd/src")
    target_sources(hdAthenea PRIVATE
        src/Embedded.cpp src/Output.cpp src/CmdMesh2Splat.cpp
        ${_usd_src}/StageRenderer.cpp ${_usd_src}/BindingPurposes.cpp ${_usd_src}/Export.cpp
        ${_usd_src}/MeshStage.cpp)
    target_include_directories(hdAthenea PRIVATE src)
    target_link_libraries(hdAthenea PRIVATE athenea::aofx_host CLI11::CLI11)
    target_compile_definitions(hdAthenea PRIVATE ATHENEA_AOFX_BUNDLE_DIR="${AOFX_BUNDLE_DIR}")
    if(TARGET athenea_aofx_mesh2splat)
        add_dependencies(hdAthenea athenea_aofx_mesh2splat athenea_aofx_splatbakefilter)
    endif()
endif()
