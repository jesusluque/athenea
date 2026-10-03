include(Catch)

# athenea_test(<name> SOURCES ... LIBS ... [LABELS ...])
#
# A Catch2 executable, registered with ctest case by case. GPU tests skip when
# the machine has no device; they never fall back to anything else. Every
# case carries one label: `display` for a binary that needs a window, `gpe`
# for one that needs gpe's device, `gpu` otherwise -- so a machine without
# a window or gpe excludes those by label (`ctest -LE display`) rather than
# reading their skips as passes. What a case skips for a capability the device lacks
# (rasterisation, ray queries) it says in its skip message, which
# scripts/remote-test.sh tabulates.
function(athenea_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;ENVIRONMENT_MODIFICATION;LABELS" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE Catch2::Catch2WithMain ${ARG_LIBS})
    athenea_set_target_defaults(${name})
    set_target_properties(${name} PROPERTIES BUILD_RPATH "${SLANG_ROOT}/lib")
    add_dependencies(${name} athenea_shaders_copy)
    # One label a binary: the most specific it needs (a window, gpe's
    # device), `gpu` otherwise -- a list of labels does not survive the
    # discovery script's quoting.
    set(label gpu)
    if(ARG_LABELS)
        list(GET ARG_LABELS 0 label)
    endif()
    set(properties ENVIRONMENT "ATHENEA_SHADER_DIR=${ATHENEA_SHADER_OUTPUT_DIR}" LABELS "${label}")
    if(ARG_ENVIRONMENT_MODIFICATION)
        list(APPEND properties ENVIRONMENT_MODIFICATION "${ARG_ENVIRONMENT_MODIFICATION}")
    endif()
    catch_discover_tests(${name}
        PROPERTIES ${properties}
        DISCOVERY_MODE PRE_TEST)
endfunction()

athenea_test(athenea_core_tests
    SOURCES core/test_hash.cpp core/test_platform.cpp
    LIBS athenea::core)

athenea_test(athenea_gpu_tests
    SOURCES gpu/test_algo.cpp gpu/test_textures.cpp gpu/test_uniforms.cpp gpu/test_paths.cpp
    LIBS athenea::gpu athenea::render)

athenea_test(athenea_scene_tests
    SOURCES scene/test_loading.cpp
    LIBS athenea::scene)
if(ATHENEA_HAVE_SPZ)
    # The SPZ test writes its fixtures with zlib.
    target_compile_definitions(athenea_scene_tests PRIVATE ATHENEA_TEST_SPZ=1)
    target_link_libraries(athenea_scene_tests PRIVATE $<IF:$<BOOL:${APPLE}>,z,ZLIB::ZLIB>)
endif()

athenea_test(athenea_render_tests
    SOURCES render/test_splat_render.cpp render/test_points.cpp render/test_ray_tracing.cpp
            render/test_splat_motion.cpp render/test_splat_lens.cpp
            $<$<BOOL:${ATHENEA_HAVE_SPZ}>:render/test_spz.cpp>
            $<$<BOOL:${ATHENEA_HAVE_WEBP}>:render/test_sog.cpp>
    LIBS athenea::render athenea::io athenea::light)
target_compile_definitions(athenea_render_tests PRIVATE ATHENEA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
if(ATHENEA_HAVE_SPZ)
    # Niantic's packer writes the fixture the SPZ reader is checked against.
    target_link_libraries(athenea_render_tests PRIVATE athenea_spz)
endif()

athenea_test(athenea_geom_tests
    SOURCES geom/test_mesh.cpp geom/test_skinning.cpp geom/test_curves.cpp geom/test_subdivision.cpp
    LIBS athenea::geom)

athenea_test(athenea_material_tests
    SOURCES material/test_texture_store.cpp material/test_lobes.cpp material/test_material_compiler.cpp
    LIBS athenea::material hio)
target_compile_options(athenea_material_tests PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-deprecated-declarations
                       -Wno-double-promotion -Wno-cast-align)
target_compile_definitions(athenea_material_tests PRIVATE ATHENEA_MATERIALX_ROOT="${ATHENEA_USD_ROOT}")
set_target_properties(athenea_material_tests PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF
                      BUILD_RPATH "${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib")

athenea_test(athenea_technique_tests
    SOURCES technique/test_visibility.cpp technique/test_display.cpp technique/test_materials.cpp
            technique/test_lights.cpp technique/test_splat_shadows.cpp
            technique/test_splat_visibility.cpp technique/test_shadow_map.cpp
            technique/test_environment.cpp
    LIBS athenea::technique athenea::scene hio)
target_compile_definitions(athenea_technique_tests PRIVATE ATHENEA_MATERIALX_ROOT="${ATHENEA_USD_ROOT}")

athenea_test(athenea_lod_tests
    SOURCES lod/test_lod.cpp
    LIBS athenea::lod)

athenea_test(athenea_volume_tests
    SOURCES volume/test_volume.cpp
    LIBS athenea::world athenea::io)
set_target_properties(athenea_volume_tests PROPERTIES BUILD_RPATH "${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib")

athenea_test(athenea_usd_tests
    SOURCES usd/test_usd.cpp usd/test_migrate.cpp
    LIBS athenea::usd)
add_dependencies(athenea_usd_tests hdAthenea)
target_compile_definitions(athenea_usd_tests PRIVATE ATHENEA_HYDRA_PLUGIN_DIR="${ATHENEA_HYDRA_PLUGIN_DIR}"
                                                 ATHENEA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_compile_options(athenea_usd_tests PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-deprecated-declarations
                       -Wno-double-promotion -Wno-cast-align)
set_target_properties(athenea_usd_tests PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF
                      BUILD_RPATH "${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib")

# Storm as the oracle for geometry: its own process, its own Metal device,
# single-sampled (OpenUSD reads the setting as its libraries load).
athenea_test(athenea_storm_oracle_tests
    SOURCES usd/test_storm_oracle.cpp
    LIBS athenea::usd hgi glf MaterialXCore MaterialXFormat
    ENVIRONMENT_MODIFICATION "HDX_MSAA_SAMPLE_COUNT=set:1")
add_dependencies(athenea_storm_oracle_tests hdAthenea)
target_compile_definitions(athenea_storm_oracle_tests PRIVATE ATHENEA_HYDRA_PLUGIN_DIR="${ATHENEA_HYDRA_PLUGIN_DIR}"
                           ATHENEA_USD_ROOT="${ATHENEA_USD_ROOT}")
target_compile_options(athenea_storm_oracle_tests PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-deprecated-declarations
                       -Wno-double-promotion -Wno-cast-align)
set_target_properties(athenea_storm_oracle_tests PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF
                      BUILD_RPATH "${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib")

athenea_test(athenea_mcp_tests
    SOURCES mcp/test_mcp.cpp
    LIBS athenea::mcp nlohmann_json::nlohmann_json)
target_compile_options(athenea_mcp_tests PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-deprecated-declarations
                       -Wno-double-promotion -Wno-cast-align)
set_target_properties(athenea_mcp_tests PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF
                      BUILD_RPATH "${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib")

# The plugin as a host loads it: UsdImagingGLEngine and HdAtheneaRendererPlugin
# by name, in an executable that does not link athenea::usd -- with the delegate's
# classes in the executable too, a template instantiated in both (make_shared
# of the render pass) binds to the executable's copy, and the pass then
# fails to recognise the plugin's own render buffers.
athenea_test(athenea_host_tests
    SOURCES usd/test_host.cpp
    LIBS athenea::render usd usdGeom usdImaging usdImagingGL hd hgi cameraUtil)
add_dependencies(athenea_host_tests hdAthenea)
target_compile_definitions(athenea_host_tests PRIVATE ATHENEA_HYDRA_PLUGIN_DIR="${ATHENEA_HYDRA_PLUGIN_DIR}")
target_include_directories(athenea_host_tests SYSTEM PRIVATE "${ATHENEA_USD_ROOT}/include")
target_link_directories(athenea_host_tests PRIVATE "${ATHENEA_USD_ROOT}/lib")
target_compile_options(athenea_host_tests PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-deprecated-declarations
                       -Wno-double-promotion -Wno-cast-align)
set_target_properties(athenea_host_tests PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF
                      BUILD_RPATH "${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib")

if(ATHENEA_BUILD_VIEW)
    athenea_test(athenea_view_tests
        SOURCES view/test_view.cpp
        LIBS athenea::view
        LABELS display)
    add_dependencies(athenea_view_tests hdAthenea)
    target_compile_options(athenea_view_tests PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-deprecated-declarations)
    set_target_properties(athenea_view_tests PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF
                          BUILD_RPATH "${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib")
endif()

athenea_test(athenea_gpu_host_tests
    SOURCES gpu_host/test_shared_device.cpp
    LIBS athenea::gpu_host
    LABELS gpe)

add_subdirectory(aofx/bundles)
athenea_test(athenea_aofx_tests
    SOURCES aofx/test_aofx.cpp aofx/test_mesh2splat.cpp
    LIBS athenea::aofx_host athenea::gpu
    LABELS gpe)
# The conversion's own bundle, checked through the host like any other.
add_dependencies(athenea_aofx_tests aofx_test_bundles athenea_aofx_mesh2splat)
target_compile_definitions(athenea_aofx_tests PRIVATE
    ATHENEA_AOFX_TEST_BUNDLES="${ATHENEA_AOFX_TEST_BUNDLES}"
    ATHENEA_AOFX_TEST_RUN_BUNDLES="${ATHENEA_AOFX_TEST_RUN_BUNDLES}"
    ATHENEA_AOFX_BUNDLES="${AOFX_BUNDLE_DIR}")
# The SDK headers against their recorded hashes: aofx compatibility is mandatory.
add_test(NAME aofx_sdk_manifest
    COMMAND ${CMAKE_COMMAND} -DSDK_DIR=${ATHENEA_AOFX_DIR}/sdk/include/aofx
            -DMANIFEST=${CMAKE_CURRENT_SOURCE_DIR}/aofx/sdk_manifest.txt
            -P ${CMAKE_CURRENT_SOURCE_DIR}/aofx/CheckSdkManifest.cmake)

athenea_test(athenea_sched_tests
    SOURCES sched/test_clock.cpp
    LIBS athenea::sched athenea::io)

# The density a mesh gets is the box it is measured over: two planes twenty
# times apart in size, converted with one grid over the pair and then each
# over its own box. The small one gets a handful of gaussians the first way
# and its share the second; the log lines say how many.
foreach(density per-model per-mesh)
    add_test(NAME mesh2splat_density_${density}
        COMMAND athenea mesh2splat ${CMAKE_CURRENT_SOURCE_DIR}/data/two_planes.usda --no-bake --no-camera
                --resolution 64 --density ${density} -o ${CMAKE_CURRENT_BINARY_DIR}/two_planes_${density}.usda)
    set_tests_properties(mesh2splat_density_${density} PROPERTIES
        ENVIRONMENT "ATHENEA_SHADER_DIR=${ATHENEA_SHADER_OUTPUT_DIR}"
        LABELS gpe)
endforeach()
set_tests_properties(mesh2splat_density_per-model PROPERTIES
    PASS_REGULAR_EXPRESSION "/World/Small -> [0-9][0-9]? splats of")
set_tests_properties(mesh2splat_density_per-mesh PROPERTIES
    PASS_REGULAR_EXPRESSION "/World/Small -> [1-9][0-9][0-9]+ splats of")

# One TBB in the process: OpenUSD's oneTBB, which OIDN (built without its CPU
# device) and everything else share.
add_test(NAME single_tbb COMMAND athenea info)
set_tests_properties(single_tbb PROPERTIES
    ENVIRONMENT "ATHENEA_SHADER_DIR=${ATHENEA_SHADER_OUTPUT_DIR}"
    PASS_REGULAR_EXPRESSION "tbb libraries  1 ")
