# Order is dependency order: each module may link only the ones above it.

add_subdirectory(core)
# The viewer's panels, as a description rather than as drawing code: `athenea view`
# and the iOS app both walk it (modules/ui/include/athenea/ui/Controls.h).
add_subdirectory(ui)
if(NOT ATHENEA_IOS)
    # sched is genlock (PTP), gpu_host is gpe: neither is built for the phone.
    add_subdirectory(sched)
endif()
add_subdirectory(image)
add_subdirectory(io)
add_subdirectory(gpu)
if(NOT ATHENEA_IOS)
    add_subdirectory(gpu_host)
endif()
add_subdirectory(scene)
add_subdirectory(render)
add_subdirectory(geom)
add_subdirectory(material)
add_subdirectory(light)
add_subdirectory(world)
add_subdirectory(technique)
add_subdirectory(lod)
add_subdirectory(usd)
if(NOT ATHENEA_IOS)
    # mcp is a stdio server, and the AOFX host loads plugin bundles -- which an
    # app may not do. The SDK is headers (aofx::aofx, from aopenfx), and
    # athenea::render takes its transforms from them; the top-level CMakeLists adds
    # it on every platform.
    add_subdirectory(mcp)
    add_subdirectory(aofx)
endif()
if(ATHENEA_BUILD_VIEW)
    add_subdirectory(view)
endif()

# What an embedder links: the renderer whole -- device, clouds, the two
# renderers, lights, LOD, techniques, files and stages -- without the viewer,
# the MCP server or the host's own plugins. No install rules and no shared
# library, on purpose: an embedder builds this tree inside its own.
add_library(athenea_engine INTERFACE)
add_library(athenea::engine ALIAS athenea_engine)
# Without athenea::usd, deliberately: an embedder that draws from its own scene
# description needs no stage, and USD brings its own TBB into a process that
# may already carry another (a compositor's OpenImageIO does). The day an
# embedder wants stages it links athenea::usd beside this, and its USD must then
# be built against the TBB the rest of the process has.
target_link_libraries(athenea_engine INTERFACE
    athenea::gpu $<TARGET_NAME_IF_EXISTS:athenea::gpu_host> athenea::scene athenea::render athenea::light athenea::lod
    athenea::technique athenea::io)
