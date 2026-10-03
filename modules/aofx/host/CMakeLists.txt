# This engine's side of the AOFX host: what it declares to aofx::host (a
# logger, nothing else -- no media stack, no inference runtime), and
# `renderEffect`, which is what a compositor's engine does for an AOFX node
# without the graph around it.
athenea_add_module(aofx_host
    SOURCES
        src/Host.cpp
        src/EffectRender.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::image
        athenea::gpu_host
        aofx::host
        aofx::aofx
        gpe)
