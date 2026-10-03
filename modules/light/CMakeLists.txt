# Lights: what a scene emits, and how a shading point samples it.
athenea_add_module(light
    SOURCES
        src/LightTable.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::render
        athenea::io)
