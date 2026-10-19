# Copyright (c) 2026 jesus luque.
athenea_add_module(scene
    SOURCES
        src/GpuClouds.cpp
        src/SplatSkinner.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::io)
