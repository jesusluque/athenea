# Copyright (c) 2026 jesus luque.
athenea_add_module(lod
    SOURCES
        src/Lod.cpp
        src/Athc.cpp
        src/SplatStream.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::scene
        athenea::render)
