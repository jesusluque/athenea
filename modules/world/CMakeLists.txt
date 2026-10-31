# Copyright (c) 2026 jesus luque.
# The scene on the device: pools of mesh data and per-instance records, read
# by every technique.
athenea_add_module(world
    SOURCES
        src/GpuScene.cpp
        src/Instancing.cpp
        src/RayTracingScene.cpp
        src/BvhScene.cpp
        src/VolumeSet.cpp
        src/VolumeSplats.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::io
        athenea::geom
        athenea::render)
