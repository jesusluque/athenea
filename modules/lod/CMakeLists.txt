# Copyright (c) 2026 jesus luque.
athenea_add_module(lod
    SOURCES
        src/Lod.cpp
        src/Athc.cpp
        src/Athz.cpp
        src/SplatStream.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::scene
        athenea::render)
# zlib's CRC-32 over a .athz page's bytes, as a loader reads them.
find_package(ZLIB REQUIRED)
if(APPLE)
    target_link_libraries(athenea_lod PRIVATE z)
else()
    target_link_libraries(athenea_lod PRIVATE ZLIB::ZLIB)
endif()
