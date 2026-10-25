# Copyright (c) 2026 jesus luque.
# Frames and pages shared with another process: the frame ring a subscriber
# reads (render_subscribe), cameras with their instants (camera_set), and live
# pages a producer's device wrote (stream_open's devicePages).
athenea_add_module(bridge
    SOURCES
        src/Frames.cpp
        src/CameraQueue.cpp
        src/ClientPages.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::scene
        athenea::render
        athenea::technique
        athenea::lod)
target_link_libraries(athenea_bridge PRIVATE nlohmann_json::nlohmann_json)
