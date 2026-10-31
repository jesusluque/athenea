# Copyright (c) 2026 jesus luque.
# Gaussian splats as video planes: the plane kernels' host side, the .athv
# container, the platform's hardware codec and the live producer of a video.
set(_video_codec src/CodecNone.cpp)
if(APPLE)
    set(_video_codec src/VideoToolbox.mm)
endif()
athenea_add_module(video
    SOURCES
        src/Planes.cpp
        src/Athv.cpp
        src/VideoStream.cpp
        ${_video_codec}
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::scene
        athenea::lod)
if(APPLE)
    target_link_libraries(athenea_video PRIVATE z "-framework VideoToolbox" "-framework CoreMedia"
                          "-framework CoreVideo" "-framework Metal" "-framework CoreFoundation")
else()
    target_link_libraries(athenea_video PRIVATE ZLIB::ZLIB)
endif()
