# Copyright (c) 2026 jesus luque.
# Gaussian splats as video planes: the plane kernels' host side, the .athv
# container and the platform's hardware encoder.
set(_video_codec src/CodecNone.cpp)
if(APPLE)
    set(_video_codec src/VideoToolbox.mm)
endif()
athenea_add_module(video
    SOURCES
        src/Planes.cpp
        src/Athv.cpp
        ${_video_codec}
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::scene)
if(APPLE)
    target_link_libraries(athenea_video PRIVATE z "-framework VideoToolbox" "-framework CoreMedia"
                          "-framework CoreVideo" "-framework CoreFoundation")
else()
    target_link_libraries(athenea_video PRIVATE ZLIB::ZLIB)
endif()
