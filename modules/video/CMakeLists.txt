# Copyright (c) 2026 jesus luque.
# Gaussian splats as video planes: the plane kernels' host side, the .athv
# container, the platform's hardware codec and the live producer of a video.
# Apple: VideoToolbox both ways. Linux and Windows: NVENC and NVDEC (the
# driver's libraries opened at run time, so nothing is linked). Anywhere
# else: no hardware codec (every factory Unsupported).
if(APPLE)
    set(_video_codec src/VideoToolbox.mm)
elseif(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_video_codec src/Cuda.cpp src/Nvenc.cpp src/Nvdec.cpp)
else()
    set(_video_codec src/CodecNone.cpp)
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
    # nvEncodeAPI.h, dynlink_cuviddec.h, dynlink_nvcuvid.h and dynlink_cuda.h
    # (MIT, FFmpeg's nv-codec-headers).
    target_include_directories(athenea_video SYSTEM PRIVATE
                               "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/nv-codec-headers/include")
endif()
