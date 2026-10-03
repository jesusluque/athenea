# Techniques: how a frame is drawn from the scene -- visibility, shading, path
# tracing, denoising, display.
athenea_add_module(technique
    SOURCES
        src/Denoiser.cpp
        src/DisplayTransform.cpp
        src/Ocio.cpp
        src/Aces2.cpp
        src/MaterialPrograms.cpp
        src/MaterialShading.cpp
        src/EmissiveTable.cpp
        src/Environment.cpp
        src/PathTracer.cpp
        src/SplatShadowMap.cpp
        src/SplatShadows.cpp
        src/SplatVisibility.cpp
        src/Visibility.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::world
        athenea::material
        athenea::light
        athenea::render)

# OpenColorIO (scripts/build-ocio.sh), as a compiler of shader text and LUTs
# for the display. Without it ViewTransform::Ocio says so.
set(ATHENEA_OCIO_ROOT "$ENV{HOME}/tools/ocio-2.5.2" CACHE PATH "OpenColorIO install")
# A parent that found OpenColorIO already has the imported target: one OCIO in
# the process, and the parent's.
if(TARGET OpenColorIO::OpenColorIO)
    set(_athenea_ocio_from "the parent project's")
else()
    find_package(OpenColorIO 2.5 CONFIG QUIET HINTS "${ATHENEA_OCIO_ROOT}" NO_DEFAULT_PATH)
    set(_athenea_ocio_from "${ATHENEA_OCIO_ROOT}")
endif()
if(TARGET OpenColorIO::OpenColorIO)
    target_link_libraries(athenea_technique PRIVATE OpenColorIO::OpenColorIO)
    target_compile_definitions(athenea_technique PRIVATE ATHENEA_HAVE_OCIO=1)
    message(STATUS "technique: OpenColorIO ${OpenColorIO_VERSION} (${_athenea_ocio_from})")
else()
    target_compile_definitions(athenea_technique PRIVATE ATHENEA_HAVE_OCIO=0)
    message(STATUS "technique: no OpenColorIO at ${ATHENEA_OCIO_ROOT}; no OCIO view")
endif()

# OIDN, GPU devices only (scripts/build-oidn.sh). Without it the engine has no
# denoiser, and says so.
set(ATHENEA_OIDN_ROOT "$ENV{HOME}/tools/oidn-2.5.1" CACHE PATH "Open Image Denoise install (GPU devices only)")
find_package(OpenImageDenoise 2.5 CONFIG QUIET HINTS "${ATHENEA_OIDN_ROOT}" NO_DEFAULT_PATH)
if(OpenImageDenoise_FOUND)
    target_link_libraries(athenea_technique PRIVATE OpenImageDenoise)
    target_compile_definitions(athenea_technique PRIVATE ATHENEA_HAVE_OIDN=1)
    message(STATUS "technique: OIDN ${OpenImageDenoise_VERSION} (${ATHENEA_OIDN_ROOT})")
else()
    target_compile_definitions(athenea_technique PRIVATE ATHENEA_HAVE_OIDN=0)
    message(STATUS "technique: no OIDN at ${ATHENEA_OIDN_ROOT}; no denoiser")
endif()
