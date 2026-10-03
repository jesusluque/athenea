# Colour: OpenColorIO as a compiler -- a config's colour spaces and views
# become generated Slang functions and the LUT textures they sample -- and the
# names a file, a material or a stage gives a colour space, resolved to one
# the config knows. Below material (texture input spaces) and technique (the
# display), which is why it is a module of its own.
athenea_add_module(colour
    SOURCES
        src/ColourCompiler.cpp
        src/ColourNames.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu)

# OpenColorIO (scripts/build-ocio.sh). Without it the names resolve by a short
# table and the one function compiled is sRGB to linear Rec.709, written here.
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
    target_link_libraries(athenea_colour PRIVATE OpenColorIO::OpenColorIO)
    target_compile_definitions(athenea_colour PRIVATE ATHENEA_HAVE_OCIO=1)
    message(STATUS "colour: OpenColorIO ${OpenColorIO_VERSION} (${_athenea_ocio_from})")
else()
    target_compile_definitions(athenea_colour PRIVATE ATHENEA_HAVE_OCIO=0)
    message(STATUS "colour: no OpenColorIO at ${ATHENEA_OCIO_ROOT}; colour spaces by table, no OCIO view")
endif()
