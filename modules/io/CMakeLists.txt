# The TBB USD was built with, so the process has one -- unless a parent found
# one already, in which case that is the one the process has and this module
# takes it (the parent then owes the same TBB to any USD it links).
if(NOT TARGET TBB::tbb)
    find_package(TBB CONFIG REQUIRED HINTS "${ATHENEA_USD_ROOT}")
endif()
athenea_add_module(io
    SOURCES
        src/PlyHeader.cpp
        src/SplatReaders.cpp
        src/PointReaders.cpp
        src/Exr.cpp
        src/Png.cpp
        src/Sog.cpp
        src/Ies.cpp
        src/SplatWriters.cpp
    PUBLIC_LIBS
        athenea::core
    PRIVATE_LIBS
        TBB::tbb
        athenea_tinyexr
        nlohmann_json::nlohmann_json)
if(ATHENEA_HAVE_SPZ)
    target_link_libraries(athenea_io PRIVATE athenea_spz)
    target_compile_definitions(athenea_io PRIVATE ATHENEA_HAVE_SPZ=1)
endif()

# SPZ version 4 is written with ZSTD where third_party/spz found it (a
# parent that brings its own spz may not say where: then version 3 alone).
if(ATHENEA_ZSTD_INCLUDE AND ATHENEA_ZSTD_LIB)
    target_include_directories(athenea_io SYSTEM PRIVATE ${ATHENEA_ZSTD_INCLUDE})
    target_link_libraries(athenea_io PRIVATE ${ATHENEA_ZSTD_LIB})
    target_compile_definitions(athenea_io PRIVATE ATHENEA_HAVE_ZSTD=1)
endif()

# .vdb: OpenVDB reads it and NanoVDB lays it out (cmake/Usd.cmake finds both).
target_sources(athenea_io PRIVATE src/Vdb.cpp)
if(ATHENEA_HAVE_OPENVDB)
    target_link_libraries(athenea_io PRIVATE ${ATHENEA_OPENVDB_LIB})
    target_include_directories(athenea_io SYSTEM PRIVATE "${ATHENEA_USD_ROOT}/include")
    target_compile_definitions(athenea_io PRIVATE ATHENEA_HAVE_OPENVDB=1)
    # OpenVDB's headers predate the warnings this engine asks for.
    set_source_files_properties(src/Vdb.cpp PROPERTIES COMPILE_OPTIONS
        "-Wno-old-style-cast;-Wno-shadow;-Wno-deprecated-declarations;-Wno-double-promotion;-Wno-cast-align;-Wno-sign-conversion;-Wno-conversion;-Wno-unused-parameter;-Wno-implicit-int-conversion;-Wno-implicit-float-conversion;-Wno-float-conversion;-Wno-shorten-64-to-32")
endif()

# SOG: zlib for the bundle, libwebp for its images. Without libwebp .sog is
# refused with a message.
find_package(ZLIB REQUIRED)
if(APPLE)
    target_link_libraries(athenea_io PRIVATE z)   # see third_party/spz: not ZLIB::ZLIB on Apple
else()
    target_link_libraries(athenea_io PRIVATE ZLIB::ZLIB)
endif()
find_path(ATHENEA_WEBP_INCLUDE webp/decode.h HINTS /opt/homebrew/opt/webp/include /usr/local/opt/webp/include)
find_library(ATHENEA_WEBP_LIB webp HINTS /opt/homebrew/opt/webp/lib /usr/local/opt/webp/lib)
if(ATHENEA_WEBP_INCLUDE AND ATHENEA_WEBP_LIB)
    target_include_directories(athenea_io SYSTEM PRIVATE ${ATHENEA_WEBP_INCLUDE})
    target_link_libraries(athenea_io PRIVATE ${ATHENEA_WEBP_LIB})
    target_compile_definitions(athenea_io PRIVATE ATHENEA_HAVE_WEBP=1)
    set(ATHENEA_HAVE_WEBP TRUE CACHE INTERNAL "")
    message(STATUS "io: reading .sog (libwebp ${ATHENEA_WEBP_LIB})")
else()
    set(ATHENEA_HAVE_WEBP FALSE CACHE INTERNAL "")
    message(STATUS "io: .sog disabled -- libwebp not found")
endif()
