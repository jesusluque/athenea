# Niantic's SPZ reference implementation (MIT, LICENSE beside it), vendored as
# openFXplayer vendors it: seven files copied from its third_party/spz rather
# than fetched, with extensions compiled out -- the engine reads gaussians, not
# metadata records.
#
# Used only to decompress: loadSpzPacked hands back the quantised bytes, and
# the dequantising happens on the GPU (shaders/athenea/scene/splat_encoding.slang).
#
# Needs zlib (SPZ v1-3) and zstd (v4). Without zstd the target is not defined
# and .spz is refused with a message; .ply and .splat are unaffected.
find_path(ATHENEA_ZSTD_INCLUDE zstd.h HINTS /opt/homebrew/opt/zstd/include /usr/local/opt/zstd/include)
find_library(ATHENEA_ZSTD_LIB zstd HINTS /opt/homebrew/opt/zstd/lib /usr/local/opt/zstd/lib)
find_package(ZLIB)
if(ATHENEA_ZSTD_INCLUDE AND ATHENEA_ZSTD_LIB AND ZLIB_FOUND)
    add_library(athenea_spz STATIC load-spz.cc splat-types.cc)
    target_include_directories(athenea_spz SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(athenea_spz PRIVATE ${ATHENEA_ZSTD_INCLUDE})
    # zlib by name on Apple: ZLIB::ZLIB carries the SDK's /usr/include as
    # -isystem, which puts the C headers ahead of libc++'s (openFXplayer's
    # finding). Elsewhere the imported target is fine.
    if(APPLE)
        target_link_libraries(athenea_spz PRIVATE ${ATHENEA_ZSTD_LIB} z)
    else()
        target_link_libraries(athenea_spz PRIVATE ${ATHENEA_ZSTD_LIB} ZLIB::ZLIB)
    endif()
    set_target_properties(athenea_spz PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Vendored code is compiled as shipped: its warnings are not our signal.
    target_compile_options(athenea_spz PRIVATE -w)
    set(ATHENEA_HAVE_SPZ TRUE CACHE INTERNAL "")
    message(STATUS "spz: reading .spz (zstd ${ATHENEA_ZSTD_LIB})")
else()
    set(ATHENEA_HAVE_SPZ FALSE CACHE INTERNAL "")
    message(STATUS "spz: .spz disabled -- zstd or zlib not found")
endif()
