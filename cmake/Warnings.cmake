# Strict warnings on the engine's own targets, and only on those: third-party
# code (slang-rhi, OpenUSD headers, gpe) keeps whatever its authors chose.

function(athenea_set_target_defaults target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
            -Wold-style-cast -Wcast-align -Woverloaded-virtual
            -Wnull-dereference -Wdouble-promotion -Wimplicit-fallthrough)
    endif()
    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
endfunction()

# athenea_add_module(<name> SOURCES ... [PUBLIC_LIBS ...] [PRIVATE_LIBS ...])
#
# A static library `athenea_<name>`, aliased `athenea::<name>`, whose public headers are
# modules/<name>/include/athenea/<name>/*.h.
function(athenea_add_module name)
    cmake_parse_arguments(ARG "" "" "SOURCES;PUBLIC_LIBS;PRIVATE_LIBS" ${ARGN})
    add_library(athenea_${name} STATIC ${ARG_SOURCES})
    add_library(athenea::${name} ALIAS athenea_${name})
    target_include_directories(athenea_${name}
        PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(athenea_${name} PUBLIC cxx_std_20)
    if(ARG_PUBLIC_LIBS)
        target_link_libraries(athenea_${name} PUBLIC ${ARG_PUBLIC_LIBS})
    endif()
    if(ARG_PRIVATE_LIBS)
        target_link_libraries(athenea_${name} PRIVATE ${ARG_PRIVATE_LIBS})
    endif()
    athenea_set_target_defaults(athenea_${name})
endfunction()
