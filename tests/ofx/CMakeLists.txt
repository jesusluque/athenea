# Copyright (c) 2026 jesus luque.
#
# athenea_ofx_host: a minimal OpenFX host that loads Athenea.ofx.bundle, renders
# one frame and writes it as an EXR, so the plugin is exercised without Nuke.
# It is built on the host support library of the OpenFX the plugin builds on
# (Natron's), without the extensions: a host that knows nothing of planes.

# expat, which HostSupport parses its plugin cache with: the system's where
# there is one (macOS, Linux), else fetched and built static (Windows).
find_package(EXPAT QUIET)
if(NOT TARGET EXPAT::EXPAT)
    include(FetchContent)
    set(EXPAT_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(EXPAT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(EXPAT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(EXPAT_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(EXPAT_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(expat
        URL https://github.com/libexpat/libexpat/releases/download/R_2_6_4/expat-2.6.4.tar.gz
        URL_HASH SHA256=fd03b7172b3bd7427a3e7a812063f74754f24542429b634e0db6511b53fb2278)
    FetchContent_MakeAvailable(expat)
    target_compile_definitions(expat INTERFACE XML_STATIC)
    add_library(EXPAT::EXPAT ALIAS expat)
endif()

file(GLOB _ofx_host_support_sources "${ATHENEA_OPENFX_DIR}/HostSupport/src/*.cpp")
add_library(athenea_ofx_host_support STATIC ${_ofx_host_support_sources})
target_include_directories(athenea_ofx_host_support PUBLIC "${ATHENEA_OPENFX_DIR}/include"
                                                          "${ATHENEA_OPENFX_DIR}/HostSupport/include")
target_link_libraries(athenea_ofx_host_support PUBLIC EXPAT::EXPAT ${CMAKE_DL_LIBS})
if(MSVC)
    # HostSupport names std::auto_ptr, which MSVC keeps behind this switch.
    target_compile_definitions(athenea_ofx_host_support PUBLIC _HAS_AUTO_PTR_ETC=1)
endif()
# OpenFX's code, not this project's: its warnings are not ours to fix.
target_compile_options(athenea_ofx_host_support PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)

add_executable(athenea_ofx_host ofx_test_host.cpp)
target_link_libraries(athenea_ofx_host PRIVATE athenea_ofx_host_support athenea::io)
athenea_set_target_defaults(athenea_ofx_host)
# HostSupport's headers do not hold to this project's warnings.
target_compile_options(athenea_ofx_host PRIVATE "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-old-style-cast;-Wno-shadow;-Wno-overloaded-virtual;-Wno-unused-parameter>")
add_dependencies(athenea_ofx_host athenea_ofx)
