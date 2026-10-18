# Copyright (c) 2026 jesus luque.
#
# athenea_ofx_host: a minimal OpenFX host that loads Athenea.ofx.bundle, renders
# one frame and writes it as an EXR, so the plugin is exercised without Nuke.
# It is built on OpenFX's own host support library.

find_package(EXPAT REQUIRED)

file(GLOB _ofx_host_support_sources "${ATHENEA_OPENFX_DIR}/HostSupport/src/*.cpp")
add_library(athenea_ofx_host_support STATIC ${_ofx_host_support_sources})
target_include_directories(athenea_ofx_host_support PUBLIC "${ATHENEA_OPENFX_DIR}/include"
                                                          "${ATHENEA_OPENFX_DIR}/HostSupport/include")
target_link_libraries(athenea_ofx_host_support PUBLIC EXPAT::EXPAT ${CMAKE_DL_LIBS})
# OpenFX's code, not this project's: its warnings are not ours to fix.
target_compile_options(athenea_ofx_host_support PRIVATE -w)

add_executable(athenea_ofx_host ofx_test_host.cpp)
target_link_libraries(athenea_ofx_host PRIVATE athenea_ofx_host_support athenea::io)
athenea_set_target_defaults(athenea_ofx_host)
# HostSupport's headers do not hold to this project's warnings.
target_compile_options(athenea_ofx_host PRIVATE "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-old-style-cast;-Wno-shadow;-Wno-overloaded-virtual;-Wno-unused-parameter>")
add_dependencies(athenea_ofx_host athenea_ofx)
