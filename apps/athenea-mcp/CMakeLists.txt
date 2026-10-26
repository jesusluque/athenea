# Copyright (c) 2026 jesus luque.
add_executable(athenea-mcp main.cpp)
target_link_libraries(athenea-mcp PRIVATE athenea::mcp)
athenea_set_target_defaults(athenea-mcp)
set_target_properties(athenea-mcp PROPERTIES
    BUILD_RPATH "${ATHENEA_ORIGIN}/../${CMAKE_INSTALL_LIBDIR};${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib"
    INSTALL_RPATH "${ATHENEA_ORIGIN}/../${CMAKE_INSTALL_LIBDIR};${SLANG_ROOT}/lib")
install(TARGETS athenea-mcp RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
