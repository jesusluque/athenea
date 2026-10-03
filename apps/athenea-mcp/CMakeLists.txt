add_executable(athenea-mcp main.cpp)
target_link_libraries(athenea-mcp PRIVATE athenea::mcp)
athenea_set_target_defaults(athenea-mcp)
set_target_properties(athenea-mcp PROPERTIES BUILD_RPATH "${SLANG_ROOT}/lib;${ATHENEA_USD_ROOT}/lib")
