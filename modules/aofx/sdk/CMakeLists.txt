# The kernel rule is aopenfx's (sdk/cmake/AofxKernel.cmake); the bundle rule is
# ours, on top of it, until aopenfx's AofxPlugin.cmake grows OUTPUT_DIR and
# LIBRARIES. Both are functions, global once included, so plugins/ and tests/
# can call them from their own directories.
list(APPEND CMAKE_MODULE_PATH "${ATHENEA_AOFX_DIR}/sdk/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AofxKernel)
include(AofxBundle)
