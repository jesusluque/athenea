# Copyright (c) 2026 jesus luque.
#
# PNanoVDB.h as the shaders include it: copied from the USD prefix with one
# change. PNanoVDB reads a grid type's layout from a table indexed by the
# type (PNANOVDB_GRID_TYPE_GET), and Slang emits that table for Metal as an
# array built in thread memory where it is used -- 20 rows of 28 words.
# Indexed by a value the compiler cannot fold (a type read from a header, or
# a constant passed into a function it did not inline), the array stays on
# the thread's stack, and kernels that kept several came out wrong on Metal
# -- different leaves wrong each run -- while CUDA was right. The table is
# read here through a function that indexes it by literals only, which
# folds.
#   cmake -DIN=<PNanoVDB.h> -DOUT=<copy> -P PatchPNanoVDB.cmake
file(READ "${IN}" text)
set(old "#define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) pnanovdb_grid_type_constants[grid_typeIn].nameIn")
set(new "#define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) pnanovdb_grid_type_get(grid_typeIn).nameIn")
string(FIND "${text}" "${old}" at)
if(at EQUAL -1)
    message(FATAL_ERROR "PatchPNanoVDB: PNANOVDB_GRID_TYPE_GET not found in ${IN}")
endif()
string(REPLACE "${old}" "${new}" text "${text}")
set(anchor "// ------------------------------------------------ Basic Lookup")
string(FIND "${text}" "${anchor}" at)
if(at EQUAL -1)
    message(FATAL_ERROR "PatchPNanoVDB: the table's end not found in ${IN}")
endif()
set(getter "// athenea: a type's row of the table by literal indices only (cmake/PatchPNanoVDB.cmake).\nPNANOVDB_FORCE_INLINE pnanovdb_grid_type_constants_t pnanovdb_grid_type_get(pnanovdb_grid_type_t grid_type)\n{\n")
foreach(t RANGE 1 19)
    string(APPEND getter "    if (grid_type == ${t}u) { return pnanovdb_grid_type_constants[${t}]; }\n")
endforeach()
string(APPEND getter "    return pnanovdb_grid_type_constants[0];\n}\n\n")
string(SUBSTRING "${text}" 0 ${at} head)
string(SUBSTRING "${text}" ${at} -1 tail)
file(WRITE "${OUT}" "${head}${getter}${tail}")
