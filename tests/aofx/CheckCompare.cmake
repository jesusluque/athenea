# Copyright (c) 2026 jesus luque.
#
# `athenea compare` against what it printed before it ran the Measure effect:
# the same files, the same arguments, the same digits. The expected text was
# written by the command when it called render::imageStats, compareHdr and
# compareImages itself; its log lines ("athenea [...") are left out.
#
#   cmake -DATHENEA=<binary> -DDATA=<tests/data/compare> -DEXPECTED=<name>
#         -DARGS=<a;b;...> -P CheckCompare.cmake
set(arguments "")
foreach(argument IN LISTS ARGS)
    if(argument MATCHES "\\.exr$")
        list(APPEND arguments "${DATA}/${argument}")
    else()
        list(APPEND arguments "${argument}")
    endif()
endforeach()
execute_process(COMMAND "${ATHENEA}" compare ${arguments}
                OUTPUT_VARIABLE printed RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "athenea compare ${ARGS} failed (${status}):\n${printed}")
endif()
string(REGEX REPLACE "athenea \\[[^\n]*\n" "" printed "${printed}")
file(READ "${DATA}/${EXPECTED}.txt" expected)
if(NOT printed STREQUAL expected)
    message(FATAL_ERROR "athenea compare ${ARGS} printed\n${printed}\nwhere it printed\n${expected}")
endif()
