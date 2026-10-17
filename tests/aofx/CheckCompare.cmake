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
# The same words and the same digits, but for a number's last digit: a mean
# is a sum the device reduces in its own order, and CUDA's last digit is not
# Metal's (0.258504 against 0.258503). Two numbers of one length that agree
# without their last character are the same number here.
if(NOT printed STREQUAL expected)
    string(REGEX MATCHALL "[^ \n]+" got "${printed}")
    string(REGEX MATCHALL "[^ \n]+" want "${expected}")
    list(LENGTH got gotCount)
    list(LENGTH want wantCount)
    set(same TRUE)
    if(NOT gotCount EQUAL wantCount)
        set(same FALSE)
    else()
        math(EXPR last "${gotCount} - 1")
        foreach(k RANGE ${last})
            list(GET got ${k} a)
            list(GET want ${k} b)
            if(NOT a STREQUAL b)
                string(LENGTH "${a}" lengthA)
                string(LENGTH "${b}" lengthB)
                if(a MATCHES "^-?[0-9.]+$" AND b MATCHES "^-?[0-9.]+$" AND lengthA EQUAL lengthB AND lengthA GREATER 2)
                    math(EXPR head "${lengthA} - 1")
                    string(SUBSTRING "${a}" 0 ${head} headA)
                    string(SUBSTRING "${b}" 0 ${head} headB)
                    if(NOT headA STREQUAL headB)
                        set(same FALSE)
                    endif()
                else()
                    set(same FALSE)
                endif()
            endif()
        endforeach()
    endif()
    if(NOT same)
        message(FATAL_ERROR "athenea compare ${ARGS} printed\n${printed}\nwhere it printed\n${expected}")
    endif()
endif()
