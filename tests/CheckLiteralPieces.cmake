# Copyright (c) 2026 jesus luque.
# MSVC refuses a string literal piece over 16380 characters (C2026), and the
# kernels several sources generate are raw strings: a piece that grew past it
# broke the Windows build while every other compiler took it. Each `R"(...)"`
# piece of the sources named here is measured; over 16000 fails, naming it.
#
#   cmake -DSOURCES=a.cpp;b.cpp -P CheckLiteralPieces.cmake
set(limit 16000)
set(failed "")
foreach(source IN LISTS SOURCES)
    file(READ "${source}" text)
    set(rest "${text}")
    while(TRUE)
        string(FIND "${rest}" "R\"(" open)
        if(open EQUAL -1)
            break()
        endif()
        math(EXPR start "${open} + 3")
        string(SUBSTRING "${rest}" ${start} -1 rest)
        string(FIND "${rest}" ")\"" close)
        if(close EQUAL -1)
            break()
        endif()
        if(close GREATER limit)
            get_filename_component(name "${source}" NAME)
            string(SUBSTRING "${rest}" 0 60 head)
            string(REPLACE "\n" " " head "${head}")
            string(APPEND failed "  ${name}: a piece of ${close} characters, starting: ${head}\n")
        endif()
        math(EXPR after "${close} + 2")
        string(SUBSTRING "${rest}" ${after} -1 rest)
    endwhile()
endforeach()
if(failed)
    message(FATAL_ERROR "raw string pieces past ${limit} characters (MSVC stops at 16380):\n${failed}"
                        "split them at a line with )\" and R\"( between.")
endif()
list(LENGTH SOURCES n)
message(STATUS "raw string pieces under ${limit} characters in ${n} sources")
