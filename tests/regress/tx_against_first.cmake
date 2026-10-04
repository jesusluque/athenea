# Copyright (c) 2026 jesus luque.
#
# A TX TRANSFER IS NO WORSE THAN THE FIRST, material by material (the
# regression gate after the pawn's head came back milky). Converts STAGE's
# PRIM twice with `athenea mesh2splat --validate` -- the TX transfer as the
# defaults make it, and the first transfer (--transfer-cells 0
# --transfer-degree 2) -- into OUT/tx and OUT/first, each with its GT|mesh|
# cloud picture per material, and fails where a material's TX relMSE is
# above SLACK times the first's. Skips (exit code 77, ctest's SKIP_RETURN_CODE)
# where the stage is not on this machine.
#
#   cmake -DATHENEA=... -DSTAGE=... -DPRIM=... -DCAMERA=... -DOUT=... [-DSIZE=512 | -DWIDTH= -DHEIGHT=]
#         [-DPATHS=512]
#         [-DSLACK=1.05] -P tx_against_first.cmake
if(NOT EXISTS "${STAGE}")
    message("tx_against_first: ${STAGE} is not on this machine")
    cmake_language(EXIT 77)
endif()
if(NOT DEFINED SIZE)
    set(SIZE 512)
endif()
if(NOT DEFINED WIDTH)
    set(WIDTH ${SIZE})
endif()
if(NOT DEFINED HEIGHT)
    set(HEIGHT ${SIZE})
endif()
if(NOT DEFINED PATHS)
    set(PATHS 512)
endif()
if(NOT DEFINED SLACK)
    set(SLACK 1.05)
endif()
foreach(mode tx first)
    set(extra "")
    if(mode STREQUAL "first")
        set(extra --transfer-cells 0 --transfer-degree 2)
    endif()
    execute_process(
        COMMAND "${ATHENEA}" mesh2splat "${STAGE}" --prim "${PRIM}" --transfer ${extra}
                --validate "${OUT}/${mode}" --validate-camera "${CAMERA}" --validate-size ${WIDTH} ${HEIGHT}
                --validate-paths ${PATHS}
        RESULT_VARIABLE code
        OUTPUT_VARIABLE said
        ERROR_VARIABLE complained)
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "tx_against_first: the ${mode} conversion failed:\n${said}\n${complained}")
    endif()
    file(READ "${OUT}/${mode}/validate.json" json)
    string(JSON n LENGTH "${json}" materials)
    math(EXPR last "${n} - 1")
    foreach(k RANGE ${last})
        string(JSON name GET "${json}" materials ${k} material)
        string(JSON rel GET "${json}" materials ${k} relMSE)
        set(${mode}_${k}_name "${name}")
        set(${mode}_${k}_rel "${rel}")
    endforeach()
endforeach()
set(failed FALSE)
foreach(k RANGE ${last})
    set(name "${tx_${k}_name}")
    set(tx "${tx_${k}_rel}")
    set(first "${first_${k}_rel}")
    if(tx GREATER first)
        # Past the slack only: first times SLACK, which CMake's math() (whole
        # numbers) cannot multiply; two numbers compared, nothing computed on
        # the data.
        execute_process(COMMAND python3 -c "import sys; sys.exit(0 if ${tx} <= ${first} * ${SLACK} else 1)"
                        RESULT_VARIABLE worse)
    else()
        set(worse 0)
    endif()
    if(worse EQUAL 0)
        message("  ${name}: TX ${tx}, first ${first}: pass")
    else()
        message("  ${name}: TX ${tx}, first ${first}: FAIL")
        set(failed TRUE)
    endif()
endforeach()
if(failed)
    message(FATAL_ERROR "tx_against_first: a material came out worse in TX than in the first transfer")
endif()
