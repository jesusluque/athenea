# Copyright (c) 2026 jesus luque.
#
# A SHADOW CATCHER DARKENS THE GROUND AS THE PATH TRACED FRAME DOES (task
# PLAY-G). Converts STAGE's PRIM's shadow with `athenea mesh2splat
# --shadow-catcher`, draws the stage's meshes with the catcher on the raster
# route -- the meshes cast no shadow of their own there (a prefiltered dome
# traces no ray), so the catcher is the whole of the ground's shadow -- and
# measures the ground in four windows against the path traced frame, each
# with `athenea compare --window` (the means are the Measure effect's, on the
# device). Fails where a window's mean is further than its tolerance from
# the GT's: TOL for the ground under and around the object, CONTACT_TOL for
# the contact window, where the catcher's 1.26 cm cell is coarse against a
# gap of a few centimetres. Skips (77) where the stage is not on this
# machine.
#
#   cmake -DATHENEA=... -DSTAGE=... -DPRIM=... -DCAMERA=... -DOUT=... [-DGT=gt.exr]
#         [-DWIDTH=960 -DHEIGHT=540] [-DPATHS=256] [-DTOL=0.10] [-DCONTACT_TOL=0.40]
#         -P catcher_against_gt.cmake
#
# The windows are the Corvette's (research 056's crops at 960 x 540): under
# the car, the bumper's contact, the open ground right and left, as X0 Y0 X1
# Y1 with rows counted from the bottom. Not the ground under the body, which
# this frame sees past the mesh car's underside -- lit, there, with no mesh
# shadowing another from a prefiltered dome -- rather than the catcher.
if(NOT EXISTS "${STAGE}")
    message("catcher_against_gt: ${STAGE} is not on this machine")
    cmake_language(EXIT 77)
endif()
if(NOT DEFINED WIDTH)
    set(WIDTH 960)
endif()
if(NOT DEFINED HEIGHT)
    set(HEIGHT 540)
endif()
if(NOT DEFINED PATHS)
    set(PATHS 256)
endif()
if(NOT DEFINED TOL)
    set(TOL 0.10)
endif()
if(NOT DEFINED CONTACT_TOL)
    set(CONTACT_TOL 0.40)
endif()
file(MAKE_DIRECTORY "${OUT}")
execute_process(
    COMMAND "${ATHENEA}" mesh2splat "${STAGE}" --prim "${PRIM}" --shadow-catcher --bake-samples 64 --bake-extra 0
            -o "${OUT}/catcher.usdc"
    RESULT_VARIABLE code OUTPUT_VARIABLE said ERROR_VARIABLE complained)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "catcher_against_gt: the catcher's conversion failed:\n${said}\n${complained}")
endif()
file(WRITE "${OUT}/with_catcher.usda"
"#usda 1.0\n(\n    subLayers = [@${STAGE}@]\n)\n\nover \"World\"\n{\n    def \"ShadowCatcher\" (\n        prepend references = @./catcher.usdc@</World/Splats>\n    )\n    {\n    }\n}\n")
if(NOT DEFINED GT)
    set(GT "${OUT}/gt.exr")
    execute_process(
        COMMAND "${ATHENEA}" stage "${STAGE}" --camera "${CAMERA}" --size ${WIDTH}x${HEIGHT} --technique rt
                --path-total ${PATHS} -o "${GT}"
        RESULT_VARIABLE code OUTPUT_VARIABLE said ERROR_VARIABLE complained)
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "catcher_against_gt: the path traced frame failed:\n${said}\n${complained}")
    endif()
endif()
execute_process(
    COMMAND "${ATHENEA}" stage "${OUT}/with_catcher.usda" --camera "${CAMERA}" --size ${WIDTH}x${HEIGHT}
            -o "${OUT}/raster.exr"
    RESULT_VARIABLE code OUTPUT_VARIABLE said ERROR_VARIABLE complained)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "catcher_against_gt: the raster frame failed:\n${said}\n${complained}")
endif()

set(names under_the_car contact open_right open_left)
set(windows "550 80 700 100" "150 110 250 130" "775 30 875 60" "30 300 130 330")
set(failed FALSE)
foreach(k RANGE 3)
    list(GET names ${k} name)
    list(GET windows ${k} window)
    separate_arguments(window)
    execute_process(COMMAND "${ATHENEA}" compare "${OUT}/raster.exr" "${GT}" --window ${window}
                    RESULT_VARIABLE code OUTPUT_VARIABLE said ERROR_VARIABLE complained)
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "catcher_against_gt: compare failed:\n${said}\n${complained}")
    endif()
    string(REGEX MATCH "image +mean ([^ ]+)" _ "${said}")
    set(got "${CMAKE_MATCH_1}")
    string(REGEX MATCH "reference +mean ([^ ]+)" _ "${said}")
    set(want "${CMAKE_MATCH_1}")
    set(tolerance ${TOL})
    if(name STREQUAL "contact")
        set(tolerance ${CONTACT_TOL})
    endif()
    # Two numbers compared, nothing computed on the data.
    execute_process(COMMAND python3 -c "import sys; sys.exit(0 if abs(${got} - ${want}) <= ${tolerance} * ${want} else 1)"
                    RESULT_VARIABLE off)
    if(off EQUAL 0)
        message("  ${name}: ${got} against ${want}: pass")
    else()
        message("  ${name}: ${got} against ${want}: FAIL (more than ${tolerance} of it apart)")
        set(failed TRUE)
    endif()
endforeach()
if(failed)
    message(FATAL_ERROR "catcher_against_gt: the catcher's ground is further from the path traced frame than allowed")
endif()
