# Slang sources compiled at run time by slang-rhi.
#
# The engine's own shaders are not embedded: they are compiled by the device
# the process opened, for the backend it opened, and a blob embedded at build
# time would pin one target. They are copied beside the binaries instead, and
# found through ATHENEA_SHADER_DIR (set at build) or <exe>/../shaders.

set(ATHENEA_SHADER_SOURCE_DIR "${ATHENEA_ROOT}/shaders")
# A parent may place the copy where its own layout wants it.
if(NOT DEFINED ATHENEA_SHADER_OUTPUT_DIR)
    set(ATHENEA_SHADER_OUTPUT_DIR "${ATHENEA_BUILD}/shaders")
endif()

# One target that copies every shader, so tests and apps depend on the same one.
function(athenea_shader_copy_target)
    if(TARGET athenea_shaders_copy)
        return()
    endif()
    file(GLOB_RECURSE _shaders CONFIGURE_DEPENDS
        "${ATHENEA_SHADER_SOURCE_DIR}/*.slang" "${ATHENEA_SHADER_SOURCE_DIR}/*.slangh"
        # MaterialX node implementations (athenea/material/mx) travel with the shaders they name.
        "${ATHENEA_SHADER_SOURCE_DIR}/*.mtlx")
    set(_outputs)
    foreach(_src IN LISTS _shaders)
        file(RELATIVE_PATH _rel "${ATHENEA_SHADER_SOURCE_DIR}" "${_src}")
        set(_dst "${ATHENEA_SHADER_OUTPUT_DIR}/${_rel}")
        add_custom_command(OUTPUT "${_dst}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}" "${_dst}"
            DEPENDS "${_src}" VERBATIM)
        list(APPEND _outputs "${_dst}")
    endforeach()
    add_custom_target(athenea_shaders_copy ALL DEPENDS ${_outputs})
    # PNanoVDB.h from the USD prefix travels with the shaders (cmake/Usd.cmake).
    if(TARGET athenea_nanovdb_header)
        add_dependencies(athenea_shaders_copy athenea_nanovdb_header)
    endif()
endfunction()

function(athenea_copy_shaders target)
    athenea_shader_copy_target()
    add_dependencies(${target} athenea_shaders_copy)
endfunction()

function(_athenea_copy_shaders_unused target)
    file(GLOB_RECURSE _shaders CONFIGURE_DEPENDS
        "${ATHENEA_SHADER_SOURCE_DIR}/*.slang" "${ATHENEA_SHADER_SOURCE_DIR}/*.slangh")
    set(_outputs)
    foreach(_src IN LISTS _shaders)
        file(RELATIVE_PATH _rel "${ATHENEA_SHADER_SOURCE_DIR}" "${_src}")
        set(_dst "${ATHENEA_SHADER_OUTPUT_DIR}/${_rel}")
        add_custom_command(OUTPUT "${_dst}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}" "${_dst}"
            DEPENDS "${_src}" VERBATIM)
        list(APPEND _outputs "${_dst}")
    endforeach()
    if(_outputs)
        add_custom_target(${target}_shaders DEPENDS ${_outputs})
        add_dependencies(${target} ${target}_shaders)
    endif()
endfunction()
