athenea_add_module(gpu
    SOURCES
        src/Device.cpp
        src/Buffer.cpp
        src/ShaderLibrary.cpp
        src/ComputeKernel.cpp
        src/CommandBatch.cpp
        src/AsyncReadback.cpp
        src/ShaderCache.cpp
        src/Texture.cpp
        src/RasterKernel.cpp
        src/RayTracingKernel.cpp
        src/algo/Mips.cpp
        src/algo/PrefixSum.cpp
        src/algo/RadixSort.cpp
    PUBLIC_LIBS
        athenea::core
        slang-rhi)
target_compile_definitions(athenea_gpu PRIVATE
    ATHENEA_BUILD_SHADER_DIR="${ATHENEA_SHADER_OUTPUT_DIR}")
if(ATHENEA_OPTIX_INCLUDE_DIR)
    target_compile_definitions(athenea_gpu PRIVATE ATHENEA_OPTIX_INCLUDE_DIR="${ATHENEA_OPTIX_INCLUDE_DIR}")
endif()
