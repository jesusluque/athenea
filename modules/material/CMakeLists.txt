# Materials: the textures they sample (TextureStore), their lobes, and what
# MaterialX graphs compile to.
athenea_add_module(material
    SOURCES
        src/TextureStore.cpp
        src/MaterialCompiler.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::colour
    PRIVATE_LIBS
        hio usdShade sdf ar tf
        MaterialXCore MaterialXFormat MaterialXGenShader MaterialXGenHw MaterialXGenSlang)
# USD's headers predate the warnings this engine asks for.
target_compile_options(athenea_material PRIVATE "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-old-style-cast;-Wno-shadow;-Wno-cast-align;-Wno-deprecated-declarations;-Wno-double-promotion>")
# PXR_STATIC=0 only where OpenUSD is not DLLs: its headers test whether the
# name is defined, and on Windows that took away every dllimport.
target_compile_definitions(athenea_material PRIVATE "$<$<NOT:$<PLATFORM_ID:Windows>>:PXR_STATIC=0>"
                           TBB_SUPPRESS_DEPRECATED_MESSAGES=1)
set_target_properties(athenea_material PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
