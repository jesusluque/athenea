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
target_compile_options(athenea_material PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-cast-align
                       -Wno-deprecated-declarations -Wno-double-promotion)
target_compile_definitions(athenea_material PRIVATE PXR_STATIC=0 TBB_SUPPRESS_DEPRECATED_MESSAGES=1)
set_target_properties(athenea_material PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
