# athenea view: a window onto a stage. GLFW for the window, Dear ImGui for panels
# drawn by the engine's own device; the frames never leave the GPU.
athenea_add_module(view
    SOURCES
        src/Window.cpp
        src/ImGuiRenderer.cpp
        src/Viewer.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::technique
        athenea::usd
    PRIVATE_LIBS
        athenea_imgui
        glfw)
# USD's headers, through StageRenderer's, predate the warnings this engine asks for.
target_compile_options(athenea_view PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-deprecated-declarations
                       -Wno-double-promotion -Wno-cast-align)
set_target_properties(athenea_view PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
