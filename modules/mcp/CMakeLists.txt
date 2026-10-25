# Copyright (c) 2026 jesus luque.
# MCP over the renderer: the protocol and the tools. Links the engine's USD
# side, since a tool's subject is a stage, and the bridge, through which
# another process feeds it and reads its frames.
athenea_add_module(mcp
    SOURCES
        src/Server.cpp
        src/Tools.cpp
        src/FrameLoop.cpp
    PUBLIC_LIBS athenea::usd athenea::bridge)
target_link_libraries(athenea_mcp PRIVATE nlohmann_json::nlohmann_json)
# The frames' clock: sched::FrameClock where the build has genlock.
if(TARGET athenea::sched)
    target_link_libraries(athenea_mcp PRIVATE athenea::sched)
    target_compile_definitions(athenea_mcp PRIVATE ATHENEA_MCP_SCHED=1)
endif()
target_compile_options(athenea_mcp PRIVATE "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-old-style-cast;-Wno-shadow;-Wno-cast-align;-Wno-deprecated-declarations;-Wno-double-promotion>")
set_target_properties(athenea_mcp PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
