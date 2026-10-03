# The viewer's panels, described once for every front end that draws them:
# Dear ImGui on the desktop, UIKit on a tablet. Nothing here knows about a
# device or a stage, so it links nothing but core.
athenea_add_module(ui
    SOURCES
        src/Controls.cpp
        src/ViewerPanels.cpp
    PUBLIC_LIBS
        athenea::core)
