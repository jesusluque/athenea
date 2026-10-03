# Geometry on the device: meshes (triangulation, normals, primvars), and later
# subdivision, curves, skinning and instancing.
athenea_add_module(geom
    SOURCES
        src/Mesh.cpp
        src/Skinner.cpp
        src/Curves.cpp
        src/Subdivision.cpp
    PUBLIC_LIBS
        athenea::core
        athenea::gpu
        athenea::scene)
