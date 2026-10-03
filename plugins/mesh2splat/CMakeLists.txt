# Electronic Arts' mesh2splat as an effect: a mesh in, gaussian splats out.
# Four entries out of one blob, which is why KERNELS names all of them: the
# counters are cleared by one thread, then a thread counts each triangle's
# cells, then one workgroup settles where each triangle's gaussians start, and
# then a thread writes each triangle's. Counting before writing is what makes
# the order of the output the mesh's own instead of a race's. A mesh over the
# share its host gives it (`thinTo`) is thinned in between: its units tallied
# twice, a threshold selected, what is kept counted and scanned again.
aofx_add_bundle(athenea_aofx_mesh2splat
    NAME Mesh2Splat
    SOURCES Mesh2Splat.cpp
    KERNELS mesh2splat ENTRY m2sClear ENTRY m2sCount ENTRY m2sScan ENTRY m2sEmit
        ENTRY m2sThinTally ENTRY m2sThinSelect ENTRY m2sThinCount ENTRY m2sThinScan
    SOURCE mesh2splat.slang)
