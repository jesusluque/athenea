# Electronic Arts' mesh2splat as an effect: a mesh in, gaussian splats out.
# Five entries out of one blob, which is why KERNELS names all of them: the
# counters are cleared by one thread, then a thread counts each triangle's
# cells, then one workgroup settles where each triangle's gaussians start, and
# then a thread writes each triangle's. Counting before writing is what makes
# the order of the output the mesh's own instead of a race's. The fifth
# measures what the surface asks of the density, where the host asks.
aofx_add_bundle(athenea_aofx_mesh2splat
    NAME Mesh2Splat
    SOURCES Mesh2Splat.cpp
    KERNELS mesh2splat ENTRY m2sClear ENTRY m2sCount ENTRY m2sScan ENTRY m2sEmit ENTRY m2sMeasure
    SOURCE mesh2splat.slang)
