# The splat bake filter: a baked cloud's indirect light averaged between
# neighbouring gaussians on its surface, a-trous, weighed by where they stand
# on the tangent plane, which way they face, which prim they came from and
# how noisy each one is. Six entries out of one blob: the counters cleared, the
# footprints measured, the cell settled, the gaussians hashed into it, then
# one pass of the filter an iteration.
aofx_add_bundle(athenea_aofx_splatbakefilter
    NAME SplatBakeFilter
    SOURCES SplatBakeFilter.cpp
    KERNELS splatbakefilter ENTRY bfClear ENTRY bfMeasure ENTRY bfCell ENTRY bfInsert ENTRY bfIterate
    SOURCE splatbakefilter.slang)
