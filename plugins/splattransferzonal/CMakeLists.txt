# The splat transfer zonal fit: a baked transfer's nine harmonics in the world
# made into one or two zonal lobes whose axes are written in each gaussian's
# own frame, so a cloud a skeleton carries turns its transfer with it
# (proposal 014 B). Two entries out of one blob: the counters cleared, then a
# thread fits each gaussian.
aofx_add_bundle(athenea_aofx_splattransferzonal
    NAME SplatTransferZonal
    SOURCES SplatTransferZonal.cpp
    KERNELS splattransferzonal ENTRY tzClear ENTRY tzFit
    SOURCE splattransferzonal.slang)
