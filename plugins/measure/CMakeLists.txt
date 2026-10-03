# Two pictures measured against each other: athenea's comparison as an effect,
# so `athenea compare` and a compositor's QC node are one implementation.
# Four entries out of one blob: a thread a row sums and bins, a thread a bin
# adds the rows, one thread walks the bins for the percentiles, and a thread a
# pixel draws the heatmap.
aofx_add_bundle(athenea_aofx_measure
    NAME Measure
    SOURCES Measure.cpp
    KERNELS measure ENTRY measureRows ENTRY measureReduce ENTRY measureFinish ENTRY measureHeatmap
    SOURCE measure.slang)
