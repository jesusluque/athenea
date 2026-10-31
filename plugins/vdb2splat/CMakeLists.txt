# Copyright (c) 2026 jesus luque.
# A volume into volumetric gaussians (proposal 103): the leaves of a sparse
# grid in, gaussians whose sum is its extinction out. One blob of kernels;
# Vdb2Splat.cpp runs them in the order its header lays out.
aofx_add_bundle(athenea_aofx_vdb2splat
    NAME Vdb2Splat
    SOURCES Vdb2Splat.cpp
    KERNELS vdb2splat ENTRY v2sClear ENTRY v2sBounds ENTRY v2sLeafMap ENTRY v2sLeafNodes ENTRY v2sCoarseMoments
            ENTRY v2sCoarseError ENTRY v2sCoarseErrorGather ENTRY v2sCutFlags ENTRY v2sCompact ENTRY v2sInitEmit
            ENTRY v2sReset ENTRY v2sPrepare ENTRY v2sForward ENTRY v2sLoss ENTRY v2sBackward ENTRY v2sAdam
            ENTRY v2sAttributes ENTRY v2sEm ENTRY v2sPrune ENTRY v2sUnder ENTRY v2sSpawn ENTRY v2sAliveFlags ENTRY v2sWrite
    SOURCE vdb2splat.slang)
