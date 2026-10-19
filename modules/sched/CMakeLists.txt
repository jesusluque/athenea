# Copyright (c) 2026 jesus luque.
athenea_add_module(sched
    SOURCES
        src/FrameClock.cpp
    PUBLIC_LIBS
        athenea::core
        genlock::genlock)
