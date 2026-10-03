athenea_add_module(sched
    SOURCES
        src/FrameClock.cpp
    PUBLIC_LIBS
        athenea::core
        genlock::genlock)
