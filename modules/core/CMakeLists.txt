athenea_add_module(core
    SOURCES
        src/Error.cpp
        src/Hash.cpp
        src/Log.cpp
        src/Platform.cpp)
find_package(Threads REQUIRED)
# dladdr (platform::moduleDir), in libdl before glibc 2.34.
target_link_libraries(athenea_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if(UNIX AND NOT APPLE)
    # A headless OpenGL context for libraries that draw with GL (Storm).
    find_library(ATHENEA_EGL_LIB EGL)
    find_path(ATHENEA_EGL_INCLUDE EGL/egl.h)
    if(ATHENEA_EGL_LIB AND ATHENEA_EGL_INCLUDE)
        target_link_libraries(athenea_core PUBLIC ${ATHENEA_EGL_LIB})
        target_compile_definitions(athenea_core PRIVATE ATHENEA_HAVE_EGL=1)
    endif()
endif()
if(APPLE)
    # Platform reaches a window's Metal layer through the Objective-C runtime.
    target_link_libraries(athenea_core PUBLIC objc "-framework CoreFoundation" "-framework CoreGraphics")
endif()
