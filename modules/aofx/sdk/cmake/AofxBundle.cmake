# aofx_add_bundle(<target> NAME <Name> SOURCES ... [KERNELS <name> ENTRY ... ]
#                 [OUTPUT_DIR <dir>] [LIBRARIES ...])
#
# The bundle layout every plugin CMakeLists in the compositor wrote out by hand:
#
#     <dir>/<Name>.aofx.bundle/Contents/Info.plist        (macOS)
#     <dir>/<Name>.aofx.bundle/Contents/<arch>/<Name>.aofx
#
# with the one setting that is easy to forget and silent when forgotten:
# default visibility, or the four AofxGet* entry points are hidden and the host
# refuses the bundle.
#
# KERNELS takes one kernel per `KERNELS <name> ENTRY <e> [ENTRY <e>...]` group;
# call aofx_add_kernel yourself for several blobs.

# A cache entry, not a directory variable: bundles are declared from other
# directories than the one that included this file.
if(APPLE)
    set(AOFX_ARCH_DIR "MacOS" CACHE INTERNAL "")
elseif(WIN32)
    set(AOFX_ARCH_DIR "Win64" CACHE INTERNAL "")
else()
    set(AOFX_ARCH_DIR "Linux-x86-64" CACHE INTERNAL "")
endif()
set(AOFX_BUNDLE_DIR "${ATHENEA_BUILD}/aofx" CACHE PATH "Where bundles built here land")
# The reverse-DNS prefix a bundle's Info.plist carries. A cache entry so a
# program that builds this engine's bundles beside its own can name them
# under its own prefix.
set(AOFX_BUNDLE_ID_PREFIX "rt.sparrow.aofxp." CACHE STRING "CFBundleIdentifier prefix for bundles built here")

function(aofx_add_bundle target)
    cmake_parse_arguments(ARG "" "NAME;OUTPUT_DIR" "SOURCES;LIBRARIES;KERNELS" ${ARGN})
    if(NOT ARG_NAME)
        message(FATAL_ERROR "aofx_add_bundle(${target}) needs NAME")
    endif()
    set(out "${ARG_OUTPUT_DIR}")
    if(NOT out)
        set(out "${AOFX_BUNDLE_DIR}")
    endif()
    set(contents "${out}/${ARG_NAME}.aofx.bundle/Contents")

    add_library(${target} MODULE ${ARG_SOURCES})
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "${ARG_NAME}" PREFIX "" SUFFIX ".aofx"
        LIBRARY_OUTPUT_DIRECTORY "${contents}/${AOFX_ARCH_DIR}"
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF)
    target_link_libraries(${target} PRIVATE aofx::aofx ${ARG_LIBRARIES})
    target_compile_features(${target} PRIVATE cxx_std_20)

    if(ARG_KERNELS)
        list(GET ARG_KERNELS 0 kernel)
        list(REMOVE_AT ARG_KERNELS 0)
        aofx_add_kernel(${target} ${kernel} ${ARG_KERNELS})
    endif()

    if(APPLE)
        file(WRITE "${contents}/Info.plist" "<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
  <key>CFBundleExecutable</key><string>${ARG_NAME}.aofx</string>
  <key>CFBundleIdentifier</key><string>${AOFX_BUNDLE_ID_PREFIX}${ARG_NAME}</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
</dict></plist>
")
    endif()
endfunction()
