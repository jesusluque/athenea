# aofx: this engine's side of the AOFX host. The SDK and the host themselves
# are aopenfx's (third_party/aopenfx, added by the top-level CMakeLists as
# aofx::aofx and aofx::host); what is here is the bundle rule the plugins and
# the tests use, and the capabilities and `renderEffect` this engine brings.
add_subdirectory(sdk)
add_subdirectory(host)
