#!/bin/sh
# Copyright (c) 2026 jesus luque.
#
# The headless Blender tests: the add-on's mesh2splat operator end to end, and
# the colour's readback float against half. Both use the GPU.
#
#   tests/blender/run.sh [out-dir]
#
# BLENDER (default /Applications/Blender.app/Contents/MacOS/Blender),
# ATHENEA_HYDRA_PLUGIN_DIR (default the macos-arm64-blender build's
# plugin/usd) and ATHENEA_CLI (an athenea binary for compare; default the
# macos-arm64-debug build's) say where things are.
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="${1:-/Users/muriel/luc/athenea-renders/blender-phase2}"
blender="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
export ATHENEA_HYDRA_PLUGIN_DIR="${ATHENEA_HYDRA_PLUGIN_DIR:-$repo/build/macos-arm64-blender/plugin/usd}"
mkdir -p "$out"
failed=0
for test in convert_and_render viewport_readback; do
    echo "== $test"
    "$blender" -b --factory-startup --python-exit-code 1 --python "$here/$test.py" -- --out "$out" \
        > "$out/$test.log" 2>&1
    status=$?
    grep -E "^(ok|FAIL) |median|all checks|failed|Error" "$out/$test.log"
    [ $status -eq 0 ] || { echo "== $test FAILED ($status), see $out/$test.log"; failed=1; }
done
exit $failed
