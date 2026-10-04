#!/bin/sh
# Copyright (c) 2026 jesus luque.
#
# The headless Blender tests of the add-on.
#
#   tests/blender/run.sh [cpu|gpu|all] [out-dir]
#
#   cpu  test_module: registration, render settings, a conversion's and an
#        export's stages and arguments. Opens no GPU device.
#   gpu  tx_module (convert with TX and a shadow catcher, render, export,
#        Blender splats, Esc) and viewport_readback. Use the GPU: one at a
#        time, on a GPU turn.
#
# BLENDER (default /Applications/Blender.app/Contents/MacOS/Blender),
# ATHENEA_HYDRA_PLUGIN_DIR (default the macos-arm64-blender build's
# plugin/usd) and ATHENEA_CLI (an athenea binary for compare; default the
# macos-arm64-debug build's) say where things are.
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
group="${1:-cpu}"
out="${2:-/Users/muriel/luc/athenea-renders/blender}"
blender="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
export ATHENEA_HYDRA_PLUGIN_DIR="${ATHENEA_HYDRA_PLUGIN_DIR:-$repo/build/macos-arm64-blender/plugin/usd}"
case "$group" in
    cpu) tests="test_module" ;;
    gpu) tests="tx_module viewport_readback" ;;
    all) tests="test_module tx_module viewport_readback" ;;
    *) echo "run.sh: cpu, gpu or all, not $group" >&2; exit 2 ;;
esac
mkdir -p "$out"
failed=0
for test in $tests; do
    echo "== $test"
    "$blender" -b --factory-startup --python-exit-code 1 --python "$here/$test.py" -- --out "$out" \
        > "$out/$test.log" 2>&1
    status=$?
    grep -E "^(ok|FAIL) |median|cancel:|all checks|failed|Error" "$out/$test.log"
    [ $status -eq 0 ] || { echo "== $test FAILED ($status), see $out/$test.log"; failed=1; }
done
exit $failed
