# Copyright (c) 2026 jesus luque.
"""What the colour costs on its way to Blender, float against half.

Blender 5.3 shows a delegate other than Storm by mapping its colour buffer
and uploading it to a texture every viewport frame (DrawTexture::
create_from_buffer); a final render maps it once and copies it
(RenderTaskDelegate::read_aov). Both go through HdAtheneaRenderBuffer::Map,
which converts the AOV on the device and reads it back. The viewport cannot
be driven headless, so this times the final path at viewport sizes with the
colour as float (the default) and as half (`athenea:colourHalf`, what the
add-on asks for in the viewport), and checks that half draws the same picture.

The final path converts half to float on the CPU after the map (read_aov's
loop), which the viewport does not -- it uploads the halves as they are -- so
the half timings here are an upper bound on the viewport's.

  blender -b --factory-startup --python-exit-code 1 \
      --python tests/blender/viewport_readback.py -- --out <dir> [--frames 12]

Uses the GPU: run on a GPU turn.
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import bpy  # noqa: E402


def main():
    args = common.arguments()
    out = os.path.abspath(args.get("out", "/Users/muriel/luc/athenea-renders/blender-phase2"))
    frames = int(args.get("frames", "12"))
    os.makedirs(out, exist_ok=True)
    checks = common.Checks()
    addon = common.enable_addon()
    engine = addon.AtheneaHydraRenderEngine
    original = engine.get_render_settings

    def with_half(self, engine_type):
        settings = dict(original(self, engine_type))
        settings['athenea:colourHalf'] = True
        return settings

    for width, height in ((1280, 720), (1920, 1080)):
        scene = common.scene_for_athenea(width, height)
        medians = {}
        images = {}
        for label, patch in (("float", original), ("half", with_half)):
            engine.get_render_settings = patch
            times = []
            for _ in range(frames):
                start = time.perf_counter()
                bpy.ops.render.render()
                times.append((time.perf_counter() - start) * 1000.0)
            medians[label] = statistics.median(times[2:] if len(times) > 4 else times)
            images[label] = common.render(os.path.join(out, f"readback_{label}_{width}x{height}"))
            print(f"{width}x{height} {label}: median {medians[label]:.1f} ms a frame over {frames}")
        engine.get_render_settings = original
        numbers = common.compare(images["half"], images["float"])
        checks.check(numbers["p99"] <= 1, f"{width}x{height}: half draws what float draws (p99 {numbers['p99']})")
        print(f"{width}x{height}: float {medians['float']:.1f} ms, half {medians['half']:.1f} ms")
    checks.finish()


main()
