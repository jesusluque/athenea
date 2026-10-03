# Copyright (c) 2026 jesus luque.
"""What the headless Blender tests share: the add-on, a scene, a render, and
`athenea compare` (the Measure effect, on the GPU) to judge it.

Run by tests/blender/run.sh; nothing here is a CPU oracle -- the numbers are
the compare command's, read from what it prints.
"""

import os
import re
import subprocess
import sys

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def arguments():
    """What follows `--` on Blender's command line, as a dict of --key value."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    found = {}
    key = None
    for word in argv:
        if word.startswith("--"):
            key = word[2:]
            found[key] = "1"
        elif key is not None:
            found[key] = word
            key = None
    return found


def enable_addon():
    sys.path.insert(0, os.path.join(REPO, "integrations", "blender"))
    import athenea_hydra
    athenea_hydra.register()
    return athenea_hydra


def athenea_cli():
    """The `athenea` binary of a desktop build, for compare."""
    candidates = [os.environ.get("ATHENEA_CLI", ""),
                  os.path.join(REPO, "build", "macos-arm64-debug", "bin", "athenea"),
                  os.path.join(REPO, "build", "macos-arm64-release", "bin", "athenea")]
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("no athenea binary for compare: set ATHENEA_CLI")


def scene_for_athenea(width=320, height=240):
    scene = bpy.context.scene
    scene.render.engine = 'ATHENEA_HYDRA'
    scene.hydra.export_method = 'USD'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'
    return scene


def render(path_stem):
    """F12 into `<stem>.exr` (32-bit float) and `<stem>.png`."""
    scene = bpy.context.scene
    bpy.ops.render.render()
    result = bpy.data.images["Render Result"]
    settings = scene.render.image_settings
    settings.file_format = 'OPEN_EXR'
    settings.color_depth = '32'
    settings.color_mode = 'RGBA'
    result.save_render(path_stem + ".exr", scene=scene)
    settings.file_format = 'PNG'
    settings.color_depth = '8'
    result.save_render(path_stem + ".png", scene=scene)
    return path_stem + ".exr"


_NUMBER = r"([-+0-9.eE]+|nan|inf)"


def compare(image, reference):
    """`athenea compare image reference`, its numbers as a dict."""
    run = subprocess.run([athenea_cli(), "compare", image, reference], capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"athenea compare failed ({run.returncode}): {run.stderr.strip()}")
    text = run.stdout
    print(text, end="")
    numbers = {}
    means = re.findall(r"^(image|reference)\s+mean " + r"\s+".join([_NUMBER] * 4), text, re.M)
    for name, *values in means:
        numbers[name + "_mean"] = [float(v) for v in values]
    hdr = re.search(r"relMSE " + _NUMBER, text)
    codes = re.search(r"8-bit\s+p99 (\d+)\s+max (\d+)\s+over 2: (\d+) of (\d+)", text)
    if hdr:
        numbers["relMSE"] = float(hdr.group(1))
    if codes:
        numbers["p99"] = int(codes.group(1))
        numbers["max"] = int(codes.group(2))
        numbers["over2"] = int(codes.group(3)) / max(int(codes.group(4)), 1)
    return numbers


class Checks:
    """Failures gathered, so one run reports every one of them."""

    def __init__(self):
        self.failed = []

    def check(self, condition, what):
        print(("ok    " if condition else "FAIL  ") + what)
        if not condition:
            self.failed.append(what)

    def finish(self):
        if self.failed:
            raise SystemExit(f"{len(self.failed)} check(s) failed: " + "; ".join(self.failed))
        print("all checks passed")
