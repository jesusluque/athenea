#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# LAMPS ADDED TO A FOUND SCENE, AGAINST THE SAME LAMPS PATH TRACED. A scene shot from a stage keeps each
# gaussian's base colour apart from the light it was found under; `neural-scene render --lamps` adds lamps to
# it with no path traced. This writes the same lamps as UsdLux lights in a layer over the stage, path traces
# that from views the scene's walk never took, and compares there:
#
#   the scene with the lamps      against the stage with the lamps      what the lamps' direct light gives
#   the scene without them        against the stage with the lamps      what was there to gain
#   the scene with the lamps      against the stage with them, one bounce   the lamps' radiometry alone
#
#   qa_relight.py --athenea <binary> --stage <dir>/<name>.usda --scene <scene.aths> --lamps lamps.txt
#                 --held held.txt --out <dir> [--size 960x540] [--focal 24] [--ray-margin M] [--exposure S]
#
# A lamp is a line of lamps.txt, as `render --lamps` reads it: `sphere x y z radius intensity r g b` or
# `spot x y z tx ty tz half-angle radius intensity r g b` (a distant light is not written to the layer).
# Nothing here computes on the data: it writes a layer, runs the engine and lays what it prints in a table.
import argparse
import math
import os
import re
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--athenea", required=True)
parser.add_argument("--stage", required=True)
parser.add_argument("--scene", required=True)
parser.add_argument("--lamps", required=True)
parser.add_argument("--held", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--size", default="960x540")
parser.add_argument("--focal", type=float, default=24.0)
parser.add_argument("--ray-margin", default="")
parser.add_argument("--exposure", default="", help="stops, for the pictures written beside the measures")
parser.add_argument("--reference-samples", type=int, default=128)
parser.add_argument("--reference-bounces", type=int, default=8)
parser.add_argument("--skip-reference", action="store_true")
parser.add_argument("--compare-backend", default="")
args = parser.parse_args()

stage = os.path.abspath(args.stage)
stage_dir = os.path.dirname(stage)
out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "qa.log"), "a")


def run(arguments):
    began = time.time()
    environment = dict(os.environ)
    if arguments[0] == "compare" and args.compare_backend:
        environment["ATHENEA_BACKEND"] = args.compare_backend
    done = subprocess.run([args.athenea] + arguments, cwd=stage_dir, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace", env=environment)
    kept = [line for line in done.stdout.splitlines() if "Node interface" not in line and line.strip()]
    log.write("$ athenea %s\n%s\n(rc %d, %.1f s)\n\n" % (" ".join(arguments), "\n".join(kept), done.returncode,
                                                         time.time() - began))
    log.flush()
    return done.returncode, kept, time.time() - began


def facing(position, target):
    """A UsdLux light sends along its -z: the rows of the matrix that turns -z towards `target`."""
    back = [p - t for p, t in zip(position, target)]
    length = math.sqrt(sum(c * c for c in back)) or 1.0
    back = [c / length for c in back]
    aside = [0.0, 0.0, 1.0] if abs(back[2]) < 0.9 else [1.0, 0.0, 0.0]
    right = [aside[1] * back[2] - aside[2] * back[1], aside[2] * back[0] - aside[0] * back[2],
             aside[0] * back[1] - aside[1] * back[0]]
    length = math.sqrt(sum(c * c for c in right))
    right = [c / length for c in right]
    up = [back[1] * right[2] - back[2] * right[1], back[2] * right[0] - back[0] * right[2],
          back[0] * right[1] - back[1] * right[0]]
    return "( (%g, %g, %g, 0), (%g, %g, %g, 0), (%g, %g, %g, 0), (%g, %g, %g, 1) )" % (*right, *up, *back, *position)


# 1. THE LAMPS AS A LAYER OVER THE STAGE.
lights = []
for k, line in enumerate(open(args.lamps)):
    words = line.split()
    if not words or words[0].startswith("#"):
        continue
    n = [float(w) for w in words[1:]]
    if words[0] == "sphere" and len(n) == 8:
        lights.append('''    def SphereLight "Lamp%d"
    {
        float inputs:radius = %g
        float inputs:intensity = %g
        color3f inputs:color = (%g, %g, %g)
        double3 xformOp:translate = (%g, %g, %g)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }''' % (k, n[3], n[4], n[5], n[6], n[7], n[0], n[1], n[2]))
    elif words[0] == "spot" and len(n) == 12:
        lights.append('''    def SphereLight "Lamp%d" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float inputs:radius = %g
        float inputs:intensity = %g
        color3f inputs:color = (%g, %g, %g)
        float inputs:shaping:cone:angle = %g
        matrix4d xformOp:transform = %s
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }''' % (k, n[7], n[8], n[9], n[10], n[11], n[6], facing(n[0:3], n[3:6])))
# The lamps alone, as a file `render --lamps` reads as well as the text one.
open(os.path.join(out, "lamps.usda"), "w").write('#usda 1.0\n\ndef Xform "AddedLamps"\n{\n%s\n}\n' % "\n".join(lights))
layer = os.path.join(out, "with_lamps.usda")
open(layer, "w").write('#usda 1.0\n(\n    subLayers = [\n        @%s@\n    ]\n)\n\ndef Xform "AddedLamps"\n{\n%s\n}\n'
                       % (stage, "\n".join(lights)))

names = [line.split()[0] for line in open(args.held) if len(line.split()) >= 7 and not line.startswith("#")]
margin = ["--ray-margin", args.ray_margin] if args.ray_margin else []
report = ["# %s: %d lamps added to %s" % (os.path.basename(stage), len(lights), os.path.basename(args.scene)), ""]


def traced(directory, bounces, what):
    os.makedirs(directory, exist_ok=True)
    if args.skip_reference and all(os.path.exists(os.path.join(directory, "%s.exr" % n)) for n in names):
        return
    rc, _, seconds = run(["stage", layer, "--cameras", os.path.abspath(args.held), "--focal", str(args.focal),
                          "--size", args.size, "--technique", "rt", "--path-total", str(args.reference_samples),
                          "--path-samples", "8", "--path-bounces", str(bounces), "--denoise", "-o",
                          os.path.join(directory, "ref.exr")] + margin)
    report.append("- %s: %d views at %s, %d paths, %d bounces, in %.0f s (rc %d)" %
                  (what, len(names), args.size, args.reference_samples, bounces, seconds, rc))


# 2. THE STAGE WITH THE LAMPS, PATH TRACED: whole, and with one bounce (the lamps' direct light and no more).
whole = os.path.join(out, "ref")
direct = os.path.join(out, "ref1")
traced(whole, args.reference_bounces, "the stage with the lamps")
traced(direct, 1, "the same with one bounce")


# 3. THE SCENE: as it was found, and with the lamps.
def drawn(stem, more):
    rc, lines, seconds = run(["neural-scene", "render", os.path.abspath(args.scene), "--views",
                              os.path.abspath(args.held), "--focal", str(args.focal), "--size", args.size, "-o",
                              os.path.join(out, stem + ".exr")] + more)
    said = [line for line in lines if line.startswith("lamps:") or " ms a frame" in line]
    report.append("- the scene %s: rc %d, %.0f s; %s" % (stem, rc, seconds, "; ".join(said)))
    if args.exposure:
        run(["neural-scene", "render", os.path.abspath(args.scene), "--views", os.path.abspath(args.held), "--focal",
             str(args.focal), "--size", args.size, "--exposure", args.exposure, "-o",
             os.path.join(out, stem + ".png")] + more)


drawn("found", ["--glass"])
drawn("lit", ["--lamps", os.path.abspath(args.lamps)])
drawn("lamps", ["--lamps", os.path.abspath(args.lamps), "--lamps-only"])


def measure(image, reference):
    _, said, _ = run(["compare", image, reference, "--mask", image, "--mask-mode", "alpha", "--min-alpha", "0.5"])
    psnr = ssim = mean = other = "--"
    for line in said:
        m = re.search(r"PSNR ([0-9.]+|inf) dB\s+SSIM ([0-9.]+)", line)
        if m:
            psnr, ssim = m.group(1), m.group(2)
        m = re.match(r"image\s+mean ([0-9.e+-]+) ([0-9.e+-]+) ([0-9.e+-]+)", line)
        if m:
            mean = "%.4f" % ((float(m.group(1)) + float(m.group(2)) + float(m.group(3))) / 3.0)
        m = re.match(r"reference\s+mean ([0-9.e+-]+) ([0-9.e+-]+) ([0-9.e+-]+)", line)
        if m:
            other = "%.4f" % ((float(m.group(1)) + float(m.group(2)) + float(m.group(3))) / 3.0)
    return psnr, ssim, mean, other


def image_of(stem, view):
    for name in ("%s.exr_%s.exr" % (stem, view), "%s_%s.exr" % (stem, view)):
        if os.path.exists(os.path.join(out, name)):
            return os.path.join(out, name)
    return os.path.join(out, "%s_%s.exr" % (stem, view))


# A STAGE THAT AUTHORS NO LIGHT OF ITS OWN is lit by the engine's default lights, and those go when a layer
# adds a lamp: the stage with the lamps is then lit by the lamps alone, and what compares with it is the scene
# with the lamps *only* -- the last column. For a stage with lights of its own it is the third.
report += ["", "| view | found, against the stage with lamps: dB, SSIM | with the lamps: dB, SSIM | mean light: scene with "
           "lamps, stage with lamps | with the lamps against one bounce: dB | the lamps only, against the stage with "
           "lamps: dB, SSIM |", "|---|---|---|---|---|---|"]
for view in names:
    before = measure(image_of("found", view), os.path.join(whole, "%s.exr" % view))
    after = measure(image_of("lit", view), os.path.join(whole, "%s.exr" % view))
    alone = measure(image_of("lit", view), os.path.join(direct, "%s.exr" % view))
    only = measure(image_of("lamps", view), os.path.join(whole, "%s.exr" % view))
    report.append("| %s | %s, %s | %s, %s | %s, %s | %s | %s, %s |" % (view, before[0], before[1], after[0], after[1],
                                                                    after[2], after[3], alone[0], only[0], only[1]))
open(os.path.join(out, "report.md"), "w").write("\n".join(report) + "\n")
print("\n".join(report))
