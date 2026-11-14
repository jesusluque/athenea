#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# WHAT A FOUND SCENE MAKES OF A STAGE, MEASURED. A stage is path traced from views no look takes (the
# reference), shot into a scene of gaussians along a walk (`neural-scene shoot`), and the scene drawn from
# those same views -- as it was found, and with its glass worked out by the eye (`render --glass`). Every
# figure is `athenea compare`'s, a kernel's: the whole frame, and where the stage says where its things are
# (balls.txt: a name, a centre, a radius) the window each takes in each view. Then the scene is handed on in
# every format and at levels of detail, and what was written is counted.
#
#   qa.py --athenea <binary> --stage <dir>/<name>.usda --out <dir> [--size 960x540] [--focal 24]
#         [--walk walk.txt] [--held held.txt] [--balls balls.txt] [--ray-margin M] [--skip-shoot]
#
# Nothing here computes on the data: it runs the engine, reads the lines it prints and lays them in a table.
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--athenea", required=True)
parser.add_argument("--stage", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--size", default="960x540")
parser.add_argument("--focal", type=float, default=24.0)
parser.add_argument("--walk", default="walk.txt")
parser.add_argument("--held", default="held.txt")
parser.add_argument("--balls", default="balls.txt")
parser.add_argument("--samples", type=int, default=32)
parser.add_argument("--reference-samples", type=int, default=128)
parser.add_argument("--bounces", type=int, default=3)
parser.add_argument("--reference-bounces", type=int, default=8)
parser.add_argument("--ray-margin", default="")
parser.add_argument("--time-step", default="")
parser.add_argument("--levels", default="200000,100000,50000")
parser.add_argument("--skip-shoot", action="store_true")
parser.add_argument("--skip-reference", action="store_true")
args = parser.parse_args()

stage_dir = os.path.dirname(os.path.abspath(args.stage))
stage_name = os.path.basename(args.stage)
out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
width, height = (int(v) for v in args.size.split("x"))
log = open(os.path.join(out, "qa.log"), "a")


def run(arguments, cwd=None):
    began = time.time()
    done = subprocess.run([args.athenea] + arguments, cwd=cwd or stage_dir, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace")
    kept = [line for line in done.stdout.splitlines() if "Node interface" not in line and line.strip()]
    log.write("$ athenea %s\n%s\n(rc %d, %.1f s)\n\n" % (" ".join(arguments), "\n".join(kept), done.returncode,
                                                         time.time() - began))
    log.flush()
    return done.returncode, kept, time.time() - began


def views_of(path):
    views = []
    for line in open(path):
        words = line.split()
        if len(words) >= 7 and not line.startswith("#"):
            views.append((words[0], [float(w) for w in words[1:4]], [float(w) for w in words[4:7]],
                          [float(w) for w in words[7:10]] if len(words) >= 10 else [0.0, 1.0, 0.0]))
    return views


def window_of(view, centre, radius):
    """The pixels a ball takes in a view: its centre through the view's lens, and its radius at that depth."""
    _, eye, target, up = view
    forward = [t - e for t, e in zip(target, eye)]
    length = math.sqrt(sum(c * c for c in forward))
    forward = [c / length for c in forward]
    right = [forward[1] * up[2] - forward[2] * up[1], forward[2] * up[0] - forward[0] * up[2],
             forward[0] * up[1] - forward[1] * up[0]]
    length = math.sqrt(sum(c * c for c in right))
    right = [c / length for c in right]
    above = [right[1] * forward[2] - right[2] * forward[1], right[2] * forward[0] - right[0] * forward[2],
             right[0] * forward[1] - right[1] * forward[0]]
    to = [c - e for c, e in zip(centre, eye)]
    depth = sum(a * b for a, b in zip(to, forward))
    if depth <= radius:
        return None
    # A focal length over an aperture 24.576 mm across: pixels a unit of x over depth.
    scale = args.focal / 24.576 * width / depth
    x = width / 2 + sum(a * b for a, b in zip(to, right)) * scale
    y = height / 2 + sum(a * b for a, b in zip(to, above)) * scale   # rows from the bottom, as compare counts
    reach = radius * scale
    x0, x1 = max(int(x - reach), 0), min(int(x + reach) + 1, width)
    y0, y1 = max(int(y - reach), 0), min(int(y + reach) + 1, height)
    return (x0, y0, x1, y1) if x1 - x0 >= 8 and y1 - y0 >= 8 else None


def compare(image, reference, window=None):
    arguments = ["compare", image, reference]
    if window:
        arguments += ["--window"] + [str(v) for v in window]
    _, lines, _ = run(arguments)
    found = {}
    for line in lines:
        m = re.search(r"PSNR ([0-9.]+|inf) dB\s+SSIM ([0-9.]+)", line)
        if m:
            found["psnr"] = float(m.group(1)) if m.group(1) != "inf" else 99.0
            found["ssim"] = float(m.group(2))
        m = re.search(r"relMSE ([0-9.e+-]+)", line)
        if m:
            found["relmse"] = float(m.group(1))
    return found


report = ["# %s" % stage_name, ""]
held = views_of(os.path.join(stage_dir, args.held))
margin = ["--ray-margin", args.ray_margin] if args.ray_margin else []
scene = os.path.join(out, "scene.aths")

# 1. THE REFERENCE: the stage path traced from the held views.
reference = {}
if not args.skip_reference:
    rc, lines, seconds = run(["stage", stage_name, "--cameras", os.path.join(stage_dir, args.held), "--focal",
                              str(args.focal), "--size", args.size, "--technique", "rt", "--path-total",
                              str(args.reference_samples), "--path-samples", "8", "--path-bounces",
                              str(args.reference_bounces), "--denoise", "-o", os.path.join(out, "ref.exr")] + margin)
    report.append("- reference: %d views path traced at %d paths, %d bounces, in %.0f s (rc %d)" %
                  (len(held), args.reference_samples, args.reference_bounces, seconds, rc))
for name, *_ in held:
    path = os.path.join(out, "%s.exr" % name)
    if os.path.exists(path):
        reference[name] = path

# 2. THE SHOOT.
if not args.skip_shoot:
    import shutil
    shutil.rmtree(scene, ignore_errors=True)
    shoot = ["neural-scene", "shoot", stage_name, scene, "--views", os.path.join(stage_dir, args.walk), "--focal",
             str(args.focal), "--size", args.size, "--samples", str(args.samples), "--pass-samples", "8", "--bounces",
             str(args.bounces), "--denoise"] + margin
    if args.time_step:
        shoot += ["--time", "0", "--time-step", args.time_step]
    rc, lines, seconds = run(shoot)
    frames = [line for line in lines if ": frame " in line]
    last = frames[-1] if frames else ""
    standing = re.search(r"(\d+) gaussians stand in (\d+) pieces", last)
    report.append("- shoot: %d frames in %.0f s (%.1f s a frame, rc %d)%s" %
                  (len(frames), seconds, seconds / max(len(frames), 1), rc,
                   "; %s gaussians stand in %s pieces" % standing.groups() if standing else ""))

# 3. THE SCENE DRAWN from the held views: as found, and with its glass by the eye.
drawn = {}
for label, extra in (("found", []), ("glass", ["--glass"])):
    prefix = os.path.join(out, label)
    rc, lines, seconds = run(["neural-scene", "render", scene, "--views", os.path.join(stage_dir, args.held),
                              "--focal", str(args.focal), "--size", args.size, "-o", prefix] + extra)
    said = [line for line in lines if line.startswith("glass:")]
    timing = [line for line in lines if " ms a frame" in line]
    report.append("- drawn %s: rc %d%s%s" % (label, rc, "; " + timing[-1].split(": ", 1)[-1] if timing else "",
                                            "; " + said[0] if said else ""))
    drawn[label] = {name: "%s_%s.exr" % (prefix, name) for name, *_ in held}

# 4. THE MEASURES: the whole frame a view.
report += ["", "## The frame, against the path tracer", "",
           "| view | found: PSNR dB | SSIM | relMSE | with --glass: PSNR dB | SSIM | relMSE |", "|---|---|---|---|---|---|---|"]
results = {"views": {}, "balls": {}}
for name, *_ in held:
    if name not in reference:
        continue
    row = {}
    for label in ("found", "glass"):
        if os.path.exists(drawn[label][name]):
            row[label] = compare(drawn[label][name], reference[name])
    results["views"][name] = row
    cells = []
    for label in ("found", "glass"):
        m = row.get(label, {})
        cells += ["%.2f" % m["psnr"] if "psnr" in m else "--", "%.3f" % m["ssim"] if "ssim" in m else "--",
                  "%.4f" % m["relmse"] if "relmse" in m else "--"]
    report.append("| %s | %s |" % (name, " | ".join(cells)))

# 5. THING BY THING, where the stage says where its things are: the mean over the views that see each.
balls_path = os.path.join(stage_dir, args.balls)
if os.path.exists(balls_path):
    report += ["", "## Material by material (the window each ball takes, the mean over the held views)", "",
               "| material | found: PSNR dB | relMSE | with --glass: PSNR dB | relMSE | views |", "|---|---|---|---|---|---|"]
    for line in open(balls_path):
        words = line.split(None, 5)
        if len(words) < 5:
            continue
        ball, centre, radius = words[0], [float(w) for w in words[1:4]], float(words[4])
        said = words[5].strip() if len(words) > 5 else ball
        sums = {"found": [0.0, 0.0, 0], "glass": [0.0, 0.0, 0]}
        for view in held:
            if view[0] not in reference:
                continue
            window = window_of(view, centre, radius)
            if window is None:
                continue
            for label in ("found", "glass"):
                if not os.path.exists(drawn[label][view[0]]):
                    continue
                m = compare(drawn[label][view[0]], reference[view[0]], window)
                if "psnr" in m and "relmse" in m:
                    sums[label][0] += m["psnr"]
                    sums[label][1] += m["relmse"]
                    sums[label][2] += 1
        cells = []
        for label in ("found", "glass"):
            total, rel, n = sums[label]
            cells += ["%.2f" % (total / n) if n else "--", "%.4f" % (rel / n) if n else "--"]
        results["balls"][ball] = {label: {"psnr": sums[label][0] / sums[label][2], "relmse": sums[label][1] / sums[label][2]}
                                  for label in sums if sums[label][2]}
        report.append("| %s (%s) | %s | %d |" % (ball, said, " | ".join(cells), sums["found"][2]))

# 6. HANDED ON: every format, the kinds apart, and the levels of detail.
report += ["", "## Handed on", ""]
web = os.path.join(out, "web")
os.makedirs(web, exist_ok=True)
rc, lines, seconds = run(["neural-scene", "export", scene, "--by-frame", "--levels", args.levels, "-o",
                          os.path.join(web, "scene")])
for line in lines:
    if re.search(r"level \d+:|gaussians in \d+ parts|cannot be met|not written", line):
        report.append("- export: %s" % line.strip())
index_path = os.path.join(web, "scene.json")
if os.path.exists(index_path):
    index = json.load(open(index_path))
    glass = sum(p.get("glass", {}).get("gaussians", 0) for p in index["parts"])
    mirror = sum(p.get("mirror", {}).get("gaussians", 0) for p in index["parts"])
    report.append("- of %d gaussians, %d are lenses and %d mirrors, set apart for a viewer to shade by the eye" %
                  (index["gaussians"], glass, mirror))
    results["export"] = {"gaussians": index["gaussians"], "glass": glass, "mirror": mirror,
                         "levels": [level["gaussians"] for level in index.get("levels", [])]}
rc, lines, seconds = run(["neural-scene", "bake", scene, "-o", os.path.join(out, "scene.athc")])
report.append("- bake (.athc): rc %d, %s" % (rc, lines[-1].strip() if lines else ""))
rc, lines, seconds = run(["neural-scene", "info", scene])
report.append("- info: %s" % (lines[0].strip() if lines else "rc %d" % rc))

open(os.path.join(out, "report.md"), "w").write("\n".join(report) + "\n")
json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
print("\n".join(report))
