#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# A FOUND SCENE OVER TIME, MEASURED. A stage that moves is shot a frame a time code through its walk
# (`neural-scene shoot --time 0 --time-step 1`); the scene then holds every frame, and the scene as it stood
# at a frame, drawn through that frame's camera (`render --frames`), should be the stage at that time. So a
# few frames are path traced apart at their own time codes and each compared with the stored frame -- the
# figures are `athenea compare`'s. Then the scene is handed on with its history (`export --by-frame
# --history`) and what was written is counted.
#
#   qa_time.py --athenea <binary> --stage <dir>/<name>.usda --out <dir> [--frames 24] [--check 1,6,12,18,24]
import argparse
import json
import os
import re
import shutil
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--athenea", required=True)
parser.add_argument("--stage", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--size", default="960x540")
parser.add_argument("--focal", type=float, default=24.0)
parser.add_argument("--walk", default="walk.txt")
parser.add_argument("--one", default="one.txt")
parser.add_argument("--check", default="1,6,12,18,24")
parser.add_argument("--samples", type=int, default=32)
parser.add_argument("--bounces", type=int, default=3)
parser.add_argument("--skip-shoot", action="store_true")
# `athenea compare` runs its kernels through gpe, which has Metal and CUDA and neither Vulkan nor D3D12: on a
# machine that renders through one of those the measures are taken through this one.
parser.add_argument("--compare-backend", default="")
args = parser.parse_args()

stage_dir = os.path.dirname(os.path.abspath(args.stage))
stage_name = os.path.basename(args.stage)
out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "qa.log"), "a")


def run(arguments):
    began = time.time()
    environment = dict(os.environ)
    if arguments[0] == "compare" and args.compare_backend:
        environment["ATHENEA_BACKEND"] = args.compare_backend
    done = subprocess.run([args.athenea] + arguments, cwd=stage_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", env=environment)
    kept = [line for line in done.stdout.splitlines() if "Node interface" not in line and line.strip()]
    log.write("$ athenea %s\n%s\n(rc %d, %.1f s)\n\n" % (" ".join(arguments), "\n".join(kept), done.returncode,
                                                         time.time() - began))
    log.flush()
    return done.returncode, kept, time.time() - began


def compare(image, reference):
    # Only where the scene holds something: a found scene has no sky.
    _, lines, _ = run(["compare", image, reference, "--mask", image, "--mask-mode", "alpha", "--min-alpha", "0.5"])
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


report = ["# %s, over time" % stage_name, ""]
scene = os.path.join(out, "scene.aths")
if not args.skip_shoot:
    shutil.rmtree(scene, ignore_errors=True)
    rc, lines, seconds = run(["neural-scene", "shoot", stage_name, scene, "--views", os.path.join(stage_dir, args.walk),
                              "--time", "0", "--time-step", "1", "--focal", str(args.focal), "--size", args.size,
                              "--samples", str(args.samples), "--pass-samples", "8", "--bounces", str(args.bounces),
                              "--denoise"])
    frames = [line for line in lines if ": frame " in line]
    report.append("- shoot: %d frames in %.0f s (%.1f s a frame, rc %d)" % (len(frames), seconds,
                                                                           seconds / max(len(frames), 1), rc))
    report += ["", "## What each frame took in", "", "| frame | covered before a path | made | gone | refined | standing |",
               "|---|---|---|---|---|---|"]
    looks = [line for line in lines if "cells of 1:" in line]
    for n, line in enumerate(frames):
        covered = re.search(r"covered ([0-9.]+)% of it", line)
        standing = re.search(r"(\d+) gaussians stand", line)
        look = looks[n] if n < len(looks) else ""
        made = re.search(r"(\d+) gaussians made", look)
        gone = re.search(r"(\d+) gone", look)
        refined = re.search(r"(\d+) refined", look)
        report.append("| %d | %s%% | %s | %s | %s | %s |" % (n + 1, covered.group(1) if covered else "--",
                                                           made.group(1) if made else "--", gone.group(1) if gone else "--",
                                                           refined.group(1) if refined else "--",
                                                           standing.group(1) if standing else "--"))

# The stored frames, drawn again each from the scene as it stood.
checks = [int(v) for v in args.check.split(",")]
last = max(checks)
rc, lines, seconds = run(["neural-scene", "render", scene, "--frames", "1", str(last), "-o", os.path.join(out, "stored.exr")])
report += ["", "- the %d stored frames drawn again: rc %d, %.0f s" % (last, rc, seconds), "",
           "## The scene as it stood, against the stage at that time", "",
           "| frame | time code | PSNR dB | SSIM | relMSE |", "|---|---|---|---|---|"]
results = {}
for n in checks:
    reference_dir = os.path.join(out, "ref_%02d" % n)
    os.makedirs(reference_dir, exist_ok=True)
    run(["stage", stage_name, "--time", str(n - 1), "--cameras", os.path.join(stage_dir, args.one), "--focal",
         str(args.focal), "--size", args.size, "--technique", "rt", "--path-total", "128", "--path-samples", "8",
         "--path-bounces", "8", "--denoise", "-o", os.path.join(reference_dir, "ref.exr")])
    reference = os.path.join(reference_dir, "t.exr")
    stored = os.path.join(out, "stored_%04d.exr" % n)
    m = compare(stored, reference) if os.path.exists(reference) and os.path.exists(stored) else {}
    results[n] = m
    report.append("| %d | %d | %s | %s | %s |" % (n, n - 1, "%.2f" % m["psnr"] if "psnr" in m else "--",
                                                 "%.3f" % m["ssim"] if "ssim" in m else "--",
                                                 "%.4f" % m["relmse"] if "relmse" in m else "--"))
# And the last frame against the first's reference: what the measure reads when the scene did not follow.
first = os.path.join(out, "ref_%02d" % checks[0], "t.exr")
stale = compare(os.path.join(out, "stored_%04d.exr" % last), first) if os.path.exists(first) else {}
if "psnr" in stale:
    report.append("")
    report.append("- for scale, frame %d against the stage at frame %d's time: %.2f dB" % (last, checks[0], stale["psnr"]))

web = os.path.join(out, "web")
os.makedirs(web, exist_ok=True)
rc, lines, seconds = run(["neural-scene", "export", scene, "--by-frame", "--history", "-o", os.path.join(web, "scene")])
report += ["", "## Handed on with its history", ""]
index_path = os.path.join(web, "scene.json")
if os.path.exists(index_path):
    index = json.load(open(index_path))
    standing = sum(p["gaussians"] + p.get("glass", {}).get("gaussians", 0) + p.get("mirror", {}).get("gaussians", 0)
                   for p in index["parts"])
    history = [h for p in index["parts"] for h in p.get("history", [])]
    past = sum(h["gaussians"] + h.get("glass", {}).get("gaussians", 0) + h.get("mirror", {}).get("gaussians", 0) for h in history)
    files = len([f for f in os.listdir(web) if f.endswith(".spz")])
    report.append("- %d parts; %d gaussians stand to the end and %d stood for a while, in %d files of history; %d files, %.1f MB"
                  % (len(index["parts"]), standing, past, len(history), files,
                     sum(os.path.getsize(os.path.join(web, f)) for f in os.listdir(web)) / 1048576.0))
else:
    report.append("- export failed (rc %d): %s" % (rc, lines[-1] if lines else ""))
open(os.path.join(out, "report.md"), "w").write("\n".join(report) + "\n")
json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
print("\n".join(report))
