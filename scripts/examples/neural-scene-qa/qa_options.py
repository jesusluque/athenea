#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# WHAT EACH OPTION OF `neural-scene shoot` DOES TO A FOUND SCENE, MEASURED. One stage, one walk, one set of
# held views path traced once; then the walk shot again with one option changed at a time, and each scene
# drawn from the held views and compared there (`athenea compare`, a kernel's). A row an option: whether it
# ran, how long, how many gaussians stand, and the frame where the scene holds something -- with how much of
# the frame that is, since an option that opens holes reads well where it holds.
#
#   qa_options.py --athenea <binary> --stage <dir>/<name>.usda --out <dir> [--size 640x360] [--focal 24]
#                 [--walk walk.txt] [--held held.txt] [--every 2] [--ray-margin M] [--only NAME,NAME]
#
# Nothing here computes on the data: it runs the engine, reads the lines it prints and lays them in a table.
import argparse
import os
import re
import shutil
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--athenea", required=True)
parser.add_argument("--stage", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--size", default="640x360")
parser.add_argument("--focal", type=float, default=24.0)
parser.add_argument("--walk", default="walk.txt")
parser.add_argument("--held", default="held.txt")
parser.add_argument("--every", type=int, default=2, help="one view of the walk in this many")
parser.add_argument("--ray-margin", default="")
parser.add_argument("--reference-samples", type=int, default=128)
parser.add_argument("--reference-bounces", type=int, default=8)
parser.add_argument("--compare-backend", default="")
parser.add_argument("--only", default="", help="names of the rows to run, comma separated (default: all)")
args = parser.parse_args()

stage_dir = os.path.dirname(os.path.abspath(args.stage))
stage_name = os.path.basename(args.stage)
out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "qa.log"), "a")

# The defaults a row changes one of: what `qa.py` shoots with.
BASE = {"--samples": "32", "--pass-samples": "8", "--bounces": "3", "--denoise": None}
ROWS = [
    ("the defaults", {}),
    ("--samples 8", {"--samples": "8"}),
    ("--samples 16", {"--samples": "16"}),
    ("--samples 64", {"--samples": "64"}),
    ("--samples 128", {"--samples": "128"}),
    ("--samples 64 --pass-samples 16", {"--samples": "64", "--pass-samples": "16"}),
    ("--samples 32 --pass-samples 32", {"--pass-samples": "32"}),
    ("--bounces 1", {"--bounces": "1"}),
    ("--bounces 6", {"--bounces": "6"}),
    ("no --denoise", {"--denoise": False}),
    ("--coarsest 1", {"--coarsest": "1"}),
    ("--coarsest 2", {"--coarsest": "2"}),
    ("--coarsest 16", {"--coarsest": "16"}),
    ("--pyramid 8", {"--pyramid": "8"}),
    ("--spread 0.7", {"--spread": "0.7"}),
    ("--spread 1.3", {"--spread": "1.3"}),
    ("--turned 0.75", {"--turned": "0.75"}),
    ("--turned 1", {"--turned": "1"}),
    ("--tolerance 0.02", {"--tolerance": "0.02"}),
    ("--tolerance 0.15", {"--tolerance": "0.15"}),
    ("--refine 1", {"--refine": "1"}),
    ("--refine 3", {"--refine": "3"}),
    ("--repaint 0.3", {"--repaint": "0.3"}),
    ("--covered-enough 0.98", {"--covered-enough": "0.98"}),
    ("--glass-opacity 0.2", {"--glass-opacity": "0.2"}),
    ("--free-each-frame", {"--free-each-frame": None}),
]


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


def lines_of(path):
    return [line for line in open(path) if len(line.split()) >= 7 and not line.startswith("#")]


# The walk thinned, and the held views, beside the results: the stage's directory is not written in.
walk = os.path.join(out, "walk.txt")
open(walk, "w").write("".join(lines_of(os.path.join(stage_dir, args.walk))[::max(args.every, 1)]))
held = os.path.join(out, "held.txt")
open(held, "w").write("".join(lines_of(os.path.join(stage_dir, args.held))))
names = [line.split()[0] for line in lines_of(held)]
margin = ["--ray-margin", args.ray_margin] if args.ray_margin else []

report = ["# %s: the options of shoot, one at a time" % stage_name, ""]
reference = os.path.join(out, "ref")
os.makedirs(reference, exist_ok=True)
if not all(os.path.exists(os.path.join(reference, "%s.exr" % name)) for name in names):
    rc, _, seconds = run(["stage", stage_name, "--cameras", held, "--focal", str(args.focal), "--size", args.size,
                          "--technique", "rt", "--path-total", str(args.reference_samples), "--path-samples", "8",
                          "--path-bounces", str(args.reference_bounces), "--denoise", "-o",
                          os.path.join(reference, "ref.exr")] + margin)
    report.append("- reference: %d views at %s, %d paths, %d bounces, in %.0f s (rc %d)" %
                  (len(names), args.size, args.reference_samples, args.reference_bounces, seconds, rc))
report += ["- the walk: %d views (one in %d of %s)" % (len(lines_of(walk)), args.every, args.walk), "",
           "| option | ended | seconds | gaussians | " + " | ".join("%s: dB, SSIM, held" % n for n in names) + " |",
           "|---|---|---|---|" + "---|" * len(names)]

only = [name.strip() for name in args.only.split(",") if name.strip()]
scene = os.path.join(out, "scene.aths")
for name, change in ROWS:
    if only and name not in only:
        continue
    options = dict(BASE)
    options.update(change)
    shoot = ["neural-scene", "shoot", stage_name, scene, "--views", walk, "--focal", str(args.focal), "--size",
             args.size] + margin
    for key, value in options.items():
        if value is False:
            continue
        shoot += [key] if value is None else [key, value]
    shutil.rmtree(scene, ignore_errors=True)
    rc, lines, seconds = run(shoot)
    stand = ""
    for line in lines:
        m = re.search(r"([0-9]+) gaussians stand", line)
        if m:
            stand = m.group(1)
    cells = []
    if rc == 0:
        run(["neural-scene", "render", scene, "--views", held, "--focal", str(args.focal), "--size", args.size, "-o",
             os.path.join(out, "found.exr")])
        for view in names:
            image = os.path.join(out, "found.exr_%s.exr" % view)
            if not os.path.exists(image):
                image = os.path.join(out, "found_%s.exr" % view)
            _, said, _ = run(["compare", image, os.path.join(reference, "%s.exr" % view), "--mask", image,
                              "--mask-mode", "alpha", "--min-alpha", "0.5"])
            psnr = ssim = cover = "--"
            for line in said:
                m = re.search(r"PSNR ([0-9.]+|inf) dB\s+SSIM ([0-9.]+)", line)
                if m:
                    psnr, ssim = m.group(1), m.group(2)
                m = re.match(r"image\s+mean [0-9.e+-]+ [0-9.e+-]+ [0-9.e+-]+ ([0-9.e+-]+)", line)
                if m:
                    cover = "%.3f" % float(m.group(1))
            cells.append("%s, %s, %s" % (psnr, ssim, cover))
    else:
        cells = [(lines[-1][:60] if lines else "") for _ in names]
    report.append("| `%s` | %s | %.0f | %s | %s |" % (name, "ok" if rc == 0 else "rc %d" % rc, seconds, stand,
                                                     " | ".join(cells)))
    open(os.path.join(out, "report-%s.md" % ("some" if only else "all")), "w").write("\n".join(report) + "\n")
shutil.rmtree(scene, ignore_errors=True)
print("\n".join(report))
