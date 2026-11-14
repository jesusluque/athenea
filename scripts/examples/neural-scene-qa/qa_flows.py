#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# EVERY WAY THROUGH `neural-scene`, RUN. A found scene (one `qa.py` left, or any `shoot` made) is taken through
# each subcommand and option that reads it -- described, drawn every way, baked, handed on in every format --
# and, where a stage is given, a short shoot that grows a file while it renders. What is checked is that each
# ends well and leaves what it says it leaves; what each printed is kept in the log. A table of them is the
# report: a workflow that broke is a row that says so.
#
#   qa_flows.py --athenea <binary> --scene <dir>.aths --out <dir> [--stage <file> --walk <file>]
#               [--size 640x360] [--compare-backend cuda]
import argparse
import glob
import json
import os
import shutil
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--athenea", required=True)
parser.add_argument("--scene", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--stage", default="")
parser.add_argument("--walk", default="")
parser.add_argument("--size", default="640x360")
parser.add_argument("--focal", default="24")
parser.add_argument("--rotate-x", default="")
args = parser.parse_args()

out = os.path.abspath(args.out)
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
log = open(os.path.join(out, "flows.log"), "w")
manifest = json.load(open(os.path.join(args.scene, "scene.json")))
cameras = manifest.get("cameras", [])
frames = manifest.get("frames", [])
snapshots = manifest.get("snapshots", [])
rows = []


def run(name, arguments, leaves, cwd=None):
    began = time.time()
    done = subprocess.run([args.athenea] + arguments, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    kept = [line for line in done.stdout.splitlines() if "Node interface" not in line and line.strip()]
    log.write("## %s\n$ athenea %s\n%s\n(rc %d, %.1f s)\n\n" % (name, " ".join(arguments), "\n".join(kept), done.returncode,
                                                               time.time() - began))
    log.flush()
    found = [p for pattern in leaves for p in glob.glob(pattern)]
    size = sum(os.path.getsize(p) for p in found if os.path.isfile(p))
    ok = done.returncode == 0 and (not leaves or len(found) > 0)
    said = next((line.strip() for line in reversed(kept) if "info]" not in line), "")
    # A build that was made without a library says so and is not a workflow that broke.
    without = "this build reads no" in said or "this build writes no" in said
    rows.append((name, "ok" if ok else "not in this build" if without else "FAILED (rc %d)" % done.returncode, "%.1f" % (time.time() - began),
                 "%d files, %.1f MB" % (len(found), size / 1048576.0) if leaves else "--", said[:110]))
    return ok or without


# A view to draw from: the first camera the scene keeps, and a file of two of them.
view = []
if cameras:
    c = cameras[0]
    view = ["--eye"] + [str(v) for v in c["eye"]] + ["--target"] + [str(v) for v in c["target"]]
views = os.path.join(out, "views.txt")
open(views, "w").write("".join("%s %s %s\n" % (c["name"], " ".join(str(v) for v in c["eye"]), " ".join(str(v) for v in c["target"]))
                              for c in cameras[:2]))
common = ["--focal", args.focal, "--size", args.size]
scene = args.scene

run("info", ["neural-scene", "info", scene], [])
run("render, a view", ["neural-scene", "render", scene] + view + common + ["-o", os.path.join(out, "view.exr")],
    [os.path.join(out, "view.exr")])
run("render, a view as a picture", ["neural-scene", "render", scene] + view + common + ["-o", os.path.join(out, "view.png")],
    [os.path.join(out, "view.png")])
run("render --views", ["neural-scene", "render", scene, "--views", views] + common + ["-o", os.path.join(out, "walk")],
    [os.path.join(out, "walk_*.exr")])
run("render --glass", ["neural-scene", "render", scene, "--views", views, "--glass"] + common + ["-o", os.path.join(out, "glass")],
    [os.path.join(out, "glass_*.exr")])
run("render --lod 2", ["neural-scene", "render", scene, "--lod", "2"] + view + common + ["-o", os.path.join(out, "lod.exr")],
    [os.path.join(out, "lod.exr")])
run("render --lod 2 --morph 0.5", ["neural-scene", "render", scene, "--lod", "2", "--morph", "0.5"] + view + common +
    ["-o", os.path.join(out, "morph.exr")], [os.path.join(out, "morph.exr")])
run("render --budget 100000", ["neural-scene", "render", scene, "--budget", "100000"] + view + common +
    ["-o", os.path.join(out, "budget.exr")], [os.path.join(out, "budget.exr")])
# The scene as it stood: at the last look of its first frame (a look taken while a frame gathers is not a
# time the scene stood at: its gaussians are the frame's own drafts).
then = str(frames[0]["lastSnapshot"]) if frames else "1"
if snapshots:
    run("render --snapshot %s" % then, ["neural-scene", "render", scene, "--snapshot", then] + view + common +
        ["-o", os.path.join(out, "snapshot.exr")], [os.path.join(out, "snapshot.exr")])
    run("render --at 0", ["neural-scene", "render", scene, "--at", str(snapshots[0].get("time", 0))] + view + common +
        ["-o", os.path.join(out, "at.exr")], [os.path.join(out, "at.exr")])
if frames:
    run("render --frame 1", ["neural-scene", "render", scene, "--frame", "1", "-o", os.path.join(out, "frame.exr")],
        [os.path.join(out, "frame.exr")])
    last = min(len(frames), 3)
    run("render --frames 1 %d" % last, ["neural-scene", "render", scene, "--frames", "1", str(last), "-o",
                                       os.path.join(out, "stored.png")], [os.path.join(out, "stored_*.png")])
run("bake", ["neural-scene", "bake", scene, "-o", os.path.join(out, "baked.athc")], [os.path.join(out, "baked.athc")])
if os.path.exists(os.path.join(out, "baked.athc")):
    run("render of the bake (.athc)", ["render", "--splats", os.path.join(out, "baked.athc"), "--lod", "1"] + view + common +
        ["-o", os.path.join(out, "baked.exr")], [os.path.join(out, "baked.exr")])
for form in ("spz", "ply", "glb"):
    run("export --format %s" % form, ["neural-scene", "export", scene, "--format", form, "-o", os.path.join(out, "x_" + form)],
        [os.path.join(out, "x_%s_*.%s" % (form, form)), os.path.join(out, "x_%s.json" % form)])
for form, turn in (("spz", ["--rotate-x", "180"]), ("ply", [])):
    first = os.path.join(out, "x_%s_000.%s" % (form, form))
    if os.path.exists(first):
        run("render of the %s handed on" % form, ["render", "--splats", first] + turn + view + common +
            ["-o", os.path.join(out, "back_%s.exr" % form)], [os.path.join(out, "back_%s.exr" % form)])
if frames:
    run("export --by-frame", ["neural-scene", "export", scene, "--by-frame", "-o", os.path.join(out, "byframe")],
        [os.path.join(out, "byframe_*.spz"), os.path.join(out, "byframe.json")])
    run("export --by-frame --history", ["neural-scene", "export", scene, "--by-frame", "--history", "-o",
                                       os.path.join(out, "history")],
        [os.path.join(out, "history_*.spz"), os.path.join(out, "history.json")])
run("export --levels", ["neural-scene", "export", scene, "--levels", "200000,50000,10000", "-o", os.path.join(out, "levels")],
    [os.path.join(out, "levels*.spz"), os.path.join(out, "levels.json")])
run("export --part 20000", ["neural-scene", "export", scene, "--part", "20000", "-o", os.path.join(out, "parts")],
    [os.path.join(out, "parts_*.spz")])
run("export --snapshot %s" % then, ["neural-scene", "export", scene, "--snapshot", then, "-o", os.path.join(out, "then")],
    [os.path.join(out, "then_*.spz")])
if args.stage and args.walk:
    # A short shoot of its own, growing a file as it renders, and the same with what moved asked about.
    stage_dir = os.path.dirname(os.path.abspath(args.stage))
    short = os.path.join(out, "short.txt")
    open(short, "w").write("".join(open(args.walk).readlines()[:3]))
    grown = os.path.join(out, "grow")
    run("shoot --grow", ["neural-scene", "shoot", os.path.basename(args.stage), os.path.join(out, "shot.aths"), "--views", short,
                        "--samples", "8", "--pass-samples", "8", "--bounces", "2", "--grow", grown] + common,
        [grown + "_*.spz", grown + ".json"], cwd=stage_dir)
    run("shoot --time-step --repaint --free-each-frame", ["neural-scene", "shoot", os.path.basename(args.stage),
        os.path.join(out, "timed.aths"), "--views", short, "--time", "0", "--time-step", "1", "--repaint", "0.3",
        "--free-each-frame", "--samples", "8", "--pass-samples", "8", "--bounces", "2"] + common,
        [os.path.join(out, "timed.aths", "scene.json")], cwd=stage_dir)

report = ["# Every way through neural-scene: %s" % os.path.basename(os.path.normpath(scene)), "",
          "| workflow | ended | seconds | left | said |", "|---|---|---|---|---|"]
report += ["| %s |" % " | ".join(row) for row in rows]
failed = [row[0] for row in rows if row[1].startswith("FAILED")]
report += ["", "%d of %d ended well%s" % (len(rows) - len(failed), len(rows), "; failed: " + ", ".join(failed) if failed else "")]
open(os.path.join(out, "report.md"), "w").write("\n".join(report) + "\n")
print("\n".join(report))
