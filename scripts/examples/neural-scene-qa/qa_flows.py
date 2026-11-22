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
#               [--size 640x360] [--compare-backend cuda] [--group-light <prim>] [--matte-prim <prim>]
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
parser.add_argument("--group-light", default="", help="the prim path of the stage's light the group workflows switch")
parser.add_argument("--matte-prim", default="", help="the prim path the matte workflows ask for (default: the first the scene shot names)")
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
    # What a scene shot now can be lit with after: lamps added with no path traced, and its light taken
    # again by a second pass over the same views (a layer of light a frame, beside the pieces).
    shot = os.path.join(out, "shot.aths")
    if cameras and os.path.exists(os.path.join(shot, "scene.json")):
        eye = cameras[0]["eye"]
        target = cameras[0]["target"]
        reach = sum((e - t) ** 2 for e, t in zip(eye, target)) ** 0.5
        lamps = os.path.join(out, "lamps.txt")
        open(lamps, "w").write("sphere %g %g %g %g 40 1 0.8 0.6\nspot %g %g %g %g %g %g 30 %g 80 0.6 0.8 1\n" %
                               (eye[0], eye[1], eye[2], 0.03 * reach, eye[0], eye[1], eye[2], target[0], target[1],
                                target[2], 0.02 * reach))
        run("render --lamps", ["neural-scene", "render", shot, "--views", views, "--lamps", lamps] + common +
            ["-o", os.path.join(out, "lamps")], [os.path.join(out, "lamps_*.exr")])
        run("render --lamps --lamps-only, one ray, no bounce",
            ["neural-scene", "render", shot, "--views", views, "--lamps", lamps, "--lamps-only", "--lamp-shadow-rays", "1",
             "--lamp-bounce", "0"] + common + ["-o", os.path.join(out, "lampsonly")], [os.path.join(out, "lampsonly_*.exr")])
        run("shoot --relight, over the scene shot", ["neural-scene", "shoot", os.path.basename(args.stage), shot, "--views",
            short, "--samples", "8", "--pass-samples", "8", "--bounces", "2", "--relight"] + common,
            [os.path.join(shot, "pieces", "*.athg.l*")], cwd=stage_dir)
        run("render --frames, after its light was taken again", ["neural-scene", "render", shot, "--frames", "1", "6", "-o",
            os.path.join(out, "relit.png")], [os.path.join(out, "relit_*.png")])
        run("render --snapshot 1, before it was", ["neural-scene", "render", shot, "--snapshot", "1"] + view + common +
            ["-o", os.path.join(out, "before.exr")], [os.path.join(out, "before.exr")])
    # A MATTE BY PRIM: the scene shot now keeps which prim each gaussian was made from. A matte of one of them
    # (--matte-prim, or the first the scene names) from a view, through the walk with glass, and the stage's
    # own matte of the same prim from the same view (its Cryptomatte), the two measured against each other.
    named = json.load(open(os.path.join(shot, "scene.json"))).get("prims", {}) if os.path.exists(
        os.path.join(shot, "scene.json")) else {}
    prim = args.matte_prim or (sorted(named)[0] if named else "")
    if cameras and prim:
        matte = os.path.join(out, "matte.exr")
        run("render --matte", ["neural-scene", "render", shot, "--matte", "one=" + prim] + view + common + ["-o", matte],
            [os.path.join(out, "matte_matte_one.exr")])
        run("render --matte --views --glass", ["neural-scene", "render", shot, "--views", views, "--glass", "--matte",
            "one=" + prim, "--matte", "all=/"] + common + ["-o", os.path.join(out, "mglass")],
            [os.path.join(out, "mglass_*_matte_one.exr"), os.path.join(out, "mglass_*_matte_all.exr")])
        run("render --matte, as a picture", ["neural-scene", "render", shot, "--matte", "one=" + prim] + view + common +
            ["-o", os.path.join(out, "mattepic.png")], [os.path.join(out, "mattepic_matte_one.png")])
        c = cameras[0]
        run("stage --matte, the stage's own", ["stage", os.path.basename(args.stage), "--eye"] + [str(v) for v in c["eye"]] +
            ["--target"] + [str(v) for v in c["target"]] + ["--up"] + [str(v) for v in c.get("up", [0, 1, 0])] +
            common + ["--matte", "one=" + prim, "-o", os.path.join(out, "stagematte.exr")],
            [os.path.join(out, "stagematte_matte_one.exr")], cwd=stage_dir)
        if os.path.exists(os.path.join(out, "stagematte_matte_one.exr")) and os.path.exists(
                os.path.join(out, "matte_matte_one.exr")):
            run("compare, the scene's matte against the stage's", ["compare", os.path.join(out, "matte_matte_one.exr"),
                os.path.join(out, "stagematte_matte_one.exr")], [])
        run("render --matte of a prim the scene does not name (refused)", ["neural-scene", "render", shot, "--matte",
            "none=/No/Such/Prim"] + view + common + ["-o", os.path.join(out, "nomatte.exr")], [])
        if rows[-1][1].startswith("FAILED"):
            rows[-1] = (rows[-1][0], "ok (refused)") + rows[-1][2:]
        else:
            rows[-1] = (rows[-1][0], "FAILED (not refused)") + rows[-1][2:]
        # THE IDS HANDED ON: `export` writes, beside every file of the scene's, which prim each of its gaussians
        # was made from (an .ids file, sixteen bits a gaussian inside gzip) and names the prims in the index. Checked by
        # what the index says against the files' sizes -- the part's own, its glass's, its mirrors', a
        # level's and a history's -- and the same export asked without them (--no-prims) leaves none.
        ids = os.path.join(out, "ids")
        for stale in glob.glob(ids + "*"):
            os.remove(stale)
        run("export, an id a gaussian (--by-frame --history --levels)", ["neural-scene", "export", shot, "--by-frame",
            "--history", "--levels", "20000,5000", "-o", ids], [ids + "_*.ids", ids + ".json"])
        if os.path.exists(ids + ".json"):
            handed = json.load(open(ids + ".json"))
            wrong = []
            files = 0

            def check(entry):
                global files
                for part in (entry, entry.get("glass") or {}, entry.get("mirror") or {}):
                    if not part.get("file"):
                        continue
                    at = os.path.join(out, part.get("ids") or "(none)")
                    # The file is gzip, whose last four bytes say how long what it holds is: two bytes a gaussian.
                    if not os.path.isfile(at) or int.from_bytes(open(at, "rb").read()[-4:], "little") != 2 * part.get("gaussians", 0):
                        wrong.append(part["file"])
                    files += 1

            for entry in handed["parts"] + [p for level in handed.get("levels", []) for p in level["parts"]]:
                check(entry)
                for span in entry.get("history", []):
                    check(span)
            if len(handed.get("prims", [])) != len(named):
                wrong.append("prims: %d in the index, %d in the scene" % (len(handed.get("prims", [])), len(named)))
            if wrong or files == 0:
                rows[-1] = (rows[-1][0], "FAILED (ids: %s)" % ", ".join(wrong[:3] or ["no file"])) + rows[-1][2:]
            else:
                rows[-1] = rows[-1][:4] + (("%d files each with its ids, %d prims named; " % (files, len(named))) + rows[-1][4],)
        noids = os.path.join(out, "noids")
        for stale in glob.glob(noids + "*"):
            os.remove(stale)
        run("export --no-prims", ["neural-scene", "export", shot, "--no-prims", "-o", noids], [noids + "_*.spz", noids + ".json"])
        if glob.glob(noids + "*.ids") or (os.path.exists(noids + ".json") and "prims" in json.load(open(noids + ".json"))):
            rows[-1] = (rows[-1][0], "FAILED (ids written)") + rows[-1][2:]
    # A GROUP OF LIGHTS: the scene shot without one of the stage's lights, that light's own layer kept by a
    # second pass under it alone, and the scene drawn with the group at a level.
    lamp = args.group_light
    if lamp:
        based = os.path.join(out, "based.aths")
        run("shoot --lights-off", ["neural-scene", "shoot", os.path.basename(args.stage), based, "--views", short,
            "--samples", "8", "--pass-samples", "8", "--bounces", "2", "--lights-off", lamp] + common,
            [os.path.join(based, "scene.json")], cwd=stage_dir)
        run("shoot --light-group --lights-on", ["neural-scene", "shoot", os.path.basename(args.stage), based, "--views", short,
            "--samples", "8", "--pass-samples", "8", "--bounces", "2", "--light-group", "lamp", "--lights-on", lamp] + common,
            [os.path.join(based, "pieces", "*.g.lamp.1")], cwd=stage_dir)
        run("render --group lamp=0.5", ["neural-scene", "render", based, "--views", views, "--group", "lamp=0.5"] + common +
            ["-o", os.path.join(out, "group")], [os.path.join(out, "group_*.exr")])
        run("render --group lamp=1,0.2,0.2 --frames", ["neural-scene", "render", based, "--frames", "1", "3", "--group",
            "lamp=1,0.2,0.2", "-o", os.path.join(out, "grouped.png")], [os.path.join(out, "grouped_*.png")])
        run("export --group lamp=0.5", ["neural-scene", "export", based, "--group", "lamp=0.5", "-o",
            os.path.join(out, "xgroup")], [os.path.join(out, "xgroup_*.spz"), os.path.join(out, "xgroup.json")])
        run("export --own-light 0, the group alone", ["neural-scene", "export", based, "--own-light", "0", "--group",
            "lamp=1", "-o", os.path.join(out, "xalone")], [os.path.join(out, "xalone_*.spz")])
        # THE SAME GROUP FOUND WITH THE SCENE, by the same paths: one shoot, the group's layer beside the pieces.
        once = os.path.join(out, "once.aths")
        run("shoot --light-groups", ["neural-scene", "shoot", os.path.basename(args.stage), once, "--views", short,
            "--samples", "8", "--pass-samples", "8", "--bounces", "2", "--light-groups", "lamp=" + lamp] + common,
            [os.path.join(once, "scene.json"), os.path.join(once, "pieces", "*.g.lamp.1")], cwd=stage_dir)
        # AND ADDED TO A SCENE ALREADY FOUND, by their planes of one path tracing of each view.
        run("shoot --light-groups --groups-only", ["neural-scene", "shoot", os.path.basename(args.stage), based, "--views", short,
            "--samples", "8", "--pass-samples", "8", "--bounces", "2", "--light-groups", "added=" + lamp, "--groups-only",
            "--light-over-base"] + common,
            [os.path.join(based, "pieces", "*.g.added.1")], cwd=stage_dir)
        run("render --group lamp=0, a scene shot with its groups", ["neural-scene", "render", once, "--views", views,
            "--group", "lamp=0"] + common + ["-o", os.path.join(out, "once")], [os.path.join(out, "once_*.exr")])
    # THE FOUND SCENE AS WHAT A PATH'S BOUNCE MEETS: a shoot of one bounce that reads the scene's own light.
    run("shoot --scene-cache --bounces 1", ["neural-scene", "shoot", os.path.basename(args.stage),
        os.path.join(out, "cached.aths"), "--views", short, "--samples", "8", "--pass-samples", "8", "--bounces", "1",
        "--scene-cache", "64"] + common, [os.path.join(out, "cached.aths", "scene.json")], cwd=stage_dir)
    run("shoot --time-step --repaint --free-each-frame", ["neural-scene", "shoot", os.path.basename(args.stage),
        os.path.join(out, "timed.aths"), "--views", short, "--time", "0", "--time-step", "1", "--repaint", "0.3",
        "--free-each-frame", "--samples", "8", "--pass-samples", "8", "--bounces", "2"] + common,
        [os.path.join(out, "timed.aths", "scene.json")], cwd=stage_dir)
    # PARENTS OF WHAT STANDS REFINED: a shoot whose gaussians name no parent that may yet go, and a level of it.
    run("shoot --laid-over-refined", ["neural-scene", "shoot", os.path.basename(args.stage),
        os.path.join(out, "laid.aths"), "--views", short, "--time", "0", "--time-step", "1", "--laid-over-refined",
        "--samples", "8", "--pass-samples", "8", "--bounces", "2"] + common,
        [os.path.join(out, "laid.aths", "scene.json")], cwd=stage_dir)
    run("export --levels, a scene shot with --laid-over-refined", ["neural-scene", "export",
        os.path.join(out, "laid.aths"), "--by-frame", "--levels", "20000", "-o", os.path.join(out, "laid")],
        [os.path.join(out, "laid.json")])

report = ["# Every way through neural-scene: %s" % os.path.basename(os.path.normpath(scene)), "",
          "| workflow | ended | seconds | left | said |", "|---|---|---|---|---|"]
report += ["| %s |" % " | ".join(row) for row in rows]
failed = [row[0] for row in rows if row[1].startswith("FAILED")]
report += ["", "%d of %d ended well%s" % (len(rows) - len(failed), len(rows), "; failed: " + ", ".join(failed) if failed else "")]
open(os.path.join(out, "report.md"), "w").write("\n".join(report) + "\n")
print("\n".join(report))
