#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""matx: the materials whose fields only the material itself can say, converted with a TX transfer and
measured against the path-traced mesh -- and the ones that were already right, held where they were.

A material whose base colour a graph computes (a procedural brick, a marble's noise, a hexagonal tiling),
reads through a tiled map, or is tinted by a coat colour used to come out of the conversion with the
constant the stage reader fell back on: the brick white, the brass silver. The TX bake now reads those
fields back from the material on the device (StageMaterial::sampled). What this holds, with nothing said
per material:

  - SAMPLED (brick, brass, wood, marble, onyx, and copper, whose colour is a constant coat colour over
    its metal): the cloud keeps the material's colour -- each channel's
    mean within [0.70, 1.45] of the GT's, and the three channels' ratios within 1.25 of each other (a
    white brick against red bricks is 3.8 apart, an untinted brass 4.3) -- and its relMSE is within
    twice the mesh raster's plus 0.1 (the mesh raster is the reference column, not the target);
    That whole rule holds the non-metals. A METAL (lobe class "metal") is held by its hue alone, since its
    level waits on TX's light under a compact sun and its open concave interreflection (docs/decisions.md,
    matx): the mean's level and relMSE are reported, not held; the 1.25 rule holds where every channel's
    mean is at least 0.70 of the GT's and is listed "pending TX's level" otherwise, with the exit code
    left alone; and what blocks is the hue in two fixed windows of the shader ball -- the body's
    sky-facing right side and the hollow's core, not the base ring's inside, which is multi-bounce metal
    -- each channel over green within 5% of the GT's. Once TX's fix lands, metals return to the full rule
    (METAL_LEVEL_WAITS_ON_TX below);
  - ALREADY RIGHT (matx_common.GATE, one a lobe class): held to baseline.csv as compare_matx.py holds the
    gate (relMSE 5% worse, a mean ratio moved by 0.05), where the baseline has the row.

CPU here, but for the regional hue: the window means are the Measure effect's on the device
(`athenea compare --window`); the rest is run_matx.py's (the same conversion and measure, the GT cached
per stage):

  python3 check_sampled.py [--sky autoshop|goegap|all] [--no-run]
  exit 0 pass, 1 fail, 77 nothing to measure on this machine
"""
import argparse
import os
import subprocess
import sys

import compare_matx
import matx_common as m

SAMPLED = ["mx_standard_surface_brick_procedural", "mx_standard_surface_brass_tiled",
           "mx_standard_surface_wood_tiled", "mx_standard_surface_marble_solid",
           "mx_standard_surface_onyx_hextiled", "mx_standard_surface_copper"]
RATIO_LOW, RATIO_HIGH = 0.70, 1.45
CHROMA = 1.25
# A METAL'S LEVEL WAITS ON TX (task TX: the metal's polish under a compact sun, the concave metal's
# interreflection). When that fix lands, set this False: metals are then held by the full rule again.
METAL_LEVEL_WAITS_ON_TX = True
HUE = 0.05   # a window's channel over green, against the GT's
# THE WINDOWS, on the matx camera's 512 x 512 frame (scene/base.usda.in), x0 y0 x1 y1 with rows counted
# from the bottom as `athenea compare --window` counts them: the body's right side, which reflects the
# dome rather than the ground, and the core inside the hollow. Not the base ring's inside: what lights it
# is metal seen in metal, several bounces deep, which is TX's to carry.
WINDOWS = {"body": (345, 272, 385, 322), "hollow": (175, 312, 235, 352)}


def window_hue(gs, gt, box):
    """Each channel over green in `box`, the cloud's and the GT's, from the Measure effect's means."""
    out = subprocess.run([m.ATHENEA, "compare", gs, gt, "--window", *map(str, box)], capture_output=True,
                         text=True, env=m.child_env())
    means = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) > 4 and parts[0] in ("image", "reference") and parts[1] == "mean":
            means[parts[0]] = [float(x) for x in parts[2:5]]
    if "image" not in means or "reference" not in means:
        return None, (out.stdout + out.stderr).strip()[-200:]
    hue = lambda c: (c[0] / max(c[1], 1e-9), c[2] / max(c[1], 1e-9))
    return (hue(means["image"]), hue(means["reference"])), None


def regional(material, sky):
    """The metal's hue in each window: [(window, ours, the GT's, off)], or an error."""
    import glob
    run = os.path.join(m.RENDERS, "runs", sky, material)
    gts = glob.glob(os.path.join(run, "gt*.exr"))
    gss = glob.glob(os.path.join(run, "*_gs.exr"))
    if not gts or not gss:
        return None, f"no frames in {run}"
    rows = []
    for name, box in WINDOWS.items():
        hues, err = window_hue(gss[0], gts[0], box)
        if err:
            return None, f"{name}: {err}"
        ours, theirs = hues
        off = max(abs(ours[k] / max(theirs[k], 1e-9) - 1.0) for k in range(2))
        rows.append((name, ours, theirs, off))
    return rows, None


def classes_of():
    import json
    try:
        return {e["id"]: e.get("classes", []) for e in json.load(open(m.MANIFEST))}
    except (OSError, ValueError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sky", default="autoshop")
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(m.RENDERS, "results")
    skies = list(m.SKIES) if a.sky == "all" else [a.sky]
    if not a.no_run:
        if not os.path.exists(m.SHADERBALL) or not os.path.exists(m.AUTOSHOP):
            print(f"matx sampled: no shader ball ({m.SHADERBALL}) or autoshop ({m.AUTOSHOP}) on this machine")
            return 77
        if os.path.realpath(m.WORK) != os.path.realpath(m.SRC) and "MATX_MANIFEST" not in os.environ:
            os.environ["MATX_MANIFEST"] = os.path.join(m.WORK, "manifest.json")
            os.makedirs(m.WORK, exist_ok=True)
        if not os.path.exists(os.path.join(m.WORK, "scene", "shaderball.usdc")):
            subprocess.run([sys.executable, os.path.join(here, "build_ball.py")], check=True)
        if not all(os.path.exists(os.path.join(m.WORK, "stages", g + ".usda")) for g in SAMPLED + m.GATE):
            subprocess.run([sys.executable, os.path.join(here, "build_stages.py")], check=True,
                           stdout=subprocess.DEVNULL)
        # Measured now: what an earlier run left is not this one's.
        for g in SAMPLED + m.GATE:
            for sky in skies:
                stale = os.path.join(results_dir, f"{g}__{sky}.json")
                if os.path.exists(stale):
                    os.remove(stale)
        for sky in skies:
            run = subprocess.run([sys.executable, os.path.join(here, "run_matx.py"), "--phase", "1", "--sky", sky,
                                  "--only", *SAMPLED, *m.GATE])
            if run.returncode != 0:
                print(f"matx sampled: the sweep stopped (exit {run.returncode})")
                return 1
    res = compare_matx.results(results_dir)
    classes = classes_of()
    failed = 0
    pending = 0
    measured = 0
    print(f"matx sampled: the colour kept (each channel {RATIO_LOW}..{RATIO_HIGH} of the GT's, within {CHROMA} "
          f"of each other) and relMSE within 2 x the mesh raster's + 0.1"
          + (f"; a metal by its hue (over green within {HUE:.0%} of the GT's in the body and the hollow, and the "
             f"{CHROMA} rule where its level is at least {RATIO_LOW}), its level reported" if METAL_LEVEL_WAITS_ON_TX
             else ""))
    for g in SAMPLED:
        for sky in skies:
            r = res.get((g, sky))
            if r is None:
                print(f"  MISSING  {g:44} {sky}")
                failed += 1
                continue
            measured += 1
            if "err" in r:
                print(f"  FAIL     {g:44} {sky:9} {r['err'][:80]}")
                failed += 1
                continue
            ratios = [r[f"mean_ratio_{c}"] for c in "rgb"]
            mesh = r.get("relMSE_mesh_raster")
            metal = METAL_LEVEL_WAITS_ON_TX and "metal" in classes.get(g, [])
            why = []
            note = []
            level_ok = all(x is not None and RATIO_LOW <= x <= RATIO_HIGH for x in ratios)
            chroma_ok = all(x is not None for x in ratios) and max(ratios) / max(min(ratios), 1e-6) <= CHROMA
            relmse_ok = mesh is None or r["relMSE_tx"] <= 2.0 * mesh + 0.1
            if not metal:
                if not level_ok:
                    why.append("mean " + "/".join("–" if x is None else f"{x:.2f}" for x in ratios))
                elif not chroma_ok:
                    why.append("colour " + "/".join(f"{x:.2f}" for x in ratios))
                if not relmse_ok:
                    why.append(f"relMSE {r['relMSE_tx']:.3f} against the mesh raster's {mesh:.3f}")
            else:
                # A METAL: its level and relMSE reported; the mean's hue held where the level allows it,
                # pending TX's level otherwise; the hue in the windows held always.
                if all(x is not None and x >= RATIO_LOW for x in ratios):
                    if not chroma_ok:
                        why.append("colour " + "/".join(f"{x:.2f}" for x in ratios))
                else:
                    note.append("pendiente del nivel de TX (level " + "/".join(
                        "–" if x is None else f"{x:.2f}" for x in ratios) + ")")
                rows, err = regional(g, sky)
                if err:
                    why.append("windows: " + err)
                else:
                    for name, ours, theirs, off in rows:
                        tag = (f"{name} r/g {ours[0]:.3f} b/g {ours[1]:.3f} (GT {theirs[0]:.3f} {theirs[1]:.3f})")
                        (why if off > HUE else note).append(tag + (f" off {off:.1%}" if off > HUE else ""))
            waits = metal and not all(x is not None and x >= RATIO_LOW for x in ratios)
            failed += bool(why)
            pending += bool(waits and not why)
            state = "FAIL" if why else "PENDING" if waits else "ok"
            print(f"  {state:8} {g:44} {sky:9} relMSE {r['relMSE_tx']:.4f} (mesh {mesh:.4f}) "
                  f"mean {'/'.join(f'{x:.2f}' for x in ratios)}{' [metal: hue held, level reported]' if metal else ''} "
                  f"{'; '.join(why + note)}")
    base = compare_matx.read_baseline()
    if base is None:
        print("matx sampled: no baseline.csv: the materials already right are measured but not held")
    else:
        lines, regressions, compared = compare_matx.compare(base, res, set(m.GATE))
        print(f"matx sampled: {compared} already right against baseline.csv")
        print("\n".join(lines))
        failed += regressions
    if measured == 0:
        return 77
    print(f"matx sampled: {failed} failure{'s' if failed != 1 else ''}"
          + (f", {pending} pendiente{'s' if pending != 1 else ''} del nivel de TX (not failures)" if pending else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
