#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""matx: the materials whose fields only the material itself can say, converted with a TX transfer and
measured against the path-traced mesh -- and the ones that were already right, held where they were.

A material whose base colour a graph computes (a procedural brick, a marble's noise, a hexagonal tiling),
reads through a tiled map, or is tinted by a coat colour used to come out of the conversion with the
constant the stage reader fell back on: the brick white, the brass silver. The TX bake now reads those
fields back from the material on the device (StageMaterial::sampled). What this holds, with nothing said
per material:

  - SAMPLED (brick, brass, wood, marble, onyx): the cloud keeps the material's colour -- each channel's
    mean within [0.70, 1.45] of the GT's, and the three channels' ratios within 1.25 of each other (a
    white brick against red bricks is 3.8 apart, an untinted brass 4.3) -- and its relMSE is within
    twice the mesh raster's plus 0.1 (the mesh raster is the reference column, not the target);
  - ALREADY RIGHT (matx_common.GATE, one a lobe class): held to baseline.csv as compare_matx.py holds the
    gate (relMSE 5% worse, a mean ratio moved by 0.05), where the baseline has the row.

CPU here; the GPU part is run_matx.py's (the same conversion and measure, the GT cached per stage):

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
           "mx_standard_surface_onyx_hextiled"]
RATIO_LOW, RATIO_HIGH = 0.70, 1.45
CHROMA = 1.25


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
    failed = 0
    measured = 0
    print(f"matx sampled: the colour kept (each channel {RATIO_LOW}..{RATIO_HIGH} of the GT's, within {CHROMA} "
          f"of each other) and relMSE within 2 x the mesh raster's + 0.1")
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
            why = []
            if any(x is None or not (RATIO_LOW <= x <= RATIO_HIGH) for x in ratios):
                why.append("mean " + "/".join("–" if x is None else f"{x:.2f}" for x in ratios))
            elif max(ratios) / max(min(ratios), 1e-6) > CHROMA:
                why.append("colour " + "/".join(f"{x:.2f}" for x in ratios))
            mesh = r.get("relMSE_mesh_raster")
            if mesh is not None and r["relMSE_tx"] > 2.0 * mesh + 0.1:
                why.append(f"relMSE {r['relMSE_tx']:.3f} against the mesh raster's {mesh:.3f}")
            failed += bool(why)
            print(f"  {'FAIL' if why else 'ok':8} {g:44} {sky:9} relMSE {r['relMSE_tx']:.4f} (mesh {mesh:.4f}) "
                  f"mean {'/'.join(f'{x:.2f}' for x in ratios)} {'; '.join(why)}")
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
    print(f"matx sampled: {failed} failure{'s' if failed != 1 else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
