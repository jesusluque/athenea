#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""matx GPU driver: every material of manifest.json x every sky, TX only, against the path-traced GT.

One run = one `athenea mesh2splat <stage> --transfer --validate <dir>`: the ball (both meshes,
one material) converted with a TX transfer, the ground left a mesh, and the frame rasterised
against athenea's path tracer under the same sky. The mesh raster is only a reference column.
Sequential, one GPU job; resumable: a run whose results/<id>__<sky>.json exists is skipped
(delete it to redo). Run in the order phase 1 autoshop, phase 1 goegap, phase 2 autoshop,
phase 2 goegap, so a cut-short sweep still has whole columns.

  python3 run_matx.py [--phase 1|2|all] [--sky autoshop|goegap|all] [--only <id>...] [--gate] [--dry]
  --gate: the regression gate's subset (matx_common.GATE, one material a lobe class, phase 1)
  env: MATX_BIN (athenea), MATX_SIZE (512), MATX_PATHS (256), MATX_BAKE (64),
       MATX_CAMERA_PIXELS (512), MATX_MAX_SPLATS (600000), MATX_TIMEOUT (900 s),
       MATX_CATCHER=1 also converts the ground's shadow catcher once per sky (mx_open_pbr_default)

Writes under MATX_OUT (default ~/luc/athenea-renders/matx/):
  runs/<sky>/<id>/       what --validate writes (validate.json, gt*.exr, mesh.exr, <Mat>_gs.exr,
                         <Mat>.png GT|mesh|cloud, clouds/) and mesh2splat.log
  results/<id>__<sky>.json   one run: validate's row + bake ms, wall s, raster ms
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import matx_common as m

SIZE = int(os.environ.get("MATX_SIZE", "512"))
PATHS = int(os.environ.get("MATX_PATHS", "256"))
BAKE = int(os.environ.get("MATX_BAKE", "64"))
CAMERA_PIXELS = int(os.environ.get("MATX_CAMERA_PIXELS", "512"))
MAX_SPLATS = int(os.environ.get("MATX_MAX_SPLATS", "600000"))
TIMEOUT = int(os.environ.get("MATX_TIMEOUT", "900"))
CAMERA = "/World/Camera"
PRIM = "/World/ShaderBall"


def sh(cmd, log=None, timeout=TIMEOUT):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out, code = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out += f"\n[matx] timeout after {timeout} s\n"
        code = -9
    if log:
        open(log, "w").write(" ".join(cmd) + "\n\n" + out)
    return out, code, time.time() - t0


def convert_args(stage, run_dir, sky):
    args = [m.ATHENEA, "mesh2splat", stage, "--prim", PRIM, "--transfer",
            "--cell-from-camera", CAMERA, "--camera-pixels", str(CAMERA_PIXELS), "--max-splats", str(MAX_SPLATS),
            "--bake-samples", str(BAKE), "--bake-extra", "0",
            "--validate", run_dir, "--validate-camera", CAMERA, "--validate-size", str(SIZE), str(SIZE),
            "--validate-paths", str(PATHS)]
    if m.SKIES[sky]:
        args += ["--validate-sky", m.SKIES[sky]]
    return args


def raster_ms(cloud_stage, out_exr):
    o, code, _ = sh([m.ATHENEA, "stage", cloud_stage, "--camera", CAMERA, "--size", f"{SIZE}x{SIZE}",
                     "--frames", "5", "-o", out_exr], timeout=300)
    mm = re.search(r"median ([\d.]+) ms", o)
    return float(mm.group(1)) if mm else None


def run_one(e, sky, results):
    res_path = os.path.join(results, f"{e['id']}__{sky}.json")
    if os.path.exists(res_path):
        return "skip"
    run_dir = os.path.join(m.RENDERS, "runs", sky, e["id"])
    os.makedirs(run_dir, exist_ok=True)
    out, code, wall = sh(convert_args(m.stage_path(e), run_dir, sky), os.path.join(run_dir, "mesh2splat.log"))
    # Whatever would make every later run fail the same way stops the sweep, with no result written.
    for fatal, code_out in (("Reentrancy avoided", 3), ("no Measure bundle", 4)):
        if fatal in out:
            print(f"[matx] '{fatal}' in {e['id']}'s log: the sweep stops (see {run_dir}/mesh2splat.log)", flush=True)
            sys.exit(code_out)
    r = {"id": e["id"], "sky": sky, "phase": e["phase"], "classes": e.get("classes", []), "material": e.get("material"),
         "wall_s": round(wall, 1), "exit": code, "bin": m.ATHENEA, "size": SIZE, "gt_paths": PATHS, "bake_samples": BAKE}
    bakes = [int(x) for x in re.findall(r"transfer baked for \d+ of \d+ gaussians .*? in (\d+) ms", out)]
    r["bake_ms"] = sum(bakes) if bakes else None
    vj = os.path.join(run_dir, "validate.json")
    if os.path.exists(vj):
        rows = json.load(open(vj)).get("materials", [])
        if rows:
            r["validate"] = rows[0]
            if "error" in rows[0]:
                r["err"] = rows[0]["error"]
    else:
        r["err"] = (out.strip().splitlines() or ["no output"])[-1][-400:]
    if code != 0 and "err" not in r:
        r["err"] = f"exit {code}: " + out.strip()[-400:]
    # Raster time: the cloud is the same under both skies, so it is timed once, under the stage's own.
    if "err" not in r and sky == "autoshop":
        name = os.path.basename(e["material"] or "")
        composed = os.path.join(run_dir, "clouds", name + ".usda")
        if os.path.exists(composed):
            r["raster_ms"] = raster_ms(composed, os.path.join(run_dir, "raster_timing.exr"))
    json.dump(r, open(res_path, "w"), indent=1)
    v = r.get("validate", {})
    print(f"[matx] {e['id']:40} {sky:9} " + (f"ERR {r['err'][:120]}" if "err" in r else
          f"relMSE {v.get('relMSE'):.4f} mesh {v.get('meshRelMSE'):.4f} splats {v.get('splats')} "
          f"bake {r['bake_ms']} ms raster {r.get('raster_ms')} ms wall {r['wall_s']} s"), flush=True)
    return "done"


def catcher(sky, results):
    """The ground's shadow catcher (optional): converted once, drawn over the mesh ground with the
    mesh ball, whole frame against the same GT the default material's run path traced."""
    res_path = os.path.join(results, f"catcher__{sky}.json")
    base_dir = os.path.join(m.RENDERS, "runs", sky, "mx_open_pbr_default")
    gts = glob.glob(os.path.join(base_dir, "gt*.exr"))
    if os.path.exists(res_path) or not gts:
        return
    d = os.path.join(m.RENDERS, "runs", sky, "catcher")
    os.makedirs(d, exist_ok=True)
    stage = os.path.join(m.WORK, "stages", "mx_open_pbr_default.usda")
    if sky != "autoshop":
        stage = os.path.join(base_dir, "sky.usda")    # what --validate-sky wrote: the same stage, the other dome
    cloud = os.path.join(d, "catcher.usdc")
    out, code, wall = sh([m.ATHENEA, "mesh2splat", os.path.join(m.WORK, "stages", "mx_open_pbr_default.usda"),
                          "--prim", PRIM, "--shadow-catcher", "--transfer", "--bake-samples", str(BAKE),
                          "--bake-extra", "0", "--no-camera", "-o", cloud], os.path.join(d, "mesh2splat.log"))
    r = {"sky": sky, "exit": code, "wall_s": round(wall, 1)}
    if code == 0:
        composed = os.path.join(d, "composed.usda")
        open(composed, "w").write(f'#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Y"\n'
                                  f'    subLayers = [@{cloud}@, @{stage}@]\n)\n')
        exr = os.path.join(d, "catcher_gs.exr")
        r["raster_ms"] = raster_ms(composed, exr)
        for tag, img in (("catcher", exr), ("mesh", os.path.join(base_dir, "mesh.exr"))):
            o, _, _ = sh([m.ATHENEA, "compare", img, gts[0]], timeout=120)
            mm = re.search(r"relMSE ([\d.e+-]+)\s+p99 relative ([\d.e+-]+)", o)
            r[tag] = {"relMSE": float(mm.group(1)), "p99rel": float(mm.group(2))} if mm else {"err": o[-300:]}
    else:
        r["err"] = out.strip()[-400:]
    json.dump(r, open(res_path, "w"), indent=1)
    print(f"[matx] catcher {sky}: {r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all")
    ap.add_argument("--sky", default="all")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.gate:
        a.phase, a.only = "1", m.GATE
    manifest = json.load(open(m.MANIFEST))
    results = os.path.join(m.RENDERS, "results")
    os.makedirs(results, exist_ok=True)
    phases = [1, 2] if a.phase == "all" else [int(a.phase)]
    skies = list(m.SKIES) if a.sky == "all" else [a.sky]
    todo = [(e, sky) for p in phases for sky in skies for e in manifest
            if e["phase"] == p and e["status"] == "ok" and (not a.only or e["id"] in a.only)]
    left = [t for t in todo if not os.path.exists(os.path.join(results, f"{t[0]['id']}__{t[1]}.json"))]
    print(f"[matx] {len(todo)} runs, {len(left)} to do; {SIZE} px, GT {PATHS} paths, bake {BAKE} paths, "
          f"{m.ATHENEA}", flush=True)
    if a.dry:
        for e, sky in left:
            print("  " + " ".join(convert_args(m.stage_path(e), os.path.join(m.RENDERS, "runs", sky, e["id"]), sky)))
        return
    t0 = time.time()
    for k, (e, sky) in enumerate(left):
        run_one(e, sky, results)
        el = time.time() - t0
        print(f"[matx] {k + 1}/{len(left)}, {el / 60:.1f} min, ~{el / (k + 1) * (len(left) - k - 1) / 60:.0f} min left",
              flush=True)
    if os.environ.get("MATX_CATCHER") == "1":
        for sky in skies:
            catcher(sky, results)


if __name__ == "__main__":
    sys.exit(main())
