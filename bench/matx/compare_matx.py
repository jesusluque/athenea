#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""matx regression check: a sweep's results against baseline.csv, CPU only (the numbers are the
ones `athenea mesh2splat --validate` measured on the GPU; this only reads them).

  python3 compare_matx.py [--results DIR] [--only <id>...]   # compare; exit 1 on a regression
  python3 compare_matx.py --gate                             # build what is missing (CPU), run the
                                                             # gate's subset (GPU, run_matx --gate), compare it
  python3 compare_matx.py --write-baseline [--results DIR]   # baseline.csv from a sweep's results

A material x sky regresses when its TX relMSE is more than 5% worse than the baseline's
(MATX_REL_SLACK, 1.05), or a channel's mean ratio (TX mean / GT mean) moved by more than 0.05
(MATX_MEAN_SLACK) -- a drift of 5% of the GT's mean either way. A run that measured in the
baseline and fails now regresses too. Exit codes: 0 none, 1 a regression, 77 nothing to compare
(no baseline.csv, or no result for any baseline row: ctest's skip).
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import matx_common as m

BASELINE = os.path.join(m.SRC, "baseline.csv")
REL_SLACK = float(os.environ.get("MATX_REL_SLACK", "1.05"))
MEAN_SLACK = float(os.environ.get("MATX_MEAN_SLACK", "0.05"))
# Quality only. Bake, raster and wall times stay in each run's JSON and are never written here,
# compared or gated on: the sweep shares the Mac with whoever is using it.
FIELDS = ["material", "sky", "relMSE_tx", "mean_ratio_r", "mean_ratio_g", "mean_ratio_b", "p99", "relMSE_mesh_raster",
          "splats", "size", "gt_paths", "bake_samples", "bin"]


def results(directory):
    out = {}
    if not os.path.isdir(directory):
        return out
    for name in os.listdir(directory):
        if not name.endswith(".json") or "__" not in name or name.startswith("catcher__"):
            continue
        r = json.load(open(os.path.join(directory, name)))
        v = r.get("validate") or {}
        row = {"material": r["id"], "sky": r["sky"], "size": r.get("size"), "gt_paths": r.get("gt_paths"),
               "bake_samples": r.get("bake_samples"), "bin": r.get("bin", "")}
        if "err" in r or "relMSE" not in v:
            row["err"] = r.get("err", "no measure")
        else:
            ratio = [a / b if b > 1e-6 else None for a, b in zip(v["mean"], v["meanGT"])]
            row.update({"relMSE_tx": v["relMSE"], "mean_ratio_r": ratio[0], "mean_ratio_g": ratio[1],
                        "mean_ratio_b": ratio[2], "p99": v.get("p99"), "relMSE_mesh_raster": v.get("meshRelMSE"),
                        "splats": v.get("splats")})
        out[(row["material"], row["sky"])] = row
    return out


def write_baseline(res):
    rows = [r for k, r in sorted(res.items()) if "err" not in r]
    with open(BASELINE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r.get(k), float) else r.get(k, "")) for k in FIELDS})
    failed = sorted(f"{k[0]} ({k[1]})" for k, r in res.items() if "err" in r)
    print(f"baseline.csv: {len(rows)} rows" + (f"; left out, failed: {', '.join(failed)}" if failed else ""))
    return 0


def read_baseline():
    if not os.path.exists(BASELINE):
        return None
    base = {}
    for r in csv.DictReader(open(BASELINE)):
        for k in ("relMSE_tx", "mean_ratio_r", "mean_ratio_g", "mean_ratio_b", "p99"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
        base[(r["material"], r["sky"])] = r
    return base


def compare(base, res, only):
    keys = [k for k in sorted(base) if not only or k[0] in only]
    lines, regressions, compared = [], 0, 0
    for k in keys:
        b, n = base[k], res.get(k)
        if n is None:
            continue
        compared += 1
        why = []
        if "err" in n:
            why.append("failed: " + n["err"][:60])
            rel = "–"
        else:
            if n["relMSE_tx"] > b["relMSE_tx"] * REL_SLACK:
                why.append(f"relMSE x{n['relMSE_tx'] / b['relMSE_tx']:.2f}")
            for c in "rgb":
                bn, nn = b[f"mean_ratio_{c}"], n[f"mean_ratio_{c}"]
                if bn is not None and nn is not None and abs(nn - bn) > MEAN_SLACK:
                    why.append(f"mean {c.upper()} {bn:.3f}->{nn:.3f}")
            rel = f"{b['relMSE_tx']:.4f} -> {n['relMSE_tx']:.4f}"
            if not why and n["relMSE_tx"] < b["relMSE_tx"] / REL_SLACK:
                why.append("better")
        bad = any(not w.startswith("better") for w in why)
        regressions += bad
        lines.append(f"{'REGRESS' if bad else 'ok':8} {k[0]:40} {k[1]:9} relMSE {rel:20} {'; '.join(why)}")
    return lines, regressions, compared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(m.RENDERS, "results"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    a = ap.parse_args()
    if a.write_baseline:
        return write_baseline(results(a.results))
    base = read_baseline()
    if base is None:
        print("no baseline.csv: write one from a sweep (--write-baseline)")
        return 77
    only = set(a.only or [])
    if a.gate:
        only = set(m.GATE)
        here = os.path.dirname(os.path.abspath(__file__))
        if os.path.realpath(m.WORK) != os.path.realpath(m.SRC) and "MATX_MANIFEST" not in os.environ:
            os.environ["MATX_MANIFEST"] = os.path.join(m.WORK, "manifest.json")
            os.makedirs(m.WORK, exist_ok=True)
        if not os.path.exists(m.SHADERBALL) or not os.path.exists(m.AUTOSHOP):
            print(f"matx gate: no shader ball ({m.SHADERBALL}) or autoshop ({m.AUTOSHOP}) on this machine")
            return 77
        if not os.path.exists(os.path.join(m.WORK, "scene", "shaderball.usdc")):
            subprocess.run([sys.executable, os.path.join(here, "build_ball.py")], check=True)
        manifest = os.environ.get("MATX_MANIFEST", m.MANIFEST)
        if not os.path.exists(manifest) or \
                not all(os.path.exists(os.path.join(m.WORK, "stages", g + ".usda")) for g in m.GATE):
            subprocess.run([sys.executable, os.path.join(here, "build_stages.py")], check=True, stdout=subprocess.DEVNULL)
        # A gate measures now: what an earlier run left is not taken as this one's.
        for g in m.GATE:
            for sky in m.SKIES:
                stale = os.path.join(a.results, f"{g}__{sky}.json")
                if os.path.exists(stale):
                    os.remove(stale)
        run = subprocess.run([sys.executable, os.path.join(here, "run_matx.py"), "--gate"])
        if run.returncode != 0:
            print(f"matx gate: the sweep stopped (exit {run.returncode})")
            return 1
    lines, regressions, compared = compare(base, results(a.results), only)
    if compared == 0:
        print(f"no result in {a.results} matches baseline.csv")
        return 77
    print(f"matx: {compared} runs against baseline.csv (relMSE slack x{REL_SLACK}, mean ratio slack {MEAN_SLACK})")
    print("\n".join(lines))
    print(f"matx: {regressions} regression{'s' if regressions != 1 else ''}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
