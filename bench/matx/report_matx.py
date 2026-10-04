#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""matx report, CPU only (python + oiiotool on the frames the GPU sweep wrote; nothing is measured here,
every number is the one `athenea mesh2splat --validate` computed on the GPU).

  python3 report_matx.py

Writes under MATX_OUT (default ~/luc/athenea-renders/matx/):
  summary.md, summary.csv     one row per material x sky
  lobes.md                    the breakdown by lobe class, worst offenders flagged
  contact_<sky>_p<phase>.png  GT | GS (TX raster) per material, labelled
"""
import csv
import glob
import json
import os
import statistics
import subprocess

import matx_common as m

TILE = 224
FLAG_REL = 0.10      # relMSE TX above this: flagged
FLAG_MEAN = 0.15     # any channel's mean off the GT's by more than this fraction: flagged


def load():
    manifest = json.load(open(m.MANIFEST))
    rows = []
    for e in manifest:
        if e["status"] != "ok":
            continue
        for sky in m.SKIES:
            p = os.path.join(m.RENDERS, "results", f"{e['id']}__{sky}.json")
            r = json.load(open(p)) if os.path.exists(p) else {"id": e["id"], "sky": sky, "missing": True}
            r["phase"] = e["phase"]
            r["classes"] = e.get("classes", [])
            rows.append(r)
    return manifest, rows


def ratio(v):
    mean, gt = v.get("mean"), v.get("meanGT")
    if not mean or not gt:
        return None
    return [round(a / b, 3) if b > 1e-6 else None for a, b in zip(mean, gt)]


def flat(r):
    v = r.get("validate", {})
    rr = ratio(v) if v else None
    return {
        "material": r["id"], "sky": r["sky"], "phase": r["phase"], "classes": "+".join(r["classes"]),
        "status": "missing" if r.get("missing") else ("error" if "err" in r else "ok"),
        "relMSE_tx": v.get("relMSE"), "relMSE_mesh_raster": v.get("meshRelMSE"),
        "mean_ratio_r": rr[0] if rr else None, "mean_ratio_g": rr[1] if rr else None,
        "mean_ratio_b": rr[2] if rr else None, "p99": v.get("p99"), "splats": v.get("splats"),
        "bake_s": round(r["bake_ms"] / 1000, 1) if r.get("bake_ms") else None, "raster_ms": r.get("raster_ms"),
        "wall_s": r.get("wall_s"), "error": (r.get("err") or "")[:160],
    }


def fmt(x, nd=4):
    if x is None or x == "":
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def flagged(f):
    why = []
    if f["relMSE_tx"] is not None and f["relMSE_tx"] > FLAG_REL:
        why.append(f"relMSE {f['relMSE_tx']:.3f}")
    rs = [f[k] for k in ("mean_ratio_r", "mean_ratio_g", "mean_ratio_b") if f[k] is not None]
    if rs and max(abs(x - 1) for x in rs) > FLAG_MEAN:
        why.append("mean " + "/".join(f"{x:.2f}" for x in rs))
    if f["relMSE_tx"] is not None and f["relMSE_mesh_raster"] is not None and f["relMSE_tx"] > f["relMSE_mesh_raster"]:
        why.append("worse than the mesh raster")
    return why


def tables(flats):
    cols = list(flats[0].keys())
    with open(os.path.join(m.RENDERS, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(flats)
    md = ["# matx: MaterialX materials, TX against athenea's path tracer", "",
          f"Shader ball, 512 px, TX (`--transfer`), GT path traced by athenea. The mesh raster is a reference column. "
          f"Mean ratio = TX mean / GT mean over the ball's pixels. Flagged: relMSE > {FLAG_REL}, a channel's mean off "
          f"by > {int(FLAG_MEAN * 100)} %, or TX worse than the mesh raster.", ""]
    for sky in m.SKIES:
        md += [f"## {sky}", "",
               "| material | classes | relMSE TX | relMSE mesh raster | mean ratio R/G/B | p99 | splats | bake s | raster ms | flag |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for f in sorted((f for f in flats if f["sky"] == sky), key=lambda f: (f["phase"], f["material"])):
            if f["status"] != "ok":
                md.append(f"| {f['material']} | {f['classes']} | {f['status']} {f['error'][:80]} | | | | | | | |")
                continue
            mr = "/".join(fmt(f[k], 2) for k in ("mean_ratio_r", "mean_ratio_g", "mean_ratio_b"))
            md.append(f"| {f['material']} | {f['classes']} | {fmt(f['relMSE_tx'])} | {fmt(f['relMSE_mesh_raster'])} | {mr} | "
                      f"{fmt(f['p99'], 3)} | {fmt(f['splats'])} | {fmt(f['bake_s'], 1)} | {fmt(f['raster_ms'], 2)} | "
                      f"{'; '.join(flagged(f))} |")
        md.append("")
    open(os.path.join(m.RENDERS, "summary.md"), "w").write("\n".join(md) + "\n")


def lobes(flats, manifest):
    md = ["# matx: TX error by lobe class", "",
          "A material counts in every class its surface shader's inputs switch on (read off the .mtlx: "
          "matx_common.classify). Its primary class is the first of: " + ", ".join(m.CLASSES) + ".", ""]
    for sky in m.SKIES:
        ok = [f for f in flats if f["sky"] == sky and f["status"] == "ok"]
        md += [f"## {sky}", "", "| class | n | median relMSE TX | mean relMSE TX | median mesh raster | worst (relMSE) | flagged |",
               "|---|---|---|---|---|---|---|"]
        stats = []
        for c in m.CLASSES:
            mem = [f for f in ok if c in f["classes"].split("+")]
            if not mem:
                md.append(f"| {c} | 0 | | | | | |")
                continue
            rel = [f["relMSE_tx"] for f in mem]
            mesh = [f["relMSE_mesh_raster"] for f in mem if f["relMSE_mesh_raster"] is not None]
            worst = sorted(mem, key=lambda f: -f["relMSE_tx"])[:3]
            nflag = sum(1 for f in mem if flagged(f))
            stats.append((statistics.median(rel), c))
            md.append(f"| {c} | {len(mem)} | {statistics.median(rel):.4f} | {statistics.mean(rel):.4f} | "
                      f"{statistics.median(mesh):.4f} | " if mesh else f"| {c} | {len(mem)} | {statistics.median(rel):.4f} | "
                      f"{statistics.mean(rel):.4f} | – | ")
            md[-1] += ", ".join(f"{f['material']} ({f['relMSE_tx']:.3f})" for f in worst) + f" | {nflag} |"
        md += ["", "Classes by median relMSE, worst first: " +
               ", ".join(f"{c} ({v:.3f})" for v, c in sorted(stats, reverse=True)), "",
               "**Worst offenders** (top 10 by relMSE TX):", ""]
        for f in sorted(ok, key=lambda f: -f["relMSE_tx"])[:10]:
            md.append(f"- {f['material']} [{f['classes']}]: relMSE {f['relMSE_tx']:.4f} (mesh raster "
                      f"{fmt(f['relMSE_mesh_raster'])}); {'; '.join(flagged(f)) or 'not flagged'}")
        bad = [f for f in flats if f["sky"] == sky and f["status"] == "error"]
        if bad:
            md += ["", "**Failed runs:**", ""] + [f"- {f['material']}: {f['error']}" for f in bad]
        md.append("")
    skipped = [e for e in manifest if e["status"] != "ok"]
    md += ["## Not run", ""] + [f"- {e['id']} ({e['status']}): {e.get('skip') or e.get('why')}" for e in skipped]
    open(os.path.join(m.RENDERS, "lobes.md"), "w").write("\n".join(md) + "\n")


def oiio(*args):
    subprocess.run(["oiiotool", *args], check=False, capture_output=True)


def tile(src, out, label, colour="1,1,1"):
    if src and os.path.exists(src):
        oiio(src, "--ch", "R,G,B", "--resize", f"{TILE}x{TILE}", "--colorconvert", "linear", "sRGB",
             "--text:x=4:y=16:size=13:color=" + colour, label, "-d", "uint8", "-o", out)
    else:
        oiio("--create", f"{TILE}x{TILE}", "3", "--fill:color=0.15,0.05,0.05", f"{TILE}x{TILE}",
             "--text:x=4:y=16:size=13:color=1,0.4,0.4", label, "-d", "uint8", "-o", out)


def sheets(flats):
    tiles = os.path.join(m.RENDERS, "sheets")
    os.makedirs(tiles, exist_ok=True)
    for sky in m.SKIES:
        for phase in (1, 2):
            sel = sorted((f for f in flats if f["sky"] == sky and f["phase"] == phase), key=lambda f: f["material"])
            if not sel:
                continue
            pairs = []
            for f in sel:
                d = os.path.join(m.RENDERS, "runs", sky, f["material"])
                gt = (glob.glob(os.path.join(d, "gt*.exr")) or [None])[0]
                gs = (glob.glob(os.path.join(d, "*_gs.exr")) or [None])[0]
                a, b, p = (os.path.join(tiles, f"{sky}_{f['material']}_{t}.png") for t in ("gt", "gs", "pair"))
                short = f["material"].replace("mx_standard_surface_", "ss ").replace("mx_open_pbr_", "op ") \
                    .replace("mx_disney_principled_", "dp ").replace("opbr_", "")
                tile(gt, a, f"GT {short[:26]}")
                lab = f"TX {fmt(f['relMSE_tx'])}" if f["status"] == "ok" else f["status"]
                tile(gs, b, lab, "1,0.85,0.3" if flagged(f) else "1,1,1")
                oiio(a, b, "--mosaic:pad=2", "2x1", "-o", p)
                pairs.append(p)
            cols = 4
            rows = (len(pairs) + cols - 1) // cols
            out = os.path.join(m.RENDERS, f"contact_{sky}_p{phase}.png")
            oiio(*pairs, "--mosaic:pad=6", f"{cols}x{rows}", "-o", out)
            print("wrote", out)


def main():
    manifest, rows = load()
    flats = [flat(r) for r in rows]
    tables(flats)
    lobes(flats, manifest)
    sheets(flats)
    done = sum(1 for f in flats if f["status"] == "ok")
    print(f"{done}/{len(flats)} runs measured; summary.md, summary.csv, lobes.md in {m.RENDERS}")


if __name__ == "__main__":
    main()
