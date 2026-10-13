#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""simplify: a cloud and its merges by pairs against the mesh path traced, from N cameras.

    python3 validate.py --stage S.usda --prim /World/Thing --cloud full.usdc --out DIR \\
        --pivot X Y Z --distance D [--up y|z] [--cameras 6] [--ratios 2 5 10 20] [--paths 1024]

For every camera on an orbit round `--pivot` (half of them level, half from 30 degrees above):
the stage path traced with the cloud's mesh (the reference, kept between runs), the full cloud
and each `athenea decimate --ratio R` of it drawn by raster under the stage's lights and camera,
each measured against the reference and against the full cloud by `athenea compare` -- relMSE,
PSNR and SSIM, every number a GPU kernel's -- and, where LPIPS_PYTHON names a Python with torch
and lpips, LPIPS (AlexNet) on the 8-bit sRGB frames. Writes DIR/table.md: gaussians, file size,
the merge's wall time, and each metric's mean over the cameras.

Nothing is measured on the CPU here: it writes cameras, runs athenea, oiiotool and LPIPS, and
averages the numbers they print. Timings are wall time on a machine that may be busy.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LPIPS_SCRIPT = os.path.join(HERE, "lpips_png.py")


def run(cmd, log=None):
    started = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log:
        with open(log, "w") as f:
            f.write(p.stdout)
    if p.returncode != 0:
        sys.exit(f"failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout[-2000:]}")
    return p.stdout, time.time() - started


def look_at(eye, target, up):
    """A camera-to-world matrix (row vectors, USD's convention) looking from eye at target."""
    f = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]]
    n = math.sqrt(sum(c * c for c in r))
    r = [c / n for c in r]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    back = [-c for c in f]   # a USD camera looks down its -Z
    return [r + [0.0], u + [0.0], back + [0.0], list(eye) + [1.0]]


def write_cameras(path, count, pivot, distance, up_axis, focal):
    up = [0.0, 1.0, 0.0] if up_axis == "y" else [0.0, 0.0, 1.0]
    names = []
    lines = ["#usda 1.0", "(", f'    upAxis = "{up_axis.upper()}"', ")", "", 'over "World"', "{"]
    for k in range(count):
        azimuth = 2.0 * math.pi * k / count
        elevation = math.radians(30.0 if k % 2 else 5.0)
        horizontal = [math.cos(azimuth), math.sin(azimuth)]
        if up_axis == "y":
            offset = [horizontal[0] * math.cos(elevation), math.sin(elevation), horizontal[1] * math.cos(elevation)]
        else:
            offset = [horizontal[0] * math.cos(elevation), horizontal[1] * math.cos(elevation), math.sin(elevation)]
        eye = [p + distance * o for p, o in zip(pivot, offset)]
        m = look_at(eye, pivot, up)
        name = f"Orbit{k}"
        names.append(name)
        rows = ", ".join("(" + ", ".join(f"{v:.9g}" for v in row) + ")" for row in m)
        lines += [f'    def Camera "{name}"', "    {",
                  f"        float focalLength = {focal}",
                  "        float horizontalAperture = 24", "        float verticalAperture = 24",
                  f"        matrix4d xformOp:transform = ({rows})",
                  '        uniform token[] xformOpOrder = ["xformOp:transform"]', "    }"]
    lines.append("}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return names


def stage_with(path, stage, cameras, cloud, prim):
    """The stage with the cameras, and with the cloud in place of the mesh (or the mesh, for None)."""
    lines = ["#usda 1.0", "(", f"    subLayers = [@{cameras}@, @{stage}@]", ")", ""]
    if cloud:
        parts = prim.strip("/").split("/")
        for depth, part in enumerate(parts):
            pad = "    " * depth
            lines += [f'{pad}over "{part}"' + (' (\n' + pad + '    active = false\n' + pad + ')' if depth == len(parts) - 1 else ""),
                      pad + "{"]
        for depth in reversed(range(len(parts))):
            lines.append("    " * depth + "}")
        lines += [f'def "SimplifyCloud" (', f"    prepend references = @{cloud}@</World/Splats>", ")", "{", "}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def compare(athenea, image, reference):
    out, _ = run([athenea, "compare", image, reference])
    rel = re.search(r"relMSE ([0-9.e+-]+)", out)
    disp = re.search(r"PSNR ([0-9.e+-]+) dB\s+SSIM ([0-9.e+-]+)", out)
    return (float(rel.group(1)) if rel else math.nan,
            float(disp.group(1)) if disp else math.nan,
            float(disp.group(2)) if disp else math.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athenea", default=os.environ.get("ATHENEA_BIN", "athenea"))
    ap.add_argument("--stage", required=True)
    ap.add_argument("--prim", required=True, help="the mesh the cloud stands for, switched off under the cloud")
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pivot", type=float, nargs=3, required=True)
    ap.add_argument("--distance", type=float, required=True)
    ap.add_argument("--up", choices=["y", "z"], default="y")
    ap.add_argument("--focal", type=float, default=35.0)
    ap.add_argument("--cameras", type=int, default=6)
    ap.add_argument("--ratios", type=float, nargs="+", default=[2, 5, 10, 20])
    ap.add_argument("--paths", type=int, default=1024)
    ap.add_argument("--size", default="512x512")
    ap.add_argument("--decimate", nargs=argparse.REMAINDER, default=[],
                    help="more options for athenea decimate, last on the line")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cams = os.path.join(a.out, "cameras.usda")
    names = write_cameras(cams, a.cameras, a.pivot, a.distance, a.up, a.focal)
    lpips_python = os.environ.get("LPIPS_PYTHON")

    # The references: the mesh, path traced, once.
    ref_stage = os.path.join(a.out, "reference.usda")
    stage_with(ref_stage, os.path.abspath(a.stage), cams, None, a.prim)
    for n in names:
        ref = os.path.join(a.out, f"ref_{n}.exr")
        if not os.path.exists(ref):
            run([a.athenea, "stage", ref_stage, "--camera", f"/World/{n}", "--size", a.size, "--technique", "rt",
                 "--path-total", str(a.paths), "-o", ref], os.path.join(a.out, f"ref_{n}.log"))

    variants = [("full", 1.0, os.path.abspath(a.cloud), 0.0)]
    for r in a.ratios:
        tag = f"{r:g}x"
        path = os.path.join(a.out, f"cloud_{tag}.usdc")
        took = 0.0
        if not os.path.exists(path):
            _, took = run([a.athenea, "decimate", os.path.abspath(a.cloud), path, "--ratio", f"{r:g}"] + a.decimate,
                          os.path.join(a.out, f"decimate_{tag}.log"))
        variants.append((tag, r, path, took))

    rows = []
    for tag, ratio, cloud, took in variants:
        st = os.path.join(a.out, f"stage_{tag}.usda")
        stage_with(st, os.path.abspath(a.stage), cams, cloud, a.prim)
        log = os.path.join(a.out, f"decimate_{tag}.log")
        count = None
        if os.path.exists(log):
            m = re.search(r"decimate: (\d+) gaussians of (\d+)", open(log).read())
            count = int(m.group(1)) if m else None
        sums = {"rel": [], "psnr": [], "ssim": [], "lpips": [], "full_psnr": [], "full_ssim": []}
        for n in names:
            frame = os.path.join(a.out, f"{tag}_{n}.exr")
            if not os.path.exists(frame):
                run([a.athenea, "stage", st, "--camera", f"/World/{n}", "--size", a.size, "-o", frame])
            rel, psnr, ssim = compare(a.athenea, frame, os.path.join(a.out, f"ref_{n}.exr"))
            sums["rel"].append(rel)
            sums["psnr"].append(psnr)
            sums["ssim"].append(ssim)
            if tag != "full":
                _, fp, fs = compare(a.athenea, frame, os.path.join(a.out, f"full_{n}.exr"))
                sums["full_psnr"].append(fp)
                sums["full_ssim"].append(fs)
            if lpips_python:
                for exr in (frame, os.path.join(a.out, f"ref_{n}.exr")):
                    png = exr[:-4] + ".png"
                    if not os.path.exists(png):
                        run(["oiiotool", exr, "--ch", "R,G,B", "--colorconvert", "linear", "sRGB", "-d", "uint8",
                             "-o", png])
                out, _ = run([lpips_python, LPIPS_SCRIPT, os.path.join(a.out, f"ref_{n}.png"), frame[:-4] + ".png"])
                sums["lpips"].append(float(out.split()[-1]))
        mean = lambda v: sum(v) / len(v) if v else math.nan
        rows.append((tag, count, os.path.getsize(cloud), took, mean(sums["rel"]), mean(sums["psnr"]),
                     mean(sums["ssim"]), mean(sums["lpips"]), mean(sums["full_psnr"]), mean(sums["full_ssim"])))

    with open(os.path.join(a.out, "table.md"), "w") as f:
        f.write(f"# {os.path.basename(a.stage)}: {len(names)} cameras, reference path traced at {a.paths} paths\n\n")
        f.write("| cloud | gaussians | file MB | merge s | relMSE | PSNR dB | SSIM | LPIPS | PSNR vs full | SSIM vs full |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for tag, count, size, took, rel, psnr, ssim, lp, fp, fs in rows:
            f.write(f"| {tag} | {count if count else '-'} | {size / 1e6:.1f} | {took:.1f} | {rel:.4f} | {psnr:.2f} | "
                    f"{ssim:.4f} | {lp:.4f} | {fp:.2f} | {fs:.4f} |\n")
    print(open(os.path.join(a.out, "table.md")).read())


if __name__ == "__main__":
    main()
