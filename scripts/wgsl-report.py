#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# The raster route's kernels as WGSL: does each compile, and what would it ask
# of a WebGPU device.
#
# Every entry point the engine dispatches to draw a splat cloud (decode, the
# projection's variants, the counters, the sorts, emit, ranges, blend) and the
# AOFX Measure effect `athenea compare` runs, compiled by slangc
# -target wgsl from the one Slang source. For each it prints:
#
#   - whether it compiles, and if not, the Slang features WGSL lacks that it
#     uses (InterlockedAdd: a WGSL atomic is a type, Atomic<T>, not a call);
#   - storage buffers (maxStorageBuffersPerShaderStage: 8 by default on the
#     web), uniforms and storage textures;
#   - workgroup memory in bytes, by WGSL's layout rules
#     (maxComputeWorkgroupStorageSize: 16384 by default, 32768 on 99% of
#     adapters);
#   - Naga's verdict (Firefox's WGSL compiler) when `naga` is on PATH or in
#     ~/.cargo/bin (cargo install naga-cli), and Tint's (Chrome's, inside
#     Dawn: scripts/wgsl-tint.cpp, built here against scripts/build-dawn.sh's
#     Dawn): its verdict on the WGSL (parse, types, uniformity), then on the
#     pipeline under the web's default limits.
#
# Nothing here touches a GPU: slangc, Naga and Dawn's null backend are all
# compilers on the CPU. The WGSL is written to --out to be read, never edited:
# the source is Slang.
#
# Usage: scripts/wgsl-report.py [--shaders DIR] [--plugins DIR] [--out DIR]
#                               [--only ENTRY,...] [--no-tint] [--markdown]
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# (module path under shaders/, entry, what it is). Plugins are under plugins/.
KERNELS = [
    ("athenea/scene/splat_validate", "splatValidate", "decode"),
    ("athenea/scene/streams", "splatStreams", "decode"),
    ("athenea/scene/splat_decode", "splatDecode", "decode"),
    ("athenea/scene/sog_decode", "sogDecode", "decode"),
    ("athenea/scene/splat_unpack", "splatUnpack", "decode"),
    ("athenea/scene/splat_kept", "splatKept", "decode"),
    ("athenea/scene/splat_kept", "splatKeptScatter", "decode"),
    ("athenea/scene/bounds_chunks", "boundsChunks", "decode"),
    ("athenea/scene/bounds_reduce", "boundsReduce", "decode"),
    ("athenea/splat/splat_project", "splatProject", "project"),
    ("athenea/splat/splat_project", "splatProjectFirst", "project"),
    ("athenea/splat/splat_project", "splatProjectPlain", "project"),
    ("athenea/splat/splat_project", "splatTransferViewless", "project"),
    ("athenea/splat/splat_project", "splatProjectCatcher", "project (play-ground)"),
    ("athenea/splat/splat_compact", "splatCompact", "counts"),
    ("athenea/splat/splat_gather_counts", "splatGatherCounts", "counts"),
    ("athenea/splat/splat_frame_counters", "splatCountersClear", "counts"),
    ("athenea/splat/splat_frame_counters", "splatCounters", "counts"),
    ("athenea/splat/splat_frame_counters", "splatCountersCloud", "counts"),
    ("athenea/algo/prefix_chunk_totals", "prefixChunkTotals", "sort"),
    ("athenea/algo/prefix_chunk_starts", "prefixChunkStarts", "sort"),
    ("athenea/algo/prefix_local", "prefixLocal", "sort"),
    ("athenea/algo/radix_histogram", "radixHistogram", "sort"),
    ("athenea/algo/radix_totals", "radixTotals", "sort"),
    ("athenea/algo/radix_starts", "radixStarts", "sort"),
    ("athenea/algo/radix_scatter", "radixScatter", "sort"),
    ("athenea/algo/radix_block", "radixBlockHistogram", "sort"),
    ("athenea/algo/radix_block", "radixBlockScatter", "sort"),
    ("athenea/splat/splat_tiles_clear", "splatTilesClear", "emit"),
    ("athenea/splat/splat_emit", "splatEmit", "emit"),
    ("athenea/splat/splat_ranges", "splatRanges", "emit"),
    ("athenea/splat/splat_blend", "splatBlend", "blend"),
    ("athenea/splat/splat_blend", "splatBlendComposite", "blend"),
    ("athenea/splat/splat_blend", "splatBlendCrypto", "blend"),
    ("athenea/splat/splat_blend", "splatBlendCompositeCrypto", "blend"),
    ("plugin:measure/measure", "measureRows", "measure"),
    ("plugin:measure/measure", "measureReduce", "measure"),
    ("plugin:measure/measure", "measureFinish", "measure"),
    ("plugin:measure/measure", "measureHeatmap", "measure"),
]

# --- WGSL layout (WGSL spec, "Memory Layout") ---------------------------------

SCALARS = {"u32": (4, 4), "i32": (4, 4), "f32": (4, 4), "f16": (2, 2), "bool": (4, 4)}


def round_up(k, n):
    return (n + k - 1) // k * k


def split_top(text):
    """Splits 'a, b<c, d>, e' on the commas outside brackets."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def const_int(text):
    m = re.fullmatch(r"(?:[iu]32\()?\s*(\d+)[iu]?\s*\)?", text.strip())
    if not m:
        raise ValueError(f"not a constant count: {text}")
    return int(m.group(1))


def layout(ty, structs):
    """(size, align) of a WGSL type, by the spec's rules."""
    ty = ty.strip()
    if ty in SCALARS:
        return SCALARS[ty]
    m = re.fullmatch(r"atomic<(\w+)>", ty)
    if m:
        return SCALARS[m.group(1)]
    m = re.fullmatch(r"vec([234])<(\w+)>", ty)
    if m:
        n, (s, _) = int(m.group(1)), SCALARS[m.group(2)]
        align = s * (4 if n == 3 else n)
        return s * n, align
    m = re.fullmatch(r"mat([234])x([234])<(\w+)>", ty)
    if m:
        cols, rows = int(m.group(1)), int(m.group(2))
        size, align = layout(f"vec{rows}<{m.group(3)}>", structs)
        return cols * round_up(align, size), align
    m = re.fullmatch(r"array<(.*)>", ty)
    if m:
        args = split_top(m.group(1))
        size, align = layout(args[0], structs)
        if len(args) < 2:
            return 0, align   # runtime-sized: storage only
        return const_int(args[1]) * round_up(align, size), align
    if ty in structs:
        return structs[ty]
    raise ValueError(f"unknown type {ty}")


def parse_structs(wgsl):
    structs = {}
    for m in re.finditer(r"struct\s+(\w+)\s*\{(.*?)\};?", wgsl, re.S):
        name, body = m.group(1), m.group(2)
        offset, max_align = 0, 1
        for field in split_top(body.replace("\n", " ")):
            fm = re.fullmatch(r"((?:@\w+\(\d+\)\s*)*)\w+\s*:\s*(.+)", field.strip())
            if not fm:
                continue
            size, align = layout(fm.group(2), structs)
            attrs = dict(re.findall(r"@(\w+)\((\d+)\)", fm.group(1)))
            align = int(attrs.get("align", align))
            size = int(attrs.get("size", size))
            offset = round_up(align, offset) + size
            max_align = max(max_align, align)
        structs[name] = (round_up(max_align, offset), max_align)
    return structs


def analyse(wgsl):
    structs = parse_structs(wgsl)
    storage_ro = len(re.findall(r"var<storage,\s*read>", wgsl))
    storage_rw = len(re.findall(r"var<storage,\s*read_write>", wgsl))
    uniforms = len(re.findall(r"var<uniform>", wgsl))
    storage_tex = len(re.findall(r":\s*texture_storage_", wgsl))
    sampled_tex = len(re.findall(r":\s*texture_(?:2d|3d|cube|2d_array)<", wgsl))
    groups = sorted({int(g) for g in re.findall(r"@group\((\d+)\)", wgsl)})
    workgroup = 0
    for m in re.finditer(r"var<workgroup>\s+\w+\s*:\s*([^;]+);", wgsl):
        workgroup += layout(m.group(1), structs)[0]
    size = re.search(r"@workgroup_size\(([^)]*)\)", wgsl)
    return {
        "storage": storage_ro + storage_rw,
        "storage_ro": storage_ro,
        "storage_rw": storage_rw,
        "uniforms": uniforms,
        "storage_tex": storage_tex,
        "sampled_tex": sampled_tex,
        "groups": groups,
        "workgroup": workgroup,
        "threads": size.group(1).replace(" ", "") if size else "?",
    }


def failures(stderr):
    """What WGSL lacks, as Slang names it: the functions an entry used."""
    used = re.findall(r"note: see using of '([^']+)'", stderr)
    errors = re.findall(r"error\[(E\d+)\]: ([^\n]+)", stderr)
    out = sorted(set(used))
    if not out:
        out = sorted({f"{code} {text.strip()[:70]}" for code, text in errors})
    return out


def tint_verdict(tint, target):
    """(language, web limits): Tint on the WGSL, then the pipeline on the web's
    default limits. The null adapter goes no further than those for workgroup
    memory, so 10 buffers / 32 KiB is this script's own count, not Tint's."""
    v = subprocess.run([tint, str(target)], capture_output=True, text=True)
    fields = dict(part.split(": ", 1) for part in v.stdout.strip().split("\t")[1:] if ": " in part)
    module, pipeline = fields.get("module", "?"), fields.get("pipeline", "?")

    def short(said):
        if said == "ok":
            return "ok"
        m = re.search(r"number of storage buffers \((\d+)\)", said)
        if m:
            return f"storage {m.group(1)}"
        m = re.search(r"workgroup storage \((\d+) bytes\)", said)
        if m:
            return f"workgroup {m.group(1)}"
        m = re.search(r"error: (.{0,110})", said)
        return (m.group(1) if m else said[:110]).strip()
    return short(module), short(pipeline)


def build_tint(here, out):
    """scripts/wgsl-tint.cpp against Dawn, once."""
    dawn = Path(os.environ.get("ATHENEA_DAWN_ROOT", Path.home() / "tools/dawn-138.0.7204.168"))
    if not (dawn / "include/dawn/webgpu.h").exists():
        return None
    binary = out / "wgsl-tint"
    source = here / "scripts/wgsl-tint.cpp"
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(["c++", "-std=c++20", "-O1", str(source), "-I", str(dawn / "include"),
                        "-L", str(dawn / "lib"), "-ldawn", f"-Wl,-rpath,{dawn / 'lib'}",
                        "-o", str(binary)], check=True)
    return str(binary)


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--shaders", default=str(here / "shaders"))
    ap.add_argument("--plugins", default=str(here / "plugins"))
    ap.add_argument("--slangc", default=str(Path.home() / "tools/slang/bin/slangc"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-tint", action="store_true",
                    help="skip Tint (built from scripts/wgsl-tint.cpp against ~/tools/dawn-*)")
    ap.add_argument("--web-limits", default="8,16384",
                    help="storage buffers per stage, workgroup bytes: what is flagged")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    out = Path(args.out or os.environ.get("TMPDIR", "/tmp")) / "wgsl"
    out.mkdir(parents=True, exist_ok=True)
    only = {e for e in args.only.split(",") if e}
    naga = shutil.which("naga") or (str(Path.home() / ".cargo/bin/naga")
                                     if (Path.home() / ".cargo/bin/naga").exists() else None)
    max_storage, max_wg = (int(x) for x in args.web_limits.split(","))
    tint = None if args.no_tint else build_tint(here, out.parent)

    rows = []
    for module, entry, what in KERNELS:
        if only and entry not in only:
            continue
        if module.startswith("plugin:"):
            source = Path(args.plugins) / (module[len("plugin:"):] + ".slang")
        else:
            source = Path(args.shaders) / (module + ".slang")
        if not source.exists():
            rows.append((what, entry, "no source", None, [], "", ""))
            continue
        target = out / f"{entry}.wgsl"
        cmd = [args.slangc, str(source), "-I", args.shaders, "-target", "wgsl",
               "-entry", entry, "-stage", "compute", "-o", str(target)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0 or not target.exists():
            missing = "entry not found" if "E38000" in run.stderr or "entryPoint" in run.stderr and "not" in run.stderr else None
            rows.append((what, entry, "no", None, failures(run.stderr) or [missing or "?"], "", ""))
            continue
        wgsl = target.read_text()
        info = analyse(wgsl)
        naga_says = ""
        if naga:
            v = subprocess.run([naga, str(target)], capture_output=True, text=True)
            naga_says = "ok" if v.returncode == 0 else (v.stderr.strip().splitlines() or ["error"])[0][:80]
        tint_says = ""
        if tint:
            tint_says = " · ".join(tint_verdict(tint, target))
        rows.append((what, entry, "yes", info, [], naga_says, tint_says))

    if args.markdown:
        print("| Group | Kernel | WGSL | Storage (ro+rw) | Uniform | Storage tex | Workgroup B | Threads | "
              "Fails / over web limits | Naga | Tint: WGSL · 8/16K |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
    for what, entry, ok, info, fails, naga_says, tint_says in rows:
        if info is None:
            line = [what, f"`{entry}`", ok, "", "", "", "", "", ", ".join(fails), "", ""]
        else:
            over = []
            if info["storage"] > max_storage:
                over.append(f"storage {info['storage']} > {max_storage}")
            if info["workgroup"] > max_wg:
                over.append(f"workgroup {info['workgroup']} > {max_wg}")
            line = [what, f"`{entry}`", ok,
                    f"{info['storage']} ({info['storage_ro']}+{info['storage_rw']})",
                    str(info["uniforms"]), str(info["storage_tex"]), str(info["workgroup"]),
                    info["threads"], "; ".join(over), naga_says, tint_says]
        if args.markdown:
            print("| " + " | ".join(line) + " |")
        else:
            print("\t".join(line))
    print(f"\nWGSL in {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
