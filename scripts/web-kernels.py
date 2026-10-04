#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# THE WEB MODULE, BUILT: the kernels of the web's raster route (proposal 072,
# preset T1) compiled from the one Slang source to WGSL, a manifest of what
# each binds and by what name, and the page's module beside them -- the
# directory a site serves as its `athenea-webgpu` renderer.
#
#   <out>/manifest.json        what the host reads: per kernel its file, its
#                              workgroup size, every parameter by name (its
#                              @binding, its kind, and for a uniform the offset
#                              of every field), its override constants, and
#                              what it asks of the device
#   <out>/kernels/<entry>.wgsl one module a kernel
#   <out>/*.js, test.html      web/athenea-webgpu, copied
#
# Every kernel must compile, pass Naga (Firefox) and Tint (Chrome, through
# Dawn's null backend: scripts/wgsl-tint.cpp, with the pipeline checked against
# the specification's default limits), and fit those limits by this script's
# own count -- 8 storage buffers a stage, 16384 bytes of workgroup memory --
# or the build fails. The uniform layouts Slang's reflection gives are checked
# against the WGSL's own (wgsl-report.py's layout rules), since the page writes
# a uniform by those offsets.
#
# Nothing here touches a GPU.
#
# Usage: scripts/web-kernels.py --out DIR [--no-tint]
import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("wgsl_report", HERE / "scripts/wgsl-report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

# The web route, in the order a frame dispatches it (web/athenea-webgpu/host.js).
KERNELS = [
    ("athenea/web/web_decode", "webDecode"),
    ("athenea/scene/bounds_chunks", "boundsChunks"),
    ("athenea/scene/bounds_reduce", "boundsReduce"),
    ("athenea/web/web_project", "webProject"),
    ("athenea/algo/prefix_chunk_totals", "prefixChunkTotals"),
    ("athenea/algo/prefix_chunk_starts", "prefixChunkStarts"),
    ("athenea/algo/prefix_local", "prefixLocal"),
    ("athenea/splat/splat_compact", "splatCompact"),
    ("athenea/algo/radix_histogram", "radixHistogram"),
    ("athenea/algo/radix_totals", "radixTotals"),
    ("athenea/algo/radix_starts", "radixStarts"),
    ("athenea/algo/radix_scatter", "radixScatter"),
    ("athenea/splat/splat_gather_counts", "splatGatherCounts"),
    ("athenea/splat/splat_emit", "splatEmit"),
    ("athenea/splat/splat_tiles_clear", "splatTilesClear"),
    ("athenea/splat/splat_ranges", "splatRanges"),
    ("athenea/web/web_blend", "webBlend"),
    ("athenea/web/web_present", "webPresent"),
]

LIMITS = {"maxStorageBuffersPerShaderStage": 8, "maxComputeWorkgroupStorageSize": 16384,
          "maxUniformBuffersPerShaderStage": 12, "maxStorageTexturesPerShaderStage": 4}


def fields_of(struct_type, prefix=""):
    """{dotted name: (offset, scalar type)} of a reflected struct, nested ones flattened."""
    out = {}
    for f in struct_type["fields"]:
        ty = f["type"]
        offset = f["binding"]["offset"]
        name = prefix + f["name"]
        if ty["kind"] == "struct":
            for k, (o, s) in fields_of(ty, name + ".").items():
                out[k] = (offset + o, s)
        elif ty["kind"] == "scalar":
            out[name] = (offset, {"uint32": "u32", "int32": "i32", "float32": "f32"}[ty["scalarType"]])
        else:
            raise SystemExit(f"uniform field {name}: a {ty['kind']}, the page writes scalars only")
    return out


def wgsl_offsets(wgsl, struct):
    """{dotted name: offset} of a WGSL struct by its own @align and layout rules."""
    structs = report.parse_structs(wgsl)
    bodies = {m.group(1): m.group(2) for m in re.finditer(r"struct\s+(\w+)\s*\{(.*?)\};?", wgsl, re.S)}

    def walk(name, base, prefix):
        out, offset = {}, 0
        for field in report.split_top(bodies[name].replace("\n", " ")):
            fm = re.fullmatch(r"((?:@\w+\(\d+\)\s*)*)(\w+)\s*:\s*(.+)", field.strip())
            if not fm:
                continue
            size, align = report.layout(fm.group(3), structs)
            attrs = dict(re.findall(r"@(\w+)\((\d+)\)", fm.group(1)))
            align = int(attrs.get("align", align))
            offset = report.round_up(align, offset)
            clean = re.sub(r"_\d+$", "", fm.group(2))
            if fm.group(3).strip() in bodies:
                out.update(walk(fm.group(3).strip(), base + offset, prefix + clean + "."))
            else:
                out[prefix + clean] = base + offset
            offset += int(attrs.get("size", size))
        return out
    return walk(struct, 0, "")


def parameter(p, wgsl):
    """One reflected parameter as the page binds it."""
    ty = p["type"]
    index = p["binding"]["index"]
    if ty["kind"] == "constantBuffer":
        element = ty["elementType"]
        fields = fields_of(element)
        m = re.search(rf"@binding\({index}\)\s*@group\(0\)\s*var<uniform>\s*\w+\s*:\s*(\w+)", wgsl)
        if not m:
            raise SystemExit(f"uniform {p['name']}: not found in the WGSL")
        wgsl_name = m.group(1)
        theirs = wgsl_offsets(wgsl, wgsl_name)
        for k, (o, _) in fields.items():
            if theirs.get(k) != o:
                raise SystemExit(f"uniform {p['name']}.{k}: reflection says {o}, the WGSL {theirs.get(k)}")
        size = report.parse_structs(wgsl)[wgsl_name][0]
        return {"binding": index, "kind": "uniform", "struct": element["name"], "size": size,
                "fields": {k: {"offset": o, "type": s} for k, (o, s) in fields.items()}}
    if ty["kind"] == "resource" and ty["baseShape"] == "structuredBuffer":
        writable = ty.get("access") == "readWrite"
        return {"binding": index, "kind": "storage" if writable else "read-only-storage"}
    if ty["kind"] == "resource" and ty["baseShape"] == "texture2D":
        return {"binding": index, "kind": "storage-texture", "access": "write-only",
                "format": {"rgba8": "rgba8unorm"}[p["format"]]}
    raise SystemExit(f"parameter {p['name']}: {ty['kind']} {ty.get('baseShape')} has no web binding here")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slangc", default=str(Path.home() / "tools/slang/bin/slangc"))
    ap.add_argument("--no-tint", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    kernels_dir = out / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)
    work = out.parent / f"{out.name}.work"   # reflection and the Tint checker, kept out of what is served
    work.mkdir(parents=True, exist_ok=True)
    naga = shutil.which("naga") or (str(Path.home() / ".cargo/bin/naga")
                                     if (Path.home() / ".cargo/bin/naga").exists() else None)
    tint = None if args.no_tint else report.build_tint(HERE, work)
    shaders = HERE / "shaders"

    manifest = {"format": "athenea-webgpu-kernels", "version": 1, "preset": "T1",
                "limits": LIMITS, "kernels": {}}
    failed = []
    for module, entry in KERNELS:
        wgsl_path = kernels_dir / f"{entry}.wgsl"
        reflection = work / f"{entry}.json"
        run = subprocess.run([args.slangc, str(shaders / f"{module}.slang"), "-I", str(shaders),
                              "-target", "wgsl", "-entry", entry, "-stage", "compute",
                              "-o", str(wgsl_path), "-reflection-json", str(reflection)],
                             capture_output=True, text=True)
        if run.returncode != 0:
            failed.append(f"{entry}: slangc: {run.stderr.strip()[:300]}")
            continue
        wgsl = wgsl_path.read_text()
        info = report.analyse(wgsl)
        verdicts = []
        if naga:
            v = subprocess.run([naga, str(wgsl_path)], capture_output=True, text=True)
            verdicts.append(("naga", "ok" if v.returncode == 0 else v.stderr.strip()[:200]))
        if tint:
            module_says, pipeline_says = report.tint_verdict(tint, wgsl_path)
            verdicts.append(("tint", module_says))
            verdicts.append(("tint-limits", pipeline_says))
        for who, said in verdicts:
            if said != "ok":
                failed.append(f"{entry}: {who}: {said}")
        if info["storage"] > LIMITS["maxStorageBuffersPerShaderStage"]:
            failed.append(f"{entry}: {info['storage']} storage buffers")
        if info["workgroup"] > LIMITS["maxComputeWorkgroupStorageSize"]:
            failed.append(f"{entry}: {info['workgroup']} bytes of workgroup memory")

        data = json.loads(reflection.read_text())
        point = next(e for e in data["entryPoints"] if e["name"] == entry)
        used = {b["name"] for b in point.get("bindings", []) if b["binding"].get("used", 1)}
        params, overrides = {}, {}
        for p in data["parameters"]:
            if p["binding"]["kind"] == "specializationConstant":
                m = re.search(rf"@id\({p['binding']['index']}\)\s*override\s+\w+\s*:\s*\w+\s*=\s*\w*\(?(\d+)",
                              wgsl)
                overrides[p["name"]] = {"id": p["binding"]["index"],
                                        "default": int(m.group(1)) if m else None}
                continue
            if p["name"] not in used or f"@binding({p['binding']['index']})" not in wgsl:
                continue
            params[p["name"]] = parameter(p, wgsl)
        manifest["kernels"][entry] = {
            "file": f"kernels/{entry}.wgsl",
            "source": f"shaders/{module}.slang",
            "workgroupSize": point["threadGroupSize"],
            "parameters": params,
            "overrides": overrides,
            "uses": {"storageBuffers": info["storage"], "uniformBuffers": info["uniforms"],
                     "storageTextures": info["storage_tex"], "workgroupBytes": info["workgroup"]},
            "checked": {who: said for who, said in verdicts},
        }
        print(f"{entry:20s} storage {info['storage']} ({info['storage_ro']}+{info['storage_rw']}) "
              f"uniform {info['uniforms']} texture {info['storage_tex']} workgroup {info['workgroup']:5d} "
              + " ".join(f"{w} {s}" for w, s in verdicts))

    if failed:
        print("\n".join(failed), file=sys.stderr)
        raise SystemExit(1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    page = HERE / "web/athenea-webgpu"
    for f in sorted(page.iterdir()):
        if f.is_file():
            shutil.copy2(f, out / f.name)
    print(f"\nthe web module in {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
