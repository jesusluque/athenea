#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""The MaterialX shader ball as USD, once, on the CPU (no pxr module needed).

Reads MaterialX's resources/Geometry/shaderball.glb (glTF 2.0 binary, two meshes,
one material; matx_common.SHADERBALL, MATX_MX_RESOURCES to point elsewhere) and
writes, through usdcat, <MATX_WORK>/scene/shaderball.usdc:

  /ShaderBall                 Xform, kind component, defaultPrim
    /Preview_Mesh             the shell: sphere, groove, base and its flat ledge
    /Calibration_Mesh         the core under it (MaterialXView puts a calibration chart on it)

glTF is metres and +Y up, so the stage is metersPerUnit 1, upAxis Y, no transform.
UVs become primvars:st with v flipped (glTF's origin is top-left, USD's bottom-left).
The glb has no material subsets (one DefaultMaterial for both meshes), so no
GeomSubsets are written: the study binds one material to both meshes.

  python3 build_ball.py
"""
import json
import os
import struct
import subprocess
import sys

import matx_common as m

GLB = m.SHADERBALL
USDCAT = m.USDCAT
OUT = os.path.join(m.WORK, "scene")


def read_glb(path):
    data = open(path, "rb").read()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        sys.exit(f"{path}: not a glTF 2.0 binary")
    jlen = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + jlen])
    off = 20 + jlen
    blen = struct.unpack_from("<I", data, off)[0]
    return doc, data[off + 8:off + 8 + blen]


def accessor(doc, binary, index):
    a = doc["accessors"][index]
    view = doc["bufferViews"][a["bufferView"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    fmt = {5126: "f", 5125: "I", 5123: "H"}[a["componentType"]]
    size = struct.calcsize(fmt)
    stride = view.get("byteStride") or size * width
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    return [struct.unpack_from("<" + fmt * width, binary, start + k * stride) for k in range(a["count"])]


def fmt3(v):
    return "(%.7g, %.7g, %.7g)" % v


def mesh_usda(name, points, normals, uvs, indices):
    lo = [min(p[a] for p in points) for a in range(3)]
    hi = [max(p[a] for p in points) for a in range(3)]
    tris = len(indices) // 3
    st = ", ".join("(%.7g, %.7g)" % (u, 1.0 - v) for u, v in uvs)
    return (
        f'    def Mesh "{name}"\n    {{\n'
        f"        uniform bool doubleSided = 1\n"
        f"        float3[] extent = [{fmt3(tuple(lo))}, {fmt3(tuple(hi))}]\n"
        f"        int[] faceVertexCounts = [{', '.join(['3'] * tris)}]\n"
        f"        int[] faceVertexIndices = [{', '.join(str(i[0]) for i in indices)}]\n"
        f"        normal3f[] normals = [{', '.join(fmt3(n) for n in normals)}] (\n"
        f'            interpolation = "vertex"\n        )\n'
        f"        point3f[] points = [{', '.join(fmt3(p) for p in points)}]\n"
        f"        texCoord2f[] primvars:st = [{st}] (\n"
        f'            interpolation = "vertex"\n        )\n'
        f'        uniform token orientation = "rightHanded"\n'
        f'        uniform token subdivisionScheme = "none"\n'
        f"    }}\n"
    ), lo, hi, tris


def main():
    os.makedirs(OUT, exist_ok=True)
    doc, binary = read_glb(GLB)
    body = ""
    info = {}
    for node in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        n = doc["nodes"][node]
        if any(k in n for k in ("matrix", "translation", "rotation", "scale")):
            sys.exit(f"node {n['name']} has a transform; this converter expects none")
        mesh = doc["meshes"][n["mesh"]]
        if len(mesh["primitives"]) != 1 or mesh["primitives"][0].get("mode", 4) != 4:
            sys.exit(f"mesh {mesh['name']}: expected one triangle primitive")
        prim = mesh["primitives"][0]
        attrs = prim["attributes"]
        text, lo, hi, tris = mesh_usda(
            mesh["name"], accessor(doc, binary, attrs["POSITION"]), accessor(doc, binary, attrs["NORMAL"]),
            accessor(doc, binary, attrs["TEXCOORD_0"]), accessor(doc, binary, prim["indices"]))
        body += text
        info[mesh["name"]] = {"triangles": tris, "min": lo, "max": hi}
    lo = [min(v["min"][a] for v in info.values()) for a in range(3)]
    hi = [max(v["max"][a] for v in info.values()) for a in range(3)]
    usda = (
        "#usda 1.0\n(\n"
        '    defaultPrim = "ShaderBall"\n'
        "    metersPerUnit = 1\n"
        '    upAxis = "Y"\n'
        '    doc = """The MaterialX shader ball (MaterialX resources/Geometry/shaderball.glb, Apache-2.0,\n'
        '    Academy Software Foundation), converted by athenea-bench/matx/build_ball.py. glTF metres, +Y up."""\n'
        ")\n\n"
        'def Xform "ShaderBall" (\n    kind = "component"\n)\n{\n'
        f"    float3[] extentsHint = [{fmt3(tuple(lo))}, {fmt3(tuple(hi))}]\n"
        + body + "}\n")
    usda_path = os.path.join(OUT, "shaderball.usda")
    open(usda_path, "w").write(usda)
    usdc_path = os.path.join(OUT, "shaderball.usdc")
    subprocess.run([USDCAT, usda_path, "-o", usdc_path], check=True)
    os.remove(usda_path)   # the text form is 10 MB; the usdc is what the stages reference
    json.dump({"source": os.path.relpath(GLB, m.MX_RESOURCES), "resources": "MaterialX resources", "meshes": info, "bounds": [lo, hi]}, open(os.path.join(OUT, "shaderball.json"), "w"),
              indent=1)
    print("wrote", usdc_path, json.dumps(info))


if __name__ == "__main__":
    main()
