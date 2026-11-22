#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# A STAGE TO QUALIFY MATERIALS ON. Fourteen balls, each of one material, on a chequered floor before a wall
# of coloured stripes -- something for glass to bend and for chrome to mirror -- under a sky and a sun. What
# `neural-scene shoot` makes of each material is then measured ball by ball against the path tracer
# (qa.sh). Written with nothing but the standard library: the stage as text, its two textures as PNGs.
#
#   make_materials_stage.py <directory>     writes materials.usda, walk.txt, held.txt, balls.txt, textures/
import math
import os
import struct
import sys
import zlib

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(os.path.join(out, "textures"), exist_ok=True)


def png(path, width, height, pixel):
    rows = b"".join(b"\x00" + bytes(c for x in range(width) for c in pixel(x, y)) for y in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


# A chequer of two greys with a coloured mark a square, so that which way up and which way round is seen.
def chequer(x, y):
    cx, cy = x // 32, y // 32
    base = 200 if (cx + cy) % 2 == 0 else 70
    if x % 32 < 6 and y % 32 < 6:
        return (220, 60, 40) if cx % 2 == 0 else (40, 90, 220)
    return (base, base, base)


# A normal map of round bumps: tangent space, +z out.
def bumps(x, y):
    u, v = (x % 32) / 32.0 - 0.5, (y % 32) / 32.0 - 0.5
    r2 = u * u + v * v
    if r2 > 0.2:
        return (128, 128, 255)
    nx, ny = -u * 2.2, -v * 2.2
    nz = math.sqrt(max(1.0 - nx * nx - ny * ny, 0.05))
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    return (int(127.5 + 127.5 * nx / n), int(127.5 + 127.5 * ny / n), int(127.5 + 127.5 * nz / n))


png(os.path.join(out, "textures", "chequer.png"), 256, 256, chequer)
png(os.path.join(out, "textures", "bumps.png"), 256, 256, bumps)

# name, how it is said, OpenPBR inputs
OPEN_PBR = [
    ("Matte", "a matt white", {"base_color": (0.8, 0.8, 0.8), "specular_roughness": 1.0, "specular_weight": 0.0}),
    ("PlasticRough", "a rough red plastic", {"base_color": (0.7, 0.08, 0.06), "specular_roughness": 0.5}),
    ("PlasticGloss", "a glossy blue plastic", {"base_color": (0.05, 0.15, 0.7), "specular_roughness": 0.08}),
    ("MetalRough", "a rough gold", {"base_color": (1.0, 0.78, 0.34), "base_metalness": 1.0, "specular_roughness": 0.5}),
    ("MetalSatin", "a satin copper", {"base_color": (0.95, 0.64, 0.54), "base_metalness": 1.0, "specular_roughness": 0.22}),
    ("Chrome", "chrome", {"base_color": (0.95, 0.95, 0.95), "base_metalness": 1.0, "specular_roughness": 0.03}),
    ("Glass", "a clear glass", {"base_color": (1, 1, 1), "transmission_weight": 1.0, "specular_roughness": 0.0, "specular_ior": 1.5}),
    ("GlassTinted", "a green glass", {"base_color": (1, 1, 1), "transmission_weight": 1.0, "transmission_color": (0.55, 0.9, 0.6), "specular_roughness": 0.0, "specular_ior": 1.5}),
    ("GlassFrosted", "a frosted glass", {"base_color": (1, 1, 1), "transmission_weight": 1.0, "specular_roughness": 0.35, "specular_ior": 1.5}),
    ("Emissive", "a lamp", {"base_color": (0.1, 0.1, 0.1), "emission_luminance": 4.0, "emission_color": (1.0, 0.55, 0.2)}),
    ("Coated", "a red under a clear coat", {"base_color": (0.6, 0.03, 0.03), "specular_roughness": 0.6, "coat_weight": 1.0, "coat_roughness": 0.03}),
    ("Water", "water", {"base_color": (1, 1, 1), "transmission_weight": 1.0, "specular_roughness": 0.0, "specular_ior": 1.33}),
]
PREVIEW = [
    ("Textured", "a chequered texture", "chequer", False),
    ("Bumped", "a grey with a normal map", None, True),
]
NAMES = [m[0] for m in OPEN_PBR] + [m[0] for m in PREVIEW]
SAID = [m[1] for m in OPEN_PBR] + [m[1] for m in PREVIEW]


def value(v):
    return "(%s)" % ", ".join("%g" % c for c in v) if isinstance(v, tuple) else "%g" % v


def kind(v):
    return "color3f" if isinstance(v, tuple) else "float"


def ball(u_count=48, v_count=24):
    points, st, counts, indices = [], [], [], []
    for v in range(v_count + 1):
        phi = math.pi * v / v_count
        for u in range(u_count + 1):
            theta = 2 * math.pi * u / u_count
            points.append((math.sin(phi) * math.cos(theta), math.cos(phi), math.sin(phi) * math.sin(theta)))
            st.append((4.0 * u / u_count, 2.0 * (1.0 - v / v_count)))
    for v in range(v_count):
        for u in range(u_count):
            a = v * (u_count + 1) + u
            counts.append(4)
            indices += [a, a + 1, a + u_count + 2, a + u_count + 1]
    return points, st, counts, indices


points, st, counts, indices = ball()
RADIUS = 0.25
PITCH = 0.8
ROW = 7

lines = ['#usda 1.0', '(', '    defaultPrim = "World"', '    metersPerUnit = 1', '    upAxis = "Y"', ')', '',
         '# Fourteen balls of one material each, to qualify what a found scene makes of a material',
         '# (scripts/examples/neural-scene-qa). Made by make_materials_stage.py.', '',
         'def Xform "World"', '{',
         '    def DomeLight "Sky"', '    {', '        float inputs:intensity = 1.0',
         '        color3f inputs:color = (0.62, 0.74, 1.0)', '    }',
         '    def DistantLight "Sun"', '    {', '        float inputs:intensity = 5.0', '        float inputs:angle = 2',
         '        color3f inputs:color = (1, 0.95, 0.86)', '        float xformOp:rotateX = -50',
         '        float xformOp:rotateY = 30', '        uniform token[] xformOpOrder = ["xformOp:rotateY", "xformOp:rotateX"]',
         '    }', '', '    def Scope "Materials"', '    {']
for name, _, inputs in OPEN_PBR:
    lines += ['        def Material "%s"' % name, '        {',
              '            token outputs:mtlx:surface.connect = </World/Materials/%s/Surface.outputs:out>' % name,
              '            def Shader "Surface"', '            {',
              '                uniform token info:id = "ND_open_pbr_surface_surfaceshader"']
    lines += ['                %s inputs:%s = %s' % (kind(v), k, value(v)) for k, v in inputs.items()]
    lines += ['                token outputs:out', '            }', '        }']
for name, _, texture, bumped in PREVIEW + [("Floor", "", "chequer", False)]:
    lines += ['        def Material "%s"' % name, '        {',
              '            token outputs:surface.connect = </World/Materials/%s/Surface.outputs:surface>' % name,
              '            def Shader "Surface"', '            {',
              '                uniform token info:id = "UsdPreviewSurface"',
              '                float inputs:roughness = %g' % (0.9 if name == "Floor" else 0.45),
              '                float inputs:metallic = 0']
    if texture:
        lines += ['                color3f inputs:diffuseColor.connect = </World/Materials/%s/Colour.outputs:rgb>' % name]
    else:
        lines += ['                color3f inputs:diffuseColor = (0.5, 0.5, 0.5)']
    if bumped:
        lines += ['                normal3f inputs:normal.connect = </World/Materials/%s/Normal.outputs:rgb>' % name]
    lines += ['                token outputs:surface', '            }',
              '            def Shader "Coords"', '            {',
              '                uniform token info:id = "UsdPrimvarReader_float2"',
              '                string inputs:varname = "st"', '                float2 outputs:result', '            }']
    if texture:
        lines += ['            def Shader "Colour"', '            {',
                  '                uniform token info:id = "UsdUVTexture"',
                  '                asset inputs:file = @./textures/%s.png@' % texture,
                  '                token inputs:sourceColorSpace = "sRGB"',
                  '                token inputs:wrapS = "repeat"', '                token inputs:wrapT = "repeat"',
                  '                float2 inputs:st.connect = </World/Materials/%s/Coords.outputs:result>' % name,
                  '                float3 outputs:rgb', '            }']
    if bumped:
        lines += ['            def Shader "Normal"', '            {',
                  '                uniform token info:id = "UsdUVTexture"',
                  '                asset inputs:file = @./textures/bumps.png@',
                  '                token inputs:sourceColorSpace = "raw"',
                  '                float4 inputs:scale = (2, 2, 2, 1)', '                float4 inputs:bias = (-1, -1, -1, 0)',
                  '                token inputs:wrapS = "repeat"', '                token inputs:wrapT = "repeat"',
                  '                float2 inputs:st.connect = </World/Materials/%s/Coords.outputs:result>' % name,
                  '                float3 outputs:rgb', '            }']
    lines += ['        }']
STRIPES = [(0.85, 0.1, 0.1), (0.95, 0.75, 0.1), (0.1, 0.65, 0.2), (0.1, 0.35, 0.9), (0.8, 0.8, 0.8), (0.55, 0.1, 0.7)]
for k, colour in enumerate(STRIPES):
    lines += ['        def Material "Stripe%d"' % k, '        {',
              '            token outputs:surface.connect = </World/Materials/Stripe%d/Surface.outputs:surface>' % k,
              '            def Shader "Surface"', '            {',
              '                uniform token info:id = "UsdPreviewSurface"',
              '                color3f inputs:diffuseColor = %s' % value(colour),
              '                float inputs:roughness = 0.8', '                token outputs:surface', '            }', '        }']
lines += ['    }', '']


def mesh(name, pts, uvs, cnt, idx, material, xform=None, normals=None):
    block = ['    def Mesh "%s" (' % name, '        prepend apiSchemas = ["MaterialBindingAPI"]', '    )', '    {',
             '        rel material:binding = </World/Materials/%s>' % material,
             '        uniform token subdivisionScheme = "none"',
             '        int[] faceVertexCounts = [%s]' % ", ".join(map(str, cnt)),
             '        int[] faceVertexIndices = [%s]' % ", ".join(map(str, idx)),
             '        point3f[] points = [%s]' % ", ".join("(%.5f, %.5f, %.5f)" % p for p in pts)]
    if uvs:
        block += ['        texCoord2f[] primvars:st = [%s] (' % ", ".join("(%.4f, %.4f)" % t for t in uvs),
                  '            interpolation = "vertex"', '        )']
    if normals:
        block += ['        normal3f[] normals = [%s] (' % ", ".join("(%.5f, %.5f, %.5f)" % n for n in normals),
                  '            interpolation = "vertex"', '        )']
    if xform:
        block += ['        double3 xformOp:translate = (%g, %g, %g)' % xform,
                  '        uniform token[] xformOpOrder = ["xformOp:translate"]']
    return block + ['    }']


# The floor, 12 m by 8 m, its chequer a square every 25 cm; and the wall behind, six stripes.
lines += mesh("Floor", [(-6, 0, -3), (6, 0, -3), (6, 0, 5), (-6, 0, 5)], [(0, 0), (12, 0), (12, 8), (0, 8)], [4],
              [0, 3, 2, 1], "Floor")
for k in range(len(STRIPES)):
    x0, x1 = -6 + 2 * k, -4 + 2 * k
    lines += mesh("Wall%d" % k, [(x0, 0, -3), (x1, 0, -3), (x1, 3, -3), (x0, 3, -3)], None, [4], [0, 1, 2, 3], "Stripe%d" % k)
centres = []
for k, name in enumerate(NAMES):
    row, column = divmod(k, ROW)
    centre = ((column - (ROW - 1) / 2) * PITCH, RADIUS, -row * 1.1)
    centres.append(centre)
    lines += mesh(name, [(p[0] * RADIUS, p[1] * RADIUS, p[2] * RADIUS) for p in points], st, counts, indices, name, centre, points)
lines += ['}', '']
open(os.path.join(out, "materials.usda"), "w").write("\n".join(lines))

# THE LOOKS: an arc before the balls, a metre and a half up, looking at their middle. And the views the
# measures are taken from, which no look took: between the looks, lower and higher.
target = (0.0, 0.3, -0.5)
walk, held = [], []
for k in range(24):
    a = math.radians(-55 + 110 * k / 23)
    walk.append("w%02d %.3f %.3f %.3f %.3f %.3f %.3f 0 1 0" % (k, 4.2 * math.sin(a), 1.5, 4.2 * math.cos(a) - 0.5, *target))
for k, (angle, height, far) in enumerate([(-40, 0.9, 3.6), (-12, 2.1, 3.9), (17, 1.0, 3.4), (43, 1.9, 4.0)]):
    a = math.radians(angle)
    held.append("h%d %.3f %.3f %.3f %.3f %.3f %.3f 0 1 0" % (k, far * math.sin(a), height, far * math.cos(a) - 0.5, *target))
open(os.path.join(out, "walk.txt"), "w").write("\n".join(walk) + "\n")
open(os.path.join(out, "held.txt"), "w").write("\n".join(held) + "\n")
# Where each ball is, for whoever measures them apart: its name, its centre, its radius, how it is said.
open(os.path.join(out, "balls.txt"), "w").write(
    "\n".join("%s %.3f %.3f %.3f %.3f %s" % (n, *c, RADIUS, s) for n, c, s in zip(NAMES, centres, SAID)) + "\n")
print("%d balls; materials.usda, walk.txt (24 looks), held.txt (4 views), balls.txt" % len(NAMES))
