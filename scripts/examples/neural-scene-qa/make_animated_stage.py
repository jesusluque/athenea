#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# A STAGE THAT MOVES, to qualify a found scene over time. A red ball crosses the floor, a chequered cube
# turns where it stands, a chrome ball stays -- and mirrors both -- before the materials stage's striped
# wall, over time codes 0 to 23. Shot through one camera that does not move (`neural-scene shoot --time 0
# --time-step 1`), each frame should take out of the scene what moved and no more, and the scene as it stood
# at each frame (`render --frame`, `export --by-frame --history`) should be that frame.
#
#   make_animated_stage.py <directory>     writes animated.usda, walk.txt, textures/chequer.png
import math
import os
import struct
import sys
import zlib

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(os.path.join(out, "textures"), exist_ok=True)
FRAMES = 24


def png(path, width, height, pixel):
    rows = b"".join(b"\x00" + bytes(c for x in range(width) for c in pixel(x, y)) for y in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def chequer(x, y):
    cx, cy = x // 32, y // 32
    base = 200 if (cx + cy) % 2 == 0 else 70
    if x % 32 < 6 and y % 32 < 6:
        return (220, 60, 40) if cx % 2 == 0 else (40, 90, 220)
    return (base, base, base)


png(os.path.join(out, "textures", "chequer.png"), 256, 256, chequer)


def ball(radius, u_count=48, v_count=24):
    points, normals, st, counts, indices = [], [], [], [], []
    for v in range(v_count + 1):
        phi = math.pi * v / v_count
        for u in range(u_count + 1):
            theta = 2 * math.pi * u / u_count
            n = (math.sin(phi) * math.cos(theta), math.cos(phi), math.sin(phi) * math.sin(theta))
            normals.append(n)
            points.append(tuple(c * radius for c in n))
            st.append((4.0 * u / u_count, 2.0 * (1.0 - v / v_count)))
    for v in range(v_count):
        for u in range(u_count):
            a = v * (u_count + 1) + u
            counts.append(4)
            indices += [a, a + 1, a + u_count + 2, a + u_count + 1]
    return points, normals, st, counts, indices


def cube(half):
    faces = [((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
             ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), ((0, 1, 0), (1, 0, 0), (0, 0, -1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1))]
    points, normals, st, counts, indices = [], [], [], [], []
    for n, u, v in faces:
        base = len(points)
        for su, sv, uv in ((-1, -1, (0, 0)), (1, -1, (1, 0)), (1, 1, (1, 1)), (-1, 1, (0, 1))):
            points.append(tuple(half * (n[k] + su * u[k] + sv * v[k]) for k in range(3)))
            normals.append(n)
            st.append(uv)
        counts.append(4)
        indices += [base, base + 1, base + 2, base + 3]
    return points, normals, st, counts, indices


def material(name, body):
    return ['        def Material "%s"' % name, '        {'] + body + ['        }']


def preview(name, colour=None, texture=None, roughness=0.6, metallic=0.0):
    body = ['            token outputs:surface.connect = </World/Materials/%s/Surface.outputs:surface>' % name,
            '            def Shader "Surface"', '            {', '                uniform token info:id = "UsdPreviewSurface"',
            '                float inputs:roughness = %g' % roughness, '                float inputs:metallic = %g' % metallic]
    if texture:
        body += ['                color3f inputs:diffuseColor.connect = </World/Materials/%s/Colour.outputs:rgb>' % name]
    else:
        body += ['                color3f inputs:diffuseColor = (%g, %g, %g)' % colour]
    body += ['                token outputs:surface', '            }']
    if texture:
        body += ['            def Shader "Coords"', '            {', '                uniform token info:id = "UsdPrimvarReader_float2"',
                 '                string inputs:varname = "st"', '                float2 outputs:result', '            }',
                 '            def Shader "Colour"', '            {', '                uniform token info:id = "UsdUVTexture"',
                 '                asset inputs:file = @./textures/%s.png@' % texture,
                 '                token inputs:sourceColorSpace = "sRGB"', '                token inputs:wrapS = "repeat"',
                 '                token inputs:wrapT = "repeat"',
                 '                float2 inputs:st.connect = </World/Materials/%s/Coords.outputs:result>' % name,
                 '                float3 outputs:rgb', '            }']
    return material(name, body)


def mesh(name, geometry, binding, ops=None):
    points, normals, st, counts, indices = geometry
    block = ['    def Mesh "%s" (' % name, '        prepend apiSchemas = ["MaterialBindingAPI"]', '    )', '    {',
             '        rel material:binding = </World/Materials/%s>' % binding,
             '        uniform token subdivisionScheme = "none"',
             '        int[] faceVertexCounts = [%s]' % ", ".join(map(str, counts)),
             '        int[] faceVertexIndices = [%s]' % ", ".join(map(str, indices)),
             '        point3f[] points = [%s]' % ", ".join("(%.5f, %.5f, %.5f)" % p for p in points)]
    if normals:
        block += ['        normal3f[] normals = [%s] (' % ", ".join("(%.5f, %.5f, %.5f)" % n for n in normals),
                  '            interpolation = "%s"' % ("vertex" if len(normals) == len(points) else "faceVarying"), '        )']
    if st:
        block += ['        texCoord2f[] primvars:st = [%s] (' % ", ".join("(%.4f, %.4f)" % t for t in st),
                  '            interpolation = "vertex"', '        )']
    return block + (ops or []) + ['    }']


STRIPES = [(0.85, 0.1, 0.1), (0.95, 0.75, 0.1), (0.1, 0.65, 0.2), (0.1, 0.35, 0.9), (0.8, 0.8, 0.8), (0.55, 0.1, 0.7)]
lines = ['#usda 1.0', '(', '    defaultPrim = "World"', '    metersPerUnit = 1', '    upAxis = "Y"',
         '    startTimeCode = 0', '    endTimeCode = %d' % (FRAMES - 1), '    timeCodesPerSecond = 24', ')', '',
         '# A stage that moves, to qualify a found scene over time (scripts/examples/neural-scene-qa).',
         '# Made by make_animated_stage.py.', '', 'def Xform "World"', '{',
         '    def DomeLight "Sky"', '    {', '        float inputs:intensity = 1.0',
         '        color3f inputs:color = (0.62, 0.74, 1.0)', '    }',
         '    def DistantLight "Sun"', '    {', '        float inputs:intensity = 5.0', '        float inputs:angle = 2',
         '        color3f inputs:color = (1, 0.95, 0.86)', '        float xformOp:rotateX = -50',
         '        float xformOp:rotateY = 30', '        uniform token[] xformOpOrder = ["xformOp:rotateY", "xformOp:rotateX"]',
         '    }', '', '    def Scope "Materials"', '    {']
lines += preview("Floor", texture="chequer", roughness=0.9)
lines += preview("Cube", texture="chequer", roughness=0.5)
lines += preview("Red", colour=(0.75, 0.08, 0.06), roughness=0.4)
lines += preview("Chrome", colour=(0.95, 0.95, 0.95), roughness=0.03, metallic=1.0)
for k, colour in enumerate(STRIPES):
    lines += preview("Stripe%d" % k, colour=colour, roughness=0.8)
lines += ['    }', '']
lines += mesh("Floor", ([(-6, 0, -3), (6, 0, -3), (6, 0, 5), (-6, 0, 5)], None, [(0, 0), (12, 0), (12, 8), (0, 8)], [4],
                        [0, 3, 2, 1]), "Floor")
for k in range(len(STRIPES)):
    x0, x1 = -6 + 2 * k, -4 + 2 * k
    lines += mesh("Wall%d" % k, ([(x0, 0, -3), (x1, 0, -3), (x1, 3, -3), (x0, 3, -3)], None, None, [4], [0, 1, 2, 3]),
                  "Stripe%d" % k)
# The ball that crosses: a metre and a half either side of the middle, in a straight line, at an even pace.
lines += mesh("Crossing", ball(0.25), "Red", [
    '        double3 xformOp:translate.timeSamples = { 0: (-1.5, 0.25, 0.6), %d: (1.5, 0.25, 0.6) }' % (FRAMES - 1),
    '        uniform token[] xformOpOrder = ["xformOp:translate"]'])
# The cube that turns: half a turn over the frames, where it stands.
lines += mesh("Turning", cube(0.3), "Cube", [
    '        double3 xformOp:translate = (-1.0, 0.3, -0.8)',
    '        float xformOp:rotateY.timeSamples = { 0: 0, %d: 180 }' % (FRAMES - 1),
    '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateY"]'])
# The ball that stays, of chrome: nothing of it moves, and what it shows does.
lines += mesh("Staying", ball(0.35), "Chrome", [
    '        double3 xformOp:translate = (1.1, 0.35, -0.9)', '        uniform token[] xformOpOrder = ["xformOp:translate"]'])
lines += ['}', '']
open(os.path.join(out, "animated.usda"), "w").write("\n".join(lines))
view = "0.0 1.6 4.4 0.0 0.35 -0.4 0 1 0"
open(os.path.join(out, "walk.txt"), "w").write("".join("t%02d %s\n" % (k, view) for k in range(FRAMES)))
open(os.path.join(out, "one.txt"), "w").write("t %s\n" % view)
print("%d frames; animated.usda, walk.txt (one camera, %d times), one.txt" % (FRAMES, FRAMES))
