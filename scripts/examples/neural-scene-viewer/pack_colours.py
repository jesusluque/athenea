#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# A GROUP'S COLOURS IN FILES OF THEIR OWN. The player reads a group of lights -- and the base colours its own
# lamp wants -- as a whole export of the scene, the same gaussians in the same order, for each gaussian's
# colour alone: three bytes of the twenty an SPZ keeps of it. This writes those three bytes beside each file
# an index names, `<file's stem>.rgb`, and names it in the index (`"rgb"` beside `"file"`): the player then
# reads that and not the file. An `.rgb` is gzip over the SPZ's colour stream as it is: three bytes a
# gaussian, in the file's order. Nothing is worked out: bytes are taken from one file and put in another.
#
#   pack_colours.py <scenes dir> <scene>            every group's index and the base colours' of scenes/<scene>.json
#   pack_colours.py <scenes dir> --index <stem> ... the indexes named, whatever scene they are of
#
# Only SPZ of versions 2 and 3 (a 16-byte header and the streams inside one gzip) are read; a file of another
# kind is left as it is and keeps no `rgb`. The index is written whole beside itself and renamed.
import argparse
import gzip
import json
import os
import struct
import sys

parser = argparse.ArgumentParser()
parser.add_argument("scenes")
parser.add_argument("scene", nargs="?")
parser.add_argument("--index", action="append", default=[])
args = parser.parse_args()

MAGIC = 0x5053474E


def colours(path):
    """The colour stream of an SPZ, or None where the file is not one this reads."""
    try:
        with gzip.open(path, "rb") as f:
            body = f.read()
    except OSError:
        return None
    if len(body) < 16:
        return None
    magic, version, count = struct.unpack_from("<III", body, 0)
    if magic != MAGIC or version not in (2, 3):
        return None
    start = 16 + count * 10   # after the header: nine bytes of place and one of opacity a gaussian
    if len(body) < start + count * 3:
        return None
    return body[start:start + count * 3]


def entries(index):
    """Every entry of an index that names a file: parts, their glass and mirrors, their history, the levels'."""
    def of(part):
        if not isinstance(part, dict):
            return
        for span in [part] + list(part.get("history", [])):
            if not isinstance(span, dict):
                continue
            for entry in (span, span.get("glass"), span.get("mirror")):
                if isinstance(entry, dict) and isinstance(entry.get("file"), str):
                    yield entry
    for part in index.get("parts", []):
        yield from of(part)
    for key in ("levels", "litLevels"):
        for level in index.get(key, []):
            for part in level.get("parts", []):
                yield from of(part)


stems = list(args.index)
if args.scene:
    with open(os.path.join(args.scenes, args.scene + ".json"), encoding="utf-8") as f:
        scene = json.load(f)
    stems += [g["stem"] for g in scene.get("groups", [])]
    if scene.get("base"):
        stems.append(scene["base"])
if not stems:
    sys.exit("no index: name a scene with groups, or --index <stem>")

for stem in stems:
    path = os.path.join(args.scenes, stem + ".json")
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    if any(isinstance(p, str) for p in index.get("parts", [])):
        print("%s: parts named by their file alone are left as they are" % stem)
    written = skipped = 0
    before = after = 0
    for entry in entries(index):
        source = os.path.join(args.scenes, entry["file"])
        stream = colours(source)
        if stream is None:
            entry.pop("rgb", None)
            skipped += 1
            continue
        name = os.path.splitext(entry["file"])[0] + ".rgb"
        with open(os.path.join(args.scenes, name), "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as out:
                out.write(stream)
        entry["rgb"] = name
        written += 1
        before += os.path.getsize(source)
        after += os.path.getsize(os.path.join(args.scenes, name))
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    os.replace(path + ".tmp", path)
    print("%s: %d files' colours written (%.1f MB where the files are %.1f MB), %d left as they are"
          % (stem, written, after / 1048576, before / 1048576, skipped))
