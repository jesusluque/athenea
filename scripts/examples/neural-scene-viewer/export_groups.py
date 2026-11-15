#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# A SCENE HANDED ON FOR THE PLAYER WITH ITS LAMPS AS GROUPS. The player draws a scene's own light and each
# group of lights' times a level it sets (index.html: "groups of lights"), and wants for that the scene with
# every group off and, a group, the same gaussians with that group's light alone as their colour. This runs
# `neural-scene export` once for each -- the same options, so the parts are the same gaussians in the same
# order -- and names the groups in the scene's index.
#
#   export_groups.py --athenea <binary> --scene <scene.aths> --out <dir>/<name> [--exposure S] [-- more export options]
#
# Writes <name>.json and its parts (the scene, groups off), <name>_g_<group>.json and its parts a group,
# <name>_base.json and its parts (each gaussian's base colour, for the player's own lamp), and "groups" and
# "base" in <name>.json. Nothing here computes on the data: it runs the engine and edits an index.
import argparse
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--athenea", required=True)
parser.add_argument("--scene", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--exposure", default="")
parser.add_argument("more", nargs="*", help="more options for export, after --")
args = parser.parse_args()

manifest = json.load(open(os.path.join(args.scene, "scene.json")))
groups = [g for g in manifest.get("lightGroups", []) if g != "dark"]
plain = ["--by-frame"] + args.more
common = plain + (["--exposure", args.exposure] if args.exposure else [])


def export(stem, levels, options=None):
    done = subprocess.run([args.athenea, "neural-scene", "export", args.scene] + (common if options is None else options) +
                          levels + ["-o", stem],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    said = [line for line in done.stdout.splitlines() if line.strip() and "info]" not in line]
    print(said[-1][:160] if said else "(rc %d)" % done.returncode)
    if done.returncode != 0:
        sys.exit(done.returncode)


# The scene with every group off: its own light alone (what glows is in it once).
export(args.out, ["--group", "*=0"])
named = []
for group in groups:
    stem = "%s_g_%s" % (args.out, group)
    # The group alone: no light of the scene's own, every other group off, and `dark` taken out of it as a
    # player's level will scale it (it is drawn at minus the levels drawn: one here).
    export(stem, ["--own-light", "0", "--group", "*=0", "--group", "%s=1" % group])
    named.append({"name": group, "stem": os.path.basename(stem)})
index_path = args.out + ".json"
index = json.load(open(index_path))
index["groups"] = named
# And, where the scene keeps them, its base colours: what the player's own lamp is multiplied by. No exposure:
# a base colour is a share, not a light.
if manifest.get("baseColour"):
    export(args.out + "_base", ["--base-colour"], plain)
    index["base"] = os.path.basename(args.out) + "_base"
    index["exposure"] = float(args.exposure) if args.exposure else 0.0
json.dump(index, open(index_path, "w"), indent=1)
print("groups: %s%s" % (", ".join(g["name"] for g in named) or "none", "; base colours" if index.get("base") else ""))
