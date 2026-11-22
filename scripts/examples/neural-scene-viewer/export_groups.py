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
#
# Where the scene names its prims, the scene's own parts (and the lit levels' and the history's) come with
# which prim each gaussian was made from -- an .ids file beside each, `prims` in the index -- for the player's
# objects. A group's parts and the base colours' are the same gaussians in the same order and are written
# without (--no-prims): the player reads an id from the scene's part.
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
parser.add_argument("--history", action="store_true", help="the scene itself with its history (export --history): what each frame found that a later one took out or refined, so that a player shows the scene whole as it stood at every frame")
parser.add_argument("--levels", default="", help="budgets of gaussians for levels of detail of the scene itself (export --levels): what a player draws first")
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


levels = ["--levels", args.levels] if args.levels else []
# The scene with every group off: its own light alone (what glows is in it once). Its levels of detail, and
# each group's and the base colours' below, are the same gaussians in the same order a level, so a player
# sets the lamps at a level as it does on the whole scene.
export(args.out, ["--group", "*=0"] + levels)
named = []
for group in groups:
    stem = "%s_g_%s" % (args.out, group)
    # The group alone: no light of the scene's own, every other group off, and `dark` taken out of it as a
    # player's level will scale it (it is drawn at minus the levels drawn: one here).
    export(stem, ["--own-light", "0", "--group", "*=0", "--group", "%s=1" % group, "--no-prims"] + levels)
    named.append({"name": group, "stem": os.path.basename(stem)})
index_path = args.out + ".json"
index = json.load(open(index_path))
index["groups"] = named
# WITH THE LAMPS ON: the levels a player draws first (`litLevels`), and the scene's history (`--history`).
# A level of the scene's own parts is unlit until its groups' parts are read; these are levels of the scene with
# every group at one, a picture as the scene was found with nothing more to wait for. And what a frame found
# that a later one took out or refined is drawn only while the timeline is before that frame: it is handed on
# as it was found, its lamps in it, since a player lays a group's part over the scene's standing parts alone --
# without it an earlier frame shows holes where the scene was later made finer. That export's own parts are
# not kept; its levels and its history are.
if args.levels or args.history:
    lit = args.out + "_lit"
    export(lit, ["--group", "*=1"] + levels + (["--history"] if args.history else []))
    lit_index = json.load(open(lit + ".json"))
    kept = set()

    def files_of(part):
        part = part if isinstance(part, dict) else {"file": part}
        glass, mirror = part.get("glass") or {}, part.get("mirror") or {}
        return [f for f in (part.get("file"), glass.get("file"), mirror.get("file"),
                            part.get("ids"), glass.get("ids"), mirror.get("ids")) if f]

    if args.levels:
        index["litLevels"] = lit_index.get("levels", [])
        for level in index["litLevels"]:
            for part in level.get("parts", []):
                kept.update(files_of(part))
    if args.history:
        spans = 0
        for own, lit_part in zip(index["parts"], lit_index["parts"]):
            if isinstance(own, dict) and isinstance(lit_part, dict) and lit_part.get("history"):
                own["history"] = lit_part["history"]
                for span in lit_part["history"]:
                    kept.update(files_of(span))
                    spans += len(span.get("spans") or [1])
        print("history: %d spans, as found" % spans)
    folder = os.path.dirname(os.path.abspath(lit))
    for name in os.listdir(folder):
        if name.startswith(os.path.basename(lit)) and name not in kept:
            os.remove(os.path.join(folder, name))
# And, where the scene keeps them, its base colours: what the player's own lamp is multiplied by. No exposure:
# a base colour is a share, not a light.
if manifest.get("baseColour"):
    export(args.out + "_base", ["--base-colour", "--no-prims"] + levels, plain)
    index["base"] = os.path.basename(args.out) + "_base"
    index["exposure"] = float(args.exposure) if args.exposure else 0.0
json.dump(index, open(index_path, "w"), indent=1)
print("groups: %s%s" % (", ".join(g["name"] for g in named) or "none", "; base colours" if index.get("base") else ""))
