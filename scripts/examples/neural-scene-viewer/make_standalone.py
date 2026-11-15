#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# THE PLAYER AS ONE FILE. The viewer page (index.html) reads three and Spark from files beside it, through an
# import map, and Spark reads its WebAssembly from a file beside itself. This writes the same page with all of
# that inside it: a page that is one file, to be opened from a disk or put anywhere, that asks nothing of a
# CDN, an import map or a server of its own.
#
#   make_standalone.py --page index.html --vendor <dir with three/ and spark/<build>/> -o player-standalone.html
#
# How: each module's source goes into the page as text; a small script makes an object URL of each, in the
# order they depend on one another, writing into each importer the URL of what it imports -- a module's
# specifier is fixed in its source -- and the WebAssembly, kept as base64, is handed to Spark the same way.
# Spark's WebGPU half and its chunks are left out: the page draws through WebGL.
#
# The page read from a disk cannot fetch the files beside it (a browser's rule for file:), so it asks for the
# scene's folder: the reader picks or drops it, and the page reads the files it was handed.
import argparse
import base64
import glob
import json
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--page", required=True)
parser.add_argument("--vendor", required=True)
parser.add_argument("-o", "--out", required=True)
args = parser.parse_args()


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


builds = sorted(glob.glob(os.path.join(args.vendor, "spark", "*", "spark.module.min.js")))
if not builds:
    sys.exit("no spark/<build>/spark.module.min.js under %s" % args.vendor)
spark_dir = os.path.dirname(builds[-1])
sources = {
    "three-core": read(os.path.join(args.vendor, "three", "three.core.min.js")),
    "three": read(os.path.join(args.vendor, "three", "three.module.min.js")),
    "pass": read(os.path.join(args.vendor, "three", "addons", "postprocessing", "Pass.js")),
    "spark": read(os.path.join(spark_dir, "spark.module.min.js")),
}
with open(os.path.join(spark_dir, "spark_rs_bg.wasm"), "rb") as f:
    wasm = base64.b64encode(f.read()).decode("ascii")

# What each source imports is written as a mark the page's script puts a URL in place of.
MARK = "@@URL:%s@@"


def swap(text, pattern, mark, what, at_least=1):
    out, n = re.subn(pattern, lambda m: m.group(1) + (MARK % mark) + m.group(2), text)
    if n < at_least:
        sys.exit("%s: the import of %s was not found; the vendored build changed" % (what, mark))
    return out


sources["three"] = swap(sources["three"], r'''(from\s*["'])\./three\.core\.min\.js(["'])''', "three-core", "three")
sources["pass"] = swap(sources["pass"], r'''(from\s*["'])three(["'])''', "three", "Pass.js")
sources["spark"] = swap(sources["spark"], r'''(from\s*["'])three(["'])''', "three", "spark")
sources["spark"] = swap(sources["spark"], r'''(from\s*["'])three/addons/postprocessing/Pass\.js(["'])''', "pass", "spark")
# Spark fetches its WebAssembly beside its own file: here, the object URL made of what the page carries.
spark, n = re.subn(r'fetch\(new URL\(\s*\w+\s*,\s*import\.meta\.url\s*\)\)', "fetch(globalThis.__sparkWasm)", sources["spark"])
if n != 1:
    sys.exit("spark: where it fetches its WebAssembly was not found; the vendored build changed")
sources["spark"] = spark

page = read(args.page)
importmap = re.search(r'\s*<script type="importmap">.*?</script>', page, re.S)
main = re.search(r'<script type="module">(.*?)</script>', page, re.S)
if not importmap or not main:
    sys.exit("%s: its import map or its module was not found" % args.page)
sources["main"] = main.group(1)
sources["main"] = swap(sources["main"], r'''(from\s*["'])three(["'])''', "three", "the page")
sources["main"] = swap(sources["main"], r'''(from\s*["'])@sparkjsdev/spark(["'])''', "spark", "the page")


def held(name, text):
    # Text in a script element ends at the first "</script": none is left whole.
    return '<script type="text/plain" id="src-%s">%s</script>\n' % (name, text.replace("</script", "<\\/script"))


order = ["three-core", "three", "pass", "spark", "main"]
loader = r"""
<style>
  #opener { position: fixed; inset: 0; z-index: 50; display: none; align-items: center; justify-content: center;
            background: rgba(8, 10, 14, 0.92); color: #e8ecf1; font: 15px/1.5 system-ui, sans-serif; padding: 24px; }
  #opener.open { display: flex; }
  #opener div { max-width: 34em; text-align: center; }
  #opener button { font: inherit; padding: 10px 18px; border-radius: 8px; border: 1px solid #5b6b7c; background: #1c2732;
                   color: inherit; cursor: pointer; margin-top: 14px; }
  #opener button:focus-visible { outline: 2px solid #7fc4e8; }
  #opener p { margin: 0 0 8px; }
  #opener small { display: block; margin-top: 14px; color: #9aa9b8; }
</style>
<div id="opener" role="dialog" aria-label="Open a scene">
  <div>
    <p><b>Open a scene</b></p>
    <p>This page is one file read from your disk, and a browser does not let such a page read the files beside it.
       Choose the folder that holds the scene (its <code>.json</code> index and its <code>.spz</code> parts), or drop it here.</p>
    <button id="openFolder" type="button">Choose the scene's folder</button>
    <input id="openInput" type="file" webkitdirectory multiple hidden>
    <small id="openSaid"></small>
  </div>
</div>
<script>
(async () => {
  const order = %(order)s;
  const text = (id) => document.getElementById("src-" + id).textContent;
  const made = {};
  for (const id of order) {
    const source = text(id).replace(/@@URL:([\w-]+)@@/g, (_, of) => made[of]);
    if (id === "spark") {
      const bytes = Uint8Array.from(atob(text("wasm")), (c) => c.charCodeAt(0));
      globalThis.__sparkWasm = URL.createObjectURL(new Blob([bytes], { type: "application/wasm" }));
    }
    made[id] = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
  }
  // A SCENE FROM A FOLDER. Read from a disk, with no address given for the scene, the page asks for the
  // folder: the files handed are addressed by object URLs under their own names, and the scene is the index
  // the address names (?scene=), or the one index among them that is neither a group's nor a level's.
  const params = new URLSearchParams(location.search);
  if (location.protocol === "file:" && !params.get("base")) {
    const opener = document.getElementById("opener");
    const input = document.getElementById("openInput");
    const said = document.getElementById("openSaid");
    opener.classList.add("open");
    const files = await new Promise((resolve) => {
      document.getElementById("openFolder").addEventListener("click", () => input.click());
      input.addEventListener("change", () => { if (input.files.length > 0) resolve([...input.files]); });
      opener.addEventListener("dragover", (event) => event.preventDefault());
      opener.addEventListener("drop", async (event) => {
        event.preventDefault();
        const found = [];
        const walk = async (entry) => {
          if (entry.isFile) found.push(await new Promise((ok, no) => entry.file(ok, no)));
          else if (entry.isDirectory) {
            const reader = entry.createReader();
            for (;;) {
              const batch = await new Promise((ok, no) => reader.readEntries(ok, no));
              if (batch.length === 0) break;
              for (const inner of batch) await walk(inner);
            }
          }
        };
        const entries = [...event.dataTransfer.items].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
        for (const entry of entries) await walk(entry);
        if (found.length > 0) resolve(found); else said.textContent = "Nothing was read from what was dropped.";
      });
    });
    const map = new Map(files.map((file) => [file.name, URL.createObjectURL(file)]));
    globalThis.__sceneFiles = map;
    globalThis.__sceneBlobs = new Map(files.map((file) => [file.name, file]));
    if (!params.get("scene")) {
      // The scene's own index holds parts and names no other index as its whole: a group's, a base colour's
      // and scenes.json are told apart by what the others say of them.
      const indexes = [];
      const named = new Set();
      for (const file of files) {
        if (!file.name.endsWith(".json") || file.name === "scenes.json" || file.name === "history.json") continue;
        try {
          const index = JSON.parse(await file.text());
          if (!Array.isArray(index.parts)) continue;
          indexes.push(file.name.slice(0, -5));
          for (const group of index.groups ?? []) named.add(group.stem);
          if (index.base) named.add(index.base);
        } catch { /* not an index */ }
      }
      const scenes = indexes.filter((name) => !named.has(name));
      if (scenes.length === 0) { said.textContent = "That folder holds no scene index (a .json with parts). Reload to choose another."; return; }
      globalThis.__sceneName = scenes.sort((a, b) => a.length - b.length)[0];
    }
    opener.classList.remove("open");
  }
  const module = document.createElement("script");
  module.type = "module";
  module.src = made.main;
  document.body.append(module);
})().catch((error) => { document.getElementById("stats").textContent = "could not start the player: " + error; });
</script>
""" % {"order": json.dumps(order)}

embedded = "".join(held(name, sources[name]) for name in order) + held("wasm", wasm)
out = page.replace(importmap.group(0), "").replace(main.group(0), embedded + loader)
with open(args.out, "w", encoding="utf-8") as f:
    f.write(out)
print("%s: %.1f MB, one file (three %s, spark %s, its WebAssembly %.1f MB as base64)" % (
    args.out, len(out.encode("utf-8")) / 1e6, "r" + (re.search(r'REVISION\s*=\s*"(\d+)"', sources["three-core"]) or [0, "?"])[1],
    os.path.basename(spark_dir), len(wasm) / 1e6))
