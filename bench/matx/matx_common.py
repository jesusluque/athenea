# Copyright (c) 2026 jesus luque.
"""What the matx study shares: paths, the two phases' material lists, the skies,
and the lobe classes read off each .mtlx (automatic, no per-material table)."""
import os
import xml.etree.ElementTree as ET

HOME = os.path.expanduser("~")


def _first(env, *candidates):
    """An environment override, else the first candidate that exists (else the first)."""
    if os.environ.get(env):
        return os.path.expanduser(os.environ[env])
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# THE SOURCE (this directory, in git): the scripts, manifest.json, baseline.csv,
# scene/base.usda.in, openpbr.sha256.
SRC = os.path.dirname(os.path.abspath(__file__))
# THE WORK DIRECTORY (generated, never in git): scene/shaderball.usdc, scene/base.usda,
# materials/, stages/, cache/openpbr/. Default: this directory, whose .gitignore keeps them out.
WORK = os.path.expanduser(os.environ.get("MATX_WORK", SRC))
# THE MANIFEST: the one in git beside the scripts, unless MATX_MANIFEST names another (the
# ctest gate writes its own under MATX_WORK, so a test never rewrites a tracked file).
MANIFEST = os.path.expanduser(os.environ.get("MATX_MANIFEST", os.path.join(SRC, "manifest.json")))
# THE RENDERS: what the GPU sweep writes (runs/, results/, summary, contact sheets).
RENDERS = os.path.expanduser(os.environ.get("MATX_OUT", os.path.join(HOME, "luc/athenea-renders/matx")))
ATHENEA = os.path.expanduser(os.environ.get("MATX_BIN", os.path.join(HOME, "luc/athenea-tx/build/macos-arm64-release/bin/athenea")))
USD_BIN = _first("MATX_USD_BIN", os.path.join(HOME, "tools/usd-26.08-mx/bin"))
USDCAT = os.path.join(USD_BIN, "usdcat")
USDCHECKER = os.path.join(USD_BIN, "usdchecker")

# MaterialX's resources: the shader ball, the examples, goegap. The toolchain's own
# MaterialX source (scripts/build-usd.sh) first.
MX_RESOURCES = _first("MATX_MX_RESOURCES",
                      os.path.join(HOME, "tools/usd-26.08-mx/src/MaterialX-1.39.5/resources"),
                      os.path.join(HOME, "tools/src/MaterialX-1.39.5/resources"),
                      os.path.join(HOME, "tools/usd-26.08-blender/resources"))
SHADERBALL = os.path.join(MX_RESOURCES, "Geometry/shaderball.glb")
MX_EXAMPLES = os.path.join(MX_RESOURCES, "Materials/Examples")
AUTOSHOP = _first("MATX_AUTOSHOP", os.path.join(HOME, "tools/assets/hdri/autoshop_01_4k.hdr"))
OPENPBR_COMMIT = "f8d6d947dfae4c9b599965a86c22826ea7a8dbfb"
OPENPBR_URL = f"https://raw.githubusercontent.com/AcademySoftwareFoundation/OpenPBR/{OPENPBR_COMMIT}/examples/"
OPENPBR_CACHE = os.path.join(WORK, "cache/openpbr")

# The stage's own dome is autoshop; "goegap" is swapped in with --validate-sky.
SKIES = {
    "autoshop": None,
    "goegap": os.path.join(MX_RESOURCES, "Lights/goegap.hdr"),
}

# THE GATE'S SUBSET (ctest -L matx): one material a lobe class, phase 1.
GATE = ["mx_open_pbr_default", "mx_standard_surface_chrome", "mx_open_pbr_carpaint", "mx_open_pbr_glass",
        "mx_open_pbr_velvet", "mx_open_pbr_ketchup"]


# THE GT CACHE: a path-traced GT is kept per material x sky and handed back to --validate, so
# every run of a stage measures its cloud against the same GT and the GT's own noise cannot
# move a result. Shared by the sweep and the ctest gate (MATX_OUT does not move it).
GT_CACHE = os.path.expanduser(os.environ.get("MATX_GT_CACHE", os.path.join(HOME, "luc/athenea-renders/matx/gt_cache")))
# What re-renders a GT besides its stage, its size and its paths:
#  - the path tracer's shaders, as the build copied them (pt_fingerprint: every directory the
#    path tracer imports from, and technique/ but for its splat_* kernels; the conversion's
#    usd/, splat/ and lod/ are left out, so a change to the bake or the raster keeps the GT);
#  - GT_EPOCH, bumped by hand in the commit that changes how the path tracer's kernel is
#    generated or bound in C++ (technique/src/PathTracer.cpp, usd/src/StageRenderer.cpp),
#    which no shader file shows.
GT_EPOCH = 1
GT_BOUNCES = 6     # --validate-bounces' default; part of the key
PT_SHADER_DIRS = ("algo", "common", "geom", "light", "material", "rt", "scene", "volume", "world", "technique")


_DIGESTS = {}


def _sha(h, path):
    """Feeds `h` a file's digest (each file read once a process: the skies are tens of MB)."""
    if path not in _DIGESTS:
        import hashlib
        d = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                d.update(block)
        _DIGESTS[path] = d.digest()
    h.update(_DIGESTS[path])


def shader_dir():
    """The shaders the binary runs: ATHENEA_SHADER_DIR, else the build's, beside bin/."""
    if os.environ.get("ATHENEA_SHADER_DIR"):
        return os.environ["ATHENEA_SHADER_DIR"]
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ATHENEA))), "shaders")


def pt_fingerprint():
    import hashlib
    root = os.path.join(shader_dir(), "athenea")
    if not os.path.isdir(root):
        raise SystemExit(f"no shaders at {root} (ATHENEA_SHADER_DIR): the GT cache cannot key the path tracer")
    h = hashlib.sha256(f"epoch {GT_EPOCH}".encode())
    for d in PT_SHADER_DIRS:
        for base, dirs, files in sorted(os.walk(os.path.join(root, d))):
            dirs.sort()
            for f in sorted(files):
                if d == "technique" and f.startswith("splat_"):
                    continue
                path = os.path.join(base, f)
                h.update(os.path.relpath(path, root).encode())
                _sha(h, path)
    return h.hexdigest()[:16]


def gt_key(entry, sky, size, paths, fingerprint):
    """What a GT depends on: the stage and every file it composes (the base scene, the ball, the
    material and its textures), the sky that replaces the dome, the frame, the paths, the bounces
    and the path tracer."""
    import hashlib
    import re
    stage = stage_path(entry)
    material = os.path.join(WORK, "materials", entry["id"] + ".usda")
    files = [stage, os.path.join(WORK, "scene", "base.usda"), os.path.join(WORK, "scene", "shaderball.usdc"), material]
    files += sorted(set(re.findall(r"@(/[^@]+)@", open(material).read())))
    files += [AUTOSHOP] + ([SKIES[sky]] if SKIES[sky] else [])
    h = hashlib.sha256(f"{sky} {size} {paths} {GT_BOUNCES} {fingerprint}".encode())
    for f in files:
        h.update(os.path.basename(f).encode())
        _sha(h, f)
    return h.hexdigest()[:20]


def gt_name(sky):
    """The file --validate reads its GT from (Mesh2SplatValidate.cpp: gt.exr, or gt_<sky's stem>.exr)."""
    return "gt.exr" if not SKIES[sky] else "gt_" + os.path.splitext(os.path.basename(SKIES[sky]))[0] + ".exr"


def stage_path(entry):
    """manifest.json keeps paths portable: a stage relative to WORK, a .mtlx relative to its root."""
    return os.path.join(WORK, entry["stage"])


def mtlx_path(entry):
    root = {"mx": MX_EXAMPLES, "openpbr": OPENPBR_CACHE}[entry["root"]]
    return os.path.join(root, entry["mtlx"])


# PHASE 1: every MaterialX example of the three families asked for. Skipped, with why.
MX_FAMILIES = ("OpenPbr", "StandardSurface", "DisneyPrincipled")
MX_SKIPPED = {
    "standard_surface_chess_set": "15 materials keyed to the OpenChessSet's pieces and UVs (an asset's look, not one material)",
    "standard_surface_look_brass_tiled": "a look (materialassigns): brass_tiled on the shell + greysphere_calibration on the core, both already in the list",
    "standard_surface_look_wood_tiled": "a look (materialassigns): wood_tiled on the shell + greysphere_calibration on the core, both already in the list",
}

# PHASE 2: proposal 086 §2, the part that sits on the shader ball and comes from
# the OpenPBR repository (Apache-2.0, pinned commit). The rest of the 52 is
# listed in PHASE2_DEFERRED with what it waits on.
PHASE2_OPENPBR = [
    # metal 7
    "gold", "silver", "copper", "chromium", "iron", "titanium", "stainless_steel",
    # anisotropy
    "aluminum_brushed",
    # opaque dielectric 6
    "gray_card", "default", "office_paper", "tire", "concrete", "plastic_pp",
    # coat
    "carpaint",
    # glass and transmission 6
    "glass", "diamond", "ice", "water", "plastic_acrylic", "sapphire",
    # thin wall
    "soapbubble",
    # volume with absorption 3
    "coffee", "blood", "honey_liquid",
    # subsurface 6
    "marble", "milk", "skin_ii", "skin_v", "ketchup", "chocolate",
    # fuzz
    "velvet",
    # thin film (coat + thin film + SSS)
    "pearl",
    # emission 2
    "light_bulb_2700k", "lcd_display_6500k",
]
PHASE2_DEFERRED = [
    {"id": "khronos_AnisotropyStrengthTest", "class": "aniso", "why": "Khronos glTF test model: its own mesh and KHR_materials_* extensions; needs a glTF->USD importer that keeps them (usdGltf not in athenea's USD, unverified)"},
    {"id": "khronos_AnisotropyRotationTest", "class": "aniso", "why": "as above"},
    {"id": "khronos_ClearCoatTest", "class": "coat", "why": "as above"},
    {"id": "khronos_TransmissionRoughnessTest", "class": "glass", "why": "as above"},
    {"id": "khronos_TransmissionThinwallTestGrid", "class": "thin wall", "why": "as above"},
    {"id": "khronos_AttenuationTest", "class": "volume", "why": "as above"},
    {"id": "khronos_SheenTestGrid", "class": "sheen", "why": "as above"},
    {"id": "khronos_IridescenceDielectricSpheres", "class": "thin film", "why": "as above"},
    {"id": "khronos_IridescenceMetallicSpheres", "class": "thin film", "why": "as above"},
    {"id": "khronos_DispersionTest", "class": "dispersion", "why": "as above"},
    {"id": "khronos_IORTestGrid", "class": "ior", "why": "as above"},
    {"id": "amd_carbon_bicolor_coat", "class": "coat (textured)", "why": "AMD GPUOpen: licence page unread (API says 'MIT Public Domain'), MaterialX 1.38 + texture package; internal use only, fetch with curl (TLS 1.3)"},
    {"id": "amd_fabric", "class": "sheen (textured)", "why": "as above"},
    {"id": "amd_marble", "class": "normal-mapped", "why": "as above"},
    {"id": "polyhaven_wood", "class": "normal-mapped", "why": "Poly Haven (CC0): asset not yet chosen; mtlx + texture download through api.polyhaven.com/files/<id>"},
    {"id": "polyhaven_stone", "class": "normal-mapped", "why": "as above"},
]

# THE LOBE CLASSES, read off the surface shader's inputs. An input that is
# connected (a graph drives it) counts as present. Every default these test is
# zero in all three nodedefs.
CLASSES = ["emission", "thin film", "glass/transmission", "volume", "thin wall", "subsurface", "sheen/velvet", "coat",
           "anisotropic", "metal", "diffuse/dielectric"]
_KEYS = {
    "open_pbr_surface": {"metal": "base_metalness", "anisotropic": "specular_roughness_anisotropy", "coat": "coat_weight",
                         "glass/transmission": "transmission_weight", "volume": "transmission_depth",
                         "thin wall": "geometry_thin_walled", "subsurface": "subsurface_weight",
                         "sheen/velvet": "fuzz_weight", "thin film": "thin_film_weight", "emission": "emission_luminance"},
    "standard_surface": {"metal": "metalness", "anisotropic": "specular_anisotropy", "coat": "coat",
                         "glass/transmission": "transmission", "volume": "transmission_depth", "thin wall": "thin_walled",
                         "subsurface": "subsurface", "sheen/velvet": "sheen", "thin film": "thin_film_thickness",
                         "emission": "emission"},
    "disney_principled": {"metal": "metallic", "anisotropic": "anisotropic", "coat": "clearcoat",
                          "glass/transmission": "specTrans", "subsurface": "subsurface", "sheen/velvet": "sheen"},
}


def _positive(inp):
    if inp is None:
        return False
    if any(inp.get(k) for k in ("nodename", "nodegraph", "output", "interfacename")):
        return True
    v = (inp.get("value") or "").strip().lower()
    if v in ("true",):
        return True
    try:
        return any(float(x) > 0 for x in v.replace(",", " ").split())
    except ValueError:
        return False


def classify(mtlx_path):
    """(classes, surface node category, textured, document colorspace)."""
    root = ET.parse(mtlx_path).getroot()
    space = root.get("colorspace", "lin_rec709")
    textured = any(e.tag in ("image", "tiledimage", "hextiledimage") for e in root.iter())
    for node in root.iter():
        if node.tag in _KEYS:
            inputs = {i.get("name"): i for i in node.findall("input")}
            found = [c for c, k in _KEYS[node.tag].items() if _positive(inputs.get(k))]
            if "volume" in found and "glass/transmission" not in found:
                found.remove("volume")
            if not any(c in found for c in ("metal", "glass/transmission", "subsurface")):
                found.append("diffuse/dielectric")
            if node.tag == "disney_principled" and "metal" not in found and "diffuse/dielectric" not in found:
                found.append("diffuse/dielectric")
            return [c for c in CLASSES if c in found], node.tag, textured, space
    return [], None, textured, space


def primary(classes):
    return classes[0] if classes else "?"
