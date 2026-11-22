#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# A SURFACE'S GRAPH AS THE ENGINE READS IT, derived from MaterialX's own. MaterialX's nodegraph for a surface
# shader (OpenPBR Surface's NG_open_pbr_surface_surfaceshader in libraries/bxdf/open_pbr_surface.mtlx, Autodesk
# Standard Surface's NG_standard_surface_surfaceshader_100 in bxdf/standard_surface.mtlx, glTF PBR's
# IMPL_gltf_pbr_surfaceshader in bxdf/gltf_pbr.mtlx) is copied with every node whose type is BSDF or VDF taken
# out and one node (athenea_open_pbr_bsdf, athenea_standard_surface_bsdf, athenea_gltf_pbr_bsdf) put before the
# emission layer under the name the surface constructor reads, fed by the nodes that fed the
# closures; the nodedef and the implementation of that node are written before it. The node's code is
# shaders/athenea/material/mx/athenea_<surface>_bsdf.slang, written by hand from the closures this takes out:
# when MaterialX is updated, run this again and read the diff of the .mtlx (a node the closures read that has
# changed its name or its inputs shows there), then run the test that holds each node to its graph --
# athenea_material_tests "[openpbr]", "[standardsurface]" and "[gltfpbr]" -- which says whether the node still builds the
# stack MaterialX's graph builds (docs/development.md, 1.5).
#
#   make_surface_node_mtlx.py <MaterialX libraries> [<surface> [<output .mtlx>]]
#
# <MaterialX libraries> is the folder that holds bxdf/ (~/tools/usd-26.08-mx/libraries as scripts/build-usd.sh
# installs it). <surface> is open_pbr, standard_surface or gltf_pbr; without one, every surface is written. The output
# defaults to the file in the repository this script is in.
import os
import re
import sys

SURFACES = {}

SURFACES["open_pbr"] = {
    "source": "open_pbr_surface.mtlx",
    "graph": "NG_open_pbr_surface_surfaceshader",
    "output": "athenea_open_pbr_surface.mtlx",
    "before": "    <!-- Emission Layer -->",
    "node": '''
    <!-- The lobes: MaterialX's fuzz over coat over the metal and the
         dielectric base (reflection over transmission, subsurface and
         diffuse), as one node (athenea_open_pbr_bsdf.slang says why). -->
    <athenea_open_pbr_bsdf name="fuzz_layer" type="BSDF">
      <input name="fuzz_weight" type="float" interfacename="fuzz_weight" />
      <input name="fuzz_color" type="color3" interfacename="fuzz_color" />
      <input name="fuzz_roughness" type="float" interfacename="fuzz_roughness" />
      <input name="coat_weight" type="float" interfacename="coat_weight" />
      <input name="coat_ior" type="float" interfacename="coat_ior" />
      <input name="coat_roughness" type="vector2" nodename="coat_roughness_vector" />
      <input name="specular_weight" type="float" interfacename="specular_weight" />
      <input name="thin_film_weight" type="float" interfacename="thin_film_weight" />
      <input name="metal_reflectivity" type="color3" nodename="metal_reflectivity" />
      <input name="specular_color" type="color3" interfacename="specular_color" />
      <input name="roughness" type="vector2" nodename="main_roughness" />
      <input name="ior" type="float" nodename="modulated_eta_s" />
      <input name="transmission_tint" type="color3" nodename="if_transmission_tint" />
      <input name="transmission_weight" type="float" interfacename="transmission_weight" />
      <input name="subsurface_color" type="color3" nodename="subsurface_color_nonnegative" />
      <input name="subsurface_brdf_factor" type="color3" nodename="subsurface_thin_walled_brdf_factor" />
      <input name="subsurface_btdf_factor" type="color3" nodename="subsurface_thin_walled_btdf_factor" />
      <input name="subsurface_weight" type="float" interfacename="subsurface_weight" />
      <input name="thin_walled" type="float" nodename="subsurface_selector" />
      <input name="base_weight" type="float" interfacename="base_weight" />
      <input name="base_color" type="color3" nodename="base_color_nonnegative" />
      <input name="base_diffuse_roughness" type="float" interfacename="base_diffuse_roughness" />
      <input name="base_metalness" type="float" interfacename="base_metalness" />
      <input name="base_darkening" type="color3" nodename="modulated_base_darkening" />
      <input name="coat_attenuation" type="color3" nodename="coat_attenuation" />
      <input name="normal" type="vector3" interfacename="geometry_normal" />
      <input name="coat_normal" type="vector3" interfacename="geometry_coat_normal" />
      <input name="tangent" type="vector3" interfacename="geometry_tangent" />
      <input name="coat_tangent" type="vector3" interfacename="geometry_coat_tangent" />
    </athenea_open_pbr_bsdf>
''',
    "head": '''<?xml version="1.0"?>
<materialx version="1.39" colorspace="lin_rec709">
  <!--
    OpenPBR Surface as the engine reads it: MaterialX's own nodegraph
    (libraries/bxdf/open_pbr_surface.mtlx, 1.39.5) with every node that is
    not a closure kept as it is (the roughnesses, the modulated index, the
    coat's darkening, the emission) and its thirty BSDF nodes (eleven lobes
    and the layer, mix, multiply and add over them) made one,
    athenea_open_pbr_bsdf, which pushes the same lobes in the same order with
    the same weights and layer marks. The graph kept 45 BSDF values in thread
    memory, each kBuildLobes wide, at every vertex of a path.
    ATHENEA_OPENPBR_GRAPH in the environment keeps MaterialX's graph
    (MaterialCompiler::create), to compare the two.

    The nodes that fed only the volume (the absorption and scattering
    coefficients of <anisotropic_vdf>) are left where they were and reach
    nothing: the generator writes only what the output reads, and the
    engine's layer of a BSDF over that VDF multiplied by one.
  -->
  <nodedef name="ND_athenea_open_pbr_bsdf" node="athenea_open_pbr_bsdf" nodegroup="pbr">
    <input name="fuzz_weight" type="float" value="0" />
    <input name="fuzz_color" type="color3" value="1, 1, 1" />
    <input name="fuzz_roughness" type="float" value="0.5" />
    <input name="coat_weight" type="float" value="0" />
    <input name="coat_ior" type="float" value="1.6" />
    <input name="coat_roughness" type="vector2" value="0, 0" />
    <input name="specular_weight" type="float" value="1" />
    <input name="thin_film_weight" type="float" value="0" />
    <input name="metal_reflectivity" type="color3" value="0.8, 0.8, 0.8" />
    <input name="specular_color" type="color3" value="1, 1, 1" />
    <input name="roughness" type="vector2" value="0.09, 0.09" />
    <input name="ior" type="float" value="1.5" />
    <input name="transmission_tint" type="color3" value="1, 1, 1" />
    <input name="transmission_weight" type="float" value="0" />
    <input name="subsurface_color" type="color3" value="0.8, 0.8, 0.8" />
    <input name="subsurface_brdf_factor" type="color3" value="0.8, 0.8, 0.8" />
    <input name="subsurface_btdf_factor" type="color3" value="0.8, 0.8, 0.8" />
    <input name="subsurface_weight" type="float" value="0" />
    <input name="thin_walled" type="float" value="0" />
    <input name="base_weight" type="float" value="1" />
    <input name="base_color" type="color3" value="0.8, 0.8, 0.8" />
    <input name="base_diffuse_roughness" type="float" value="0" />
    <input name="base_metalness" type="float" value="0" />
    <input name="base_darkening" type="color3" value="1, 1, 1" />
    <input name="coat_attenuation" type="color3" value="1, 1, 1" />
    <input name="normal" type="vector3" defaultgeomprop="Nworld" />
    <input name="coat_normal" type="vector3" defaultgeomprop="Nworld" />
    <input name="tangent" type="vector3" defaultgeomprop="Tworld" />
    <input name="coat_tangent" type="vector3" defaultgeomprop="Tworld" />
    <output name="out" type="BSDF" />
  </nodedef>
  <implementation name="IM_athenea_open_pbr_bsdf_genslang" nodedef="ND_athenea_open_pbr_bsdf" file="athenea/material/mx/athenea_open_pbr_bsdf.slang" function="athenea_open_pbr_bsdf" target="genslang" />

''',
}

SURFACES["standard_surface"] = {
    "source": "standard_surface.mtlx",
    "graph": "NG_standard_surface_surfaceshader_100",
    "output": "athenea_standard_surface.mtlx",
    "before": "    <!-- Emission Layer -->",
    "node": '''
    <!-- The lobes: MaterialX's coat over the metal and the dielectric base
         (reflection over transmission and, under the sheen, subsurface and
         diffuse), as one node (athenea_standard_surface_bsdf.slang says
         why). -->
    <athenea_standard_surface_bsdf name="coat_layer" type="BSDF">
      <input name="coat" type="float" interfacename="coat" />
      <input name="coat_ior" type="float" interfacename="coat_IOR" />
      <input name="coat_roughness" type="vector2" nodename="coat_roughness_vector" />
      <input name="metalness" type="float" interfacename="metalness" />
      <input name="metal_ior" type="color3" nodename="artistic_ior" output="ior" />
      <input name="metal_extinction" type="color3" nodename="artistic_ior" output="extinction" />
      <input name="specular" type="float" interfacename="specular" />
      <input name="specular_color" type="color3" interfacename="specular_color" />
      <input name="specular_ior" type="float" interfacename="specular_IOR" />
      <input name="roughness" type="vector2" nodename="main_roughness" />
      <input name="transmission" type="float" interfacename="transmission" />
      <input name="transmission_color" type="color3" interfacename="transmission_color" />
      <input name="transmission_roughness" type="vector2" nodename="transmission_roughness" />
      <input name="sheen" type="float" interfacename="sheen" />
      <input name="sheen_color" type="color3" interfacename="sheen_color" />
      <input name="sheen_roughness" type="float" interfacename="sheen_roughness" />
      <input name="subsurface" type="float" interfacename="subsurface" />
      <input name="subsurface_color" type="color3" nodename="coat_affected_subsurface_color" />
      <input name="thin_walled" type="float" nodename="subsurface_selector" />
      <input name="base" type="float" interfacename="base" />
      <input name="base_color" type="color3" nodename="coat_affected_diffuse_color" />
      <input name="diffuse_roughness" type="float" interfacename="diffuse_roughness" />
      <input name="coat_attenuation" type="color3" nodename="coat_attenuation" />
      <input name="normal" type="vector3" interfacename="normal" />
      <input name="coat_normal" type="vector3" interfacename="coat_normal" />
      <input name="tangent" type="vector3" nodename="main_tangent" />
      <input name="coat_tangent" type="vector3" nodename="coat_tangent" />
    </athenea_standard_surface_bsdf>
''',
    "head": '''<?xml version="1.0"?>
<materialx version="1.39" colorspace="lin_rec709">
  <!--
    Autodesk Standard Surface as the engine reads it: MaterialX's own
    nodegraph (libraries/bxdf/standard_surface.mtlx, 1.39.5) with every node
    that is not a closure kept as it is (the roughnesses, the rotated
    tangents, the coat's gamma on the colours, the artistic index, the
    emission) and its sixteen BSDF nodes (eight lobes and the layer, mix and
    multiply over them) made one, athenea_standard_surface_bsdf, which pushes
    the same lobes in the same order with the same weights and layer marks.
    The graph kept 29 BSDF values in thread memory, each kBuildLobes wide, at
    every vertex of a path. ATHENEA_STANDARD_SURFACE_GRAPH in the environment
    keeps MaterialX's graph (MaterialCompiler::create), to compare the two.

    The node that fed only the subsurface's radius (subsurface_radius_scaled)
    is left where it was and reaches nothing: the generator writes only what
    the output reads, and the engine's subsurface is a diffuse lobe.
  -->
  <nodedef name="ND_athenea_standard_surface_bsdf" node="athenea_standard_surface_bsdf" nodegroup="pbr">
    <input name="coat" type="float" value="0" />
    <input name="coat_ior" type="float" value="1.5" />
    <input name="coat_roughness" type="vector2" value="0.01, 0.01" />
    <input name="metalness" type="float" value="0" />
    <input name="metal_ior" type="color3" value="0.18, 0.42, 1.37" />
    <input name="metal_extinction" type="color3" value="3.42, 2.35, 1.77" />
    <input name="specular" type="float" value="1" />
    <input name="specular_color" type="color3" value="1, 1, 1" />
    <input name="specular_ior" type="float" value="1.5" />
    <input name="roughness" type="vector2" value="0.04, 0.04" />
    <input name="transmission" type="float" value="0" />
    <input name="transmission_color" type="color3" value="1, 1, 1" />
    <input name="transmission_roughness" type="vector2" value="0.04, 0.04" />
    <input name="sheen" type="float" value="0" />
    <input name="sheen_color" type="color3" value="1, 1, 1" />
    <input name="sheen_roughness" type="float" value="0.3" />
    <input name="subsurface" type="float" value="0" />
    <input name="subsurface_color" type="color3" value="1, 1, 1" />
    <input name="thin_walled" type="float" value="0" />
    <input name="base" type="float" value="0.8" />
    <input name="base_color" type="color3" value="1, 1, 1" />
    <input name="diffuse_roughness" type="float" value="0" />
    <input name="coat_attenuation" type="color3" value="1, 1, 1" />
    <input name="normal" type="vector3" defaultgeomprop="Nworld" />
    <input name="coat_normal" type="vector3" defaultgeomprop="Nworld" />
    <input name="tangent" type="vector3" defaultgeomprop="Tworld" />
    <input name="coat_tangent" type="vector3" defaultgeomprop="Tworld" />
    <output name="out" type="BSDF" />
  </nodedef>
  <implementation name="IM_athenea_standard_surface_bsdf_genslang" nodedef="ND_athenea_standard_surface_bsdf" file="athenea/material/mx/athenea_standard_surface_bsdf.slang" function="athenea_standard_surface_bsdf" target="genslang" />

''',
}

SURFACES["gltf_pbr"] = {
    "source": "gltf_pbr.mtlx",
    "graph": "IMPL_gltf_pbr_surfaceshader",
    "output": "athenea_gltf_pbr.mtlx",
    "before": "    <!-- Emission -->",
    "node": '''
    <!-- The lobes: MaterialX's clearcoat over sheen over the metal and the
         dielectric base (reflection over transmission and diffuse), each
         reflection under the iridescence and bare, as one node
         (athenea_gltf_pbr_bsdf.slang says why). -->
    <athenea_gltf_pbr_bsdf name="clearcoat_layer" type="BSDF">
      <input name="clearcoat" type="float" interfacename="clearcoat" />
      <input name="clearcoat_roughness" type="vector2" nodename="clearcoat_roughness_uv" />
      <input name="sheen" type="float" nodename="sheen_intensity" />
      <input name="sheen_color" type="color3" nodename="sheen_color_normalized" />
      <input name="sheen_roughness" type="float" nodename="sheen_roughness_sq" />
      <input name="iridescence" type="float" interfacename="iridescence" />
      <input name="base_color" type="color3" interfacename="base_color" />
      <input name="metallic" type="float" interfacename="metallic" />
      <input name="dielectric_f0" type="color3" nodename="dielectric_f0" />
      <input name="dielectric_f90" type="color3" nodename="dielectric_f90" />
      <input name="roughness" type="vector2" nodename="roughness_uv" />
      <input name="transmission" type="float" interfacename="transmission" />
      <input name="ior" type="float" interfacename="ior" />
      <input name="normal" type="vector3" interfacename="normal" />
      <input name="clearcoat_normal" type="vector3" interfacename="clearcoat_normal" />
      <input name="tangent" type="vector3" nodename="selected_tangent" />
      <input name="clearcoat_tangent" type="vector3" interfacename="tangent" />
    </athenea_gltf_pbr_bsdf>
''',
    "head": '''<?xml version="1.0"?>
<materialx version="1.39" colorspace="lin_rec709">
  <!--
    glTF PBR as the engine reads it: MaterialX's own nodegraph
    (libraries/bxdf/gltf_pbr.mtlx, 1.39.5) with every node that is not a
    closure kept as it is (the dielectric's F0 and F90, the roughnesses, the
    sheen's intensity, the turned tangent, the emission, the alpha modes) and
    its fifteen BSDF nodes (eight lobes and the layer and mix over them) and
    one VDF made one, athenea_gltf_pbr_bsdf, which pushes the same lobes in
    the same order with the same weights and layer marks. The graph kept its
    BSDF values in thread memory, each kBuildLobes wide, at every vertex of a
    path. ATHENEA_GLTF_PBR_GRAPH in the environment keeps MaterialX's graph
    (MaterialCompiler::create), to compare the two.

    The nodes that fed only the volume (the attenuation coefficient of
    <anisotropic_vdf>, which the graph itself left unread) are left where
    they were and reach nothing: the generator writes only what the output
    reads.
  -->
  <nodedef name="ND_athenea_gltf_pbr_bsdf" node="athenea_gltf_pbr_bsdf" nodegroup="pbr">
    <input name="clearcoat" type="float" value="0" />
    <input name="clearcoat_roughness" type="vector2" value="0, 0" />
    <input name="sheen" type="float" value="0" />
    <input name="sheen_color" type="color3" value="0, 0, 0" />
    <input name="sheen_roughness" type="float" value="0" />
    <input name="iridescence" type="float" value="0" />
    <input name="base_color" type="color3" value="1, 1, 1" />
    <input name="metallic" type="float" value="1" />
    <input name="dielectric_f0" type="color3" value="0.04, 0.04, 0.04" />
    <input name="dielectric_f90" type="color3" value="1, 1, 1" />
    <input name="roughness" type="vector2" value="1, 1" />
    <input name="transmission" type="float" value="0" />
    <input name="ior" type="float" value="1.5" />
    <input name="normal" type="vector3" defaultgeomprop="Nworld" />
    <input name="clearcoat_normal" type="vector3" defaultgeomprop="Nworld" />
    <input name="tangent" type="vector3" defaultgeomprop="Tworld" />
    <input name="clearcoat_tangent" type="vector3" defaultgeomprop="Tworld" />
    <output name="out" type="BSDF" />
  </nodedef>
  <implementation name="IM_athenea_gltf_pbr_bsdf_genslang" nodedef="ND_athenea_gltf_pbr_bsdf" file="athenea/material/mx/athenea_gltf_pbr_bsdf.slang" function="athenea_gltf_pbr_bsdf" target="genslang" />

''',
}


def derive(libraries, name, output):
    surface = SURFACES[name]
    src = open(os.path.join(libraries, "bxdf", surface["source"])).read()
    a = src.index('  <nodegraph name="' + surface["graph"] + '"')
    b = src.index("</nodegraph>", a) + len("</nodegraph>")
    g = src[a:b]
    # Every BSDF and VDF node out.
    g, n = re.subn(r'\n    <(\w+) name="\w+" type="(?:BSDF|VDF)">.*?</\1>', "", g, flags=re.S)
    print(name + ":", n, "closure nodes removed", file=sys.stderr)
    k = g.index(surface["before"])
    g = g[:k] + surface["node"].lstrip("\n") + "\n" + g[k:]
    open(output, "w").write(surface["head"] + g + "\n</materialx>\n")
    print("written", os.path.normpath(output), file=sys.stderr)


if len(sys.argv) < 2 or (len(sys.argv) > 2 and sys.argv[2] not in SURFACES):
    sys.exit("usage: make_surface_node_mtlx.py <MaterialX libraries> [" + "|".join(SURFACES) + " [<output .mtlx>]]")
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shaders", "athenea", "material", "mx")
for name in [sys.argv[2]] if len(sys.argv) > 2 else list(SURFACES):
    derive(sys.argv[1], name, sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, SURFACES[name]["output"]))
