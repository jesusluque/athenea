#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
#
# OPENPBR SURFACE'S GRAPH AS THE ENGINE READS IT, derived from MaterialX's own. MaterialX's nodegraph
# NG_open_pbr_surface_surfaceshader (libraries/bxdf/open_pbr_surface.mtlx) is copied with every node whose
# type is BSDF or VDF taken out and one node, athenea_open_pbr_bsdf, put before the emission layer, fed by the
# nodes that fed the closures; the nodedef and the implementation of that node are written before it. The
# node's code is shaders/athenea/material/mx/athenea_open_pbr_bsdf.slang, written by hand from the closures
# this takes out: when MaterialX is updated, run this again and read the diff of the .mtlx (a node the
# closures read that has changed its name or its inputs shows there), then run the test that holds the node to
# the graph -- athenea_material_tests "[openpbr]" -- which says whether the node still builds the stack
# MaterialX's graph builds (docs/development.md, 1.5).
#
#   make_open_pbr_mtlx.py <MaterialX libraries> [<output .mtlx>]
#
# <MaterialX libraries> is the folder that holds bxdf/open_pbr_surface.mtlx (~/tools/usd-26.08-mx/libraries
# as scripts/build-usd.sh installs it). The output defaults to the file in the repository this script is in.
import os
import re
import sys

if len(sys.argv) < 2:
    sys.exit("usage: make_open_pbr_mtlx.py <MaterialX libraries> [<output .mtlx>]")
libraries = sys.argv[1]
output = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "shaders", "athenea", "material", "mx",
    "athenea_open_pbr_surface.mtlx")
src = open(os.path.join(libraries, "bxdf", "open_pbr_surface.mtlx")).read()
a=src.index('  <nodegraph name="NG_open_pbr_surface_surfaceshader"'); b=src.index('</nodegraph>',a)+len('</nodegraph>')
g=src[a:b]
# drop every BSDF and VDF node
g,n=re.subn(r'\n    <(\w+) name="\w+" type="(?:BSDF|VDF)">.*?</\1>', '', g, flags=re.S)
print(n,'closure nodes removed',file=sys.stderr)
node='''
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
'''
k=g.index('    <!-- Emission Layer -->')
g=g[:k]+node.lstrip('\n')+'\n'+g[k:]
head='''<?xml version="1.0"?>
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

'''
open(output, 'w').write(head + g + '\n</materialx>\n')
print('written', os.path.normpath(output), file=sys.stderr)
