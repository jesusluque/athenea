// Copyright (c) 2026 jesus luque.
//
// MaterialX 1.39.5 split its hardware generator out of MaterialXGenShader
// into MaterialXGenHw; Blender ships 1.39.4, where it is still inside
// MaterialXGenShader. These headers give the 1.39.5 names over the 1.39.4
// classes, so the Slang generator (1.39.5's MaterialXGenSlang, compiled
// here into MaterialX_v1_39_4) links against Blender's libraries.
#pragma once
#include <MaterialXCore/Library.h>
#define MX_GENHW_API MATERIALX_SYMBOL_EXPORT
