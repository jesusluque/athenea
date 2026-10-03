// Copyright (c) 2026 jesus luque.
//
// 1.39.4 declares the HW identifiers in HwShaderGenerator.h. Two constants
// 1.39.5's nodes use are new; they are defined in src/HwCompat.cpp.
#pragma once
#include <MaterialXGenHw/Export.h>
#include <MaterialXCore/Value.h>
#include <MaterialXGenShader/HwShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN
namespace HW
{
extern MX_GENHW_API const TypedValue<Vector3> VEC3_ZERO;
extern MX_GENHW_API const TypedValue<Vector3> VEC3_ONE;
}
MATERIALX_NAMESPACE_END
