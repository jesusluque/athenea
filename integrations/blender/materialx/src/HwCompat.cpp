// Copyright (c) 2026 jesus luque.
//
// The HW constants MaterialX 1.39.5 added and its nodes use, with 1.39.5's
// values (MaterialXGenHw/HwConstants.cpp).
#include <MaterialXGenHw/HwConstants.h>

MATERIALX_NAMESPACE_BEGIN
namespace HW
{
const TypedValue<Vector3> VEC3_ZERO = TypedValue(Vector3(0.f, 0.f, 0.f));
const TypedValue<Vector3> VEC3_ONE = TypedValue(Vector3(1.f, 1.f, 1.f));
}
MATERIALX_NAMESPACE_END
