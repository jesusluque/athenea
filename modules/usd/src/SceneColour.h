// Copyright (c) 2026 jesus luque.
//
// THE COLOUR SPACE A PRIMVAR WAS AUTHORED IN, AS THE SCENE INDEX CARRIES IT.
//
// A Hydra 1 delegate's Get hands over a primvar's value alone; UsdImaging's
// scene index keeps beside each colour-role primvar the space USD resolves
// for it (`HdPrimvarSchema::GetColorSpace`: the attribute's own colorSpace,
// else an ancestor's ColorSpaceAPI). The values go to the device as
// authored, and a kernel brings them into the working space there
// (colour::ColourCompiler::convertWords).
#pragma once

#include <map>
#include <string>

#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/tokens.h>

PXR_NAMESPACE_OPEN_SCOPE

/// Every colour primvar of `id` -- one with a colour space, or the colour
/// role -- by name, with its space: lin_rec709_scene where it is a colour
/// and none is said (USD's default). Empty where there is none, or no scene
/// index is behind the delegate.
inline std::map<std::string, std::string> primvarColourSpaces(HdSceneDelegate* delegate, const SdfPath& id) {
    std::map<std::string, std::string> spaces;
    const HdSceneIndexBaseRefPtr index = delegate->GetRenderIndex().GetTerminalSceneIndex();
    if (!index) {
        return spaces;
    }
    const HdSceneIndexPrim prim = index->GetPrim(id);
    if (!prim.dataSource) {
        return spaces;
    }
    const HdPrimvarsSchema primvars = HdPrimvarsSchema::GetFromParent(prim.dataSource);
    if (!primvars.IsDefined()) {
        return spaces;
    }
    for (const TfToken& name : primvars.GetPrimvarNames()) {
        const HdPrimvarSchema primvar = primvars.GetPrimvar(name);
        std::string said;
        if (const HdTokenDataSourceHandle space = primvar.GetColorSpace()) {
            said = space->GetTypedValue(0.0F).GetString();
        }
        bool colour = !said.empty();
        if (const HdTokenDataSourceHandle role = primvar.GetRole()) {
            colour = colour || role->GetTypedValue(0.0F) == HdPrimvarRoleTokens->color;
        }
        if (colour) {
            spaces[name.GetString()] = said.empty() ? std::string("lin_rec709_scene") : said;
        }
    }
    return spaces;
}

PXR_NAMESPACE_CLOSE_SCOPE
