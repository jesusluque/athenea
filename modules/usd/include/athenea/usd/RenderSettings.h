// Copyright (c) 2026 jesus luque.
//
// A UsdRender settings prim as Hydra resolved it: the products it names,
// their vars, the purposes it includes and its namespaced settings -- what
// `StageRenderer::renderSettings` reads from the delegate's bprim and
// `renderProducts` renders and writes.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace athenea::usd {

/// One `RenderVar`: the AOV it reads (`sourceName`, an AOV of the delegate's
/// for `sourceType` "raw") under the name its layer takes in the product.
struct RenderVarInfo {
    std::string name;         ///< the var prim's name: the layer's
    std::string sourceName;   ///< "color", "depth", "primId", "Neye", "primvars:st"... (RenderMan's "Ci", "z" read as colour and depth)
    std::string sourceType;   ///< "raw" (the default), "lpe", "primvar"...
    std::string dataType;     ///< as authored: "color3f", "float", "int"...
};

/// One `RenderProduct`: an image of its vars at its resolution from its camera.
struct RenderProductInfo {
    std::string                path;
    std::string                name;   ///< productName: where the image goes
    std::string                type;   ///< productType: "raster"
    uint32_t                   width = 0;
    uint32_t                   height = 0;
    std::string                camera;   ///< the product's camera prim, else the settings'
    bool                       disableMotionBlur = false;
    bool                       disableDepthOfField = false;
    std::vector<RenderVarInfo> vars;
};

/// One `RenderPass` whose `renderSource` is the settings prim: a pipeline's
/// unit of work, with the renderer it names (`HydraRenderPassAPI`, 26.08).
struct RenderPassInfo {
    std::string path;
    std::string passType;       ///< as authored, a pipeline's own vocabulary
    std::string rendererName;   ///< `hydra:rendererName`, empty when unsaid
    bool        forThisRenderer = true;   ///< unsaid, or athenea
};

struct RenderSettingsInfo {
    std::string                        path;
    bool                               active = false;   ///< the scene's active settings prim
    std::vector<std::string>           includedPurposes;   ///< "default", "render", "proxy", "guide"
    std::vector<std::string>           materialBindingPurposes;
    std::string                        renderingColorSpace;
    std::string                        camera;
    bool                               disableMotionBlur = false;
    bool                               disableDepthOfField = false;
    std::vector<RenderProductInfo>     products;
    std::vector<RenderPassInfo>        passes;     ///< the passes that render from this prim
    std::map<std::string, std::string> settings;   ///< the `athenea:` namespaced settings, as text
    uint32_t                           syncs = 0;   ///< how many times Hydra synced the prim (a probe)
};

}   // namespace athenea::usd
