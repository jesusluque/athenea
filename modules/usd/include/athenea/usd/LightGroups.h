// Copyright (c) 2026 jesus luque.
//
// LIGHT GROUPS: A CAR'S LAMPS SWITCHED, DIMMED AND TINTED AT RUN TIME, with
// nothing baked again (docs/decisions.md, light groups; proposals 062/066).
//
// Light is linear. A group of lamps k -- the low beam, the tail lamps -- is
// baked once with every other light off and the group at radiance one, and
// what each gaussian sends out then is its layer: so every state of the car
// is
//
//   colour = base + sum_k w_k layer_k
//   w_k    = level_k x emissionLuminance_k x emissionColor_k
//
// where `base` is the cloud with every lamp off and the levels come from a
// state the stage names and from per-group levels a host sets.
//
//   the file    the cloud's layers, one primvar a group,
//               `primvars:athenea:splat:lightLayer:<group>` (color3f[], linear,
//               per unit of the group's radiance), written by `athenea lights
//               merge` from one cloud a group (`mergeLightLayers`)
//   the stage   the groups and states as namespaced attributes on Scope prims
//               (the `.lights.usda` sidecar, until AtheneaLightGroupAPI):
//               children of a `LightGroups` Scope are groups
//               (`athenea:lightGroup:emissionLuminance`, `:emissionColor`,
//               `:initialOn`), children of a `LightStates` Scope are states
//               (`athenea:lightState:target:<group>`, a level each; a group a
//               state does not name is at 0)
//   the frame   the render settings `athenea:lightState` (a state's name;
//               empty: each group at `initialOn`) and `athenea:lightLevel:<group>`
//               (a level that stands over the state's), resolved into the
//               weights once a frame (`LightRig::weights`) -- a few numbers,
//               the only thing that changes when a lamp is switched.
//
// Rules, sequences and time curves (a blinker, a xenon lamp warming up) are
// read by the web side and not here yet: a state and a level are what a
// frame takes.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <pxr/usd/usd/stage.h>

#include "athenea/core/Result.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::usd {

/// A group of lamps as the stage says it: its name, its radiance on, and
/// whether it starts on.
struct LightGroup {
    std::string          name;
    std::string          prim;
    /// `athenea:lightGroup:emissionLuminance` (or `:radiance`), 1 where it
    /// says nothing: linear radiance in scene units, what the layer is per
    /// unit of.
    float                luminance = 1.0F;
    /// `athenea:lightGroup:emissionColor` (or `:color`), white where it says
    /// nothing.
    std::array<float, 3> colour{1.0F, 1.0F, 1.0F};
    bool                 initialOn = false;
};

/// A state: the level of each group it names (`athenea:lightState:target:<group>`).
struct LightState {
    std::string                  name;
    std::string                  prim;
    std::map<std::string, float> targets;
};

/// THE STAGE'S LAMPS: every group and state it carries, by name -- the first
/// prim of a name wins, so a car and its light catcher that each carry the
/// sidecar's Scopes are one rig.
struct LightRig {
    std::vector<LightGroup> groups;
    std::vector<LightState> states;

    [[nodiscard]] bool empty() const noexcept { return groups.empty(); }
    [[nodiscard]] const LightState* state(const std::string& name) const;

    /// The level of each group: `levels[group]` where it is there, else the
    /// state's target where `state` names one (0 for a group it does not
    /// name), else 1 for a group that starts on and 0 for one that does not.
    [[nodiscard]] std::map<std::string, float> levelsFor(const std::string& state,
                                                         const std::map<std::string, float>& levels) const;
    /// What each group's layer is added times: its level times its radiance
    /// and colour, linear rgb.
    [[nodiscard]] std::map<std::string, std::array<float, 3>> weights(
        const std::string& state, const std::map<std::string, float>& levels) const;
};

/// The rig a stage carries; empty where it carries none.
[[nodiscard]] LightRig readLightRig(const pxr::UsdStageRefPtr& stage);

/// THE RENDER SETTINGS THAT SAY WHICH LAMPS ARE LIT: `athenea:lightState`,
/// and `athenea:lightLevel:` followed by a group's name.
inline constexpr const char* kLightStateSetting = "athenea:lightState";
inline constexpr const char* kLightLevelSetting = "athenea:lightLevel:";

/// One group's layer cloud, for `mergeLightLayers`.
struct LightLayerFile {
    std::string           group;
    std::filesystem::path path;
};

/// What `athenea lights merge` is asked.
struct LightMergeOptions {
    std::filesystem::path       base;     ///< the cloud with every lamp off
    std::vector<LightLayerFile> layers;   ///< one cloud a group, the same gaussians
    /// The `.lights.usda` sidecar whose LightGroups, LightStates and
    /// LightRules Scopes are copied under the cloud's prim; empty copies none.
    std::filesystem::path       lights;
    std::filesystem::path       output;
    std::string                 prim;     ///< the ParticleField; empty: the first
    /// How far a layer's position may be from the base's, relative to the
    /// coordinate (and absolute below 1).
    float                       tolerance = 1.0e-4F;
    /// A layer none of whose gaussians sends out more than this per unit of
    /// radiance is dropped, said so; 0 drops only a layer that is all black.
    float                       dropBelow = 0.0F;
};

/// What it did, group by group.
struct LightMergeReport {
    uint32_t gaussians = 0;
    struct Layer {
        std::string group;
        uint32_t    lit = 0;        ///< gaussians above `dropBelow`
        bool        dropped = false;
    };
    std::vector<Layer>       layers;
    std::vector<std::string> copied;   ///< the sidecar's Scopes, as written
};

/// THE BASE CLOUD WITH ITS LAMPS: each layer cloud checked against the base
/// -- the same number of gaussians, each where the base's is, counted on the
/// device -- its constant term made light per unit of radiance on the device
/// (light_layers.slang's `lightLayerFromCloud`), and written onto a copy of
/// the base as `primvars:athenea:splat:lightLayer:<group>`. Refuses a layer
/// whose gaussians are not the base's.
[[nodiscard]] Result<LightMergeReport> mergeLightLayers(gpu::ShaderLibrary& library,
                                                        const LightMergeOptions& options);

}   // namespace athenea::usd
