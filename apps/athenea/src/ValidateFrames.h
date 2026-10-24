// Copyright (c) 2026 jesus luque.
//
// WHAT EACH MEASURED MATERIAL'S FRAME LEAVES OUT (Mesh2SplatValidate.h): the
// meshes kept out of its conversion and the meshes its frame switches off.
// Bookkeeping over prim paths, apart from the measure so a test can hold it.
#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "athenea/usd/MeshStage.h"

namespace athenea::cli {

/// A name a file can have: the material prim's last element.
inline std::string validateFileName(const std::string& material) {
    if (material.empty()) {
        return "no_material";
    }
    std::string name = material.substr(material.find_last_of('/') + 1);
    for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
            c = '_';
        }
    }
    return name;
}

/// One measured material's frame.
struct ValidateFrame {
    usd::MaterialGroup       group;    ///< the material measured
    std::vector<std::string> hidden;   ///< left out of its conversion, sorted
    std::vector<std::string> off;      ///< switched off in its frame, sorted
};

/// The frames of a measure: one for each of `stage`'s groups that `materials`
/// names (prim paths or names; empty names every one). `hidden` is the job's
/// own, left out of everything.
///
/// NARROWING THE MEASURE NARROWS WHAT IS MEASURED, NOT THE STAGE. What a
/// frame leaves out is worked out over every group of the stage: a cloud
/// that stands for the whole stage is drawn with all of its meshes off, and
/// a conversion of one material leaves every other material's meshes out.
/// Worked out over the named groups alone, a flattened set measured on one
/// material was drawn with the rest of its meshes still standing in the
/// cloud -- black, and fighting the gaussians for depth -- which read as the
/// file's panels and walls coming back black.
inline std::vector<ValidateFrame> validateFrames(const std::vector<usd::MaterialGroup>& stage,
                                                 const std::vector<std::string>& materials,
                                                 const std::vector<std::string>& hidden, bool wholeCloud) {
    const auto sorted = [](std::vector<std::string>& paths) {
        std::sort(paths.begin(), paths.end());
        paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
    };
    const auto named = [&](const usd::MaterialGroup& g) {
        return materials.empty() || std::any_of(materials.begin(), materials.end(), [&](const std::string& want) {
                   return want == g.material || want == validateFileName(g.material);
               });
    };
    std::vector<ValidateFrame> frames;
    for (const usd::MaterialGroup& group : stage) {
        if (!named(group)) {
            continue;
        }
        ValidateFrame frame;
        frame.group = group;
        // ONLY THIS MATERIAL'S MESHES CONVERTED: every other mesh of the
        // stage is left out of the conversion, and stays a mesh in the frame.
        frame.hidden = hidden;
        for (const usd::MaterialGroup& other : stage) {
            for (const std::string& mesh : other.meshes) {
                if (!std::binary_search(group.meshes.begin(), group.meshes.end(), mesh)) {
                    frame.hidden.push_back(mesh);
                }
            }
        }
        sorted(frame.hidden);
        frame.off = group.meshes;
        if (wholeCloud) {
            for (const usd::MaterialGroup& other : stage) {
                frame.off.insert(frame.off.end(), other.meshes.begin(), other.meshes.end());
            }
        }
        sorted(frame.off);
        frames.push_back(std::move(frame));
    }
    return frames;
}

}   // namespace athenea::cli
