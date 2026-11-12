// Copyright (c) 2026 jesus luque.
//
// A NEURAL SCENE ON DISK: a directory, `<name>.aths`, of a scene kept in
// pieces. A piece is a part of the scene that is converted, fitted, loaded
// and changed on its own -- a floor of a building, a prop, a character: its
// gaussians, and the neural appearance they are drawn by. The manifest says
// what the pieces are and where they stand, what lights and cameras the
// scene was made with, and when each piece last changed.
//
//   <name>.aths/
//     scene.json                 the manifest: this header's structs
//     pieces/<id>.usda|.usdc|…   a piece's gaussians (a USD ParticleField as
//                                athenea mesh2splat writes one, or a splat file)
//     pieces/<id>.athn           its neural appearance (NeuralFile.h), once fitted
//
// Every path in the manifest is relative to the directory, so a scene moves
// as a directory does. The manifest is read and written here and nothing is
// computed: counts, bounds and measures are what the device said when the
// piece was added or fitted.
//
// `revision` goes up each time a piece's gaussians or appearance are written,
// and `static` says its gaussians do not move from frame to frame: what a
// renderer keeps of a piece (its matt colours, a still frame's geometry) it
// keeps while the revision stands. Two stamps say what kind of change the
// last one was -- where the piece stands, or what it is made of -- since
// what must be done again differs: a piece that only moved keeps its matt
// colours, a piece that was only repainted keeps its place in every sort.
//
// A SCENE THAT IS FOUND, NOT ONLY CONVERTED. A piece says where it came from
// (`origin`): converted whole from geometry, discovered by a render -- the
// gaussians one camera saw at one time, placed in the world -- or generated
// by something that fills what no camera has seen. `snapshots` is the
// scene's history: each a time, the camera that looked, and the pieces it
// added or changed, so that the scene at a time is the pieces of the
// snapshots up to it.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "athenea/core/Result.h"

namespace athenea::io {

/// What a piece's appearance was fitted to, and how close it came.
struct NeuralSceneFit {
    uint32_t    steps = 0;
    std::string against;        ///< "harmonics": the piece's own baked harmonics
    double      relMse = 0.0;
    double      p99 = 0.0;      ///< 8-bit code values
    uint32_t    max = 0;
};

struct NeuralScenePiece {
    std::string            id;            ///< a file-safe name, unique in the scene
    std::string            prim;          ///< where it came from in the source stage, if anywhere
    std::string            cloud;         ///< its gaussians, relative to the scene
    uint32_t               gaussians = 0;
    std::array<float, 3>   boundsMin{};
    std::array<float, 3>   boundsMax{};
    /// Where the piece stands: object to world, row major, identity where it
    /// was converted in the scene's own space.
    std::array<double, 16> transform{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::string            appearance;    ///< its `.athn`, relative to the scene; empty: its harmonics
    bool                   lit = false;   ///< the appearance answers a light rather than holding one
    bool                   split = false; ///< two networks: what is the same from everywhere, and what the view adds
    uint32_t               revision = 1;
    bool                   isStatic = true;
    /// The `revision` at which the piece last moved, and at which what it is
    /// made of last changed (its gaussians' colours, its appearance).
    uint32_t               movedAt = 1;
    uint32_t               paintedAt = 1;
    /// "converted" (from geometry, whole), "discovered" (what a camera saw)
    /// or "generated" (filled in by another system); and, where a camera
    /// found it, which and at what time.
    std::string            origin = "converted";
    std::string            foundBy;
    double                 foundAt = 0.0;
    std::optional<NeuralSceneFit> fit;
};

struct NeuralSceneLight {
    std::string          name;
    std::string          kind;                       ///< "distant" | "dome"
    std::array<float, 3> towards{0.0F, 0.0F, 1.0F};  ///< a distant light: a unit vector towards it
    std::array<float, 3> radiance{1.0F, 1.0F, 1.0F};
};

struct NeuralSceneCamera {
    std::string           name;
    std::array<double, 3> eye{};
    std::array<double, 3> target{};
    std::array<double, 3> up{0.0, 1.0, 0.0};
    double                focal = 35.0;   ///< mm
};

/// One look at the scene: when, through which camera, and the pieces it
/// added or changed.
struct NeuralSceneSnapshot {
    double                   time = 0.0;
    std::string              camera;
    std::vector<std::string> pieces;
};

struct NeuralScene {
    static constexpr uint32_t kVersion = 1;

    std::string name;
    std::string upAxis = "Y";             ///< "Y" | "Z"
    double      metersPerUnit = 1.0;
    std::string source;                   ///< the stage the pieces were converted from
    std::string licence;                  ///< what the source asks of whoever uses it
    std::vector<NeuralScenePiece>  pieces;
    std::vector<NeuralSceneLight>  lights;
    std::vector<NeuralSceneCamera> cameras;
    std::vector<NeuralSceneSnapshot> snapshots;

    [[nodiscard]] NeuralScenePiece* piece(const std::string& id);
    /// The box round every piece's own, as the manifest has them.
    void bounds(std::array<float, 3>& min, std::array<float, 3>& max) const;
};

/// `directory`/scene.json.
[[nodiscard]] Result<NeuralScene> readNeuralScene(const std::filesystem::path& directory);
/// Writes the manifest, making the directory and its `pieces` where they are not.
[[nodiscard]] Result<void> writeNeuralScene(const std::filesystem::path& directory, const NeuralScene& scene);

}   // namespace athenea::io
