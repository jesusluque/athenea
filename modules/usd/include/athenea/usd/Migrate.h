// Copyright (c) 2026 jesus luque.
//
// lucabRTrender's files under this engine's names. athenea is lucabRTrender
// renamed (e8ef1eb): the codeless schemas Lrt*API are Athenea*API, every
// `lrt:` namespace in a property's name is `athenea:` (primvars, render
// settings, outputs), hydra:rendererName's "lrt" is "athenea", and a .lrtc is
// a .athc. A stage written before the rename names none of what this engine
// reads; `migrate` writes it again under the new names, layer by layer, with
// the USD API -- values, metadata, time samples, connections and relationship
// targets are moved, not rebuilt.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "athenea/core/Result.h"

namespace athenea::usd {

struct MigrateOptions {
    /// Follows sublayers, references, payloads, value clips and asset-valued
    /// attributes to the layers, packages and .lrtc files they name, and
    /// migrates each that lies under `root` into the same place under the
    /// output's directory, so the copies name each other as the originals do.
    bool recursive = false;
    /// What `recursive` may write a copy of; empty is the input's directory.
    /// It must hold the input.
    std::filesystem::path root;
};

struct MigrateChange {
    enum class Kind {
        Schema,      ///< an apiSchemas entry
        Property,    ///< a property's name (a primvar, an output, a setting)
        Target,      ///< a connection or relationship target naming a renamed property
        Value,       ///< a value naming the renderer
        Metadata,    ///< a customData / customLayerData key
        Asset,       ///< an asset path rewritten (.lrtc to .athc, or to a migrated copy)
        Anchored,    ///< a relative asset path made absolute: its file was not copied
        File,        ///< a file written
        Warning,     ///< something left as it was, and why
    };
    Kind        kind = Kind::Warning;
    std::string file;   ///< the layer (or file) it was found in
    std::string where;  ///< the spec path, where there is one
    std::string from;
    std::string to;
};

struct MigrateReport {
    std::vector<MigrateChange> changes;
    /// Files written, in the order they were.
    std::vector<std::filesystem::path> written;

    [[nodiscard]] size_t count(MigrateChange::Kind kind) const;
    /// Renames of any kind but File and Warning: zero for a file already migrated.
    [[nodiscard]] size_t renames() const;
};

[[nodiscard]] const char* toString(MigrateChange::Kind kind) noexcept;

/// Migrates `in` -- a USD layer (.usda, .usdc, .usd), a package (.usdz) or a
/// .lrtc -- into `out`. A file already migrated is written unchanged and
/// reports no rename. `out` may not be `in`, and with `recursive` no copy may
/// land on its original.
[[nodiscard]] Result<MigrateReport> migrate(const std::filesystem::path& in, const std::filesystem::path& out,
                                            const MigrateOptions& options = {});

}   // namespace athenea::usd
