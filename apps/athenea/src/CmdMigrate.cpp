// Copyright (c) 2026 jesus luque.
//
// `athenea migrate`: lucabRTrender's files under this engine's names -- the
// schemas, the `lrt:` properties and settings, and .lrtc clouds -- written as
// copies. The CPU renames and copies; no data is decoded (usd::migrate).
#include "Commands.h"

#include <CLI/CLI.hpp>

#include <cstdio>
#include <memory>
#include <string>

#include "athenea/usd/Migrate.h"

namespace athenea::cli {

void addMigrate(CLI::App& app) {
    struct Options {
        std::string input, output, root;
        bool recursive = false;
        bool quiet = false;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand("migrate",
                                   "rewrite a lucabRTrender stage (.usda/.usdc/.usd/.usdz) or .lrtc under athenea's "
                                   "names, as a copy");
    cmd->add_option("input", o->input, ".usda / .usdc / .usd / .usdz / .lrtc")->required()->check(CLI::ExistingFile);
    cmd->add_option("-o,--output", o->output, "where the copy goes: the same kind of file (.athc for a .lrtc)")
        ->required();
    cmd->add_flag("-r,--recursive", o->recursive,
                  "also migrate the layers, packages and .lrtc files it names that lie under --root, into the same "
                  "places under the output's directory");
    cmd->add_option("--root", o->root, "what --recursive may copy (default: the input's directory)");
    cmd->add_flag("-q,--quiet", o->quiet, "print the totals only");
    cmd->callback([o] {
        usd::MigrateOptions options;
        options.recursive = o->recursive;
        options.root = o->root;
        auto report = usd::migrate(o->input, o->output, options);
        if (!report) {
            cli::fail(report.error());
        }
        using Kind = usd::MigrateChange::Kind;
        for (const usd::MigrateChange& c : report->changes) {
            if (o->quiet && c.kind != Kind::Warning) {
                continue;
            }
            if (c.kind == Kind::File) {
                std::printf("%-9s %s -> %s\n", usd::toString(c.kind), c.from.c_str(), c.to.c_str());
            } else {
                std::printf("%-9s %s %s: %s%s%s\n", usd::toString(c.kind), c.file.c_str(), c.where.c_str(),
                            c.from.c_str(), c.to.empty() ? "" : " -> ", c.to.c_str());
            }
        }
        std::printf("migrate: %zu files written; %zu schemas, %zu properties, %zu targets, %zu values, %zu metadata "
                    "keys, %zu assets renamed, %zu paths anchored, %zu warnings\n",
                    report->written.size(), report->count(Kind::Schema), report->count(Kind::Property),
                    report->count(Kind::Target), report->count(Kind::Value), report->count(Kind::Metadata),
                    report->count(Kind::Asset), report->count(Kind::Anchored), report->count(Kind::Warning));
    });
}

}   // namespace athenea::cli
