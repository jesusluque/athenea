// Copyright (c) 2026 jesus luque.
//
// The tools, as a table: one entry a tool, its schema beside it, so the
// listing and the dispatch cannot disagree about what exists.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Session.h"

namespace athenea::mcp {

struct ToolEntry {
    std::string    name;
    std::string    description;
    nlohmann::json schema;
    std::function<nlohmann::json(Session&, const nlohmann::json&)> handler;
};

[[nodiscard]] const std::vector<ToolEntry>& toolTable();

/// A tool's answer: MCP wants content blocks, and every one of these is text.
[[nodiscard]] nlohmann::json textAnswer(std::string text, bool isError = false);

}   // namespace athenea::mcp
