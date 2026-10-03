// Copyright (c) 2026 jesus luque.
//
// Where the engine finds its own files: the image this code is linked into,
// which for a program is its executable and for a plugin is the plugin.
#include "athenea/core/Platform.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

using namespace athenea;

TEST_CASE("a program's module is its executable", "[core][platform]") {
    // Core is a static library: linked into this test binary, the image
    // holding moduleDir is the executable itself.
    const std::filesystem::path module = platform::moduleDir();
    REQUIRE_FALSE(module.empty());
    CHECK(std::filesystem::is_directory(module));
    CHECK(std::filesystem::equivalent(module, platform::executableDir()));
}
