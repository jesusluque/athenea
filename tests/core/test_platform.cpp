// Copyright (c) 2026 jesus luque.
//
// Where the engine finds its own files: the image this code is linked into,
// which for a program is its executable and for a plugin is the plugin. And a
// file written whole or not at all.
#include "athenea/core/Platform.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using namespace athenea;

TEST_CASE("a program's module is its executable", "[core][platform]") {
    // Core is a static library: linked into this test binary, the image
    // holding moduleDir is the executable itself.
    const std::filesystem::path module = platform::moduleDir();
    REQUIRE_FALSE(module.empty());
    CHECK(std::filesystem::is_directory(module));
    CHECK(std::filesystem::equivalent(module, platform::executableDir()));
}

// A FILE WRITTEN WHOLE OR NOT AT ALL. A writer that fails half way leaves
// nothing under the name it was given -- not its half, and not the partial
// file it wrote it in -- and a file already there stays as it was; one that
// succeeds replaces it, and the partial file is gone.
TEST_CASE("a file written atomically is whole or absent", "[core][platform]") {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "athenea_atomic_test";
    std::filesystem::remove_all(dir);
    REQUIRE(std::filesystem::create_directories(dir));
    const std::filesystem::path target = dir / "cloud.usda";
    const auto contents = [](const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    };
    const auto put = [](const std::filesystem::path& path, const std::string& text) {
        std::ofstream out(path, std::ios::binary);
        out << text;
    };
    const auto leftovers = [&dir, &target]() {
        size_t n = 0;
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            n += entry.path() != target ? 1 : 0;
        }
        return n;
    };

    // The partial name keeps the extension a writer picks its format by.
    CHECK(platform::partialPathFor(target).extension() == ".usda");
    CHECK(platform::partialPathFor(target).parent_path() == dir);

    // Failing half way, with nothing there before: nothing there after.
    auto failed = platform::writeAtomically(target, [&](const std::filesystem::path& partial) -> Result<void> {
        put(partial, "half a clo");
        return Error(ErrorCode::IoFailure, "the device ran out");
    });
    CHECK_FALSE(failed);
    CHECK_FALSE(std::filesystem::exists(target));
    CHECK(leftovers() == 0);

    // Succeeding: the file is there, whole, and nothing beside it.
    REQUIRE(platform::writeAtomically(target, [&](const std::filesystem::path& partial) -> Result<void> {
        put(partial, "a whole cloud");
        return ok();
    }));
    CHECK(contents(target) == "a whole cloud");
    CHECK(leftovers() == 0);

    // Failing over a file already there: the old one stays as it was.
    auto again = platform::writeAtomically(target, [&](const std::filesystem::path& partial) -> Result<void> {
        put(partial, "half of another");
        return Error(ErrorCode::IoFailure, "killed");
    });
    CHECK_FALSE(again);
    CHECK(contents(target) == "a whole cloud");
    CHECK(leftovers() == 0);

    std::filesystem::remove_all(dir);
}
