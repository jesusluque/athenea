// Copyright (c) 2026 jesus luque.
//
// The engine's shaders found from where its code was loaded: a program in
// bin/, the Hydra plugin three directories down a build tree, a package
// another program unpacks with the shaders beside the library. No device.
#include "athenea/gpu/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <random>
#include <string>

using namespace athenea;

namespace {

/// A scratch tree under the temporary directory, removed when it goes.
struct ScratchTree {
    std::filesystem::path root;
    ScratchTree() {
        std::random_device seed;
        root = std::filesystem::temp_directory_path() / ("athenea-paths-" + std::to_string(seed()));
        std::filesystem::create_directories(root);
    }
    ~ScratchTree() {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
    }
    std::filesystem::path make(const std::filesystem::path& relative) const {
        std::filesystem::create_directories(root / relative);
        return std::filesystem::canonical(root / relative);
    }
};

}   // namespace

TEST_CASE("shaders are found from the module, the executable's way and the plugin's", "[gpu][paths]") {
    ScratchTree tree;
    const auto shaders = tree.make("build/shaders");
    tree.make("build/shaders/athenea");
    const auto bin = tree.make("build/bin");
    const auto plugin = tree.make("build/plugin/usd/hdAthenea");

    const std::filesystem::path fromBin[] = {bin};
    CHECK(gpu::findShaderDirectory(fromBin) == shaders);
    const std::filesystem::path fromPlugin[] = {plugin};
    CHECK(gpu::findShaderDirectory(fromPlugin) == shaders);
}

TEST_CASE("the module is asked before the executable", "[gpu][paths]") {
    ScratchTree tree;
    const auto packaged = tree.make("addon/shaders");
    tree.make("addon/shaders/athenea");
    const auto library = tree.make("addon");
    const auto hostShaders = tree.make("host/shaders");
    tree.make("host/shaders/athenea");
    const auto host = tree.make("host/MacOS");

    const std::filesystem::path starts[] = {library, host};
    CHECK(gpu::findShaderDirectory(starts) == packaged);
    const std::filesystem::path hostOnly[] = {std::filesystem::path{}, host};
    CHECK(gpu::findShaderDirectory(hostOnly) == hostShaders);
}

TEST_CASE("a shaders directory without the engine's tree is not the engine's", "[gpu][paths]") {
    ScratchTree tree;
    tree.make("app/shaders/somethingElse");
    const auto deep = tree.make("app/a/b/c/d");
    const std::filesystem::path starts[] = {tree.make("app/MacOS"), deep};
    CHECK(gpu::findShaderDirectory(starts).empty());
}
