// Copyright (c) 2026 jesus luque.
#include "athenea/io/NeuralScene.h"

#include <algorithm>
#include <fstream>
#include <system_error>

#include <nlohmann/json.hpp>

namespace athenea::io {
namespace {

using Json = nlohmann::json;

constexpr const char* kFormat = "athenea-neural-scene";

template <typename T, size_t N>
Json listOf(const std::array<T, N>& values) {
    Json list = Json::array();
    for (const T& v : values) {
        list.push_back(v);
    }
    return list;
}

template <typename T, size_t N>
void into(const Json& from, const char* key, std::array<T, N>& values) {
    const auto found = from.find(key);
    if (found == from.end() || !found->is_array() || found->size() != N) {
        return;
    }
    for (size_t k = 0; k < N; ++k) {
        values[k] = (*found)[k].get<T>();
    }
}

}   // namespace

NeuralScenePiece* NeuralScene::piece(const std::string& id) {
    const auto found = std::find_if(pieces.begin(), pieces.end(), [&](const NeuralScenePiece& p) { return p.id == id; });
    return found == pieces.end() ? nullptr : &*found;
}

void NeuralScene::bounds(std::array<float, 3>& min, std::array<float, 3>& max) const {
    min = {0.0F, 0.0F, 0.0F};
    max = {0.0F, 0.0F, 0.0F};
    bool any = false;
    for (const NeuralScenePiece& p : pieces) {
        if (p.gaussians == 0) {
            continue;
        }
        for (size_t k = 0; k < 3; ++k) {
            min[k] = any ? std::min(min[k], p.boundsMin[k]) : p.boundsMin[k];
            max[k] = any ? std::max(max[k], p.boundsMax[k]) : p.boundsMax[k];
        }
        any = true;
    }
}

Result<std::vector<uint32_t>> readSeen(const std::filesystem::path& path, uint32_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Error(ErrorCode::NotFound, "cannot open " + path.string());
    }
    std::vector<uint32_t> seen(count);
    in.read(reinterpret_cast<char*>(seen.data()), static_cast<std::streamsize>(seen.size() * sizeof(uint32_t)));
    if (!in || in.peek() != std::ifstream::traits_type::eof()) {
        return Error(ErrorCode::InvalidArgument,
                     path.string() + " is not a record of " + std::to_string(count) + " gaussians");
    }
    return seen;
}

Result<void> writeSeen(const std::filesystem::path& path, const std::vector<uint32_t>& seen) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(seen.data()), static_cast<std::streamsize>(seen.size() * sizeof(uint32_t)));
    if (!out) {
        return Error(ErrorCode::IoFailure, "cannot write " + path.string());
    }
    return ok();
}

Result<NeuralScene> readNeuralScene(const std::filesystem::path& directory) {
    const std::filesystem::path path = directory / "scene.json";
    std::ifstream in(path);
    if (!in) {
        return Error(ErrorCode::NotFound, "cannot open " + path.string());
    }
    const Json root = Json::parse(in, nullptr, false);
    if (root.is_discarded() || !root.is_object() || root.value("format", std::string{}) != kFormat) {
        return Error(ErrorCode::InvalidArgument, path.string() + " is not a neural scene's manifest");
    }
    const uint32_t version = root.value("version", 0u);
    if (version != NeuralScene::kVersion) {
        return Error(ErrorCode::Unsupported, path.string() + " is version " + std::to_string(version) +
                                                 " of a neural scene; this reads " +
                                                 std::to_string(NeuralScene::kVersion));
    }
    NeuralScene scene;
    scene.name = root.value("name", std::string{});
    scene.upAxis = root.value("upAxis", std::string{"Y"});
    scene.metersPerUnit = root.value("metersPerUnit", 1.0);
    if (const auto source = root.find("source"); source != root.end() && source->is_object()) {
        scene.source = source->value("stage", std::string{});
        scene.licence = source->value("licence", std::string{});
    }
    if (const auto pieces = root.find("pieces"); pieces != root.end() && pieces->is_array()) {
        for (const Json& from : *pieces) {
            NeuralScenePiece p;
            p.id = from.value("id", std::string{});
            p.prim = from.value("prim", std::string{});
            p.cloud = from.value("cloud", std::string{});
            p.gaussians = from.value("gaussians", 0u);
            if (const auto box = from.find("bounds"); box != from.end() && box->is_object()) {
                into(*box, "min", p.boundsMin);
                into(*box, "max", p.boundsMax);
            }
            into(from, "transform", p.transform);
            p.revision = from.value("revision", 1u);
            p.isStatic = from.value("static", true);
            p.seen = from.value("seen", std::string{});
            p.discovered = from.value("discovered", 0u);
            p.movedAt = from.value("movedAt", p.revision);
            p.paintedAt = from.value("paintedAt", p.revision);
            if (const auto origin = from.find("origin"); origin != from.end() && origin->is_object()) {
                p.origin = origin->value("kind", std::string{"converted"});
                p.foundBy = origin->value("camera", std::string{});
                p.foundAt = origin->value("time", 0.0);
            }
            if (const auto look = from.find("appearance"); look != from.end() && look->is_object()) {
                p.appearance = look->value("file", std::string{});
                p.lit = look->value("lit", false);
                p.split = look->value("split", false);
                if (const auto fit = look->find("fit"); fit != look->end() && fit->is_object()) {
                    NeuralSceneFit f;
                    f.steps = fit->value("steps", 0u);
                    f.against = fit->value("against", std::string{});
                    f.relMse = fit->value("relMSE", 0.0);
                    f.p99 = fit->value("p99", 0.0);
                    f.max = fit->value("max", 0u);
                    p.fit = f;
                }
            }
            if (p.id.empty() || p.cloud.empty()) {
                return Error(ErrorCode::InvalidArgument, path.string() + ": a piece wants an id and a cloud");
            }
            scene.pieces.push_back(std::move(p));
        }
    }
    if (const auto lights = root.find("lights"); lights != root.end() && lights->is_array()) {
        for (const Json& from : *lights) {
            NeuralSceneLight l;
            l.name = from.value("name", std::string{});
            l.kind = from.value("kind", std::string{"distant"});
            into(from, "towards", l.towards);
            into(from, "radiance", l.radiance);
            scene.lights.push_back(std::move(l));
        }
    }
    if (const auto cameras = root.find("cameras"); cameras != root.end() && cameras->is_array()) {
        for (const Json& from : *cameras) {
            NeuralSceneCamera c;
            c.name = from.value("name", std::string{});
            into(from, "eye", c.eye);
            into(from, "target", c.target);
            into(from, "up", c.up);
            c.focal = from.value("focal", 35.0);
            scene.cameras.push_back(std::move(c));
        }
    }
    if (const auto snapshots = root.find("snapshots"); snapshots != root.end() && snapshots->is_array()) {
        for (const Json& from : *snapshots) {
            NeuralSceneSnapshot s;
            s.time = from.value("time", 0.0);
            s.camera = from.value("camera", std::string{});
            if (const auto pieces = from.find("pieces"); pieces != from.end() && pieces->is_array()) {
                for (const Json& id : *pieces) {
                    s.pieces.push_back(id.get<std::string>());
                }
            }
            scene.snapshots.push_back(std::move(s));
        }
    }
    return scene;
}

Result<void> writeNeuralScene(const std::filesystem::path& directory, const NeuralScene& scene) {
    std::error_code failed;
    std::filesystem::create_directories(directory / "pieces", failed);
    if (failed) {
        return Error(ErrorCode::IoFailure, "cannot make " + (directory / "pieces").string());
    }
    Json root;
    root["format"] = kFormat;
    root["version"] = NeuralScene::kVersion;
    root["name"] = scene.name;
    root["upAxis"] = scene.upAxis;
    root["metersPerUnit"] = scene.metersPerUnit;
    root["source"] = {{"stage", scene.source}, {"licence", scene.licence}};
    std::array<float, 3> min{};
    std::array<float, 3> max{};
    scene.bounds(min, max);
    root["bounds"] = {{"min", listOf(min)}, {"max", listOf(max)}};
    root["pieces"] = Json::array();
    for (const NeuralScenePiece& p : scene.pieces) {
        Json piece;
        piece["id"] = p.id;
        piece["prim"] = p.prim;
        piece["cloud"] = p.cloud;
        piece["gaussians"] = p.gaussians;
        piece["bounds"] = {{"min", listOf(p.boundsMin)}, {"max", listOf(p.boundsMax)}};
        piece["transform"] = listOf(p.transform);
        piece["revision"] = p.revision;
        piece["static"] = p.isStatic;
        if (!p.seen.empty()) {
            piece["seen"] = p.seen;
            piece["discovered"] = p.discovered;
        }
        piece["movedAt"] = p.movedAt;
        piece["paintedAt"] = p.paintedAt;
        piece["origin"] = {{"kind", p.origin}, {"camera", p.foundBy}, {"time", p.foundAt}};
        if (!p.appearance.empty()) {
            Json look = {{"file", p.appearance}, {"lit", p.lit}, {"split", p.split}};
            if (p.fit) {
                look["fit"] = {{"steps", p.fit->steps},
                               {"against", p.fit->against},
                               {"relMSE", p.fit->relMse},
                               {"p99", p.fit->p99},
                               {"max", p.fit->max}};
            }
            piece["appearance"] = std::move(look);
        }
        root["pieces"].push_back(std::move(piece));
    }
    root["lights"] = Json::array();
    for (const NeuralSceneLight& l : scene.lights) {
        root["lights"].push_back(
            {{"name", l.name}, {"kind", l.kind}, {"towards", listOf(l.towards)}, {"radiance", listOf(l.radiance)}});
    }
    root["cameras"] = Json::array();
    for (const NeuralSceneCamera& c : scene.cameras) {
        root["cameras"].push_back({{"name", c.name},
                                   {"eye", listOf(c.eye)},
                                   {"target", listOf(c.target)},
                                   {"up", listOf(c.up)},
                                   {"focal", c.focal}});
    }
    root["snapshots"] = Json::array();
    for (const NeuralSceneSnapshot& s : scene.snapshots) {
        root["snapshots"].push_back({{"time", s.time}, {"camera", s.camera}, {"pieces", s.pieces}});
    }
    const std::filesystem::path path = directory / "scene.json";
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        return Error(ErrorCode::IoFailure, "cannot write " + path.string());
    }
    out << root.dump(2) << "\n";
    if (!out) {
        return Error(ErrorCode::IoFailure, "writing " + path.string() + " failed");
    }
    return ok();
}

}   // namespace athenea::io
