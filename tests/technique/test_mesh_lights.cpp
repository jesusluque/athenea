// Copyright (c) 2026 jesus luque.
//
// Mesh lights (technique::MeshLights) on their own: a planar cluster's
// samples against the density its triangles' shares describe, the tree's
// choices against the probabilities it reports, the clusters' power against
// EmissiveTable's, and a moved emitter refitted without its clusters
// changing. Everything is measured by kernels; the host reads counters.
#include "../gpu/GpuTest.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "athenea/geom/Mesh.h"
#include "athenea/light/LightTable.h"
#include "athenea/material/MaterialCompiler.h"
#include "athenea/material/TextureStore.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/technique/EmissiveTable.h"
#include "athenea/technique/MaterialPrograms.h"
#include "athenea/technique/MeshLights.h"
#include "athenea/world/GpuScene.h"

using namespace athenea;

namespace {

/// Two quads side by side, of different widths, in the z = 0 plane facing +z
/// (four triangles of three areas), shifted along x by `shiftX`.
std::shared_ptr<const geom::GpuMesh> panel(geom::MeshBuilder& builder, float shiftX, uint64_t topology) {
    static std::vector<float> points;
    points = {0.0F + shiftX, 0.0F, 0.0F, 1.0F + shiftX, 0.0F, 0.0F, 3.0F + shiftX, 0.0F, 0.0F,
              0.0F + shiftX, 1.0F, 0.0F, 1.5F + shiftX, 1.0F, 0.0F, 3.0F + shiftX, 1.0F, 0.0F};
    static const std::vector<int32_t> counts{4, 4};
    static const std::vector<int32_t> indices{0, 1, 4, 3, 1, 2, 5, 4};
    geom::MeshInput in;
    in.source = "panel";
    in.topology = topology;
    in.points = {std::as_bytes(std::span<const float>(points)), false};
    in.faceVertexCounts = counts;
    in.faceVertexIndices = indices;
    in.smoothNormals = false;
    auto mesh = builder.build(in);
    if (!mesh) FAIL(mesh.error().toString());
    return std::make_shared<const geom::GpuMesh>(std::move(*mesh));
}

struct Rig {
    std::unique_ptr<material::MaterialCompiler> compiler;
    std::unique_ptr<material::TextureStore>     textures;
    std::optional<geom::MeshBuilder>            builder;
    std::optional<world::GpuScene>              scene;
    std::optional<technique::MaterialPrograms>  programs;
    std::optional<light::LightTable>            table;
    std::optional<technique::MeshLights>        lights;
    gpu::Buffer                                 records;
    gpu::Buffer                                 blob;
    technique::MaterialFrame                    frame;
    render::Projection                          projection;
};

void makeRig(test::Gpu& gpu, Rig& rig) {
    std::vector<std::filesystem::path> shaderPaths;
    for (const std::string& path : gpu.device->shaderSearchPaths()) {
        shaderPaths.emplace_back(path);
    }
    auto compiler = material::MaterialCompiler::create({ATHENEA_MATERIALX_ROOT}, shaderPaths);
    if (!compiler) FAIL(compiler.error().toString());
    rig.compiler = std::move(*compiler);
    auto textures = material::TextureStore::create(*gpu.library);
    if (!textures) FAIL(textures.error().toString());
    rig.textures = std::move(*textures);
    auto builder = geom::MeshBuilder::create(*gpu.library);
    auto scene = world::GpuScene::create(*gpu.library);
    auto programs = technique::MaterialPrograms::create(*gpu.library);
    auto table = light::LightTable::create(*gpu.library);
    auto lights = technique::MeshLights::create(*gpu.library);
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!table) FAIL(table.error().toString());
    if (!lights) FAIL(lights.error().toString());
    rig.builder.emplace(std::move(*builder));
    rig.scene.emplace(std::move(*scene));
    rig.programs.emplace(std::move(*programs));
    rig.table.emplace(std::move(*table));
    rig.lights.emplace(std::move(*lights));
    auto glowing = rig.compiler->compileXml(
        "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
        "  <uniform_edf name=\"edf\" type=\"EDF\">\n"
        "    <input name=\"color\" type=\"color3\" value=\"1.0, 0.5, 0.25\" />\n"
        "  </uniform_edf>\n"
        "  <surface name=\"shader\" type=\"surfaceshader\">\n"
        "    <input name=\"edf\" type=\"EDF\" nodename=\"edf\" />\n"
        "  </surface>\n"
        "  <surfacematerial name=\"material\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
        "  </surfacematerial>\n</materialx>\n");
    if (!glowing) FAIL(glowing.error().toString());
    REQUIRE(rig.programs->setModules(std::span<const material::CompiledMaterial>(&*glowing, 1)));
    const std::vector<float> blob = material::MaterialCompiler::parameters(
        *glowing, *rig.textures, [](const std::string&) { return 0xFFFFFFFFu; });
    REQUIRE(rig.textures->commit());
    const std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}, {1, 0, technique::kMaterialEmits, 0}};
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*gpu.device, rows, "materials.records");
    auto blobBuffer = gpu::Buffer::fromSpan<float>(*gpu.device, blob, "materials.blob");
    REQUIRE(records);
    REQUIRE(blobBuffer);
    rig.records = std::move(*records);
    rig.blob = std::move(*blobBuffer);
    render::Camera camera = render::Camera::lookingAt({1.5, 0.5, 4.0}, {1.5, 0.5, 0.0});
    rig.projection = render::projectionFor(camera, 64, 48);
    rig.frame.programs = &*rig.programs;
    rig.frame.scene = &*rig.scene;
    rig.frame.records = &rig.records;
    rig.frame.blob = &rig.blob;
    rig.frame.textures = rig.textures.get();
    rig.frame.lights = &*rig.table;
}

/// The frame's mesh lights, as Engine::render makes them: planned, the
/// region reserved, the table set, then written.
void buildLights(test::Gpu& gpu, Rig& rig, std::span<const world::MeshInstance> instances) {
    REQUIRE(rig.scene->update(instances, rig.projection));
    technique::MeshLightInputs inputs;
    inputs.sets.emplace_back();
    for (auto& w : inputs.weights) {
        w = {1.0F, 1.0F, 1.0F};
    }
    if (auto planned = rig.lights->plan(rig.frame, rig.projection, inputs); !planned) {
        FAIL(planned.error().toString());
    }
    rig.table->reserveMesh(rig.lights->wordsNeeded());
    REQUIRE(rig.table->set(std::span<const light::Light>()));
    gpu::CommandBatch batch(*gpu.device);
    if (auto written = rig.lights->write(batch, rig.frame, *rig.table, inputs); !written) {
        FAIL(written.error().toString());
    }
    REQUIRE(batch.submit(true));
}

uint32_t regionWord(test::Gpu& gpu, const Rig& rig, uint32_t at) {
    uint32_t value = 0;
    REQUIRE(rig.table->words().read(*gpu.device, uint64_t{at} * 4, 4, &value));
    return value;
}

}   // namespace

TEST_CASE("a mesh light cluster's samples follow its triangles' shares, and the tree says what it chose",
          "[technique][lights][meshlights][chi2]") {
    ATHENEA_REQUIRE_GPU(gpu);
    Rig rig;
    makeRig(*gpu, rig);
    // The panel, and a second far along x: two clusters, so the tree has a
    // root to choose at.
    std::vector<world::MeshInstance> instances(2);
    instances[0].mesh = panel(*rig.builder, 0.0F, 1);
    instances[1].mesh = panel(*rig.builder, 100.0F, 2);
    for (world::MeshInstance& m : instances) {
        m.material = 1;
        m.lightFlags = technique::kInstanceEmits;
    }
    buildLights(*gpu, rig, instances);
    std::printf("  %u emitting triangles in %u clusters\n", rig.lights->triangles(), rig.lights->clusters());
    REQUIRE(rig.lights->triangles() == 8);
    REQUIRE(rig.lights->clusters() == 2);

    const auto kernel = [&](const char* entry) {
        auto made = gpu::ComputeKernel::create(*gpu->library, "athenea/test/mesh_light_check", entry);
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    };
    gpu::ComputeKernel draw = kernel("meshDraw");
    gpu::ComputeKernel tally = kernel("meshTally");
    gpu::ComputeKernel expect = kernel("meshExpect");
    gpu::ComputeKernel statistic = kernel("meshStatistic");
    const uint32_t bins = 48;
    const uint32_t samples = 1u << 20;
    gpu::Buffer binOf = test::uintBuffer(*gpu->device, samples, "mesh.binOf");
    gpu::Buffer flags = test::uintBuffer(*gpu->device, samples, "mesh.flags");
    auto observed = gpu::Buffer::fromSpan<float>(*gpu->device, std::vector<float>(bins * bins + 1, 0.0F), "mesh.observed");
    auto expected = gpu::Buffer::fromSpan<float>(*gpu->device, std::vector<float>(bins * bins + 1, 0.0F), "mesh.expected");
    auto result = gpu::Buffer::fromSpan<float>(*gpu->device, std::vector<float>(8, 0.0F), "mesh.result");
    REQUIRE(observed);
    REQUIRE(expected);
    REQUIRE(result);
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["region"].setBinding(rig.table->words().rhi());
        cursor["binOf"].setBinding(binOf.rhi());
        cursor["flags"].setBinding(flags.rhi());
        cursor["observed"].setBinding(observed->rhi());
        cursor["expected"].setBinding(expected->rhi());
        cursor["result"].setBinding(result->rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["bins"].setData(bins);
        p["samples"].setData(samples);
        p["cluster"].setData(uint32_t{0});
        p["words"].setData(rig.table->meshWords());
        const float point[4] = {1.2F, 0.4F, 1.5F, 0.0F};
        const float normal[4] = {0.0F, 0.0F, -1.0F, 0.0F};
        p["point"].setData(point, sizeof(point));
        p["normal"].setData(normal, sizeof(normal));
    };
    {
        gpu::CommandBatch batch(*gpu->device);
        draw.dispatch(batch, {samples, 1, 1}, bind);
        tally.dispatch(batch, {1, 1, 1}, bind);
        expect.dispatch(batch, {bins * bins, 1, 1}, bind);
        statistic.dispatch(batch, {1, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    }
    float r[8] = {};
    REQUIRE(result->read(*gpu->device, 0, sizeof(r), r));
    const float z = (r[0] - r[1]) / std::sqrt(2.0F * r[1]);
    std::printf("  planar cluster: chi2 %.1f on %.0f dof (z %.2f), %.4f of the samples drawn, the density integrates "
                "to %.4f; %.0f samples whose density meshClusterPdf disputes, %.0f choices the tree disputes\n",
                double(r[0]), double(r[1]), double(z), double(r[2]), double(r[5]), double(r[3]), double(r[4]));
    CHECK(std::abs(z) < 4.0F);
    CHECK(std::abs(r[5] - 1.0F) < 0.01F);
    CHECK(std::abs(r[2] - 1.0F) < 0.001F);
    CHECK(r[3] == 0.0F);
    CHECK(r[4] == 0.0F);
}

TEST_CASE("mesh lights carry the power EmissiveTable weighs, and a moved emitter keeps its clusters",
          "[technique][lights][meshlights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    Rig rig;
    makeRig(*gpu, rig);
    std::vector<world::MeshInstance> instances(3);
    instances[0].mesh = panel(*rig.builder, 0.0F, 1);
    instances[1].mesh = panel(*rig.builder, 7.0F, 2);
    instances[2].mesh = panel(*rig.builder, -4.0F, 3);
    instances[2].objectToWorld = aofx::xform::translation({0.0, 3.0, -2.0});
    for (world::MeshInstance& m : instances) {
        m.material = 1;
        m.lightFlags = technique::kInstanceEmits;
    }
    buildLights(*gpu, rig, instances);
    const uint32_t triangles = rig.lights->triangles();
    REQUIRE(triangles == 12);
    float meshTotal = 0.0F;
    REQUIRE(rig.table->words().read(*gpu->device, 3 * 4, 4, &meshTotal));
    auto emissive = technique::EmissiveTable::create(*gpu->library);
    REQUIRE(emissive);
    REQUIRE(emissive->build(rig.frame, 2));
    std::printf("  %u clusters; their power %.6f, EmissiveTable's %.6f\n", rig.lights->clusters(), double(meshTotal),
                double(emissive->totalPower()));
    CHECK(std::abs(meshTotal - emissive->totalPower()) <= 1.0e-4F * emissive->totalPower());

    // Which cluster each triangle is in, as the rebuild left it.
    const uint32_t slotBase = regionWord(*gpu, rig, 10);
    const auto membership = [&]() {
        gpu::BufferDesc desc;
        desc.bytes = uint64_t{triangles} * 2 * 4;
        desc.elementBytes = 4;
        auto copy = gpu::Buffer::create(*gpu->device, desc);
        REQUIRE(copy);
        gpu::CommandBatch batch(*gpu->device);
        batch.encoder()->copyBuffer(copy->rhi(), 0, rig.table->words().rhi(), uint64_t{slotBase} * 4, desc.bytes);
        batch.markDirty();
        REQUIRE(batch.submit(true));
        return std::move(*copy);
    };
    const gpu::Buffer before = membership();
    const uint64_t rebuilds = rig.lights->rebuilds();
    // Moved and stretched: a refit, not a rebuild, and the same clusters.
    for (int frame = 1; frame <= 3; ++frame) {
        instances[0].objectToWorld =
            aofx::xform::translation({0.1 * frame, 0.0, 0.0}) * aofx::xform::scaling({1.0 + 0.2 * frame, 1.0, 1.0});
        buildLights(*gpu, rig, instances);
    }
    const gpu::Buffer after = membership();
    auto differ = render::countDifferent(*gpu->library, before, after, triangles * 2);
    REQUIRE(differ);
    float movedTotal = 0.0F;
    REQUIRE(rig.table->words().read(*gpu->device, 3 * 4, 4, &movedTotal));
    std::printf("  three frames of a moving, stretching emitter: %llu rebuilds, %llu refits, %llu membership words "
                "differ; power %.6f, from %.6f\n",
                static_cast<unsigned long long>(rig.lights->rebuilds() - rebuilds),
                static_cast<unsigned long long>(rig.lights->refits()), static_cast<unsigned long long>(*differ),
                double(movedTotal), double(meshTotal));
    CHECK(rig.lights->rebuilds() == rebuilds);
    CHECK(rig.lights->refits() == 3);
    CHECK(*differ == 0);
    // Stretched 1.6 along x, the panel's power grows by 0.6 of a third.
    CHECK(std::abs(movedTotal - meshTotal * (1.0F + 0.6F / 3.0F)) <= 1.0e-3F * meshTotal);
}
