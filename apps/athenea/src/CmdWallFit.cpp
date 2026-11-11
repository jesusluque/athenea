// Copyright (c) 2026 jesus luque.
//
// athenea wall-fit: a cloud taught what a wall of light does to it.
//
// The pictures are the path tracer's: the same mesh baked into the same
// gaussians with one texel of the wall lit at a time (athenea mesh2splat, a
// RectLight where the texel is). What a wall showing any picture does is the
// sum of those, times the picture's texels -- light adds -- so the fit is
// shown single texels and then held against whole pictures it never saw,
// from cameras it never stood at, the frame they should be summed from the
// baked clouds.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/io/Exr.h"
#include "athenea/render/NeuralAppearance.h"
#include "athenea/render/NeuralWall.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"

namespace athenea::cli {
namespace {

struct WallFitOptions {
    std::string         patches;
    unsigned            steps = 4000;
    std::string         size = "960x540";
    std::vector<double> target;
    double              distance = 0.0;
    double              focal = 35.0;
    unsigned            heldOut = 4;
    unsigned            lobes = 1;
    float               startLevel = -1.0F;
    float               startGain = 0.05F;
    float               rateUv = 2.0e-3F;
    float               rateLevel = 1.0e-2F;
    float               rateGain = 1.0e-2F;
    float               rateDecay = 0.1F;
    bool                frozen = false;
    unsigned            wholeEvery = 1;   ///< one step in this many is a whole picture, a colour a texel; 0: none is
    std::string         output;
};

struct Shown {
    const char* name;
    uint32_t    pattern;
    uint32_t    argument;
};

int run(const WallFitOptions& o) {
    render::NeuralWallSettings how;
    if (std::sscanf(o.size.c_str(), "%ux%u", &how.width, &how.height) != 2 || how.width == 0 || how.height == 0) {
        err("--size wants WIDTHxHEIGHT\n");
        return kExitFailure;
    }
    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    auto raster = render::TileRasterizer::create(library);
    if (!raster) return report(raster.error());
    auto wall = render::NeuralWall::create(library);
    if (!wall) return report(wall.error());

    render::WallRect rect;
    uint32_t wide = 0;
    uint32_t high = 0;
    std::vector<std::unique_ptr<scene::GpuSplats>> clouds;   // by texel
    {
        std::ifstream in(o.patches);
        if (!in) {
            err("cannot open %s\n", o.patches.c_str());
            return kExitFailure;
        }
        const std::filesystem::path beside = std::filesystem::path(o.patches).parent_path();
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream words(line);
            std::string kind;
            if (!(words >> kind)) {
                continue;
            }
            if (kind == "wall") {
                if (!(words >> wide >> high >> rect.centre[0] >> rect.centre[1] >> rect.centre[2] >> rect.axisU[0] >>
                      rect.axisU[1] >> rect.axisU[2] >> rect.axisV[0] >> rect.axisV[1] >> rect.axisV[2] >>
                      rect.halfU >> rect.halfV) ||
                    wide == 0 || high == 0) {
                    err("a wall line is: wall W H cx cy cz ux uy uz vx vy vz halfU halfV\n");
                    return kExitFailure;
                }
                clouds.resize(size_t{wide} * high);
                continue;
            }
            uint32_t texel = 0;
            std::string file;
            if (kind != "patch" || !(words >> texel >> file) || texel >= clouds.size()) {
                err("%s: a line is `wall ...` first, then `patch <texel> <file>`\n", o.patches.c_str());
                return kExitFailure;
            }
            auto cloud = usd::uploadParticleField(library, *loader, beside / file);
            if (!cloud) return report(cloud.error());
            clouds[texel] = std::make_unique<scene::GpuSplats>(std::move(*cloud));
        }
    }
    const uint32_t texels = wide * high;
    if (texels == 0) {
        err("%s names no wall\n", o.patches.c_str());
        return kExitFailure;
    }
    for (uint32_t k = 0; k < texels; ++k) {
        if (!clouds[k]) {
            err("the wall's texel %u has no cloud\n", k);
            return kExitFailure;
        }
        if (clouds[k]->count != clouds[0]->count) {
            err("texel %u's cloud holds %u gaussians and the first %u: not the same gaussians\n", k, clouds[k]->count,
                clouds[0]->count);
            return kExitFailure;
        }
    }
    const scene::GpuSplats& cloud = *clouds[0];

    // The pictures the wall holds: each texel alone, which is what the fit is
    // shown, and then whole pictures, which is what it is held against.
    const std::vector<Shown> whole{{"ramp", 1, 0},
                                   {"checker", 2, 1},
                                   {"random", 3, 11},
                                   {"window", 4, (high / 2) * wide + wide / 4},
                                   {"white", 5, 0}};
    // And one more, written anew each time the fit is shown a whole picture.
    const uint32_t scratch = texels + static_cast<uint32_t>(whole.size());
    const uint32_t images = scratch + 1;
    // A frozen fit reads the whole wall's mean and moves only its gain: what a wall with no picture would be.
    const float startLevel = o.frozen ? 64.0F : o.startLevel;
    {
        auto started = wall->start(cloud, rect, o.lobes, wide, high, images, startLevel, o.startGain);
        if (!started) return report(started.error());
    }
    for (uint32_t k = 0; k < texels; ++k) {
        auto made = wall->pattern(k, 0, k);
        if (!made) return report(made.error());
    }
    for (size_t k = 0; k < whole.size(); ++k) {
        auto made = wall->pattern(texels + static_cast<uint32_t>(k), whole[k].pattern, whole[k].argument);
        if (!made) return report(made.error());
    }

    // The cloud itself gives no light here: a lit neural cloud under no light is black, and the wall adds its own.
    auto dark = render::makeNeuralCloud(library, cloud, 1, true);
    if (!dark) return report(dark.error());
    const render::NeuralLight none;
    render::SplatInstance student{&cloud, render::Mat4::identity(), {}};
    student.neural = &*dark;
    student.neuralLight = &none;
    student.neuralLightCount = 0;
    student.wall = &*wall;

    const auto d = [](float v) { return static_cast<double>(v); };
    const scene::Bounds& b = cloud.bounds;
    const render::Vec3 target = o.target.size() == 3
                                    ? render::Vec3{o.target[0], o.target[1], o.target[2]}
                                    : render::Vec3{(d(b.min[0]) + d(b.max[0])) * 0.5, (d(b.min[1]) + d(b.max[1])) * 0.5,
                                                   (d(b.min[2]) + d(b.max[2])) * 0.5};
    const double extent = std::max({d(b.max[0]) - d(b.min[0]), d(b.max[1]) - d(b.min[1]), d(b.max[2]) - d(b.min[2])});
    const double distance = o.distance > 0.0 ? o.distance : std::max(extent, 1.0e-3) * 1.4;
    const auto cameraAt = [&](uint32_t k, double phase) {
        const double a = 6.283185307179586 * std::fmod(0.5 + (k + phase) * 0.7548776662466927, 1.0);
        const double h = -0.25 + 1.15 * std::fmod(0.5 + (k + phase) * 0.5698402909980532, 1.0);
        const double r = std::sqrt(std::max(1.0 - h * h, 0.0));
        render::Camera camera = render::Camera::lookingAt(
            target + render::Vec3{r * std::sin(a), h, r * std::cos(a)} * distance, target, {0.0, 1.0, 0.0});
        camera.lens.focal = o.focal;
        return camera;
    };

    render::RenderSettings settings;
    settings.width = how.width;
    settings.height = how.height;
    const uint32_t pixels = how.width * how.height;
    gpu::Buffer summed;
    {
        gpu::BufferDesc desc;
        desc.bytes = uint64_t{pixels} * 16;
        desc.elementBytes = 16;
        desc.label = "wall.summed";
        auto made = gpu::Buffer::create(**device, desc);
        if (!made) return report(made.error());
        summed = *made;
    }
    render::RenderTargets one;
    render::RenderTargets ours;
    // The frame a wall showing picture `image` should give from `camera`: every texel's cloud drawn, times the texel.
    const auto shouldBe = [&](const render::Camera& camera, uint32_t image) -> Result<void> {
        for (uint32_t k = 0; k < texels; ++k) {
            const render::SplatInstance baked{clouds[k].get(), render::Mat4::identity(), {}};
            auto drawn = raster->render(camera, std::span<const render::SplatInstance>(&baked, 1), settings, one);
            if (!drawn) return std::move(drawn).error();
            ATHENEA_TRY(wall->accumulate(one.colour, summed, pixels, image, k, k == 0));
        }
        return ok();
    };
    struct Measure {
        double   relMse = 0.0;
        double   p99 = 0.0;
        uint32_t max = 0;
    };
    const auto compare = [&](Measure& m, double share) -> Result<void> {
        auto hdr = render::compareHdr(library, ours.colour, summed, settings.width, settings.height);
        if (!hdr) return std::move(hdr).error();
        auto codes = render::compareImages(library, ours.colour, summed, settings.width, settings.height);
        if (!codes) return std::move(codes).error();
        m.relMse += hdr->relMse * share;
        m.p99 += static_cast<double>(codes->p99) * share;
        m.max = std::max(m.max, codes->max);
        return ok();
    };
    // Whole pictures from the held-out cameras; `write`: the first camera's frames, under that prefix.
    const auto measure = [&](const std::string& write, std::vector<Measure>* each) -> Result<Measure> {
        Measure m;
        const double share = 1.0 / (static_cast<double>(whole.size()) * o.heldOut);
        for (size_t at = 0; at < whole.size(); ++at) {
            const uint32_t image = texels + static_cast<uint32_t>(at);
            Measure own;
            for (uint32_t k = 0; k < o.heldOut; ++k) {
                const render::Camera camera = cameraAt(k, 0.5);
                ATHENEA_TRY(shouldBe(camera, image));
                wall->show(image);
                auto drawn = raster->render(camera, std::span<const render::SplatInstance>(&student, 1), settings, ours);
                if (!drawn) return std::move(drawn).error();
                ATHENEA_TRY(compare(m, share));
                ATHENEA_TRY(compare(own, 1.0 / o.heldOut));
                if (!write.empty() && k == 0) {
                    auto depth = ours.depth.readAll<float>(**device);
                    if (!depth) return std::move(depth).error();
                    for (const auto& [what, buffer] : {std::pair<const char*, const gpu::Buffer*>{"picture", &summed},
                                                       std::pair<const char*, const gpu::Buffer*>{"wall", &ours.colour}}) {
                        auto colour = buffer->readAll<float>(**device);
                        if (!colour) return std::move(colour).error();
                        const std::string path = write + "_" + whole[at].name + "_" + what + ".exr";
                        ATHENEA_TRY(io::writeExr(path, settings.width, settings.height, *colour, *depth));
                        out("wrote %s\n", path.c_str());
                    }
                }
            }
            if (each != nullptr) {
                each->push_back(own);
            }
        }
        return m;
    };

    out("%u gaussians before a wall of %u x %u texels, %u read%s a gaussian%s\n", cloud.count, wide, high, o.lobes,
        o.lobes == 1 ? "" : "s", o.frozen ? ", frozen on the wall's mean" : "");
    auto before = measure({}, nullptr);
    if (!before) return report(before.error());
    out("before: relMSE %.5f, p99 %.1f, max %u (whole pictures, held-out cameras, against the baked clouds summed)\n",
        before->relMse, before->p99, before->max);
    const auto start = std::chrono::steady_clock::now();
    const unsigned every = std::max(o.steps / 10, 1u);
    render::RenderTargets pictured;
    for (unsigned k = 0; k < o.steps; ++k) {
        // The texels in an order that visits all of them before any twice, where their number allows.
        const uint32_t texel = static_cast<uint32_t>((static_cast<uint64_t>(k) * 7) % texels);
        const render::Camera camera = cameraAt(k, 0.0);
        // A WHOLE PICTURE NOW AND THEN. A texel alone is a dim frame, and what a gaussian is off by under
        // each adds up under a picture that lights them all: so some steps are a picture of a colour a
        // texel, another each time, and the frame it should be the baked clouds summed.
        const bool wholePicture = o.wholeEvery != 0 && k % o.wholeEvery == o.wholeEvery - 1;
        const gpu::Buffer* wanted = &pictured.colour;
        if (wholePicture) {
            // Pictures of every grain: a colour a texel, a colour a block of two or of four, and a few
            // texels lit among dark ones -- so that where on the wall a gaussian reads matters to the fit.
            const uint32_t turn = (k / o.wholeEvery) % 4;
            const uint32_t seed = 1000u + k * 97u;
            auto made = turn == 0   ? wall->pattern(scratch, 3, seed)
                        : turn == 1 ? wall->pattern(scratch, 7, (seed << 2) | 0u)
                        : turn == 2 ? wall->pattern(scratch, 7, (seed << 2) | 1u)
                                    : wall->pattern(scratch, 8, seed);
            if (!made) return report(made.error());
            auto all = shouldBe(camera, scratch);
            if (!all) return report(all.error());
            wall->show(scratch);
            wanted = &summed;
        } else {
            const render::SplatInstance baked{clouds[texel].get(), render::Mat4::identity(), {}};
            auto drawn = raster->render(camera, std::span<const render::SplatInstance>(&baked, 1), settings, pictured);
            if (!drawn) return report(drawn.error());
            wall->show(texel);
        }
        const float falling = std::pow(o.rateDecay, static_cast<float>(k) / static_cast<float>(std::max(o.steps, 1u)));
        how.rateUv = o.frozen ? 0.0F : o.rateUv * falling;
        how.rateLevel = o.frozen ? 0.0F : o.rateLevel * falling;
        how.rateGain = o.rateGain * falling;
        const bool ask = k == 0 || (k + 1) % every == 0;
        auto loss = wall->step(*raster, camera, student, *wanted, how, ask);
        if (!loss) return report(loss.error());
        if (*loss) {
            auto now = measure({}, nullptr);
            if (!now) return report(now.error());
            out("step %u: relMSE %.5f; whole pictures: relMSE %.5f, p99 %.1f, max %u\n", k + 1, **loss, now->relMse,
                now->p99, now->max);
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::vector<Measure> each;
    auto after = measure(o.output, &each);
    if (!after) return report(after.error());
    out("after %u steps in %.1f s (%.1f ms a step): whole pictures relMSE %.5f, p99 %.1f, max %u\n", o.steps, seconds,
        o.steps > 0 ? seconds * 1000.0 / o.steps : 0.0, after->relMse, after->p99, after->max);
    for (size_t k = 0; k < each.size(); ++k) {
        out("  %s: relMSE %.5f, p99 %.1f, max %u\n", whole[k].name, each[k].relMse, each[k].p99, each[k].max);
    }

    // What the wall costs a frame: the cloud drawn with and without it, the clock waiting for the device.
    render::RenderSettings timed = settings;
    timed.timeStages = true;
    render::SplatInstance unlit = student;
    unlit.wall = nullptr;
    const render::Camera camera = cameraAt(0, 0.5);
    double least[2] = {1.0e30, 1.0e30};
    wall->show(texels);
    for (int pass = 0; pass < 40; ++pass) {
        for (int which = 0; which < 2; ++which) {
            const render::SplatInstance& drawnAs = which == 0 ? unlit : student;
            auto stats = raster->render(camera, std::span<const render::SplatInstance>(&drawnAs, 1), timed, ours);
            if (!stats) return report(stats.error());
            least[which] = std::min(least[which], stats->totalMs);
        }
    }
    out("a frame of %ux%u: %.2f ms without the wall, %.2f ms with it\n", settings.width, settings.height, least[0],
        least[1]);
    // A STILL FRAME: the camera and the cloud as they were, the wall showing another picture each frame --
    // what a set with a fixed camera is. Timed, and held against the same picture's frame drawn whole.
    render::RenderSettings stillTimed = timed;
    stillTimed.still = true;
    double leastStill = 1.0e30;
    uint32_t kept = 0;
    render::RenderTargets stillFrame;
    for (int pass = 0; pass < 40; ++pass) {
        wall->show(texels + static_cast<uint32_t>(pass % static_cast<int>(whole.size())));
        auto stats = raster->render(camera, std::span<const render::SplatInstance>(&student, 1), stillTimed, stillFrame);
        if (!stats) return report(stats.error());
        if (stats->still) {
            leastStill = std::min(leastStill, stats->totalMs);
            ++kept;
        }
    }
    auto wholeFrame = raster->render(camera, std::span<const render::SplatInstance>(&student, 1), settings, ours);
    if (!wholeFrame) return report(wholeFrame.error());
    auto same = render::compareImages(library, stillFrame.colour, ours.colour, settings.width, settings.height);
    if (!same) return report(same.error());
    out("a still frame, the wall's picture another each time: %.2f ms (%u of 40 kept the last frame's geometry); "
        "against that picture's frame drawn whole: max %u code values\n",
        leastStill, kept, same->max);
    return 0;
}

}   // namespace

void addWallFit(CLI::App& app) {
    auto options = std::make_shared<WallFitOptions>();
    WallFitOptions& o = *options;
    CLI::App* cmd = app.add_subcommand(
        "wall-fit", "a cloud taught what a wall of light does to it: where each gaussian reads the wall's picture");
    cmd->add_option("--patches", o.patches,
                    "a manifest: the wall, and one mesh baked into the same gaussians under each of its texels alone")
        ->required();
    cmd->add_option("--steps", o.steps, "steps of the fit, a camera and a texel each");
    cmd->add_option("--size", o.size, "WIDTHxHEIGHT of the frames fitted and measured");
    cmd->add_option("--target", o.target, "what every camera looks at: x y z")->expected(3);
    cmd->add_option("--distance", o.distance, "every camera's distance from the target");
    cmd->add_option("--focal", o.focal, "focal length, mm");
    cmd->add_option("--held-out", o.heldOut, "cameras the fit is measured from and never fitted from");
    cmd->add_option("--lobes", o.lobes, "reads of the wall a gaussian keeps");
    cmd->add_option("--start-level", o.startLevel, "how wide a read starts, as a level of the wall's mip chain");
    cmd->add_option("--start-gain", o.startGain, "how strong a gaussian's reads start, together");
    cmd->add_option("--rate-uv", o.rateUv, "Adam's rate for where a read is");
    cmd->add_option("--rate-level", o.rateLevel, "Adam's rate for how wide a read is");
    cmd->add_option("--rate-gain", o.rateGain, "Adam's rate for the log of a read's gain");
    cmd->add_option("--rate-decay", o.rateDecay, "what the rates have fallen to by the last step, as a factor");
    cmd->add_option("--whole-every", o.wholeEvery, "one step in this many is fitted to a whole picture, a colour a texel, rather than to one texel (0: none)");
    cmd->add_flag("--frozen", o.frozen, "every read is the whole wall's mean and only its gain is learnt: the wall with no picture");
    cmd->add_option("-o,--output", o.output, "a prefix: <prefix>_<picture>_picture.exr and _wall.exr, the first held-out camera's");
    cmd->callback([options] {
        if (const int code = run(*options); code != 0) {
            throw CLI::RuntimeError(code);
        }
    });
}

}   // namespace athenea::cli
