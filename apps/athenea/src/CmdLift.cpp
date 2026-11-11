// Copyright (c) 2026 jesus luque.
//
// athenea lift: a mask made on some cameras' frames of a cloud, carried to its
// gaussians and drawn as the matte another camera sees (render/SplatLift.h).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/io/Exr.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/SplatLift.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::cli {
namespace {

struct LiftOptions {
    std::vector<std::string> splats;
    std::vector<double>      from;     ///< the lifting cameras' eyes, three numbers each
    std::vector<std::string> masks;    ///< a painted mask a lifting camera, or none: --rect
    std::vector<double>      rect;     ///< left bottom right top, fractions of the frame
    std::vector<double>      eye;      ///< the camera the matte is drawn for
    std::vector<double>      target;
    std::vector<double>      up{0.0, 1.0, 0.0};
    double                   focal = 35.0;
    std::string              size = "1920x1080";
    unsigned                 degree = 3;
    bool                     hard = false;
    float                    threshold = 0.5F;
    bool                     byTrace = false;
    std::string              output = "lift.exr";
    unsigned                 turn = 0;   ///< frames of the --eye camera going once round the target, each keyed
};

int run(const LiftOptions& o) {
    uint32_t width = 0;
    uint32_t height = 0;
    if (std::sscanf(o.size.c_str(), "%ux%u", &width, &height) != 2 || width == 0 || height == 0) {
        err("--size wants WIDTHxHEIGHT\n");
        return kExitFailure;
    }
    if (o.from.empty() || o.from.size() % 3 != 0) {
        err("--from wants the lifting cameras' positions, three numbers each\n");
        return kExitFailure;
    }
    const size_t cameras = o.from.size() / 3;
    if (o.masks.empty() == (o.rect.size() != 4)) {
        err("give the masks either as --rect LEFT BOTTOM RIGHT TOP or as --mask, one picture a --from\n");
        return kExitFailure;
    }
    if (!o.masks.empty() && o.masks.size() != cameras) {
        err("--mask wants one picture for each of the %zu --from cameras\n", cameras);
        return kExitFailure;
    }
    if (o.eye.size() != 3) {
        err("--eye wants the position of the camera the matte is drawn for\n");
        return kExitFailure;
    }
    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    auto raster = render::TileRasterizer::create(library);
    if (!raster) return report(raster.error());

    std::vector<std::unique_ptr<scene::GpuSplats>> clouds;
    std::vector<render::SplatInstance> instances;
    scene::Bounds all;
    uint32_t slots = 0;
    for (const std::string& path : o.splats) {
        auto splats = scene::loadSplatFile(*loader, path, o.degree);
        if (!splats) return report(splats.error());
        for (size_t k = 0; k < 3; ++k) {
            all.min[k] = clouds.empty() ? splats->bounds.min[k] : std::min(all.min[k], splats->bounds.min[k]);
            all.max[k] = clouds.empty() ? splats->bounds.max[k] : std::max(all.max[k], splats->bounds.max[k]);
        }
        slots += splats->count;
        clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*splats)));
    }
    if (clouds.empty()) {
        err("nothing to lift a mask to: give --splats\n");
        return kExitFailure;
    }
    for (const auto& cloud : clouds) {
        instances.push_back({cloud.get(), render::Mat4::identity(), {}});
    }
    const auto d = [](float v) { return static_cast<double>(v); };
    const render::Vec3 target = o.target.size() == 3
                                    ? render::Vec3{o.target[0], o.target[1], o.target[2]}
                                    : render::Vec3{(d(all.min[0]) + d(all.max[0])) * 0.5,
                                                   (d(all.min[1]) + d(all.max[1])) * 0.5,
                                                   (d(all.min[2]) + d(all.max[2])) * 0.5};
    const render::Vec3 up = o.up.size() == 3 ? render::Vec3{o.up[0], o.up[1], o.up[2]} : render::Vec3{0, 1, 0};
    const auto cameraAt = [&](const double* at) {
        render::Camera camera = render::Camera::lookingAt({at[0], at[1], at[2]}, target, up);
        camera.lens.focal = o.focal;
        return camera;
    };

    render::RenderSettings settings;
    settings.width = width;
    settings.height = height;
    const uint64_t pixels = uint64_t{width} * height;

    const auto kernel = [&](const char* entry) {
        return gpu::ComputeKernel::create(library, "athenea/splat/splat_lift_mask", entry);
    };
    auto rectKernel = kernel("liftMaskRect");
    auto imageKernel = kernel("liftMaskImage");
    auto keyKernel = kernel("liftKey");
    if (!rectKernel) return report(rectKernel.error());
    if (!imageKernel) return report(imageKernel.error());
    if (!keyKernel) return report(keyKernel.error());
    const auto floats = [&](uint32_t channels, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = pixels * channels * sizeof(float);
        desc.elementBytes = channels * sizeof(float);
        desc.label = label;
        return gpu::Buffer::create(**device, desc);
    };
    auto mask = floats(1, "lift.mask");
    auto keyed = floats(4, "lift.keyed");
    auto noImage = floats(4, "lift.noImage");
    if (!mask) return report(mask.error());
    if (!keyed) return report(keyed.error());
    if (!noImage) return report(noImage.error());
    // Where a key is written: the one buffer a frame is read back from, unless a measure keeps several.
    const gpu::Buffer* keyedInto = nullptr;
    const auto dispatch = [&](gpu::ComputeKernel& which, const gpu::Buffer& image, const gpu::Buffer& matte) {
        gpu::CommandBatch batch(**device);
        which.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["image"].setBinding(image.rhi());
            cursor["matte"].setBinding(matte.rhi());
            cursor["mask"].setBinding(mask->rhi());
            cursor["keyed"].setBinding(keyedInto != nullptr ? keyedInto->rhi() : keyed->rhi());
            cursor["params"]["width"].setData(width);
            cursor["params"]["height"].setData(height);
            const bool rect = o.rect.size() == 4;
            cursor["params"]["left"].setData(rect ? static_cast<float>(o.rect[0]) : 0.0F);
            cursor["params"]["bottom"].setData(rect ? static_cast<float>(o.rect[1]) : 0.0F);
            cursor["params"]["right"].setData(rect ? static_cast<float>(o.rect[2]) : 1.0F);
            cursor["params"]["top"].setData(rect ? static_cast<float>(o.rect[3]) : 1.0F);
        });
        return batch.submit(true);
    };

    auto lift = render::makeSplatLift(**device, slots);
    if (!lift) return report(lift.error());
    for (size_t c = 0; c < cameras; ++c) {
        // This camera's mask: the rectangle, or the picture painted for it.
        if (o.masks.empty()) {
            if (auto made = dispatch(*rectKernel, *noImage, *mask); !made) return report(made.error());
        } else {
            auto read = io::readExr(o.masks[c]);
            if (!read) return report(read.error());
            if (read->width != width || read->height != height) {
                err("%s is %ux%u and the frame %ux%u\n", o.masks[c].c_str(), read->width, read->height, width,
                    height);
                return kExitFailure;
            }
            auto painted = gpu::Buffer::fromSpan<float>(**device, read->rgba, "lift.painted");
            if (!painted) return report(painted.error());
            if (auto made = dispatch(*imageKernel, *painted, *mask); !made) return report(made.error());
        }
        const render::Camera camera = cameraAt(&o.from[c * 3]);
        render::RenderSettings lifting = settings;
        render::RenderTargets targets;
        if (o.byTrace) {
            lifting.trace = true;
        } else {
            render::liftWhileBlending(lifting, *lift, *mask);
        }
        auto drawn = raster->render(camera, instances, lifting, targets);
        if (!drawn) return report(drawn.error());
        if (o.byTrace) {
            if (auto lifted = render::liftMask(library, targets, *mask, *lift); !lifted) return report(lifted.error());
        }
    }

    // The camera the matte is for: its frame, traced, and the matte from it.
    settings.trace = true;
    gpu::Buffer matte;
    render::MatteSettings how;
    how.hard = o.hard;
    how.threshold = o.threshold;
    const auto keyFrom = [&](const double* eye, const std::string& path) -> Result<void> {
        render::RenderTargets view;
        auto drawn = raster->render(cameraAt(eye), instances, settings, view);
        if (!drawn) return std::move(drawn).error();
        ATHENEA_TRY(render::drawMatte(library, view, *lift, how, matte));
        ATHENEA_TRY(dispatch(*keyKernel, view.colour, matte));
        auto rgba = keyed->readAll<float>(**device);
        if (!rgba) return std::move(rgba).error();
        auto depth = view.depth.readAll<float>(**device);
        if (!depth) return std::move(depth).error();
        return io::writeExr(path, width, height, *rgba, *depth);
    };
    if (auto written = keyFrom(o.eye.data(), o.output); !written) return report(written.error());
    // HOW MUCH THE KEY SHIMMERS as the camera goes by: three frames half a
    // degree apart round the target, the middle one against the mean of its
    // neighbours (`compareFlicker`), for the keyed frames and for the whole
    // ones they were keyed from. The camera moves, so both have parallax in
    // them; what is read is the key's against the cloud's own.
    {
        std::array<render::RenderTargets, 3> views;
        std::array<gpu::Buffer, 3> keys;
        const double dx = o.eye[0] - target.x;
        const double dz = o.eye[2] - target.z;
        for (int k = 0; k < 3; ++k) {
            const double a = 0.5 * (k - 1) * 3.14159265358979 / 180.0;
            const double eye[3] = {target.x + dx * std::cos(a) + dz * std::sin(a), o.eye[1],
                                   target.z - dx * std::sin(a) + dz * std::cos(a)};
            auto drawn = raster->render(cameraAt(eye), instances, settings, views[static_cast<size_t>(k)]);
            if (!drawn) return report(drawn.error());
            auto made = floats(4, "lift.keyed.measure");
            if (!made) return report(made.error());
            keys[static_cast<size_t>(k)] = *made;
            if (auto m = render::drawMatte(library, views[static_cast<size_t>(k)], *lift, how, matte); !m) {
                return report(m.error());
            }
            keyedInto = &keys[static_cast<size_t>(k)];
            auto keyedNow = dispatch(*keyKernel, views[static_cast<size_t>(k)].colour, matte);
            keyedInto = nullptr;
            if (!keyedNow) return report(keyedNow.error());
        }
        auto whole = render::compareFlicker(library, views[0].colour, views[1].colour, views[2].colour, width, height);
        if (!whole) return report(whole.error());
        auto key = render::compareFlicker(library, keys[0], keys[1], keys[2], width, height);
        if (!key) return report(key.error());
        out("three frames half a degree apart, the middle against its neighbours' mean: the keyed frames relMSE %.6f, "
            "p99 %u, max %u; the whole frames %.6f, %u, %u\n",
            key->hdr.relMse, key->codes.p99, key->codes.max, whole->hdr.relMse, whole->codes.p99, whole->codes.max);
    }
    // THE SAME KEY AS THE CAMERA GOES ROUND: the --eye camera turned about the
    // target's up axis, a frame a step -- what a matte carried by the
    // gaussians is for, since nothing is keyed again as the camera moves.
    if (o.turn > 0) {
        const std::string stem = o.output.size() > 4 ? o.output.substr(0, o.output.size() - 4) : o.output;
        const double dx = o.eye[0] - target.x;
        const double dy = o.eye[1] - target.y;
        const double dz = o.eye[2] - target.z;
        for (unsigned f = 0; f < o.turn; ++f) {
            const double a = 6.283185307179586 * f / o.turn;
            const double eye[3] = {target.x + dx * std::cos(a) + dz * std::sin(a), target.y + dy,
                                   target.z - dx * std::sin(a) + dz * std::cos(a)};
            char name[32];
            std::snprintf(name, sizeof(name), "_%03u.exr", f);
            if (auto written = keyFrom(eye, stem + name); !written) return report(written.error());
        }
        out("wrote %u frames of the camera going round: %s_NNN.exr\n", o.turn, stem.c_str());
    }
    out("wrote %s: %u gaussians, %zu lifting camera%s, %ux%u\n", o.output.c_str(), slots, cameras,
        cameras == 1 ? "" : "s", width, height);
    return 0;
}

}   // namespace

void addLift(CLI::App& app) {
    auto options = std::make_shared<LiftOptions>();
    LiftOptions& o = *options;
    CLI::App* cmd = app.add_subcommand("lift", "a mask on some cameras' frames, as the matte another camera sees");
    cmd->add_option("--splats", o.splats, ".ply / .splat / .spz / .sog files")->required();
    cmd->add_option("--from", o.from, "the cameras the mask is made on: x y z, once a camera");
    cmd->add_option("--mask", o.masks, "a painted mask a --from camera: an EXR of the frame's size, white inside");
    cmd->add_option("--rect", o.rect, "or the same rectangle of every --from frame: LEFT BOTTOM RIGHT TOP, 0..1")
        ->expected(4);
    cmd->add_option("--eye", o.eye, "the camera the matte is drawn for: x y z")->expected(3);
    cmd->add_option("--target", o.target, "what every camera looks at: x y z")->expected(3);
    cmd->add_option("--up", o.up, "up vector")->expected(3);
    cmd->add_option("--focal", o.focal, "focal length, mm");
    cmd->add_option("--size", o.size, "WIDTHxHEIGHT");
    cmd->add_option("--degree", o.degree, "harmonic degree cap 0..3");
    cmd->add_flag("--hard", o.hard, "a gaussian is in the mask or out of it");
    cmd->add_option("--threshold", o.threshold, "--hard: the share of a gaussian's weight inside the mask");
    cmd->add_flag("--by-trace", o.byTrace, "lift through each frame's trace rather than while blending");
    cmd->add_option("--turn", o.turn, "also this many frames of the --eye camera going once round the target (about y), keyed: <output>_NNN.exr");
    cmd->add_option("-o,--output", o.output, "EXR: the --eye frame keyed by the matte");
    cmd->callback([options] {
        if (const int code = run(*options); code != 0) {
            throw CLI::RuntimeError(code);
        }
    });
}

}   // namespace athenea::cli
