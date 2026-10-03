// Copyright (c) 2026 jesus luque.
//
// `athenea compare`: what one or two EXR images hold, measured on the GPU.
// The CPU reads the files; the means, the largest values and the differences
// are kernels (render::imageStats, compareHdr, compareImages), the same ones
// the tests use. One image: its statistics. Two: each one's, and how far the
// first is from the second, the second taken as the reference.
#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/io/Exr.h"
#include "athenea/render/ReferenceRenderer.h"

namespace athenea::cli {
namespace {

struct Options {
    std::string           image;
    std::string           reference;
    std::vector<uint32_t> window;   // x0 y0 x1 y1, bottom row first
};

Result<gpu::Buffer> upload(gpu::Device& device, const io::ExrPixels& pixels, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = pixels.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    desc.label = label;
    return gpu::Buffer::create(device, desc, pixels.rgba.data());
}

void printStats(const char* name, const render::ImageStats& stats) {
    std::printf("%-9s mean %.6g %.6g %.6g %.6g   max %.6g %.6g %.6g %.6g   (%llu pixels)\n", name,
                stats.mean[0], stats.mean[1], stats.mean[2], stats.mean[3], stats.max[0], stats.max[1],
                stats.max[2], stats.max[3], static_cast<unsigned long long>(stats.pixels));
}

}   // namespace

void addCompare(CLI::App& app) {
    auto* cmd = app.add_subcommand("compare", "what one or two EXR images hold, and how far apart they are, "
                                              "measured on the GPU");
    auto o = std::make_shared<Options>();
    cmd->add_option("image", o->image, "the image measured (.exr)")->required();
    cmd->add_option("reference", o->reference, "the image it is compared with (.exr, same size)");
    cmd->add_option("--window", o->window,
                    "X0 Y0 X1 Y1: only the pixels in [X0, X1) x [Y0, Y1), rows counted from the bottom "
                    "(default: the whole image)")
        ->expected(4);
    cmd->callback([o] {
        const auto fail = [](const Error& error) {
            std::fprintf(stderr, "compare: %s\n", error.toString().c_str());
            throw CLI::RuntimeError(1);
        };
        auto device = gpu::Device::create();
        if (!device) fail(device.error());
        gpu::ShaderLibrary library(*device);

        auto image = io::readExr(o->image);
        if (!image) fail(image.error());
        auto imageBuffer = upload(**device, *image, "compare.image");
        if (!imageBuffer) fail(imageBuffer.error());

        std::array<uint32_t, 4> w{0, 0, 0, 0};
        if (o->window.size() == 4) {
            w = {o->window[0], o->window[1], o->window[2], o->window[3]};
        }
        auto stats = render::imageStats(library, *imageBuffer, image->width, image->height, w[0], w[1], w[2], w[3]);
        if (!stats) fail(stats.error());
        printStats("image", *stats);
        if (o->reference.empty()) {
            return;
        }

        auto reference = io::readExr(o->reference);
        if (!reference) fail(reference.error());
        if (reference->width != image->width || reference->height != image->height) {
            fail(Error::make(ErrorCode::InvalidArgument, "{}x{} against {}x{}", image->width, image->height,
                             reference->width, reference->height));
        }
        auto referenceBuffer = upload(**device, *reference, "compare.reference");
        if (!referenceBuffer) fail(referenceBuffer.error());
        auto referenceStats =
            render::imageStats(library, *referenceBuffer, image->width, image->height, w[0], w[1], w[2], w[3]);
        if (!referenceStats) fail(referenceStats.error());
        printStats("reference", *referenceStats);

        // The differences are over the whole image: they are distributions,
        // and a window is for the means above.
        auto hdr = render::compareHdr(library, *imageBuffer, *referenceBuffer, image->width, image->height);
        if (!hdr) fail(hdr.error());
        std::printf("hdr       relMSE %.6g   p99 relative %.4g   max relative %.4g\n", hdr->relMse, hdr->p99Relative,
                    hdr->maxRelative);
        auto ldr = render::compareImages(library, *imageBuffer, *referenceBuffer, image->width, image->height);
        if (!ldr) fail(ldr.error());
        std::printf("8-bit     p99 %u   max %u   over 2: %llu of %llu pixels\n", ldr->p99, ldr->max,
                    static_cast<unsigned long long>(ldr->over2), static_cast<unsigned long long>(ldr->pixels));
    });
}

}   // namespace athenea::cli
