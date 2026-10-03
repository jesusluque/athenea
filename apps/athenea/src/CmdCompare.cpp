// Copyright (c) 2026 jesus luque.
//
// `athenea compare`: what one or two EXR images hold, measured on the GPU.
// The CPU reads the files; the means, the largest values and the differences
// are the Measure effect's kernels (plugins/measure), run through the AOFX
// host as a compositor runs them, so this command and a QC node in
// openFXplayer are one implementation. One image: its statistics. Two: each
// one's, and how far the first is from the second, the second taken as the
// reference.
#include <algorithm>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Commands.h"
#include "aofx/Effect.h"
#include "athenea/aofx/EffectRegistry.h"
#include "athenea/aofx/EffectRender.h"
#include "athenea/gpu_host/Context.h"
#include "athenea/image/Image.h"
#include "athenea/io/Exr.h"

namespace athenea::cli {
namespace {

struct Options {
    std::string              image;
    std::string              reference;
    std::vector<uint32_t>    window;   // x0 y0 x1 y1, bottom row first
    std::string              heatmap;
    std::string              show = "codes";
    double                   gain = 1.0;
    std::vector<std::string> paths;
};

/// The file's pixels in an image on the device, rows as the file has them
/// (bottom first).
Result<image::ImagePtr> load(const io::ExrPixels& pixels) {
    auto image = image::Image::create({0, 0, static_cast<int32_t>(pixels.width), static_cast<int32_t>(pixels.height)});
    if (!image) return std::move(image).error();
    auto floats = (*image)->floats();
    const auto stride = static_cast<size_t>((*image)->stride());
    for (uint32_t y = 0; y < pixels.height; ++y) {
        const size_t row = size_t{y} * pixels.width * 4;
        std::copy_n(pixels.rgba.begin() + static_cast<std::ptrdiff_t>(row), size_t{pixels.width} * 4,
                    floats.begin() + static_cast<std::ptrdiff_t>(size_t{y} * stride * 4));
    }
    return *image;
}

/// A count the effect attached as two exact floats, high and low 24 bits.
unsigned long long countAt(const std::vector<float>& values, size_t at) {
    return (static_cast<unsigned long long>(values[at]) << 24U) + static_cast<unsigned long long>(values[at + 1]);
}

/// "source" / "reference": mean, max and sum of each channel, then the
/// window's pixels. The mean printed is the sum over the pixels in double,
/// as it always was.
void printStats(const char* name, const std::vector<float>& values) {
    const unsigned long long pixels = countAt(values, 12);
    double mean[4];
    for (size_t c = 0; c < 4; ++c) {
        mean[c] = static_cast<double>(values[8 + c]) / static_cast<double>(pixels);
    }
    std::printf("%-9s mean %.6g %.6g %.6g %.6g   max %.6g %.6g %.6g %.6g   (%llu pixels)\n", name, mean[0],
                mean[1], mean[2], mean[3], static_cast<double>(values[4]), static_cast<double>(values[5]),
                static_cast<double>(values[6]), static_cast<double>(values[7]), pixels);
}

int showIndex(const std::string& show) {
    if (show == "source") return 0;
    if (show == "difference") return 1;
    if (show == "relative") return 2;
    return 3;
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
    cmd->add_option("--heatmap", o->heatmap, "also write where the two differ, as an EXR");
    cmd->add_option("--show", o->show, "what --heatmap draws")
        ->check(CLI::IsMember({"source", "difference", "relative", "codes"}));
    cmd->add_option("--gain", o->gain, "what --heatmap is multiplied by");
    cmd->add_option("--path", o->paths, "extra bundle directories (after $AOFX_PLUGIN_PATH)");
    cmd->callback([o] {
        const auto fail = [](const Error& error) {
            std::fprintf(stderr, "compare: ");
            cli::fail(error);
        };
        gpu_host::Context* context = gpu_host::installProcessContext();
        if (context == nullptr || context->compute() == nullptr) {
            fail(Error(ErrorCode::Unsupported, "no GPU compute device for AOFX kernels (gpe has no backend here)"));
        }
        aofx_host::EffectRegistry registry;
        for (const std::string& path : o->paths) {
            registry.addSearchPath(path);
        }
#ifdef ATHENEA_AOFX_BUNDLE_DIR
        registry.addSearchPath(ATHENEA_AOFX_BUNDLE_DIR);
#endif
        registry.scan(context);
        aofx::Effect* effect = registry.find("rt.sparrow.aofx.measure");
        if (effect == nullptr) {
            fail(Error(ErrorCode::NotFound, "no Measure bundle on the AOFX search path (try `athenea aofx list`)"));
        }

        auto image = io::readExr(o->image);
        if (!image) fail(image.error());
        std::optional<io::ExrPixels> reference;
        if (!o->reference.empty()) {
            auto read = io::readExr(o->reference);
            if (!read) fail(read.error());
            if (read->width != image->width || read->height != image->height) {
                fail(Error::make(ErrorCode::InvalidArgument, "{}x{} against {}x{}", image->width, image->height,
                                 read->width, read->height));
            }
            reference = std::move(*read);
        }

        aofx_host::EffectJob job;
        auto source = load(*image);
        if (!source) fail(source.error());
        job.inputs.push_back({"Source", *source});
        if (reference) {
            auto loaded = load(*reference);
            if (!loaded) fail(loaded.error());
            job.inputs.push_back({"Reference", *loaded});
        }
        if (o->window.size() == 4) {
            job.params.push_back(aofx::ParamValue{
                "window",
                {static_cast<double>(o->window[0]), static_cast<double>(o->window[1]),
                 static_cast<double>(o->window[2]), static_cast<double>(o->window[3])},
                {}});
        }
        job.params.push_back(aofx::ParamValue{"mode", {static_cast<double>(showIndex(o->show))}, {}});
        job.params.push_back(aofx::ParamValue{"gain", {o->gain}, {}});
        auto rendered = aofx_host::renderEffect(*context, *effect, job);
        if (!rendered) fail(rendered.error());
        const image::Image& out = **rendered;

        const std::vector<float>* stats = out.attached("source");
        if (stats == nullptr || stats->size() < 14) {
            fail(Error(ErrorCode::DeviceFailure, "the Measure effect attached no statistics"));
        }
        printStats("image", *stats);
        if (reference) {
            const std::vector<float>* referenceStats = out.attached("reference");
            const std::vector<float>* hdr = out.attached("hdr");
            const std::vector<float>* codes = out.attached("codes");
            if (referenceStats == nullptr || referenceStats->size() < 14 || hdr == nullptr || hdr->size() < 6 ||
                codes == nullptr || codes->size() < 7) {
                fail(Error(ErrorCode::DeviceFailure, "the Measure effect attached no differences"));
            }
            printStats("reference", *referenceStats);
            // The differences are over the whole image: they are
            // distributions, and a window is for the means above.
            const unsigned long long pixels = countAt(*hdr, 4);
            const double relMse =
                pixels > 0 ? static_cast<double>((*hdr)[3]) / static_cast<double>(pixels) : 0.0;
            std::printf("hdr       relMSE %.6g   p99 relative %.4g   max relative %.4g\n", relMse,
                        static_cast<double>((*hdr)[1]), static_cast<double>((*hdr)[2]));
            std::printf("8-bit     p99 %u   max %u   over 2: %llu of %llu pixels\n",
                        static_cast<unsigned>((*codes)[0]), static_cast<unsigned>((*codes)[1]), countAt(*codes, 2),
                        countAt(*codes, 5));
        }

        if (!o->heatmap.empty()) {
            const auto width = static_cast<uint32_t>(out.bounds().width());
            const auto height = static_cast<uint32_t>(out.bounds().height());
            std::vector<float> rgba(size_t{width} * height * 4);
            const auto floats = out.floats();
            const auto stride = static_cast<size_t>(out.stride());
            for (uint32_t y = 0; y < height; ++y) {
                std::copy_n(floats.begin() + static_cast<std::ptrdiff_t>(size_t{y} * stride * 4), size_t{width} * 4,
                            rgba.begin() + static_cast<std::ptrdiff_t>(size_t{y} * width * 4));
            }
            if (auto written = io::writeExr(o->heatmap, width, height, rgba, {}, false); !written) {
                fail(written.error());
            }
        }
    });
}

}   // namespace athenea::cli
