// Copyright (c) 2026 jesus luque.
#include "athenea/render/NeuralWall.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include "athenea/gpu/Device.h"

namespace athenea::render {
namespace {

constexpr float kBeta1 = 0.9F;
constexpr float kBeta2 = 0.999F;

/// Zeroed on purpose: sums and moments start from nothing.
Result<gpu::Buffer> zeroed(gpu::Device& device, uint64_t words, uint32_t elementBytes, const char* label) {
    const std::vector<uint32_t> zeros(std::max<uint64_t>(words, 4), 0u);
    gpu::BufferDesc desc;
    desc.bytes = zeros.size() * sizeof(uint32_t);
    desc.elementBytes = elementBytes;
    desc.label = label;
    return gpu::Buffer::create(device, desc, zeros.data());
}

}   // namespace

Result<NeuralWall> NeuralWall::create(gpu::ShaderLibrary& library) {
    NeuralWall wall;
    wall.library_ = &library;
    const auto make = [&](gpu::ComputeKernel& kernel, const char* entry) -> Result<void> {
        auto made = gpu::ComputeKernel::create(library, "athenea/neural/neural_wall", entry);
        if (!made) return std::move(made).error();
        kernel = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(make(wall.init_, "neuralWallInit"));
    ATHENEA_TRY(make(wall.pattern_, "neuralWallPattern"));
    ATHENEA_TRY(make(wall.pyramid_, "neuralWallPyramid"));
    ATHENEA_TRY(make(wall.shade_, "neuralWallShade"));
    ATHENEA_TRY(make(wall.error_, "neuralWallError"));
    ATHENEA_TRY(make(wall.clear_, "neuralWallClear"));
    ATHENEA_TRY(make(wall.backward_, "neuralWallBackward"));
    ATHENEA_TRY(make(wall.adam_, "neuralWallAdam"));
    ATHENEA_TRY(make(wall.accumulate_, "neuralWallAccumulate"));
    return wall;
}

/// Every name the module declares, bound to the buffer of this wall it
/// means; a dispatch then names the few that are its own.
void NeuralWall::bind(rhi::ShaderCursor cursor) const {
    cursor["positions"].setBinding(cloud_->positions.rhi());
    cursor["shape"].setBinding(cloud_->shape.rhi());
    cursor["visible"].setBinding(colourGrad_.rhi());
    cursor["wallLatent"].setBinding(latent_.rhi());
    cursor["wallImage"].setBinding(images_.rhi());
    cursor["proj"].setBinding(spare_.rhi());
    cursor["made"].setBinding(spare_.rhi());
    cursor["imageOut"].setBinding(spare_.rhi());
    cursor["colourGrad"].setBinding(colourGrad_.rhi());
    cursor["gradient"].setBinding(spare_.rhi());
    cursor["gradientRead"].setBinding(grad_.rhi());
    cursor["moment1"].setBinding(m_.rhi());
    cursor["moment2"].setBinding(v_.rhi());
    cursor["student"].setBinding(pixelGrad_.rhi());
    cursor["teacher"].setBinding(pixelGrad_.rhi());
    cursor["pixelGrad"].setBinding(spare_.rhi());
    cursor["loss"].setBinding(lossRows_.rhi());
    cursor["cleared"].setBinding(spare_.rhi());
    cursor["frameIn"].setBinding(pixelGrad_.rhi());
    cursor["frameOut"].setBinding(spare_.rhi());
    rhi::ShaderCursor p = cursor["params"];
    p["count"].setData(count_);
    p["base"].setData(uint32_t{0});
    p["lobes"].setData(lobes_);
    p["image"].setData(shown_);
    p["width"].setData(width_);
    p["height"].setData(height_);
    p["levels"].setData(levels_);
    p["level"].setData(uint32_t{0});
    p["centre"].setData(wall_.centre);
    p["halfU"].setData(wall_.halfU);
    p["axisU"].setData(wall_.axisU);
    p["halfV"].setData(wall_.halfV);
    p["axisV"].setData(wall_.axisV);
    p["startLevel"].setData(startLevel_);
    p["startGain"].setData(startGain_);
}

Result<void> NeuralWall::start(const scene::GpuSplats& cloud, const WallRect& wall, uint32_t lobes, uint32_t width,
                               uint32_t height, uint32_t images, float startLevel, float startGain) {
    if (cloud.count == 0 || width == 0 || height == 0 || images == 0 || lobes == 0) {
        return Error(ErrorCode::InvalidArgument, "a wall wants gaussians, texels, a picture and a lobe");
    }
    gpu::Device& device = library_->device();
    cloud_ = &cloud;
    wall_ = wall;
    count_ = cloud.count;
    lobes_ = lobes;
    width_ = width;
    height_ = height;
    imageCount_ = images;
    levels_ = 1;
    texels_ = width * height;
    for (uint32_t w = width, h = height; w > 1 || h > 1;) {
        w = std::max(w >> 1, 1u);
        h = std::max(h >> 1, 1u);
        texels_ += w * h;
        ++levels_;
    }
    startLevel_ = startLevel < 0.0F ? static_cast<float>(levels_ > 1 ? levels_ - 2 : 0) : startLevel;
    startGain_ = startGain;
    steps_ = 0;
    const uint64_t numbers = uint64_t{count_} * lobes_ * kWallLobe;
    const auto assign = [&](gpu::Buffer& into, uint64_t words, uint32_t element, const char* label) -> Result<void> {
        auto made = zeroed(device, words, element, label);
        if (!made) return std::move(made).error();
        into = *made;
        return ok();
    };
    ATHENEA_TRY(assign(latent_, numbers, 4, "wall.reads"));
    ATHENEA_TRY(assign(grad_, numbers, 4, "wall.grad"));
    ATHENEA_TRY(assign(m_, numbers, 4, "wall.m"));
    ATHENEA_TRY(assign(v_, numbers, 4, "wall.v"));
    ATHENEA_TRY(assign(colourGrad_, uint64_t{count_} * 4, 4, "wall.colourGrad"));
    ATHENEA_TRY(assign(images_, uint64_t{texels_} * images * 4, 16, "wall.images"));
    ATHENEA_TRY(assign(spare_, 64, 4, "wall.spare"));
    if (!pixelGrad_.valid()) {
        ATHENEA_TRY(assign(pixelGrad_, 16, 16, "wall.pixelGrad"));
        ATHENEA_TRY(assign(lossRows_, 4, 4, "wall.loss"));
        pixels_ = 0;
    }
    gpu::CommandBatch batch(device);
    init_.dispatch(batch, {count_, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["made"].setBinding(latent_.rhi());
        cursor["wallLatent"].setBinding(grad_.rhi());
    });
    return batch.submit(true);
}

Result<void> NeuralWall::pyramid(gpu::CommandBatch& batch, uint32_t image) {
    for (uint32_t level = 1; level < levels_; ++level) {
        const uint32_t w = std::max(width_ >> level, 1u);
        const uint32_t h = std::max(height_ >> level, 1u);
        pyramid_.dispatch(batch, {w * h, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor);
            cursor["imageOut"].setBinding(images_.rhi());
            cursor["wallImage"].setBinding(spare_.rhi());
            cursor["params"]["image"].setData(image);
            cursor["params"]["level"].setData(level);
        });
    }
    return ok();
}

Result<void> NeuralWall::pattern(uint32_t image, uint32_t pattern, uint32_t argument) {
    if (image >= imageCount_) {
        return Error(ErrorCode::InvalidArgument, "the wall holds no such picture");
    }
    gpu::CommandBatch batch(library_->device());
    pattern_.dispatch(batch, {width_ * height_, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["imageOut"].setBinding(images_.rhi());
        cursor["wallImage"].setBinding(spare_.rhi());
        cursor["params"]["image"].setData(image);
        cursor["params"]["pattern"].setData(pattern);
        cursor["params"]["argument"].setData(argument);
    });
    ATHENEA_TRY(pyramid(batch, image));
    return batch.submit();
}

Result<void> NeuralWall::picture(uint32_t image, const gpu::Buffer& texels) {
    if (image >= imageCount_) {
        return Error(ErrorCode::InvalidArgument, "the wall holds no such picture");
    }
    gpu::CommandBatch batch(library_->device());
    pattern_.dispatch(batch, {width_ * height_, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["frameIn"].setBinding(texels.rhi());
        cursor["imageOut"].setBinding(images_.rhi());
        cursor["wallImage"].setBinding(spare_.rhi());
        cursor["params"]["image"].setData(image);
        cursor["params"]["pattern"].setData(uint32_t{6});
    });
    ATHENEA_TRY(pyramid(batch, image));
    return batch.submit();
}

void NeuralWall::shade(gpu::CommandBatch& batch, uint32_t base, const gpu::Buffer& proj, const gpu::Buffer& visible) {
    if (count_ == 0) {
        return;
    }
    shade_.dispatch(batch, {count_, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["proj"].setBinding(proj.rhi());
        cursor["visible"].setBinding(visible.rhi());
        cursor["params"]["base"].setData(base);
    });
}

Result<void> NeuralWall::reserve(const NeuralWallSettings& settings) {
    const uint32_t pixels = settings.width * settings.height;
    if (pixels == pixels_) {
        return ok();
    }
    gpu::Device& device = library_->device();
    auto grad = zeroed(device, uint64_t{pixels} * 4, 16, "wall.pixelGrad");
    if (!grad) return std::move(grad).error();
    auto rows = zeroed(device, settings.height, 4, "wall.loss");
    if (!rows) return std::move(rows).error();
    pixelGrad_ = *grad;
    lossRows_ = *rows;
    pixels_ = pixels;
    return ok();
}

Result<std::optional<double>> NeuralWall::step(TileRasterizer& rasterizer, const Camera& camera,
                                               const SplatInstance& student, const gpu::Buffer& teacher,
                                               const NeuralWallSettings& settings, bool readLoss) {
    if (student.splats == nullptr || student.splats->count != count_ || student.wall != this) {
        return Error(ErrorCode::InvalidArgument, "the instance is not the cloud this wall's reads are of");
    }
    ATHENEA_TRY(reserve(settings));
    gpu::Device& device = library_->device();
    RenderSettings frame;
    frame.width = settings.width;
    frame.height = settings.height;
    auto drawn = rasterizer.render(camera, std::span<const SplatInstance>(&student, 1), frame, drawn_);
    if (!drawn) return std::move(drawn).error();
    ++steps_;
    const float t = static_cast<float>(steps_);
    const auto fit = [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        rhi::ShaderCursor p = cursor["params"];
        p["frameWidth"].setData(settings.width);
        p["frameHeight"].setData(settings.height);
        p["rateUv"].setData(settings.rateUv);
        p["rateLevel"].setData(settings.rateLevel);
        p["rateGain"].setData(settings.rateGain);
        p["beta1"].setData(kBeta1);
        p["beta2"].setData(kBeta2);
        p["epsilon"].setData(1.0e-8F);
        p["correction1"].setData(1.0F - std::pow(kBeta1, t));
        p["correction2"].setData(1.0F - std::pow(kBeta2, t));
    };
    {
        gpu::CommandBatch batch(device);
        clear_.dispatch(batch, {count_ * 4, 1, 1}, [&](rhi::ShaderCursor cursor) {
            fit(cursor);
            cursor["cleared"].setBinding(colourGrad_.rhi());
            cursor["colourGrad"].setBinding(grad_.rhi());
            cursor["visible"].setBinding(grad_.rhi());
            cursor["params"]["count"].setData(count_ * 4);
        });
        clear_.dispatch(batch, {settings.height, 1, 1}, [&](rhi::ShaderCursor cursor) {
            fit(cursor);
            cursor["cleared"].setBinding(lossRows_.rhi());
            cursor["loss"].setBinding(spare_.rhi());
            cursor["params"]["count"].setData(settings.height);
        });
        error_.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
            fit(cursor);
            cursor["student"].setBinding(drawn_.colour.rhi());
            cursor["teacher"].setBinding(teacher.rhi());
            cursor["pixelGrad"].setBinding(pixelGrad_.rhi());
            cursor["frameIn"].setBinding(teacher.rhi());
        });
        ATHENEA_TRY(batch.submit());
    }
    // The same frame again, the blend carrying each pixel's error to the gaussians it is made of.
    RenderSettings carrying = frame;
    carrying.contribution = &colourGrad_;
    carrying.contributionMode = 4;
    carrying.finalColour = &pixelGrad_;
    drawn = rasterizer.render(camera, std::span<const SplatInstance>(&student, 1), carrying, again_);
    if (!drawn) return std::move(drawn).error();
    {
        gpu::CommandBatch batch(device);
        backward_.dispatch(batch, {count_, 1, 1}, [&](rhi::ShaderCursor cursor) {
            fit(cursor);
            cursor["gradient"].setBinding(grad_.rhi());
            cursor["gradientRead"].setBinding(latent_.rhi());
        });
        const uint32_t numbers = count_ * lobes_ * kWallLobe;
        adam_.dispatch(batch, {numbers, 1, 1}, [&](rhi::ShaderCursor cursor) {
            fit(cursor);
            cursor["made"].setBinding(latent_.rhi());
            cursor["wallLatent"].setBinding(colourGrad_.rhi());
            cursor["params"]["count"].setData(numbers);
        });
        ATHENEA_TRY(batch.submit(readLoss));
    }
    if (!readLoss) {
        return std::optional<double>{};
    }
    std::vector<uint32_t> rows(settings.height);
    ATHENEA_TRY(lossRows_.read(device, 0, rows.size() * sizeof(uint32_t), rows.data()));
    uint64_t total = 0;
    for (const uint32_t row : rows) {
        total += row;
    }
    return std::optional<double>{static_cast<double>(total) / 1024.0 / static_cast<double>(pixels_)};
}

Result<void> NeuralWall::accumulate(const gpu::Buffer& frame, gpu::Buffer& into, uint32_t pixels, uint32_t image,
                                    uint32_t texel, bool first) {
    gpu::CommandBatch batch(library_->device());
    accumulate_.dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["frameIn"].setBinding(frame.rhi());
        cursor["frameOut"].setBinding(into.rhi());
        cursor["params"]["count"].setData(uint32_t{first ? 0u : 1u});
        cursor["params"]["image"].setData(image);
        cursor["params"]["texel"].setData(texel);
        cursor["params"]["byTexel"].setData(uint32_t{1});
        cursor["params"]["frameWidth"].setData(pixels);
        cursor["params"]["frameHeight"].setData(uint32_t{1});
    });
    return batch.submit();
}

}   // namespace athenea::render
