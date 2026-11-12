// Copyright (c) 2026 jesus luque.
//
// A WALL OF LIGHT AS A TEXTURE EACH GAUSSIAN LOOKS UP
// (shaders/athenea/neural/neural_wall.slang): an LED wall whose picture is
// known every frame, and a cloud whose gaussians each keep where on it they
// read, how wide and how much. A frame is then lit by whatever the wall shows
// at the cost of a filtered read or two a gaussian, with no light sampled and
// no ray.
#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/render/Camera.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::render {

/// neural_wall.slang's kWallLobe: u, v, the level, and the log of a gain a channel.
inline constexpr uint32_t kWallLobe = 6;

/// Where the wall stands, in the cloud's own space: its centre, its two unit
/// axes (u to the right of its picture, v up it) and half its size along each.
struct WallRect {
    std::array<float, 3> centre{0.0F, 0.0F, 0.0F};
    std::array<float, 3> axisU{1.0F, 0.0F, 0.0F};
    std::array<float, 3> axisV{0.0F, 1.0F, 0.0F};
    float                halfU = 0.5F;
    float                halfV = 0.5F;
};

struct NeuralWallSettings {
    uint32_t width = 640;    ///< the fit's frame
    uint32_t height = 360;
    /// Adam's rates: u and v (in walls a step), the level, the log of the gain.
    float    rateUv = 2.0e-3F;
    float    rateLevel = 1.0e-2F;
    float    rateGain = 1.0e-2F;
};

class NeuralWall {
public:
    [[nodiscard]] static Result<NeuralWall> create(gpu::ShaderLibrary& library);

    /// A wall of `width` x `height` texels holding `images` pictures, and
    /// `cloud`'s reads of it, `lobes` a gaussian, started where each
    /// gaussian's normal meets the wall: `startLevel` wide (a pyramid level;
    /// negative: the widest but one) and `startGain` strong together.
    /// `gridLevel` not negative: in place of reads each gaussian keeps a
    /// weight a texel of that level of the wall's pictures (0: the wall's own
    /// texels, 1: half as many a side) and a gain a channel -- what a matt
    /// thing takes from a wall, which is most of it and no read's shape.
    [[nodiscard]] Result<void> start(const scene::GpuSplats& cloud, const WallRect& wall, uint32_t lobes,
                                     uint32_t width, uint32_t height, uint32_t images, float startLevel = -1.0F,
                                     float startGain = 0.05F, int gridLevel = -1);

    /// Picture `image` written on the device and its pyramid built:
    /// neuralWallPattern's patterns (0: the one texel `argument` lit).
    [[nodiscard]] Result<void> pattern(uint32_t image, uint32_t pattern, uint32_t argument);
    /// Picture `image` from `texels` (width x height float4, bottom row
    /// first), and its pyramid: what a wall showing a video is given a frame.
    [[nodiscard]] Result<void> picture(uint32_t image, const gpu::Buffer& texels);

    /// THE WALL'S PICTURES MADE FINER THAN THE FIT: `levels` more levels (each
    /// twice the texels a side) and `images` pictures, those there were gone.
    /// The reads stay as they were learnt -- where on the wall, how wide --
    /// and now read pictures of the wall's real resolution; `accumulate`
    /// still counts texels as the fit did. Not for fitting on.
    [[nodiscard]] Result<void> finer(uint32_t levels, uint32_t images);

    /// Which picture the wall shows in the frames drawn from here on.
    void show(uint32_t image) noexcept { shown_ = image; }

    /// What the rasteriser calls for an instance that names this wall: the
    /// wall's light added to the colours of the cloud's gaussians in view.
    void shade(gpu::CommandBatch& batch, uint32_t base, const gpu::Buffer& proj, const gpu::Buffer& visible);

    /// One step of the fit from one camera: `student` (an instance that names
    /// this wall) drawn under the picture shown should be the frame
    /// `teacher` holds (a frame's colour, width x height), and the reads are
    /// moved towards it. With `readLoss`, the frame's relMSE before the step.
    [[nodiscard]] Result<std::optional<double>> step(TileRasterizer& rasterizer, const Camera& camera,
                                                      const SplatInstance& student, const gpu::Buffer& teacher,
                                                      const NeuralWallSettings& settings, bool readLoss = false);

    /// `frame` times a colour into `into` (both width x height float4):
    /// over what `into` holds, or with `first` over nothing. The colour is
    /// what picture `image` holds at `texel` -- a wall's picture is a sum of
    /// its texels and so is the frame it lights.
    [[nodiscard]] Result<void> accumulate(const gpu::Buffer& frame, gpu::Buffer& into, uint32_t pixels,
                                          uint32_t image, uint32_t texel, bool first);

    [[nodiscard]] uint32_t count() const noexcept { return count_; }
    [[nodiscard]] uint32_t lobes() const noexcept { return lobes_; }
    /// Numbers a gaussian keeps: six a read, or a grid's cells and three.
    [[nodiscard]] uint32_t stride() const noexcept { return cells_ != 0 ? cells_ + 3 : lobes_ * kWallLobe; }
    [[nodiscard]] uint32_t cells() const noexcept { return cells_; }
    [[nodiscard]] uint32_t width() const noexcept { return width_; }
    [[nodiscard]] uint32_t height() const noexcept { return height_; }
    [[nodiscard]] uint32_t levels() const noexcept { return levels_; }
    [[nodiscard]] const gpu::Buffer& reads() const noexcept { return latent_; }
    [[nodiscard]] const gpu::Buffer& images() const noexcept { return images_; }

private:
    void bind(rhi::ShaderCursor cursor) const;
    [[nodiscard]] Result<void> pyramid(gpu::CommandBatch& batch, uint32_t image);
    [[nodiscard]] Result<void> reserve(const NeuralWallSettings& settings);

    gpu::ShaderLibrary* library_ = nullptr;
    gpu::ComputeKernel  init_, pattern_, pyramid_, shade_, error_, clear_, backward_, adam_, accumulate_;
    const scene::GpuSplats* cloud_ = nullptr;
    WallRect            wall_;
    uint32_t            count_ = 0;
    uint32_t            lobes_ = 1;
    uint32_t            width_ = 0;
    uint32_t            height_ = 0;
    uint32_t            levels_ = 0;
    uint32_t            texels_ = 0;      ///< a pyramid's
    uint32_t            imageCount_ = 0;
    uint32_t            shown_ = 0;
    uint32_t            finer_ = 0;
    uint32_t            cells_ = 0;
    uint32_t            gridLevel_ = 0;
    uint32_t            steps_ = 0;
    uint32_t            pixels_ = 0;
    float               startLevel_ = 0.0F;
    float               startGain_ = 0.05F;
    gpu::Buffer         latent_, images_;
    gpu::Buffer         grad_, m_, v_, colourGrad_, pixelGrad_, lossRows_;
    RenderTargets       drawn_, again_;
    gpu::Buffer         spare_;           ///< what a kernel that reads no frame is handed for one
};

}   // namespace athenea::render
