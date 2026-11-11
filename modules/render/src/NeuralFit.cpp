// Copyright (c) 2026 jesus luque.
#include "athenea/render/NeuralFit.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "aofx/Transform.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"

namespace athenea::render {
namespace {

/// neural_fit.slang's kFitRecord.
constexpr uint32_t kFitRecord = 1 + kNeuralIn + kNeuralHidden * 2 + 3 + kNeuralHidden * 2;
constexpr float kBeta1 = 0.9F;
constexpr float kBeta2 = 0.999F;

/// Zeroed on purpose: sums and moments start from nothing, and a buffer
/// nobody wrote is not a buffer of zeros on every device.
Result<gpu::Buffer> zeroed(gpu::Device& device, uint64_t words, const char* label) {
    const std::vector<uint32_t> zeros(std::max<uint64_t>(words, 1), 0u);
    gpu::BufferDesc desc;
    desc.bytes = zeros.size() * sizeof(uint32_t);
    desc.elementBytes = sizeof(uint32_t);
    desc.label = label;
    return gpu::Buffer::create(device, desc, zeros.data());
}

}   // namespace

Result<NeuralFit> NeuralFit::create(gpu::ShaderLibrary& library) {
    NeuralFit fit;
    fit.library_ = &library;
    const auto make = [&](gpu::ComputeKernel& kernel, const char* entry) -> Result<void> {
        auto made = gpu::ComputeKernel::create(library, "athenea/neural/neural_fit", entry);
        if (!made) return std::move(made).error();
        kernel = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(make(fit.error_, "neuralFitError"));
    ATHENEA_TRY(make(fit.clear_, "neuralFitClear"));
    ATHENEA_TRY(make(fit.backward_, "neuralFitBackward"));
    ATHENEA_TRY(make(fit.weights_, "neuralFitWeights"));
    ATHENEA_TRY(make(fit.adam_, "neuralFitAdam"));
    return fit;
}

Result<void> NeuralFit::reserve(const NeuralCloud& neural, const NeuralFitSettings& settings) {
    gpu::Device& device = library_->device();
    const auto assign = [&](gpu::Buffer& into, uint64_t words, const char* label) -> Result<void> {
        auto made = zeroed(device, words, label);
        if (!made) return std::move(made).error();
        into = *made;
        return ok();
    };
    if (neural.count != count_ || !records_.valid()) {
        count_ = neural.count;
        steps_ = 0;
        ATHENEA_TRY(assign(colourGrad_, uint64_t{count_} * 3, "fit.colourGrad"));
        ATHENEA_TRY(assign(records_, uint64_t{count_} * kFitRecord, "fit.records"));
        ATHENEA_TRY(assign(latentGrad_, uint64_t{count_} * kNeuralStride, "fit.latentGrad"));
        ATHENEA_TRY(assign(latentM_, uint64_t{count_} * kNeuralStride, "fit.latentM"));
        ATHENEA_TRY(assign(latentV_, uint64_t{count_} * kNeuralStride, "fit.latentV"));
        ATHENEA_TRY(assign(weightGrad_, kNeuralWeights, "fit.weightGrad"));
        ATHENEA_TRY(assign(weightM_, kNeuralWeights, "fit.weightM"));
        ATHENEA_TRY(assign(weightV_, kNeuralWeights, "fit.weightV"));
    }
    const uint32_t pixels = settings.width * settings.height;
    if (pixels != pixels_ || !pixelGrad_.valid()) {
        pixels_ = pixels;
        ATHENEA_TRY(assign(pixelGrad_, uint64_t{pixels} * 4, "fit.pixelGrad"));
        ATHENEA_TRY(assign(lossRows_, settings.height, "fit.loss"));
    }
    return ok();
}

Result<std::optional<double>> NeuralFit::step(TileRasterizer& rasterizer, const Camera& camera,
                                              const SplatInstance& instance, NeuralCloud& neural,
                                              const NeuralFitSettings& settings, bool readLoss,
                                              const NeuralFitLight* light) {
    if (instance.splats == nullptr || instance.splats->count != neural.count) {
        return Error(ErrorCode::InvalidArgument, "the neural cloud is not this cloud's, gaussian for gaussian");
    }
    ATHENEA_TRY(reserve(neural, settings));
    gpu::Device& device = library_->device();
    const scene::GpuSplats& cloud = *instance.splats;

    RenderSettings frame;
    frame.width = settings.width;
    frame.height = settings.height;
    SplatInstance byHarmonics = instance;
    byHarmonics.neural = nullptr;
    SplatInstance byLatents = instance;
    byLatents.neural = &neural;
    const bool lit = neural.lit && light != nullptr;
    if (neural.lit != (light != nullptr)) {
        return Error(ErrorCode::InvalidArgument, "a lit neural cloud is fitted under a light, and only a lit one is");
    }
    const bool pictured = lit && light->picture != nullptr;
    if (pictured && (light->picture->splats == nullptr || light->picture->splats->count != neural.count)) {
        return Error(ErrorCode::InvalidArgument, "the picture's cloud is not this cloud's gaussians");
    }
    if (lit) {
        if (pictured) {
            byHarmonics = *light->picture;
            byHarmonics.neural = nullptr;
            byHarmonics.relight = false;
        } else {
            byHarmonics.relight = true;
        }
        byLatents.neuralLight = &light->light;
    }
    const SplatLights* teacherLights = lit && !pictured ? light->teacher : nullptr;

    // What the frame should be, and what it is.
    auto drawn = rasterizer.render(camera, std::span<const SplatInstance>(&byHarmonics, 1), frame, teacher_, {},
                                   nullptr, teacherLights);
    if (!drawn) return std::move(drawn).error();
    // A shadowed neural cloud takes what reaches it from the same shadow scene the picture was made with.
    const SplatLights* studentLights =
        lit && neural.shadowed ? (light->student != nullptr ? light->student : light->teacher) : nullptr;
    const bool measured = studentLights != nullptr && studentLights->visibilityFactors != nullptr &&
                          studentLights->visibilityLights > 0;
    drawn = rasterizer.render(camera, std::span<const SplatInstance>(&byLatents, 1), frame, student_, {}, nullptr,
                              studentLights);
    if (!drawn) return std::move(drawn).error();

    const Projection projection = projectionFor(camera, settings.width, settings.height);
    const Vec3 eye = aofx::xform::inverseAffine(instance.objectToWorld).point(projection.eyeWorld);
    ++steps_;
    const float t = static_cast<float>(steps_);

    // What reaches each gaussian, where the cloud is shadowed: known once the frame that carries the
    // error back has cast its shadows.
    uint32_t shadowLights = 0;
    // Every name the module declares is bound, to the buffer of this fit it
    // means; a kernel reads the few it uses.
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["student"].setBinding(student_.colour.rhi());
        cursor["teacher"].setBinding(teacher_.colour.rhi());
        cursor["pixelGrad"].setBinding(pixelGrad_.rhi());
        cursor["loss"].setBinding(lossRows_.rhi());
        cursor["colourGrad"].setBinding(colourGrad_.rhi());
        cursor["positions"].setBinding(cloud.positions.rhi());
        cursor["shape"].setBinding(cloud.shape.rhi());
        cursor["shadowFactors"].setBinding(shadowLights == 0 ? latentGrad_.rhi()
                                           : measured        ? studentLights->visibilityFactors->rhi()
                                                             : rasterizer.shadowFactors().rhi());
        cursor["params"]["shadowLights"].setData(shadowLights);
        cursor["latent"].setBinding(neural.latent.rhi());
        cursor["weights"].setBinding(neural.weights.rhi());
        cursor["records"].setBinding(records_.rhi());
        cursor["gradient"].setBinding(latentGrad_.rhi());
        cursor["recordsRead"].setBinding(records_.rhi());
        cursor["value"].setBinding(neural.latent.rhi());
        cursor["moment1"].setBinding(latentM_.rhi());
        cursor["moment2"].setBinding(latentV_.rhi());
        cursor["gradientRead"].setBinding(latentGrad_.rhi());
        cursor["cleared"].setBinding(colourGrad_.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["width"].setData(settings.width);
        p["height"].setData(settings.height);
        p["count"].setData(count_);
        p["base"].setData(uint32_t{0});
        p["eyeX"].setData(static_cast<float>(eye.x));
        p["eyeY"].setData(static_cast<float>(eye.y));
        p["eyeZ"].setData(static_cast<float>(eye.z));
        p["stride"].setData(kNeuralStride);
        p["firstOfSecond"].setData(kNeuralFeatures);
        p["rate"].setData(settings.rateFeatures);
        p["rateSecond"].setData(settings.rateBase);
        p["beta1"].setData(kBeta1);
        p["beta2"].setData(kBeta2);
        p["epsilon"].setData(1.0e-8F);
        p["correction1"].setData(1.0F - std::pow(kBeta1, t));
        p["correction2"].setData(1.0F - std::pow(kBeta2, t));
        p["sparse"].setData(uint32_t{1});
        // The network's gradient from a share of the gaussians, another each
        // step: it is a sum over hundreds of thousands, and an eighth of them
        // says the same at an eighth of the reads.
        const uint32_t every = count_ > 20000 ? 8u : 1u;
        p["every"].setData(every);
        p["phase"].setData(steps_ % every);
        Vec3 towards{0.0, 0.0, 1.0};
        NeuralLight told;
        if (lit) {
            told = light->light;
            towards = aofx::xform::inverseAffine(instance.objectToWorld)
                          .direction({told.towards[0], told.towards[1], told.towards[2]});
            const double length = std::sqrt(towards.x * towards.x + towards.y * towards.y + towards.z * towards.z);
            if (length > 0.0) {
                towards = towards * (1.0 / length);
            }
        }
        p["lit"].setData(uint32_t{lit ? 1u : 0u});
        p["lightX"].setData(static_cast<float>(towards.x));
        p["lightY"].setData(static_cast<float>(towards.y));
        p["lightZ"].setData(static_cast<float>(towards.z));
        p["radianceR"].setData(told.radiance[0]);
        p["radianceG"].setData(told.radiance[1]);
        p["radianceB"].setData(told.radiance[2]);
    };
    {
        gpu::CommandBatch batch(device);
        clear_.dispatch(batch, {count_ * 3, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor);
            cursor["params"]["count"].setData(count_ * 3);
        });
        clear_.dispatch(batch, {settings.height, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor);
            cursor["cleared"].setBinding(lossRows_.rhi());
            cursor["params"]["count"].setData(settings.height);
        });
        error_.dispatch(batch, {settings.width, settings.height, 1}, bind);
        ATHENEA_TRY(batch.submit());
    }
    // The same frame again, the blend carrying each pixel's error to the
    // gaussians it is made of.
    RenderSettings carrying = frame;
    carrying.contribution = &colourGrad_;
    carrying.contributionMode = 4;
    carrying.finalColour = &pixelGrad_;
    drawn = rasterizer.render(camera, std::span<const SplatInstance>(&byLatents, 1), carrying, again_, {}, nullptr,
                              studentLights);
    if (!drawn) return std::move(drawn).error();
    shadowLights = studentLights == nullptr ? 0u
                   : measured                ? studentLights->visibilityLights
                                             : rasterizer.shadowFactorLights();
    if (!drawn) return std::move(drawn).error();
    {
        gpu::CommandBatch batch(device);
        backward_.dispatch(batch, {count_, 1, 1}, bind);
        weights_.dispatch(batch, {kNeuralWeights, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor);
            cursor["gradient"].setBinding(weightGrad_.rhi());
        });
        adam_.dispatch(batch, {count_ * kNeuralStride, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor);
            // Writes nothing here; kept off the buffer this kernel reads.
            cursor["gradient"].setBinding(weightGrad_.rhi());
            cursor["params"]["count"].setData(count_ * kNeuralStride);
        });
        adam_.dispatch(batch, {kNeuralWeights, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor);
            cursor["gradient"].setBinding(latentGrad_.rhi());
            cursor["value"].setBinding(neural.weights.rhi());
            cursor["moment1"].setBinding(weightM_.rhi());
            cursor["moment2"].setBinding(weightV_.rhi());
            cursor["gradientRead"].setBinding(weightGrad_.rhi());
            cursor["params"]["count"].setData(kNeuralWeights);
            cursor["params"]["stride"].setData(uint32_t{1});
            cursor["params"]["firstOfSecond"].setData(uint32_t{1});
            cursor["params"]["rate"].setData(settings.rateWeights);
            cursor["params"]["sparse"].setData(uint32_t{0});
        });
        ATHENEA_TRY(batch.submit(readLoss));
    }
    if (!readLoss) {
        return std::optional<double>{};
    }
    // Counters come back, as everything that crosses from the device here
    // does: a word a row of the frame.
    std::vector<uint32_t> rows(settings.height);
    ATHENEA_TRY(lossRows_.read(device, 0, rows.size() * sizeof(uint32_t), rows.data()));
    uint64_t total = 0;
    for (const uint32_t row : rows) {
        total += row;
    }
    return std::optional<double>{static_cast<double>(total) / 1024.0 / static_cast<double>(pixels_)};
}

}   // namespace athenea::render
