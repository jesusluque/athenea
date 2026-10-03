// Copyright (c) 2026 jesus luque.
//
// OpenColorIO as a compiler. A processor's GPU shader becomes a generated
// Slang module exporting one function, its LUTs become textures a kernel
// fills from OCIO's values, and its dynamic properties uniforms bound by
// name. Every texel is transformed on the device; the shader text and the
// LUT values are what OCIO computes on the host, the one exception this
// route has (docs/decisions.md, "Colour").
#include "athenea/colour/ColourCompiler.h"

#include <cstdio>
#include <vector>

#include "ColourConfig.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu/Texture.h"

namespace athenea::colour {

struct ColourFunction::State {
    std::string module;        ///< athenea_cs_<hash>
    std::string entry;         ///< atheneaCs_<hash>
    std::string body;          ///< the function's text, without its module line
    std::string description;
    struct Lut {
        std::string  name;       ///< the texture as the shader names it
        std::string  sampler;    ///< and its sampler
        gpu::Texture texture;
        gpu::Sampler filter;
    };
    std::vector<Lut> luts;
#if ATHENEA_HAVE_OCIO
    OCIO::ConstProcessorRcPtr processor;   ///< keeps the dynamic properties the uniforms read alive
    OCIO::GpuShaderDescRcPtr  shader;
#endif
};

bool ocioBuilt() noexcept {
    return ATHENEA_HAVE_OCIO != 0;
}

uint64_t fnv1a(const std::string& text) noexcept {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char c : text) {
        hash = (hash ^ c) * 1099511628211ULL;
    }
    return hash;
}

const std::string& ColourFunction::module() const noexcept {
    static const std::string none;
    return state_ != nullptr ? state_->module : none;
}

const std::string& ColourFunction::entry() const noexcept {
    static const std::string none;
    return state_ != nullptr ? state_->entry : none;
}

const std::string& ColourFunction::body() const noexcept {
    static const std::string none;
    return state_ != nullptr ? state_->body : none;
}

const std::string& ColourFunction::description() const noexcept {
    static const std::string none;
    return state_ != nullptr ? state_->description : none;
}

void ColourFunction::bind(rhi::ShaderCursor cursor) const {
    if (state_ == nullptr) {
        return;
    }
    for (const State::Lut& lut : state_->luts) {
        cursor[lut.name.c_str()].setBinding(lut.texture.rhi());
        cursor[lut.sampler.c_str()].setBinding(lut.filter.rhi());
    }
#if ATHENEA_HAVE_OCIO
    if (!state_->shader) {
        return;
    }
    // Dynamic properties, read as they stand now (exposure, contrast, gamma,
    // a grading's values): whatever the processor was built with, by name.
    for (unsigned i = 0; i < state_->shader->getNumUniforms(); ++i) {
        OCIO::GpuShaderDesc::UniformData data;
        const char* name = state_->shader->getUniform(i, data);
        rhi::ShaderCursor field = cursor[name];
        if (!field.isValid()) {
            continue;
        }
        switch (data.m_type) {
        case OCIO::UNIFORM_DOUBLE: field.setData(static_cast<float>(data.m_getDouble())); break;
        case OCIO::UNIFORM_BOOL: field.setData(int32_t{data.m_getBool() ? 1 : 0}); break;
        case OCIO::UNIFORM_FLOAT3: {
            const OCIO::Float3& v = data.m_getFloat3();
            field.setData(v.data(), sizeof(float) * 3);
            break;
        }
        default: break;   // vectors: refused when the function was compiled
        }
    }
#endif
}

namespace {

struct Names {
    std::string module;
    std::string entry;
    std::string prefix;
};

Names namesFor(uint64_t hash) {
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(hash));
    return {std::string("athenea_cs_") + hex, std::string("atheneaCs_") + hex, std::string("athenea_") + hex + "_"};
}

/// The module loaded under its name, so that a kernel's source can import
/// it: the function public, everything it samples the module's own.
Result<void> loadModule(gpu::ShaderLibrary& library, const ColourFunction::State& state) {
    std::string text = state.body;
    const std::string signature = "float4 " + state.entry + "(";
    if (const size_t at = text.find(signature); at != std::string::npos) {
        text.insert(at, "public ");
    }
    auto program = library.loadSource(state.module, "module " + state.module + ";\n" + text, {});
    if (!program) return std::move(program).error();
    return ok();
}

}   // namespace

#if !ATHENEA_HAVE_OCIO

// WITHOUT OPENCOLORIO: the names the engine always knew, and the one function
// a texture needs that its view cannot give -- sRGB to linear, for a 16-bit or
// float file said to be sRGB.

Result<std::unique_ptr<ColourCompiler>> ColourCompiler::create(gpu::ShaderLibrary& library,
                                                               const std::string& configUri) {
    auto compiler = std::unique_ptr<ColourCompiler>(new ColourCompiler());
    compiler->library_ = &library;
    compiler->device_ = &library.device();
    compiler->uri_ = configUri;
    auto config = std::make_shared<ColourConfig>();
    config->uri = configUri;
    config->cacheId = "table";
    config->label = "the built-in names (no OpenColorIO)";
    compiler->config_ = config;
    compiler->names_.config_ = config;
    compiler->names_.working_ = "lin_rec709";
    compiler->names_.srgb_ = "srgb_texture";
    return compiler;
}

Result<ColourFunction> ColourCompiler::function(const std::string& src, const std::string& dst) {
    const bool decode = src == names_.srgb_ && dst == names_.working_;
    if (!decode && src != dst) {
        return Error::make(ErrorCode::Unsupported, "colour: '{}' to '{}' needs OpenColorIO (scripts/build-ocio.sh)",
                           src, dst);
    }
    const uint64_t hash = fnv1a(config_->cacheId + "\n" + src + "\n" + dst);
    if (const auto found = cache_.find(hash); found != cache_.end()) {
        return found->second;
    }
    auto state = std::make_shared<ColourFunction::State>();
    const Names names = namesFor(hash);
    state->module = names.module;
    state->entry = names.entry;
    state->description = "built-in: " + src + " -> " + dst;
    state->body = decode ? "float4 " + names.entry +
                               "(float4 c) {\n"
                               "    const float3 x = c.rgb;\n"
                               "    return float4(select(x <= 0.04045, x / 12.92, pow((x + 0.055) / 1.055, 2.4)), c.a);\n"
                               "}\n"
                         : "float4 " + names.entry + "(float4 c) { return c; }\n";
    ATHENEA_TRY(loadModule(*library_, *state));
    ColourFunction function;
    function.state_ = std::move(state);
    cache_.emplace(hash, function);
    return function;
}

Result<ColourFunction> ColourCompiler::displayView(const std::string&, const std::string&, const std::string&,
                                                   const std::string&) {
    return Error(ErrorCode::Unsupported, "no OpenColorIO in this build (scripts/build-ocio.sh)");
}

#else

namespace {

/// OCIO's HLSL samples its LUTs with `Sample`, which takes derivatives a
/// compute kernel does not have; the tables are single level, so an explicit
/// level zero samples the same texels.
std::string explicitLevels(const std::string& text) {
    const std::string needle = ".Sample(";
    std::string out;
    size_t at = 0;
    for (;;) {
        const size_t found = text.find(needle, at);
        if (found == std::string::npos) {
            out.append(text, at, std::string::npos);
            break;
        }
        out.append(text, at, found - at);
        size_t end = found + needle.size();
        int depth = 1;
        while (end < text.size() && depth > 0) {
            depth += text[end] == '(' ? 1 : text[end] == ')' ? -1 : 0;
            ++end;
        }
        out += ".SampleLevel(";
        out.append(text, found + needle.size(), end - 1 - (found + needle.size()));
        out += ", 0.0)";
        at = end;
    }
    return out;
}

const char* kFill = R"(
module athenea_colour_fill;

StructuredBuffer<float> values;
RWTexture2D<float4>     target2;
RWTexture3D<float4>     target3;
uniform uint            width;
uniform uint            height;
uniform uint            depth;
uniform uint            channels;   // 1 red, 3 rgb

float4 texel(uint i) {
    return channels == 1 ? float4(values[i], 0.0, 0.0, 1.0)
                         : float4(values[i * 3], values[i * 3 + 1], values[i * 3 + 2], 1.0);
}

// OCIO's rows, as it lays them out, into a texture of four channels.
[shader("compute")]
[numthreads(16, 16, 1)]
void colourFill2(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    target2[tid.xy] = texel(tid.y * width + tid.x);
}

// A 3D LUT's values as OCIO's own OpenGL upload takes them: x fastest, then
// y, then z -- the order its shader text samples in.
[shader("compute")]
[numthreads(8, 8, 8)]
void colourFill3(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height || tid.z >= depth) {
        return;
    }
    target3[tid] = texel((tid.z * height + tid.y) * width + tid.x);
}
)";

rhi::SamplerDesc samplerFor(OCIO::Interpolation interpolation) {
    rhi::SamplerDesc desc;
    const rhi::TextureFilteringMode mode = interpolation == OCIO::INTERP_NEAREST ? rhi::TextureFilteringMode::Point
                                                                                  : rhi::TextureFilteringMode::Linear;
    desc.minFilter = mode;
    desc.magFilter = mode;
    desc.mipFilter = rhi::TextureFilteringMode::Point;
    desc.addressU = rhi::TextureAddressingMode::ClampToEdge;
    desc.addressV = rhi::TextureAddressingMode::ClampToEdge;
    desc.addressW = rhi::TextureAddressingMode::ClampToEdge;
    return desc;
}

/// The processor's shader, its LUTs filled on the device, its module loaded.
Result<std::shared_ptr<ColourFunction::State>> compileProcessor(gpu::ShaderLibrary& library,
                                                                OCIO::ConstProcessorRcPtr processor, uint64_t hash,
                                                                const std::string& label) {
    gpu::Device& device = library.device();
    auto state = std::make_shared<ColourFunction::State>();
    const Names names = namesFor(hash);
    state->module = names.module;
    state->entry = names.entry;
    state->processor = std::move(processor);
    try {
        state->shader = OCIO::GpuShaderDesc::CreateShaderDesc();
        state->shader->setLanguage(OCIO::GPU_LANGUAGE_HLSL_DX11);
        state->shader->setFunctionName(names.entry.c_str());
        state->shader->setResourcePrefix(names.prefix.c_str());
        state->shader->setAllowTexture1D(false);
        state->processor->getDefaultGPUProcessor()->extractGpuShaderInfo(state->shader);
        state->body = explicitLevels(state->shader->getShaderText());
        for (unsigned i = 0; i < state->shader->getNumUniforms(); ++i) {
            OCIO::GpuShaderDesc::UniformData data;
            const char* name = state->shader->getUniform(i, data);
            if (data.m_type != OCIO::UNIFORM_DOUBLE && data.m_type != OCIO::UNIFORM_BOOL &&
                data.m_type != OCIO::UNIFORM_FLOAT3) {
                return Error::make(ErrorCode::Unsupported, "OCIO: uniform '{}' is an array, which is not bound", name);
            }
        }
    } catch (const OCIO::Exception& e) {
        // OCIO reports by throwing; the engine does not, so this is where
        // its exceptions stop.
        return Error::make(ErrorCode::InvalidArgument, "OCIO: {}", e.what());
    }
    state->description = std::string("OCIO ") + OCIO::GetVersion() + ": " + label;
    ATHENEA_TRY(loadModule(library, *state));

    // The LUTs: OCIO's values uploaded as they are, laid into textures by a kernel.
    const unsigned flat = state->shader->getNumTextures();
    const unsigned cubes = state->shader->getNum3DTextures();
    if (flat + cubes == 0) {
        return state;
    }
    auto fillProgram = library.loadSource("athenea_colour_fill", kFill, {"colourFill2", "colourFill3"});
    if (!fillProgram) return std::move(fillProgram).error();
    auto fill2 = gpu::ComputeKernel::create(library, "athenea_colour_fill", "colourFill2");
    if (!fill2) return std::move(fill2).error();
    auto fill3 = gpu::ComputeKernel::create(library, "athenea_colour_fill", "colourFill3");
    if (!fill3) return std::move(fill3).error();
    gpu::CommandBatch batch(device);
    std::vector<gpu::Buffer> uploads;   // alive until the batch has run
    const auto lutOf = [&](const char* texture, const char* sampler, uint32_t w, uint32_t h, uint32_t d,
                           uint32_t channels, OCIO::Interpolation interpolation, const float* values) -> Result<void> {
        ColourFunction::State::Lut lut;
        lut.name = texture;
        lut.sampler = sampler;
        gpu::TextureDesc desc;
        desc.type = d > 1 ? rhi::TextureType::Texture3D : rhi::TextureType::Texture2D;
        desc.width = w;
        desc.height = h;
        desc.depth = d;
        desc.format = rhi::Format::RGBA32Float;
        desc.usage = rhi::TextureUsage::ShaderResource | rhi::TextureUsage::UnorderedAccess;
        desc.label = texture;
        auto made = gpu::Texture::create(device, desc);
        if (!made) return std::move(made).error();
        lut.texture = std::move(*made);
        auto filter = gpu::Sampler::create(device, samplerFor(interpolation));
        if (!filter) return std::move(filter).error();
        lut.filter = std::move(*filter);
        gpu::BufferDesc upload;
        upload.bytes = uint64_t{w} * h * d * channels * sizeof(float);
        upload.elementBytes = sizeof(float);
        upload.label = "colour.lut";
        auto buffer = gpu::Buffer::create(device, upload, values);
        if (!buffer) return std::move(buffer).error();
        auto level = lut.texture.view(0);
        if (!level) return std::move(level).error();
        const gpu::ComputeKernel& fill = d > 1 ? *fill3 : *fill2;
        fill.dispatch(batch, {w, h, d}, [&](rhi::ShaderCursor cursor) {
            cursor["values"].setBinding(buffer->rhi());
            cursor[d > 1 ? "target3" : "target2"].setBinding((*level).get());
            cursor["width"].setData(w);
            cursor["height"].setData(h);
            cursor["depth"].setData(d);
            cursor["channels"].setData(channels);
        });
        uploads.push_back(std::move(*buffer));
        state->luts.push_back(std::move(lut));
        return ok();
    };
    for (unsigned i = 0; i < flat; ++i) {
        const char* texture = nullptr;
        const char* sampler = nullptr;
        unsigned w = 0, h = 0;
        OCIO::GpuShaderDesc::TextureType channel = OCIO::GpuShaderDesc::TEXTURE_RGB_CHANNEL;
        OCIO::GpuShaderDesc::TextureDimensions dimensions = OCIO::GpuShaderDesc::TEXTURE_2D;
        OCIO::Interpolation interpolation = OCIO::INTERP_LINEAR;
        state->shader->getTexture(i, texture, sampler, w, h, channel, dimensions, interpolation);
        const float* values = nullptr;
        state->shader->getTextureValues(i, values);
        ATHENEA_TRY(lutOf(texture, sampler, w, h, 1, channel == OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL ? 1u : 3u,
                          interpolation, values));
    }
    for (unsigned i = 0; i < cubes; ++i) {
        const char* texture = nullptr;
        const char* sampler = nullptr;
        unsigned edge = 0;
        OCIO::Interpolation interpolation = OCIO::INTERP_LINEAR;
        state->shader->get3DTexture(i, texture, sampler, edge, interpolation);
        const float* values = nullptr;
        state->shader->get3DTextureValues(i, values);
        ATHENEA_TRY(lutOf(texture, sampler, edge, edge, edge, 3, interpolation, values));
    }
    ATHENEA_TRY(batch.submit(true));
    return state;
}

/// The config in use's name for `wanted`, else the studio config's.
std::string nameIn(const OCIO::ConstConfigRcPtr& config, const char* wanted) {
    if (config) {
        if (OCIO::ConstColorSpaceRcPtr space = config->getColorSpace(wanted)) {
            return space->getName();
        }
    }
    return {};
}

}   // namespace

Result<std::unique_ptr<ColourCompiler>> ColourCompiler::create(gpu::ShaderLibrary& library,
                                                               const std::string& configUri) {
    auto compiler = std::unique_ptr<ColourCompiler>(new ColourCompiler());
    compiler->library_ = &library;
    compiler->device_ = &library.device();
    compiler->uri_ = configUri;
    auto config = std::make_shared<ColourConfig>();
    config->uri = configUri;
    try {
        config->config = OCIO::Config::CreateFromFile(configUri.c_str());
        config->studio = configUri == kStudioConfig ? config->config : OCIO::Config::CreateFromFile(kStudioConfig);
        config->cacheId = config->config->getCacheID();
        config->label = config->config->getName();
    } catch (const OCIO::Exception& e) {
        return Error::make(ErrorCode::InvalidArgument, "OCIO: {}", e.what());
    }
    compiler->config_ = config;
    compiler->names_.config_ = config;
    // The working space and sRGB as the config in use names them. A config
    // without them leaves them to the studio config's names, which the
    // functions then reach across configs.
    compiler->names_.working_ = nameIn(config->config, kWorkingSpace);
    if (compiler->names_.working_.empty()) {
        compiler->names_.working_ = nameIn(config->studio, kWorkingSpace);
    }
    compiler->names_.srgb_ = nameIn(config->config, "srgb_texture");
    if (compiler->names_.srgb_.empty()) {
        compiler->names_.srgb_ = nameIn(config->studio, "srgb_texture");
    }
    return compiler;
}

Result<ColourFunction> ColourCompiler::function(const std::string& src, const std::string& dst) {
    const uint64_t hash = fnv1a(config_->cacheId + "\n" + src + "\n" + dst);
    if (const auto found = cache_.find(hash); found != cache_.end()) {
        return found->second;
    }
    OCIO::ConstProcessorRcPtr processor;
    try {
        // Each name in the config that knows it: the config in use first; one
        // only the studio config knows is reached through the two configs'
        // interchange spaces.
        const auto holder = [&](const std::string& name) {
            return config_->config->getColorSpace(name.c_str()) != nullptr ? config_->config : config_->studio;
        };
        const OCIO::ConstConfigRcPtr from = holder(src);
        const OCIO::ConstConfigRcPtr to = holder(dst);
        processor = from == to ? from->getProcessor(src.c_str(), dst.c_str())
                               : OCIO::Config::GetProcessorFromConfigs(from, src.c_str(), to, dst.c_str());
    } catch (const OCIO::Exception& e) {
        return Error::make(ErrorCode::InvalidArgument, "OCIO: {}", e.what());
    }
    auto state = compileProcessor(*library_, std::move(processor), hash, config_->label + " / " + src + " -> " + dst);
    if (!state) return std::move(state).error();
    ColourFunction function;
    function.state_ = std::move(*state);
    cache_.emplace(hash, function);
    return function;
}

Result<ColourFunction> ColourCompiler::displayView(const std::string& src, const std::string& displayName,
                                                   const std::string& viewName, const std::string& look) {
    OCIO::ConstProcessorRcPtr processor;
    std::string display;
    std::string view;
    try {
        const OCIO::ConstConfigRcPtr& config = config_->config;
        display = displayName.empty() ? config->getDefaultDisplay() : displayName;
        view = viewName.empty() ? config->getDefaultView(display.c_str()) : viewName;
        const uint64_t hash =
            fnv1a(config_->cacheId + "\ndisplay\n" + src + "\n" + display + "\n" + view + "\n" + look);
        if (const auto found = cache_.find(hash); found != cache_.end()) {
            return found->second;
        }
        OCIO::DisplayViewTransformRcPtr transform = OCIO::DisplayViewTransform::Create();
        transform->setSrc(src.c_str());
        transform->setDisplay(display.c_str());
        transform->setView(view.c_str());
        if (!look.empty()) {
            OCIO::LegacyViewingPipelineRcPtr pipeline = OCIO::LegacyViewingPipeline::Create();
            pipeline->setDisplayViewTransform(transform);
            pipeline->setLooksOverrideEnabled(true);
            pipeline->setLooksOverride(look.c_str());
            processor = pipeline->getProcessor(config);
        } else {
            processor = config->getProcessor(OCIO::ConstTransformRcPtr(transform));
        }
        auto state = compileProcessor(*library_, std::move(processor), hash,
                                      config_->label + " / " + display + " / " + view);
        if (!state) return std::move(state).error();
        ColourFunction function;
        function.state_ = std::move(*state);
        cache_.emplace(hash, function);
        return function;
    } catch (const OCIO::Exception& e) {
        return Error::make(ErrorCode::InvalidArgument, "OCIO: {}", e.what());
    }
}

#endif

}   // namespace athenea::colour
