// Copyright (c) 2026 jesus luque.
#include "athenea/material/TextureStore.h"

#include <pxr/usd/ar/resolver.h>

#include <algorithm>
#include <cctype>
#include <cstring>

#include <pxr/imaging/hio/image.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usdShade/udimUtils.h>

#include <cstdio>

#include "athenea/core/Log.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace athenea::material {

namespace {

constexpr uint32_t kTextureSlots = 1024;   // texture_table.slang's kTextureSlots
constexpr uint32_t kSamplerSlots = 16;

/// shaders/athenea/material/texture_table.slang's TextureRecord.
struct TextureRecord {
    uint32_t slot = 0;
    uint32_t udimBase = 0;
    uint32_t flags = 0;
    uint32_t pad0 = 0;
    float    width = 0.0F;
    float    height = 0.0F;
    float    pad1 = 0.0F;
    float    pad2 = 0.0F;
};
static_assert(sizeof(TextureRecord) == 32);

struct Layout {
    uint32_t channels = 0;
    uint32_t component = 0;   // texture_decode.slang: 0 unorm8, 1 unorm16, 2 float16, 3 float32
    uint32_t componentBytes = 1;
    bool     srgb = false;
};

bool layoutOf(HioFormat format, Layout& out) {
    const auto set = [&](uint32_t channels, uint32_t component, uint32_t bytes, bool srgb) {
        out = {channels, component, bytes, srgb};
        return true;
    };
    switch (format) {
    case HioFormatUNorm8: return set(1, 0, 1, false);
    case HioFormatUNorm8Vec2: return set(2, 0, 1, false);
    case HioFormatUNorm8Vec3: return set(3, 0, 1, false);
    case HioFormatUNorm8Vec4: return set(4, 0, 1, false);
    case HioFormatUNorm8srgb: return set(1, 0, 1, true);
    case HioFormatUNorm8Vec2srgb: return set(2, 0, 1, true);
    case HioFormatUNorm8Vec3srgb: return set(3, 0, 1, true);
    case HioFormatUNorm8Vec4srgb: return set(4, 0, 1, true);
    case HioFormatUInt16: return set(1, 1, 2, false);
    case HioFormatUInt16Vec2: return set(2, 1, 2, false);
    case HioFormatUInt16Vec3: return set(3, 1, 2, false);
    case HioFormatUInt16Vec4: return set(4, 1, 2, false);
    case HioFormatFloat16: return set(1, 2, 2, false);
    case HioFormatFloat16Vec2: return set(2, 2, 2, false);
    case HioFormatFloat16Vec3: return set(3, 2, 2, false);
    case HioFormatFloat16Vec4: return set(4, 2, 2, false);
    case HioFormatFloat32: return set(1, 3, 4, false);
    case HioFormatFloat32Vec2: return set(2, 3, 4, false);
    case HioFormatFloat32Vec3: return set(3, 3, 4, false);
    case HioFormatFloat32Vec4: return set(4, 3, 4, false);
    default: return false;
    }
}

rhi::TextureAddressingMode addressOf(Wrap wrap) {
    switch (wrap) {
    case Wrap::Clamp: return rhi::TextureAddressingMode::ClampToEdge;
    case Wrap::Mirror: return rhi::TextureAddressingMode::MirrorRepeat;
    case Wrap::Black: return rhi::TextureAddressingMode::ClampToBorder;
    default: return rhi::TextureAddressingMode::Wrap;
    }
}

}   // namespace

Result<std::unique_ptr<TextureStore>> TextureStore::create(gpu::ShaderLibrary& library) {
    auto store = std::unique_ptr<TextureStore>(new TextureStore());
    store->device_ = &library.device();
    store->library_ = &library;
    auto decode = gpu::ComputeKernel::create(library, "athenea/material/texture_decode", "textureDecode");
    if (!decode) return std::move(decode).error();
    store->decode_ = std::move(*decode);
    auto mips = gpu::MipGenerator::create(library);
    if (!mips) return std::move(mips).error();
    store->mips_ = std::make_unique<gpu::MipGenerator>(std::move(*mips));
    store->sampler(Wrap::Repeat, Wrap::Repeat, Filter::Linear);   // slot 0, for anything unset
    ATHENEA_TRY(store->writeRecords());
    return store;
}

bool TextureStore::isExternal(const std::string& path) { return path.rfind("aofx://", 0) == 0; }

uint32_t TextureStore::request(const std::string& path, const std::string& given, bool latLong) {
    // One entry for an external texture, whatever a material says of it: the
    // host's working space arrives as it is.
    const std::string space = isExternal(path) ? std::string("raw") : given;
    const auto key = std::make_pair(path, space);
    if (const auto found = ids_.find(key); found != ids_.end()) {
        if (latLong && !entries_[found->second].info.latLong) {
            entries_[found->second].info.latLong = true;
            entries_[found->second].pending = true;   // its mips again, as a lat-long's
        }
        return found->second;
    }
    const uint32_t id = static_cast<uint32_t>(entries_.size());
    Entry entry;
    entry.info.path = path;
    entry.info.space = space;
    entry.info.udim = UsdShadeUdimUtils::IsUdimIdentifier(path);
    entry.info.latLong = latLong;
    entries_.push_back(std::move(entry));
    ids_.emplace(key, id);
    return id;
}

uint32_t TextureStore::sampler(Wrap s, Wrap t, Filter filter) {
    const auto key = std::make_tuple(s, t, filter);
    if (const auto found = samplerIds_.find(key); found != samplerIds_.end()) {
        return found->second;
    }
    if (samplers_.size() >= kSamplerSlots) {
        log::warn("materials: more than {} sampler modes; the rest share the first", kSamplerSlots);
        return 0;
    }
    rhi::SamplerDesc desc;
    desc.addressU = addressOf(s);
    desc.addressV = addressOf(t);
    const rhi::TextureFilteringMode mode =
        filter == Filter::Nearest ? rhi::TextureFilteringMode::Point : rhi::TextureFilteringMode::Linear;
    desc.minFilter = mode;
    desc.magFilter = mode;
    desc.mipFilter = rhi::TextureFilteringMode::Linear;
    desc.maxAnisotropy = 8;
    auto made = gpu::Sampler::create(*device_, desc);
    if (!made) {
        log::warn("materials: {}", made.error().toString());
        return 0;
    }
    samplers_.push_back(std::move(*made));
    const uint32_t id = static_cast<uint32_t>(samplers_.size() - 1);
    samplerIds_.emplace(key, id);
    return id;
}

/// A name with case, spaces, punctuation and extension taken out.
std::string normalisedName(const std::string& name) {
    const size_t dot = name.find_last_of('.');
    std::string key;
    for (size_t k = 0; k < (dot == std::string::npos ? name.size() : dot); ++k) {
        const unsigned char c = static_cast<unsigned char>(name[k]);
        if (std::isalnum(c) != 0) {
            key += static_cast<char>(std::tolower(c));
        }
    }
    return key;
}

std::filesystem::path TextureStore::besideByName(const std::string& name) const {
    if (!besideBuilt_) {
        besideBuilt_ = true;
        std::error_code failed;
        auto entry = std::filesystem::recursive_directory_iterator(
            search_, std::filesystem::directory_options::skip_permission_denied, failed);
        const auto end = std::filesystem::recursive_directory_iterator();
        for (; !failed && entry != end; entry.increment(failed)) {
            // Two levels: the stage's folder and its `textures` beside it,
            // not somebody's whole home directory.
            if (entry.depth() >= 2) {
                entry.disable_recursion_pending();
            }
            if (entry->is_regular_file(failed)) {
                beside_.emplace(normalisedName(entry->path().filename().string()), entry->path());
            }
        }
    }
    const auto found = beside_.find(normalisedName(name));
    return found != beside_.end() ? found->second : std::filesystem::path();
}

Result<const TextureStore::Decoder*> TextureStore::decoderFor(const std::string& space) {
    if (const auto found = decoders_.find(space); found != decoders_.end()) {
        return &found->second;
    }
    auto function = colour_->function(space, colour_->names().working());
    if (!function) return std::move(function).error();
    // texture_decode's entry with the function between reading a texel and
    // storing it; alpha is coverage, not colour, and is left as it is.
    const std::string source = "import athenea.material.texture_decode;\nimport " + function->module() +
                               ";\n"
                               "[shader(\"compute\")]\n"
                               "[numthreads(16, 16, 1)]\n"
                               "void textureDecodeColour(uint3 tid: SV_DispatchThreadID) {\n"
                               "    float4 texel;\n"
                               "    if (!decodeTexel(tid, texel)) {\n"
                               "        return;\n"
                               "    }\n"
                               "    storeTexel(tid, float4(" + function->entry() + "(texel).rgb, texel.a));\n"
                               "}\n";
    char name[48];
    std::snprintf(name, sizeof(name), "athenea_texdec_%016llx",
                  static_cast<unsigned long long>(colour::fnv1a(source)));
    auto program = library_->loadSource(name, "module " + std::string(name) + ";\n" + source, {"textureDecodeColour"});
    if (!program) return std::move(program).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "textureDecodeColour");
    if (!kernel) return std::move(kernel).error();
    Decoder decoder;
    decoder.kernel = std::move(*kernel);
    decoder.function = std::move(*function);
    return &decoders_.emplace(space, std::move(decoder)).first->second;
}

Result<uint32_t> TextureStore::loadFile(const std::string& path, const std::string& space, TextureInfo& info) {
    if (slots_.size() >= kTextureSlots) {
        return Error::make(ErrorCode::OutOfMemory, "more than {} texture files", kTextureSlots);
    }
    // Hio is asked for the file as it is: what its pixels mean is decided
    // here, by name, and done on the device.
    const HioImage::SourceColorSpace source = HioImage::Auto;
    // WHAT A PATH MIGHT MEAN, IN ORDER.
    //
    // As given first: an absolute path, a package path (Hio reads inside a
    // usdz), or one the asset resolver already answered. Then the resolver's
    // own answer, for a search-path asset. Then the stage's folder, which is
    // what a material means by `./textures/paint.jpg` and what no material
    // network carries. Each one is tried once and the first that opens wins.
    std::vector<std::string> tries{path};
    if (const std::string resolved = ArGetResolver().Resolve(path).GetPathString(); !resolved.empty()) {
        tries.push_back(resolved);
    }
    if (!search_.empty() && !std::filesystem::path(path).is_absolute()) {
        std::string relative = path;
        while (relative.rfind("./", 0) == 0) {
            relative.erase(0, 2);
        }
        tries.push_back((search_ / relative).string());
        // And beside the stage, by name alone: an asset moved next to its
        // textures rather than above them still finds them.
        tries.push_back((search_ / std::filesystem::path(relative).filename()).string());
        // LAST: THE NAME WITH THE PACKAGING TAKEN OUT.
        //
        // A file's spelling changes on the way through a packager -- spaces
        // dropped, case changed, .jpg written as .jpeg -- and then a network
        // that names `carpet_BaseColor.jpeg` sits next to a file called
        // `carpet_Base Color.jpg` and neither renderer nor artist can see why
        // nothing loads. Matching on what such a rename cannot change finds
        // it. Only after every honest path has failed, and it says so.
        if (const std::filesystem::path found = besideByName(std::filesystem::path(relative).filename().string());
            !found.empty()) {
            tries.push_back(found.string());
        }
    }
    HioImageSharedPtr image;
    std::string opened;
    for (const std::string& candidate : tries) {
        image = HioImage::OpenForReading(candidate, 0, 0, source, /*suppressErrors=*/true);
        if (image) {
            opened = candidate;
            break;
        }
    }
    if (!image) {
        return Error::make(ErrorCode::IoFailure, "cannot read image '{}' (tried {} path(s))", path, tries.size());
    }
    if (opened != path) {
        log::debug("materials: '{}' read as '{}'", path, opened);
    }
    Layout layout;
    if (!layoutOf(image->GetFormat(), layout)) {
        return Error::make(ErrorCode::Unsupported, "image '{}': pixel format {} not read", path,
                           static_cast<int>(image->GetFormat()));
    }
    // WHAT THE NAME MEANS, AND WHAT THE FILE SAYS WHEN THE NAME LEAVES IT.
    //
    // Raw and the working space are read as they are. 8-bit sRGB stays 8-bit
    // behind an sRGB view, which the hardware decodes. Anything else -- a
    // float file said to be sRGB, ACEScg, a camera's log -- is brought into
    // the working space by the decode kernel, through the function compiled
    // for its colour space; a name nothing knows is read as the file says.
    const bool eightBit = layout.component == 0;
    const colour::FileEncoding encoding =
        eightBit && layout.srgb ? colour::FileEncoding::Srgb8 : colour::FileEncoding::Linear;
    colour::ResolvedSpace resolved = colour_->names().resolve(space, encoding);
    if (resolved.kind == colour::SpaceKind::Unknown) {
        info.error = "colour space '" + space + "' is not known; read as the file says";
        resolved = colour_->names().resolve({}, encoding);
    }
    const bool srgbView = resolved.kind == colour::SpaceKind::Srgb && eightBit && srgbFastPath_;
    const bool convert = resolved.kind == colour::SpaceKind::Other ||
                         (resolved.kind == colour::SpaceKind::Srgb && !srgbView);
    const Decoder* decoder = nullptr;
    if (convert) {
        auto made = decoderFor(resolved.name);
        if (!made) return std::move(made).error();
        decoder = *made;
    }
    const uint32_t w = static_cast<uint32_t>(image->GetWidth());
    const uint32_t h = static_cast<uint32_t>(image->GetHeight());
    const size_t bytes = size_t{w} * h * layout.channels * layout.componentBytes;
    std::vector<uint32_t> words((bytes + 3) / 4, 0);
    HioImage::StorageSpec spec;
    spec.width = static_cast<int>(w);
    spec.height = static_cast<int>(h);
    spec.depth = 1;
    spec.format = image->GetFormat();
    spec.flipped = false;
    spec.data = words.data();
    if (!image->Read(spec)) {
        return Error::make(ErrorCode::IoFailure, "cannot decode image '{}'", path);
    }
    info.width = w;
    info.height = h;

    // 8-bit stays 8-bit when it is read as it is; what a function brings into
    // the working space is light, in half floats; floats keep their precision.
    gpu::TextureDesc desc;
    desc.width = w;
    desc.height = h;
    desc.mipCount = 0;
    desc.format = eightBit && !convert ? rhi::Format::RGBA8Unorm
                  : layout.component == 3 ? rhi::Format::RGBA32Float
                                          : rhi::Format::RGBA16Float;
    desc.usage = rhi::TextureUsage::ShaderResource | rhi::TextureUsage::UnorderedAccess |
                 rhi::TextureUsage::CopySource;
    desc.label = path;
    auto texture = gpu::Texture::create(*device_, desc);
    if (!texture) return std::move(texture).error();
    gpu::BufferDesc upload;
    upload.bytes = std::max<uint64_t>(words.size(), 1) * 4;
    upload.elementBytes = 4;
    upload.label = "texture.bytes";
    auto buffer = gpu::Buffer::create(*device_, upload, words.data());
    if (!buffer) return std::move(buffer).error();
    auto level0 = texture->view(0);
    if (!level0) return std::move(level0).error();
    // Where a store into the texture does not convert (CUDA), the kernel packs
    // the texels into a buffer and the buffer is copied into level 0.
    const uint32_t packedKind = device_->caps().convertingStores ? 0u : gpu::packedKindOf(desc.format);
    if (!device_->caps().convertingStores && packedKind == 0) {
        return Error::make(ErrorCode::Unsupported, "'{}': this device's stores do not convert and no kernel packs "
                                                   "its format", path);
    }
    gpu::Buffer packed;
    if (packedKind != 0) {
        gpu::BufferDesc packedDesc;
        packedDesc.bytes = uint64_t{w} * h * texture->texelBytes();
        packedDesc.elementBytes = 4;
        packedDesc.label = "texture.packed";
        auto made = gpu::Buffer::create(*device_, packedDesc);
        if (!made) return std::move(made).error();
        packed = std::move(*made);
    }
    gpu::CommandBatch batch(*device_);
    const gpu::ComputeKernel& kernel = decoder != nullptr ? decoder->kernel : decode_;
    kernel.dispatch(batch, {w, h, 1}, [&](rhi::ShaderCursor cursor) {
        if (decoder != nullptr) {
            decoder->function.bind(cursor);
        }
        cursor["bytes"].setBinding(buffer->rhi());
        cursor["level"].setBinding((*level0).get());
        cursor["packed"].setBinding(packedKind != 0 ? packed.rhi() : nullptr);
        cursor["params"]["packedKind"].setData(packedKind);
        rhi::ShaderCursor p = cursor["params"];
        p["width"].setData(w);
        p["height"].setData(h);
        p["channels"].setData(layout.channels);
        p["component"].setData(layout.component);
        p["rowPixels"].setData(uint32_t{0});
        p["bottomFirst"].setData(uint32_t{0});
    });
    if (packedKind != 0) {
        const uint32_t rowPitch = w * texture->texelBytes();
        batch.encoder()->copyBufferToTexture(texture->rhi(), 0, 0, {0, 0, 0}, packed.rhi(), 0,
                                             uint64_t{rowPitch} * h, rowPitch, {w, h, 1});
        batch.markDirty();
    }
    ATHENEA_TRY(mips_->generate(batch, *texture, srgbView, info.latLong));
    ATHENEA_TRY(batch.submit(true));
    info.decode = srgbView                                ? "srgb view"
                  : decoder != nullptr                    ? decoder->function.description()
                  : resolved.kind == colour::SpaceKind::Raw ? "raw"
                                                          : "working";

    Slot slot;
    if (srgbView) {
        rhi::TextureViewDesc view;
        view.format = rhi::Format::RGBA8UnormSrgb;
        if (SLANG_FAILED(device_->rhi()->createTextureView(texture->rhi(), view, slot.view.writeRef()))) {
            return Error::make(ErrorCode::DeviceFailure, "no sRGB view of '{}'", path);
        }
    } else {
        rhi::TextureViewDesc view;
        if (SLANG_FAILED(device_->rhi()->createTextureView(texture->rhi(), view, slot.view.writeRef()))) {
            return Error::make(ErrorCode::DeviceFailure, "no view of '{}'", path);
        }
    }
    slot.texture = std::move(*texture);
    slots_.push_back(std::move(slot));
    return static_cast<uint32_t>(slots_.size() - 1);
}

Result<size_t> TextureStore::commit() {
    size_t loaded = 0;
    bool changed = false;
    for (Entry& entry : entries_) {
        if (!entry.pending) {
            continue;
        }
        entry.pending = false;
        changed = true;
        if (isExternal(entry.info.path)) {
            continue;   // filled by updateExternal, never read from disk
        }
        if (colour_ == nullptr) {
            // The names, and the functions they compile to, from the studio
            // config: made once, when the first file is read.
            auto made = colour::ColourCompiler::create(*library_, colour::kStudioConfig);
            if (!made) return std::move(made).error();
            colour_ = std::move(*made);
        }
        if (entry.info.udim) {
            entry.tiles.assign(100, 0);
            for (const auto& [tilePath, tile] : UsdShadeUdimUtils::ResolveUdimTilePaths(entry.info.path, SdfLayerHandle())) {
                const int number = std::atoi(tile.c_str());
                if (number < 1001 || number > 1100) {
                    continue;
                }
                auto slot = loadFile(tilePath, entry.info.space, entry.info);
                if (!slot) {
                    log::warn("materials: {}", slot.error().toString());
                    continue;
                }
                entry.tiles[static_cast<size_t>(number - 1001)] = *slot + 1;
                entry.info.loaded = true;
                ++loaded;
            }
            if (!entry.info.loaded) {
                entry.info.error = "no UDIM tiles found";
                log::warn("materials: '{}': no UDIM tiles found", entry.info.path);
            }
            continue;
        }
        auto slot = loadFile(entry.info.path, entry.info.space, entry.info);
        if (!slot) {
            entry.info.error = slot.error().toString();
            log::warn("materials: {}", entry.info.error);
            continue;
        }
        entry.slot = *slot;
        entry.info.loaded = true;
        ++loaded;
    }
    if (changed) {
        ATHENEA_TRY(writeRecords());
    }
    return loaded;
}

Result<void> TextureStore::updateExternal(const std::string& name, const gpu::Buffer& rgba, uint32_t width,
                                          uint32_t height, uint32_t rowPixels) {
    if (!isExternal(name)) {
        return Error::make(ErrorCode::InvalidArgument, "'{}' is not an external texture (aofx://...)", name);
    }
    if (width == 0 || height == 0 || rowPixels < width ||
        rgba.bytes() < (uint64_t{rowPixels} * (height - 1) + width) * 16) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': a {}x{} plane of {} a row does not fit its buffer",
                           name, width, height, rowPixels);
    }
    const uint32_t id = request(name, "raw");
    Entry& entry = entries_[id];
    const bool reshape = !entry.info.loaded || slots_[entry.slot].texture.width() != width ||
                         slots_[entry.slot].texture.height() != height;
    if (reshape) {
        if (!entry.info.loaded && slots_.size() >= kTextureSlots) {
            return Error::make(ErrorCode::OutOfMemory, "more than {} textures", kTextureSlots);
        }
        gpu::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.mipCount = 0;
        desc.format = rhi::Format::RGBA16Float;
        desc.usage = rhi::TextureUsage::ShaderResource | rhi::TextureUsage::UnorderedAccess |
                     rhi::TextureUsage::CopySource | rhi::TextureUsage::CopyDestination;
        desc.label = name;
        auto texture = gpu::Texture::create(*device_, desc);
        if (!texture) return std::move(texture).error();
        Slot slot;
        rhi::TextureViewDesc view;
        if (SLANG_FAILED(device_->rhi()->createTextureView(texture->rhi(), view, slot.view.writeRef()))) {
            return Error::make(ErrorCode::DeviceFailure, "no view of '{}'", name);
        }
        slot.texture = std::move(*texture);
        if (entry.info.loaded) {
            slots_[entry.slot] = std::move(slot);
        } else {
            slots_.push_back(std::move(slot));
            entry.slot = static_cast<uint32_t>(slots_.size() - 1);
        }
        entry.info.width = width;
        entry.info.height = height;
        entry.info.loaded = true;
        entry.info.error.clear();
        entry.pending = false;
        ATHENEA_TRY(writeRecords());
    }
    gpu::Texture& texture = slots_[entry.slot].texture;
    auto level0 = texture.view(0);
    if (!level0) return std::move(level0).error();
    const uint32_t packedKind = device_->caps().convertingStores ? 0u : gpu::packedKindOf(texture.desc().format);
    gpu::Buffer packed;
    if (packedKind != 0) {
        gpu::BufferDesc packedDesc;
        packedDesc.bytes = uint64_t{width} * height * texture.texelBytes();
        packedDesc.elementBytes = 4;
        packedDesc.label = "texture.packed";
        auto made = gpu::Buffer::create(*device_, packedDesc);
        if (!made) return std::move(made).error();
        packed = std::move(*made);
    }
    gpu::CommandBatch batch(*device_);
    decode_.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["bytes"].setBinding(rgba.rhi());
        cursor["level"].setBinding((*level0).get());
        cursor["packed"].setBinding(packedKind != 0 ? packed.rhi() : nullptr);
        rhi::ShaderCursor p = cursor["params"];
        p["packedKind"].setData(packedKind);
        p["width"].setData(width);
        p["height"].setData(height);
        p["channels"].setData(uint32_t{4});
        p["component"].setData(uint32_t{3});
        p["rowPixels"].setData(rowPixels);
        p["bottomFirst"].setData(uint32_t{1});
    });
    if (packedKind != 0) {
        const uint32_t rowPitch = width * texture.texelBytes();
        batch.encoder()->copyBufferToTexture(texture.rhi(), 0, 0, {0, 0, 0}, packed.rhi(), 0,
                                             uint64_t{rowPitch} * height, rowPitch, {width, height, 1});
        batch.markDirty();
    }
    ATHENEA_TRY(mips_->generate(batch, texture, false));
    // The packed buffer is this batch's only; waiting for it is the price of
    // a device whose stores do not convert.
    return batch.submit(packedKind != 0);
}

Result<void> TextureStore::writeRecords() {
    std::vector<TextureRecord> records(std::max<size_t>(entries_.size(), 1));
    std::vector<uint32_t> cells;
    for (size_t k = 0; k < entries_.size(); ++k) {
        Entry& entry = entries_[k];
        TextureRecord& record = records[k];
        record.width = static_cast<float>(entry.info.width);
        record.height = static_cast<float>(entry.info.height);
        if (!entry.info.loaded) {
            continue;
        }
        record.flags = 1;
        if (entry.info.udim) {
            record.flags |= 2;
            entry.udimBase = static_cast<uint32_t>(cells.size());
            record.udimBase = entry.udimBase;
            cells.insert(cells.end(), entry.tiles.begin(), entry.tiles.end());
        } else {
            record.slot = entry.slot;
        }
    }
    cells.resize(std::max<size_t>(cells.size(), 1), 0);
    auto made = gpu::Buffer::fromSpan<TextureRecord>(*device_, records, "textures.records");
    if (!made) return std::move(made).error();
    records_ = std::move(*made);
    auto tiles = gpu::Buffer::fromSpan<uint32_t>(*device_, cells, "textures.udim");
    if (!tiles) return std::move(tiles).error();
    udim_ = std::move(*tiles);
    return ok();
}

void TextureStore::bind(rhi::ShaderCursor table) const {
    for (size_t k = 0; k < slots_.size(); ++k) {
        table["textures"][static_cast<uint32_t>(k)].setBinding(slots_[k].view.get());
    }
    for (size_t k = 0; k < kSamplerSlots; ++k) {
        table["samplers"][static_cast<uint32_t>(k)].setBinding(samplers_[k < samplers_.size() ? k : 0].rhi());
    }
    table["records"].setBinding(records_.rhi());
    table["udim"].setBinding(udim_.rhi());
}

}   // namespace athenea::material
