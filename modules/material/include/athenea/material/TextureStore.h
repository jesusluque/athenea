// Copyright (c) 2026 jesus luque.
//
// The textures materials sample, on the device. A file is decoded by Hio on
// the host (decompression, as the rules allow) and its bytes go up whole; a
// kernel expands its channels and flips it to v-up (texture_decode.slang), and
// its mip chain is made on the device -- averaged as light for sRGB colour,
// which is sampled through an sRGB view. UDIM sets resolve to a slot per tile.
// Materials reach all of it through one table (athenea/material/texture_table).
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Texture.h"
#include "athenea/gpu/algo/Mips.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::material {

/// How a file's components are to be read: MaterialX's and UsdUVTexture's
/// `colorSpace`/`sourceColorSpace`.
enum class ColourSpace : uint8_t {
    Auto,   ///< the file says: 8-bit images are sRGB unless tagged otherwise, float ones linear
    Raw,    ///< data (normals, roughness): no decoding
    Srgb,   ///< sRGB-encoded colour
};

enum class Wrap : uint8_t { Repeat, Clamp, Mirror, Black };
enum class Filter : uint8_t { Linear, Nearest };

struct TextureInfo {
    std::string path;
    ColourSpace space = ColourSpace::Auto;
    bool        loaded = false;
    bool        udim = false;
    uint32_t    width = 0;    ///< the first tile's, for a UDIM set
    uint32_t    height = 0;
    std::string error;        ///< why it did not load
};

class TextureStore {
public:
    [[nodiscard]] static Result<std::unique_ptr<TextureStore>> create(gpu::ShaderLibrary& library);

    /// The id of `path` -- a file path, one holding "<UDIM>", or a path
    /// inside a package (`shot.usdz[textures/paint.jpg]`) -- read as `space`.
    /// Loaded at the next commit; until then, and if it cannot be, samples
    /// report it missing and materials use their defaults.
    uint32_t request(const std::string& path, ColourSpace space = ColourSpace::Auto);

    /// WHERE A RELATIVE PATH IS RELATIVE TO.
    ///
    /// A material that names `./textures/paint.jpg` means the folder its
    /// layer sits in, and nothing in a Hydra material network says which
    /// folder that is: UsdPreviewSurface's `file` arrives resolved, a
    /// MaterialX network's does not, and the same asset therefore loaded
    /// half its textures and lost the other half (the Mustang's carpet).
    /// The stage's own directory is set here when it opens, and an
    /// unresolved path is tried against it before it is given up on.
    void setSearchPath(const std::filesystem::path& directory) { search_ = directory; }
    [[nodiscard]] const std::filesystem::path& searchPath() const noexcept { return search_; }

    /// The sampler slot for these modes, shared by every texture asking the same.
    uint32_t sampler(Wrap s, Wrap t, Filter filter = Filter::Linear);

    /// Decodes and uploads what was requested since the last commit, mips
    /// made on the device. Returns how many files loaded.
    [[nodiscard]] Result<size_t> commit();

    /// A texture no file holds: a name in the `aofx://` scheme
    /// (`aofx://slot1`) that a material's file input names and a host fills
    /// from the device every frame. Always raw -- the host's working space
    /// arrives as it is -- and never looked for on disk; until it is filled,
    /// samples report it missing as any unloaded texture does.
    [[nodiscard]] static bool isExternal(const std::string& path);
    /// Fill the external texture `name` from `rgba`: float4 texels,
    /// `rowPixels` a row, bottom row first, as a host's image plane holds
    /// them. A copy and the mips, recorded on the device's queue and not
    /// waited for; the texture is made again only when the size changes.
    [[nodiscard]] Result<void> updateExternal(const std::string& name, const gpu::Buffer& rgba,
                                              uint32_t width, uint32_t height, uint32_t rowPixels);

    /// Binds the table under `table` (a TextureTable parameter block).
    void bind(rhi::ShaderCursor table) const;

    [[nodiscard]] const TextureInfo& info(uint32_t id) const { return entries_[id].info; }
    [[nodiscard]] uint32_t count() const noexcept { return static_cast<uint32_t>(entries_.size()); }
    /// The texture in `slot`, for tests.
    [[nodiscard]] const gpu::Texture& slotTexture(uint32_t slot) const { return slots_[slot].texture; }

private:
    struct Slot {
        gpu::Texture                   texture;
        rhi::ComPtr<rhi::ITextureView> view;   ///< sRGB for 8-bit colour, else the texture's own
    };
    struct Entry {
        TextureInfo           info;
        std::vector<uint32_t> tiles;   ///< a UDIM set: slot + 1 per cell of 1001..1100, 0 where none
        uint32_t              slot = 0;
        uint32_t              udimBase = 0;
        bool                  pending = true;
    };

    TextureStore() = default;
    [[nodiscard]] Result<uint32_t> loadFile(const std::string& path, ColourSpace space, TextureInfo& info);
    /// A file under the search path whose name matches but for what a
    /// packager changes; empty if there is none.
    [[nodiscard]] std::filesystem::path besideByName(const std::string& name) const;
    [[nodiscard]] Result<void> writeRecords();

    gpu::Device*                                          device_ = nullptr;
    gpu::ComputeKernel                                    decode_;
    std::unique_ptr<gpu::MipGenerator>                    mips_;
    std::filesystem::path                                 search_;
    /// The files under `search_`, by a name with case, spaces, punctuation
    /// and extension taken out: the last resort for a path whose spelling the
    /// asset's own packaging changed. Built once, on the first miss.
    mutable std::map<std::string, std::filesystem::path>  beside_;
    mutable bool                                          besideBuilt_ = false;
    std::vector<Entry>                                    entries_;
    std::map<std::pair<std::string, ColourSpace>, uint32_t> ids_;
    std::vector<Slot>                                     slots_;
    std::vector<gpu::Sampler>                             samplers_;
    std::map<std::tuple<Wrap, Wrap, Filter>, uint32_t>    samplerIds_;
    gpu::Buffer                                           records_;
    gpu::Buffer                                           udim_;
};

}   // namespace athenea::material
