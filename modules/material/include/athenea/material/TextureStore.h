// Copyright (c) 2026 jesus luque.
//
// The textures materials sample, on the device. A file is decoded by Hio on
// the host (decompression, as the rules allow) and its bytes go up whole; a
// kernel expands its channels and flips it to v-up (texture_decode.slang), and
// its mip chain is made on the device -- averaged as light for sRGB colour,
// which is sampled through an sRGB view. A file in any other colour space is
// brought into the working space by the decode kernel itself, a generated
// one that calls the OpenColorIO function compiled for that space
// (colour::ColourCompiler). UDIM sets resolve to a slot per tile. Materials
// reach all of it through one table (athenea/material/texture_table).
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

#include "athenea/colour/ColourCompiler.h"
#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Texture.h"
#include "athenea/gpu/algo/Mips.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::material {

enum class Wrap : uint8_t { Repeat, Clamp, Mirror, Black };
enum class Filter : uint8_t { Linear, Nearest };

struct TextureInfo {
    std::string path;
    /// The colour space as the material, light or file input named it:
    /// MaterialX's `colorspace`, UsdUVTexture's `sourceColorSpace`, USD's
    /// `colorSpace` metadata. Empty or `auto`: the file says (8-bit images
    /// sRGB unless tagged otherwise, the rest linear). Resolved by
    /// colour::ColourNames when the file is read.
    std::string space;
    /// How it was read: "srgb view", "raw", "working" or the function's
    /// description; empty before it was.
    std::string decode;
    bool        loaded = false;
    bool        udim = false;
    bool        latLong = false;   ///< a dome's lat-long: its mips keep the solid-angle mean
    uint32_t    width = 0;    ///< the first tile's, for a UDIM set
    uint32_t    height = 0;
    std::string error;        ///< why it did not load, or a colour space nothing knew (read as the file says)
};

class TextureStore {
public:
    [[nodiscard]] static Result<std::unique_ptr<TextureStore>> create(gpu::ShaderLibrary& library);

    /// The id of `path` -- a file path, one holding "<UDIM>", or a path
    /// inside a package (`shot.usdz[textures/paint.jpg]`) -- read in the
    /// colour space `space` names (TextureInfo::space). Loaded at the next
    /// commit; until then, and if it cannot be, samples report it missing
    /// and materials use their defaults.
    /// `latLong`: the image is a dome's lat-long, whose mips keep its
    /// solid-angle mean (gpu::MipGenerator); asked once, it stays so.
    uint32_t request(const std::string& path, const std::string& space = {}, bool latLong = false);

    /// Whether 8-bit sRGB files stay 8-bit behind an sRGB view (the default)
    /// or go through the compiled sRGB function like any other space: the
    /// second exists to check the first against. For files read after the
    /// call.
    void setSrgbFastPath(bool enabled) { srgbFastPath_ = enabled; }

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
    [[nodiscard]] Result<uint32_t> loadFile(const std::string& path, const std::string& space, TextureInfo& info);
    /// The decode kernel that brings `space` (the config's name) into the
    /// working space: athenea_texdec_<hash>, made once per space.
    struct Decoder {
        gpu::ComputeKernel     kernel;
        colour::ColourFunction function;
    };
    [[nodiscard]] Result<const Decoder*> decoderFor(const std::string& space);
    /// A file under the search path whose name matches but for what a
    /// packager changes; empty if there is none.
    [[nodiscard]] std::filesystem::path besideByName(const std::string& name) const;
    [[nodiscard]] Result<void> writeRecords();

    gpu::Device*                                          device_ = nullptr;
    gpu::ShaderLibrary*                                   library_ = nullptr;
    gpu::ComputeKernel                                    decode_;
    /// Made at the first commit that reads a file: the studio config's names
    /// and the functions compiled from it.
    std::unique_ptr<colour::ColourCompiler>               colour_;
    std::map<std::string, Decoder>                        decoders_;
    bool                                                  srgbFastPath_ = true;
    std::unique_ptr<gpu::MipGenerator>                    mips_;
    std::filesystem::path                                 search_;
    /// The files under `search_`, by a name with case, spaces, punctuation
    /// and extension taken out: the last resort for a path whose spelling the
    /// asset's own packaging changed. Built once, on the first miss.
    mutable std::map<std::string, std::filesystem::path>  beside_;
    mutable bool                                          besideBuilt_ = false;
    std::vector<Entry>                                    entries_;
    std::map<std::pair<std::string, std::string>, uint32_t> ids_;
    std::vector<Slot>                                     slots_;
    std::vector<gpu::Sampler>                             samplers_;
    std::map<std::tuple<Wrap, Wrap, Filter>, uint32_t>    samplerIds_;
    gpu::Buffer                                           records_;
    gpu::Buffer                                           udim_;
};

}   // namespace athenea::material
