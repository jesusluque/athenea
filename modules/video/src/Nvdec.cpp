// Copyright (c) 2026 jesus luque.
//
// The NVIDIA side of Codec.h's decoder: NVDEC's HEVC decoder, on Linux and
// Windows, matching what VideoToolbox gives on Apple -- HEVC Main (Main 10 at
// 10 bits), 4:2:0, full range or refused, one picture out for each access
// unit in, synchronously -- and the luma (and at 8 bits the second plane) as
// textures of the engine's CUDA device.
//
// NOTHING IS LINKED. The driver's decoder library (nvcuvid.dll,
// libnvcuvid.so.1) and the CUDA driver (Cuda.h) are opened at run time
// through platform::SharedLibrary, so a build runs where no NVIDIA driver is
// (the factories say Unsupported). The API's types are the MIT headers of
// nv-codec-headers (third_party/nv-codec-headers: dynlink_cuviddec.h,
// dynlink_nvcuvid.h).
//
// THE SAMPLES STAY ON THE DEVICE. NVDEC's parser takes the unit as Annex B
// (its length prefixes rewritten as start codes: bytes moved, not computed
// on), decodes into its own surfaces on the engine's CUDA context, and the
// picture is mapped and copied (cuMemcpy2D, device to device) into a pitched
// allocation the DecodedFrame owns, so a frame outlives the decoder's
// surfaces as VideoToolbox's pixel buffers do. `LumaTextures::wrap` copies
// that into textures of the device (a CUDA array cannot alias linear memory),
// on the device's stream. Nothing here computes on a sample.
#include "athenea/video/Codec.h"

#include <algorithm>
#include <deque>
#include <mutex>
#include <utility>

#include "Cuda.h"   // dynlink_cuda.h, which the decoder's headers build on

#include <ffnvcodec/dynlink_nvcuvid.h>

#include "athenea/core/Platform.h"
#include "athenea/gpu/Device.h"
#include "athenea/video/Athv.h"

namespace athenea::video {
namespace {

#if defined(_WIN32)
constexpr const char* kDecodeLibrary = "nvcuvid.dll";
#else
constexpr const char* kDecodeLibrary = "libnvcuvid.so.1";
#endif

constexpr uint32_t kHvc1 = 0x31637668;   // 'hvc1'

/// The decoder's entry points, loaded once a process, beside the CUDA
/// driver's.
struct Driver {
    const cuda::Driver*       cuda = nullptr;
    platform::SharedLibrary   library;
    tcuvidGetDecoderCaps*     cuvidGetDecoderCaps = nullptr;
    tcuvidCreateDecoder*      cuvidCreateDecoder = nullptr;
    tcuvidDestroyDecoder*     cuvidDestroyDecoder = nullptr;
    tcuvidDecodePicture*      cuvidDecodePicture = nullptr;
    tcuvidGetDecodeStatus*    cuvidGetDecodeStatus = nullptr;
    tcuvidMapVideoFrame64*    cuvidMapVideoFrame = nullptr;
    tcuvidUnmapVideoFrame64*  cuvidUnmapVideoFrame = nullptr;
    tcuvidCtxLockCreate*      cuvidCtxLockCreate = nullptr;
    tcuvidCtxLockDestroy*     cuvidCtxLockDestroy = nullptr;
    tcuvidCreateVideoParser*  cuvidCreateVideoParser = nullptr;
    tcuvidParseVideoData*     cuvidParseVideoData = nullptr;
    tcuvidDestroyVideoParser* cuvidDestroyVideoParser = nullptr;
};

Result<const Driver*> driver() {
    static std::mutex              guard;
    static std::unique_ptr<Driver> loaded;
    static Error                   failed(ErrorCode::Unsupported, "");
    static bool                    tried = false;
    const std::lock_guard<std::mutex> held(guard);
    if (loaded != nullptr) return static_cast<const Driver*>(loaded.get());
    if (tried) return failed;
    tried = true;
    auto d = std::make_unique<Driver>();
    auto c = cuda::driver("NVDEC");
    if (!c) {
        failed = c.error();
        return failed;
    }
    d->cuda = *c;
    auto library = platform::SharedLibrary::open(kDecodeLibrary);
    if (!library) {
        failed = Error::make(ErrorCode::Unsupported, "video: no NVDEC in this driver ({})", library.error().message());
        return failed;
    }
    d->library = std::move(*library);
    const platform::SharedLibrary& l = d->library;
    if (!cuda::bind(l, d->cuvidGetDecoderCaps, "cuvidGetDecoderCaps") ||
        !cuda::bind(l, d->cuvidCreateDecoder, "cuvidCreateDecoder") ||
        !cuda::bind(l, d->cuvidDestroyDecoder, "cuvidDestroyDecoder") ||
        !cuda::bind(l, d->cuvidDecodePicture, "cuvidDecodePicture") ||
        !cuda::bind(l, d->cuvidGetDecodeStatus, "cuvidGetDecodeStatus") ||
        !cuda::bind(l, d->cuvidMapVideoFrame, "cuvidMapVideoFrame64") ||
        !cuda::bind(l, d->cuvidUnmapVideoFrame, "cuvidUnmapVideoFrame64") ||
        !cuda::bind(l, d->cuvidCtxLockCreate, "cuvidCtxLockCreate") ||
        !cuda::bind(l, d->cuvidCtxLockDestroy, "cuvidCtxLockDestroy") ||
        !cuda::bind(l, d->cuvidCreateVideoParser, "cuvidCreateVideoParser") ||
        !cuda::bind(l, d->cuvidParseVideoData, "cuvidParseVideoData") ||
        !cuda::bind(l, d->cuvidDestroyVideoParser, "cuvidDestroyVideoParser")) {
        failed = Error(ErrorCode::Unsupported, "video: the NVDEC library lacks an entry point this decoder needs");
        return failed;
    }
    loaded = std::move(d);
    return static_cast<const Driver*>(loaded.get());
}

/// The context a decoder (or the textures) work on: the engine's own when its
/// device is CUDA, else the first device's primary context (retained).
struct Context {
    CUcontext context = nullptr;
    CUstream  stream = nullptr;
    CUdevice  device = 0;
    bool      primary = false;
};

Result<Context> contextFor(const cuda::Driver& d, gpu::Device* device) {
    Context c;
    if (device != nullptr && device->backend() == gpu::Backend::CUDA) {
        const gpu::NativeHandles native = device->native();
        c.context = reinterpret_cast<CUcontext>(native.device.value);
        c.stream = reinterpret_cast<CUstream>(native.queue.value);
    }
    if (c.context == nullptr) {
        if (d.cuDeviceGet(&c.device, 0) != CUDA_SUCCESS ||
            d.cuDevicePrimaryCtxRetain(&c.context, c.device) != CUDA_SUCCESS) {
            return Error(ErrorCode::Unsupported, "video: no CUDA device for NVDEC");
        }
        c.primary = true;
    }
    return c;
}

/// Pitched allocations of one picture's size (luma, then the second plane),
/// handed out to frames and taken back when a frame goes: a decode allocates
/// only while more frames are alive than ever before.
class FramePool {
public:
    FramePool(const cuda::Driver& d, CUcontext context, uint32_t width, uint32_t height, uint32_t sample)
        : d_(d), context_(context), width_(width), height_(height), sample_(sample) {}
    ~FramePool() {
        const cuda::Pushed pushed(d_, context_);
        for (const auto& [p, pitch] : free_) d_.cuMemFree(p);
    }
    FramePool(const FramePool&) = delete;
    FramePool& operator=(const FramePool&) = delete;

    Result<std::pair<CUdeviceptr, size_t>> take() {
        {
            const std::lock_guard<std::mutex> held(mutex_);
            if (!free_.empty()) {
                auto out = free_.back();
                free_.pop_back();
                return out;
            }
        }
        CUdeviceptr p = 0;
        size_t pitch = 0;
        ATHENEA_TRY(cuda::check(d_, d_.cuMemAllocPitch(&p, &pitch, size_t{width_} * sample_, size_t{height_} * 3 / 2, 16),
                                "a decoded frame's memory"));
        return std::make_pair(p, pitch);
    }
    void give(CUdeviceptr p, size_t pitch) {
        const std::lock_guard<std::mutex> held(mutex_);
        free_.emplace_back(p, pitch);
    }

private:
    const cuda::Driver&                     d_;
    CUcontext                               context_;
    uint32_t                                width_, height_, sample_;
    std::mutex                              mutex_;
    std::vector<std::pair<CUdeviceptr, size_t>> free_;
};

class NvFrame final : public DecodedFrame {
public:
    NvFrame(std::shared_ptr<FramePool> pool, CUdeviceptr p, size_t pitch, uint32_t w, uint32_t h, uint32_t bits)
        : pool_(std::move(pool)), p_(p), pitch_(pitch), w_(w), h_(h), bits_(bits) {}
    ~NvFrame() override { pool_->give(p_, pitch_); }
    uint32_t width() const noexcept override { return w_; }
    uint32_t height() const noexcept override { return h_; }
    uint32_t bits() const noexcept override { return bits_; }
    CUdeviceptr luma() const noexcept { return p_; }
    /// The second plane, (Cb, Cr) pairs, height / 2 rows after the luma.
    CUdeviceptr chroma() const noexcept { return p_ + pitch_ * h_; }
    size_t pitch() const noexcept { return pitch_; }

private:
    std::shared_ptr<FramePool> pool_;
    CUdeviceptr                p_;
    size_t                     pitch_;
    uint32_t                   w_, h_, bits_;
};

class NvDecoder final : public VideoDecoder {
public:
    ~NvDecoder() override {
        if (d_ == nullptr) return;
        {
            const cuda::Pushed pushed(*d_->cuda, context_.context);
            if (parser_ != nullptr) d_->cuvidDestroyVideoParser(parser_);
            if (decoder_ != nullptr) d_->cuvidDestroyDecoder(decoder_);
            if (lock_ != nullptr) d_->cuvidCtxLockDestroy(lock_);
        }
        pool_.reset();   // frames still alive hold it, and free it on their way
        if (context_.primary) d_->cuda->cuDevicePrimaryCtxRelease(context_.device);
    }

    static Result<std::unique_ptr<VideoDecoder>> create(uint32_t bits, gpu::Device* device) {
        if (bits != 8 && bits != 10) {
            return Error::make(ErrorCode::InvalidArgument, "video: NVDEC decodes 8 or 10 bits, not {}", bits);
        }
        auto loaded = driver();
        if (!loaded) return std::move(loaded).error();
        std::unique_ptr<NvDecoder> n(new NvDecoder());
        n->d_ = *loaded;
        n->bits_ = bits;
        auto context = contextFor(*n->d_->cuda, device);
        if (!context) {
            n->d_ = nullptr;
            return std::move(context).error();
        }
        n->context_ = *context;
        const cuda::Pushed pushed(*n->d_->cuda, n->context_.context);
        ATHENEA_TRY(cuda::check(*n->d_->cuda, n->d_->cuvidCtxLockCreate(&n->lock_, n->context_.context),
                                "NVDEC's context lock"));
        CUVIDPARSERPARAMS p{};
        p.CodecType = cudaVideoCodec_HEVC;
        p.ulMaxNumDecodeSurfaces = 1;   // the sequence callback says how many
        p.ulMaxDisplayDelay = 0;        // a picture out in the call its unit went in
        p.ulErrorThreshold = 0;
        p.pUserData = n.get();
        p.pfnSequenceCallback = &NvDecoder::onSequence;
        p.pfnDecodePicture = &NvDecoder::onDecode;
        p.pfnDisplayPicture = &NvDecoder::onDisplay;
        ATHENEA_TRY(cuda::check(*n->d_->cuda, n->d_->cuvidCreateVideoParser(&n->parser_, &p), "NVDEC's parser"));
        return std::unique_ptr<VideoDecoder>(std::move(n));
    }

    Result<std::shared_ptr<DecodedFrame>> decode(std::span<const uint8_t> unit,
                                                 const std::vector<std::vector<uint8_t>>& sets,
                                                 bool keyframe) override {
        static const std::vector<std::vector<uint8_t>> kNone;
        const std::lock_guard<std::mutex> held(mutex_);
        const cuda::Pushed pushed(*d_->cuda, context_.context);
        const std::vector<uint8_t> bytes = annexB(unit, keyframe ? sets : kNone);
        CUVIDSOURCEDATAPACKET packet{};
        packet.flags = CUVID_PKT_ENDOFPICTURE | CUVID_PKT_TIMESTAMP;
        packet.payload_size = static_cast<tcu_ulong>(bytes.size());
        packet.payload = bytes.data();
        packet.timestamp = static_cast<CUvideotimestamp>(units_++);
        failed_ = ok();
        const CUresult parsed = d_->cuvidParseVideoData(parser_, &packet);
        if (!failed_) return std::move(failed_).error();
        ATHENEA_TRY(cuda::check(*d_->cuda, parsed, "NVDEC's parser"));
        if (ready_.empty()) {
            if (decoder_ == nullptr) {
                return Error(ErrorCode::InvalidArgument, "video: a frame before any key frame with parameter sets");
            }
            return Error(ErrorCode::DeviceFailure, "video: NVDEC gave no picture for an access unit");
        }
        const CUVIDPARSERDISPINFO shown = ready_.front();
        ready_.pop_front();
        CUVIDGETDECODESTATUS status{};
        if (d_->cuvidGetDecodeStatus(decoder_, shown.picture_index, &status) == CUDA_SUCCESS &&
            (status.decodeStatus == cuvidDecodeStatus_Error ||
             status.decodeStatus == cuvidDecodeStatus_Error_Concealed)) {
            return Error::make(ErrorCode::DeviceFailure, "video: NVDEC could not decode an access unit (status {})",
                               static_cast<int>(status.decodeStatus));
        }
        CUVIDPROCPARAMS proc{};
        proc.progressive_frame = shown.progressive_frame;
        proc.top_field_first = shown.top_field_first;
        proc.output_stream = context_.stream;
        unsigned long long surface = 0;
        unsigned int surfacePitch = 0;
        ATHENEA_TRY(cuda::check(*d_->cuda,
                                d_->cuvidMapVideoFrame(decoder_, shown.picture_index, &surface, &surfacePitch, &proc),
                                "mapping a decoded picture"));
        auto copied = copyOut(surface, surfacePitch);
        const CUresult unmapped = d_->cuvidUnmapVideoFrame(decoder_, surface);
        if (!copied) return std::move(copied).error();
        ATHENEA_TRY(cuda::check(*d_->cuda, unmapped, "unmapping a decoded picture"));
        return std::move(*copied);
    }

    bool hardware() const noexcept override { return true; }

private:
    NvDecoder() = default;

    uint32_t sample() const noexcept { return bits_ > 8 ? 2 : 1; }

    /// The mapped surface (luma, then the second plane `surfaceHeight_` rows
    /// down) into a frame of the pool, on the stream, finished on return.
    Result<std::shared_ptr<DecodedFrame>> copyOut(unsigned long long surface, unsigned int surfacePitch) {
        auto taken = pool_->take();
        if (!taken) return std::move(taken).error();
        const auto [p, pitch] = *taken;
        auto frame = std::make_shared<NvFrame>(pool_, p, pitch, width_, height_, bits_);
        const auto copy = [&](CUdeviceptr from, CUdeviceptr to, uint32_t rows) {
            CUDA_MEMCPY2D c{};
            c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            c.srcDevice = from;
            c.srcPitch = surfacePitch;
            c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            c.dstDevice = to;
            c.dstPitch = pitch;
            c.WidthInBytes = size_t{width_} * sample();
            c.Height = rows;
            return d_->cuda->cuMemcpy2DAsync(&c, context_.stream);
        };
        ATHENEA_TRY(cuda::check(*d_->cuda, copy(surface, p, height_), "copying a decoded luma plane"));
        ATHENEA_TRY(cuda::check(*d_->cuda,
                                copy(surface + uint64_t{surfacePitch} * surfaceHeight_, frame->chroma(), height_ / 2),
                                "copying a decoded second plane"));
        ATHENEA_TRY(cuda::check(*d_->cuda, d_->cuda->cuStreamSynchronize(context_.stream), "a decoded frame's copy"));
        return std::shared_ptr<DecodedFrame>(std::move(frame));
    }

    /// A sequence header: the decoder made (or made again) for its size.
    /// Returns the decode surfaces the parser cycles through; 0 stops it.
    int sequence(const CUVIDEOFORMAT& f) {
        if (f.codec != cudaVideoCodec_HEVC || f.chroma_format != cudaVideoChromaFormat_420) {
            failed_ = Error(ErrorCode::Unsupported, "video: NVDEC takes HEVC 4:2:0 only");
            return 0;
        }
        if (uint32_t{f.bit_depth_luma_minus8} + 8 != bits_) {
            failed_ = Error::make(ErrorCode::InvalidArgument, "video: the stream is {} bits; its header says {}",
                                  uint32_t{f.bit_depth_luma_minus8} + 8, bits_);
            return 0;
        }
        // The planes are full range. NVDEC hands the samples over as coded,
        // so video range would not be stretched here; it is refused anyway,
        // as VideoToolbox's side refuses it, so a file reads the same on both.
        if (f.video_signal_description.video_full_range_flag == 0) {
            failed_ = Error(ErrorCode::InvalidArgument,
                            "video: the stream's SPS says video range (16-235); video planes are full range");
            return 0;
        }
        const auto w = static_cast<uint32_t>(f.display_area.right - f.display_area.left);
        const auto h = static_cast<uint32_t>(f.display_area.bottom - f.display_area.top);
        const int surfaces = std::max<int>(f.min_num_decode_surfaces, 1);
        if (decoder_ != nullptr && f.coded_width == codedWidth_ && f.coded_height == codedHeight_ && w == width_ &&
            h == height_) {
            return surfaces_;
        }
        if (decoder_ != nullptr) {
            d_->cuvidDestroyDecoder(decoder_);
            decoder_ = nullptr;
        }
        CUVIDDECODECAPS caps{};
        caps.eCodecType = cudaVideoCodec_HEVC;
        caps.eChromaFormat = cudaVideoChromaFormat_420;
        caps.nBitDepthMinus8 = bits_ - 8;
        if (d_->cuvidGetDecoderCaps(&caps) != CUDA_SUCCESS || caps.bIsSupported == 0) {
            failed_ = Error::make(ErrorCode::Unsupported, "video: this GPU's NVDEC does not decode HEVC at {} bits",
                                  bits_);
            return 0;
        }
        if (f.coded_width > caps.nMaxWidth || f.coded_height > caps.nMaxHeight) {
            failed_ = Error::make(ErrorCode::Unsupported, "video: NVDEC decodes up to {} x {}, not {} x {}",
                                  caps.nMaxWidth, caps.nMaxHeight, f.coded_width, f.coded_height);
            return 0;
        }
        CUVIDDECODECREATEINFO info{};
        info.ulWidth = f.coded_width;
        info.ulHeight = f.coded_height;
        info.ulMaxWidth = f.coded_width;
        info.ulMaxHeight = f.coded_height;
        info.ulNumDecodeSurfaces = static_cast<tcu_ulong>(surfaces);
        info.CodecType = cudaVideoCodec_HEVC;
        info.ChromaFormat = cudaVideoChromaFormat_420;
        info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
        info.bitDepthMinus8 = bits_ - 8;
        info.display_area.left = static_cast<short>(f.display_area.left);
        info.display_area.top = static_cast<short>(f.display_area.top);
        info.display_area.right = static_cast<short>(f.display_area.right);
        info.display_area.bottom = static_cast<short>(f.display_area.bottom);
        info.OutputFormat = bits_ > 8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
        info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
        info.ulTargetWidth = w;
        info.ulTargetHeight = h;
        info.ulNumOutputSurfaces = 2;
        info.vidLock = lock_;
        if (const CUresult r = d_->cuvidCreateDecoder(&decoder_, &info); r != CUDA_SUCCESS) {
            decoder_ = nullptr;
            failed_ = cuda::check(*d_->cuda, r, "creating NVDEC's decoder");
            return 0;
        }
        codedWidth_ = f.coded_width;
        codedHeight_ = f.coded_height;
        width_ = w;
        height_ = h;
        surfaceHeight_ = h;   // the output surface's rows: the target height
        surfaces_ = surfaces;
        pool_ = std::make_shared<FramePool>(*d_->cuda, context_.context, w, h, sample());
        return surfaces;
    }

    static int CUDAAPI onSequence(void* user, CUVIDEOFORMAT* f) { return static_cast<NvDecoder*>(user)->sequence(*f); }

    static int CUDAAPI onDecode(void* user, CUVIDPICPARAMS* p) {
        auto* n = static_cast<NvDecoder*>(user);
        if (n->decoder_ == nullptr) return 0;
        if (const CUresult r = n->d_->cuvidDecodePicture(n->decoder_, p); r != CUDA_SUCCESS) {
            n->failed_ = cuda::check(*n->d_->cuda, r, "decoding a picture");
            return 0;
        }
        return 1;
    }

    static int CUDAAPI onDisplay(void* user, CUVIDPARSERDISPINFO* shown) {
        if (shown != nullptr) static_cast<NvDecoder*>(user)->ready_.push_back(*shown);
        return 1;
    }

    const Driver*                   d_ = nullptr;
    Context                         context_;
    uint32_t                        bits_ = 8;
    std::mutex                      mutex_;
    CUvideoctxlock                  lock_ = nullptr;
    CUvideoparser                   parser_ = nullptr;
    CUvideodecoder                  decoder_ = nullptr;
    unsigned int                    codedWidth_ = 0, codedHeight_ = 0;
    uint32_t                        width_ = 0, height_ = 0, surfaceHeight_ = 0;
    int                             surfaces_ = 1;
    uint64_t                        units_ = 0;
    std::deque<CUVIDPARSERDISPINFO> ready_;
    Result<void>                    failed_ = ok();
    std::shared_ptr<FramePool>      pool_;
};

// --- the decoded planes as textures of the CUDA device ----------------------------

class NvLumaTextures final : public LumaTextures {
public:
    ~NvLumaTextures() override {
        if (d_ != nullptr && context_.primary) d_->cuDevicePrimaryCtxRelease(context_.device);
    }

    static Result<std::unique_ptr<LumaTextures>> create(gpu::Device& device) {
        if (device.backend() != gpu::Backend::CUDA) {
            return Error(ErrorCode::Unsupported, "video: NVDEC's frames are read as textures by a CUDA device only");
        }
        auto loaded = cuda::driver("NVDEC's textures");
        if (!loaded) return std::move(loaded).error();
        std::unique_ptr<NvLumaTextures> t(new NvLumaTextures());
        t->d_ = *loaded;
        t->device_ = &device;
        auto context = contextFor(*t->d_, &device);
        if (!context) {
            t->d_ = nullptr;
            return std::move(context).error();
        }
        t->context_ = *context;
        return std::unique_ptr<LumaTextures>(std::move(t));
    }

    Result<LumaTexture> wrap(const DecodedFrame& frame) override {
        const auto* f = dynamic_cast<const NvFrame*>(&frame);
        if (f == nullptr) return Error(ErrorCode::InvalidArgument, "video: a frame NVDEC did not decode");
        const bool wide = f->bits() > 8;
        LumaTexture out;
        ATHENEA_TRY(plane(f->width(), f->height(), wide ? rhi::Format::R16Unorm : rhi::Format::R8Unorm,
                          "video.luma", f->luma(), f->pitch(), size_t{f->width()} * (wide ? 2 : 1), out.texture));
        // The second plane, (Cb, Cr) a 2 x 2 block, as RG8Unorm: what a frame
        // with its colour in chroma reads (8 bits only, as on Metal).
        if (!wide) {
            ATHENEA_TRY(plane(f->width() / 2, f->height() / 2, rhi::Format::RG8Unorm, "video.chroma", f->chroma(),
                              f->pitch(), f->width(), out.chroma));
        }
        const cuda::Pushed pushed(*d_, context_.context);
        ATHENEA_TRY(cuda::check(*d_, d_->cuStreamSynchronize(context_.stream), "a decoded frame into textures"));
        return out;
    }

private:
    NvLumaTextures() = default;

    /// A `w` x `h` texture of `format`, `rowBytes` of each of its rows copied
    /// from `from` (`pitch` apart) on the device's stream.
    Result<void> plane(uint32_t w, uint32_t h, rhi::Format format, const char* label, CUdeviceptr from, size_t pitch,
                       size_t rowBytes, rhi::ComPtr<rhi::ITexture>& texture) {
        rhi::TextureDesc desc;
        desc.type = rhi::TextureType::Texture2D;
        desc.size = {w, h, 1};
        desc.mipCount = 1;
        desc.arrayLength = 1;
        desc.format = format;
        desc.usage = rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopyDestination;
        desc.label = label;
        if (SLANG_FAILED(device_->rhi()->createTexture(desc, nullptr, texture.writeRef()))) {
            return Error::make(ErrorCode::DeviceFailure, "video: no {} x {} texture for a decoded plane", w, h);
        }
        rhi::NativeHandle handle;
        if (SLANG_FAILED(texture->getNativeHandle(&handle)) || handle.type != rhi::NativeHandleType::CUarray) {
            return Error(ErrorCode::DeviceFailure, "video: a texture of the CUDA device that is not a CUDA array");
        }
        CUDA_MEMCPY2D c{};
        c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        c.srcDevice = from;
        c.srcPitch = pitch;
        c.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        c.dstArray = reinterpret_cast<CUarray>(handle.value);
        c.WidthInBytes = rowBytes;
        c.Height = h;
        const cuda::Pushed pushed(*d_, context_.context);
        return cuda::check(*d_, d_->cuMemcpy2DAsync(&c, context_.stream), "copying a decoded plane into a texture");
    }

    const cuda::Driver* d_ = nullptr;
    gpu::Device*        device_ = nullptr;
    Context             context_;
};

}   // namespace

Result<std::unique_ptr<VideoDecoder>> createHardwareDecoder(uint32_t codec, uint32_t bits, gpu::Device* device) {
    if (codec != kHvc1) return Error::make(ErrorCode::Unsupported, "video: codec {:#x} is not HEVC", codec);
    return NvDecoder::create(bits, device);
}

Result<std::unique_ptr<LumaTextures>> createLumaTextures(gpu::Device& device) {
    return NvLumaTextures::create(device);
}

}   // namespace athenea::video
