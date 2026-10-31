// Copyright (c) 2026 jesus luque.
//
// The NVIDIA side of Codec.h's encoder: NVENC's HEVC encoder, on Linux and
// Windows, matching what VideoToolbox gives on Apple -- HEVC Main (Main 10 at
// 10 bits), 4:2:0, full range in the VUI, no B pictures, each frame out
// before the next goes in, length-prefixed NAL units with the parameter sets
// (VPS, SPS, PPS) beside a key frame's unit.
//
// NOTHING IS LINKED. The driver's encoder library (nvEncodeAPI64.dll,
// libnvidia-encode.so.1) and the CUDA driver (nvcuda.dll, libcuda.so.1) are
// opened at run time through platform::SharedLibrary, so a build runs where
// no NVIDIA driver is (the factory says Unsupported). The API's types are the
// MIT headers of nv-codec-headers (third_party/nv-codec-headers).
//
// THE SAMPLES STAY ON THE DEVICE. The session is opened on the engine's own
// CUDA context (gpu::Device::native()), and its input is one pitched NV12 (or
// P010) allocation of that context registered once as a CUDA device pointer.
// `encodeDevice` copies the planes PlanesCodec laid into it on the device
// (cuMemcpy2D, device to device, on the device's stream); `encode`, from host
// spans, is the same copy from the host. The second plane is mid grey when no
// chroma is given (cuMemsetD8/D16). Nothing here computes on a sample.
#include "athenea/video/Codec.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>   // GUID and RECT, which nvEncodeAPI.h takes from it
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

#include <ffnvcodec/dynlink_cuda.h>
#include <ffnvcodec/nvEncodeAPI.h>

#include "athenea/core/Platform.h"
#include "athenea/gpu/Device.h"

namespace athenea::video {
namespace {

#if defined(_WIN32)
constexpr const char* kEncodeLibrary = "nvEncodeAPI64.dll";
constexpr const char* kCudaLibrary = "nvcuda.dll";
#else
constexpr const char* kEncodeLibrary = "libnvidia-encode.so.1";
constexpr const char* kCudaLibrary = "libcuda.so.1";
#endif

// The two the dynlink header has no typedef for.
typedef CUresult CUDAAPI tcuMemsetD8Async_(CUdeviceptr, unsigned char, size_t, CUstream);
typedef CUresult CUDAAPI tcuMemsetD16Async_(CUdeviceptr, unsigned short, size_t, CUstream);
typedef NVENCSTATUS NVENCAPI tNvEncodeAPICreateInstance(NV_ENCODE_API_FUNCTION_LIST*);
typedef NVENCSTATUS NVENCAPI tNvEncodeAPIGetMaxSupportedVersion(uint32_t*);

/// The driver's entry points, loaded once a process.
struct Driver {
    platform::SharedLibrary      cudaLibrary, encodeLibrary;
    tcuInit*                     cuInit = nullptr;
    tcuDeviceGet*                cuDeviceGet = nullptr;
    tcuDevicePrimaryCtxRetain*   cuDevicePrimaryCtxRetain = nullptr;
    tcuDevicePrimaryCtxRelease*  cuDevicePrimaryCtxRelease = nullptr;
    tcuCtxPushCurrent_v2*        cuCtxPushCurrent = nullptr;
    tcuCtxPopCurrent_v2*         cuCtxPopCurrent = nullptr;
    tcuMemAllocPitch_v2*         cuMemAllocPitch = nullptr;
    tcuMemFree_v2*               cuMemFree = nullptr;
    tcuMemcpy2DAsync_v2*         cuMemcpy2DAsync = nullptr;
    tcuMemsetD8Async_*           cuMemsetD8Async = nullptr;
    tcuMemsetD16Async_*          cuMemsetD16Async = nullptr;
    tcuStreamSynchronize*        cuStreamSynchronize = nullptr;
    tcuGetErrorName*             cuGetErrorName = nullptr;
    NV_ENCODE_API_FUNCTION_LIST  nvenc{};
};

template <typename F>
bool bind(const platform::SharedLibrary& library, F*& to, const char* name) {
    to = reinterpret_cast<F*>(library.symbol(name));
    return to != nullptr;
}

Result<Driver*> driver() {
    static std::mutex          guard;
    static std::unique_ptr<Driver> loaded;
    static Error               failed(ErrorCode::Unsupported, "");
    static bool                tried = false;
    const std::lock_guard<std::mutex> held(guard);
    if (loaded != nullptr) return loaded.get();
    if (tried) return failed;
    tried = true;
    auto d = std::make_unique<Driver>();
    auto cuda = platform::SharedLibrary::open(kCudaLibrary);
    if (!cuda) {
        failed = Error::make(ErrorCode::Unsupported, "video: no NVIDIA driver for NVENC ({})", cuda.error().message());
        return failed;
    }
    d->cudaLibrary = std::move(*cuda);
    const platform::SharedLibrary& c = d->cudaLibrary;
    if (!bind(c, d->cuInit, "cuInit") || !bind(c, d->cuDeviceGet, "cuDeviceGet") ||
        !bind(c, d->cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") ||
        !bind(c, d->cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2") ||
        !bind(c, d->cuCtxPushCurrent, "cuCtxPushCurrent_v2") || !bind(c, d->cuCtxPopCurrent, "cuCtxPopCurrent_v2") ||
        !bind(c, d->cuMemAllocPitch, "cuMemAllocPitch_v2") || !bind(c, d->cuMemFree, "cuMemFree_v2") ||
        !bind(c, d->cuMemcpy2DAsync, "cuMemcpy2DAsync_v2") || !bind(c, d->cuMemsetD8Async, "cuMemsetD8Async") ||
        !bind(c, d->cuMemsetD16Async, "cuMemsetD16Async") ||
        !bind(c, d->cuStreamSynchronize, "cuStreamSynchronize") || !bind(c, d->cuGetErrorName, "cuGetErrorName")) {
        failed = Error(ErrorCode::Unsupported, "video: the CUDA driver lacks an entry point NVENC needs");
        return failed;
    }
    auto enc = platform::SharedLibrary::open(kEncodeLibrary);
    if (!enc) {
        failed = Error::make(ErrorCode::Unsupported, "video: no NVENC in this driver ({})", enc.error().message());
        return failed;
    }
    d->encodeLibrary = std::move(*enc);
    tNvEncodeAPIGetMaxSupportedVersion* maxVersion = nullptr;
    tNvEncodeAPICreateInstance* create = nullptr;
    if (!bind(d->encodeLibrary, maxVersion, "NvEncodeAPIGetMaxSupportedVersion") ||
        !bind(d->encodeLibrary, create, "NvEncodeAPICreateInstance")) {
        failed = Error(ErrorCode::Unsupported, "video: the NVENC library has no NvEncodeAPICreateInstance");
        return failed;
    }
    uint32_t most = 0;
    const uint32_t wanted = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
    if (maxVersion(&most) != NV_ENC_SUCCESS || most < wanted) {
        failed = Error::make(ErrorCode::Unsupported,
                             "video: the driver's NVENC is API {}.{}; this build wants {}.{} (a newer driver)",
                             most >> 4, most & 0xF, NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);
        return failed;
    }
    d->nvenc.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    if (create(&d->nvenc) != NV_ENC_SUCCESS || d->nvenc.nvEncOpenEncodeSessionEx == nullptr) {
        failed = Error(ErrorCode::Unsupported, "video: NvEncodeAPICreateInstance failed");
        return failed;
    }
    if (d->cuInit(0) != CUDA_SUCCESS) {
        failed = Error(ErrorCode::Unsupported, "video: cuInit failed (no CUDA device)");
        return failed;
    }
    loaded = std::move(d);
    return loaded.get();
}

/// The context current on this thread while it lives.
class Pushed {
public:
    Pushed(const Driver& d, CUcontext context) : d_(d) { d_.cuCtxPushCurrent(context); }
    ~Pushed() {
        CUcontext popped = nullptr;
        d_.cuCtxPopCurrent(&popped);
    }
    Pushed(const Pushed&) = delete;
    Pushed& operator=(const Pushed&) = delete;

private:
    const Driver& d_;
};

/// The NAL unit type of an HEVC NAL unit's first header byte.
uint32_t nalType(uint8_t first) { return (first >> 1u) & 0x3Fu; }

class NvEncoder final : public VideoEncoder {
public:
    ~NvEncoder() override {
        if (d_ == nullptr) return;
        const Pushed pushed(*d_, context_);
        const NV_ENCODE_API_FUNCTION_LIST& f = d_->nvenc;
        if (encoder_ != nullptr) {
            if (registered_ != nullptr) f.nvEncUnregisterResource(encoder_, registered_);
            if (bitstream_ != nullptr) f.nvEncDestroyBitstreamBuffer(encoder_, bitstream_);
            f.nvEncDestroyEncoder(encoder_);
        }
        if (input_ != 0) d_->cuMemFree(input_);
        if (primary_) d_->cuDevicePrimaryCtxRelease(cudaDevice_);
    }

    static Result<std::unique_ptr<VideoEncoder>> create(const EncoderSettings& s) {
        auto loaded = driver();
        if (!loaded) return std::move(loaded).error();
        Driver* d = *loaded;
        if (s.width == 0 || s.height == 0 || (s.width & 1u) != 0 || (s.height & 1u) != 0) {
            return Error::make(ErrorCode::InvalidArgument, "video: NVENC takes even sizes, not {} x {}", s.width,
                               s.height);
        }
        if (s.bits != 8 && s.bits != 10) {
            return Error::make(ErrorCode::InvalidArgument, "video: NVENC encodes 8 or 10 bits, not {}", s.bits);
        }
        std::unique_ptr<NvEncoder> e(new NvEncoder());
        e->d_ = d;
        e->settings_ = s;
        // The engine's own context when the device is CUDA: its buffers are
        // then this session's input. Otherwise the first device's primary
        // context, for host samples.
        if (s.device != nullptr && s.device->backend() == gpu::Backend::CUDA) {
            const gpu::NativeHandles native = s.device->native();
            e->context_ = reinterpret_cast<CUcontext>(native.device.value);
            e->stream_ = reinterpret_cast<CUstream>(native.queue.value);
            e->takesDevice_ = e->context_ != nullptr;
        }
        if (e->context_ == nullptr) {
            if (d->cuDeviceGet(&e->cudaDevice_, 0) != CUDA_SUCCESS ||
                d->cuDevicePrimaryCtxRetain(&e->context_, e->cudaDevice_) != CUDA_SUCCESS) {
                return Error(ErrorCode::Unsupported, "video: no CUDA device for NVENC");
            }
            e->primary_ = true;
        }
        ATHENEA_TRY(e->open());
        return std::unique_ptr<VideoEncoder>(std::move(e));
    }

    Result<std::vector<EncodedUnit>> encode(std::span<const std::byte> luma, std::span<const std::byte> chroma,
                                            uint64_t index) override {
        const EncoderSettings& s = settings_;
        const size_t row = size_t{s.width} * sample();
        if (luma.size() != row * s.height) {
            return Error::make(ErrorCode::InvalidArgument, "video: {} bytes of luma for {} x {}", luma.size(), s.width,
                               s.height);
        }
        if (!chroma.empty() && (sample() != 1 || chroma.size() != size_t{s.width} * (s.height / 2))) {
            return Error::make(ErrorCode::InvalidArgument, "video: {} bytes of chroma for {} x {}", chroma.size(),
                               s.width, s.height);
        }
        const Pushed pushed(*d_, context_);
        ATHENEA_TRY(copyIn(CU_MEMORYTYPE_HOST, luma.data(), 0, row, s.height, 0));
        if (!chroma.empty()) {
            ATHENEA_TRY(copyIn(CU_MEMORYTYPE_HOST, chroma.data(), 0, row, s.height / 2, chromaOffset()));
            greyed_ = false;
        } else {
            ATHENEA_TRY(grey());
        }
        return picture(index);
    }

    Result<std::vector<EncodedUnit>> encodeDevice(const DevicePicture& p, uint64_t index) override {
        if (!takesDevice_) {
            return Error(ErrorCode::Unsupported, "video: this NVENC session is not on the engine's CUDA context");
        }
        const EncoderSettings& s = settings_;
        const size_t row = size_t{s.width} * sample();
        auto luma = devicePointer(p.luma);
        if (!luma) return std::move(luma).error();
        CUdeviceptr chroma = 0;
        if (p.chroma != nullptr) {
            if (sample() != 1) return Error(ErrorCode::InvalidArgument, "video: chroma is 8 bits only");
            auto c = devicePointer(p.chroma);
            if (!c) return std::move(c).error();
            chroma = *c;
        }
        const Pushed pushed(*d_, context_);
        ATHENEA_TRY(copyIn(CU_MEMORYTYPE_DEVICE, nullptr, *luma + p.lumaOffset, p.lumaPitch != 0 ? p.lumaPitch : row,
                           s.height, 0));
        if (p.chroma != nullptr) {
            ATHENEA_TRY(copyIn(CU_MEMORYTYPE_DEVICE, nullptr, chroma + p.chromaOffset,
                               p.chromaPitch != 0 ? p.chromaPitch : row, s.height / 2, chromaOffset()));
            greyed_ = false;
        } else {
            ATHENEA_TRY(grey());
        }
        return picture(index);
    }

    bool takesDevice() const noexcept override { return takesDevice_; }

    Result<std::vector<EncodedUnit>> finish() override {
        // No B pictures and no look-ahead: every frame came out with its own
        // call, and the end of the stream flushes nothing.
        const Pushed pushed(*d_, context_);
        NV_ENC_PIC_PARAMS eos{};
        eos.version = NV_ENC_PIC_PARAMS_VER;
        eos.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
        ATHENEA_TRY(check(d_->nvenc.nvEncEncodePicture(encoder_, &eos), "the end of the stream"));
        return std::vector<EncodedUnit>{};
    }

    bool hardware() const noexcept override { return true; }

private:
    NvEncoder() = default;

    size_t   sample() const noexcept { return settings_.bits > 8 ? 2 : 1; }
    uint64_t chromaOffset() const noexcept { return uint64_t{pitch_} * settings_.height; }

    Result<void> check(NVENCSTATUS status, const char* what) const {
        if (status == NV_ENC_SUCCESS) return ok();
        const char* last = encoder_ != nullptr && d_->nvenc.nvEncGetLastErrorString != nullptr
                               ? d_->nvenc.nvEncGetLastErrorString(encoder_)
                               : nullptr;
        return Error::make(status == NV_ENC_ERR_OUT_OF_MEMORY ? ErrorCode::OutOfMemory : ErrorCode::DeviceFailure,
                           "video: NVENC failed at {} ({}{}{})", what, static_cast<int>(status),
                           last != nullptr && *last != '\0' ? ": " : "", last != nullptr ? last : "");
    }

    Result<void> cuda(CUresult r, const char* what) const {
        if (r == CUDA_SUCCESS) return ok();
        const char* name = nullptr;
        d_->cuGetErrorName(r, &name);
        constexpr int kOutOfMemory = 2;   // CUDA_ERROR_OUT_OF_MEMORY
        return Error::make(static_cast<int>(r) == kOutOfMemory ? ErrorCode::OutOfMemory : ErrorCode::DeviceFailure,
                           "video: CUDA failed at {} ({})", what, name != nullptr ? name : "?");
    }

    Result<CUdeviceptr> devicePointer(rhi::IBuffer* buffer) const {
        rhi::NativeHandle handle;
        if (buffer == nullptr || SLANG_FAILED(buffer->getNativeHandle(&handle)) ||
            handle.type != rhi::NativeHandleType::CUdeviceptr) {
            return Error(ErrorCode::InvalidArgument, "video: a plane that is not a CUDA buffer");
        }
        return static_cast<CUdeviceptr>(handle.value);
    }

    /// `rows` rows of `rowBytes` into the input at `at`, from the host or the
    /// device, on the stream; finished before it returns.
    Result<void> copyIn(CUmemorytype from, const void* host, CUdeviceptr device, size_t rowBytes, uint32_t rows,
                        uint64_t at) {
        CUDA_MEMCPY2D c{};
        c.srcMemoryType = from;
        c.srcHost = host;
        c.srcDevice = device;
        c.srcPitch = rowBytes;
        c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        c.dstDevice = input_ + at;
        c.dstPitch = pitch_;
        c.WidthInBytes = size_t{settings_.width} * sample();
        c.Height = rows;
        ATHENEA_TRY(cuda(d_->cuMemcpy2DAsync(&c, stream_), "the copy into the encoder's input"));
        return cuda(d_->cuStreamSynchronize(stream_), "the copy into the encoder's input");
    }

    /// The second plane at mid grey: 128, or 0x8000 as P010 holds 512.
    Result<void> grey() {
        if (greyed_) return ok();
        const size_t bytes = size_t{pitch_} * (settings_.height / 2);
        const CUresult r = sample() == 1 ? d_->cuMemsetD8Async(input_ + chromaOffset(), 128, bytes, stream_)
                                         : d_->cuMemsetD16Async(input_ + chromaOffset(), 0x8000, bytes / 2, stream_);
        ATHENEA_TRY(cuda(r, "the grey chroma"));
        ATHENEA_TRY(cuda(d_->cuStreamSynchronize(stream_), "the grey chroma"));
        greyed_ = true;
        return ok();
    }

    Result<void> open() {
        const EncoderSettings& s = settings_;
        const Pushed pushed(*d_, context_);
        const NV_ENCODE_API_FUNCTION_LIST& f = d_->nvenc;
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session{};
        session.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        session.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        session.device = context_;
        session.apiVersion = NVENCAPI_VERSION;
        if (const NVENCSTATUS st = f.nvEncOpenEncodeSessionEx(&session, &encoder_); st != NV_ENC_SUCCESS) {
            encoder_ = nullptr;
            return Error::make(ErrorCode::Unsupported, "video: no NVENC session on this device ({})",
                               static_cast<int>(st));
        }
        const GUID preset = NV_ENC_PRESET_P4_GUID;
        const NV_ENC_TUNING_INFO tuning = s.realTime ? NV_ENC_TUNING_INFO_LOW_LATENCY : NV_ENC_TUNING_INFO_HIGH_QUALITY;
        NV_ENC_PRESET_CONFIG presetConfig{};
        presetConfig.version = NV_ENC_PRESET_CONFIG_VER;
        presetConfig.presetCfg.version = NV_ENC_CONFIG_VER;
        ATHENEA_TRY(check(f.nvEncGetEncodePresetConfigEx(encoder_, NV_ENC_CODEC_HEVC_GUID, preset, tuning,
                                                         &presetConfig),
                          "the preset"));
        config_ = presetConfig.presetCfg;
        config_.version = NV_ENC_CONFIG_VER;
        config_.profileGUID = s.bits > 8 ? NV_ENC_HEVC_PROFILE_MAIN10_GUID : NV_ENC_HEVC_PROFILE_MAIN_GUID;
        const uint32_t key = std::max<uint32_t>(s.keyInterval, 1);
        config_.gopLength = key;
        config_.frameIntervalP = 1;   // I and P only: a frame out for each frame in
        NV_ENC_RC_PARAMS& rc = config_.rcParams;
        rc.version = NV_ENC_RC_PARAMS_VER;
        rc.enableLookahead = 0;
        rc.lookaheadDepth = 0;
        rc.enableAQ = 0;
        rc.enableTemporalAQ = 0;
        rc.multiPass = NV_ENC_MULTI_PASS_DISABLED;
        if (s.megabits > 0.0) {
            rc.rateControlMode = NV_ENC_PARAMS_RC_VBR;
            rc.averageBitRate = static_cast<uint32_t>(std::min(s.megabits * 1.0e6, 4.0e9));
            rc.maxBitRate = static_cast<uint32_t>(std::min(s.megabits * 2.0e6, 4.0e9));
        } else {
            // Constant quality as one QP for every picture: 51 at quality
            // 0, 0 at 1 (HEVC's QP scale, linear in the setting).
            const uint32_t qp =
                static_cast<uint32_t>(std::lround((1.0 - std::clamp(s.quality, 0.0, 1.0)) * 51.0));
            rc.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
            rc.constQP = {qp, qp, qp};
        }
        NV_ENC_CONFIG_HEVC& h = config_.encodeCodecConfig.hevcConfig;
        h.idrPeriod = key;
        h.repeatSPSPPS = 1;   // the parameter sets on every IDR
        h.disableSPSPPS = 0;
        h.outputAUD = 0;
        h.chromaFormatIDC = 1;
        h.inputBitDepth = s.bits > 8 ? NV_ENC_BIT_DEPTH_10 : NV_ENC_BIT_DEPTH_8;
        h.outputBitDepth = h.inputBitDepth;
        h.sliceMode = 0;
        h.sliceModeData = 0;
        // Full range, as the planes are: a decoder that read video range
        // would stretch every sample from 16-235 (VideoToolbox refuses it).
        h.hevcVUIParameters.videoSignalTypePresentFlag = 1;
        h.hevcVUIParameters.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
        h.hevcVUIParameters.videoFullRangeFlag = 1;
        h.hevcVUIParameters.colourDescriptionPresentFlag = 0;
        NV_ENC_INITIALIZE_PARAMS init{};
        init.version = NV_ENC_INITIALIZE_PARAMS_VER;
        init.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
        init.presetGUID = preset;
        init.tuningInfo = tuning;
        init.encodeWidth = s.width;
        init.encodeHeight = s.height;
        init.darWidth = s.width;
        init.darHeight = s.height;
        init.maxEncodeWidth = s.width;
        init.maxEncodeHeight = s.height;
        init.frameRateNum = static_cast<uint32_t>(std::lround(std::max(s.rate, 1.0) * 1000.0));
        init.frameRateDen = 1000;
        init.enableEncodeAsync = 0;
        init.enablePTD = 1;
        init.encodeConfig = &config_;
        ATHENEA_TRY(check(f.nvEncInitializeEncoder(encoder_, &init), "initialisation"));

        // The input: one pitched allocation, luma then the second plane.
        size_t pitch = 0;
        ATHENEA_TRY(cuda(d_->cuMemAllocPitch(&input_, &pitch, size_t{s.width} * sample(), size_t{s.height} * 3 / 2, 16),
                         "the encoder's input"));
        pitch_ = static_cast<uint32_t>(pitch);
        format_ = s.bits > 8 ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_NV12;
        NV_ENC_REGISTER_RESOURCE reg{};
        reg.version = NV_ENC_REGISTER_RESOURCE_VER;
        reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        reg.width = s.width;
        reg.height = s.height;
        reg.pitch = pitch_;
        reg.resourceToRegister = reinterpret_cast<void*>(input_);
        reg.bufferFormat = format_;
        reg.bufferUsage = NV_ENC_INPUT_IMAGE;
        ATHENEA_TRY(check(f.nvEncRegisterResource(encoder_, &reg), "registering the input"));
        registered_ = reg.registeredResource;
        NV_ENC_CREATE_BITSTREAM_BUFFER out{};
        out.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        ATHENEA_TRY(check(f.nvEncCreateBitstreamBuffer(encoder_, &out), "the output buffer"));
        bitstream_ = out.bitstreamBuffer;
        return ok();
    }

    /// The input as it stands now through the encoder, and its unit split
    /// from Annex B into length-prefixed NAL units and parameter sets.
    Result<std::vector<EncodedUnit>> picture(uint64_t index) {
        const NV_ENCODE_API_FUNCTION_LIST& f = d_->nvenc;
        NV_ENC_MAP_INPUT_RESOURCE map{};
        map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
        map.registeredResource = registered_;
        ATHENEA_TRY(check(f.nvEncMapInputResource(encoder_, &map), "mapping the input"));
        NV_ENC_PIC_PARAMS pic{};
        pic.version = NV_ENC_PIC_PARAMS_VER;
        pic.inputWidth = settings_.width;
        pic.inputHeight = settings_.height;
        pic.inputPitch = pitch_;
        pic.inputBuffer = map.mappedResource;
        pic.outputBitstream = bitstream_;
        pic.bufferFmt = map.mappedBufferFmt;
        pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
        pic.inputTimeStamp = index;
        if (index % std::max<uint32_t>(settings_.keyInterval, 1) == 0) {
            pic.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
        }
        const NVENCSTATUS encoded = f.nvEncEncodePicture(encoder_, &pic);
        f.nvEncUnmapInputResource(encoder_, map.mappedResource);
        ATHENEA_TRY(check(encoded, "encoding a picture"));
        NV_ENC_LOCK_BITSTREAM lock{};
        lock.version = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream = bitstream_;
        ATHENEA_TRY(check(f.nvEncLockBitstream(encoder_, &lock), "reading the picture's bits"));
        EncodedUnit u;
        u.index = index;
        u.keyframe = lock.pictureType == NV_ENC_PIC_TYPE_IDR;
        split({static_cast<const uint8_t*>(lock.bitstreamBufferPtr), lock.bitstreamSizeInBytes}, u);
        f.nvEncUnlockBitstream(encoder_, bitstream_);
        std::vector<EncodedUnit> out;
        out.push_back(std::move(u));
        return out;
    }

    /// Annex B (start codes) into the unit's 4-byte big-endian lengths, the
    /// VPS, SPS and PPS out to `parameterSets` as VideoToolbox gives them.
    static void split(std::span<const uint8_t> b, EncodedUnit& u) {
        const size_t n = b.size();
        // Where each NAL unit's payload starts: after a 00 00 01.
        std::vector<size_t> starts;
        for (size_t i = 0; i + 2 < n; ++i) {
            if (b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1) {
                starts.push_back(i + 3);
                i += 2;
            }
        }
        for (size_t k = 0; k < starts.size(); ++k) {
            size_t end = k + 1 < starts.size() ? starts[k + 1] - 3 : n;
            while (end > starts[k] && b[end - 1] == 0) --end;   // a 4-byte start code's zero, trailing zeros
            if (end <= starts[k]) continue;
            const std::span<const uint8_t> nal = b.subspan(starts[k], end - starts[k]);
            const uint32_t type = nalType(nal[0]);
            if (type >= 32 && type <= 34) {   // VPS, SPS, PPS
                u.parameterSets.emplace_back(nal.begin(), nal.end());
                continue;
            }
            if (type == 35) continue;   // an access unit delimiter
            const auto size = static_cast<uint32_t>(nal.size());
            const uint8_t length[4] = {static_cast<uint8_t>(size >> 24), static_cast<uint8_t>(size >> 16),
                                       static_cast<uint8_t>(size >> 8), static_cast<uint8_t>(size)};
            u.unit.insert(u.unit.end(), length, length + 4);
            u.unit.insert(u.unit.end(), nal.begin(), nal.end());
        }
    }

    Driver*                 d_ = nullptr;
    EncoderSettings         settings_;
    CUcontext               context_ = nullptr;
    CUstream                stream_ = nullptr;
    CUdevice                cudaDevice_ = 0;
    bool                    primary_ = false;
    bool                    takesDevice_ = false;
    bool                    greyed_ = false;
    void*                   encoder_ = nullptr;
    NV_ENC_CONFIG           config_{};
    CUdeviceptr             input_ = 0;
    uint32_t                pitch_ = 0;
    NV_ENC_BUFFER_FORMAT    format_ = NV_ENC_BUFFER_FORMAT_NV12;
    NV_ENC_REGISTERED_PTR   registered_ = nullptr;
    NV_ENC_OUTPUT_PTR       bitstream_ = nullptr;
};

}   // namespace

Result<std::unique_ptr<VideoEncoder>> createHardwareEncoder(const EncoderSettings& settings) {
    return NvEncoder::create(settings);
}

}   // namespace athenea::video
