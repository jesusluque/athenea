// Copyright (c) 2026 jesus luque.
//
// The Apple side of Codec.h: VideoToolbox's HEVC encoder and decoder, both
// required to run on the hardware block, and CVMetalTextureCache for the
// decoded frame's luma plane as a Metal texture over the same IOSurface,
// adopted by slang-rhi (createTextureFromNativeHandle). Full-range
// bi-planar 4:2:0 both ways ('420f', or 'xf20' at 10 bits), so neither side
// converts a sample. The CPU copies samples into the encoder's buffer and
// fills chroma with mid grey; nothing else touches a value.
#include "athenea/video/Codec.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>

#import <CoreFoundation/CoreFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <VideoToolbox/VideoToolbox.h>

#include "athenea/gpu/Device.h"

namespace athenea::video {
namespace {

OSType pixelFormatFor(uint32_t bits) {
    return bits > 8 ? kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
                    : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
}

void setNumber(CFMutableDictionaryRef d, CFStringRef key, int32_t v) {
    CFNumberRef n = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &v);
    CFDictionarySetValue(d, key, n);
    CFRelease(n);
}

CFMutableDictionaryRef newDictionary() {
    return CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
                                     &kCFTypeDictionaryValueCallBacks);
}

/// Buffer attributes both sides ask for: the format, IOSurface-backed, and
/// readable by Metal.
CFMutableDictionaryRef newBufferAttributes(uint32_t bits, uint32_t width, uint32_t height) {
    CFMutableDictionaryRef a = newDictionary();
    setNumber(a, kCVPixelBufferPixelFormatTypeKey, static_cast<int32_t>(pixelFormatFor(bits)));
    if (width != 0) {
        setNumber(a, kCVPixelBufferWidthKey, static_cast<int32_t>(width));
        setNumber(a, kCVPixelBufferHeightKey, static_cast<int32_t>(height));
    }
    CFMutableDictionaryRef surface = newDictionary();
    CFDictionarySetValue(a, kCVPixelBufferIOSurfacePropertiesKey, surface);
    CFRelease(surface);
    CFDictionarySetValue(a, kCVPixelBufferMetalCompatibilityKey, kCFBooleanTrue);
    return a;
}

// --- encoder -------------------------------------------------------------------

class VtEncoder final : public VideoEncoder {
public:
    ~VtEncoder() override {
        if (session_ != nullptr) {
            VTCompressionSessionInvalidate(session_);
            CFRelease(session_);
        }
    }

    static Result<std::unique_ptr<VideoEncoder>> create(const EncoderSettings& s) {
        std::unique_ptr<VtEncoder> e(new VtEncoder());
        e->settings_ = s;
        CFMutableDictionaryRef spec = newDictionary();
        CFDictionarySetValue(spec, kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder, kCFBooleanTrue);
        CFDictionarySetValue(spec, kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,
                             kCFBooleanTrue);
        CFMutableDictionaryRef source = newBufferAttributes(s.bits, s.width, s.height);
        const OSStatus made = VTCompressionSessionCreate(
            kCFAllocatorDefault, static_cast<int32_t>(s.width), static_cast<int32_t>(s.height), kCMVideoCodecType_HEVC,
            spec, source, kCFAllocatorDefault, &VtEncoder::output, e.get(), &e->session_);
        CFRelease(spec);
        CFRelease(source);
        if (made != noErr || e->session_ == nullptr) {
            return Error::make(ErrorCode::Unsupported,
                               "video: the hardware HEVC encoder refused {} x {} at {} bits (VideoToolbox {})",
                               s.width, s.height, s.bits, static_cast<int>(made));
        }
        VTCompressionSessionRef v = e->session_;
        VTSessionSetProperty(v, kVTCompressionPropertyKey_RealTime, s.realTime ? kCFBooleanTrue : kCFBooleanFalse);
        VTSessionSetProperty(v, kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse);
        VTSessionSetProperty(v, kVTCompressionPropertyKey_ProfileLevel,
                             s.bits > 8 ? kVTProfileLevel_HEVC_Main10_AutoLevel : kVTProfileLevel_HEVC_Main_AutoLevel);
        const int32_t key = static_cast<int32_t>(std::max<uint32_t>(s.keyInterval, 1));
        CFNumberRef keyN = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &key);
        VTSessionSetProperty(v, kVTCompressionPropertyKey_MaxKeyFrameInterval, keyN);
        CFRelease(keyN);
        const double rate = s.rate;
        CFNumberRef rateN = CFNumberCreate(kCFAllocatorDefault, kCFNumberDoubleType, &rate);
        VTSessionSetProperty(v, kVTCompressionPropertyKey_ExpectedFrameRate, rateN);
        CFRelease(rateN);
        if (s.megabits > 0.0) {
            const int64_t bps = static_cast<int64_t>(s.megabits * 1.0e6);
            CFNumberRef b = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &bps);
            VTSessionSetProperty(v, kVTCompressionPropertyKey_AverageBitRate, b);
            CFRelease(b);
        } else {
            const float q = static_cast<float>(std::clamp(s.quality, 0.0, 1.0));
            CFNumberRef qN = CFNumberCreate(kCFAllocatorDefault, kCFNumberFloat32Type, &q);
            const OSStatus set = VTSessionSetProperty(v, kVTCompressionPropertyKey_Quality, qN);
            CFRelease(qN);
            if (set != noErr) {
                return Error::make(ErrorCode::Unsupported, "video: the HEVC encoder takes no constant quality ({})",
                                   static_cast<int>(set));
            }
        }
        CFBooleanRef hw = nullptr;
        if (VTSessionCopyProperty(v, kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder,
                                  kCFAllocatorDefault, &hw) == noErr && hw != nullptr) {
            e->hardware_ = CFBooleanGetValue(hw);
            CFRelease(hw);
        }
        VTCompressionSessionPrepareToEncodeFrames(v);
        return std::unique_ptr<VideoEncoder>(std::move(e));
    }

    Result<std::vector<EncodedUnit>> encode(std::span<const std::byte> luma, std::span<const std::byte> chroma,
                                            uint64_t index) override {
        const EncoderSettings& s = settings_;
        const size_t sample = s.bits > 8 ? 2 : 1;
        if (luma.size() != size_t{s.width} * s.height * sample) {
            return Error::make(ErrorCode::InvalidArgument, "video: {} bytes of luma for {} x {}", luma.size(), s.width,
                               s.height);
        }
        CVPixelBufferPoolRef pool = VTCompressionSessionGetPixelBufferPool(session_);
        CVPixelBufferRef pb = nullptr;
        if (pool == nullptr || CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb) != kCVReturnSuccess) {
            return Error(ErrorCode::DeviceFailure, "video: the encoder gave no buffer to fill");
        }
        CVPixelBufferLockBaseAddress(pb, 0);
        auto* y = static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(pb, 0));
        const size_t yStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
        const size_t row = size_t{s.width} * sample;
        for (uint32_t r = 0; r < s.height; ++r) {
            std::memcpy(y + r * yStride, luma.data() + r * row, row);
        }
        // The second plane as the planes laid it (4:2:0 given, so the
        // encoder's own downsampling never decides), or mid grey.
        auto* uv = static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(pb, 1));
        const size_t uvStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
        const size_t uvRows = CVPixelBufferGetHeightOfPlane(pb, 1);
        if (!chroma.empty()) {
            if (sample != 1 || chroma.size() != size_t{s.width} * (s.height / 2) || uvRows < s.height / 2) {
                CVPixelBufferUnlockBaseAddress(pb, 0);
                CVPixelBufferRelease(pb);
                return Error::make(ErrorCode::InvalidArgument, "video: {} bytes of chroma for {} x {}", chroma.size(),
                                   s.width, s.height);
            }
            for (uint32_t r = 0; r < s.height / 2; ++r) {
                std::memcpy(uv + r * uvStride, chroma.data() + size_t{r} * s.width, s.width);
            }
        } else if (sample == 1) {
            std::memset(uv, 128, uvStride * uvRows);
        } else {
            auto* words = reinterpret_cast<uint16_t*>(uv);
            std::fill(words, words + uvStride * uvRows / 2, uint16_t{0x8000});
        }
        CVPixelBufferUnlockBaseAddress(pb, 0);
        const int32_t scale = 1000;
        const CMTime pts = CMTimeMake(static_cast<int64_t>(index) * scale, static_cast<int32_t>(s.rate * scale));
        const CMTime duration = CMTimeMake(scale, static_cast<int32_t>(s.rate * scale));
        CFMutableDictionaryRef props = nullptr;
        if (index % std::max<uint32_t>(s.keyInterval, 1) == 0) {
            props = newDictionary();
            CFDictionarySetValue(props, kVTEncodeFrameOptionKey_ForceKeyFrame, kCFBooleanTrue);
        }
        const OSStatus st = VTCompressionSessionEncodeFrame(session_, pb, pts, duration, props,
                                                            reinterpret_cast<void*>(static_cast<uintptr_t>(index)),
                                                            nullptr);
        if (props != nullptr) CFRelease(props);
        CVPixelBufferRelease(pb);
        if (st != noErr) {
            return Error::make(ErrorCode::DeviceFailure, "video: frame {} was not encoded ({})", index,
                               static_cast<int>(st));
        }
        // Each frame out before the next goes in: the producer keeps no
        // frame's side data waiting, and a frame's bytes are its own.
        VTCompressionSessionCompleteFrames(session_, pts);
        return take();
    }

    Result<std::vector<EncodedUnit>> finish() override {
        VTCompressionSessionCompleteFrames(session_, kCMTimeInvalid);
        return take();
    }

    bool hardware() const noexcept override { return hardware_; }

private:
    VtEncoder() = default;

    Result<std::vector<EncodedUnit>> take() {
        std::lock_guard lock(mutex_);
        if (failed_ != noErr) {
            return Error::make(ErrorCode::DeviceFailure, "video: the encoder failed ({})", static_cast<int>(failed_));
        }
        std::vector<EncodedUnit> out;
        out.swap(done_);
        return out;
    }

    static void output(void* self, void* frame, OSStatus status, VTEncodeInfoFlags /*flags*/,
                       CMSampleBufferRef sample) {
        auto* e = static_cast<VtEncoder*>(self);
        std::lock_guard lock(e->mutex_);
        if (status != noErr || sample == nullptr) {
            e->failed_ = status != noErr ? status : -1;
            return;
        }
        EncodedUnit u;
        u.index = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(frame));
        u.keyframe = true;
        CFArrayRef attachments = CMSampleBufferGetSampleAttachmentsArray(sample, false);
        if (attachments != nullptr && CFArrayGetCount(attachments) > 0) {
            auto d = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(attachments, 0));
            CFBooleanRef notSync = nullptr;
            if (CFDictionaryGetValueIfPresent(d, kCMSampleAttachmentKey_NotSync,
                                              reinterpret_cast<const void**>(&notSync))) {
                u.keyframe = !CFBooleanGetValue(notSync);
            }
        }
        CMBlockBufferRef block = CMSampleBufferGetDataBuffer(sample);
        const size_t n = CMBlockBufferGetDataLength(block);
        u.unit.resize(n);
        CMBlockBufferCopyDataBytes(block, 0, n, u.unit.data());
        if (u.keyframe) {
            CMFormatDescriptionRef desc = CMSampleBufferGetFormatDescription(sample);
            size_t count = 0;
            int nalHeader = 0;
            CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(desc, 0, nullptr, nullptr, &count, &nalHeader);
            for (size_t k = 0; k < count; ++k) {
                const uint8_t* p = nullptr;
                size_t size = 0;
                if (CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(desc, k, &p, &size, nullptr, nullptr) ==
                    noErr) {
                    u.parameterSets.emplace_back(p, p + size);
                }
            }
        }
        e->done_.push_back(std::move(u));
    }

    EncoderSettings          settings_;
    VTCompressionSessionRef  session_ = nullptr;
    bool                     hardware_ = false;
    std::mutex               mutex_;
    std::vector<EncodedUnit> done_;
    OSStatus                 failed_ = noErr;
};

// --- decoder -------------------------------------------------------------------

class VtFrame final : public DecodedFrame {
public:
    VtFrame(CVPixelBufferRef pb, uint32_t bits) : pb_(pb), bits_(bits) {}
    ~VtFrame() override { CVPixelBufferRelease(pb_); }
    uint32_t width() const noexcept override { return static_cast<uint32_t>(CVPixelBufferGetWidthOfPlane(pb_, 0)); }
    uint32_t height() const noexcept override { return static_cast<uint32_t>(CVPixelBufferGetHeightOfPlane(pb_, 0)); }
    uint32_t bits() const noexcept override { return bits_; }
    CVPixelBufferRef buffer() const noexcept { return pb_; }

private:
    CVPixelBufferRef pb_;
    uint32_t         bits_;
};

class VtDecoder final : public VideoDecoder {
public:
    explicit VtDecoder(uint32_t bits) : bits_(bits) {}
    ~VtDecoder() override { reset(); }

    Result<std::shared_ptr<DecodedFrame>> decode(std::span<const uint8_t> unit,
                                                 const std::vector<std::vector<uint8_t>>& sets,
                                                 bool keyframe) override {
        if (keyframe && !sets.empty() && sets != sets_) {
            ATHENEA_TRY(open(sets));
        }
        if (session_ == nullptr) {
            return Error(ErrorCode::InvalidArgument, "video: a frame before any key frame with parameter sets");
        }
        CMBlockBufferRef block = nullptr;
        if (CMBlockBufferCreateWithMemoryBlock(kCFAllocatorDefault, const_cast<uint8_t*>(unit.data()), unit.size(),
                                               kCFAllocatorNull, nullptr, 0, unit.size(), 0, &block) != noErr) {
            return Error(ErrorCode::DeviceFailure, "video: no block for an access unit");
        }
        CMSampleBufferRef sample = nullptr;
        const size_t size = unit.size();
        const OSStatus made = CMSampleBufferCreateReady(kCFAllocatorDefault, block, format_, 1, 0, nullptr, 1, &size,
                                                        &sample);
        CFRelease(block);
        if (made != noErr) {
            return Error::make(ErrorCode::DeviceFailure, "video: no sample for an access unit ({})",
                               static_cast<int>(made));
        }
        __block CVImageBufferRef image = nullptr;
        __block OSStatus result = noErr;
        VTDecodeInfoFlags info = 0;
        const OSStatus st = VTDecompressionSessionDecodeFrameWithOutputHandler(
            session_, sample, 0, &info,
            ^(OSStatus status, VTDecodeInfoFlags, CVImageBufferRef buffer, CMTime, CMTime) {
              result = status;
              if (status == noErr && buffer != nullptr) {
                  image = CVPixelBufferRetain(buffer);
              }
            });
        VTDecompressionSessionWaitForAsynchronousFrames(session_);
        CFRelease(sample);
        if (st != noErr || result != noErr || image == nullptr) {
            if (image != nullptr) CVPixelBufferRelease(image);
            return Error::make(ErrorCode::DeviceFailure, "video: the decoder refused an access unit ({}, {})",
                               static_cast<int>(st), static_cast<int>(result));
        }
        return std::shared_ptr<DecodedFrame>(std::make_shared<VtFrame>(image, bits_));
    }

    bool hardware() const noexcept override { return hardware_; }

private:
    void reset() {
        if (session_ != nullptr) {
            VTDecompressionSessionInvalidate(session_);
            CFRelease(session_);
            session_ = nullptr;
        }
        if (format_ != nullptr) {
            CFRelease(format_);
            format_ = nullptr;
        }
    }

    Result<void> open(const std::vector<std::vector<uint8_t>>& sets) {
        reset();
        std::vector<const uint8_t*> ptrs;
        std::vector<size_t> sizes;
        for (const auto& s : sets) {
            ptrs.push_back(s.data());
            sizes.push_back(s.size());
        }
        OSStatus st = CMVideoFormatDescriptionCreateFromHEVCParameterSets(kCFAllocatorDefault, ptrs.size(), ptrs.data(),
                                                                          sizes.data(), 4, nullptr, &format_);
        if (st != noErr) {
            return Error::make(ErrorCode::InvalidArgument, "video: parameter sets the decoder cannot read ({})",
                               static_cast<int>(st));
        }
        // The planes are full range, and this decoder asks for full range
        // out: a stream that says video range would come back stretched
        // from 16-235, every sample skewed, so it is refused.
        const auto full = static_cast<CFBooleanRef>(
            CMFormatDescriptionGetExtension(format_, kCMFormatDescriptionExtension_FullRangeVideo));
        if (full == nullptr || !CFBooleanGetValue(full)) {
            reset();
            return Error(ErrorCode::InvalidArgument,
                         "video: the stream's SPS says video range (16-235); video planes are full range");
        }
        CFMutableDictionaryRef spec = newDictionary();
        CFDictionarySetValue(spec, kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
                             kCFBooleanTrue);
        CFMutableDictionaryRef out = newBufferAttributes(bits_, 0, 0);
        st = VTDecompressionSessionCreate(kCFAllocatorDefault, format_, spec, out, nullptr, &session_);
        CFRelease(spec);
        CFRelease(out);
        if (st != noErr || session_ == nullptr) {
            session_ = nullptr;
            return Error::make(ErrorCode::Unsupported, "video: no hardware HEVC decoder for this stream ({})",
                               static_cast<int>(st));
        }
        CFBooleanRef hw = nullptr;
        if (VTSessionCopyProperty(session_, kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
                                  kCFAllocatorDefault, &hw) == noErr && hw != nullptr) {
            hardware_ = CFBooleanGetValue(hw);
            CFRelease(hw);
        }
        sets_ = sets;
        return ok();
    }

    uint32_t                          bits_;
    VTDecompressionSessionRef         session_ = nullptr;
    CMVideoFormatDescriptionRef       format_ = nullptr;
    std::vector<std::vector<uint8_t>> sets_;
    bool                              hardware_ = false;
};

// --- the luma plane as a Metal texture -------------------------------------------

class VtLumaTextures final : public LumaTextures {
public:
    ~VtLumaTextures() override {
        if (cache_ != nullptr) CFRelease(cache_);
    }

    static Result<std::unique_ptr<LumaTextures>> create(gpu::Device& device) {
        rhi::DeviceNativeHandles handles;
        if (SLANG_FAILED(device.rhi()->getNativeDeviceHandles(&handles)) ||
            handles.handles[0].type != rhi::NativeHandleType::MTLDevice) {
            return Error(ErrorCode::Unsupported, "video: decoded frames are read as textures by a Metal device only");
        }
        std::unique_ptr<VtLumaTextures> t(new VtLumaTextures());
        t->device_ = &device;
        id<MTLDevice> mtl = (__bridge id<MTLDevice>)reinterpret_cast<void*>(handles.handles[0].value);
        if (CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr, mtl, nullptr, &t->cache_) != kCVReturnSuccess) {
            return Error(ErrorCode::DeviceFailure, "video: no Metal texture cache");
        }
        return std::unique_ptr<LumaTextures>(std::move(t));
    }

    Result<LumaTexture> wrap(const DecodedFrame& frame) override {
        const auto* f = dynamic_cast<const VtFrame*>(&frame);
        if (f == nullptr) {
            return Error(ErrorCode::InvalidArgument, "video: a frame VideoToolbox did not decode");
        }
        const bool wide = f->bits() > 8;
        CVMetalTextureRef cv = nullptr;
        const size_t w = f->width(), h = f->height();
        if (CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache_, f->buffer(), nullptr,
                                                      wide ? MTLPixelFormatR16Unorm : MTLPixelFormatR8Unorm, w, h, 0,
                                                      &cv) != kCVReturnSuccess ||
            cv == nullptr) {
            return Error(ErrorCode::DeviceFailure, "video: the decoded luma plane is not a Metal texture");
        }
        id<MTLTexture> mtl = CVMetalTextureGetTexture(cv);
        rhi::TextureDesc desc;
        desc.type = rhi::TextureType::Texture2D;
        desc.size = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
        desc.mipCount = 1;
        desc.arrayLength = 1;
        desc.format = wide ? rhi::Format::R16Unorm : rhi::Format::R8Unorm;
        desc.usage = rhi::TextureUsage::ShaderResource;
        desc.label = "video.luma";
        rhi::NativeHandle handle;
        handle.type = rhi::NativeHandleType::MTLTexture;
        handle.value = reinterpret_cast<uint64_t>((__bridge void*)mtl);
        LumaTexture out;
        if (SLANG_FAILED(device_->rhi()->createTextureFromNativeHandle(handle, desc, out.texture.writeRef()))) {
            CFRelease(cv);
            return Error(ErrorCode::DeviceFailure, "video: slang-rhi did not adopt the luma texture");
        }
        // The CVMetalTexture holds the IOSurface the texture reads; the frame
        // holds the pixel buffer. Both live as long as the texture is used.
        auto pixels = std::shared_ptr<const void>(CVPixelBufferRetain(f->buffer()),
                                                  [](const void* p) { CVPixelBufferRelease(static_cast<CVPixelBufferRef>(const_cast<void*>(p))); });
        // The second plane, (Cb, Cr) a 2 x 2 block, as RG8Unorm over the same
        // memory: what a frame with its colour in chroma reads (8 bits).
        CVMetalTextureRef cvChroma = nullptr;
        if (!wide && CVPixelBufferGetPlaneCount(f->buffer()) > 1 &&
            CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache_, f->buffer(), nullptr,
                                                      MTLPixelFormatRG8Unorm, w / 2, h / 2, 1,
                                                      &cvChroma) == kCVReturnSuccess &&
            cvChroma != nullptr) {
            id<MTLTexture> mtlChroma = CVMetalTextureGetTexture(cvChroma);
            rhi::TextureDesc cd = desc;
            cd.size = {static_cast<uint32_t>(w / 2), static_cast<uint32_t>(h / 2), 1};
            cd.format = rhi::Format::RG8Unorm;
            cd.label = "video.chroma";
            rhi::NativeHandle ch;
            ch.type = rhi::NativeHandleType::MTLTexture;
            ch.value = reinterpret_cast<uint64_t>((__bridge void*)mtlChroma);
            if (SLANG_FAILED(device_->rhi()->createTextureFromNativeHandle(ch, cd, out.chroma.writeRef()))) {
                out.chroma = nullptr;
            }
        }
        out.keep = std::shared_ptr<void>(cv, [pixels, cvChroma](void* p) {
            CFRelease(static_cast<CFTypeRef>(p));
            if (cvChroma != nullptr) CFRelease(cvChroma);
        });
        if (++wrapped_ % 64 == 0) {
            CVMetalTextureCacheFlush(cache_, 0);
        }
        return out;
    }

private:
    VtLumaTextures() = default;
    gpu::Device*           device_ = nullptr;
    CVMetalTextureCacheRef cache_ = nullptr;
    uint64_t               wrapped_ = 0;
};

}   // namespace

Result<std::unique_ptr<VideoEncoder>> createHardwareEncoder(const EncoderSettings& settings) {
    return VtEncoder::create(settings);
}

Result<std::unique_ptr<VideoDecoder>> createHardwareDecoder(uint32_t codec, uint32_t bits) {
    if (codec != 0x31637668) {   // 'hvc1'
        return Error::make(ErrorCode::Unsupported, "video: codec {:#x} is not HEVC", codec);
    }
    return std::unique_ptr<VideoDecoder>(std::make_unique<VtDecoder>(bits));
}

Result<std::unique_ptr<LumaTextures>> createLumaTextures(gpu::Device& device) {
    return VtLumaTextures::create(device);
}

}   // namespace athenea::video
