// Copyright (c) 2026 jesus luque.
//
// The Apple side of Codec.h: VideoToolbox's HEVC encoder, required to run on
// the hardware block, fed full-range bi-planar 4:2:0 ('420f', or 'xf20' at
// 10 bits), so it converts no sample. The CPU copies samples into the
// encoder's buffer and fills chroma with mid grey; nothing else touches a
// value.
#include "athenea/video/Codec.h"

#include <algorithm>
#include <cstring>
#include <mutex>

#import <CoreFoundation/CoreFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <VideoToolbox/VideoToolbox.h>

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

    Result<std::vector<EncodedUnit>> encode(std::span<const std::byte> luma, uint64_t index) override {
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
        // Chroma at mid grey: no data rides there.
        auto* uv = static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(pb, 1));
        const size_t uvStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
        const size_t uvRows = CVPixelBufferGetHeightOfPlane(pb, 1);
        if (sample == 1) {
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

}   // namespace

Result<std::unique_ptr<VideoEncoder>> createHardwareEncoder(const EncoderSettings& settings) {
    return VtEncoder::create(settings);
}

}   // namespace athenea::video
