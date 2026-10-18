// Copyright (c) 2026 jesus luque.
//
// "Athenea 3DGS": a gaussian splat cloud drawn by athenea's rasteriser, as an
// OpenFX generator a compositor (Nuke) puts in its graph.
//
// The node draws one cloud through one camera at the project's format and the
// frame's time, as linear premultiplied RGBA floats -- what a comp merges --
// with nothing of the engine in the host's process (RenderServer.h says why).
//
// THE CAMERA. A comp's camera is a node with a translate, a rotate in some
// order, a focal length and an aperture; the knobs here are those, so a
// Camera node's are linked to them by expression and the cloud moves with the
// plate. The matrix is built as Nuke builds its own: scale, then rotation in
// the order chosen (ZXY by default, Nuke's), then translation; looking down
// -z with +y up.
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include "RenderServer.h"
#include "ofxsImageEffect.h"

namespace {

constexpr const char* kPluginId = "uk.co.lucab.athenea.splats";
constexpr const char* kCloud = "cloud";
constexpr const char* kTranslate = "translate";
constexpr const char* kRotate = "rotate";
constexpr const char* kRotOrder = "rot_order";
constexpr const char* kFocal = "focal";
constexpr const char* kHaperture = "haperture";
constexpr const char* kVaperture = "vaperture";
constexpr const char* kNear = "near";
constexpr const char* kFar = "far";
constexpr const char* kTimeOffset = "time_offset";
constexpr const char* kOrders[] = {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"};

using Mat = std::array<double, 16>;

Mat multiply(const Mat& a, const Mat& b) {
    Mat out{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) sum += a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = sum;
        }
    }
    return out;
}

Mat rotationAbout(char axis, double degrees) {
    const double t = degrees * M_PI / 180.0;
    const double c = std::cos(t);
    const double s = std::sin(t);
    switch (axis) {
        case 'X': return {1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1};
        case 'Y': return {c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1};
        default: return {c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    }
}

/// Camera to world, as Nuke's Camera builds it: the rotations applied in the
/// order named (the first letter first), then the translation.
Mat cameraMatrix(const double translate[3], const double rotate[3], int order) {
    const char* letters = kOrders[order];
    Mat rotation{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (int k = 0; k < 3; ++k) {
        const char axis = letters[k];
        const double angle = rotate[axis - 'X'];
        rotation = multiply(rotationAbout(axis, angle), rotation);   // later rotations act after
    }
    rotation[3] = translate[0];
    rotation[7] = translate[1];
    rotation[11] = translate[2];
    return rotation;
}

class SplatEffect : public OFX::ImageEffect {
public:
    explicit SplatEffect(OfxImageEffectHandle handle) : ImageEffect(handle) {
        output_ = fetchClip(kOfxImageEffectOutputClipName);
        cloud_ = fetchStringParam(kCloud);
        translate_ = fetchDouble3DParam(kTranslate);
        rotate_ = fetchDouble3DParam(kRotate);
        order_ = fetchChoiceParam(kRotOrder);
        focal_ = fetchDoubleParam(kFocal);
        haperture_ = fetchDoubleParam(kHaperture);
        vaperture_ = fetchDoubleParam(kVaperture);
        near_ = fetchDoubleParam(kNear);
        far_ = fetchDoubleParam(kFar);
        timeOffset_ = fetchDoubleParam(kTimeOffset);
    }

    void getClipPreferences(OFX::ClipPreferencesSetter& preferences) override {
        preferences.setOutputFrameVarying(true);
        preferences.setOutputPremultiplication(OFX::eImagePreMultiplied);
    }

    /// The project's format: a generator has no input to take one from.
    bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod) override {
        const OfxPointD size = getProjectSize();
        const OfxPointD offset = getProjectOffset();
        rod.x1 = offset.x;
        rod.y1 = offset.y;
        rod.x2 = offset.x + size.x;
        rod.y2 = offset.y + size.y;
        (void)args;
        return true;
    }

    void render(const OFX::RenderArguments& args) override {
        std::unique_ptr<OFX::Image> dst(output_->fetchImage(args.time));
        if (!dst) OFX::throwSuiteStatusException(kOfxStatFailed);
        if (dst->getPixelDepth() != OFX::eBitDepthFloat || dst->getPixelComponents() != OFX::ePixelComponentRGBA) {
            OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
        }
        const OfxRectI bounds = dst->getRegionOfDefinition();
        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        clear(*dst, args.renderWindow);

        std::string cloud;
        cloud_->getValueAtTime(args.time, cloud);
        if (cloud.empty() || width <= 0 || height <= 0) {
            return;   // nothing chosen yet: a transparent frame
        }
        athenea::ofx::FrameRequest request;
        request.cloud = cloud;
        request.width = static_cast<uint32_t>(width);
        request.height = static_cast<uint32_t>(height);
        request.time = args.time + timeOffset_->getValueAtTime(args.time);
        double t[3];
        double r[3];
        translate_->getValueAtTime(args.time, t[0], t[1], t[2]);
        rotate_->getValueAtTime(args.time, r[0], r[1], r[2]);
        int order = 4;
        order_->getValueAtTime(args.time, order);
        request.camera.cameraToWorld = cameraMatrix(t, r, order);
        request.camera.focal = focal_->getValueAtTime(args.time);
        request.camera.haperture = haperture_->getValueAtTime(args.time);
        // The vertical aperture the format's own aspect gives, pixel aspect
        // counted, unless one is set: Nuke fits the horizontal one.
        const double asked = vaperture_->getValueAtTime(args.time);
        request.camera.vaperture =
            asked > 0.0 ? asked : request.camera.haperture * height / (width * getProjectPixelAspectRatio());
        request.camera.nearZ = near_->getValueAtTime(args.time);
        request.camera.farZ = far_->getValueAtTime(args.time);

        athenea::ofx::Frame frame;
        std::string error;
        {
            // One frame at a time to one server: the host may call render
            // from several threads, and the server answers one call at once.
            std::lock_guard<std::mutex> hold(lock_);
            if (!server_) server_ = std::make_unique<athenea::ofx::RenderServer>();
            if (!server_->render(request, frame, error)) {
                sendMessage(OFX::Message::eMessageError, "", "Athenea 3DGS: " + error);
                OFX::throwSuiteStatusException(kOfxStatFailed);
            }
        }
        // Both bottom row first, so rows copy as they are.
        const OfxRectI& window = args.renderWindow;
        for (int y = window.y1; y < window.y2; ++y) {
            if (abort()) return;
            auto* row = static_cast<float*>(dst->getPixelAddress(window.x1, y));
            if (row == nullptr) continue;
            const size_t from = (size_t(y - bounds.y1) * size_t(width) + size_t(window.x1 - bounds.x1)) * 4;
            std::memcpy(row, frame.rgba.data() + from, size_t(window.x2 - window.x1) * 4 * sizeof(float));
        }
    }

private:
    static void clear(OFX::Image& image, const OfxRectI& window) {
        for (int y = window.y1; y < window.y2; ++y) {
            if (void* row = image.getPixelAddress(window.x1, y)) {
                std::memset(row, 0, size_t(window.x2 - window.x1) * 4 * sizeof(float));
            }
        }
    }

    OFX::Clip*         output_ = nullptr;
    OFX::StringParam*  cloud_ = nullptr;
    OFX::Double3DParam* translate_ = nullptr;
    OFX::Double3DParam* rotate_ = nullptr;
    OFX::ChoiceParam*  order_ = nullptr;
    OFX::DoubleParam*  focal_ = nullptr;
    OFX::DoubleParam*  haperture_ = nullptr;
    OFX::DoubleParam*  vaperture_ = nullptr;
    OFX::DoubleParam*  near_ = nullptr;
    OFX::DoubleParam*  far_ = nullptr;
    OFX::DoubleParam*  timeOffset_ = nullptr;
    std::mutex         lock_;
    std::unique_ptr<athenea::ofx::RenderServer> server_;
};

OFX::DoubleParamDescriptor* number(OFX::ImageEffectDescriptor& desc, OFX::PageParamDescriptor& page,
                                   const char* name, const char* label, const char* hint, double value,
                                   double low, double high) {
    OFX::DoubleParamDescriptor* param = desc.defineDoubleParam(name);
    param->setLabels(label, label, label);
    param->setScriptName(name);
    param->setHint(hint);
    param->setDefault(value);
    param->setDisplayRange(low, high);
    param->setAnimates(true);
    page.addChild(*param);
    return param;
}

}   // namespace

mDeclarePluginFactory(SplatEffectFactory, {}, {});

void SplatEffectFactory::describe(OFX::ImageEffectDescriptor& desc) {
    desc.setLabels("Athenea 3DGS", "Athenea 3DGS", "Athenea 3DGS");
    desc.setPluginGrouping("Athenea");
    desc.setPluginDescription(
        "A gaussian splat cloud drawn by athenea's rasteriser through the node's camera, at the project's "
        "format and the frame's time: linear premultiplied RGBA. The engine runs in a process of its own "
        "(athenea-mcp), so nothing of it is loaded into the host.");
    desc.addSupportedContext(OFX::eContextGenerator);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(false);
    desc.setSupportsTiles(false);   // a whole frame a render: the server draws whole frames
    desc.setTemporalClipAccess(false);
    desc.setRenderThreadSafety(OFX::eRenderInstanceSafe);
}

void SplatEffectFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context) {
    if (context == OFX::eContextGeneral) {
        // General wants a source; it is optional and unused (a format comes
        // from the project), there for hosts that offer no generator context.
        OFX::ClipDescriptor* source = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
        source->addSupportedComponent(OFX::ePixelComponentRGBA);
        source->setOptional(true);
    }
    OFX::ClipDescriptor* output = desc.defineClip(kOfxImageEffectOutputClipName);
    output->addSupportedComponent(OFX::ePixelComponentRGBA);
    output->setSupportsTiles(false);

    OFX::PageParamDescriptor* page = desc.definePageParam("Athenea");

    OFX::StringParamDescriptor* cloud = desc.defineStringParam(kCloud);
    cloud->setLabels("Cloud", "Cloud", "Cloud");
    cloud->setScriptName(kCloud);
    cloud->setHint("The gaussian cloud: a .usdc or .usda holding a ParticleField (what athenea exports).");
    cloud->setStringType(OFX::eStringTypeFilePath);
    cloud->setFilePathExists(true);
    cloud->setAnimates(false);
    page->addChild(*cloud);

    OFX::Double3DParamDescriptor* translate = desc.defineDouble3DParam(kTranslate);
    translate->setLabels("Translate", "Translate", "Translate");
    translate->setScriptName(kTranslate);
    translate->setHint("Where the camera stands, in the cloud's units (link a Camera node's translate).");
    translate->setDefault(0.0, 0.0, 5.0);
    translate->setDoubleType(OFX::eDoubleTypePlain);
    page->addChild(*translate);

    OFX::Double3DParamDescriptor* rotate = desc.defineDouble3DParam(kRotate);
    rotate->setLabels("Rotate", "Rotate", "Rotate");
    rotate->setScriptName(kRotate);
    rotate->setHint("The camera's rotation in degrees about x, y and z (link a Camera node's rotate).");
    rotate->setDefault(0.0, 0.0, 0.0);
    rotate->setDoubleType(OFX::eDoubleTypeAngle);
    page->addChild(*rotate);

    OFX::ChoiceParamDescriptor* order = desc.defineChoiceParam(kRotOrder);
    order->setLabels("Rotation order", "Rotation order", "Rotation order");
    order->setScriptName(kRotOrder);
    order->setHint("The order the rotations apply in, first letter first: ZXY is Nuke's default.");
    for (const char* name : kOrders) order->appendOption(name);
    order->setDefault(4);
    order->setAnimates(false);
    page->addChild(*order);

    number(desc, *page, kFocal, "Focal length", "mm (link a Camera node's focal)", 50.0, 5.0, 300.0);
    number(desc, *page, kHaperture, "Horizontal aperture", "mm (link a Camera node's haperture)", 24.576, 1.0, 100.0);
    number(desc, *page, kVaperture, "Vertical aperture",
           "mm; 0 takes the format's aspect, as Nuke fits a camera to its format", 0.0, 0.0, 100.0);
    number(desc, *page, kNear, "Near", "near clipping plane, scene units", 0.1, 0.001, 10.0);
    number(desc, *page, kFar, "Far", "far clipping plane, scene units", 10000.0, 1.0, 100000.0);
    number(desc, *page, kTimeOffset, "Time offset",
           "added to the frame number to give the cloud's USD time (an animated, skinned cloud)", 0.0, -1000.0,
           1000.0);
}

OFX::ImageEffect* SplatEffectFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) {
    return new SplatEffect(handle);
}

namespace OFX::Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
    static SplatEffectFactory factory(kPluginId, 0, 1);
    ids.push_back(&factory);
}
}   // namespace OFX::Plugin
