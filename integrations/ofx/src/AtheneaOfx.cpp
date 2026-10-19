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
//
// THE LAYERS. Beside the colour a frame carries its depth (view z), P (where
// each pixel lies in the world) and the Cryptomatte's three layers. A host
// that takes planes (Natron, through OpenFX's multi-plane extension) gets
// them all as layers of the one output; one that does not (Nuke takes only
// its own planes from an OpenFX node) shows the layer the `output` knob picks
// in RGBA. Only the layers the host asks for are drawn and read back.
//
// THE SKY. A relightable cloud (a transfer) is lit by the stage's domes: the
// `sky` knob puts another image on all of them, turned by `sky_rotation`, and
// `sky_visible` off keeps the background transparent for a comp while the sky
// still lights.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "RenderServer.h"
#include "nuke/fnOfxExtensions.h"
#include "ofxNatron.h"
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
constexpr const char* kSky = "sky";
constexpr const char* kSkyRotation = "sky_rotation";
constexpr const char* kSkyVisible = "sky_visible";
constexpr const char* kOutput = "output";
constexpr const char* kOrders[] = {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"};

/// A LAYER OF THE FRAME: its name as the server knows it ("" for the two the
/// frame always carries), as a host shows it, and its channels.
struct Layer {
    const char* server;
    const char* name;
    const char* channels[4];
    int         count;
};
constexpr Layer kLayers[] = {
    {"", "rgba", {"R", "G", "B", "A"}, 4},
    {"", "depth", {"Z", nullptr, nullptr, nullptr}, 1},
    {"P", "P", {"X", "Y", "Z", "A"}, 4},
    {"CryptoObject00", "CryptoObject00", {"R", "G", "B", "A"}, 4},
    {"CryptoObject01", "CryptoObject01", {"R", "G", "B", "A"}, 4},
    {"CryptoObject02", "CryptoObject02", {"R", "G", "B", "A"}, 4},
};
constexpr int kLayerCount = static_cast<int>(sizeof(kLayers) / sizeof(kLayers[0]));
constexpr int kDepthLayer = 1;

/// A layer as the multi-plane extension names one: the colour is Foundry's
/// plane, the others Natron's custom planes -- the plane, its label, each
/// channel (ofxNatron.h).
std::string planeOf(const Layer& layer) {
    if (std::strcmp(layer.name, "rgba") == 0) return kFnOfxImagePlaneColour;
    std::string plane =
        std::string(kNatronOfxImageComponentsPlaneName) + layer.name + kNatronOfxImageComponentsPlaneLabel + layer.name;
    for (int c = 0; c < layer.count; ++c) {
        plane += std::string(kNatronOfxImageComponentsPlaneChannel) + layer.channels[c];
    }
    return plane;
}

/// Which of kLayers a plane the host asks for is, or -1.
int layerOfPlane(const std::string& plane) {
    for (int k = 0; k < kLayerCount; ++k) {
        if (plane == planeOf(kLayers[k])) return k;
    }
    return -1;
}

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
        sky_ = fetchStringParam(kSky);
        skyRotation_ = fetchDoubleParam(kSkyRotation);
        skyVisible_ = fetchBooleanParam(kSkyVisible);
        outputLayer_ = fetchChoiceParam(kOutput);
    }

    /// The planes this node makes, for a host that takes planes: all of them.
    OfxStatus getClipComponents(const OFX::ClipComponentsArguments&, OFX::ClipComponentsSetter& components) override {
        for (const Layer& layer : kLayers) {
            components.addClipPlane(*output_, planeOf(layer));
        }
        components.setPassThroughClip(nullptr, 0.0, 0);
        return kOfxStatOK;
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
        // The layers to fill and the images to fill them in: every plane the
        // host asked for, or, from a host without planes, the one image whose
        // RGBA shows the layer the `output` knob picks.
        struct Target {
            std::unique_ptr<OFX::Image> image;
            int                         layer = 0;
        };
        std::vector<Target> targets;
        if (OFX::getImageEffectHostDescription()->isMultiPlanar && !args.planes.empty()) {
            for (const std::string& plane : args.planes) {
                const int layer = layerOfPlane(plane);
                if (layer < 0) continue;
                Target target;
                target.image.reset(output_->fetchImagePlane(args.time, args.renderView, plane.c_str()));
                target.layer = layer;
                if (target.image) targets.push_back(std::move(target));
            }
        } else {
            Target target;
            target.image.reset(output_->fetchImage(args.time));
            outputLayer_->getValueAtTime(args.time, target.layer);
            if (target.image) targets.push_back(std::move(target));
        }
        if (targets.empty()) OFX::throwSuiteStatusException(kOfxStatFailed);
        const OfxRectI bounds = targets.front().image->getRegionOfDefinition();
        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        for (Target& target : targets) {
            if (target.image->getPixelDepth() != OFX::eBitDepthFloat) {
                OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
            }
            clear(*target.image, args.renderWindow);
        }

        std::string cloud;
        cloud_->getValueAtTime(args.time, cloud);
        if (cloud.empty() || width <= 0 || height <= 0) {
            return;   // nothing chosen yet: transparent frames
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
        sky_->getValueAtTime(args.time, request.sky);
        request.skyRotation = skyRotation_->getValueAtTime(args.time);
        request.skyVisible = skyVisible_->getValueAtTime(args.time);
        for (const Target& target : targets) {
            const std::string server = kLayers[target.layer].server;
            if (!server.empty() && std::find(request.layers.begin(), request.layers.end(), server) == request.layers.end()) {
                request.layers.push_back(server);
            }
        }

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
        const OfxRectI& window = args.renderWindow;
        for (Target& target : targets) {
            const Layer& layer = kLayers[target.layer];
            const float* source = layer.server[0] != '\0'     ? frame.layers[layer.server].data()
                                  : target.layer == kDepthLayer ? frame.depth.data()
                                                                : frame.rgba.data();
            const size_t from_channels = target.layer == kDepthLayer ? 1 : 4;
            const size_t channels = size_t(target.image->getPixelComponentCount());
            const size_t count = size_t(window.x2 - window.x1);
            for (int y = window.y1; y < window.y2; ++y) {
                if (abort()) return;
                auto* row = static_cast<float*>(target.image->getPixelAddress(window.x1, y));
                if (row == nullptr) continue;
                // Both bottom row first, so rows copy as they are.
                const size_t from = size_t(y - bounds.y1) * size_t(width) + size_t(window.x1 - bounds.x1);
                if (channels == from_channels) {
                    std::memcpy(row, source + from * channels, count * channels * sizeof(float));
                    continue;
                }
                // The depth shown in RGBA, by a host without planes: its one
                // value copied into every channel.
                for (size_t x = 0; x < count; ++x) {
                    for (size_t c = 0; c < channels; ++c) {
                        row[x * channels + c] = source[(from + x) * from_channels + std::min(c, from_channels - 1)];
                    }
                }
            }
        }
    }

private:
    static void clear(OFX::Image& image, const OfxRectI& window) {
        const size_t channels = size_t(image.getPixelComponentCount());
        for (int y = window.y1; y < window.y2; ++y) {
            if (void* row = image.getPixelAddress(window.x1, y)) {
                std::memset(row, 0, size_t(window.x2 - window.x1) * channels * sizeof(float));
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
    OFX::StringParam*  sky_ = nullptr;
    OFX::DoubleParam*  skyRotation_ = nullptr;
    OFX::BooleanParam* skyVisible_ = nullptr;
    OFX::ChoiceParam*  outputLayer_ = nullptr;
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
    // Every layer at once where the host takes planes: one draw makes them all.
    if (OFX::getImageEffectHostDescription()->isMultiPlanar) {
        desc.setIsMultiPlanar(true);
        desc.setPassThroughForNotProcessedPlanes(OFX::ePassThroughLevelBlockAllNonRenderedPlanes);
        desc.setRenderAllPlanes(true);
    }
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

    OFX::StringParamDescriptor* sky = desc.defineStringParam(kSky);
    sky->setLabels("Sky", "Sky", "Sky");
    sky->setScriptName(kSky);
    sky->setHint("An HDR image (lat-long .hdr or .exr) on every dome of the stage: a relightable (transfer) cloud is "
                 "lit by it. Empty keeps the stage's own sky; a stage with no dome gets one.");
    sky->setStringType(OFX::eStringTypeFilePath);
    sky->setFilePathExists(true);
    sky->setAnimates(false);
    page->addChild(*sky);
    number(desc, *page, kSkyRotation, "Sky rotation", "degrees the sky turns about the stage's up axis", 0.0, -180.0,
           180.0);
    OFX::BooleanParamDescriptor* visible = desc.defineBooleanParam(kSkyVisible);
    visible->setLabels("Sky visible", "Sky visible", "Sky visible");
    visible->setScriptName(kSkyVisible);
    visible->setHint("Off: the background is transparent, to merge over a plate; the sky still lights the cloud.");
    visible->setDefault(true);
    visible->setAnimates(false);
    page->addChild(*visible);

    OFX::ChoiceParamDescriptor* layerChoice = desc.defineChoiceParam(kOutput);
    layerChoice->setLabels("Output", "Output", "Output");
    layerChoice->setScriptName(kOutput);
    layerChoice->setHint("The layer shown in RGBA by a host that takes no planes (Nuke): rgba, depth (view z in every "
                         "channel), P (world position, alpha 1 where drawn) or a Cryptomatte layer. A host that takes "
                         "planes (Natron) gets every layer as a plane of its own, whatever this says.");
    for (const Layer& layer : kLayers) layerChoice->appendOption(layer.name);
    layerChoice->setDefault(0);
    layerChoice->setAnimates(false);
    page->addChild(*layerChoice);
}

OFX::ImageEffect* SplatEffectFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) {
    return new SplatEffect(handle);
}

static SplatEffectFactory gFactory(kPluginId, 0, 2);
mRegisterPluginFactoryInstance(gFactory)
