// Copyright (c) 2026 jesus luque.
//
// athenea_ofx_host: the least of an OpenFX host that renders one frame of an
// image effect and writes it as an EXR -- enough to drive Athenea.ofx.bundle
// as Nuke would, without Nuke.
//
//   athenea_ofx_host --bundle-dir <dir> [--plugin <id>] --width W --height H
//                    --frame T --out <file.exr> [name=value ...]
//
// The project is W x H, square pixels, 24 fps. The effect is made in the
// Generator context, its output clip is float RGBA premultiplied, whole
// frames, rows bottom first. A parameter holds one value at every time: there
// is no animation. A value is set as `x,y,z` for a 3D parameter, an index for
// a choice and the text itself for a string.
//
// It is built on OpenFX's host support library, which reports errors by
// exceptions; they are caught in main.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// OpenFX's host headers are not self-contained: this is the order they need.
// clang-format off
#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxPixels.h"
#include "ofxhBinary.h"
#include "ofxhPropertySuite.h"
#include "ofxhClip.h"
#include "ofxhParam.h"
#include "ofxhMemory.h"
#include "ofxhImageEffect.h"
#include "ofxhPluginAPICache.h"
#include "ofxhPluginCache.h"
#include "ofxhHost.h"
#include "ofxhImageEffectAPI.h"
// clang-format on

#include "athenea/io/Exr.h"

namespace {

namespace fx = OFX::Host::ImageEffect;
namespace param = OFX::Host::Param;

struct Project {
    int    width = 0;
    int    height = 0;
    double frame = 0.0;
};
Project gProject;

OfxStatus printMessage(const char* type, const char* format, va_list args) {
    std::fprintf(stderr, "plugin %s: ", type);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    // No one to ask: a question is answered yes.
    return std::strcmp(type, kOfxMessageQuestion) == 0 ? kOfxStatReplyYes : kOfxStatOK;
}

// A PARAMETER'S VALUE, as the descriptor's default first and the command line
// after. Numbers are kept as doubles whatever the type (an integer, a choice
// and a boolean are exact in one); a string as its text.
class Stored {
public:
    Stored(const OFX::Host::Property::Set& props, int count, bool integer, bool text) : text_(text) {
        if (text) {
            string_ = props.getStringProperty(kOfxParamPropDefault);
            return;
        }
        for (int i = 0; i < count; ++i) {
            values_.push_back(integer ? props.getIntProperty(kOfxParamPropDefault, i)
                                      : props.getDoubleProperty(kOfxParamPropDefault, i));
        }
    }
    virtual ~Stored() = default;

    bool assign(const std::string& value) {
        if (text_) {
            string_ = value;
            return true;
        }
        std::vector<double> parsed;
        std::stringstream in(value);
        for (std::string item; std::getline(in, item, ',');) {
            char* end = nullptr;
            parsed.push_back(std::strtod(item.c_str(), &end));
            if (end == item.c_str() || *end != '\0') return false;
        }
        if (parsed.size() != values_.size()) return false;
        values_ = parsed;
        return true;
    }

protected:
    bool                text_;
    std::vector<double> values_;
    std::string         string_;
};

class DoubleParam : public param::DoubleInstance, public Stored {
public:
    DoubleParam(param::Descriptor& d, param::SetInstance* set) : DoubleInstance(d, set), Stored(d.getProperties(), 1, false, false) {}
    OfxStatus get(double& v) override { v = values_[0]; return kOfxStatOK; }
    OfxStatus get(OfxTime, double& v) override { return get(v); }
    OfxStatus set(double v) override { values_[0] = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, double v) override { return set(v); }
    OfxStatus derive(OfxTime, double& v) override { v = 0.0; return kOfxStatOK; }
    OfxStatus integrate(OfxTime t1, OfxTime t2, double& v) override { v = values_[0] * (t2 - t1); return kOfxStatOK; }
};

class Double2DParam : public param::Double2DInstance, public Stored {
public:
    Double2DParam(param::Descriptor& d, param::SetInstance* set) : Double2DInstance(d, set), Stored(d.getProperties(), 2, false, false) {}
    OfxStatus get(double& x, double& y) override { x = values_[0]; y = values_[1]; return kOfxStatOK; }
    OfxStatus get(OfxTime, double& x, double& y) override { return get(x, y); }
    OfxStatus set(double x, double y) override { values_ = {x, y}; return kOfxStatOK; }
    OfxStatus set(OfxTime, double x, double y) override { return set(x, y); }
};

class Double3DParam : public param::Double3DInstance, public Stored {
public:
    Double3DParam(param::Descriptor& d, param::SetInstance* set) : Double3DInstance(d, set), Stored(d.getProperties(), 3, false, false) {}
    OfxStatus get(double& x, double& y, double& z) override { x = values_[0]; y = values_[1]; z = values_[2]; return kOfxStatOK; }
    OfxStatus get(OfxTime, double& x, double& y, double& z) override { return get(x, y, z); }
    OfxStatus set(double x, double y, double z) override { values_ = {x, y, z}; return kOfxStatOK; }
    OfxStatus set(OfxTime, double x, double y, double z) override { return set(x, y, z); }
};

class IntegerParam : public param::IntegerInstance, public Stored {
public:
    IntegerParam(param::Descriptor& d, param::SetInstance* set) : IntegerInstance(d, set), Stored(d.getProperties(), 1, true, false) {}
    OfxStatus get(int& v) override { v = static_cast<int>(values_[0]); return kOfxStatOK; }
    OfxStatus get(OfxTime, int& v) override { return get(v); }
    OfxStatus set(int v) override { values_[0] = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, int v) override { return set(v); }
};

class ChoiceParam : public param::ChoiceInstance, public Stored {
public:
    ChoiceParam(param::Descriptor& d, param::SetInstance* set) : ChoiceInstance(d, set), Stored(d.getProperties(), 1, true, false) {}
    OfxStatus get(int& v) override { v = static_cast<int>(values_[0]); return kOfxStatOK; }
    OfxStatus get(OfxTime, int& v) override { return get(v); }
    OfxStatus set(int v) override { values_[0] = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, int v) override { return set(v); }
};

class BooleanParam : public param::BooleanInstance, public Stored {
public:
    BooleanParam(param::Descriptor& d, param::SetInstance* set) : BooleanInstance(d, set), Stored(d.getProperties(), 1, true, false) {}
    OfxStatus get(bool& v) override { v = values_[0] != 0.0; return kOfxStatOK; }
    OfxStatus get(OfxTime, bool& v) override { return get(v); }
    OfxStatus set(bool v) override { values_[0] = v ? 1.0 : 0.0; return kOfxStatOK; }
    OfxStatus set(OfxTime, bool v) override { return set(v); }
};

class StringParam : public param::StringInstance, public Stored {
public:
    StringParam(param::Descriptor& d, param::SetInstance* set) : StringInstance(d, set), Stored(d.getProperties(), 1, false, true) {}
    OfxStatus get(std::string& v) override { v = string_; return kOfxStatOK; }
    OfxStatus get(OfxTime, std::string& v) override { return get(v); }
    OfxStatus set(const char* v) override { string_ = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, const char* v) override { return set(v); }
};

// THE OUTPUT CLIP: one float RGBA image of the project's size, which the
// plugin fetches and writes and main reads afterwards. Any other clip (a
// General context's optional source) is unconnected.
class Clip : public fx::ClipInstance {
public:
    Clip(fx::Instance* effect, fx::ClipDescriptor& desc) : ClipInstance(effect, desc) {}
    ~Clip() override {
        if (image_) image_->releaseReference();
    }

    const std::vector<float>& pixels() const { return pixels_; }

    const std::string& getUnmappedBitDepth() const override { static const std::string v(kOfxBitDepthFloat); return v; }
    const std::string& getUnmappedComponents() const override { static const std::string v(kOfxImageComponentRGBA); return v; }
    const std::string& getPremult() const override { static const std::string v(kOfxImagePreMultiplied); return v; }
    double getAspectRatio() const override { return 1.0; }
    double getFrameRate() const override { return 24.0; }
    void getFrameRange(double& start, double& end) const override { start = end = gProject.frame; }
    const std::string& getFieldOrder() const override { static const std::string v(kOfxImageFieldNone); return v; }
    bool getConnected() const override { return isOutput(); }
    double getUnmappedFrameRate() const override { return 24.0; }
    void getUnmappedFrameRange(double& start, double& end) const override { getFrameRange(start, end); }
    bool getContinuousSamples() const override { return false; }
    OfxRectD getRegionOfDefinition(OfxTime) const override { return {0.0, 0.0, double(gProject.width), double(gProject.height)}; }

    fx::Image* getImage(OfxTime, const OfxRectD*) override {
        if (!isOutput()) return nullptr;
        if (!image_) {
            pixels_.assign(size_t(gProject.width) * size_t(gProject.height) * 4, 0.0f);
            const OfxRectI bounds{0, 0, gProject.width, gProject.height};
            image_ = new fx::Image(*this, 1.0, 1.0, pixels_.data(), bounds, bounds,
                                   gProject.width * 4 * int(sizeof(float)), kOfxImageFieldNone, "output");
        }
        // The plugin releases what it fetched; the clip keeps its own reference.
        image_->addReference();
        return image_;
    }

private:
    std::vector<float> pixels_;
    fx::Image*         image_ = nullptr;
};

class Effect : public fx::Instance {
public:
    Effect(fx::ImageEffectPlugin* plugin, fx::Descriptor& desc, const std::string& context)
        : Instance(plugin, desc, context, false) {}

    const std::string& getDefaultOutputFielding() const override { static const std::string v(kOfxImageFieldNone); return v; }
    fx::ClipInstance* newClipInstance(fx::Instance*, fx::ClipDescriptor* desc, int) override { return new Clip(this, *desc); }

    OfxStatus vmessage(const char* type, const char*, const char* format, va_list args) override {
        return printMessage(type, format, args);
    }
    OfxStatus setPersistentMessage(const char* type, const char* id, const char* format, va_list args) override {
        return vmessage(type, id, format, args);
    }
    OfxStatus clearPersistentMessage() override { return kOfxStatOK; }

    void getProjectSize(double& x, double& y) const override { x = gProject.width; y = gProject.height; }
    void getProjectOffset(double& x, double& y) const override { x = y = 0.0; }
    void getProjectExtent(double& x, double& y) const override { getProjectSize(x, y); }
    double getProjectPixelAspectRatio() const override { return 1.0; }
    double getEffectDuration() const override { return 1.0; }
    double getFrameRate() const override { return 24.0; }
    double getFrameRecursive() const override { return gProject.frame; }
    void getRenderScaleRecursive(double& x, double& y) const override { x = y = 1.0; }

    param::Instance* newParam(const std::string&, param::Descriptor& d) override {
        const std::string& type = d.getType();
        if (type == kOfxParamTypeDouble) return new DoubleParam(d, this);
        if (type == kOfxParamTypeDouble2D) return new Double2DParam(d, this);
        if (type == kOfxParamTypeDouble3D) return new Double3DParam(d, this);
        if (type == kOfxParamTypeInteger) return new IntegerParam(d, this);
        if (type == kOfxParamTypeChoice) return new ChoiceParam(d, this);
        if (type == kOfxParamTypeBoolean) return new BooleanParam(d, this);
        if (type == kOfxParamTypeString) return new StringParam(d, this);
        if (type == kOfxParamTypePushButton) return new param::PushbuttonInstance(d, this);
        if (type == kOfxParamTypeGroup) return new param::GroupInstance(d, this);
        if (type == kOfxParamTypePage) return new param::PageInstance(d, this);
        std::fprintf(stderr, "athenea_ofx_host: parameter type %s is not supported\n", type.c_str());
        return nullptr;
    }
    OfxStatus editBegin(const std::string&) override { return kOfxStatOK; }
    OfxStatus editEnd() override { return kOfxStatOK; }

    void progressStart(const std::string&, const std::string&) override {}
    void progressEnd() override {}
    bool progressUpdate(double) override { return true; }
    double timeLineGetTime() override { return gProject.frame; }
    void timeLineGotoTime(double) override {}
    void timeLineGetBounds(double& t1, double& t2) override { t1 = t2 = gProject.frame; }
};

class Host : public fx::Host {
public:
    Host() {
        _properties.setStringProperty(kOfxPropName, "uk.co.lucab.athenea.ofxtesthost");
        _properties.setStringProperty(kOfxPropLabel, "athenea OFX test host");
        _properties.setIntProperty(kOfxImageEffectHostPropIsBackground, 1);
        _properties.setIntProperty(kOfxImageEffectPropSupportsOverlays, 0);
        _properties.setIntProperty(kOfxImageEffectPropSupportsMultiResolution, 0);
        _properties.setIntProperty(kOfxImageEffectPropSupportsTiles, 0);
        _properties.setIntProperty(kOfxImageEffectPropTemporalClipAccess, 0);
        _properties.setStringProperty(kOfxImageEffectPropSupportedComponents, kOfxImageComponentRGBA, 0);
        _properties.setStringProperty(kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextGenerator, 0);
        _properties.setStringProperty(kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextGeneral, 1);
        _properties.setIntProperty(kOfxImageEffectPropSupportsMultipleClipDepths, 0);
        _properties.setIntProperty(kOfxImageEffectPropSupportsMultipleClipPARs, 0);
        _properties.setIntProperty(kOfxParamHostPropSupportsCustomInteract, 0);
        _properties.setIntProperty(kOfxParamHostPropMaxParameters, -1);
        _properties.setIntProperty(kOfxParamHostPropMaxPages, 0);
    }

    fx::Instance* newInstance(void*, fx::ImageEffectPlugin* plugin, fx::Descriptor& desc, const std::string& context) override {
        return new Effect(plugin, desc, context);
    }
    fx::Descriptor* makeDescriptor(fx::ImageEffectPlugin* plugin) override { return new fx::Descriptor(plugin); }
    fx::Descriptor* makeDescriptor(const fx::Descriptor& root, fx::ImageEffectPlugin* plugin) override {
        return new fx::Descriptor(root, plugin);
    }
    fx::Descriptor* makeDescriptor(const std::string& bundlePath, fx::ImageEffectPlugin* plugin) override {
        return new fx::Descriptor(bundlePath, plugin);
    }
    OfxStatus vmessage(const char* type, const char*, const char* format, va_list args) override {
        return printMessage(type, format, args);
    }
    OfxStatus setPersistentMessage(const char* type, const char* id, const char* format, va_list args) override {
        return vmessage(type, id, format, args);
    }
    OfxStatus clearPersistentMessage() override { return kOfxStatOK; }
};

bool failed(const char* action, OfxStatus status) {
    if (status == kOfxStatOK || status == kOfxStatReplyDefault) return false;
    std::fprintf(stderr, "athenea_ofx_host: %s returned %d\n", action, status);
    return true;
}

int run(int argc, char** argv) {
    std::string bundleDir;
    std::string pluginId = "uk.co.lucab.athenea.splats";
    std::string out;
    std::vector<std::string> overrides;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--bundle-dir" && hasValue) bundleDir = argv[++i];
        else if (arg == "--plugin" && hasValue) pluginId = argv[++i];
        else if (arg == "--width" && hasValue) gProject.width = std::atoi(argv[++i]);
        else if (arg == "--height" && hasValue) gProject.height = std::atoi(argv[++i]);
        else if (arg == "--frame" && hasValue) gProject.frame = std::atof(argv[++i]);
        else if (arg == "--out" && hasValue) out = argv[++i];
        else if (arg.find('=') != std::string::npos && arg.rfind("--", 0) != 0) overrides.push_back(arg);
        else {
            std::fprintf(stderr, "athenea_ofx_host: unexpected argument '%s'\n", arg.c_str());
            return 2;
        }
    }
    if (bundleDir.empty() || out.empty() || gProject.width <= 0 || gProject.height <= 0) {
        std::fprintf(stderr, "usage: athenea_ofx_host --bundle-dir <dir> [--plugin <id>] --width W --height H "
                             "--frame T --out <file.exr> [name=value ...]\n");
        return 2;
    }

    // The plugins are scanned where asked and the cache is never read or
    // written, so a run leaves no file behind and sees the bundle as built.
    Host host;
    OFX::Host::PluginCache* cache = OFX::Host::PluginCache::getPluginCache();
    cache->setCacheVersion("athenea_ofx_host");
    cache->prependFileToPath(std::filesystem::absolute(bundleDir).string());
    fx::PluginCache effects(host);
    effects.registerInCache(*cache);
    // The cache goes before the host it was registered with, and after the
    // instance.
    struct Unload {
        ~Unload() { OFX::Host::PluginCache::clearPluginCache(); }
    } unload;
    cache->scanPluginFiles();

    fx::ImageEffectPlugin* plugin = effects.getPluginById(pluginId);
    if (plugin == nullptr) {
        std::fprintf(stderr, "athenea_ofx_host: no plugin %s under %s\n", pluginId.c_str(), bundleDir.c_str());
        return 1;
    }
    std::unique_ptr<fx::Instance> instance(plugin->createInstance(kOfxImageEffectContextGenerator, nullptr));
    if (!instance) {
        std::fprintf(stderr, "athenea_ofx_host: %s would not make an instance in the Generator context\n",
                     pluginId.c_str());
        return 1;
    }
    for (const std::string& assignment : overrides) {
        const size_t equals = assignment.find('=');
        const std::string name = assignment.substr(0, equals);
        auto* stored = dynamic_cast<Stored*>(instance->getParam(name));
        if (stored == nullptr || !stored->assign(assignment.substr(equals + 1))) {
            std::fprintf(stderr, "athenea_ofx_host: cannot set %s\n", assignment.c_str());
            return 2;
        }
    }

    if (failed("createInstanceAction", instance->createInstanceAction())) return 1;
    if (!instance->getClipPreferences()) {
        std::fprintf(stderr, "athenea_ofx_host: getClipPreferences failed\n");
        return 1;
    }
    const OfxPointD scale{1.0, 1.0};
    const OfxRectI window{0, 0, gProject.width, gProject.height};
    const double frame = gProject.frame;
    if (failed("beginRenderAction", instance->beginRenderAction(frame, frame, 1.0, false, scale, true, false))) return 1;
    const OfxStatus rendered = instance->renderAction(frame, kOfxImageFieldNone, window, scale, true, false, false);
    const OfxStatus ended = instance->endRenderAction(frame, frame, 1.0, false, scale, true, false);
    if (failed("renderAction", rendered) || failed("endRenderAction", ended)) return 1;

    auto* output = dynamic_cast<Clip*>(instance->getClip(kOfxImageEffectOutputClipName));
    if (output == nullptr || output->pixels().empty()) {
        std::fprintf(stderr, "athenea_ofx_host: the plugin never fetched its output image\n");
        return 1;
    }
    // Both OpenFX and the writer take rows bottom first.
    if (auto written = athenea::io::writeExr(out, uint32_t(gProject.width), uint32_t(gProject.height),
                                             output->pixels(), {}, false);
        !written) {
        std::fprintf(stderr, "athenea_ofx_host: %s\n", written.error().message().c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d)\n", out.c_str(), gProject.width, gProject.height);
    return 0;
}

}   // namespace

int main(int argc, char** argv) {
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "athenea_ofx_host: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "athenea_ofx_host: OpenFX host support threw\n");
    }
    return status;
}
