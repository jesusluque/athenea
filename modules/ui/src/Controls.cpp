// Copyright (c) 2026 jesus luque.
#include "athenea/ui/Controls.h"

namespace athenea::ui {

const std::vector<Choice>& techniques() {
    // What the delegate's athenea:technique takes, and nothing else: "rt" is the
    // traced route, which is the one that path traces (StageRenderer.h).
    static const std::vector<Choice> list{{"raster", "Raster"}, {"rt", "Path traced"}};
    return list;
}

const std::vector<Choice>& visibilityRoutes() {
    static const std::vector<Choice> list{{"automatic", "Automatic"},
                                          {"raster", "Raster"},
                                          {"rays", "Rays"},
                                          {"bvh", "Compute BVH"}};
    return list;
}

const std::vector<Choice>& aovs() {
    // The matte's three layers are written as they are, so a compositor can
    // take them; the first entry is what a frame shows unless asked otherwise.
    static const std::vector<Choice> list{{"color", "Colour"},
                                          {"depth", "Depth"},
                                          {"primId", "Prim id"},
                                          {"instanceId", "Instance id"},
                                          {"elementId", "Element id"},
                                          {"Neye", "Eye normal"},
                                          {"normal", "World normal"},
                                          {"cryptomatte", "Cryptomatte"},
                                          {"CryptoObject00", "CryptoObject00"},
                                          {"CryptoObject01", "CryptoObject01"},
                                          {"CryptoObject02", "CryptoObject02"}};
    return list;
}

const std::vector<Choice>& viewTransforms() {
    static const std::vector<Choice> list{
        {"standard", "Standard"}, {"agx", "AgX"}, {"aces2", "ACES 2.0"}, {"ocio", "OpenColorIO"}};
    return list;
}

const std::vector<Choice>& displayEncodings() {
    static const std::vector<Choice> list{
        {"srgb", "sRGB"}, {"rec709", "BT.1886"}, {"p3", "Display P3"}, {"linearP3", "Linear P3 (EDR)"}};
    return list;
}

std::string labelOf(const std::vector<Choice>& choices, const std::string& value) {
    for (const Choice& choice : choices) {
        if (choice.value == value) {
            return choice.label;
        }
    }
    return value;
}

}   // namespace athenea::ui
