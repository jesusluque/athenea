// Copyright (c) 2026 jesus luque.
#include "athenea/ui/ViewerPanels.h"

#include <utility>

namespace athenea::ui {
namespace {

Control toggle(std::string id, std::string label, bool& value, std::string note = {}) {
    Control control;
    control.kind = Control::Kind::Toggle;
    control.id = std::move(id);
    control.label = std::move(label);
    control.note = std::move(note);
    control.flag = [&value] { return value; };
    control.setFlag = [&value](bool set) { value = set; };
    return control;
}

Control slider(std::string id, std::string label, double& value, double minimum, double maximum,
               int decimals = 2, std::string unit = {}) {
    Control control;
    control.kind = Control::Kind::Slider;
    control.id = std::move(id);
    control.label = std::move(label);
    control.minimum = minimum;
    control.maximum = maximum;
    control.decimals = decimals;
    control.unit = std::move(unit);
    control.number = [&value] { return value; };
    control.setNumber = [&value](double set) { value = set; };
    return control;
}

Control stepper(std::string id, std::string label, int& value, int minimum, int maximum) {
    Control control;
    control.kind = Control::Kind::Stepper;
    control.id = std::move(id);
    control.label = std::move(label);
    control.minimum = minimum;
    control.maximum = maximum;
    control.decimals = 0;
    control.number = [&value] { return double(value); };
    control.setNumber = [&value](double set) { value = int(set); };
    return control;
}

Control choice(std::string id, std::string label, std::string& value, const std::vector<Choice>& choices) {
    Control control;
    control.kind = Control::Kind::Choice;
    control.id = std::move(id);
    control.label = std::move(label);
    control.choices = choices;
    control.chosen = [&value] { return value; };
    control.choose = [&value](const std::string& set) { value = set; };
    return control;
}

Control action(std::string id, std::string label, std::function<void()> what) {
    Control control;
    control.kind = Control::Kind::Action;
    control.id = std::move(id);
    control.label = std::move(label);
    control.act = std::move(what);
    return control;
}

Control reading(std::string id, std::string label, std::function<std::string()> what) {
    Control control;
    control.kind = Control::Kind::Reading;
    control.id = std::move(id);
    control.label = std::move(label);
    control.reading = std::move(what);
    return control;
}

/// A control that is there only when `when` says so; an empty function is
/// always.
Control only(Control control, std::function<bool()> when) {
    control.shown = std::move(when);
    return control;
}

}   // namespace

std::vector<Panel> viewerPanels(ViewerSettings& settings, ViewerFacts facts, ViewerActions actions) {
    ViewerSettings* s = &settings;
    const auto tracing = [s] { return s->technique == "rt"; };

    Panel view;
    view.id = "view";
    view.title = "View";

    Section camera;
    camera.id = "camera";
    camera.title = "Camera";
    camera.controls.push_back(slider("focal", "Focal length", s->focal, 8.0, 200.0, 0, "mm"));
    camera.controls.push_back(slider("fstop", "Aperture", s->fStop, 0.0, 22.0, 1, "f"));
    camera.controls.push_back(
        only(slider("focus", "Focus distance", s->focusDistance, 0.01, 100.0, 2), [s] { return s->fStop > 0.0; }));
    if (actions.frameAll) {
        camera.controls.push_back(action("frame", "Frame the stage", actions.frameAll));
    }
    view.sections.push_back(std::move(camera));

    Section route;
    route.id = "route";
    route.title = "How it is drawn";
    route.controls.push_back(choice("technique", "Technique", s->technique, techniques()));
    route.controls.push_back(choice("visibility", "Visibility", s->visibility, visibilityRoutes()));
    route.controls.push_back(choice("aov", "Output", s->aov, aovs()));
    route.controls.push_back(slider("scale", "Render scale", s->renderScale, 0.25, 1.0, 2));
    view.sections.push_back(std::move(route));

    Section light;
    light.id = "light";
    light.title = "Light";
    if (facts.stageHasLights) {
        auto hasLights = facts.stageHasLights;
        light.controls.push_back(only(toggle("defaultLights", "Default lights", s->defaultLights,
                                             "a sky and a sun, for a stage that authors none"),
                                      [hasLights] { return !hasLights(); }));
    }
    light.controls.push_back(toggle("cloudShadows", "Cloud shadows", s->cloudShadows));
    light.controls.push_back(
        only(slider("shadowDensity", "Shadow density", s->shadowDensity, 0.0, 4.0, 2),
             [s] { return s->cloudShadows; }));
    view.sections.push_back(std::move(light));

    Section paths;
    paths.id = "paths";
    paths.title = "Path tracing";
    paths.controls.push_back(only(stepper("samples", "Paths a frame", s->pathSamples, 1, 64), tracing));
    paths.controls.push_back(only(stepper("bounces", "Bounces", s->pathBounces, 0, 16), tracing));
    paths.controls.push_back(only(toggle("denoise", "Denoise", s->denoise), tracing));
    if (facts.paths) {
        paths.controls.push_back(only(reading("accumulated", "Accumulated", facts.paths), tracing));
    }
    view.sections.push_back(std::move(paths));

    Section display;
    display.id = "display";
    display.title = "Display";
    {
        Control transform = choice("viewTransform", "View transform", s->viewTransform, viewTransforms());
        if (facts.ocioAvailable) {
            // OpenColorIO is only an entry where the build has it.
            auto ocio = facts.ocioAvailable;
            transform.choices.clear();
            for (const Choice& entry : viewTransforms()) {
                if (entry.value != "ocio" || ocio()) {
                    transform.choices.push_back(entry);
                }
            }
        }
        display.controls.push_back(std::move(transform));

        Control encoding = choice("displayEncoding", "Display", s->displayEncoding, displayEncodings());
        if (facts.extendedRange) {
            auto edr = facts.extendedRange;
            encoding.choices.clear();
            for (const Choice& entry : displayEncodings()) {
                if (entry.value != "linearP3" || edr()) {
                    encoding.choices.push_back(entry);
                }
            }
        }
        display.controls.push_back(std::move(encoding));
    }
    display.controls.push_back(slider("exposure", "Exposure", s->exposure, -8.0, 8.0, 1, "stops"));
    view.sections.push_back(std::move(display));

    // What the stage offers to pick between: an animation's clips, a level
    // of detail, a shirt. The engine does not know which, so every set is a
    // menu, read once here -- the panels are built again for another stage.
    if (facts.variantSets) {
        Section variants;
        variants.id = "variants";
        variants.title = "Variants";
        auto sets = facts.variantSets;
        auto choose = actions.chooseVariant;
        for (const VariantSet& set : sets()) {
            Control menu;
            menu.kind = Control::Kind::Choice;
            menu.id = "variant:" + set.prim + "{" + set.name + "}";
            menu.label = set.name;
            menu.note = set.variants.size() > 1 ? std::to_string(set.variants.size()) + " on " + set.prim
                                                 : set.prim;
            for (const std::string& variant : set.variants) {
                menu.choices.push_back({variant, variant});
            }
            const std::string prim = set.prim;
            const std::string name = set.name;
            // Read, not copied: the selection is the stage's, and a variant
            // set may be switched by something other than this menu.
            menu.chosen = [sets, prim, name] {
                for (const VariantSet& now : sets()) {
                    if (now.prim == prim && now.name == name) {
                        return now.selected;
                    }
                }
                return std::string();
            };
            menu.choose = [choose, prim, name](const std::string& variant) {
                if (choose) {
                    choose(prim, name, variant);
                }
            };
            variants.controls.push_back(std::move(menu));
        }
        if (!variants.controls.empty()) {
            view.sections.push_back(std::move(variants));
        }
    }

    // THE SKY, AS A LIST TO WALK: a menu a dome, holding every image on
    // offer, the first entry putting the stage's own opinion back; and the
    // turn, since an environment is only right once it is turned. What
    // `athenea view --hdri` offers, said once for both front ends.
    if (facts.domes) {
        Section sky;
        sky.id = "sky";
        sky.title = "Sky";
        auto domes = facts.domes;
        const std::vector<std::string> skies = facts.skies ? facts.skies() : std::vector<std::string>();
        const std::vector<Dome> now = domes();
        for (const Dome& dome : now) {
            const std::string prim = dome.prim;
            const auto current = [domes, prim] {
                for (const Dome& d : domes()) {
                    if (d.prim == prim) {
                        return d;
                    }
                }
                return Dome{};
            };
            Control menu;
            menu.kind = Control::Kind::Choice;
            menu.id = "sky:" + prim;
            menu.label = now.size() == 1 ? std::string("Environment") : dome.name;
            menu.choices.push_back({"", "The stage's own"});
            for (const std::string& path : skies) {
                const size_t slash = path.find_last_of('/');
                std::string stem = slash == std::string::npos ? path : path.substr(slash + 1);
                if (const size_t dot = stem.find_last_of('.'); dot != std::string::npos) {
                    stem.resize(dot);
                }
                menu.choices.push_back({path, stem});
            }
            menu.chosen = [current] { return current().texture; };
            auto set = actions.setDomeTexture;
            menu.choose = [set, prim](const std::string& texture) {
                if (set) {
                    set(prim, texture);
                }
            };
            sky.controls.push_back(std::move(menu));

            Control turn;
            turn.kind = Control::Kind::Slider;
            turn.id = "turn:" + prim;
            turn.label = "Turn";
            turn.minimum = -180.0;
            turn.maximum = 180.0;
            turn.decimals = 0;
            turn.unit = "deg";
            turn.number = [current] { return double(current().rotation); };
            auto rotate = actions.setDomeRotation;
            turn.setNumber = [rotate, prim](double degrees) {
                if (rotate) {
                    rotate(prim, float(degrees));
                }
            };
            sky.controls.push_back(std::move(turn));
        }
        if (!now.empty() && actions.chooseSkyFolder) {
            Control folder = action("skyFolder", skies.empty() ? "Choose a folder of skies" : "Another folder of skies",
                                    actions.chooseSkyFolder);
            sky.controls.push_back(std::move(folder));
        }
        if (!sky.controls.empty()) {
            view.sections.push_back(std::move(sky));
        }
    }

    Section time;
    time.id = "time";
    time.title = "Time";
    if (facts.timeRange) {
        auto range = facts.timeRange;
        const auto animated = [range] {
            const auto [first, last] = range();
            return last > first;
        };
        // The range is the selected clip's, which moves when another is
        // picked, so the slider asks for it rather than holding numbers.
        Control at = slider("time", "Time", s->time, 0.0, 1.0, 0, "");
        at.range = range;
        at.shown = animated;
        time.controls.push_back(std::move(at));
        time.controls.push_back(only(toggle("playing", "Play", s->playing), animated));
        time.controls.push_back(only(toggle("everyFrame", "Every frame", s->everyFrame,
                                            "one time code a drawn frame, not the stage's rate"),
                                     animated));
        if (actions.toStart) {
            time.controls.push_back(only(action("toStart", "Back to the start", actions.toStart), animated));
        }
        if (actions.stepBack) {
            time.controls.push_back(only(action("stepBack", "Previous frame", actions.stepBack), animated));
        }
        if (actions.stepForward) {
            time.controls.push_back(only(action("stepForward", "Next frame", actions.stepForward), animated));
        }
    }
    time.controls.push_back(slider("shutter", "Shutter", s->shutter, 0.0, 1.0, 2, "frames"));
    view.sections.push_back(std::move(time));

    Section frame;
    frame.id = "frame";
    frame.title = "This frame";
    if (facts.device) {
        frame.controls.push_back(reading("device", "Device", facts.device));
    }
    if (facts.frameTiming) {
        frame.controls.push_back(reading("timing", "Frame", facts.frameTiming));
    }
    if (facts.sceneCounts) {
        frame.controls.push_back(reading("counts", "Scene", facts.sceneCounts));
    }
    if (facts.status) {
        auto status = facts.status;
        frame.controls.push_back(
            only(reading("status", "Last error", status), [status] { return !status().empty(); }));
    }
    if (actions.reloadStage) {
        frame.controls.push_back(action("reload", "Open another stage", actions.reloadStage));
    }
    view.sections.push_back(std::move(frame));

    return {std::move(view)};
}

}   // namespace athenea::ui
