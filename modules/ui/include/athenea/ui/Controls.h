// Copyright (c) 2026 jesus luque.
//
// What a viewer's panels are, said once.
//
// `athenea view` draws its panels with Dear ImGui and the iOS app draws the same
// ones with UIKit, and neither is where a menu's entries should live: a route
// added to the renderer has to appear in both, and a table written twice is a
// table that disagrees with itself. So the panels are described here -- what
// the controls are, what they range over, what a menu's entries are, and how
// to read and write the value -- and each front end walks the description and
// draws it the way its platform draws things.
//
// Nothing here knows about a device, a stage or a window: a control carries
// the two functions that reach its value, and whoever builds the description
// closes over whatever it lives in.
#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace athenea::ui {

/// An entry of a menu: what the engine calls it, and what a person reads.
struct Choice {
    std::string value;
    std::string label;
};

/// One row of a panel.
struct Control {
    enum class Kind {
        Toggle,    ///< a switch
        Slider,    ///< a number over a range
        Stepper,   ///< a whole number over a small range
        Choice,    ///< one of `choices`
        Action,    ///< a button that does something
        Reading,   ///< a number or a line the frame wrote; not editable
    };

    Kind        kind = Kind::Reading;
    std::string id;      ///< stable, so a front end can key its rows by it
    std::string label;
    std::string note;    ///< a quieter line under the label, or empty

    // Slider and Stepper.
    double      minimum = 0.0;
    double      maximum = 1.0;
    bool        logarithmic = false;
    int         decimals = 2;
    std::string unit;    ///< "mm", "stops", "deg": shown after the number

    // Choice.
    std::vector<Choice> choices;

    // Where the value lives. Only the pair its kind uses is set.
    std::function<double()>                 number;
    std::function<void(double)>             setNumber;
    std::function<bool()>                   flag;
    std::function<void(bool)>               setFlag;
    std::function<std::string()>            chosen;   ///< a value, never an index
    std::function<void(const std::string&)> choose;
    std::function<void()>                   act;
    std::function<std::string()>            reading;

    /// A range that moves -- a timeline whose clip was switched. Set, it is
    /// what the control ranges over instead of `minimum` and `maximum`.
    std::function<std::pair<double, double>()> range;

    /// When a control only applies sometimes -- a path tracer's bounces under
    /// the raster route, a focus distance at a pinhole. Empty means always.
    std::function<bool()> shown;
    /// Shown but not editable, and said to be so.
    std::function<bool()> enabled;

    [[nodiscard]] bool isShown() const { return !shown || shown(); }
    [[nodiscard]] std::pair<double, double> bounds() const {
        return range ? range() : std::pair<double, double>{minimum, maximum};
    }
    [[nodiscard]] bool isEnabled() const { return !enabled || enabled(); }
};

/// Controls that belong together, under a heading.
struct Section {
    std::string          id;
    std::string          title;
    std::vector<Control> controls;

    [[nodiscard]] bool isShown() const {
        for (const Control& control : controls) {
            if (control.isShown()) {
                return true;
            }
        }
        return false;
    }
};

/// A panel: a window on the desktop, a screen or a sidebar on a tablet.
struct Panel {
    std::string          id;
    std::string          title;
    std::vector<Section> sections;
};

// --- the tables the menus are made of -------------------------------------------
//
// One list per menu, here rather than in a front end: `athenea view` and the iOS
// app show the same entries because they read the same list.

[[nodiscard]] const std::vector<Choice>& techniques();     ///< athenea:technique
[[nodiscard]] const std::vector<Choice>& visibilityRoutes();   ///< athenea:visibility
[[nodiscard]] const std::vector<Choice>& aovs();               ///< what a frame can show
[[nodiscard]] const std::vector<Choice>& viewTransforms();     ///< Standard, AgX, ACES 2.0, OCIO
[[nodiscard]] const std::vector<Choice>& displayEncodings();   ///< sRGB, BT.1886, Display P3, linear P3

/// The label for `value` in `choices`, or the value itself where the list has
/// no entry for it.
[[nodiscard]] std::string labelOf(const std::vector<Choice>& choices, const std::string& value);

}   // namespace athenea::ui
