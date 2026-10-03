// Copyright (c) 2026 jesus luque.
#include "athenea/view/Viewer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>

#include <imgui.h>

#include "athenea/core/Platform.h"
#include <imgui_impl_glfw.h>

#include "athenea/core/Log.h"
#include "athenea/gpu/Texture.h"
#include "athenea/io/Exr.h"
#include "athenea/io/Png.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/technique/DisplayTransform.h"
#include "athenea/ui/GaussianPanel.h"
#include "athenea/usd/StageRenderer.h"
#include "athenea/view/ImGuiRenderer.h"
#include "athenea/view/Window.h"

namespace athenea::view {

namespace {

constexpr rhi::Format kSurfaceFormat = rhi::Format::BGRA8Unorm;   // encoded by the display transform
constexpr std::array<float, 3> kBackground{0.0F, 0.0F, 0.0F};

/// THE SKIES A PANEL CAN OFFER: every image in the directories asked for, plus
/// the folder each dome's own sky sits in.
///
/// That second half is what makes the option unnecessary on most stages: a
/// dome usually points at one file of a library of them, and the library is
/// the folder. Gathered once, when the window opens -- a panel that reads a
/// directory every frame is a panel that stutters.
std::vector<std::filesystem::path> skiesOnOffer(std::span<const std::filesystem::path> asked,
                                                std::span<const usd::StageDome> domes) {
    namespace fs = std::filesystem;
    std::vector<fs::path> folders(asked.begin(), asked.end());
    for (const usd::StageDome& dome : domes) {
        if (!dome.texture.empty()) {
            folders.push_back(fs::path(dome.texture).parent_path());
        }
    }
    std::vector<fs::path> found;
    for (const fs::path& folder : folders) {
        std::error_code failed;
        if (folder.empty() || !fs::is_directory(folder, failed)) {
            continue;
        }
        for (const fs::directory_entry& entry : fs::directory_iterator(folder, failed)) {
            std::string suffix = entry.path().extension().string();
            std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (suffix == ".hdr" || suffix == ".exr") {
                found.push_back(entry.path());
            }
        }
    }
    // By the name, not by the path: the same sky in two libraries is one
    // entry, since the combo shows a stem and two identical rows help nobody.
    // The first folder named wins, which is the order they were asked for.
    std::vector<fs::path> kept;
    for (const fs::path& sky : found) {
        const auto same = [&](const fs::path& already) {
            return already.filename() == sky.filename();
        };
        if (std::none_of(kept.begin(), kept.end(), same)) {
            kept.push_back(sky);
        }
    }
    std::sort(kept.begin(), kept.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename() < b.filename();
    });
    return kept;
}

/// The frame as shown, panels included, into a float texture, read back and
/// written: output, the one image this viewer reads. Returns how many pixels
/// the display made brighter than the background, counted by a kernel.
Result<uint64_t> snapshot(usd::StageRenderer& stage, technique::DisplayTransform& display, ImGuiRenderer& ui,
                          const technique::DisplaySettings& settings, const std::string& aov, uint32_t width,
                          uint32_t height, const std::filesystem::path& path) {
    gpu::Device& device = stage.device();
    gpu::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = rhi::Format::RGBA32Float;
    desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::RenderTarget |
                 rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopySource;
    desc.label = "view.snapshot";
    auto texture = gpu::Texture::create(device, desc);
    if (!texture) return std::move(texture).error();
    auto source = stage.displaySource(aov);
    if (!source) return std::move(source).error();
    auto view = texture->view(0);
    if (!view) return std::move(view).error();
    auto count = gpu::ComputeKernel::create(stage.library(), "athenea/view/snapshot_count", "snapshotCount");
    if (!count) return std::move(count).error();
    gpu::BufferDesc one;
    one.bytes = 4;
    one.elementBytes = 4;
    one.label = "view.snapshot.lit";
    auto lit = gpu::Buffer::create(device, one);
    if (!lit) return std::move(lit).error();
    gpu::CommandBatch batch(device);
    ATHENEA_TRY(display.run(batch, *source, settings, texture->rhi(), width, height));
    ATHENEA_TRY(ui.render(batch, ImGui::GetDrawData(), (*view).get(), rhi::Format::RGBA32Float, width, height));
    ATHENEA_TRY(batch.submit(true));
    {
        gpu::CommandBatch counting(device);
        count->dispatch(counting, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["shown"].setBinding((*view).get());
            cursor["lit"].setBinding(lit->rhi());
            cursor["params"]["width"].setData(width);
            cursor["params"]["height"].setData(height);
            cursor["params"]["threshold"].setData(0.25F);
        });
        ATHENEA_TRY(counting.submit(true));
    }
    uint32_t litPixels = 0;
    ATHENEA_TRY(lit->read(device, 0, sizeof(litPixels), &litPixels));
    auto bytes = texture->read(device, 0, 0);
    if (!bytes) return std::move(bytes).error();
    // Top row first on the device; bottom row first for the EXR writer.
    std::vector<float> rgba(size_t{width} * height * 4);
    const size_t row = size_t{width} * 4 * sizeof(float);
    for (size_t y = 0; y < height; ++y) {
        std::memcpy(rgba.data() + (height - 1 - y) * size_t{width} * 4, bytes->data() + y * row, row);
    }
    ATHENEA_TRY(io::writeExr(path, width, height, rgba, {}, false));
    return uint64_t{litPixels};
}

/// THE FRAME AS SHOWN, INTO A PNG: what a capture of a run is made of.
///
/// `snapshot` above keeps the numbers, which is what one frame is for. A run
/// is hundreds of them, and hundreds of float EXRs of a retina window is
/// tens of gigabytes: this draws the same thing into eight bits a channel --
/// the display transform has already encoded it -- and writes a PNG, panels
/// and all.
Result<void> capture(usd::StageRenderer& stage, technique::DisplayTransform& display, ImGuiRenderer& ui,
                     const technique::DisplaySettings& settings, const std::string& aov, uint32_t width,
                     uint32_t height, const std::filesystem::path& path) {
    gpu::Device& device = stage.device();
    gpu::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = rhi::Format::RGBA8Unorm;
    desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::RenderTarget |
                 rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopySource;
    desc.label = "view.capture";
    auto texture = gpu::Texture::create(device, desc);
    if (!texture) return std::move(texture).error();
    auto source = stage.displaySource(aov);
    if (!source) return std::move(source).error();
    auto view = texture->view(0);
    if (!view) return std::move(view).error();
    {
        gpu::CommandBatch batch(device);
        ATHENEA_TRY(display.run(batch, *source, settings, texture->rhi(), width, height));
        ATHENEA_TRY(ui.render(batch, ImGui::GetDrawData(), (*view).get(), rhi::Format::RGBA8Unorm, width, height));
        ATHENEA_TRY(batch.submit(true));
    }
    auto bytes = texture->read(device, 0, 0);
    if (!bytes) return std::move(bytes).error();
    return io::writePng(path, width, height,
                        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(bytes->data()),
                                                 bytes->size()));
}

/// A camera turning about a point: the free camera.
struct Orbit {
    std::array<double, 3> target{0.0, 0.0, 0.0};
    double                distance = 10.0;
    double                yaw = 0.6;
    double                pitch = 0.35;
    double                radius = 5.0;   ///< the framed scene's, for clipping planes
};

std::array<double, 3> upOf(char axis) {
    return axis == 'Z' ? std::array<double, 3>{0.0, 0.0, 1.0} : std::array<double, 3>{0.0, 1.0, 0.0};
}

render::Camera cameraOf(const Orbit& orbit, char axis, double focal, double fStop = 0.0,
                        double focus = 0.0) {
    const double cp = std::cos(orbit.pitch);
    const std::array<double, 3> away = axis == 'Z'
                                           ? std::array<double, 3>{cp * std::sin(orbit.yaw), -cp * std::cos(orbit.yaw),
                                                                   std::sin(orbit.pitch)}
                                           : std::array<double, 3>{cp * std::sin(orbit.yaw), std::sin(orbit.pitch),
                                                                   cp * std::cos(orbit.yaw)};
    const std::array<double, 3> eye{orbit.target[0] + away[0] * orbit.distance,
                                    orbit.target[1] + away[1] * orbit.distance,
                                    orbit.target[2] + away[2] * orbit.distance};
    const std::array<double, 3> up = upOf(axis);
    render::Camera camera = render::Camera::lookingAt({eye[0], eye[1], eye[2]},
                                                      {orbit.target[0], orbit.target[1], orbit.target[2]},
                                                      {up[0], up[1], up[2]});
    camera.lens.focal = focal;
    // A diaphragm, when the panel opened one: what is at `focus` stays sharp
    // and everything else spreads into its circle of confusion.
    camera.lens.fStop = fStop;
    camera.lens.focusDistance = focus > 0.0 ? focus : orbit.distance;
    camera.lens.nearZ = std::max(orbit.distance * 1e-3, 1e-4);
    camera.lens.farZ = orbit.distance + orbit.radius * 8.0 + 1.0;
    return camera;
}

void frameBounds(Orbit& orbit, const scene::Bounds& bounds, double focal) {
    for (size_t k = 0; k < 3; ++k) {
        orbit.target[k] = 0.5 * (double(bounds.min[k]) + double(bounds.max[k]));
    }
    const double dx = double(bounds.max[0]) - double(bounds.min[0]);
    const double dy = double(bounds.max[1]) - double(bounds.min[1]);
    const double dz = double(bounds.max[2]) - double(bounds.min[2]);
    orbit.radius = std::max(0.5 * std::sqrt(dx * dx + dy * dy + dz * dz), 1e-3);
    const double halfFov = std::atan(0.5 * 18.672 / focal);
    orbit.distance = orbit.radius / std::sin(halfFov) * 1.05;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct Choice {
    const char* label;
    const char* value;
};

constexpr std::array<Choice, 2> kTechniques{{{"Raster", "raster"}, {"Path traced", "rt"}}};
constexpr std::array<Choice, 4> kVisibility{
    {{"Automatic", "automatic"}, {"Raster", "raster"}, {"Rays", "rays"}, {"Compute BVH", "bvh"}}};
constexpr std::array<Choice, 11> kAovs{{{"Colour", "color"},
                                        {"Depth", "depth"},
                                        {"Prim id", "primId"},
                                        {"Instance id", "instanceId"},
                                        {"Element id", "elementId"},
                                        {"Eye normal", "Neye"},
                                        {"World normal", "normal"},
                                        // The matte: a colour a prim, then the
                                        // three layers as they are written.
                                        {"Cryptomatte", "cryptomatte"},
                                        {"CryptoObject00", "CryptoObject00"},
                                        {"CryptoObject01", "CryptoObject01"},
                                        {"CryptoObject02", "CryptoObject02"}}};

bool combo(const char* label, int& index, std::span<const Choice> choices) {
    bool changed = false;
    if (ImGui::BeginCombo(label, choices[static_cast<size_t>(index)].label)) {
        for (size_t k = 0; k < choices.size(); ++k) {
            if (ImGui::Selectable(choices[k].label, static_cast<size_t>(index) == k)) {
                changed = changed || static_cast<size_t>(index) != k;
                index = static_cast<int>(k);
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

/// A PANEL OF THE SHARED DESCRIPTION (athenea::ui), drawn with Dear ImGui: a
/// collapsing header a section, a reading its label and its text, a note as
/// the row's tooltip. What the iOS app draws with UIKit from the same table.
void drawPanel(const ui::Panel& panel) {
    for (const ui::Section& section : panel.sections) {
        if (!section.isShown()) {
            continue;
        }
        ImGui::PushID(section.id.c_str());
        if (ImGui::CollapsingHeader(section.title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            for (const ui::Control& control : section.controls) {
                if (!control.isShown()) {
                    continue;
                }
                ImGui::PushID(control.id.c_str());
                ImGui::BeginDisabled(!control.isEnabled());
                switch (control.kind) {
                case ui::Control::Kind::Reading: {
                    const std::string text = control.reading ? control.reading() : std::string();
                    if (control.label.empty()) {
                        ImGui::Separator();
                        ImGui::TextWrapped("%s", text.c_str());
                    } else {
                        ImGui::TextDisabled("%s", control.label.c_str());
                        ImGui::SameLine(150.0F);
                        ImGui::TextWrapped("%s", text.c_str());
                    }
                    break;
                }
                case ui::Control::Kind::Toggle: {
                    bool on = control.flag && control.flag();
                    if (ImGui::Checkbox(control.label.c_str(), &on) && control.setFlag) {
                        control.setFlag(on);
                    }
                    break;
                }
                case ui::Control::Kind::Action:
                    if (ImGui::Button(control.label.c_str()) && control.act) {
                        control.act();
                    }
                    break;
                case ui::Control::Kind::Slider:
                case ui::Control::Kind::Stepper: {
                    const auto [lo, hi] = control.bounds();
                    float value = control.number ? float(control.number()) : 0.0F;
                    const std::string format = "%." + std::to_string(control.decimals) + "f " + control.unit;
                    if (ImGui::SliderFloat(control.label.c_str(), &value, float(lo), float(hi), format.c_str(),
                                           control.logarithmic ? ImGuiSliderFlags_Logarithmic : 0) &&
                        control.setNumber) {
                        control.setNumber(double(value));
                    }
                    break;
                }
                case ui::Control::Kind::Choice: {
                    const std::string now = control.chosen ? control.chosen() : std::string();
                    if (ImGui::BeginCombo(control.label.c_str(), ui::labelOf(control.choices, now).c_str())) {
                        for (const ui::Choice& entry : control.choices) {
                            if (ImGui::Selectable(entry.label.c_str(), entry.value == now) && control.choose) {
                                control.choose(entry.value);
                            }
                        }
                        ImGui::EndCombo();
                    }
                    break;
                }
                }
                ImGui::EndDisabled();
                if (!control.note.empty() && ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", control.note.c_str());
                }
                ImGui::PopID();
            }
        }
        ImGui::PopID();
    }
}

int indexOf(std::span<const Choice> choices, const std::string& value) {
    for (size_t k = 0; k < choices.size(); ++k) {
        if (value == choices[k].value) {
            return static_cast<int>(k);
        }
    }
    return 0;
}

}   // namespace

Result<ViewStats> runViewer(const ViewOptions& options) {
    auto opened = usd::StageRenderer::open(options.stage);
    if (!opened) return std::move(opened).error();
    usd::StageRenderer& stage = **opened;
    gpu::Device& device = stage.device();
    gpu::ShaderLibrary& library = stage.library();

    auto window = Window::open("athenea view - " + options.stage.filename().string(), options.width, options.height,
                               options.visible);
    if (!window) return std::move(window).error();
    rhi::ComPtr<rhi::ISurface> surface = device.rhi()->createSurface((*window)->handle());
    if (!surface) {
        return Error(ErrorCode::DeviceFailure, "cannot make a surface for the window");
    }
    (*window)->matchSurfaceToBacking();
    // Extended range: the surface in floats, linear P3 with 1.0 at the
    // reference white, and ACES 2.0 filling the headroom the screen has.
    rhi::Format surfaceFormat = kSurfaceFormat;
    double headroom = 1.0;
    if (options.edr && (*window)->enableExtendedRange()) {
        surfaceFormat = rhi::Format::RGBA16Float;
        headroom = (*window)->extendedRangeHeadroom();
        athenea::log::info("athenea view: extended range on, headroom {:.2f}", headroom);
    }
    uint32_t surfaceWidth = 0;
    uint32_t surfaceHeight = 0;
    const auto configure = [&](uint32_t w, uint32_t h) -> Result<void> {
        rhi::SurfaceConfig config;
        config.format = surfaceFormat;
        config.usage = rhi::TextureUsage::Present | rhi::TextureUsage::RenderTarget |
                       rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
        config.width = w;
        config.height = h;
        config.vsync = true;
        if (SLANG_FAILED(surface->configure(config))) {
            return Error::make(ErrorCode::DeviceFailure, "cannot configure a {}x{} surface", w, h);
        }
        surfaceWidth = w;
        surfaceHeight = h;
        return ok();
    };

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    struct ContextGuard {
        ~ContextGuard() {
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
    };
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui_ImplGlfw_InitForOther((*window)->glfw(), true);
    const ContextGuard contextGuard;
    auto ui = ImGuiRenderer::create(library);
    if (!ui) return std::move(ui).error();
    auto display = technique::DisplayTransform::create(library);
    if (!display) return std::move(display).error();
    const bool ocio = !options.ocioConfig.empty() || !options.ocioDisplay.empty() || !options.ocioView.empty();
    if (ocio) {
        technique::OcioView chosen;
        if (!options.ocioConfig.empty()) {
            chosen.config = options.ocioConfig;
        }
        chosen.display = options.ocioDisplay;
        chosen.view = options.ocioView;
        ATHENEA_TRY(display->setOcio(chosen));
        athenea::log::info("athenea view: {}", display->ocioDescription());
    }

    // The selections asked for on the command line, before the cameras, the
    // variant sets and the time range are read: each one recomposes the stage.
    for (const std::string& selection : options.variants) {
        ATHENEA_TRY(stage.setVariantSelection(selection));
    }

    // What the panels set. The cameras and the variant sets are re-read
    // whenever a variant is chosen, since choosing one recomposes the stage:
    // another animation, another level of detail, and possibly another camera.
    std::vector<std::string> cameras = stage.cameras();
    int cameraIndex = 0;   // 0: the free camera
    for (size_t k = 0; k < cameras.size(); ++k) {
        if (cameras[k] == options.camera) {
            cameraIndex = static_cast<int>(k) + 1;
        }
    }
    std::vector<usd::StageVariantSet> variantSets = stage.variantSets();
    // THE SKIES THE PANEL OFFERS, and the domes they can go on. A dome is the
    // one light whose whole character is a file, so swapping it relights the
    // frame with nothing else on the stage touched -- which is the only way
    // to see, rather than be told, what a relightable cloud buys.
    std::vector<usd::StageDome> domes = stage.domes();
    // And every light the stage brought, so that what is not the sky can be
    // taken away while trying skies: a fixed sun is a highlight no image
    // explains.
    std::vector<usd::StageLight> lights = stage.lights();
    const std::vector<std::filesystem::path> skies = skiesOnOffer(options.hdriPaths, domes);
    // The timeline's extent: what the stage's samples occupy, not what its
    // root layer declares, because a variant set of seventy animations
    // declares the longest of them for all of them.
    std::pair<double, double> range = stage.animationRange();
    const auto recompose = [&] {
        const std::string was = cameraIndex == 0 ? std::string() : cameras[static_cast<size_t>(cameraIndex - 1)];
        cameras = stage.cameras();
        cameraIndex = 0;
        for (size_t k = 0; k < cameras.size(); ++k) {
            if (cameras[k] == was) {
                cameraIndex = static_cast<int>(k) + 1;
            }
        }
        variantSets = stage.variantSets();
        domes = stage.domes();
        lights = stage.lights();
        range = stage.animationRange();
    };
    int technique = indexOf(kTechniques, options.technique);
    int visibility = indexOf(kVisibility, options.visibility);
    int aov = options.isolate.empty() ? (options.aov.empty() ? 0 : indexOf(kAovs, options.aov))
                                      : indexOf(kAovs, "cryptomatte");
    // OCIO when one was given; ACES 2.0 for extended range; AgX otherwise.
    int viewTransform = ocio ? 3 : surfaceFormat == rhi::Format::RGBA16Float ? 2 : 1;
    int displayEncoding = surfaceFormat == rhi::Format::RGBA16Float ? 3 : 0;
    float exposure = 0.0F;
    float renderScale = 1.0F;
    double time = range.first;
    // The timeline: playing advances the time by the wall clock at the
    // stage's timeCodesPerSecond and wraps at the end. What frame N shows is
    // the stage's business (SetTime); when it is drawn is the clock's.
    //
    // Unless `everyFrame`, which advances one time code a drawn frame instead
    // -- so a stage at 30 fps that draws at ten shows every pose in order,
    // three times slower than life. Which is what you want when what you are
    // looking at is how a wing moves between two of them, and what you do not
    // want when you are judging the speed of the motion itself.
    bool playing = options.play && range.second > range.first;
    bool everyFrame = options.everyFrame;
    double drawnTime = time;
    uint32_t distinctTimes = 0;
    auto lastTick = std::chrono::steady_clock::now();
    double focal = 35.0;
    bool cloudShadows = true;
    float shadowDensity = 1.0F;
    float fStop = float(options.fStop);
    float focusDistance = float(options.focus > 0.0 ? options.focus : 1.0);
    // What a variant combo is typing into, and what it picked: the pick is
    // applied after the combos are drawn, since it rebuilds them.
    std::array<char, 64> variantFilter{};
    std::optional<std::tuple<std::string, std::string, std::string>> chosen;
    // The same for the sky: which dome, and the image to put on it (empty is
    // the one the stage itself authored). Applied after the panel, since
    // setting it rereads the domes the loop is walking.
    std::array<char, 64> skyFilter{};
    std::optional<std::pair<std::string, std::string>> skyChosen;
    std::optional<usd::StagePick> picked;
    // WHAT THE PICKED PRIM'S GAUSSIANS ARE MADE OF, AND WHAT WE SAY THEY ARE.
    //
    // A Cryptomatte id is a selection: every gaussian carrying it came from
    // one prim. `reading` is what they carry, counted on the device when the
    // pick changes; `said` is the row the frame holds over them, which starts
    // as the reading so that moving one slider does not flatten the rest.
    uint32_t measuredId = 0;
    render::SplatIdReading reading;
    render::SplatOverride said;
    bool saying = false;
    std::string status;
    Orbit orbit;
    bool framed = false;
    const char up = stage.upAxis();
    // WHAT THE FRAME IS ASKED TO KEEP.
    //
    // The three Cryptomatte layers are asked for whenever the rasteriser is
    // drawing, not only when the matte is being shown. A gaussian writes no
    // `primId`, so the matte is the only name a pixel of splats has: without
    // it a click on a cloud answers with nothing at all, and the panel that
    // says what that prim is made of never appears. The colour is the same
    // colour either way -- the matte rides the walk the blend already does,
    // which `athenea_render_tests "[crypto]"` checks to the bit.
    //
    // The traced splat route writes no matte, so under `rt` this asks for
    // nothing and the panel says why rather than going quiet.
    const auto request = [&] {
        std::vector<std::string> outputs{"primId", "instanceId"};
        const std::string chosen = kAovs[static_cast<size_t>(aov)].value;
        const bool rasterising = kTechniques[static_cast<size_t>(technique)].value == std::string_view("raster");
        if (chosen == "cryptomatte" || rasterising) {
            outputs.push_back("CryptoObject00");
            outputs.push_back("CryptoObject01");
            outputs.push_back("CryptoObject02");
        }
        if (chosen != "color" && chosen != "depth" && chosen != "primId" && chosen != "instanceId" &&
            chosen != "cryptomatte") {
            outputs.push_back(chosen);
        }
        stage.requestOutputs(outputs);
    };
    request();
    ATHENEA_TRY(stage.setMeshVisibility(kVisibility[static_cast<size_t>(visibility)].value));
    stage.setLightSamples(options.lightSamples);
    stage.setChooseLights(options.chooseLights);
    int pathSamples = static_cast<int>(std::max(options.pathSamples, 1u));
    int pathBounces = static_cast<int>(options.pathBounces);
    int pathTotal = static_cast<int>(std::max(options.pathTotal, 1u));
    bool denoise = options.denoise;
    stage.setPathSamples(static_cast<uint32_t>(pathSamples));
    stage.setPathBounces(static_cast<uint32_t>(pathBounces));
    stage.setPathTotal(static_cast<uint32_t>(pathTotal));
    stage.setDenoise(denoise);
    // A camera of the viewer's own authors no shutter, so this is how one is
    // said. Pushed every time it changes rather than every frame: a shutter
    // change re-dirties every prim so they sample about the new one.
    float shutter = std::max(options.shutter, 0.0F);
    float shutterSet = -1.0F;
    const bool stageLit = stage.hasLights();
    bool defaultLights = !stageLit && options.defaultLights;
    ATHENEA_TRY(stage.setDefaultLights(defaultLights));
    // The first frame of a technique compiles its kernels for every material
    // on the stage -- minutes, the first time, for a stage like the chess set
    // -- and the window cannot draw while it does. One frame says so first.
    std::array<bool, kTechniques.size()> techniqueDrawn{};
    // ATHENEA_VIEW_SWITCH_AT=N: at frame N the Technique selector goes to the
    // other technique, as a click on it would -- so the sequence a person
    // reports (open in one, switch to the other) can be run with --frames
    // and --snapshot rather than described.
    const std::string switchAtText = platform::env("ATHENEA_VIEW_SWITCH_AT");
    const long switchAt = switchAtText.empty() ? -1L : std::strtol(switchAtText.c_str(), nullptr, 10);
    int announcedFor = -1;
    // ATHENEA_VIEW_ORBIT=R: the free camera turns R radians about its target
    // every frame, as a drag would -- so what moving the camera costs can be
    // measured with --frames rather than felt.
    const std::string orbitText = platform::env("ATHENEA_VIEW_ORBIT");
    const double orbitStep = orbitText.empty() ? 0.0 : std::strtod(orbitText.c_str(), nullptr);

    // The id whose matte is shown alone, white on black: what the Isolate
    // button on a pick sets. 0 shows every id in its own colour.
    uint32_t cryptoIsolate = 0;
    const auto displaySettings = [&] {
        technique::DisplaySettings settings;
        settings.cryptoIsolate = cryptoIsolate;
        settings.view = static_cast<technique::ViewTransform>(viewTransform);
        settings.display = static_cast<technique::DisplayEncoding>(displayEncoding);
        settings.peakLuminance = static_cast<float>(100.0 * headroom);
        settings.exposure = exposure;
        settings.background = kBackground;
        settings.nearZ = static_cast<float>(std::max(orbit.distance - orbit.radius, orbit.distance * 1e-2));
        settings.farZ = static_cast<float>(orbit.distance + orbit.radius);
        return settings;
    };
    // THE GAUSSIANS PANEL, described in athenea::ui and drawn by drawPanel. Its
    // numbers are the report of the frame just drawn, taken once a frame; the
    // engine gathers them only while the panel is open, and the device's
    // counts in it are read without waiting, so they can be a frame behind.
    usd::GaussianStats gaussianReport;
    bool timeSplatStages = false;
    bool gaussiansOpen = true;
    const ui::Panel gaussians =
        ui::gaussianPanel([&gaussianReport]() -> const ui::GaussianReport& { return gaussianReport; },
                          timeSplatStages);
    uint64_t snapshotLit = 0;
    std::vector<double> drawMs;
    std::vector<double> frameMs;
    ImVec2 pressedAt{0.0F, 0.0F};
    uint32_t frames = 0;
    while (!(*window)->shouldClose() && (options.frames == 0 || frames < options.frames)) {
        const auto frameStart = std::chrono::steady_clock::now();
        (*window)->pollEvents();
        const auto [fbw, fbh] = (*window)->framebufferSize();
        if (fbw == 0 || fbh == 0) {
            continue;   // minimised
        }
        if (fbw != surfaceWidth || fbh != surfaceHeight) {
            ATHENEA_TRY(configure(fbw, fbh));
        }
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuiIO& io = ImGui::GetIO();
        const uint32_t rw = std::max<uint32_t>(1, static_cast<uint32_t>(float(fbw) * renderScale));
        const uint32_t rh = std::max<uint32_t>(1, static_cast<uint32_t>(float(fbh) * renderScale));

        if (switchAt >= 0 && static_cast<long>(frames) == switchAt) {
            technique = technique == 0 ? 1 : 0;
        }
        // The free camera, from the mouse the panels do not want.
        if (!io.WantCaptureMouse) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                pressedAt = io.MousePos;
            }
            const ImVec2 delta = io.MouseDelta;
            const bool pan = ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
                             (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && io.KeyShift);
            if (cameraIndex == 0) {
                if (pan) {
                    const render::Camera camera = cameraOf(orbit, up, focal);
                    const double step = orbit.distance * 0.0015;
                    for (size_t k = 0; k < 3; ++k) {
                        orbit.target[k] += (-camera.cameraToWorld.at(int(k), 0) * delta.x +
                                            camera.cameraToWorld.at(int(k), 1) * delta.y) *
                                           step;
                    }
                } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    orbit.yaw -= delta.x * 0.005;
                    orbit.pitch = std::clamp(orbit.pitch + delta.y * 0.005, -1.55, 1.55);
                } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
                    orbit.distance *= std::exp(delta.y * 0.005);
                }
                if (io.MouseWheel != 0.0F) {
                    orbit.distance *= std::pow(0.9, double(io.MouseWheel));
                }
                orbit.yaw += orbitStep;
            }
            // A click that did not drag picks.
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !io.KeyShift) {
                const float moved = std::hypot(io.MousePos.x - pressedAt.x, io.MousePos.y - pressedAt.y);
                if (moved < 3.0F) {
                    const float px = io.MousePos.x * io.DisplayFramebufferScale.x * float(rw) / float(fbw);
                    const float py = io.MousePos.y * io.DisplayFramebufferScale.y * float(rh) / float(fbh);
                    if (px >= 0.0F && py >= 0.0F) {
                        auto hit = stage.pick(static_cast<uint32_t>(px), static_cast<uint32_t>(py));
                        if (hit) {
                            picked = *hit;
                        } else {
                            status = hit.error().toString();
                        }
                    }
                }
            }
        }
        if (!io.WantCaptureKeyboard) {
            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                (*window)->requestClose();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_F)) {
                framed = false;
                cameraIndex = 0;
            }
        }

        // Panels.
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340, 420), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("View")) {
            const char* cameraLabel = cameraIndex == 0 ? "Free" : cameras[static_cast<size_t>(cameraIndex - 1)].c_str();
            if (ImGui::BeginCombo("Camera", cameraLabel)) {
                if (ImGui::Selectable("Free", cameraIndex == 0)) {
                    cameraIndex = 0;
                }
                for (size_t k = 0; k < cameras.size(); ++k) {
                    if (ImGui::Selectable(cameras[k].c_str(), cameraIndex == static_cast<int>(k) + 1)) {
                        cameraIndex = static_cast<int>(k) + 1;
                    }
                }
                ImGui::EndCombo();
            }
            if (cameraIndex == 0) {
                float f = static_cast<float>(focal);
                if (ImGui::SliderFloat("Focal (mm)", &f, 8.0F, 200.0F, "%.0f")) {
                    focal = f;
                }
                if (ImGui::Button("Frame all (F)")) {
                    framed = false;
                }
                // A diaphragm for the viewer's own camera: the raster spreads
                // each splat by its circle of confusion, the path tracer
                // samples the disk, and both focus at the same distance.
                ImGui::SliderFloat("f-stop", &fStop, 0.0F, 22.0F, fStop > 0.0F ? "f/%.1f" : "pinhole");
                if (fStop > 0.0F) {
                    ImGui::SliderFloat("Focus", &focusDistance, 0.01F, 100.0F, "%.2f", ImGuiSliderFlags_Logarithmic);
                }
            }
            // One combo a variant set the stage carries, whatever they mean:
            // variant sets are USD's own "pick one of these", so an asset that
            // packages its animations as variants -- or its levels of detail,
            // or its shirts -- is offered here without the viewer knowing what
            // it is offering. A set with many variants gets a filter box,
            // which is what makes seventy of them usable.
            for (size_t k = 0; k < variantSets.size(); ++k) {
                const usd::StageVariantSet& set = variantSets[k];
                const bool filtered = set.variants.size() > 12;
                ImGui::PushID(static_cast<int>(k));
                if (ImGui::BeginCombo(set.name.c_str(), set.selected.c_str())) {
                    if (ImGui::IsWindowAppearing()) {
                        variantFilter[0] = '\0';
                    }
                    if (filtered) {
                        ImGui::SetNextItemWidth(-1.0F);
                        ImGui::InputTextWithHint("##filter", "filter", variantFilter.data(),
                                                 variantFilter.size());
                        ImGui::Separator();
                    }
                    for (const std::string& variant : set.variants) {
                        if (filtered && variantFilter[0] != '\0' &&
                            variant.find(variantFilter.data()) == std::string::npos) {
                            continue;
                        }
                        if (ImGui::Selectable(variant.c_str(), variant == set.selected) &&
                            variant != set.selected) {
                            chosen = {set.prim, set.name, variant};
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::PopID();
            }
            // Applied outside the loop: the selection recomposes the stage,
            // and the vector being walked is one of the things rebuilt.
            if (chosen) {
                if (auto set = stage.setVariantSelection(std::get<0>(*chosen), std::get<1>(*chosen),
                                                        std::get<2>(*chosen));
                    !set) {
                    status = set.error().toString();
                } else {
                    recompose();
                    time = std::clamp(time, range.first, range.second);
                }
                chosen.reset();
            }
            // THE STAGE'S LIGHTS, ONE SWITCH EACH. Switched after the list
            // is drawn, since switching one rereads the list and the domes.
            if (!lights.empty()) {
                ImGui::SeparatorText("Lights");
                std::optional<std::pair<std::string, bool>> switched;
                for (const usd::StageLight& light : lights) {
                    bool on = light.on;
                    const std::string label = light.name + "##" + light.prim;
                    if (ImGui::Checkbox(label.c_str(), &on)) {
                        switched = {light.prim, on};
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s, %s", light.type.c_str(), light.prim.c_str());
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", light.type.c_str());
                }
                if (switched) {
                    if (auto set = stage.setLightOn(switched->first, switched->second); !set) {
                        status = set.error().toString();
                    }
                    // A dome switched off leaves the Sky block with it.
                    lights = stage.lights();
                    domes = stage.domes();
                }
            }
            // THE SKY, AS A LIST YOU CAN WALK. One combo a dome, holding
            // every image found beside the one it already carries, plus
            // whatever `--hdri` added; the first entry puts the stage's own
            // opinion back, because a session layer is what this writes into
            // and clearing it is how that opinion goes away.
            if (!domes.empty() && !skies.empty()) {
                ImGui::SeparatorText("Sky");
                for (size_t k = 0; k < domes.size(); ++k) {
                    const usd::StageDome& dome = domes[k];
                    ImGui::PushID(static_cast<int>(k));
                    const std::string label = domes.size() == 1 ? std::string("Environment") : dome.name;
                    const std::string shown =
                        dome.texture.empty() ? std::string("a colour")
                                             : std::filesystem::path(dome.texture).stem().string();
                    const bool filtered = skies.size() > 12;
                    if (ImGui::BeginCombo(label.c_str(), shown.c_str())) {
                        if (ImGui::IsWindowAppearing()) {
                            skyFilter[0] = '\0';
                            ImGui::SetKeyboardFocusHere();
                        }
                        if (filtered) {
                            ImGui::InputTextWithHint("##skyfilter", "filter", skyFilter.data(),
                                                     skyFilter.size());
                            ImGui::Separator();
                        }
                        if (ImGui::Selectable("(the stage's own)", dome.texture.empty())) {
                            skyChosen = {dome.prim, std::string()};
                        }
                        for (const std::filesystem::path& sky : skies) {
                            const std::string name = sky.stem().string();
                            if (filtered && skyFilter[0] != '\0' &&
                                name.find(skyFilter.data()) == std::string::npos) {
                                continue;
                            }
                            if (ImGui::Selectable(name.c_str(), sky.string() == dome.texture)) {
                                skyChosen = {dome.prim, sky.string()};
                            }
                        }
                        ImGui::EndCombo();
                    }
                    // An environment is only right once it is turned, and
                    // turning it is the quickest proof that a reflection is
                    // reading the sky rather than a constant.
                    float turn = dome.rotation;
                    if (ImGui::SliderFloat("Turn", &turn, -180.0F, 180.0F, "%.0f deg")) {
                        if (auto set = stage.setDomeRotation(dome.prim, turn); !set) {
                            status = set.error().toString();
                        } else {
                            domes[k].rotation = turn;
                        }
                    }
                    ImGui::PopID();
                }
                if (skyChosen) {
                    if (auto set = stage.setDomeTexture(skyChosen->first, skyChosen->second); !set) {
                        status = set.error().toString();
                    } else {
                        // The dome's record changed, so the prepared sky --
                        // its harmonics and its prefiltered chain -- is
                        // rebuilt on the next frame and not on the ones after.
                        domes = stage.domes();
                    }
                    skyChosen.reset();
                }
                ImGui::Separator();
            }
            if (combo("Technique", technique, kTechniques)) {
                // Raster keeps a matte and the traced route does not, so what
                // the frame is asked for changes with the route.
                request();
            }
            // What the frame's clouds cast on its meshes and on each other,
            // from a map at each light: no ray, so it is the same switch on
            // every device.
            if (ImGui::Checkbox("Cloud shadows", &cloudShadows)) {
                stage.setCloudShadows(cloudShadows);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("A transmittance map at each light: a floor under a cloud is dark, and "
                                  "gaussians shadow each other. No ray is traced.");
            }
            if (cloudShadows) {
                ImGui::Indent();
                // The density a compositor means: 1 is what the cloud's own
                // opacity says, and the only value that is a measurement.
                if (ImGui::SliderFloat("Shadow density", &shadowDensity, 0.0F, 4.0F, "%.2f")) {
                    stage.setCloudShadowDensity(shadowDensity);
                }
                ImGui::Unindent();
            }
            if (!stageLit && ImGui::Checkbox("Default lights (the stage has none)", &defaultLights)) {
                ATHENEA_TRY(stage.setDefaultLights(defaultLights));
            }
            if (kTechniques[static_cast<size_t>(technique)].value == std::string_view("rt")) {
                // The path tracer's own settings: the delegate's defaults are
                // one bounce, which lights a room little more than the raster.
                ImGui::Indent();
                if (ImGui::SliderInt("Paths per frame", &pathSamples, 1, 64)) {
                    stage.setPathSamples(static_cast<uint32_t>(pathSamples));
                }
                if (ImGui::SliderInt("Bounces", &pathBounces, 0, 16)) {
                    stage.setPathBounces(static_cast<uint32_t>(pathBounces));
                }
                if (ImGui::Checkbox("Denoise", &denoise)) {
                    stage.setDenoise(denoise);
                }
                // A viewport keeps gathering while the camera is still and
                // starts again when it moves; the count says which.
                ImGui::Text("%u paths a pixel", stage.pathAccumulated());
                ImGui::Unindent();
            }
            if (combo("Mesh visibility", visibility, kVisibility)) {
                ATHENEA_TRY(stage.setMeshVisibility(kVisibility[static_cast<size_t>(visibility)].value));
            }
            if (combo("Output", aov, kAovs)) {
                request();
            }
            ImGui::Separator();
            const char* views[] = {"Standard", "AgX", "ACES 2.0", "OCIO"};
            ImGui::Combo("View transform", &viewTransform, views, ocio ? 4 : 3);
            if (viewTransform == 3) {
                ImGui::TextUnformatted(display->ocioDescription().c_str());
            }
            const char* displays[] = {"sRGB", "Rec.709 (BT.1886)", "Display P3", "Linear P3 (extended range)"};
            ImGui::Combo("Display", &displayEncoding, displays, surfaceFormat == rhi::Format::RGBA16Float ? 4 : 3);
            ImGui::SliderFloat("Exposure", &exposure, -8.0F, 8.0F, "%.1f stops");
            // What a cloud's motion blur is integrated over. A stage camera
            // brings its own, and then this is what the free camera uses.
            ImGui::SliderFloat("Shutter", &shutter, 0.0F, 1.0F, "%.2f frames");
            ImGui::SliderFloat("Render scale", &renderScale, 0.25F, 1.0F, "%.2f");
            const double start = range.first;
            const double end = range.second;
            if (end > start) {
                float t = static_cast<float>(time);
                if (ImGui::SliderFloat("Time", &t, float(start), float(end), "%.1f")) {
                    time = t;
                    playing = false;
                }
                if (ImGui::Button(playing ? "Pause" : "Play")) {
                    playing = !playing;
                    lastTick = std::chrono::steady_clock::now();
                }
                ImGui::SameLine();
                ImGui::Checkbox("Every frame", &everyFrame);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Advance one frame a drawn frame instead of by the clock: "
                                      "every pose is shown, slower than life.");
                }
                ImGui::SameLine();
                if (ImGui::Button("|<")) {
                    time = start;
                    playing = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("<")) {
                    time = std::max(start, std::floor(time - 1.0 + 0.5));
                    playing = false;
                }
                ImGui::SameLine();
                if (ImGui::Button(">")) {
                    time = std::min(end, std::floor(time + 1.0 + 0.5));
                    playing = false;
                }
                ImGui::SameLine();
                ImGui::Text("%.1f fps", stage.timeCodesPerSecond());
            }
            if (playing) {
                const auto now = std::chrono::steady_clock::now();
                const double seconds = std::chrono::duration<double>(now - lastTick).count();
                lastTick = now;
                // A time code a drawn frame, or as much of the timeline as
                // the wall clock went through while this frame was drawn.
                time += everyFrame ? 1.0 : seconds * stage.timeCodesPerSecond();
                if (time > end) {
                    time = start + std::fmod(time - start, std::max(end - start, 1e-9));
                }
            }
            ImGui::Separator();
            ImGui::Text("%s on %s", device.caps().apiName.c_str(), device.caps().adapterName.c_str());
            ImGui::Text("%u x %u, draw %.2f ms, frame %.2f ms", rw, rh, drawMs.empty() ? 0.0 : drawMs.back(),
                        frameMs.empty() ? 0.0 : frameMs.back());
            // A prim named on the command line: its id is only known once a
            // frame has built the manifest, so it is looked up here.
            if (!options.isolate.empty() && cryptoIsolate == 0) {
                const auto manifest = stage.cryptoManifest();
                const auto found = manifest.find(options.isolate);
                if (found != manifest.end()) {
                    cryptoIsolate = found->second;
                } else if (!manifest.empty() && status.empty()) {
                    status = "no prim '" + options.isolate + "' in the frame's matte";
                }
            }
            // WHAT THE FRAME HELD, from the counts the passes already had.
            const usd::StageRenderer::Counters counters = stage.counters();
            if (counters.splats > 0) {
                ImGui::Text("%u gaussians, %u visible, %u pairs", counters.splats, counters.visibleSplats,
                            counters.splatPairs);
            }
            if (counters.meshInstances > 0 || counters.lights > 0) {
                ImGui::Text("%u mesh instances, %u lights%s", counters.meshInstances, counters.lights,
                            counters.cryptomatte ? ", matte kept" : "");
            }
            if (!status.empty()) {
                ImGui::TextWrapped("%s", status.c_str());
            }
        }
        ImGui::End();
        // PICKED HAS A PANEL OF ITS OWN, and not the bottom of another one.
        //
        // What a pixel is, and what the prim behind it is made of, used to sit
        // under everything else in View: on any window shorter than the panel
        // it was below the fold, behind a scrollbar nobody thinks to drag
        // after clicking. It is its own window now, open from the start, and
        // it says what to do when nothing is picked rather than being empty.
        ImGui::SetNextWindowPos(ImVec2(360, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(360, 420), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Picked")) {
            const usd::StageRenderer::Counters counters = stage.counters();
        if (picked) {
            ImGui::TextWrapped("Picked %s (instance %d)", picked->prim.c_str(), picked->instance);
            // WHAT THE MATTE CALLS IT. A gaussian writes no prim id, so
            // for a cloud this is the only name a pixel has -- and one
            // click pulls that id's matte on its own.
            if (picked->cryptoId != 0) {
                ImGui::TextWrapped("Matte %s (%08x, %.0f%% of the pixel)",
                                   picked->cryptoName.empty() ? "unnamed" : picked->cryptoName.c_str(),
                                   picked->cryptoId, double(picked->cryptoCoverage) * 100.0);
                const bool isolated = cryptoIsolate == picked->cryptoId;
                if (ImGui::Button(isolated ? "Show every id" : "Isolate this matte")) {
                    cryptoIsolate = isolated ? 0u : picked->cryptoId;
                    // The matte is shown by the Cryptomatte output, so
                    // the button takes the view there when it is not.
                    if (!isolated && kAovs[static_cast<size_t>(aov)].value != "cryptomatte") {
                        aov = indexOf(kAovs, "cryptomatte");
                        request();
                    }
                }
                // WHAT THOSE GAUSSIANS ARE MADE OF, counted once when the
                // pick changes: the id is the whole selection, so the
                // answer is about the prim and not about the pixel.
                if (measuredId != picked->cryptoId) {
                    measuredId = picked->cryptoId;
                    reading = {};
                    if (auto got = stage.measureSplatId(measuredId); !got) {
                        status = got.error().toString();
                    } else {
                        reading = *got;
                    }
                    // The sliders start where the cloud already is, and
                    // a row already held for this id wins over that.
                    said = render::SplatOverride{};
                    said.id = measuredId;
                    saying = false;
                    for (const render::SplatOverride& row : stage.splatOverrides()) {
                        if (row.id == measuredId) {
                            said = row;
                            saying = true;
                        }
                    }
                    if (!saying && reading.hasPbr) {
                        said.metallic = reading.mean[0];
                        said.roughness = reading.mean[1];
                        said.transmission = reading.mean[2];
                    }
                }
                if (reading.count > 0) {
                    ImGui::Text("%u gaussians carry it", reading.count);
                    if (reading.hasPbr) {
                        // The range, not only the mean: a material whose
                        // roughness came out of a map gives each gaussian
                        // its own, and a panel that showed one number
                        // would be claiming they are all the same.
                        const char* names[] = {"metallic", "roughness", "transmission"};
                        for (uint32_t c = 0; c < 3; ++c) {
                            ImGui::Text("%-13s %.3f", names[c], double(reading.mean[c]));
                            if (reading.high[c] - reading.low[c] > 1.0e-3F) {
                                ImGui::SameLine();
                                ImGui::TextDisabled("(%.3f to %.3f)", double(reading.low[c]),
                                                    double(reading.high[c]));
                            }
                        }
                        ImGui::SeparatorText("Say otherwise");
                        bool moved = false;
                        moved |= ImGui::SliderFloat("Metallic##said", &said.metallic, 0.0F, 1.0F, "%.3f");
                        moved |= ImGui::SliderFloat("Roughness##said", &said.roughness, 0.0F, 1.0F, "%.3f");
                        moved |= ImGui::SliderFloat("Transmission##said", &said.transmission, 0.0F, 1.0F,
                                                    "%.3f");
                        moved |= ImGui::ColorEdit3("Tint##said", said.tint.data(),
                                                   ImGuiColorEditFlags_Float);
                        moved |= ImGui::Checkbox("The tint is the colour##said", &said.replaceColour);
                        if (moved) {
                            stage.setSplatOverride(said);
                            saying = true;
                        }
                        if (saying) {
                            ImGui::TextDisabled("every gaussian of this prim, this frame only");
                            if (ImGui::Button("Put the file back")) {
                                stage.clearSplatOverride(measuredId);
                                saying = false;
                                said = render::SplatOverride{};
                                said.id = measuredId;
                                said.metallic = reading.mean[0];
                                said.roughness = reading.mean[1];
                                said.transmission = reading.mean[2];
                            }
                            ImGui::SameLine();
                        }
                        if (!stage.splatOverrides().empty() && ImGui::Button("Put every prim back")) {
                            stage.clearSplatOverrides();
                            saying = false;
                        }
                    } else {
                        ImGui::TextDisabled("this cloud carries no material to change");
                    }
                }
            } else {
                // NEVER SILENT. A pixel with no id has a reason, and the
                // reason decides what to do about it.
                const bool tracing =
                    kTechniques[static_cast<size_t>(technique)].value == std::string_view("rt");
                if (tracing) {
                    ImGui::TextDisabled("no matte: the traced splat route writes none");
                    if (ImGui::Button("Switch to Raster")) {
                        technique = indexOf(kTechniques, "raster");
                        request();
                    }
                } else if (!counters.cryptomatte) {
                    ImGui::TextDisabled("no matte in this frame");
                } else {
                    ImGui::TextDisabled("nothing named this pixel: a cloud converted without ids, "
                                        "or a surface the matte does not cover");
                }
            }
        }
            if (!picked) {
                ImGui::TextDisabled("Click a pixel.");
                ImGui::TextWrapped("A gaussian writes no prim id, so a cloud is named by its "
                                   "Cryptomatte: pick one and this says how many gaussians share "
                                   "that name, what they are made of, and lets you say otherwise.");
            }
        }
        ImGui::End();
        ImGui::SetNextWindowPos(ImVec2(10, 440), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340, 420), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Stage")) {
            const std::function<void(const std::string&)> tree = [&](const std::string& path) {
                for (const usd::StagePrim& prim : stage.children(path)) {
                    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
                    if (!prim.hasChildren) {
                        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                    }
                    if (picked && picked->prim == prim.path) {
                        flags |= ImGuiTreeNodeFlags_Selected;
                    }
                    const bool open = ImGui::TreeNodeEx(prim.path.c_str(), flags, "%s  %s%s", prim.name.c_str(),
                                                        prim.type.c_str(), prim.instance ? " (instance)" : "");
                    if (open && prim.hasChildren) {
                        tree(prim.path);
                        ImGui::TreePop();
                    }
                }
            };
            tree("/");
        }
        ImGui::End();

        // The shutter the panel holds, pushed when it moves and not every
        // frame: a change re-dirties every prim so they sample about the new
        // one. Whichever camera is looking -- a control that silently loses
        // to the stage camera's own shutter is a broken control.
        if (shutter != shutterSet) {
            stage.setShutter(0.0, double(shutter));
            shutterSet = shutter;
        }

        // What the Gaussians panel asks of the frame: counted while it is
        // open, its stages timed while its switch is on.
        stage.setGaussianStats(gaussiansOpen);
        stage.setTimeSplatStages(gaussiansOpen && timeSplatStages);

        // The frame.
        const auto drawStart = std::chrono::steady_clock::now();
        const std::string techniqueName = kTechniques[static_cast<size_t>(technique)].value;
        Result<void> drawn = ok();
        const size_t techniqueAt = static_cast<size_t>(technique);
        const bool announce = !techniqueDrawn[techniqueAt] && frames > 0 && announcedFor != technique;
        if (announce) {
            // Shown over the last frame; the frame that compiles is the next.
            announcedFor = technique;
            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(ImVec2(viewport->Size.x * 0.5F, viewport->Size.y * 0.5F), ImGuiCond_Always,
                                    ImVec2(0.5F, 0.5F));
            ImGui::Begin("##preparing", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing);
            ImGui::Text("Preparing %s: its kernels compile for this stage's materials.",
                        kTechniques[techniqueAt].label);
            ImGui::Text("The window waits until they have; next time they come from the cache.");
            ImGui::End();
        } else if (cameraIndex == 0) {
            drawn = stage.draw(cameraOf(orbit, up, focal, double(fStop), double(focusDistance)), time, rw, rh,
                               techniqueName);
            if (drawn && !framed) {
                auto bounds = stage.bounds();
                if (bounds && bounds->has_value()) {
                    frameBounds(orbit, **bounds, focal);
                    framed = true;
                    drawn = stage.draw(cameraOf(orbit, up, focal, double(fStop), double(focusDistance)), time, rw, rh,
                               techniqueName);
                }
            }
        } else {
            drawn = stage.draw(cameras[static_cast<size_t>(cameraIndex - 1)], time, rw, rh, techniqueName);
        }
        drawMs.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - drawStart).count());
        status = drawn ? std::string() : drawn.error().toString();
        if (drawn && !announce) {
            techniqueDrawn[techniqueAt] = true;
        }
        if (drawn) {
            if (distinctTimes == 0 || time != drawnTime) {
                ++distinctTimes;
            }
            drawnTime = time;
        }

        // Drawn after the frame, so its numbers are this frame's.
        gaussianReport = stage.gaussianStats();
        ImGui::SetNextWindowPos(ImVec2(730, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 620), ImGuiCond_FirstUseEver);
        gaussiansOpen = ImGui::Begin("Gaussians");
        if (gaussiansOpen) {
            drawPanel(gaussians);
        }
        ImGui::End();

        rhi::ComPtr<rhi::ITexture> image = surface->acquireNextImage();
        ImGui::Render();
        if (!image) {
            continue;
        }
        gpu::CommandBatch batch(device);
        if (drawn) {
            auto source = stage.displaySource(kAovs[static_cast<size_t>(aov)].value);
            if (source) {
                ATHENEA_TRY(display->run(batch, *source, displaySettings(), image.get(), fbw, fbh));
            }
        }
        ATHENEA_TRY((*ui)->render(batch, ImGui::GetDrawData(), image->getDefaultView(), surfaceFormat, fbw, fbh));
        ATHENEA_TRY(batch.submit(false));
        // Every frame as it was shown, panels included, where a capture was
        // asked for: the same path `--snapshot` takes, taken each time round.
        if (!options.capture.empty() && drawn) {
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%05u.png", frames);
            ATHENEA_TRY(capture(stage, *display, **ui, displaySettings(), kAovs[static_cast<size_t>(aov)].value,
                            fbw, fbh, options.capture / name));
        }
        const bool last = options.frames != 0 && frames + 1 == options.frames;
        if (last && !options.snapshot.empty() && drawn) {
            auto shot = snapshot(stage, *display, **ui, displaySettings(), kAovs[static_cast<size_t>(aov)].value,
                                 fbw, fbh, options.snapshot);
            if (!shot) return std::move(shot).error();
            snapshotLit = *shot;
        }
        if (SLANG_FAILED(surface->present())) {
            return Error(ErrorCode::DeviceFailure, "cannot present to the window");
        }
        ++frames;
        frameMs.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart).count());
    }
    {
        gpu::CommandBatch finish(device);
        ATHENEA_TRY(finish.submit(true));
    }
    ui->reset();
    surface.setNull();
    ViewStats stats;
    stats.frames = frames;
    stats.lastTime = drawnTime;
    stats.distinctTimes = distinctTimes;
    stats.snapshotLitPixels = snapshotLit;
    stats.medianDrawMs = median(drawMs);
    stats.medianFrameMs = median(frameMs);
    return stats;
}

}   // namespace athenea::view
