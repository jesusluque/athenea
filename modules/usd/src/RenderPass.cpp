// Copyright (c) 2026 jesus luque.
#include "RenderPass.h"
#include "Camera.h"

#include "RenderParam.h"

#include <pxr/imaging/hd/camera.h>

#include <algorithm>
#include <array>

#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/renderPassState.h>

#include "RenderBuffer.h"
#include "RenderDelegate.h"
#include "athenea/core/Log.h"
#include "athenea/usd/HydraCamera.h"

PXR_NAMESPACE_OPEN_SCOPE

void HdAtheneaRenderPass::_Execute(HdRenderPassStateSharedPtr const& state, TfTokenVector const& renderTags) {
    if (_engine == nullptr) {
        return;
    }
    GfRect2i window;
    const CameraUtilFraming& framing = state->GetFraming();
    if (framing.IsValid()) {
        window = framing.dataWindow;
    } else {
        const GfVec4f vp = state->GetViewport();
        window = GfRect2i(GfVec2i(0), int(vp[2]), int(vp[3]));
    }
    const unsigned width = static_cast<unsigned>(std::max(window.GetWidth(), 1));
    const unsigned height = static_cast<unsigned>(std::max(window.GetHeight(), 1));

    // OUT OF MEMORY is handled here, for every host at once (athenea view,
    // the CLI, usdview, Blender): what the engine can give back it gives
    // back, one thing more is given up (Engine::relieveMemory), and the step
    // is tried again, once. A frame that still fails is reported -- logged,
    // and kept for StageRenderer -- and the next frame tries again from the
    // lower level.
    const auto relieved = [this](const athenea::Error& error) {
        if (error.code() != athenea::ErrorCode::OutOfMemory) {
            return false;
        }
        const std::string did = _engine->relieveMemory();
        if (did.empty()) {
            athenea::log::error("hdAthenea: {}; nothing is left to give back", error.toString());
            return false;
        }
        athenea::log::warn("hdAthenea: {}; trying again with {}", error.toString(), did);
        return true;
    };
    if (auto uploaded = _engine->commit(); !uploaded) {
        athenea::Error error = std::move(uploaded).error();
        bool recovered = false;
        if (relieved(error)) {
            auto again = _engine->commit();
            recovered = again.hasValue();
            if (!recovered) {
                error = std::move(again).error();
            }
        }
        if (!recovered) {
            athenea::log::error("hdAthenea: {}", error.toString());
            _engine->noteFrameError(std::move(error));
            return;
        }
    }
    const GfMatrix4d proj = state->GetProjectionMatrix();
    athenea::render::Projection projection =
        athenea::usd::projectionFromHydra(state->GetWorldToViewMatrix(), proj, width, height);
    // A diaphragm asked for by a caller with a camera of its own -- `athenea view`
    // and `athenea stage --eye` build one out of two matrices, and a lens does not
    // fit in a matrix. It wins over a stage camera's, as the shutter does.
    if (_delegate != nullptr && !_delegate->GetDisableDepthOfField()) {
        if (const auto lens = _delegate->GetLensOverride(); lens) {
            projection.lensRadius = lens->first;
            projection.focusDistance = lens->second;
        }
    }
    // The camera's exposure, in stops, which the renderer applies over the
    // frame it draws; the projection matrix alone does not carry it.
    if (const HdCamera* camera = state->GetCamera(); camera != nullptr) {
        projection.exposure = static_cast<double>(camera->GetExposure());
        // The diaphragm and the distortion, which only the path tracer's own
        // rays can honour. HdCamera's focal length is already in scene units.
        if (camera->GetFStop() > 0.0F && (_delegate == nullptr || !_delegate->GetDisableDepthOfField())) {
            projection.lensRadius = static_cast<double>(camera->GetFocalLength()) /
                                    (2.0 * static_cast<double>(camera->GetFStop()));
            projection.focusDistance = static_cast<double>(camera->GetFocusDistance());
        }
        // A camera that moves under the shutter: view to world at its two
        // samples (the camera's transform, then the flip to +z).
        if (const auto* moving = dynamic_cast<const HdAtheneaCamera*>(camera); moving != nullptr && moving->Moves()) {
            const athenea::render::Mat4 flip = aofx::xform::scaling({1.0, 1.0, -1.0});
            projection.cameraMoves = true;
            projection.viewToWorldStart = athenea::usd::fromUsd(moving->GetTransformStart()) * flip;
            projection.viewToWorldEnd = athenea::usd::fromUsd(moving->GetTransformEnd()) * flip;
            projection.cameraTimeStart = moving->GetTimeStart();
            projection.cameraTimeEnd = moving->GetTimeEnd();
        }
        if (camera->GetLensDistortionType() == HdCameraTokens->standard) {
            projection.distortionK1 = static_cast<double>(camera->GetLensDistortionK1());
            projection.distortionK2 = static_cast<double>(camera->GetLensDistortionK2());
        }
        // The shutter, for Sync to sample at. Sync ran before this pass, so a
        // change here reaches the prims on the next frame: every prim's
        // transform and primvars dirtied through the delegate's scene index,
        // which every chain built for this renderer holds.
        if (auto* param = _delegate != nullptr ? static_cast<HdAtheneaRenderParam*>(_delegate->GetRenderParam())
                                               : nullptr;
            param != nullptr) {
            // A product that switched motion blur off draws the frame, so
            // its prims are sampled at the frame and not about the shutter.
            const bool blurs = _delegate == nullptr || !_delegate->GetDisableMotionBlur();
            // A camera of the engine's own authors no shutter; `athenea:shutter`
            // is how a caller says one. Consulted here rather than set behind
            // this line's back, which would be overwritten on the next frame.
            const std::optional<std::pair<double, double>> asked =
                _delegate != nullptr ? _delegate->GetShutterOverride() : std::nullopt;
            const double open = blurs ? (asked ? asked->first : camera->GetShutterOpen()) : 0.0;
            const double close = blurs ? (asked ? asked->second : camera->GetShutterClose()) : 0.0;
            _engine->setShutter(open, close);
            if (open != param->GetShutterOpen() || close != param->GetShutterClose()) {
                param->SetShutter(open, close);
                if (_delegate == nullptr || !_delegate->ResampleAllPrims()) {
                    athenea::log::warn("hdAthenea: the shutter changed and no scene index of this renderer's is in the "
                                   "chain to resample the prims through; they keep the old samples");
                }
            }
        }
    }

    athenea::render::RenderSettings settings;
    settings.width = width;
    settings.height = height;
    // Every AOV binding: what each reads, and what the frame must compute.
    struct Output {
        HdAtheneaRenderBuffer*      buffer;
        athenea::usd::AovSource     source;
    };
    std::vector<Output> outputs;
    athenea::usd::AovRequest request;
    for (HdRenderPassAovBinding const& binding : state->GetAovBindings()) {
        auto* buffer = dynamic_cast<HdAtheneaRenderBuffer*>(binding.renderBuffer);
        if (buffer == nullptr) {
            continue;
        }
        const std::string& name = binding.aovName.GetString();
        athenea::usd::AovSource source;
        if (binding.aovName == HdAovTokens->color) {
            source.kind = athenea::usd::AovKind::Colour;
            if (binding.clearValue.IsHolding<GfVec4f>()) {
                const GfVec4f c = binding.clearValue.UncheckedGet<GfVec4f>();
                settings.background = {c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]};
            }
        } else if (binding.aovName == HdAovTokens->depth) {
            source.kind = athenea::usd::AovKind::Depth;
        } else if (binding.aovName == HdAovTokens->primId) {
            source.kind = athenea::usd::AovKind::PrimId;
            request.ids = true;
        } else if (binding.aovName == HdAovTokens->instanceId) {
            source.kind = athenea::usd::AovKind::InstanceId;
            request.ids = true;
        } else if (binding.aovName == HdAovTokens->elementId) {
            source.kind = athenea::usd::AovKind::ElementId;
            request.ids = true;
        } else if (binding.aovName == TfToken("albedo")) {
            source.kind = athenea::usd::AovKind::Albedo;
            request.aux = true;
        } else if (binding.aovName == TfToken("shadingNormal")) {
            source.kind = athenea::usd::AovKind::ShadingNormal;
            request.aux = true;
        } else if (binding.aovName == HdAovTokens->Neye) {
            source.kind = athenea::usd::AovKind::EyeNormal;
            request.normals = true;
        } else if (binding.aovName == HdAovTokens->normal) {
            source.kind = athenea::usd::AovKind::WorldNormal;
            request.normals = true;
        } else if (name.rfind("CryptoObject", 0) == 0 && name.size() == 14) {
            // CryptoObject00, 01, 02: the three layers of the one matte this
            // engine keeps. Anything else under the name is not a layer.
            const char tens = name[12];
            const char units = name[13];
            if (tens != '0' || units < '0' || units > '2') {
                continue;
            }
            source.kind = athenea::usd::AovKind::Crypto;
            source.primvar = static_cast<uint32_t>(units - '0');
            request.cryptomatte = true;
        } else if (name.rfind("primvars:", 0) == 0) {
            source.kind = athenea::usd::AovKind::Primvar;
            source.primvar = static_cast<uint32_t>(request.primvars.size());
            request.primvars.push_back(name.substr(9));
        } else if (name.rfind("lightGroup:", 0) == 0) {
            source.kind = athenea::usd::AovKind::LightGroup;
            source.primvar = static_cast<uint32_t>(request.lightGroups.size());
            request.lightGroups.push_back(name.substr(11));
        } else {
            continue;
        }
        outputs.push_back({buffer, source});
    }

    const auto technique = _delegate != nullptr ? _delegate->GetTechnique() : athenea::usd::Technique::Raster;
    const bool settle = _delegate != nullptr && _delegate->GetSettleStreams();
    const auto visibility =
        _delegate != nullptr ? _delegate->GetMeshVisibility() : athenea::usd::MeshVisibility::Automatic;
    if (_delegate != nullptr) {
        _engine->setLightSamples(_delegate->GetLightSamples());
        _engine->setChooseLights(_delegate->GetChooseLights());
        _engine->setSplatShadows(_delegate->GetSplatShadows());
        _engine->setSplatTransferIndirect(_delegate->GetSplatTransferIndirect());
        _engine->setSplatReflections(_delegate->GetSplatReflections());
        _engine->setCloudShadows(_delegate->GetCloudShadows());
        _engine->setDomePrefiltered(_delegate->GetDomePrefiltered());
        _engine->setCloudShadowResolution(_delegate->GetCloudShadowResolution());
        _engine->setCloudShadowTerms(_delegate->GetCloudShadowTerms());
        _engine->setCloudShadowDensity(_delegate->GetCloudShadowDensity());
        _engine->setAntialias(_delegate->GetAntialias());
        _engine->setPathSamples(_delegate->GetPathSamples());
        _engine->setPathBounces(_delegate->GetPathBounces());
        _engine->setPathTotal(_delegate->GetPathTotal());
        _engine->setDenoise(_delegate->GetDenoise());
        _engine->setPathAdaptive(_delegate->GetPathAdaptive());
        _engine->setPathMis(_delegate->GetPathMis());
        _engine->setPathError(_delegate->GetPathError());
        _engine->setMotionBuckets(_delegate->GetDisableMotionBlur() ? 1u : _delegate->GetMotionBuckets());
    }
    if (auto drawn =
            _engine->render(projection, settings, *_targets, technique, settle, &renderTags, request, visibility);
        !drawn) {
        athenea::Error error = std::move(drawn).error();
        bool recovered = false;
        if (relieved(error)) {
            // Committed again first: a stream given up is opened again there,
            // at the budget the step down allows.
            athenea::Result<void> again = athenea::ok();
            if (auto recommitted = _engine->commit(); !recommitted) {
                again = std::move(recommitted).error();
            } else {
                again = _engine->render(projection, settings, *_targets, technique, settle, &renderTags, request,
                                        visibility);
            }
            recovered = again.hasValue();
            if (!recovered) {
                error = std::move(again).error();
            }
        }
        if (!recovered) {
            athenea::log::error("hdAthenea: {}", error.toString());
            _engine->noteFrameError(std::move(error));
            return;
        }
    }
    for (const Output& output : outputs) {
        HdAtheneaRenderBuffer* buffer = output.buffer;
        if (buffer->GetWidth() != width || buffer->GetHeight() != height) {
            continue;
        }
        const HdFormat component = HdGetComponentFormat(buffer->GetFormat());
        athenea::usd::AovLayout layout;
        layout.channels = static_cast<uint32_t>(HdGetComponentCount(buffer->GetFormat()));
        layout.componentBytes = static_cast<uint32_t>(HdDataSizeOfFormat(component));
        switch (component) {
        case HdFormatUNorm8: layout.componentKind = 0; break;
        case HdFormatFloat16: layout.componentKind = 1; break;
        case HdFormatFloat32: layout.componentKind = 2; break;
        case HdFormatInt32: layout.componentKind = 3; break;
        default:
            athenea::log::warn("hdAthenea: render buffer format {} is not filled", static_cast<int>(buffer->GetFormat()));
            continue;
        }
        std::array<double, 16> hostProjection{};
        std::copy(proj.data(), proj.data() + 16, hostProjection.begin());
        buffer->SetPendingFill([engine = _engine, targets = _targets, source = output.source, layout,
                                hostProjection](std::span<uint8_t> into) {
            if (auto written = engine->writeAov(*targets, source, layout, hostProjection.data(), into); !written) {
                athenea::log::error("hdAthenea: {}", written.error().toString());
            }
        });
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
