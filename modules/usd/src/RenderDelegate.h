// Copyright (c) 2026 jesus luque.
//
// athenea as a Hydra 2.0 render delegate: USD is the scene, the engine
// is the renderer. ParticleField3DGaussianSplat and Points are drawn; cameras
// come through HdCamera; AOVs are colour and depth.
#pragma once

#include <optional>
#include <utility>
#include <memory>

#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneIndex.h>

#include "Engine.h"
#include "RenderParam.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdAtheneaRenderDelegate final : public HdRenderDelegate {
public:
    HdAtheneaRenderDelegate();
    explicit HdAtheneaRenderDelegate(HdRenderSettingsMap const& settings);
    /// On a device the host opened (Engine::create's second form); what
    /// `StageRenderer::open(path, device)` builds. The plugin registry's
    /// delegates, made by Hydra itself, open their own.
    explicit HdAtheneaRenderDelegate(std::shared_ptr<athenea::gpu::Device> device);
    ~HdAtheneaRenderDelegate() override;

    TfTokenVector const& GetSupportedRprimTypes() const override;
    TfTokenVector const& GetSupportedSprimTypes() const override;
    TfTokenVector const& GetSupportedBprimTypes() const override;
    TfTokenVector        GetRenderSettingsNamespaces() const override;
    HdRenderParam* GetRenderParam() const override { return _param.get(); }

    /// Kept, so that a change the scene does not carry can be sent through it.
    void SetTerminalSceneIndex(const HdSceneIndexBaseRefPtr& terminalSceneIndex) override;
    /// Every prim's transform and primvars sampled again -- the shutter
    /// changed -- through the HdAtheneaResampleSceneIndex in the host's chain.
    /// False when none is there (a chain not built for this renderer).
    bool ResampleAllPrims() const;
    /// Every prim at and beneath `root` has its visibility read again: what
    /// an instancer's visibility changing needs, since its prototypes carry
    /// their own and Hydra does not dirty them for the instancer's. False
    /// when no HdAtheneaResampleSceneIndex is in the chain.
    bool DirtyVisibilityBelow(const SdfPath& root) const;
    HdResourceRegistrySharedPtr GetResourceRegistry() const override { return _registry; }

    HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex* index,
                                           HdRprimCollection const& collection) override;
    HdInstancer* CreateInstancer(HdSceneDelegate* delegate, SdfPath const& id) override;
    void DestroyInstancer(HdInstancer* instancer) override;

    HdRprim* CreateRprim(TfToken const& typeId, SdfPath const& rprimId) override;
    void DestroyRprim(HdRprim* rprim) override;
    HdSprim* CreateSprim(TfToken const& typeId, SdfPath const& sprimId) override;
    HdSprim* CreateFallbackSprim(TfToken const& typeId) override;
    void DestroySprim(HdSprim* sprim) override;
    HdBprim* CreateBprim(TfToken const& typeId, SdfPath const& bprimId) override;
    HdBprim* CreateFallbackBprim(TfToken const& typeId) override;
    void DestroyBprim(HdBprim* bprim) override;
    void CommitResources(HdChangeTracker*) override {}

    HdAovDescriptor GetDefaultAovDescriptor(TfToken const& name) const override;

    /// `athenea:technique`: "raster" (default) or "rt".
    HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override;
    TfTokenVector GetMaterialRenderContexts() const override;
    /// The binding purpose a mesh's material is resolved with, falling back
    /// to the all-purpose binding: a renderer's "full", not Storm's "preview".
    /// (StageRenderer resolves render settings' purposes before this is asked.)
    TfToken GetMaterialBindingPurpose() const override { return HdTokens->full; }
    [[nodiscard]] athenea::usd::Technique GetTechnique() const;
    /// `athenea:settleStreams`: false (default) for a viewport, which lets streamed
    /// assets fill in over frames; true for an image that must be complete.
    [[nodiscard]] bool GetSettleStreams() const;

    /// Samples per light per pixel ("athenea:lightSamples"), at least one.
    [[nodiscard]] uint32_t GetLightSamples() const;
    /// Whether to choose one light a sample ("athenea:chooseLights").
    [[nodiscard]] bool GetChooseLights() const;
    /// `athenea:splatShadows`: a relit splat casts a shadow ray against the
    /// cloud's own proxies.
    [[nodiscard]] bool GetSplatShadows() const;
    [[nodiscard]] bool GetCloudShadows() const;
    [[nodiscard]] uint32_t GetCloudShadowResolution() const;
    [[nodiscard]] uint32_t GetCloudShadowTerms() const;
    [[nodiscard]] float GetCloudShadowDensity() const;
    [[nodiscard]] bool GetAntialias() const;
    /// Paths a pixel a path traced frame gathers ("athenea:pathSamples"), at
    /// least one, and bounces after the first hit ("athenea:pathBounces").
    [[nodiscard]] uint32_t GetPathSamples() const;
    [[nodiscard]] uint32_t GetPathBounces() const;
    [[nodiscard]] uint32_t GetMotionBuckets() const;
    /// A render product's (or its settings prim's) disableMotionBlur and
    /// disableDepthOfField: "athenea:disableMotionBlur" draws one shutter slice,
    /// "athenea:disableDepthOfField" a pinhole through the camera's lens.
    [[nodiscard]] bool GetDisableMotionBlur() const;
    /// `athenea:shutter`: open and close for a camera that authors none of its
    /// own, which is every camera the engine makes for itself.
    [[nodiscard]] std::optional<std::pair<double, double>> GetShutterOverride() const;
    /// (lens radius, focus distance) for a camera the engine made itself.
    [[nodiscard]] std::optional<std::pair<double, double>> GetLensOverride() const;
    [[nodiscard]] bool GetDisableDepthOfField() const;
    /// Paths a pixel at which a path traced frame is finished
    /// ("athenea:pathTotal"); one, the default, never accumulates.
    [[nodiscard]] uint32_t GetPathTotal() const;
    /// Denoise a path traced frame once it is gathered ("athenea:denoise").
    [[nodiscard]] bool GetDenoise() const;
    /// Adaptive sampling ("athenea:pathAdaptive") and its relative error target
    /// ("athenea:pathError").
    [[nodiscard]] bool GetPathAdaptive() const;
    /// "athenea:splatReflections": a cloud reflects itself rather than only the
    /// prepared sky, at one ray a gaussian a frame (false, the default).
    [[nodiscard]] bool GetSplatReflections() const;

    /// "athenea:splatTransferIndirect": a transferred cloud adds the indirect
    /// half it carries (true, the default). Off draws the same cloud with its
    /// direct transfer alone, which is what the interreflection is worth.
    [[nodiscard]] bool GetSplatTransferIndirect() const;

    /// "athenea:pathMis": light and material sampling weighed (true, the default).
    [[nodiscard]] bool GetPathMis() const;
    [[nodiscard]] float GetPathError() const;
    /// `athenea:visibility`: "automatic" (default), "raster", "rays" or "bvh".
    [[nodiscard]] athenea::usd::MeshVisibility GetMeshVisibility() const;

    /// True when a device opened; a delegate without one draws nothing.
    [[nodiscard]] bool HasEngine() const { return _engine != nullptr; }
    /// Why there is no engine, where there is none: a host with no console to
    /// read -- an app on a phone -- says why instead of only that.
    [[nodiscard]] const std::string& EngineError() const { return _engineError; }
    /// The engine's device; only valid when HasEngine().
    [[nodiscard]] athenea::gpu::Device& GetEngineDevice() const { return _engine->device(); }
    [[nodiscard]] athenea::gpu::ShaderLibrary& GetEngineLibrary() const { return _engine->library(); }
    /// The engine itself, for what a task cannot ask for through Hydra: the
    /// bake, which is a frame at points a caller names rather than at pixels.
    /// Only valid when HasEngine(), and only from the thread that renders.
    [[nodiscard]] athenea::usd::Engine& GetEngine() const { return *_engine; }

private:
    void _Setup();

    /// The host's device, or null to open one. Kept for `_Setup` only: the
    /// engine holds its own share once made.
    std::shared_ptr<athenea::gpu::Device>  _device;
    std::unique_ptr<athenea::usd::Engine>  _engine;

    std::string _engineError;
    std::unique_ptr<HdAtheneaRenderParam>  _param;
    HdResourceRegistrySharedPtr        _registry;
    HdSceneIndexBasePtr                _terminal;   ///< weak: the render index owns it
};

/// Registers the scene indices this renderer needs (light linking), once.
/// Hosts that build the delegate themselves must call it; the plugin does.
void HdAtheneaRegisterSceneIndices();

PXR_NAMESPACE_CLOSE_SCOPE
