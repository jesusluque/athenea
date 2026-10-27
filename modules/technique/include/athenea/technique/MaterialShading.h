// Copyright (c) 2026 jesus luque.
//
// A visibility buffer shaded by its instances' MaterialX materials: one
// kernel over MaterialPrograms' generated dispatch that reconstructs each
// pixel's surface (material_surface.slang), evaluates its material into a
// lobe stack, and lights it -- by the frame's lights, with shadows where the
// device traces rays, linking, and either every light at every pixel or one
// chosen a sample by its power. A frame with no lights at all falls back to
// the headlight, a unit light from the eye, as HeadlightShading draws
// unshaded meshes.
//
// Where the device traces rays it is three kernels a frame: drawLobes
// evaluates the material and writes its lobe samples' directions,
// traceShadows traces every light sample's and lobe sample's shadow ray into
// one bit each, and shadeMaterials -- which holds no intersector -- lights
// the pixel by them. A ray query beside the lobe stack miscompiled on Metal.
#pragma once

#include <optional>
#include <string>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Texture.h"
#include "athenea/render/Camera.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/light/LtcTable.h"
#include "athenea/technique/MaterialPrograms.h"
#include "athenea/technique/Visibility.h"

namespace athenea::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace athenea::technique {

class MaterialShading {
public:
    [[nodiscard]] static Result<MaterialShading> create(gpu::ShaderLibrary& library);

    /// The kernel is rebuilt when `programs` dispatches to a different set.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs);
    /// What the mesh lights' closed form needs before a frame's batch opens:
    /// the LTC table (fitted on the device the first time, a wait) and its
    /// kernels. A frame drawn without it samples them instead.
    [[nodiscard]] Result<void> prepareClosedForm();
    /// The same, for a frame with light groups (their buffers declared) or
    /// without; `shade` switches as the frame asks.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs, bool groups);
    /// And with the cloud shadow map, which a frame has only where a cloud
    /// casts one: a frame without compiles the kernel it always did.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs, bool groups, bool clouds);
    /// And with the domes read prefiltered (`MaterialFrame::domeLighting`):
    /// a frame that samples them compiles the kernel it always did.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs, bool groups, bool clouds, bool domes);

    [[nodiscard]] Result<void> shade(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                                     const render::Projection& projection, const MaterialFrame& frame,
                                     render::RenderTargets& out);

    /// ATHENEA_STAGES: submit and wait after each kernel, and keep how long
    /// each took (milliseconds), so a frame can say where its meshes went.
    struct StageTimes {
        double lobes = 0.0;    ///< drawLobes
        double shadows = 0.0;  ///< traceShadows
        double shade = 0.0;    ///< shadeMaterials
        double domes = 0.0;    ///< domeShade, the domes read prefiltered
        double reflections = 0.0;   ///< shadeReflections, the screen-space reflections
    };
    void timeStages(bool on) noexcept { timeStages_ = on; }
    [[nodiscard]] const StageTimes& stageTimes() const noexcept { return times_; }

private:
    gpu::ShaderLibrary*              library_ = nullptr;
    gpu::Device*                     device_ = nullptr;
    std::optional<gpu::ComputeKernel> kernel_;
    /// The shadow rays' kernel (traceShadows), and the materials it was made for.
    std::optional<gpu::ComputeKernel> trace_;
    std::string                      traceModule_;
    /// One bit a light sample a pixel: what traceShadows found blocked.
    gpu::Buffer                      shadowBits_;
    /// The kernel that draws the lobe samples' directions (drawLobes), and them.
    std::optional<gpu::ComputeKernel> lobes_;
    gpu::Buffer                      lobeDirs_;
    std::string                      module_;
    bool                             groups_ = false;
    bool                             clouds_ = false;
    bool                             domes_ = false;
    /// The device would not make the kernel with the domes read prefiltered:
    /// every kernel after is made with them sampled.
    bool                             domesRefused_ = false;
    /// The domes read prefiltered (dome_shade.slang), and what the shading
    /// kernel leaves it a pixel: (diffuse albedo, coverage), (glossy albedo,
    /// alpha), (position, normal).
    std::optional<gpu::ComputeKernel> domeKernel_;
    gpu::Texture                     domeDiffuse_;
    gpu::Texture                     domeGlossy_;
    gpu::Texture                     domePoint_;
    rhi::ComPtr<rhi::ITextureView>   domeDiffuseView_;
    rhi::ComPtr<rhi::ITextureView>   domeGlossyView_;
    rhi::ComPtr<rhi::ITextureView>   domePointView_;
    /// What stands for the cloud map where a frame has none.
    gpu::Texture                     domeNoMap_;
    rhi::ComPtr<rhi::ITextureView>   domeNoMapView_;
    /// The table of materials the same everywhere (tabulateMaterials), made
    /// for the materials it was compiled with, and the texture it fills.
    std::optional<gpu::ComputeKernel> table_;
    std::optional<gpu::ComputeKernel> tabled_;   ///< shadeTabled: the pixels the table shades
    std::string                      tableModule_;
    gpu::Texture                     tableTexture_;
    rhi::ComPtr<rhi::ITextureView>   tableView_;
    /// shadeEvaluated: the pixels whose material reflects only, under the
    /// domes alone, evaluated without the lights' machinery; and the mask of
    /// the pixels it took, which the shading kernel skips. Empty where the
    /// device would not make it: the shading kernel then shades them all.
    std::optional<gpu::ComputeKernel> evaluated_;
    std::optional<gpu::ComputeKernel> unlit_;   ///< shadeUnlit: the unlit surfaces' pixels
    std::optional<gpu::ComputeKernel> reflections_;   ///< shadeReflections: the screen-space reflections
    gpu::Buffer                       reflectionSource_;   ///< what was shaded, read by the reflections
    // THE MESH LIGHTS IN CLOSED FORM: the LTC table (fitted once), the
    // tiles' lightcuts, the shadow ratio and its filtered texture, and the
    // kernels that make the last three.
    std::optional<light::LtcTable>    ltc_;
    std::optional<gpu::ComputeKernel> meshCut_;
    std::optional<gpu::ComputeKernel> meshFilter_;
    gpu::Texture                      meshCutTexture_;
    rhi::ComPtr<rhi::ITextureView>    meshCutView_;
    gpu::Texture                      meshShadowTexture_;
    rhi::ComPtr<rhi::ITextureView>    meshShadowView_;
    gpu::Buffer                       meshRatio_;
    [[nodiscard]] Result<void> ensureMeshTextures(uint32_t cutWidth, uint32_t cutHeight, uint32_t width,
                                                  uint32_t height);
    gpu::Texture                     evaluatedMask_;
    rhi::ComPtr<rhi::ITextureView>   evaluatedMaskView_;
    bool                             timeStages_ = false;
    StageTimes                       times_;
    /// Whether the kernel in use reads the shadow rays' answers (traceShadows').
    bool                             shadowed_ = false;
    /// The device would not make the shadowed kernel: every kernel after is
    /// made without shadow rays (setPrograms).
    bool                             shadowsRefused_ = false;
};

}   // namespace athenea::technique
