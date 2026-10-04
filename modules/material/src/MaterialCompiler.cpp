// Copyright (c) 2026 jesus luque.
#include "athenea/material/MaterialCompiler.h"

#include "athenea/core/Log.h"

#include <atomic>
#include <cstdio>
#include <functional>
#include <sstream>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenHw/HwConstants.h>
#include <MaterialXGenHw/Nodes/HwSurfaceNode.h>
#include <MaterialXGenShader/Exception.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Syntax.h>
#include <MaterialXGenShader/TypeDesc.h>
#include <MaterialXGenShader/Util.h>
#include <MaterialXGenSlang/SlangShaderGenerator.h>

namespace mx = MaterialX;

namespace athenea::material {

namespace {

const std::string kFunctionPlaceholder = "ATHENEA_MATERIAL_FUNCTION";
/// MaterialX's UsdPreviewSurface nodegraph, replaced by the engine's own.
const std::string kPreviewSurfaceGraph = "IMP_UsdPreviewSurface_surfaceshader";

/// The surface node: no light loop. Its BSDF graph runs once, pushing lobes;
/// what it weights them by becomes the material's lobe stack.
class AtheneaSurfaceNode : public mx::HwSurfaceNode {
public:
    static mx::ShaderNodeImplPtr create() { return std::make_shared<AtheneaSurfaceNode>(); }

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override {
        if (stage.getName() != mx::Stage::PIXEL) {
            return;
        }
        const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
        const std::string prefix = mx::HW::T_VERTEX_DATA_INSTANCE + ".";
        const mx::ShaderOutput* output = node.getOutput();
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, true, context, stage);
        shadergen.emitLineEnd(stage);
        shadergen.emitScopeBegin(stage);
        shadergen.emitLine("float3 N = normalize(" + prefix + mx::HW::T_NORMAL_WORLD + ")", stage);
        shadergen.emitLine("float3 V = normalize(" + mx::HW::T_VIEW_POSITION + " - " + prefix +
                               mx::HW::T_POSITION_WORLD + ")",
                           stage);
        shadergen.emitLine("float3 P = " + prefix + mx::HW::T_POSITION_WORLD, stage);
        shadergen.emitLine("float3 L = float3(0.0, 0.0, 0.0)", stage);
        shadergen.emitLine("float occlusion = 1.0", stage);
        shadergen.emitLineBegin(stage);
        shadergen.emitString("float surfaceOpacity = ", stage);
        shadergen.emitInput(node.getInput(getOpacityInputName()), context, stage);
        shadergen.emitLineEnd(stage);
        const std::string outColor = output->getVariable() + ".color";
        const std::string outTransparency = output->getVariable() + ".transparency";
        // A THIN WALL is the surface's, not a BSDF's (MaterialX puts it on the
        // `surface` constructor; OpenPBR wires `geometry_thin_walled` there):
        // said before the BSDF builds, so the dielectrics it pushes know.
        if (const mx::ShaderInput* thin = node.getInput("thin_walled")) {
            shadergen.emitLineBegin(stage);
            shadergen.emitString("gAtheneaThinWalled = ", stage);
            shadergen.emitInput(thin, context, stage);
            shadergen.emitLineEnd(stage);
        }
        if (const mx::ShaderInput* bsdfInput = node.getInput(getBsdfInputName())) {
            if (const mx::ShaderNode* bsdf = bsdfInput->getConnectedSibling()) {
                shadergen.emitLine("ClosureData closureData = makeClosureData(CLOSURE_TYPE_REFLECTION, L, V, N, P, "
                                   "occlusion)",
                                   stage);
                shadergen.emitFunctionCall(*bsdf, context, stage);
                shadergen.emitLine("gAtheneaResult = atheneaFinishStack(" + bsdf->getOutput()->getVariable() +
                                       ", gAtheneaResult.emission, surfaceOpacity)",
                                   stage);
            }
        }
        if (const mx::ShaderInput* edfInput = node.getInput(getEdfInputName())) {
            if (const mx::ShaderNode* edf = edfInput->getConnectedSibling()) {
                shadergen.emitScopeBegin(stage);
                shadergen.emitLine("ClosureData closureData = makeClosureData(CLOSURE_TYPE_EMISSION, L, V, N, P, "
                                   "occlusion)",
                                   stage);
                shadergen.emitFunctionCall(*edf, context, stage);
                shadergen.emitLine(outColor + " += " + edf->getOutput()->getVariable(), stage);
                shadergen.emitScopeEnd(stage);
            }
        }
        shadergen.emitLine(outTransparency + " = float3(1.0 - surfaceOpacity)", stage);
        shadergen.emitScopeEnd(stage);
        shadergen.emitLineBreak(stage);
    }
};

/// A surface's opacity alone: what a shadow ray asks a cut-out, so the BSDF,
/// the emission and everything only they read are dead code the compiler
/// drops. The function around it turns the transparency into
/// `gAtheneaResult.opacity`, as it does for the lobes.
class AtheneaOpacitySurfaceNode : public mx::HwSurfaceNode {
public:
    static mx::ShaderNodeImplPtr create() { return std::make_shared<AtheneaOpacitySurfaceNode>(); }

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override {
        if (stage.getName() != mx::Stage::PIXEL) {
            return;
        }
        const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
        const mx::ShaderOutput* output = node.getOutput();
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, true, context, stage);
        shadergen.emitLineEnd(stage);
        shadergen.emitScopeBegin(stage);
        shadergen.emitLineBegin(stage);
        shadergen.emitString("float surfaceOpacity = ", stage);
        shadergen.emitInput(node.getInput(getOpacityInputName()), context, stage);
        shadergen.emitLineEnd(stage);
        shadergen.emitLine(output->getVariable() + ".transparency = float3(1.0 - surfaceOpacity)", stage);
        shadergen.emitScopeEnd(stage);
        shadergen.emitLineBreak(stage);
    }
};

/// The genglsl reference's surface node: the BSDF graph as MaterialX
/// evaluates it for one light, gAtheneaReferenceL, its response left in
/// gAtheneaReferenceResponse.
class AtheneaReferenceSurfaceNode : public mx::HwSurfaceNode {
public:
    static mx::ShaderNodeImplPtr create() { return std::make_shared<AtheneaReferenceSurfaceNode>(); }

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override {
        if (stage.getName() != mx::Stage::PIXEL) {
            return;
        }
        const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
        const std::string prefix = mx::HW::T_VERTEX_DATA_INSTANCE + ".";
        const mx::ShaderOutput* output = node.getOutput();
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, true, context, stage);
        shadergen.emitLineEnd(stage);
        shadergen.emitScopeBegin(stage);
        shadergen.emitLine("float3 N = normalize(" + prefix + mx::HW::T_NORMAL_WORLD + ")", stage);
        shadergen.emitLine("float3 V = normalize(" + mx::HW::T_VIEW_POSITION + " - " + prefix +
                               mx::HW::T_POSITION_WORLD + ")",
                           stage);
        shadergen.emitLine("float3 P = " + prefix + mx::HW::T_POSITION_WORLD, stage);
        shadergen.emitLine("float3 L = gAtheneaReferenceL", stage);
        shadergen.emitLine("float occlusion = 1.0", stage);
        shadergen.emitLineBegin(stage);
        shadergen.emitString("float surfaceOpacity = ", stage);
        shadergen.emitInput(node.getInput(getOpacityInputName()), context, stage);
        shadergen.emitLineEnd(stage);
        if (const mx::ShaderInput* bsdfInput = node.getInput(getBsdfInputName())) {
            if (const mx::ShaderNode* bsdf = bsdfInput->getConnectedSibling()) {
                shadergen.emitLine("ClosureData closureData = makeClosureData(CLOSURE_TYPE_REFLECTION, L, V, N, P, "
                                   "occlusion)",
                                   stage);
                shadergen.emitFunctionCall(*bsdf, context, stage);
                shadergen.emitLine("gAtheneaReferenceResponse = " + bsdf->getOutput()->getVariable() + ".response",
                                   stage);
            }
        }
        if (const mx::ShaderInput* edfInput = node.getInput(getEdfInputName())) {
            if (const mx::ShaderNode* edf = edfInput->getConnectedSibling()) {
                shadergen.emitScopeBegin(stage);
                shadergen.emitLine("ClosureData closureData = makeClosureData(CLOSURE_TYPE_EMISSION, L, V, N, P, "
                                   "occlusion)",
                                   stage);
                shadergen.emitFunctionCall(*edf, context, stage);
                shadergen.emitLine(output->getVariable() + ".color += " + edf->getOutput()->getVariable(), stage);
                shadergen.emitScopeEnd(stage);
            }
        }
        shadergen.emitLine(output->getVariable() + ".transparency = float3(1.0 - surfaceOpacity)", stage);
        shadergen.emitScopeEnd(stage);
        shadergen.emitLineBreak(stage);
    }
};

mx::ShaderNodeImplPtr createTransformPoint();
mx::ShaderNodeImplPtr createTransformVector();
mx::ShaderNodeImplPtr createTransformNormal();

class AtheneaSlangShaderGenerator : public mx::SlangShaderGenerator {
public:
    AtheneaSlangShaderGenerator(mx::TypeSystemPtr types, ClosureVariant variant)
        : mx::SlangShaderGenerator(types), variant_(variant) {
        const bool lobes = variant != ClosureVariant::GenglslReference;
        registerImplementation("IM_surface_" + TARGET, variant == ClosureVariant::Opacity ? AtheneaOpacitySurfaceNode::create
                                                       : lobes ? AtheneaSurfaceNode::create
                                                               : AtheneaReferenceSurfaceNode::create);
        // Transforms between object, world and a coordinate system bound to
        // the prim, from the shading point's own transform: genslang's read
        // matrices from uniforms nothing set (identity) and knew no systems.
        registerImplementation("IM_transformpoint_vector3_" + TARGET, createTransformPoint);
        registerImplementation("IM_transformvector_vector3_" + TARGET, createTransformVector);
        registerImplementation("IM_transformnormal_vector3_" + TARGET, createTransformNormal);
        // The closure and shader types live in material_runtime.slang.
        const auto aggregate = [&](mx::TypeDesc type, const std::string& name, const std::string& value) {
            _syntax->registerTypeSyntax(type, std::make_shared<mx::AggregateTypeSyntax>(
                                                  _syntax.get(), name, value, mx::EMPTY_STRING, mx::EMPTY_STRING,
                                                  mx::EMPTY_STRING));
        };
        if (lobes) {
            aggregate(mx::Type::BSDF, "BSDF", "athenea_bsdf_zero()");
        }
        aggregate(mx::Type::VDF, "VDF", "VDF(float3(0.0),float3(1.0))");
        aggregate(mx::Type::SURFACESHADER, "surfaceshader", "surfaceshader(float3(0.0),float3(0.0))");
        aggregate(mx::Type::VOLUMESHADER, "volumeshader", "volumeshader(float3(0.0),float3(0.0))");
        aggregate(mx::Type::DISPLACEMENTSHADER, "displacementshader", "displacementshader(float3(0.0),1.0)");
        aggregate(mx::Type::LIGHTSHADER, "lightshader", "lightshader(float3(0.0),float3(0.0))");
    }

    const ClosureVariant variant_;

    /// Filled by generate: the blob layout, in declaration order.
    mutable std::vector<MaterialSlot> slots;
    mutable uint32_t words = 0;

    /// What colour space each file is read in, by the path itself.
    ///
    /// A shader port does not always carry the colour space its document
    /// input had: MaterialX puts one on a port for a colour management system
    /// to act on, and this generator registers none -- it does the decode on
    /// the device, in `TextureStore`, where the texture already is. So the
    /// document is read for it before generating, and a file input whose port
    /// says nothing takes its answer from here. Keyed by the path because that
    /// is what both sides have: two nodes reading one file in two colour
    /// spaces would be a document contradicting itself.
    mutable std::map<std::string, std::string> fileColourSpaces;

    mx::ShaderPtr generate(const std::string& name, mx::ElementPtr element, mx::GenContext& context) const override {
        slots.clear();
        words = 0;
        mx::ShaderPtr shader = createShader(name, element, context);
        mx::ScopedFloatFormatting fmt(mx::Value::FloatFormatFixed);
        mx::ShaderStage& ps = shader->getStage(mx::Stage::PIXEL);
        emitMaterial(shader->getGraph(), context, ps);
        replaceTokens(_tokenSubstitutions, ps);
        SlangSyntaxFromGlsl(ps);
        return shader;
    }

    /// A primvar's scene slot as the blob holds it, read in generated code.
    std::string primvarSlotRead(const std::string& primvar) const {
        MaterialSlot& slot = addSlot(MaterialSlot::Kind::Primvar, primvar, 1);
        slot.name = primvar;
        return "uint(atheneaBlob(" + std::to_string(slot.offset) + "u))";
    }

private:
    MaterialSlot& addSlot(MaterialSlot::Kind kind, const std::string& variable, uint32_t count) const {
        MaterialSlot slot;
        slot.kind = kind;
        slot.variable = variable;
        slot.offset = words;
        slot.words = count;
        words += count;
        slots.push_back(std::move(slot));
        return slots.back();
    }

    static uint32_t wordsOf(mx::TypeDesc type) {
        if (type == mx::Type::FLOAT || type == mx::Type::INTEGER || type == mx::Type::BOOLEAN) return 1;
        if (type == mx::Type::VECTOR2) return 2;
        if (type == mx::Type::VECTOR3 || type == mx::Type::COLOR3) return 3;
        if (type == mx::Type::VECTOR4 || type == mx::Type::COLOR4) return 4;
        if (type == mx::Type::MATRIX33) return 9;
        if (type == mx::Type::MATRIX44) return 16;
        return 0;
    }

    static std::string readOf(mx::TypeDesc type, uint32_t offset) {
        const auto at = [&](uint32_t k) { return "atheneaBlob(" + std::to_string(offset + k) + "u)"; };
        const auto list = [&](uint32_t n) {
            std::string s;
            for (uint32_t k = 0; k < n; ++k) {
                s += (k ? ", " : "") + at(k);
            }
            return s;
        };
        if (type == mx::Type::FLOAT) return at(0);
        if (type == mx::Type::INTEGER) return "int(" + at(0) + ")";
        if (type == mx::Type::BOOLEAN) return "(" + at(0) + " != 0.0)";
        if (type == mx::Type::VECTOR2) return "float2(" + list(2) + ")";
        if (type == mx::Type::VECTOR3 || type == mx::Type::COLOR3) return "float3(" + list(3) + ")";
        if (type == mx::Type::VECTOR4 || type == mx::Type::COLOR4) return "float4(" + list(4) + ")";
        if (type == mx::Type::MATRIX33) return "float3x3(" + list(9) + ")";
        if (type == mx::Type::MATRIX44) return "float4x4(" + list(16) + ")";
        return {};
    }

    static std::vector<float> valueOf(const mx::ValuePtr& value, mx::TypeDesc type) {
        std::vector<float> out(wordsOf(type), 0.0F);
        if (!value) {
            if (type == mx::Type::MATRIX33 || type == mx::Type::MATRIX44) {
                const size_t n = type == mx::Type::MATRIX33 ? 3 : 4;
                for (size_t k = 0; k < n; ++k) out[k * n + k] = 1.0F;
            }
            return out;
        }
        if (value->isA<float>()) out[0] = value->asA<float>();
        else if (value->isA<int>()) out[0] = static_cast<float>(value->asA<int>());
        else if (value->isA<bool>()) out[0] = value->asA<bool>() ? 1.0F : 0.0F;
        else if (value->isA<mx::Vector2>()) { auto v = value->asA<mx::Vector2>(); out = {v[0], v[1]}; }
        else if (value->isA<mx::Vector3>()) { auto v = value->asA<mx::Vector3>(); out = {v[0], v[1], v[2]}; }
        else if (value->isA<mx::Color3>()) { auto v = value->asA<mx::Color3>(); out = {v[0], v[1], v[2]}; }
        else if (value->isA<mx::Vector4>()) { auto v = value->asA<mx::Vector4>(); out = {v[0], v[1], v[2], v[3]}; }
        else if (value->isA<mx::Color4>()) { auto v = value->asA<mx::Color4>(); out = {v[0], v[1], v[2], v[3]}; }
        else if (value->isA<mx::Matrix33>()) {
            auto m = value->asA<mx::Matrix33>();
            for (size_t r = 0; r < 3; ++r) for (size_t c = 0; c < 3; ++c) out[r * 3 + c] = m[r][c];
        } else if (value->isA<mx::Matrix44>()) {
            auto m = value->asA<mx::Matrix44>();
            for (size_t r = 0; r < 4; ++r) for (size_t c = 0; c < 4; ++c) out[r * 4 + c] = m[r][c];
        }
        return out;
    }

    static int intInput(const mx::ShaderNode* node, const std::string& name, int fallback) {
        const mx::ShaderInput* input = node != nullptr ? node->getInput(name) : nullptr;
        if (input == nullptr || !input->getValue()) return fallback;
        const mx::ValuePtr v = input->getValue();
        return v->isA<int>() ? v->asA<int>() : fallback;
    }

    void emitMaterial(const mx::ShaderGraph& graph, mx::GenContext& context, mx::ShaderStage& stage) const {
        const bool lobes = variant_ != ClosureVariant::GenglslReference;
        emitLine(lobes ? "import athenea.material.material_runtime;" : "import athenea.material.material_inputs;", stage, false);
        emitLineBreak(stage);
        emitTypeDefinitions(context, stage);
        emitConstants(context, stage);

        // The vertex data MaterialX's nodes read, filled from the shading point.
        const mx::VariableBlock& vertexData = stage.getInputBlock(mx::HW::VERTEX_DATA);
        emitLine("struct VertexData", stage, false);
        emitScopeBegin(stage);
        emitLine("float4 SV_Position", stage);
        for (size_t i = 0; i < vertexData.size(); ++i) {
            emitLineBegin(stage);
            emitVariableDeclaration(vertexData[i], mx::EMPTY_STRING, context, stage, false);
            emitLineEnd(stage);
        }
        emitScopeEnd(stage, true);
        emitLine("static VertexData " + mx::HW::T_VERTEX_DATA_INSTANCE, stage);
        emitLineBreak(stage);

        // Uniforms as module statics, assigned from the blob.
        std::vector<const mx::ShaderPort*> uniforms;
        for (const auto& it : stage.getUniformBlocks()) {
            const mx::VariableBlock& block = *it.second;
            if (block.getName() == mx::HW::LIGHT_DATA) {
                continue;
            }
            for (size_t i = 0; i < block.size(); ++i) {
                const mx::ShaderPort* port = block[i];
                emitLineBegin(stage);
                // No initial value: materials differing only in values share one source.
                emitVariableDeclaration(port, "static", context, stage, false);
                emitLineEnd(stage);
                uniforms.push_back(port);
            }
        }
        emitLineBreak(stage);
        emitLibraryInclude("stdlib/genslang/lib/mx_math.slang", context, stage);
        emitLine("#define DIRECTIONAL_ALBEDO_METHOD 0", stage, false);
        emitLine("#define AIRY_FRESNEL_ITERATIONS 2", stage, false);
        emitLineBreak(stage);
        if (!lobes) {
            // genglsl's closures call into the environment and transmission
            // code: none and opacity, which read no uniforms.
            emitSpecularEnvironment(context, stage);
            emitTransmissionRender(context, stage);
        }
        _tokenSubstitutions[mx::ShaderGenerator::T_FILE_TRANSFORM_UV] = "mx_transform_uv.glsl";
        _tokenSubstitutions[mx::HW::T_TEX_SAMPLER_SIGNATURE] = "SamplerTexture2D tex_sampler";
        emitFunctionDefinitions(graph, context, stage);

        emitLine("public void " + kFunctionPlaceholder + "(MaterialInputs inputs, uint blob)", stage, false);
        emitFunctionBodyBegin(graph, context, stage);
        // The opacity alone leaves the lobe stack as it was: a kernel may be
        // in the middle of reading one when a shadow ray asks.
        emitLine(lobes && variant_ != ClosureVariant::Opacity ? "atheneaBeginMaterial(inputs, blob)"
                                                               : "atheneaBeginInputs(inputs, blob)",
                 stage);
        const std::string vd = mx::HW::T_VERTEX_DATA_INSTANCE + ".";
        for (size_t i = 0; i < vertexData.size(); ++i) {
            const mx::ShaderPort* port = vertexData[i];
            const std::string& name = port->getName();
            const mx::TypeDesc type = port->getType();
            std::string value;
            const auto builtin = [&](const std::string& token, const char* field) {
                if (name == token) value = std::string("inputs.") + field;
            };
            builtin(mx::HW::T_POSITION_WORLD, "positionWorld");
            builtin(mx::HW::T_NORMAL_WORLD, "normalWorld");
            builtin(mx::HW::T_TANGENT_WORLD, "tangentWorld");
            builtin(mx::HW::T_BITANGENT_WORLD, "bitangentWorld");
            builtin(mx::HW::T_POSITION_OBJECT, "positionObject");
            builtin(mx::HW::T_NORMAL_OBJECT, "normalObject");
            builtin(mx::HW::T_TANGENT_OBJECT, "tangentObject");
            builtin(mx::HW::T_BITANGENT_OBJECT, "bitangentObject");
            const auto primvar = [&](const std::string& primvarName, const std::string& fallback) {
                MaterialSlot& slot = addSlot(MaterialSlot::Kind::Primvar, name, 1);
                slot.name = primvarName;
                const std::string read = "atheneaPrimvar(uint(atheneaBlob(" + std::to_string(slot.offset) + "u)), " +
                                         fallback + ")";
                if (type == mx::Type::FLOAT) value = read + ".x";
                else if (type == mx::Type::INTEGER) value = "int(" + read + ".x)";
                else if (type == mx::Type::BOOLEAN) value = "(" + read + ".x != 0.0)";
                else if (type == mx::Type::VECTOR2) value = read + ".xy";
                else if (type == mx::Type::VECTOR3 || type == mx::Type::COLOR3) value = read + ".xyz";
                else value = read;
            };
            if (name.rfind(mx::HW::T_TEXCOORD + "_", 0) == 0) {
                const std::string index = name.substr(mx::HW::T_TEXCOORD.size() + 1);
                primvar(index == "0" ? "st" : "st" + index, "float4(0.0)");
            } else if (name.rfind(mx::HW::T_COLOR + "_", 0) == 0) {
                primvar("displayColor", "inputs.displayColor");
            } else if (name.rfind(mx::HW::T_IN_GEOMPROP + "_", 0) == 0) {
                primvar(name.substr(mx::HW::T_IN_GEOMPROP.size() + 1), "float4(0.0)");
            }
            if (!value.empty()) {
                emitLine(vd + port->getVariable() + " = " + value, stage);
            }
        }
        for (const mx::ShaderPort* port : uniforms) {
            const mx::TypeDesc type = port->getType();
            const std::string& variable = port->getVariable();
            // The world matrices, from the shading point's transform, as the
            // float4x4 mx_matrix_mul applies them.
            if (port->getName() == mx::HW::T_WORLD_MATRIX) {
                emitLine(variable + " = atheneaMatrixApplying(atheneaWorldFromObject())", stage);
                continue;
            }
            if (port->getName() == mx::HW::T_WORLD_INVERSE_MATRIX) {
                emitLine(variable + " = atheneaMatrixApplying(atheneaInverse(atheneaWorldFromObject()))", stage);
                continue;
            }
            if (port->getName() == mx::HW::T_WORLD_TRANSPOSE_MATRIX) {
                emitLine(variable + " = atheneaMatrixApplyingTranspose(atheneaWorldFromObject())", stage);
                continue;
            }
            if (port->getName() == mx::HW::T_WORLD_INVERSE_TRANSPOSE_MATRIX) {
                emitLine(variable + " = atheneaMatrixApplyingTranspose(atheneaInverse(atheneaWorldFromObject()))", stage);
                continue;
            }
            if (port->getName() == mx::HW::T_VIEW_POSITION) {
                emitLine(variable + " = inputs.viewPosition", stage);
                continue;
            }
            if (port->getName() == mx::HW::T_FRAME) {
                emitLine(variable + " = inputs.frame", stage);
                continue;
            }
            if (port->getName() == mx::HW::T_TIME) {
                emitLine(variable + " = inputs.time", stage);
                continue;
            }
            if (type == mx::Type::FILENAME) {
                MaterialSlot& slot = addSlot(MaterialSlot::Kind::Texture, variable, 2);
                slot.name = port->getValue() ? port->getValue()->getValueString() : std::string();
                const mx::ShaderNode* node = port->getNode();
                // The colour space as the document names it, resolved when the
                // file is read (colour::ColourNames, in TextureStore): MaterialX
                // reads a file in its colorspace, the document's (linear)
                // unless it says otherwise; UsdUVTexture says sourceColorSpace instead: auto, raw or sRGB.
                std::string space = port->getColorSpace();
                if (space.empty()) {
                    if (const auto found = fileColourSpaces.find(slot.name); found != fileColourSpaces.end()) {
                        space = found->second;
                    }
                }
                const mx::ShaderInput* source = node != nullptr ? node->getInput("sourceColorSpace") : nullptr;
                const std::string usd = source != nullptr && source->getValue() ? source->getValue()->getValueString()
                                                                                : std::string();
                if (source != nullptr) {
                    // "auto" is not a MaterialX colour space; the delegate
                    // writes it where USD said the file decides, which is what
                    // `sourceColorSpace = auto` means and what a file with no
                    // colour space at all gets from the scene index. An
                    // explicit sRGB wins over a document's default.
                    slot.space = usd == "sRGB" || space.empty() ? (usd.empty() ? std::string("auto") : usd) : space;
                } else {
                    slot.space = space.empty() ? std::string("lin_rec709") : space;
                }
                const auto wrap = [](int mode) {
                    // MaterialX address modes: 0 constant, 1 clamp, 2 periodic, 3 mirror.
                    return mode == 0 ? Wrap::Black : mode == 1 ? Wrap::Clamp : mode == 3 ? Wrap::Mirror : Wrap::Repeat;
                };
                slot.wrapS = wrap(intInput(node, "uaddressmode", 2));
                slot.wrapT = wrap(intInput(node, "vaddressmode", 2));
                slot.filter = intInput(node, "filtertype", 1) == 0 ? Filter::Nearest : Filter::Linear;
                emitLine(variable + " = atheneaTextureOf(atheneaBlob(" + std::to_string(slot.offset) + "u), atheneaBlob(" +
                             std::to_string(slot.offset + 1) + "u))",
                         stage);
                continue;
            }
            const uint32_t count = wordsOf(type);
            if (count == 0) {
                continue;   // strings, shader-typed uniforms: their declared defaults
            }
            MaterialSlot& slot = addSlot(MaterialSlot::Kind::Value, variable, count);
            slot.value = valueOf(port->getValue(), type);
            emitLine(variable + " = " + readOf(type, slot.offset), stage);
        }

        // The graph, as GenSlang orders it.
        const mx::ShaderGraphOutputSocket* outputSocket = graph.getOutputSocket();
        if (graph.hasClassification(mx::ShaderNode::Classification::SHADER | mx::ShaderNode::Classification::SURFACE)) {
            emitFunctionCalls(graph, context, stage, mx::ShaderNode::Classification::TEXTURE);
            for (mx::ShaderGraphOutputSocket* socket : graph.getOutputSockets()) {
                if (socket->getConnection()) {
                    const mx::ShaderNode* upstream = socket->getConnection()->getNode();
                    if (upstream->getParent() == &graph &&
                        (upstream->hasClassification(mx::ShaderNode::Classification::CLOSURE) ||
                         upstream->hasClassification(mx::ShaderNode::Classification::SHADER))) {
                        emitFunctionCall(*upstream, context, stage);
                    }
                }
            }
        } else {
            emitFunctionCalls(graph, context, stage);
        }
        if (const mx::ShaderOutput* connection = outputSocket->getConnection()) {
            const std::string variable = connection->getVariable();
            if (graph.hasClassification(mx::ShaderNode::Classification::SURFACE)) {
                if (variant_ == ClosureVariant::Opacity) {
                    emitLine("gAtheneaOpacity = saturate(1.0 - dot(" + variable + ".transparency, float3(1.0 / 3.0)))",
                             stage);
                } else if (lobes) {
                    emitLine("gAtheneaResult.emission = " + variable + ".color", stage);
                    emitLine("gAtheneaResult.opacity = saturate(1.0 - dot(" + variable +
                                 ".transparency, float3(1.0 / 3.0)))",
                             stage);
                } else {
                    emitLine("gAtheneaReferenceEmission = " + variable + ".color", stage);
                }
            } else {
                const mx::TypeDesc type = outputSocket->getType();
                std::string rgb = variable;
                if (type == mx::Type::FLOAT || type == mx::Type::INTEGER) rgb = "float3(float(" + variable + "))";
                else if (type.isFloat2()) rgb = "float3(" + variable + ", 0.0)";
                else if (type.isFloat4()) rgb = variable + ".rgb";
                emitLine((lobes ? "gAtheneaResult.emission = " : "gAtheneaReferenceEmission = ") + rgb, stage);
            }
        }
        emitFunctionBodyEnd(graph, context, stage);
    }
};

/// transformpoint, transformvector and transformnormal between "world",
/// "object" (or "model") and any other name, which is a coordinate system
/// bound to the prim (UsdShadeCoordSysAPI): its transform to world reaches
/// the mesh as three constant primvars, "atheneaCoordSys_NAME_0" to "_2", the
/// rows of a 3x4. An empty space is world.
class AtheneaTransformNode : public mx::ShaderNodeImpl {
public:
    enum class Kind { Point, Vector, Normal };
    explicit AtheneaTransformNode(Kind kind) : kind_(kind) {}

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override {
        DEFINE_SHADER_STAGE(stage, mx::Stage::PIXEL) {
            const auto& generator = static_cast<const AtheneaSlangShaderGenerator&>(context.getShaderGenerator());
            const mx::ShaderInput* in = node.getInput("in");
            const mx::ShaderOutput* output = node.getOutput();
            if (in == nullptr || output == nullptr) {
                throw mx::ExceptionShaderGenError("transform node '" + node.getName() + "' has no in or out");
            }
            const auto space = [&](const char* input) {
                const mx::ShaderInput* port = node.getInput(input);
                const std::string name = port != nullptr ? port->getValueString() : std::string();
                if (name.empty() || name == "world") {
                    return std::string("0u, 0u, 0u, 0u");
                }
                if (name == "object" || name == "model") {
                    return std::string("1u, 0u, 0u, 0u");
                }
                const std::string prefix = "atheneaCoordSys_" + name + "_";
                return "2u, " + generator.primvarSlotRead(prefix + "0") + ", " + generator.primvarSlotRead(prefix + "1") +
                       ", " + generator.primvarSlotRead(prefix + "2");
            };
            const std::string from = space("fromspace");
            const std::string to = space("tospace");
            generator.emitLineBegin(stage);
            generator.emitOutput(output, true, false, context, stage);
            generator.emitString(" = atheneaTransformBetween(" + generator.getUpstreamResult(in, context) + ", " +
                                     (kind_ == Kind::Point ? "1.0" : "0.0") + ", " +
                                     (kind_ == Kind::Normal ? "true" : "false") + ", " + from + ", " + to + ")",
                                 stage);
            generator.emitLineEnd(stage);
        }
    }

private:
    Kind kind_;
};

mx::ShaderNodeImplPtr createTransformPoint() { return std::make_shared<AtheneaTransformNode>(AtheneaTransformNode::Kind::Point); }
mx::ShaderNodeImplPtr createTransformVector() { return std::make_shared<AtheneaTransformNode>(AtheneaTransformNode::Kind::Vector); }
mx::ShaderNodeImplPtr createTransformNormal() { return std::make_shared<AtheneaTransformNode>(AtheneaTransformNode::Kind::Normal); }

std::string hexHash(const std::string& text) {
    // FNV-1a, 64 bits: a name for equal source, not a secret.
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : text) {
        h = (h ^ c) * 1099511628211ULL;
    }
    char out[17];
    std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(h));
    return out;
}

}   // namespace

struct MaterialCompiler::Impl {
    mx::DocumentPtr                          libraries;            ///< standard, the engine's images and closures
    mx::DocumentPtr                          referenceLibraries;   ///< standard and the engine's images
    mx::FileSearchPath                       sourcePaths;
    std::shared_ptr<AtheneaSlangShaderGenerator> generator;
    std::shared_ptr<AtheneaSlangShaderGenerator> reference;
    std::shared_ptr<AtheneaSlangShaderGenerator> opacity;
};

MaterialCompiler::~MaterialCompiler() = default;

Result<std::unique_ptr<MaterialCompiler>> MaterialCompiler::create(
    const std::vector<std::filesystem::path>& materialxRoots, const std::vector<std::filesystem::path>& shaderPaths) {
    try {
        mx::FileSearchPath libraryPaths;
        std::vector<std::filesystem::path> searchPaths;
        for (const auto& root : materialxRoots) {
            libraryPaths.append(mx::FilePath(root.string()));
            searchPaths.push_back(root);
            searchPaths.push_back(root / "libraries");
        }
        mx::DocumentPtr standard = mx::createDocument();
        mx::loadLibraries({"libraries"}, libraryPaths, standard);
        return create(std::shared_ptr<void>(standard), searchPaths, shaderPaths);
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::InternalError, "materials: MaterialX: {}", e.what());
    }
}

Result<std::unique_ptr<MaterialCompiler>> MaterialCompiler::create(
    const std::shared_ptr<void>& libraries, const std::vector<std::filesystem::path>& librarySearchPaths,
    const std::vector<std::filesystem::path>& shaderPaths) {
    auto compiler = std::unique_ptr<MaterialCompiler>(new MaterialCompiler());
    compiler->impl_ = std::make_unique<Impl>();
    Impl& impl = *compiler->impl_;
    try {
        const mx::DocumentPtr standard = std::static_pointer_cast<mx::Document>(libraries);
        if (!standard || standard->getNodeDefs().empty()) {
            return Error(ErrorCode::NotFound, "materials: no MaterialX libraries");
        }
        // The engine's node implementations, found beside its shaders.
        impl.libraries = mx::createDocument();
        impl.libraries->importLibrary(standard);
        impl.referenceLibraries = mx::createDocument();
        impl.referenceLibraries->importLibrary(standard);
        bool found = false;
        for (const auto& shaders : shaderPaths) {
            const std::filesystem::path images = shaders / "athenea/material/mx/athenea_genslang_images.mtlx";
            const std::filesystem::path closures = shaders / "athenea/material/mx/athenea_genslang_closures.mtlx";
            if (std::filesystem::exists(images) && std::filesystem::exists(closures)) {
                mx::DocumentPtr imageDoc = mx::createDocument();
                mx::readFromXmlFile(imageDoc, mx::FilePath(images.string()));
                mx::DocumentPtr closureDoc = mx::createDocument();
                mx::readFromXmlFile(closureDoc, mx::FilePath(closures.string()));
                impl.libraries->importLibrary(imageDoc);
                impl.libraries->importLibrary(closureDoc);
                impl.referenceLibraries->importLibrary(imageDoc);
                // UsdPreviewSurface with its opacity as coverage, in place of
                // MaterialX's graph (the file says why); the reference keeps
                // MaterialX's, as genglsl does.
                const std::filesystem::path preview = shaders / "athenea/material/mx/athenea_usd_preview_surface.mtlx";
                if (std::filesystem::exists(preview)) {
                    if (impl.libraries->getNodeGraph(kPreviewSurfaceGraph)) {
                        impl.libraries->removeNodeGraph(kPreviewSurfaceGraph);
                    }
                    mx::DocumentPtr previewDoc = mx::createDocument();
                    mx::readFromXmlFile(previewDoc, mx::FilePath(preview.string()));
                    impl.libraries->importLibrary(previewDoc);
                }
                found = true;
            }
            impl.sourcePaths.append(mx::FilePath(shaders.string()));
        }
        if (!found) {
            return Error(ErrorCode::NotFound,
                         "materials: athenea/material/mx/athenea_genslang_{images,closures}.mtlx are not in the shader paths");
        }
        for (const auto& path : librarySearchPaths) {
            impl.sourcePaths.append(mx::FilePath(path.string()));
        }
        impl.generator = std::make_shared<AtheneaSlangShaderGenerator>(mx::TypeSystem::create(), ClosureVariant::Lobes);
        impl.generator->registerTypeDefs(impl.libraries);
        impl.reference =
            std::make_shared<AtheneaSlangShaderGenerator>(mx::TypeSystem::create(), ClosureVariant::GenglslReference);
        impl.reference->registerTypeDefs(impl.referenceLibraries);
        impl.opacity = std::make_shared<AtheneaSlangShaderGenerator>(mx::TypeSystem::create(), ClosureVariant::Opacity);
        impl.opacity->registerTypeDefs(impl.libraries);
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::InternalError, "materials: MaterialX: {}", e.what());
    }
    return compiler;
}

bool MaterialCompiler::cutsOut(const std::shared_ptr<void>& document) {
    const auto doc = std::static_pointer_cast<mx::Document>(document);
    if (!doc) {
        return false;
    }
    for (const mx::ElementPtr& element : doc->traverseTree()) {
        const mx::NodePtr node = element->asA<mx::Node>();
        if (!node) {
            continue;
        }
        // The material's own surface node, which hdMtlx puts at the root:
        // the document carries the libraries too, whose implementation
        // graphs wire `opacity` up in every surface.
        if (node->getParent() != doc) {
            continue;
        }
        // A threshold cuts; an opacity under one is coverage, cut by lot.
        const auto driven = [](const mx::InputPtr& input) {
            return input && (!input->getNodeName().empty() || !input->getNodeGraphString().empty() ||
                             !input->getInterfaceName().empty());
        };
        const mx::InputPtr threshold = node->getInput("opacityThreshold");
        if (driven(threshold)) {
            return true;   // driven by a graph: assume it cuts
        }
        if (threshold) {
            const mx::ValuePtr value = threshold->getValue();
            if (value && value->isA<float>() && value->asA<float>() > 0.0F) {
                return true;
            }
        }
        // OpenPBR calls it `geometry_opacity`, and it is coverage there too:
        // a glass sparrow's feather cards read theirs off a mask and were
        // drawn whole, their mask a fractional opacity nothing cut by.
        for (const char* name : {"opacity", "geometry_opacity"}) {
            const mx::InputPtr opacity = node->getInput(name);
            if (driven(opacity)) {
                return true;
            }
            if (opacity) {
                const mx::ValuePtr value = opacity->getValue();
                if (value && value->isA<float>() && value->asA<float>() < 1.0F) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool MaterialCompiler::transparentOpacity(const std::shared_ptr<void>& document) {
    // UsdPreviewSurface 2.6, `opacityMode`: in `transparent` (0, the default)
    // a fractional opacity is not coverage. The diffuse contribution goes down
    // and what is left lets light straight through, WHILE THE SPECULAR AND THE
    // EMISSIVE STAY AT FULL WEIGHT -- a window at opacity 0 still reflects the
    // sky. `presence` (1) is the older reading, where the whole response
    // scales and the surface is there by lot. An `opacityThreshold` makes it
    // a cutout in either mode, and a mode driven by a graph is taken as
    // presence: a cut that should have happened is the worse error.
    const auto doc = std::static_pointer_cast<mx::Document>(document);
    if (!doc) {
        return false;
    }
    const auto driven = [](const mx::InputPtr& input) {
        return input && (!input->getNodeName().empty() || !input->getNodeGraphString().empty() ||
                         !input->getInterfaceName().empty());
    };
    for (const mx::ElementPtr& element : doc->traverseTree()) {
        const mx::NodePtr node = element->asA<mx::Node>();
        if (!node || node->getParent() != doc || node->getCategory() != "UsdPreviewSurface") {
            continue;
        }
        const mx::InputPtr threshold = node->getInput("opacityThreshold");
        if (driven(threshold)) {
            return false;
        }
        if (threshold && threshold->getValue() && threshold->getValue()->isA<float>() &&
            threshold->getValue()->asA<float>() > 0.0F) {
            return false;
        }
        const mx::InputPtr mode = node->getInput("opacityMode");
        if (driven(mode) || (mode && mode->getValue() && mode->getValue()->isA<int>() &&
                             mode->getValue()->asA<int>() == 1)) {
            return false;
        }
        const mx::InputPtr opacity = node->getInput("opacity");
        if (driven(opacity)) {
            return true;
        }
        return opacity && opacity->getValue() && opacity->getValue()->isA<float>() &&
               opacity->getValue()->asA<float>() < 1.0F;
    }
    return false;
}

bool MaterialCompiler::transmits(const std::shared_ptr<void>& document) {
    // A surface shader whose transmission is not zero: OpenPBR's
    // `transmission_weight`, standard_surface's and glTF's `transmission`, or
    // a UsdPreviewSurface whose opacity is transparency. Driven by a graph
    // counts as transmitting: a bit left open where the glass is clear is
    // the smaller error.
    const auto doc = std::static_pointer_cast<mx::Document>(document);
    if (!doc) {
        return false;
    }
    if (transparentOpacity(document)) {
        return true;
    }
    for (const mx::ElementPtr& element : doc->traverseTree()) {
        const mx::NodePtr node = element->asA<mx::Node>();
        if (!node) {
            continue;
        }
        for (const char* name : {"transmission_weight", "transmission"}) {
            const mx::InputPtr input = node->getInput(name);
            if (!input) {
                continue;
            }
            if (!input->getNodeName().empty() || !input->getNodeGraphString().empty() ||
                !input->getInterfaceName().empty()) {
                return true;
            }
            if (input->getValue() && input->getValue()->isA<float>() && input->getValue()->asA<float>() > 0.0F) {
                return true;
            }
        }
    }
    return false;
}

std::optional<VolumeCoefficients> MaterialCompiler::volumeCoefficients(const std::shared_ptr<void>& document) {
    const auto doc = std::static_pointer_cast<mx::Document>(document);
    if (!doc) {
        return std::nullopt;
    }
    const auto driven = [](const mx::InputPtr& input) {
        return input && (!input->getNodeName().empty() || !input->getNodeGraphString().empty() ||
                         !input->getInterfaceName().empty());
    };
    // The value an input holds, or the nodedef's default, unless a graph
    // drives it: a medium has no point to evaluate a graph at.
    static std::atomic<int> said{0};
    const auto vec3 = [&](const mx::NodePtr& node, const char* name, std::array<float, 3>& into) {
        const mx::InputPtr input = node->getInput(name);
        if (driven(input)) {
            if (said.fetch_add(1) < 4) {
                athenea::log::warn("material: a volume's '{}' on {} is driven by a graph; the medium takes the declared "
                          "default, since it has no surface to evaluate a graph at",
                          name, node->getName());
            }
            return;
        }
        mx::ValuePtr value = input ? input->getValue() : nullptr;
        if (!value) {
            const mx::NodeDefPtr def = node->getNodeDef();
            const mx::InputPtr declared = def ? def->getActiveInput(name) : nullptr;
            value = declared ? declared->getValue() : nullptr;
        }
        if (value && value->isA<mx::Vector3>()) {
            const mx::Vector3 v = value->asA<mx::Vector3>();
            into = {v[0], v[1], v[2]};
        } else if (value && value->isA<mx::Color3>()) {
            const mx::Color3 v = value->asA<mx::Color3>();
            into = {v[0], v[1], v[2]};
        }
    };
    const auto scalar = [&](const mx::NodePtr& node, const char* name, float& into) {
        const mx::InputPtr input = node->getInput(name);
        if (driven(input)) {
            return;
        }
        const mx::ValuePtr value = input ? input->getValue() : nullptr;
        if (value && value->isA<float>()) {
            into = value->asA<float>();
        }
    };
    for (const mx::ElementPtr& element : doc->traverseTree()) {
        const mx::NodePtr node = element->asA<mx::Node>();
        if (!node || node->getParent() != doc || node->getType() != mx::VOLUME_SHADER_TYPE_STRING) {
            continue;
        }
        VolumeCoefficients out;
        if (node->getCategory() != "volume") {
            athenea::log::warn("material: volume shader '{}' is not a MaterialX volume node; the medium takes its primvars",
                      node->getCategory());
            return std::nullopt;
        }
        if (const mx::InputPtr vdfInput = node->getInput("vdf"); vdfInput) {
            const mx::NodePtr vdf = vdfInput->getConnectedNode();
            if (vdf && vdf->getCategory() == "anisotropic_vdf") {
                vec3(vdf, "absorption", out.absorption);
                vec3(vdf, "scattering", out.scattering);
                scalar(vdf, "anisotropy", out.anisotropy);
            } else if (vdf && vdf->getCategory() == "absorption_vdf") {
                vec3(vdf, "absorption", out.absorption);
            } else if (vdf) {
                athenea::log::warn("material: vdf '{}' is not anisotropic_vdf or absorption_vdf; the medium takes what it "
                          "can read of it, which is nothing",
                          vdf->getCategory());
            }
        }
        if (const mx::InputPtr edfInput = node->getInput("edf"); edfInput) {
            const mx::NodePtr edf = edfInput->getConnectedNode();
            if (edf && edf->getCategory() == "uniform_edf") {
                vec3(edf, "color", out.emission);
            } else if (edf) {
                athenea::log::warn("material: edf '{}' is not uniform_edf; the medium emits nothing", edf->getCategory());
            }
        }
        return out;
    }
    return std::nullopt;
}

std::shared_ptr<void> MaterialCompiler::libraries() const {
    return impl_->libraries;
}

Result<CompiledMaterial> MaterialCompiler::compileXml(const std::string& xml, const std::string& element,
                                                     ClosureVariant variant) {
    try {
        mx::DocumentPtr doc = mx::createDocument();
        mx::readFromXmlString(doc, xml);
        doc->importLibrary(variant != ClosureVariant::GenglslReference ? impl_->libraries : impl_->referenceLibraries);
        return compileDocument(doc, element, variant);
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::InvalidArgument, "materials: MaterialX: {}", e.what());
    }
}

Result<CompiledMaterial> MaterialCompiler::compileDocument(const std::shared_ptr<void>& document,
                                                           const std::string& element, ClosureVariant variant) {
    const Impl& impl = *impl_;
    const std::shared_ptr<AtheneaSlangShaderGenerator>& generator = variant == ClosureVariant::Lobes     ? impl.generator
                                                                : variant == ClosureVariant::Opacity ? impl.opacity
                                                                                                     : impl.reference;
    const bool ours = variant != ClosureVariant::GenglslReference;
    try {
        const mx::DocumentPtr given = std::static_pointer_cast<mx::Document>(document);
        // A document built elsewhere (hdMtlx) carries the libraries its USD
        // loaded but not the engine's implementations. Where the document
        // and the compiler's libraries define the same element, the
        // compiler's wins: the engine's UsdPreviewSurface graph over
        // MaterialX's, and -- for a compiler given libraries of its own
        // ($ATHENEA_MATERIALX_ROOT) -- its definitions and implementations
        // over the host's older ones, whose source files would otherwise be
        // included beside its own.
        const mx::DocumentPtr& libraries = ours ? impl.libraries : impl.referenceLibraries;
        mx::DocumentPtr doc = mx::createDocument();
        doc->copyContentFrom(given);
        const std::vector<mx::ElementPtr> children = doc->getChildren();   // a copy: removing changes the list
        for (const mx::ElementPtr& child : children) {
            if (!ours && child->getName() == kPreviewSurfaceGraph) {
                continue;   // the reference keeps MaterialX's graph
            }
            // Library content only -- definitions, implementations and a
            // definition's graph -- and only of the same kind: a material
            // may be named like a library's typedef ("material").
            const mx::ElementPtr theirs = libraries->getChild(child->getName());
            const bool definition = child->isA<mx::NodeDef>() || child->isA<mx::Implementation>() ||
                                    child->isA<mx::TypeDef>() ||
                                    (child->isA<mx::NodeGraph>() && child->hasAttribute(mx::InterfaceElement::NODE_DEF_ATTRIBUTE));
            if (definition && theirs && theirs->getCategory() == child->getCategory()) {
                doc->removeChild(child->getName());
            }
        }
        doc->importLibrary(libraries);
        mx::TypedElementPtr renderable;
        if (!element.empty()) {
            renderable = doc->getDescendant(element) ? doc->getDescendant(element)->asA<mx::TypedElement>() : nullptr;
        } else {
            const auto found = mx::findRenderableElements(doc);
            if (!found.empty()) {
                renderable = found.front();
            }
        }
        if (!renderable) {
            return Error::make(ErrorCode::NotFound, "materials: no renderable element '{}'", element);
        }
        // The generator names the surface's variables after the renderable
        // element, which hdMtlx names after the material prim: every material
        // on a stage is a different name, so materials identical but for
        // their values hashed apart and were compiled one module each (the
        // standard shader ball's seventeen were thirteen modules of five
        // sources). Nothing refers to the renderable by name: one name for
        // all of them.
        static const std::string kRenderableName = "athenea_surface";
        if (renderable->getParent() == doc && renderable->getName() != kRenderableName &&
            !doc->getChild(kRenderableName)) {
            renderable->setName(kRenderableName);
        }
        // What each file's colour space is, read off the document while it is
        // still a document (see `fileColourSpaces`).
        generator->fileColourSpaces.clear();
        const std::function<void(const mx::ElementPtr&)> readSpaces = [&](const mx::ElementPtr& parent) {
            for (const mx::ElementPtr& child : parent->getChildren()) {
                if (const mx::NodePtr node = child->asA<mx::Node>()) {
                    for (const mx::InputPtr& input : node->getInputs()) {
                        if (input->getType() != "filename" || !input->getValue()) {
                            continue;
                        }
                        const std::string space = input->getActiveColorSpace();
                        if (!space.empty()) {
                            generator->fileColourSpaces[input->getValue()->getValueString()] = space;
                        }
                    }
                } else if (child->isA<mx::NodeGraph>()) {
                    readSpaces(child);
                }
            }
        };
        readSpaces(doc);

        mx::GenContext context(generator);
        context.registerSourceCodeSearchPath(impl.sourcePaths);
        mx::GenOptions& options = context.getOptions();
        options.shaderInterfaceType = mx::SHADER_INTERFACE_COMPLETE;
        options.hwSpecularEnvironmentMethod = mx::SPECULAR_ENVIRONMENT_NONE;
        options.hwTransmissionRenderMethod = mx::TRANSMISSION_OPACITY;
        options.hwMaxActiveLightSources = 0;
        options.fileTextureVerticalFlip = false;
        options.hwDirectionalAlbedoMethod = mx::DIRECTIONAL_ALBEDO_ANALYTIC;
        mx::ShaderPtr shader = generator->generate("athenea_material", renderable, context);
        CompiledMaterial compiled;
        const std::string source = shader->getSourceCode(mx::Stage::PIXEL);
        const std::string hash = hexHash(source);
        const char* modulePrefix = variant == ClosureVariant::Lobes     ? "athenea_mat_"
                                   : variant == ClosureVariant::Opacity ? "athenea_opa_"
                                                                        : "athenea_ref_";
        const char* functionPrefix = variant == ClosureVariant::Lobes     ? "athenea_material_"
                                     : variant == ClosureVariant::Opacity ? "athenea_opacity_"
                                                                          : "athenea_reference_";
        compiled.module = modulePrefix + hash;
        compiled.function = functionPrefix + hash;
        compiled.variant = variant;
        compiled.source = source;
        for (size_t at = compiled.source.find(kFunctionPlaceholder); at != std::string::npos;
             at = compiled.source.find(kFunctionPlaceholder, at)) {
            compiled.source.replace(at, kFunctionPlaceholder.size(), compiled.function);
        }
        compiled.slots = generator->slots;
        compiled.words = generator->words;
        return compiled;
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::ShaderFailure, "materials: MaterialX could not generate '{}': {}", element,
                           e.what());
    }
}

std::vector<float> MaterialCompiler::parameters(const CompiledMaterial& material, TextureStore& textures,
                                                const std::function<uint32_t(const std::string&)>& slotOf) {
    std::vector<float> words(material.words, 0.0F);
    for (const MaterialSlot& slot : material.slots) {
        switch (slot.kind) {
        case MaterialSlot::Kind::Value:
            for (size_t k = 0; k < slot.value.size() && k < slot.words; ++k) {
                words[slot.offset + k] = slot.value[k];
            }
            break;
        case MaterialSlot::Kind::Texture:
            words[slot.offset] = slot.name.empty() ? 4294967295.0F
                                                   : static_cast<float>(textures.request(slot.name, slot.space));
            words[slot.offset + 1] = static_cast<float>(textures.sampler(slot.wrapS, slot.wrapT, slot.filter));
            break;
        case MaterialSlot::Kind::Primvar:
            words[slot.offset] = static_cast<float>(slotOf(slot.name));
            break;
        }
    }
    return words;
}

}   // namespace athenea::material
