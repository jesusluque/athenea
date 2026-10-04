// Copyright (c) 2026 jesus luque.
//
// A WGSL file through Tint, Chrome's WGSL compiler, on the CPU.
//
// Dawn's null backend: an adapter with no GPU behind it, which still runs
// every check Dawn makes before it would hand a shader to Metal or Vulkan --
// Tint's parser, its resolver and its uniformity analysis (a barrier under a
// branch on group memory is an error there and not in Naga), and the device
// limits a pipeline is held to when its layout is made: storage buffers per
// stage, workgroup memory, invocations per workgroup. The limits are the
// web's defaults unless raised, so what passes here is what a browser on the
// default limits would take.
//
// Nothing is dispatched and no GPU is opened.
//
// Build (scripts/wgsl-report.py --tint does it when it is missing):
//   c++ -std=c++20 -O1 scripts/wgsl-tint.cpp -I ~/tools/dawn-138.0.7204.168/include \
//       -L ~/tools/dawn-138.0.7204.168/lib -ldawn \
//       -Wl,-rpath,$HOME/tools/dawn-138.0.7204.168/lib -o build/wgsl-tint
// Usage: wgsl-tint [--storage N] [--workgroup BYTES] file.wgsl...
//   prints, a line per file, "module:" (Tint on the WGSL) and "pipeline:" (the
//   auto layout against the limits), each "ok" or Dawn's message; exits
//   non-zero if any failed.
#include <dawn/webgpu.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

WGPUStringView view(const std::string& s) { return WGPUStringView{s.data(), s.size()}; }
std::string text(WGPUStringView v) { return v.data ? std::string(v.data, v.length == WGPU_STRLEN ? std::strlen(v.data) : v.length) : std::string(); }

struct Wait {
    bool done = false;
};

void pump(WGPUInstance instance, const bool& done) {
    for (int i = 0; i < 100000 && !done; ++i) {
        wgpuInstanceProcessEvents(instance);
    }
}

}   // namespace

int main(int argc, char** argv) {
    uint32_t storage = 0;
    uint32_t workgroup = 0;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--storage") && i + 1 < argc) storage = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--workgroup") && i + 1 < argc) workgroup = uint32_t(std::atoi(argv[++i]));
        else files.emplace_back(argv[i]);
    }
    if (files.empty()) {
        std::fprintf(stderr, "usage: wgsl-tint [--storage N] [--workgroup BYTES] file.wgsl...\n");
        return 2;
    }

    WGPUInstanceDescriptor instanceDesc = WGPU_INSTANCE_DESCRIPTOR_INIT;
    WGPUInstance instance = wgpuCreateInstance(&instanceDesc);
    if (!instance) {
        std::fprintf(stderr, "no WebGPU instance\n");
        return 2;
    }

    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    options.backendType = WGPUBackendType_Null;
    options.featureLevel = WGPUFeatureLevel_Core;
    struct AdapterState {
        bool done = false;
        WGPUAdapter adapter = nullptr;
        std::string message;
    } adapterState;
    WGPURequestAdapterCallbackInfo adapterCb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    adapterCb.mode = WGPUCallbackMode_AllowProcessEvents;
    adapterCb.callback = [](WGPURequestAdapterStatus, WGPUAdapter adapter, WGPUStringView message, void* u, void*) {
        auto* s = static_cast<AdapterState*>(u);
        s->adapter = adapter;
        s->message = text(message);
        s->done = true;
    };
    adapterCb.userdata1 = &adapterState;
    wgpuInstanceRequestAdapter(instance, &options, adapterCb);
    pump(instance, adapterState.done);
    if (!adapterState.adapter) {
        std::fprintf(stderr, "no null adapter: %s\n", adapterState.message.c_str());
        return 2;
    }

    // The web's defaults, raised only where asked and only as far as the
    // null adapter goes (its own limits are the defaults for workgroup
    // memory: 16384).
    WGPULimits supported = WGPU_LIMITS_INIT;
    wgpuAdapterGetLimits(adapterState.adapter, &supported);
    WGPULimits limits = WGPU_LIMITS_INIT;
    if (storage != 0) limits.maxStorageBuffersPerShaderStage = std::min(storage, supported.maxStorageBuffersPerShaderStage);
    if (workgroup != 0) limits.maxComputeWorkgroupStorageSize = std::min(workgroup, supported.maxComputeWorkgroupStorageSize);
    if ((storage != 0 && storage > supported.maxStorageBuffersPerShaderStage) ||
        (workgroup != 0 && workgroup > supported.maxComputeWorkgroupStorageSize)) {
        std::fprintf(stderr, "wgsl-tint: the null adapter goes to %u storage buffers and %u workgroup bytes\n",
                     supported.maxStorageBuffersPerShaderStage, supported.maxComputeWorkgroupStorageSize);
    }
    WGPUDeviceDescriptor deviceDesc = WGPU_DEVICE_DESCRIPTOR_INIT;
    deviceDesc.requiredLimits = &limits;
    struct DeviceState {
        bool done = false;
        WGPUDevice device = nullptr;
        std::string message;
    } deviceState;
    WGPURequestDeviceCallbackInfo deviceCb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    deviceCb.mode = WGPUCallbackMode_AllowProcessEvents;
    deviceCb.callback = [](WGPURequestDeviceStatus, WGPUDevice device, WGPUStringView message, void* u, void*) {
        auto* s = static_cast<DeviceState*>(u);
        s->device = device;
        s->message = text(message);
        s->done = true;
    };
    deviceCb.userdata1 = &deviceState;
    wgpuAdapterRequestDevice(adapterState.adapter, &deviceDesc, deviceCb);
    pump(instance, deviceState.done);
    if (!deviceState.device) {
        std::fprintf(stderr, "no null device: %s\n", deviceState.message.c_str());
        return 2;
    }
    WGPUDevice device = deviceState.device;

    int failed = 0;
    for (const std::string& file : files) {
        std::ifstream in(file);
        std::stringstream buffer;
        buffer << in.rdbuf();
        const std::string source = buffer.str();
        if (source.empty()) {
            std::printf("%s: cannot read\n", file.c_str());
            ++failed;
            continue;
        }

        // Two scopes: the module (Tint: parse, resolve, uniformity) and the
        // pipeline (its auto layout against the device's limits), so a
        // kernel over a limit still says whether its WGSL is right.
        struct ScopeState {
            bool done = false;
            WGPUErrorType type = WGPUErrorType_NoError;
            std::string message;
        };
        const auto popScope = [&](ScopeState& scope) {
            WGPUPopErrorScopeCallbackInfo popCb = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
            popCb.mode = WGPUCallbackMode_AllowProcessEvents;
            popCb.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView message, void* u, void*) {
                auto* s = static_cast<ScopeState*>(u);
                s->type = type;
                s->message = text(message);
                s->done = true;
            };
            popCb.userdata1 = &scope;
            wgpuDevicePopErrorScope(device, popCb);
            pump(instance, scope.done);
        };
        const auto said = [](const ScopeState& scope) {
            if (scope.type == WGPUErrorType_NoError) return std::string("ok");
            std::string first = scope.message;
            for (char& c : first) {
                if (c == '\n' || c == '\t') c = ' ';
            }
            return first.substr(0, 600);
        };

        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
        WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
        wgsl.code = view(source);
        WGPUShaderModuleDescriptor moduleDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        moduleDesc.nextInChain = &wgsl.chain;
        WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &moduleDesc);
        ScopeState moduleScope;
        popScope(moduleScope);

        ScopeState pipelineScope;
        WGPUComputePipeline pipeline = nullptr;
        if (moduleScope.type == WGPUErrorType_NoError) {
            wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
            WGPUComputePipelineDescriptor pipelineDesc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
            pipelineDesc.compute.module = module;   // the sole entry point; layout: auto
            pipeline = wgpuDeviceCreateComputePipeline(device, &pipelineDesc);
            popScope(pipelineScope);
        } else {
            pipelineScope.type = WGPUErrorType_Validation;
            pipelineScope.message = "no module";
        }
        std::printf("%s\tmodule: %s\tpipeline: %s\n", file.c_str(), said(moduleScope).c_str(),
                    said(pipelineScope).c_str());
        if (moduleScope.type != WGPUErrorType_NoError || pipelineScope.type != WGPUErrorType_NoError) {
            ++failed;
        }
        if (pipeline) wgpuComputePipelineRelease(pipeline);
        if (module) wgpuShaderModuleRelease(module);
    }
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapterState.adapter);
    wgpuInstanceRelease(instance);
    return failed == 0 ? 0 : 1;
}
