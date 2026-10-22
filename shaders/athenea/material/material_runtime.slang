// Copyright (c) 2026 jesus luque.
//
// What code generated from a MaterialX graph (MaterialCompiler) runs against:
// MaterialX's closure and shader types as this engine has them, its texture
// wrappers over the texture table, the shading point's inputs, and the build
// stack its BSDF nodes push lobes onto.
//
// A BSDF value is not a response but a weight per built lobe: a BSDF node
// pushes its lobe once and returns a value weighting it; mix, layer, add and
// multiply combine values the way genglsl combines responses (layer scales
// the base by the top's throughput at the view direction). A value used by
// two branches is therefore two weightings of the same lobes, not two copies.
module material_runtime;
__exported import athenea.material.lobes;
__exported import athenea.material.material_inputs;

// ---- MaterialX's types and constants -----------------------------------------

// ClosureData, makeClosureData, the CLOSURE_TYPE_ constants and M_PI come
// from MaterialX's own mx_closure_type.glsl and mx_math.slang, which generated
// code includes once.

// How many lobes a material may build before the heaviest kMaxLobes are
// kept. It also sizes every BSDF value a MaterialX graph passes between its
// nodes (BSDF::weights), so it is most of what a shading kernel keeps in
// thread memory. A lobe pushed past it is dropped (atheneaPushLobe), not written
// past the end. ATHENEA_BUILD_LOBES is set lower where a GPU gives a thread less
// stack (athenea::gpu::Device).
//
// A link-time constant: the module that dispatches to a frame's materials
// (MaterialPrograms) exports as many as they can push, so a room of
// UsdPreviewSurfaces keeps five where an OpenPBR coat over a sheen needs
// sixteen -- thread memory is what a material evaluated at every pixel
// spends its time on. A kernel that imports no materials keeps the default.
#ifdef ATHENEA_BUILD_LOBES
public extern static const uint kBuildLobes = ATHENEA_BUILD_LOBES;
#else
public extern static const uint kBuildLobes = 16;
#endif

public struct BSDF {
    public float3 weights[kBuildLobes];   // per built lobe
    public float3 throughput;
};

public BSDF athenea_bsdf_zero() {
    BSDF b;
    for (uint k = 0; k < kBuildLobes; ++k) {
        b.weights[k] = float3(0.0);
    }
    b.throughput = float3(1.0);
    return b;
}

// ---- the build stack --------------------------------------------------------------

/// WHETHER THE CLOSURES MARK THEIR LOBES for a conversion to read back
/// (lobeTint, lobeLayersOver; material_sample.slang): a link-time constant,
/// false unless the kernel exports it true -- the path tracer's bake does --
/// so every other kernel compiles exactly what it did before the marks
/// existed. The shadowed shading kernel stands at what Metal will build.
public extern static const bool kAtheneaMarkLobes = false;

public static Lobe  gAtheneaLobes[kBuildLobes];
public static uint  gAtheneaLobeCount;
/// The surface constructor's `thin_walled` (OpenPBR's `geometry_thin_walled`):
/// set before its BSDF builds, so every dielectric it pushes is a thin sheet.
public static bool  gAtheneaThinWalled;
/// Where a material's opacity alone is left (ClosureVariant::Opacity), apart
/// from the lobe stack a kernel may be reading when a shadow ray asks.
public static float gAtheneaOpacity;

/// Pushes `lobe` (weight unset) and returns its index, or kBuildLobes when full.
public uint atheneaPushLobe(Lobe lobe, float3 N, float3 X) {
    if (gAtheneaLobeCount >= kBuildLobes) {
        return kBuildLobes;
    }
    lobe.normal = normalize(N);
    lobe.tangent = X;
    lobe.weight = float3(0.0);
    if (lobe.kind == kLobeDielectric) {
        // A thin wall has no inside to be seen from.
        if (gAtheneaThinWalled) {
            lobe.flags |= kFlagThinWalled;
        } else if (gAtheneaInputs.inside) {
            lobe.flags |= kFlagInside;
        }
    }
    gAtheneaLobes[gAtheneaLobeCount] = lobe;
    return gAtheneaLobeCount++;
}

public Lobe atheneaLobe(uint kind) {
    Lobe lobe;
    lobe.kind = kind;
    lobe.flags = 0;
    lobe.scatter = kScatterReflect;
    lobe.pad0 = 0;
    lobe.weight = float3(0.0);
    lobe.roughness = 0.0;
    lobe.colour0 = float3(0.0);
    lobe.ior = 1.5;
    lobe.colour1 = float3(0.0);
    lobe.exponent = 5.0;
    lobe.colour2 = float3(1.0);
    lobe.pad1 = kAtheneaMarkLobes ? 1.0 : 0.0;   // the tint (lobeTint): none yet
    lobe.alpha = float2(0.0);
    lobe.pad2 = float2(kAtheneaMarkLobes ? 1.0 : 0.0);
    lobe.normal = float3(0.0, 0.0, 1.0);
    lobe.pad3 = 0.0;
    lobe.tangent = float3(1.0, 0.0, 0.0);
    lobe.pad4 = 0.0;
    return lobe;
}

/// The value weighting pushed lobe `index` by `weight`.
public BSDF atheneaBsdfOf(uint index, float3 weight, float3 throughput) {
    BSDF b = athenea_bsdf_zero();
    if (index < kBuildLobes) {
        b.weights[index] = weight;
    }
    b.throughput = throughput;
    return b;
}

/// The final lobes, into gAtheneaResult where they are read: every built lobe
/// weighted by `bsdf`, those that carry nothing dropped, the heaviest
/// kMaxLobes kept. Its emission stays what it was. Written in place: a stack
/// returned and assigned was one more copy of it in thread memory.
public void atheneaFinishStack(BSDF bsdf, float opacity) {
    // THE WEIGHTS ARE COPIED OUT FIRST, before a lobe is written to the stack.
    // On CUDA the BSDF arrives as a pointer to the caller's temporary, and
    // nvcc laid this stack over that temporary's memory: each lobe kept
    // overwrote the weights of the lobes still to be read, so lobes of zero
    // weight survived with what a written lobe left there (1, 0, 1) -- an
    // OpenPBR ground kept five lobes where it has two, lit pink and ten times
    // too bright, and every bake came out black. Metal and Vulkan read the
    // weights as they are; a copy taken before the stack is touched is the
    // same on all three.
    float3 weights[kBuildLobes];
    for (uint k = 0; k < kBuildLobes; ++k) {
        weights[k] = bsdf.weights[k];
    }
    gAtheneaResult.count = 0;
    gAtheneaResult.opacity = opacity;
    for (uint k = 0; k < kMaxLobes; ++k) {
        gAtheneaResult.lobes[k].kind = kLobeNone;
    }
    for (uint k = 0; k < gAtheneaLobeCount; ++k) {
        const float3 w = max(weights[k], float3(0.0));
        const float heft = dot(w, float3(0.2126, 0.7152, 0.0722));
        if (heft <= 1e-6) {
            continue;
        }
        Lobe lobe = gAtheneaLobes[k];
        lobe.weight = w;
        if (gAtheneaResult.count < kMaxLobes) {
            gAtheneaResult.lobes[gAtheneaResult.count++] = lobe;
            continue;
        }
        // Full: replace the lightest.
        uint lightest = 0;
        float least = 3.4e38;
        for (uint j = 0; j < kMaxLobes; ++j) {
            const float h = dot(gAtheneaResult.lobes[j].weight, float3(0.2126, 0.7152, 0.0722));
            if (h < least) {
                least = h;
                lightest = j;
            }
        }
        if (heft > least) {
            gAtheneaResult.lobes[lightest] = lobe;
        }
    }
}

// Where the material function leaves its result.
public static LobeStack gAtheneaResult;

public void atheneaBeginMaterial(MaterialInputs inputs, uint blob) {
    atheneaBeginInputs(inputs, blob);
    gAtheneaLobeCount = 0;
    gAtheneaThinWalled = false;
    gAtheneaResult = emptyStack();
}

/// What a mesh with no material shows: its displayColor, diffusely, with its
/// displayOpacity -- as Storm draws it.
public void atheneaFallbackMaterial(MaterialInputs inputs) {
    atheneaBeginMaterial(inputs, 0);
    const float3 colour = atheneaPrimvar(kSlotDisplayColor, inputs.displayColor).rgb;
    const float opacity = saturate(atheneaPrimvar(kSlotDisplayOpacity, float4(inputs.displayColor.w)).x);
    Lobe lobe = atheneaLobe(kLobeOrenNayar);
    lobe.colour0 = colour;
    const uint index = atheneaPushLobe(lobe, inputs.normalWorld, inputs.tangentWorld);
    atheneaFinishStack(atheneaBsdfOf(index, float3(1.0), float3(0.0)), opacity);
}
