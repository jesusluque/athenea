// Copyright (c) 2026 jesus luque.
//
// A WALL OF LIGHT AS A TEXTURE EACH GAUSSIAN LOOKS UP.
//
// An LED wall is a light whose picture is known every frame, and what it does
// to a gaussian is linear in that picture: a row of numbers, one a texel of
// the wall, that says how much of each reaches the eye off the gaussian. That
// row is not kept. What a surface takes from a wall is a patch of it -- wide
// for a matt thing, narrow for a shiny one, cut where something stands in the
// way -- and a patch of a picture is a read of its mip chain: where (u, v),
// how wide (the level), and how much (a gain a channel). So a gaussian keeps
// `lobes` such reads, six numbers each, and its colour under any picture the
// wall shows is
//
//     sum over its lobes of gain * wall(u, v, level)
//
// one or two filtered texture reads a gaussian, whatever the wall's
// resolution (after Azzarelli et al., who learn a gaussian's texture
// coordinate on a known display). The gain holds what the gaussian is made
// of as well as how much of the wall it sees: nothing else of the cloud is
// read, so any cloud can stand before a wall. The reads are learnt: the picture the
// frame should be, under a wall picture the fit knows, pulls on u, v, the
// level and the gain through the read's own derivatives.
//
// The wall's pictures live in one buffer, pyramid after pyramid: level zero
// row by row from the bottom, then each level half the last (`wallOffset`).
module neural_wall;
import athenea.common.color;
import athenea.common.packing;
import athenea.splat.frame;

/// A lobe: u, v (0 .. 1 across the wall), the level, and the log of its gain a channel.
public static const uint kWallLobe = 6;

struct NeuralWallParams {
    uint  count;        // gaussians; or the numbers a clear or Adam walks
    uint  base;         // the cloud's first slot in the frame
    uint  lobes;
    uint  image;        // which of the buffer's pictures
    uint  width;        // the wall's texels
    uint  height;
    uint  levels;
    uint  level;        // neuralWallPyramid: the level written
    float3 centre;      // the wall, in the cloud's space: its centre,
    float  halfU;
    float3 axisU;       // its two unit axes and half its size along each
    float  halfV;
    float3 axisV;
    float  startLevel;
    float  startGain;
    uint   pattern;     // neuralWallPattern: what picture
    uint   argument;
    uint   frameWidth;
    uint   frameHeight;
    float  rateUv; float rateLevel; float rateGain;
    float  beta1; float beta2; float epsilon;
    float  correction1; float correction2;
    float  tintR; float tintG; float tintB;   // neuralWallAccumulate: what the frame is multiplied by
    uint   texel;       // neuralWallAccumulate: or, with `byTexel`, the picture's texel that says so
    uint   byTexel;
    /// THE PICTURES FINER THAN THE FIT. The reads were learnt on a wall of few texels and the wall
    /// shows pictures of this many levels more: a read's level counts from the fit's own, and the
    /// fit's texels are this level of a picture's pyramid.
    uint   finer;
    /// A GRID IN PLACE OF READS. What a matt thing takes from a wall is most of the wall, more from
    /// here and less from there, and no read or two of a mip chain is that shape. With `cells` not
    /// zero a gaussian keeps instead a weight a cell of a coarse level of the wall's pictures
    /// (`gridLevel` above the fit's own, `cells` texels) and one gain a channel: `cells + 3`
    /// numbers, all as their logs, and its colour is gain * sum of weight * that level's texel.
    uint   cells;
    uint   gridLevel;
};

StructuredBuffer<float4>           positions;
StructuredBuffer<uint>             shape;
StructuredBuffer<uint>             visible;
StructuredBuffer<float>            wallLatent;   // kWallLobe * lobes a gaussian
StructuredBuffer<float4>           wallImage;
RWStructuredBuffer<ProjRecord>     proj;
RWStructuredBuffer<float>          made;         // the reads being started, or moved
RWStructuredBuffer<float4>         imageOut;     // a picture being written, and its pyramid
StructuredBuffer<uint>             colourGrad;   // four words a slot, as the blend summed them
RWStructuredBuffer<float>          gradient;
StructuredBuffer<float>            gradientRead;
RWStructuredBuffer<float>          moment1;
RWStructuredBuffer<float>          moment2;
StructuredBuffer<float4>           student;
StructuredBuffer<float4>           teacher;
RWStructuredBuffer<float4>         pixelGrad;
RWStructuredBuffer<uint>           loss;
RWStructuredBuffer<uint>           cleared;
StructuredBuffer<float4>           frameIn;
RWStructuredBuffer<float4>         frameOut;
ConstantBuffer<NeuralWallParams>   params;

uint2 wallSize(uint level) {
    return uint2(max(params.width >> level, 1u), max(params.height >> level, 1u));
}

/// Texels before `level` in a pyramid.
uint wallOffset(uint level) {
    uint offset = 0;
    for (uint l = 0; l < level; ++l) {
        const uint2 size = wallSize(l);
        offset += size.x * size.y;
    }
    return offset;
}

uint wallTexels() { return wallOffset(params.levels); }

/// A read of one level, and how it changes with u and v.
struct WallRead {
    float3 value;
    float3 du;
    float3 dv;
    float3 dLevel;
};

WallRead wallLevelRead(uint image, uint level, float u, float v) {
    const uint2 size = wallSize(level);
    const uint first = image * wallTexels() + wallOffset(level);
    const float x = u * float(size.x) - 0.5;
    const float y = v * float(size.y) - 0.5;
    const float x0 = floor(x);
    const float y0 = floor(y);
    const float tx = x - x0;
    const float ty = y - y0;
    const int xa = clamp(int(x0), 0, int(size.x) - 1);
    const int xb = clamp(int(x0) + 1, 0, int(size.x) - 1);
    const int ya = clamp(int(y0), 0, int(size.y) - 1);
    const int yb = clamp(int(y0) + 1, 0, int(size.y) - 1);
    const float3 t00 = wallImage[first + uint(ya) * size.x + uint(xa)].rgb;
    const float3 t10 = wallImage[first + uint(ya) * size.x + uint(xb)].rgb;
    const float3 t01 = wallImage[first + uint(yb) * size.x + uint(xa)].rgb;
    const float3 t11 = wallImage[first + uint(yb) * size.x + uint(xb)].rgb;
    WallRead r;
    r.value = lerp(lerp(t00, t10, tx), lerp(t01, t11, tx), ty);
    r.du = lerp(t10 - t00, t11 - t01, ty) * float(size.x);
    r.dv = lerp(t01 - t00, t11 - t10, tx) * float(size.y);
    r.dLevel = float3(0.0);
    return r;
}

/// The wall's picture at (u, v), as wide as `level` says: between two levels
/// of the pyramid, and its derivatives in all three.
WallRead wallRead(uint image, float u, float v, float level) {
    const float top = float(params.levels - 1);
    const float l = clamp(level + float(params.finer), 0.0, top);
    const uint below = uint(floor(l));
    const uint above = min(below + 1, params.levels - 1);
    const float t = l - float(below);
    const float uu = saturate(u);
    const float vv = saturate(v);
    const WallRead a = wallLevelRead(image, below, uu, vv);
    const WallRead b = wallLevelRead(image, above, uu, vv);
    WallRead r;
    r.value = lerp(a.value, b.value, t);
    r.du = lerp(a.du, b.du, t);
    r.dv = lerp(a.dv, b.dv, t);
    r.dLevel = b.value - a.value;
    return r;
}

/// WHERE A GAUSSIAN'S READS START: where its normal, turned towards the wall,
/// meets the wall's plane -- what a matt surface takes most of -- or, where
/// it does not, the wall's point nearest the gaussian. Wide, and faint: the
/// fit narrows and raises them. Lobes after the first start a little aside.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralWallInit(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint row = i * 4;
    const float3x3 axes = axesOfQuaternion(decodeQuaternion(shape[row]));
    const float3 scale = float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2]));
    const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    float3 normal = float3(axes[0][shortest], axes[1][shortest], axes[2][shortest]);
    const float3 x = positions[i].xyz;
    const float3 toWall = params.centre - x;
    if (dot(normal, toWall) < 0.0) {
        normal = -normal;
    }
    const float3 wallNormal = cross(params.axisU, params.axisV);
    const float facing = dot(wallNormal, normal);
    const float away = dot(wallNormal, toWall);
    float3 hit = x + wallNormal * away;   // the wall's plane, nearest the gaussian
    if (abs(facing) > 1.0e-4 && away / facing > 0.0) {
        hit = x + normal * (away / facing);
    }
    const float3 on = hit - params.centre;
    const float u = saturate(dot(on, params.axisU) / (2.0 * params.halfU) + 0.5);
    const float v = saturate(dot(on, params.axisV) / (2.0 * params.halfV) + 0.5);
    if (params.cells != 0) {
        // A grid starts as the wall's mean: every cell the same, and the gain what a read's would be.
        const uint first = i * (params.cells + 3);
        for (uint c = 0; c < params.cells; ++c) {
            made[first + c] = log(1.0 / float(params.cells));
        }
        const float start = log(max(params.startGain, 1.0e-6));
        made[first + params.cells] = start;
        made[first + params.cells + 1] = start;
        made[first + params.cells + 2] = start;
        return;
    }
    for (uint lobe = 0; lobe < params.lobes; ++lobe) {
        const uint at = (i * params.lobes + lobe) * kWallLobe;
        const float du = lobe == 0 ? 0.0 : (hash01(i * 2654435761u + lobe * 97u) - 0.5) * 0.5;
        const float dv = lobe == 0 ? 0.0 : (hash01(i * 40503u + lobe * 131u + 7u) - 0.5) * 0.5;
        made[at] = saturate(u + du);
        made[at + 1] = saturate(v + dv);
        made[at + 2] = params.startLevel;
        const float gain = log(max(params.startGain / float(params.lobes), 1.0e-6));
        made[at + 3] = gain;
        made[at + 4] = gain;
        made[at + 5] = gain;
    }
}

/// A picture for the wall, written on the device: level zero of picture
/// `image`. 0: one texel lit (`argument`), white. 1: left to right, red into
/// blue. 2: a checker of `argument` texels, warm and cool. 3: a colour a
/// texel, at random. 4: a bright window, `argument` its texel, on a dim wall.
/// 5: all of it, white. 6: the picture `frameIn` holds. 7: a colour a block of
/// texels. 8: one texel in four lit, the rest dark.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralWallPattern(uint3 tid: SV_DispatchThreadID) {
    const uint texels = params.width * params.height;
    if (tid.x >= texels) {
        return;
    }
    const uint x = tid.x % params.width;
    const uint y = tid.x / params.width;
    float3 c = float3(0.0);
    switch (params.pattern) {
    case 0: c = tid.x == params.argument ? float3(1.0) : float3(0.0); break;
    case 1: {
        const float t = (float(x) + 0.5) / float(params.width);
        c = float3(1.0 - t, 0.15, t);
        break;
    }
    case 2: {
        const uint cell = max(params.argument, 1u);
        c = ((x / cell + y / cell) & 1u) != 0u ? float3(1.0, 0.7, 0.3) : float3(0.1, 0.3, 0.9);
        break;
    }
    case 3:
        c = float3(hash01(tid.x * 3u + params.argument), hash01(tid.x * 3u + 1u + params.argument),
                   hash01(tid.x * 3u + 2u + params.argument));
        break;
    case 4: {
        const int wx = int(params.argument % params.width);
        const int wy = int(params.argument / params.width);
        const bool inside = abs(int(x) - wx) <= 1 && abs(int(y) - wy) <= 0;
        c = inside ? float3(1.0, 0.95, 0.8) : float3(0.03, 0.04, 0.08);
        break;
    }
    case 6: c = frameIn[tid.x].rgb; break;
    case 7: {
        // A colour a block of texels, at random: `argument`'s low two bits say how many texels a block is
        // wide (2, 4, 8, 16), the rest is the seed.
        const uint cell = 2u << (params.argument & 3u);
        const uint block = (y / cell) * 4099u + (x / cell);
        const uint seed = params.argument >> 2;
        c = float3(hash01(block * 3u + seed), hash01(block * 3u + 1u + seed), hash01(block * 3u + 2u + seed));
        break;
    }
    case 8: {
        // A texel in four lit, at random and brightly, the rest dark: `argument` is the seed.
        const bool on = hash01(tid.x * 7u + params.argument) < 0.25;
        c = on ? 2.0 * float3(hash01(tid.x * 3u + params.argument), hash01(tid.x * 3u + 1u + params.argument),
                              hash01(tid.x * 3u + 2u + params.argument))
               : float3(0.0);
        break;
    }   // a picture somebody holds: `frameIn`, the wall's size
    default: c = float3(1.0); break;
    }
    imageOut[params.image * wallTexels() + tid.x] = float4(c, 1.0);
}

/// A level of a picture's pyramid from the one below it: the mean of the
/// texels under each of its own.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralWallPyramid(uint3 tid: SV_DispatchThreadID) {
    const uint2 size = wallSize(params.level);
    if (tid.x >= size.x * size.y || params.level == 0) {
        return;
    }
    const uint2 under = wallSize(params.level - 1);
    const uint first = params.image * wallTexels();
    const uint from = first + wallOffset(params.level - 1);
    const uint x = tid.x % size.x;
    const uint y = tid.x / size.x;
    // Two texels under each along an axis that halved, one along an axis already a texel wide.
    const uint spanX = under.x > size.x ? 2u : 1u;
    const uint spanY = under.y > size.y ? 2u : 1u;
    float4 sum = float4(0.0);
    for (uint dy = 0; dy < spanY; ++dy) {
        for (uint dx = 0; dx < spanX; ++dx) {
            sum += imageOut[from + min(y * spanY + dy, under.y - 1) * under.x + min(x * spanX + dx, under.x - 1)];
        }
    }
    imageOut[first + wallOffset(params.level) + tid.x] = sum / float(spanX * spanY);
}

/// What the wall's picture adds to each gaussian's colour in this frame.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralWallShade(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    if (visible[slot] == 0u) {
        return;
    }
    float3 sum = float3(0.0);
    if (params.cells != 0) {
        const uint row = i * (params.cells + 3);
        const uint first = params.image * wallTexels() + wallOffset(params.finer + params.gridLevel);
        for (uint c = 0; c < params.cells; ++c) {
            sum += exp(wallLatent[row + c]) * wallImage[first + c].rgb;
        }
        sum *= exp(float3(wallLatent[row + params.cells], wallLatent[row + params.cells + 1],
                          wallLatent[row + params.cells + 2]));
        proj[slot].colour.xyz += sum;
        return;
    }
    for (uint lobe = 0; lobe < params.lobes; ++lobe) {
        const uint at = (i * params.lobes + lobe) * kWallLobe;
        const float3 gain = exp(float3(wallLatent[at + 3], wallLatent[at + 4], wallLatent[at + 5]));
        sum += gain * wallRead(params.image, wallLatent[at], wallLatent[at + 1], wallLatent[at + 2]).value;
    }
    proj[slot].colour.xyz += sum;
}

/// The error's derivative at each pixel: neural_fit.slang's, the same relMSE.
[shader("compute")]
[numthreads(16, 16, 1)]
void neuralWallError(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.frameWidth || tid.y >= params.frameHeight) {
        return;
    }
    const uint at = tid.y * params.frameWidth + tid.x;
    const float3 wanted = teacher[at].rgb;
    const float3 d = student[at].rgb - wanted;
    const float level = dot(wanted, wanted) / 3.0 + 1.0e-2;
    pixelGrad[at] = float4(clamp(2.0 * d / level, float3(-50.0), float3(50.0)), 0.0);
    InterlockedAdd(loss[tid.y], uint(min(dot(d, d) / (3.0 * level), 64.0) * 1024.0 + 0.5));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void neuralWallClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.count) {
        cleared[tid.x] = 0u;
    }
}

/// One gaussian's reads, pulled on by what the blend carried back to its colour.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralWallBackward(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    const int3 summed = int3(int(colourGrad[slot * 4]), int(colourGrad[slot * 4 + 1]), int(colourGrad[slot * 4 + 2]));
    const bool seen = any(summed != int3(0));
    const float3 pulled = float3(summed) / kColourGradScale;
    if (params.cells != 0) {
        const uint row = i * (params.cells + 3);
        if (!seen) {
            for (uint k = 0; k < params.cells + 3; ++k) {
                gradient[row + k] = 0.0;
            }
            return;
        }
        const uint first = params.image * wallTexels() + wallOffset(params.finer + params.gridLevel);
        const float3 gain = exp(float3(wallLatent[row + params.cells], wallLatent[row + params.cells + 1],
                                       wallLatent[row + params.cells + 2]));
        float3 sum = float3(0.0);
        for (uint c = 0; c < params.cells; ++c) {
            const float w = exp(wallLatent[row + c]);
            const float3 texel = wallImage[first + c].rgb;
            sum += w * texel;
            // Each is kept as its log: its own derivative is the weight's times the weight.
            gradient[row + c] = dot(pulled * gain, texel) * w;
        }
        gradient[row + params.cells] = pulled.x * gain.x * sum.x;
        gradient[row + params.cells + 1] = pulled.y * gain.y * sum.y;
        gradient[row + params.cells + 2] = pulled.z * gain.z * sum.z;
        return;
    }
    for (uint lobe = 0; lobe < params.lobes; ++lobe) {
        const uint at = (i * params.lobes + lobe) * kWallLobe;
        if (!seen) {
            for (uint k = 0; k < kWallLobe; ++k) {
                gradient[at + k] = 0.0;
            }
            continue;
        }
        const float3 gain = exp(float3(wallLatent[at + 3], wallLatent[at + 4], wallLatent[at + 5]));
        const WallRead r = wallRead(params.image, wallLatent[at], wallLatent[at + 1], wallLatent[at + 2]);
        const float3 through = pulled * gain;
        gradient[at] = dot(through, r.du);
        gradient[at + 1] = dot(through, r.dv);
        gradient[at + 2] = dot(through, r.dLevel);
        // The gain is kept as its log: its own derivative is the gain's times the gain.
        gradient[at + 3] = through.x * r.value.x;
        gradient[at + 4] = through.y * r.value.y;
        gradient[at + 5] = through.z * r.value.z;
    }
}

/// Adam, a number a thread, and each number kept where it means something:
/// u and v on the wall, the level among the pyramid's.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralWallAdam(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float g = gradientRead[i];
    if (g == 0.0) {
        return;
    }
    // A grid's numbers are all logs of a gain: the last kind.
    const uint which = params.cells != 0 ? 3u : i % kWallLobe;
    const float m = params.beta1 * moment1[i] + (1.0 - params.beta1) * g;
    const float v = params.beta2 * moment2[i] + (1.0 - params.beta2) * g * g;
    moment1[i] = m;
    moment2[i] = v;
    const float rate = which < 2 ? params.rateUv : which == 2 ? params.rateLevel : params.rateGain;
    float moved = made[i] - rate * (m / params.correction1) / (sqrt(v / params.correction2) + params.epsilon);
    if (which < 2) {
        moved = saturate(moved);
    } else if (which == 2) {
        moved = clamp(moved, 0.0, float(params.levels - 1 - params.finer));
    } else {
        moved = clamp(moved, -14.0, 6.0);
    }
    made[i] = moved;
}

/// A frame added into another, times a colour: a wall's picture is a sum of
/// its texels, and so is the frame it lights -- the frame under each texel
/// alone, times what the picture holds there (`byTexel`), or times a tint.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralWallAccumulate(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.frameWidth * params.frameHeight) {
        return;
    }
    const float3 tint = params.byTexel != 0 ? wallImage[params.image * wallTexels() + wallOffset(params.finer) + params.texel].rgb
                                            : float3(params.tintR, params.tintG, params.tintB);
    const float4 f = frameIn[tid.x];
    const float4 held = params.count != 0 ? frameOut[tid.x] : float4(0.0);
    frameOut[tid.x] = float4(held.rgb + f.rgb * tint, max(held.a, f.a));
}
