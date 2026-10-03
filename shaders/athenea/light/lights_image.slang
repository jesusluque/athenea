// Copyright (c) 2026 jesus luque.
//
// A dome's image, and the warp that samples by it. This is the half of the
// light module that needs a texture table, kept apart so that a kernel
// without one can still light a surface: the splat projection lives below
// material in the module order and cannot have it.
//
// sampleLightImaged and lightPdfImaged are what a kernel with a table calls;
// they use the image where a dome has one and defer to the core where it does
// not.
module lights_image;
__exported import athenea.light.lights;
__exported import athenea.material.material_inputs;

/// A dome's image along `wi`, white where it has none.
public float3 domeImage(LightRecord l, float3 wi) {
    if (l.texture == 0xFFFFFFFF) {
        return float3(1.0);
    }
    const float2 uv = domeUv(l, wi);
    bool found;
    const float4 value = sampleTexture(gTextures, l.texture, l.sampler, uv, float2(0.0), float2(0.0), found);
    return found ? value.rgb : float3(1.0);
}


/// One texel of one level, read exactly. Filtering would blend neighbours,
/// and then the four children of a cell would not sum to their parent: the
/// descent's probabilities would stop telescoping and the closed-form density
/// would no longer be the one the samples were drawn with.
public float domeTexelOf(LightRecord l, int2 xy, uint level) {
    const TextureRecord record = gTextures.records[l.texture];
    const int2 size = int2(max(int(record.width) >> level, 1), max(int(record.height) >> level, 1));
    const int2 at = clamp(xy, int2(0, 0), size - 1);
    const float3 rgb = gTextures.textures[record.slot].Load(int3(at, int(level))).rgb;
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

/// The same by uv, for a density that is asked about a direction.
float domeLuminance(LightRecord l, float2 uv, float level) {
    const TextureRecord record = gTextures.records[l.texture];
    const uint mip = uint(max(level, 0.0));
    const int2 size = int2(max(int(record.width) >> mip, 1), max(int(record.height) >> mip, 1));
    return domeTexelOf(l, int2(int(uv.x * float(size.x)), int(uv.y * float(size.y))), mip);
}

/// A lat-long texel covers 2 pi^2 sin(theta) du dv of solid angle.
static const float kTwoPiSquared = 2.0 * 3.14159265358979 * 3.14159265358979;
static const uint  kDomeNoCdf = 0xFFFFFFFF;

/// Whether a dome is worth sampling by its image rather than around the
/// surface. It must have a distribution (LightTable builds one for every
/// dome whose image loaded, dome_cdf.slang) with something in it. And it
/// must vary: an image that barely does is better served by the cosine,
/// which never draws a direction below the surface, where the image's
/// distribution sends half its samples for a flat sky. One with a sun in it
/// is the opposite. The 1x1 level gives the mean and a middle level the
/// spread, so the choice costs a handful of reads.
///
/// The warp this replaced descended the mip chain, splitting each cell
/// among its children, and was off from the start: a chi-square read z 61543
/// over a million samples. The distribution is now the exact two-level CDF
/// Cycles builds (kernel/light/background.h, Apache-2.0).
public bool domeHasImage(LightRecord l) {
    if (l.texture == 0xFFFFFFFF || l.cdf == kDomeNoCdf) {
        return false;
    }
    if ((gTextures.records[l.texture].flags & 1u) == 0) {
        return false;
    }
    if (!(iesValues[l.cdf + 3] > 0.0)) {
        return false;
    }
    const TextureRecord record = gTextures.records[l.texture];
    const float top = ceil(log2(max(max(record.width, record.height), 1.0)));
    const float average = domeLuminance(l, float2(0.5, 0.5), top);
    if (!(average > 0.0)) {
        return false;
    }
    // Four quarters of the image against its mean: enough to tell a sky with
    // a sun from a sky without one, and it reads the chain already built.
    const float level = max(top - 1.0, 0.0);
    float spread = 0.0;
    for (uint k = 0; k < 4; ++k) {
        const float2 uv = float2(k % 2 == 0 ? 0.25 : 0.75, k < 2 ? 0.25 : 0.75);
        spread = max(spread, abs(domeLuminance(l, uv, level) - average));
    }
    return spread > 0.25 * average;
}

/// The largest k in [0, count) whose running sum at `at + k` is at most
/// `target`: the cell a number drawn in proportion lands in. A cell of no
/// weight shares its start with the next and is passed over.
uint domeCdfSearch(uint at, uint count, float target) {
    uint lo = 0;
    uint hi = count - 1;
    while (lo < hi) {
        const uint mid = (lo + hi + 1) >> 1;
        if (iesValues[at + mid] <= target) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

/// The density of `wi` in solid angle: the texel's share of the marginal
/// times its share of its row, spread over the texel's uv, over the solid
/// angle a unit of uv covers there,
///
///     pdf(wi) = p(row) p(column | row) width height / (2 pi^2 sin(theta))
///
/// read from the same sums the sample was searched in, so the two agree to
/// the rounding of a difference.
public float domeImagePdf(LightRecord l, float3 wi) {
    if (!domeHasImage(l)) {
        return 0.0;
    }
    const uint base = l.cdf;
    const uint w = uint(iesValues[base]);
    const uint h = uint(iesValues[base + 1]);
    const float total = iesValues[base + 3];
    const float2 uv = domeUv(l, wi);
    const float sinTheta = sin(uv.y * 3.14159265358979);
    if (sinTheta <= 1.0e-6) {
        return 0.0;
    }
    const uint i = min(uint(max(uv.x, 0.0) * float(w)), w - 1);
    const uint j = min(uint(max(uv.y, 0.0) * float(h)), h - 1);
    const uint marginal = base + 4;
    const float rowWeight = iesValues[marginal + j + 1] - iesValues[marginal + j];
    const uint row = base + 5 + h + j * (w + 1);
    const float rowTotal = iesValues[row + w];
    if (!(rowWeight > 0.0) || !(rowTotal > 0.0)) {
        return 0.0;
    }
    const float cell = iesValues[row + i + 1] - iesValues[row + i];
    const float pUv = (rowWeight / total) * (cell / rowTotal) * float(w) * float(h);
    return pUv / (kTwoPiSquared * sinTheta);
}

/// A row by the marginal with u.y, a column by that row's conditional with
/// u.x, and the place inside the texel from what is left of each number --
/// so both are spent rather than merely consumed.
public LightSample domeImageSample(LightRecord l, float2 u) {
    LightSample s;
    s.wi = float3(0.0, 0.0, 1.0);
    s.distance = 1.0e30;
    s.radiance = float3(0.0);
    s.pdf = 0.0;
    s.delta = false;
    s.valid = false;
    const uint base = l.cdf;
    const uint w = uint(iesValues[base]);
    const uint h = uint(iesValues[base + 1]);
    const float total = iesValues[base + 3];
    const uint marginal = base + 4;
    const float ty = u.y * total;
    const uint j = domeCdfSearch(marginal, h, ty);
    const float m0 = iesValues[marginal + j];
    const float m1 = iesValues[marginal + j + 1];
    if (!(m1 > m0)) {
        return s;
    }
    const uint row = base + 5 + h + j * (w + 1);
    const float tx = u.x * iesValues[row + w];
    const uint i = domeCdfSearch(row, w, tx);
    const float c0 = iesValues[row + i];
    const float c1 = iesValues[row + i + 1];
    if (!(c1 > c0)) {
        return s;
    }
    const float2 inside = float2(saturate((tx - c0) / (c1 - c0)), saturate((ty - m0) / (m1 - m0)));
    const float2 uv = (float2(float(i), float(j)) + inside) / float2(float(w), float(h));
    s.wi = domeDirection(l, uv);
    // The density of the direction, as anyone asking about it is told: a
    // sample on a texel's very edge may come back through domeUv into its
    // neighbour, and then this, not the texel it was drawn in, is what MIS
    // and the chi-square must see.
    s.pdf = domeImagePdf(l, s.wi);
    s.radiance = lightEmission(l) * domeImage(l, s.wi);
    s.valid = s.pdf > 0.0 && any(s.radiance > float3(0.0));
    return s;
}
/// One sample of `l`, by its image where it has one.
public LightSample sampleLightImaged(LightRecord l, float3 p, float3 n, float2 u) {
    if (l.kind == kLightDome) {
        if (domeHasImage(l)) {
            return domeImageSample(l, u);
        }
        LightSample s = sampleLight(l, p, n, u);
        s.radiance *= domeImage(l, s.wi);
        return s;
    }
    return sampleLight(l, p, n, u);
}

/// The density that sample was drawn with.
public float lightPdfImaged(LightRecord l, float3 p, float3 n, float3 wi) {
    if (l.kind == kLightDome && domeHasImage(l)) {
        return domeImagePdf(l, wi);
    }
    return lightPdf(l, p, n, wi);
}

/// lightHit, with a dome's image.
public LightHit lightHitImaged(LightRecord l, float3 p, float3 wi) {
    LightHit h = lightHit(l, p, wi);
    if (h.valid && l.kind == kLightDome) {
        h.radiance *= domeImage(l, wi);
        h.valid = any(h.radiance > float3(0.0));
    }
    return h;
}

/// What a dome puts in a direction nothing was drawn in.
public float3 domeBackground(LightRecord l, float3 wi) {
    return lightEmission(l) * domeImage(l, wi);
}
