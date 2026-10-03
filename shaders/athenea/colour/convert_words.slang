// Copyright (c) 2026 jesus luque.
//
// COLOURS THAT ARE NOT TEXTURES, INTO THE WORKING SPACE, WHERE THEY LIE.
//
// A material's constant, a light's colour, a prim's displayColor, a point's
// colour: each is uploaded as it was authored, in the space it was authored
// in, and converted on the device in the buffer that holds it. The host only
// notes where the colours are and what space they are in (bookkeeping).
//
// The entry, `convertColourWords`, is generated for each source space
// (colour::ColourCompiler::convertWords): this module's `colourWordsAt`,
// `loadColourWords` and `storeColourWords` around the compiled colour
// function (`atheneaCs_<hash>`), as texture_decode's entry is around its
// own. A colour is three floats, `colourStride` words apart from the next,
// or (`colourHalves`) four halves in two words, r|g and b|a; alpha is never
// converted. One dispatch converts one run: `colourCount` colours from word
// `colourFirst`, or (`colourListed`) at the words `colourList` names -- a
// material's constants, scattered through its row.
module convert_words;

public RWStructuredBuffer<uint> colourWords;
public uniform uint             colourFirst;
public uniform uint             colourCount;
public uniform uint             colourStride;   // words from one colour to the next
public uniform uint             colourHalves;   // 1: f16 r|g, b|a in two words
public StructuredBuffer<uint>   colourList;     // the first word of each colour, where listed
public uniform uint             colourListed;   // 1: `colourList` says where each colour is

/// The first word of the colour this thread converts; false past the run.
public bool colourWordsAt(uint3 tid, out uint word) {
    word = 0;
    if (tid.x >= colourCount) {
        return false;
    }
    word = colourListed != 0 ? colourList[tid.x] : colourFirst + tid.x * colourStride;
    return true;
}

public float3 loadColourWords(uint word) {
    if (colourHalves != 0) {
        const uint rg = colourWords[word];
        const uint ba = colourWords[word + 1];
        return float3(f16tof32(rg & 0xFFFFu), f16tof32(rg >> 16), f16tof32(ba & 0xFFFFu));
    }
    return float3(asfloat(colourWords[word]), asfloat(colourWords[word + 1]), asfloat(colourWords[word + 2]));
}

public void storeColourWords(uint word, float3 c) {
    if (colourHalves != 0) {
        const uint ba = colourWords[word + 1];
        colourWords[word] = f32tof16(c.r) | (f32tof16(c.g) << 16);
        colourWords[word + 1] = f32tof16(c.b) | (ba & 0xFFFF0000u);
        return;
    }
    colourWords[word] = asuint(c.r);
    colourWords[word + 1] = asuint(c.g);
    colourWords[word + 2] = asuint(c.b);
}
