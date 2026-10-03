// Copyright (c) 2026 jesus luque.
//
// How many words of a buffer are not the word expected: what a test of a
// carried array reads back instead of the array (the levels of detail and
// .athc carrying a transfer, task TX).
module words_check;

struct WordsParams {
    uint count;      ///< words
    uint expected;   ///< what each must be
    uint pad0;
    uint pad1;
};

StructuredBuffer<uint>   words;
RWStructuredBuffer<uint> wrong;   ///< [0] words off, [1] words read
ConstantBuffer<WordsParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void wordsCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    if (words[tid.x] != params.expected) {
        InterlockedAdd(wrong[0], 1u);
    }
    InterlockedAdd(wrong[1], 1u);
}
