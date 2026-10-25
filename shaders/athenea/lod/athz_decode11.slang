// Copyright (c) 2026 jesus luque.
//
// The athz decoder for tables of up to 2^11 entries (8 KiB of workgroup
// memory, inside WebGPU's 16 384 bytes). athz_decode.slangh is the body.
module athz_decode11;
#define ATHZ_TABLE_LOG_MAX 11
#include "athz_decode.slangh"
