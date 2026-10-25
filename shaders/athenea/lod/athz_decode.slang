// Copyright (c) 2026 jesus luque.
//
// The athz decoder for tables of up to 2^12 entries (16 KiB of workgroup
// memory): Metal, CUDA, Vulkan. athz_decode.slangh is the body.
module athz_decode;
#define ATHZ_TABLE_LOG_MAX 12
#include "athz_decode.slangh"
