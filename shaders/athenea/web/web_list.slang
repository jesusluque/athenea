// Copyright (c) 2026 jesus luque.
//
// THE DRAW LIST the web's level of detail hands the projection: one buffer,
// word 0 the count the cut wants drawn (a prefix sum's total, written there),
// and from word kListHead -- 256 bytes on, a storage binding's alignment --
// the pool indices to draw, in pool order. A list, not a copy: what the cut
// leaves out is never touched again (docs/decisions.md, "The web viewer").
module web_list;

public static const uint kListHead = 64;
