// Copyright (c) 2026 jesus luque.
//
// Where a command's lines go. In the `athenea` binary, stdout and stderr, as
// printf would put them. Run inside another program through the plugin's
// entry point (Embedded.cpp) -- Blender's add-on converting a mesh -- they go
// to that program's sink instead, so it can show them where its user looks:
// a terminal is not where a Blender user is.
#pragma once

namespace athenea::cli {

/// A host's sink: `error` is 1 for what the command writes to stderr. `text`
/// is what one call printed (usually one line, with its newline).
using LineSink = void (*)(int error, const char* text, void* user);

/// Replaces where `out` and `err` write; null restores stdout and stderr.
/// Not thread-safe: set around one command, which the entry point serialises.
void setLineSink(LineSink sink, void* user);

/// printf to stdout, or to the sink.
void out(const char* format, ...) __attribute__((format(printf, 1, 2)));
/// printf to stderr, or to the sink.
void err(const char* format, ...) __attribute__((format(printf, 1, 2)));

}   // namespace athenea::cli
