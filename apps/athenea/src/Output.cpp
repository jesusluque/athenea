// Copyright (c) 2026 jesus luque.
#include "Output.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <string>

namespace athenea::cli {
namespace {

std::atomic<bool> gCancel{false};
LineSink gSink = nullptr;
void*    gUser = nullptr;

void emit(int error, const char* format, va_list args) {
    if (gSink == nullptr) {
        std::vfprintf(error != 0 ? stderr : stdout, format, args);
        return;
    }
    va_list again;
    va_copy(again, args);
    const int length = std::vsnprintf(nullptr, 0, format, again);
    va_end(again);
    if (length <= 0) {
        return;
    }
    std::string text(static_cast<size_t>(length), '\0');
    std::vsnprintf(text.data(), text.size() + 1, format, args);
    gSink(error, text.c_str(), gUser);
}

}   // namespace

void setLineSink(LineSink sink, void* user) {
    gSink = sink;
    gUser = user;
}

void requestCancel(bool on) { gCancel.store(on, std::memory_order_relaxed); }

bool cancelRequested() { return gCancel.load(std::memory_order_relaxed); }

void out(const char* format, ...) {
    va_list args;
    va_start(args, format);
    emit(0, format, args);
    va_end(args);
}

void err(const char* format, ...) {
    va_list args;
    va_start(args, format);
    emit(1, format, args);
    va_end(args);
}

}   // namespace athenea::cli
