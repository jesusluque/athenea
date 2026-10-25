#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""Renditions of a .athv video played out in real time on standard output: the Python standard library only.

Reads one or more `.athv` files (`athenea video-encode`, one per rendition:
the same frames at other sizes and qualities, each with its own
`--rendition` id) and writes their records to stdout at the files' frame
rate, looping, frame-aligned: every rendition's pictures of presentation
frame k before any of frame k + 1. Each message is a u32 little-endian
length and then that many bytes:

  ATVM  the manifest (JSON): the rate as a fraction, the loop's frame count,
        and per rendition its id, gaussians, SH degree, quality, pictures a
        frame and their size, mean Mbit/s, codec string and hvcC. Sent first
        and again every --manifest-every frames (a second by default), just
        before that frame's pictures.
  ATVF  a picture, byte for byte as the file holds it but for two fields at
        fixed offsets: the presentation frame (offset 8, u32: counts on
        across loops, the same for the same instant in every rendition) and
        the PTP time (offset 12, u64: CLOCK_REALTIME in ns when it is sent).

    python3 scripts/examples/athv_stream.py r0.athv r1.athv r2.athv     # forever, at the files' rate
    python3 scripts/examples/athv_stream.py r0.athv --rate 24 --loops 2 --stats
    python3 scripts/examples/athv_stream.py r0.athv | \\
        srt-live-transmit file://con "srt://127.0.0.1:9000?mode=caller&latency=120"

Pacing is by the monotonic clock against a fixed schedule (frame k is due at
start + k / rate), so the rate does not drift; a frame sent late is sent at
once and the schedule kept. It sleeps to 3 ms before a frame's time and
spins the rest, as a sleeping process is woken up to ~10 ms late. `--stats`
prints, every second on stderr, what was sent.

The messages are docs/operations.md §8.2 ("Video sent as a stream"); the file
and its records are modules/video/include/athenea/video/Athv.h.
"""
import argparse
import json
import mmap
import os
import struct
import sys
import time

import athv_read as athv

SPIN_NS = 3_000_000


def manifest_message(manifest, frame):
    payload = json.dumps(manifest, separators=(",", ":")).encode()
    body = struct.pack("<4sHHIQIHHI", b"ATVM", athv.MANIFEST, 0xFFFF, frame, time.time_ns(), 0, 2, 0, len(payload))
    body += payload
    body += b"\0" * (-len(body) % 4)
    body = bytearray(body)
    struct.pack_into("<I", body, athv.BYTES_AT, len(body))
    return struct.pack("<I", len(body)) + bytes(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("athv", nargs="+", help="the .athv files, one a rendition")
    ap.add_argument("--rate", type=float, default=0.0, help="frames a second (default: the files')")
    ap.add_argument("--loops", type=int, default=0, help="times the frames are played (default 0: forever)")
    ap.add_argument("--manifest-every", type=int, default=0, help="frames between manifests (default: a second)")
    ap.add_argument("--stats", action="store_true", help="a line a second on stderr: frames, Mbit/s, lateness")
    a = ap.parse_args()

    files = []
    for path in a.athv:
        with open(path, "rb") as f:
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        h, _lay, records, _first, entry = athv.describe(data)
        # Records by source frame, slices in order.
        by_frame = {}
        for at, size in records:
            source, = struct.unpack_from("<I", data, at + 32)
            by_frame.setdefault(source, []).append((at, size))
        entry["file"] = os.path.basename(path)
        files.append(dict(data=data, header=h, frames=by_frame, entry=entry))
    ids = [f["header"]["rendition"] for f in files]
    if len(set(ids)) != len(ids):
        raise SystemExit("athv_stream: two files are rendition %d" % next(i for i in ids if ids.count(i) > 1))
    first = files[0]["header"]
    for f in files[1:]:
        h = f["header"]
        if (h["rateNumerator"], h["rateDenominator"]) != (first["rateNumerator"], first["rateDenominator"]) or \
                len(f["frames"]) != len(files[0]["frames"]):
            raise SystemExit("athv_stream: %s is not the same frames at the same rate as %s"
                             % (f["entry"]["file"], files[0]["entry"]["file"]))
    files.sort(key=lambda f: f["header"]["rendition"])
    frames = len(files[0]["frames"])
    rate = a.rate if a.rate > 0 else first["rate"]
    num, den = (first["rateNumerator"], first["rateDenominator"]) if a.rate <= 0 else (round(a.rate * 1000), 1000)
    for f in files:
        f["entry"]["meanMbps"] = round(f["entry"]["meanBytesPerFrame"] * 8 * rate / 1e6, 2)
    manifest = dict(version=1, fps=dict(num=num, den=den), frames=frames, renditions=[f["entry"] for f in files])
    every = a.manifest_every if a.manifest_every > 0 else max(1, round(rate))

    out = sys.stdout.buffer
    period_ns = 1.0e9 / rate
    start = time.monotonic_ns()
    k = 0
    sent_bytes = sent_frames = worst_late = 0
    last_stats = start
    try:
        while a.loops <= 0 or k < a.loops * frames:
            due = start + int(round(k * period_ns))
            now = time.monotonic_ns()
            if due - now > SPIN_NS:
                time.sleep((due - now - SPIN_NS) / 1.0e9)
            while time.monotonic_ns() < due:
                pass
            worst_late = max(worst_late, time.monotonic_ns() - due)
            if k % every == 0:
                m = manifest_message(manifest, k)
                out.write(m)
                sent_bytes += len(m)
            source = k % frames
            for f in files:
                for at, size in f["frames"][source]:
                    record = bytearray(f["data"][at:at + size])
                    struct.pack_into("<I", record, athv.FRAME_AT, k)
                    struct.pack_into("<Q", record, athv.PTP_AT, time.time_ns())
                    out.write(struct.pack("<I", size))
                    out.write(record)
                    sent_bytes += 4 + size
            out.flush()
            k += 1
            sent_frames += 1
            if a.stats:
                now = time.monotonic_ns()
                if now - last_stats >= 1_000_000_000:
                    span = (now - last_stats) / 1.0e9
                    sys.stderr.write("athv_stream: frame %d, %.1f frames/s, %.1f Mbit/s, latest %.1f ms late\n"
                                     % (k, sent_frames / span, sent_bytes * 8 / span / 1.0e6, worst_late / 1.0e6))
                    sent_frames = sent_bytes = worst_late = 0
                    last_stats = now
    except (BrokenPipeError, KeyboardInterrupt):
        # The reader went away: nothing more to flush at exit.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


if __name__ == "__main__":
    main()
