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
  ATVA  ancillary data (JSON), right after every manifest and whenever its
        file changes: --anc FILE.json, or --env URL to send an environment
        block (the dome the frames were drawn under; docs/operations.md).
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
once and the schedule kept. It sleeps to --spin-ms (12) before a frame's
time and spins the rest: a sleeping process is woken up to ~10 ms late (3 ms
of spin left 7 ms of jitter, 12 ms left 0.5 ms). `--stats`
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



def json_message(magic, kind, payload_object, frame):
    """The fixed prefix, a u32 length and JSON, padded to four bytes: a manifest or ancillary data."""
    payload = json.dumps(payload_object, separators=(",", ":")).encode()
    body = struct.pack("<4sHHIQIHHI", magic, kind, 0xFFFF, frame, time.time_ns(), 0, 2, 0, len(payload))
    body += payload
    body += b"\0" * (-len(body) % 4)
    body = bytearray(body)
    struct.pack_into("<I", body, athv.BYTES_AT, len(body))
    return struct.pack("<I", len(body)) + bytes(body)


def manifest_message(manifest, frame):
    return json_message(b"ATVM", athv.MANIFEST, manifest, frame)


# The sparrow's dome, upright (camera_sky_upright.usda): the DomeLight turned rotateX(90) on a Z-up stage, so a
# direction d in its own frame (pole +Y) is (dx, -dz, dy) in the stage's world, its pole world +Z. The cloud's own
# space (where decoded positions are) is the stage's world turned by /Live's rotateX(270): world = (x, z, -y), a
# world direction w is (wx, -wz, wy) there. Together d is (dx, -dy, -dz) in the cloud's space: toWorld = these rows.
DOME_TO_CLOUD = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]


def environment_block(url):
    return dict(environment=dict(url=url, mime="image/vnd.radiance", format="latlong", toWorld=DOME_TO_CLOUD,
                                 intensity=1.0, exposure=0.0, background=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("athv", nargs="+", help="the .athv files, one a rendition")
    ap.add_argument("--rate", type=float, default=0.0, help="frames a second (default: the files')")
    ap.add_argument("--loops", type=int, default=0, help="times the frames are played (default 0: forever)")
    ap.add_argument("--manifest-every", type=int, default=0, help="frames between manifests (default: a second)")
    ap.add_argument("--anc", metavar="JSON", help="ancillary data to send (a JSON file), re-read when it changes")
    ap.add_argument("--env", metavar="URL", help="send an environment block naming this URL (the sparrow's dome)")
    ap.add_argument("--spin-ms", type=float, default=12.0, help="ms before a frame's time spent spinning, not asleep")
    ap.add_argument("--stats", action="store_true", help="a line a second on stderr: frames, Mbit/s, lateness")
    a = ap.parse_args()
    spin_ns = int(a.spin_ms * 1e6)

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
    anc = {"mtime": None, "payload": None}

    def ancillary():
        """The ancillary payload, and whether it changed since it was last sent."""
        if a.anc:
            try:
                mtime = os.stat(a.anc).st_mtime_ns
            except OSError:
                return anc["payload"], False
            if mtime != anc["mtime"]:
                with open(a.anc) as f:
                    anc["payload"] = json.load(f)
                changed = anc["mtime"] is not None
                anc["mtime"] = mtime
                return anc["payload"], changed
            return anc["payload"], False
        if a.env and anc["payload"] is None:
            anc["payload"] = environment_block(a.env)
        return anc["payload"], False

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
            if due - now > spin_ns:
                time.sleep((due - now - spin_ns) / 1.0e9)
            while time.monotonic_ns() < due:
                pass
            worst_late = max(worst_late, time.monotonic_ns() - due)
            payload, changed = ancillary()
            if k % every == 0 or changed:
                if k % every == 0:
                    m = manifest_message(manifest, k)
                    out.write(m)
                    sent_bytes += len(m)
                if payload is not None:
                    m = json_message(b"ATVA", 2, payload, k)
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
