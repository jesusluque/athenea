#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""A live splat stream fed from another process: the Python standard library only.

Starts `athenea-mcp`, has it open a stage whose ParticleField names a stream
(AtheneaLiveSourceAPI; this script writes one if --stage is not given), asks it
for the stream's shared-memory ring (stream_open), and then, epoch after
epoch, writes SPZ files into the ring and commits them (stream_commit) as the
stream's chunks, rendering a frame after each epoch (render, a PNG).

    python3 scripts/examples/live_producer.py --mcp build/macos-arm64-release/bin/athenea-mcp \
        --epochs 6 --out /tmp/live a.spz b.spz c.spz

Each epoch puts every file in one chunk, turned by one place an epoch, so
consecutive frames differ.

With --subscribe the frames come the other way: the script queues cameras
with their PTP instants (camera_set, the system clock in nanoseconds, as the
pages' ptpTime), subscribes to the frames (render_subscribe, colour as shown,
eight bits, and depth, in poll mode: a reader with no device of its own),
and reads each epoch's frame from the shared memory the engine wrote it
into -- a memoryview over the slot, no copy -- writing it as a PNG. It says
it is done with a frame by writing the ring's `released` word. The pages are SPZ files as they are on disk; the
engine decompresses them on its loader threads. A PLY becomes an SPZ with
`athenea flatten` or Niantic's own tools.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time
import zlib
from multiprocessing import shared_memory


class Mcp:
    """JSON-RPC over the server's stdin and stdout, one message a line."""

    def __init__(self, binary):
        self.process = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.serial = 0
        self.ask("initialize", {"protocolVersion": "2025-06-18"})
        self.notify("notifications/initialized")

    def ask(self, method, params):
        self.serial += 1
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.serial, "method": method,
                                             "params": params}) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def notify(self, method):
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.process.stdin.flush()

    def tool(self, name, arguments):
        reply = self.ask("tools/call", {"name": name, "arguments": arguments})
        result = reply.get("result", {})
        text = "".join(block.get("text", "") for block in result.get("content", []))
        if result.get("isError") or "error" in reply:
            sys.exit(f"{name}: {text or reply.get('error')}")
        return text

    def close(self):
        self.process.stdin.close()
        self.process.wait()


def write_stage(path, stream, chunks, chunk_splats):
    with open(path, "w") as out:
        out.write(f'''#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)
def Xform "World"
{{
    def ParticleField3DGaussianSplat "Live" (
        prepend apiSchemas = ["AtheneaLiveSourceAPI"]
    )
    {{
        string primvars:athenea:live:stream = "{stream}" ( interpolation = "constant" )
        int primvars:athenea:live:chunks = {chunks} ( interpolation = "constant" )
        int primvars:athenea:live:chunkSplats = {chunk_splats} ( interpolation = "constant" )
    }}
    def Camera "Shot"
    {{
        float2 clippingRange = (0.1, 1000)
        float focalLength = 30
        double3 xformOp:translate = (0, 0, 6)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
''')


def open_ring(name):
    # Python 3.13 can be told not to track a region it did not make; before
    # that its resource tracker would remove the server's ring at exit.
    try:
        return shared_memory.SharedMemory(name=name, create=False, track=False)
    except TypeError:
        ring = shared_memory.SharedMemory(name=name, create=False)
        from multiprocessing import resource_tracker
        resource_tracker.unregister(ring._name, "shared_memory")
        return ring


# THE FRAME RING (bridge/Frames.h, RingHeader): where the words a reader
# needs lie in the shared memory render_subscribe answers.
RING_COMPLETED = 56      # u64: the newest frame the engine has seen written
RING_RELEASED = 64       # u64: the newest frame this reader is done with (poll mode)
RING_RECORDS = 528       # RingRecord[64], 256 bytes each
RECORD_BYTES = 256
RECORDS = 64


def ring_word(ring, offset):
    return struct.unpack_from("<Q", ring.buf, offset)[0]


def frame_record(ring, frame):
    """The record of `frame` (its slot, instant and epoch), or None when the ring has moved on."""
    at = RING_RECORDS + (frame % RECORDS) * RECORD_BYTES
    kept, slot, instant, epoch = struct.unpack_from("<QQQQ", ring.buf, at)
    if kept != frame or ring_word(ring, at) != frame:
        return None
    return {"slot": slot, "instant": instant, "epoch": epoch}


def write_png(path, width, height, rgba):
    """Eight-bit RGBA rows, top first, as they lie: a container, no arithmetic on the pixels."""
    row = width * 4
    raw = b"".join(b"\x00" + bytes(rgba[y * row:(y + 1) * row]) for y in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    with open(path, "wb") as out:
        out.write(b"\x89PNG\r\n\x1a\n")
        out.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
        out.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        out.write(chunk(b"IEND", b""))


def pose_at(base, degrees, up):
    """`base` (camera to world, row by row) turned `degrees` about the world's `up` axis through the origin."""
    import math
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if up == "z":
        turn = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    else:
        turn = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    rows = [base[0:4], base[4:8], base[8:12]]
    out = []
    for r in range(3):
        out += [sum(turn[r][k] * rows[k][col] for k in range(3)) for col in range(4)]
    return out + [0, 0, 0, 1]


def subscribe(mcp, args, pages, ring_pages):
    width, height = 640, 360
    now = time.time_ns()
    # Cameras with their instants: a slow move across the run.
    span = (args.epochs + 2) * 0.5e9
    base = [float(v) for v in args.pose.split()] if args.pose else [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 6, 0, 0, 0, 1]
    for k in range(5):
        mcp.tool("camera_set", {"ptpTime": int(now + k * span / 4),
                                "pose": pose_at(base, args.turn * (k / 4 - 0.5), args.up),
                                "focal": args.focal, "haperture": args.aperture,
                                "vaperture": args.aperture * height / width, "near": 0.01, "far": 1000})
    info = json.loads(mcp.tool("render_subscribe", {"aovs": [{"name": "color", "format": "rgba8"}, "depth"],
                                                    "width": width, "height": height, "rate": 30,
                                                    "sync": "poll"}))
    ring = open_ring(info["ring"])
    data = open_ring(info["data"])
    colour = info["aovs"][0]
    try:
        for epoch in range(1, args.epochs + 1):
            listed, at = [], 0
            epoch_pages = pages[(epoch - 1) % len(pages)] if args.sequence else \
                [pages[(chunk + epoch) % len(pages)] for chunk in range(len(pages))]
            for chunk, page in enumerate(epoch_pages):
                ring_pages.buf[at:at + len(page)] = page
                listed.append({"chunk": chunk, "op": "replace", "offset": at, "bytes": len(page),
                               "ptpTime": time.time_ns()})
                at += len(page)
            pushed = time.time_ns()
            mcp.tool("stream_commit", {"name": args.stream, "epoch": epoch, "pages": listed})
            # Read frames as they come until one shows this epoch.
            seen, deadline = ring_word(ring, RING_COMPLETED), time.time() + 10
            while time.time() < deadline:
                newest = ring_word(ring, RING_COMPLETED)
                if newest == seen:
                    time.sleep(0.001)
                    continue
                seen = newest
                record = frame_record(ring, newest)
                if record is not None and record["epoch"] == epoch:
                    start = record["slot"] * info["slotBytes"] + colour["offset"]
                    pixels = data.buf[start:start + colour["bytes"]]   # where the engine wrote it
                    frame = os.path.join(args.out, f"subscribed_{epoch:03d}.png")
                    write_png(frame, width, height, pixels)
                    del pixels
                    struct.pack_into("<Q", ring.buf, RING_RELEASED, newest)
                    print(f"epoch {epoch}: frame {newest}, {(time.time_ns() - pushed) / 1e6:.1f} ms after the "
                          f"push; wrote {frame}")
                    break
                struct.pack_into("<Q", ring.buf, RING_RELEASED, newest)
            else:
                print(f"epoch {epoch}: no frame showed it in 10 s")
        print(mcp.tool("subscription_status", {}))
        mcp.tool("render_unsubscribe", {})
    finally:
        ring.close()
        data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spz", nargs="+", help="SPZ files, one chunk each (or, with --sequence, a directory)")
    parser.add_argument("--mcp", default="athenea-mcp", help="the athenea-mcp binary")
    parser.add_argument("--stage", help="a stage naming the stream; written beside --out when not given")
    parser.add_argument("--stream", default="example-live", help="the stream's name")
    parser.add_argument("--chunk-splats", type=int, default=1 << 20, help="the most gaussians a file holds")
    parser.add_argument("--camera", default="/World/Shot", help="the camera the frames are drawn through")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--out", default=".", help="where the frames (and the stage) go")
    parser.add_argument("--subscribe", action="store_true",
                        help="read the frames from shared memory (render_subscribe) instead of rendering them")
    parser.add_argument("--sequence", action="store_true",
                        help="the files are a directory of pages f<frame>_c<chunk>.spz (athenea flatten --range): "
                             "epoch k is frame k's pages")
    parser.add_argument("--pose", help="with --subscribe: the camera to world, sixteen numbers row by row")
    parser.add_argument("--turn", type=float, default=20.0,
                        help="with --subscribe: degrees the camera turns about the up axis over the run")
    parser.add_argument("--up", default="y", choices=["y", "z"], help="the stage's up axis")
    parser.add_argument("--focal", type=float, default=30.0, help="the camera's focal length, mm")
    parser.add_argument("--aperture", type=float, default=24.576, help="its horizontal aperture, mm")
    args = parser.parse_args()

    if args.sequence:
        import glob
        import re
        found = {}
        for path in glob.glob(os.path.join(args.spz[0], "f*_c*.spz")):
            m = re.search(r"f(\d+)_c(\d+)\.spz$", path)
            found.setdefault(int(m.group(1)), {})[int(m.group(2))] = path
        frames = [found[f] for f in sorted(found)][:args.epochs]
        pages = [[open(chunks[c], "rb").read() for c in sorted(chunks)] for chunks in frames]
        ring_bytes = max(sum(map(len, p)) for p in pages)
        chunk_count = max(len(p) for p in pages)
    else:
        pages = [open(path, "rb").read() for path in args.spz]
        ring_bytes = sum(map(len, pages))
        chunk_count = len(pages)
    os.makedirs(args.out, exist_ok=True)
    stage = args.stage or os.path.join(args.out, "live.usda")
    if not args.stage:
        write_stage(stage, args.stream, chunk_count, args.chunk_splats)

    mcp = Mcp(args.mcp)
    mcp.tool("open_stage", {"stage": stage})
    opened = json.loads(mcp.tool("stream_open", {"name": args.stream, "chunks": chunk_count,
                                                 "ringBytes": max(ring_bytes, 1 << 16)}))
    ring = open_ring(opened["ring"])
    if args.subscribe:
        try:
            subscribe(mcp, args, pages, ring)
            mcp.tool("stream_close", {"name": args.stream})
        finally:
            ring.close()
            mcp.close()
        return
    try:
        for epoch in range(1, args.epochs + 1):
            listed, at = [], 0
            for chunk in range(len(pages)):
                page = pages[(chunk + epoch) % len(pages)]
                ring.buf[at:at + len(page)] = page
                listed.append({"chunk": chunk, "op": "replace", "offset": at, "bytes": len(page)})
                at += len(page)
            # The bytes are copied out before the answer: the ring is free again.
            print(mcp.tool("stream_commit", {"name": args.stream, "epoch": epoch, "pages": listed}))
            # A frame draws the last epoch whose pages are all on the device:
            # the first render hands this one to the loaders, the second,
            # once they have decompressed it, shows it.
            mcp.tool("render", {"camera": args.camera, "width": 640, "height": 360})
            time.sleep(0.2)
            frame = os.path.join(args.out, f"live_{epoch:03d}.png")
            mcp.tool("render", {"camera": args.camera, "width": 640, "height": 360, "output": frame})
            print("wrote", frame)
        mcp.tool("stream_close", {"name": args.stream})
    finally:
        ring.close()
        mcp.close()


if __name__ == "__main__":
    main()
