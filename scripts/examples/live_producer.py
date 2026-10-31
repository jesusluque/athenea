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
consecutive frames differ. The pages are SPZ files as they are on disk; the
engine decompresses them on its loader threads. A PLY becomes an SPZ with
`athenea flatten` or Niantic's own tools.
"""
import argparse
import json
import os
import subprocess
import sys
import time
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spz", nargs="+", help="SPZ files, one chunk each")
    parser.add_argument("--mcp", default="athenea-mcp", help="the athenea-mcp binary")
    parser.add_argument("--stage", help="a stage naming the stream; written beside --out when not given")
    parser.add_argument("--stream", default="example-live", help="the stream's name")
    parser.add_argument("--chunk-splats", type=int, default=1 << 20, help="the most gaussians a file holds")
    parser.add_argument("--camera", default="/World/Shot", help="the camera the frames are drawn through")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--out", default=".", help="where the frames (and the stage) go")
    args = parser.parse_args()

    pages = [open(path, "rb").read() for path in args.spz]
    os.makedirs(args.out, exist_ok=True)
    stage = args.stage or os.path.join(args.out, "live.usda")
    if not args.stage:
        write_stage(stage, args.stream, len(pages), args.chunk_splats)

    mcp = Mcp(args.mcp)
    mcp.tool("open_stage", {"stage": stage})
    opened = json.loads(mcp.tool("stream_open", {"name": args.stream, "chunks": len(pages),
                                                 "ringBytes": max(sum(map(len, pages)), 1 << 16)}))
    ring = open_ring(opened["ring"])
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
