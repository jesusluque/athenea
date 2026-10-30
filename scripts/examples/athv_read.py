#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""A receiver's side of athv_stream.py, and a .athv's codec facts: the Python standard library only.

Reading a stream (stdin, as athv_stream.py, or a transport after it, writes
it): parses the length-prefixed messages for --seconds, checks each one's
magic and length, and prints, per rendition, records a second and Mbit/s;
for the frames (the first record of each new presentation frame) the gaps
between arrivals and between the sender's PTP stamps (median and the largest
departure from the rate), and frames missing from the count. A message whose
magic is neither ATVF nor ATVM is counted and the stream resynchronised on
the next "ATVF".

    python3 scripts/examples/athv_stream.py a.athv b.athv | python3 scripts/examples/athv_read.py --seconds 10

Describing a file (--info): its header, the plane layout it implies, the
records' sizes, the first record, and what a WebCodecs VideoDecoder is
configured with -- the codec string and the hvcC (HEVCDecoderConfigurationRecord)
built from that record's own VPS, SPS and PPS (--hvcc writes it to a file).

    python3 scripts/examples/athv_read.py --info v.athv --hvcc v.hvcC

The messages are docs/operations.md §8.2 ("Video sent as a stream"); the file
and its records are modules/video/include/athenea/video/Athv.h.
"""
import argparse
import base64
import json
import statistics
import struct
import sys
import time

HEADER_BYTES = 128
# Every record's fixed prefix (Athv.h, AthvRecordPrefix).
MAGIC_AT, TYPE_AT, RENDITION_AT, FRAME_AT, PTP_AT, BYTES_AT, VERSION_AT, FLAGS_AT = 0, 4, 6, 8, 12, 20, 24, 26
PICTURE, MANIFEST, ANCILLARY = 0, 1, 2


# --- the file --------------------------------------------------------------

def parse_header(b):
    if len(b) < HEADER_BYTES or b[0:4] != b"ATHV":
        raise SystemExit("athv_read: not a .athv header")
    (version, codec, bits, rest, chunk, columns, most, flags, rendition) = struct.unpack_from("<9I", b, 4)
    if version != 2:
        raise SystemExit("athv_read: a version %d .athv; this reads version 2 (athenea video-encode writes it)"
                         % version)
    rate, = struct.unpack_from("<d", b, 40)
    num, den, frames = struct.unpack_from("<3I", b, 48)
    quality, = struct.unpack_from("<f", b, 60)
    block, high_bits = struct.unpack_from("<2I", b, 64)
    target, = struct.unpack_from("<f", b, 72)
    geometry_interval, = struct.unpack_from("<I", b, 76)
    geometry_bits, = struct.unpack_from("<I", b, 80)
    return dict(version=version, codec=struct.pack("<I", codec).decode("latin-1"), bits=bits, restPerColour=rest,
                chunkSplats=chunk, columns=columns, maxSplats=most, hiInVideo=bool(flags & 1),
                antialiased=bool(flags & 2), chroma=bool(flags & 4), stableOrder=bool(flags & 8),
                rendition=rendition, rate=rate,
                rateNumerator=num, rateDenominator=den,
                sourceFrames=frames, quality=round(quality, 4), positionBlock=block, positionHighBits=high_bits,
                positionTarget=round(target, 6), geometry=bool(flags & 16), geometryStagger=bool(flags & 32),
                relightable=bool(flags & 64),
                geometryInterval=geometry_interval, geometryPositionBits=(geometry_bits & 255) or 16,
                geometryScaleBits=((geometry_bits >> 8) & 255) or 10, geometryOrder=((geometry_bits >> 16) & 255) or 1,
                geometryRotationBits=(geometry_bits >> 24) or 10)


def layout(h):
    """The mosaic the header implies (Planes.cpp, PlaneLayout::make)."""
    # Relightable (flag bit 6): one occlusion attribute and plane after the colours.
    extra = 1 if h.get("relightable") else 0
    attributes = 14 + 3 * h["restPerColour"] + extra
    first = 6 if h["hiInVideo"] else 3
    # With colour in chroma: 8 opacity / scale / rotation planes and a Y plane a colour quantity.
    planes = first + (8 + 1 + h["restPerColour"] + extra if h["chroma"] else attributes - 3)
    if h.get("geometry"):
        # The geometry layer: the opacity and the colours only.
        planes = 1 + (1 + h["restPerColour"] if h["chroma"] else 3 + 3 * h["restPerColour"]) + extra
    chunks = (h["maxSplats"] + h["chunkSplats"] - 1) // h["chunkSplats"]
    tile_height = chunks * (h["chunkSplats"] // 512)
    rows = (planes + h["columns"] - 1) // h["columns"]
    return dict(attributes=attributes, planes=planes, tileHeight=tile_height, width=h["columns"] * 512,
                height=rows * tile_height)


def walk(data):
    """(offset, bytes) of every record of a file."""
    out = []
    at = HEADER_BYTES
    while at < len(data):
        if data[at:at + 4] != b"ATVF":
            raise SystemExit("athv_read: a broken record at byte %d" % at)
        size, = struct.unpack_from("<I", data, at + BYTES_AT)
        if size < 104 or size > len(data) - at:
            raise SystemExit("athv_read: a record of %d bytes at byte %d" % (size, at))
        out.append((at, size))
        at += size
    return out


def parse_record(b, at=0):
    if b[at:at + 4] != b"ATVF":
        raise SystemExit("athv_read: not an ATVF record")
    _type, rendition, frame, ptp, size, version, flags = struct.unpack_from("<HHIQIHH", b, at + 4)
    (index, source, slice_, slices, first, splats, source_splats) = struct.unpack_from("<7I", b, at + 28)
    t, = struct.unpack_from("<d", b, at + 56)
    label = b[at + 64:at + 80].split(b"\0")[0].decode("ascii", "replace")
    (n_ranges, hi_raw, hi_stored, n_sets, unit_bytes, geometry_stored) = struct.unpack_from("<6I", b, at + 80)
    p = at + 104
    ranges = struct.unpack_from("<%df" % n_ranges, b, p)
    p += 4 * n_ranges
    high = bytes(b[p:p + hi_stored])
    p += hi_stored
    p = (p - at + 3) // 4 * 4 + at
    geometry = bytes(b[p:p + geometry_stored])
    p += geometry_stored
    p = (p - at + 3) // 4 * 4 + at
    sets = []
    for _ in range(n_sets):
        n, = struct.unpack_from("<I", b, p)
        sets.append(bytes(b[p + 4:p + 4 + n]))
        p += 4 + n
        p = (p - at + 3) // 4 * 4 + at
    unit = bytes(b[p:p + unit_bytes])
    return dict(size=size, rendition=rendition, frame=frame, ptp=ptp, version=version, key=bool(flags & 1),
                geometryKey=bool(flags & 2), geometry=geometry, geometryStored=geometry_stored,
                index=index, source=source, slice=slice_, slices=slices, firstSplat=first, splats=splats,
                sourceSplats=source_splats, time=t, timecode=label, ranges=ranges, highRaw=hi_raw,
                highStored=hi_stored, high=high, sets=sets, unit=unit)


# --- HEVC parameter sets ---------------------------------------------------

class Bits:
    def __init__(self, rbsp):
        self.b = rbsp
        self.at = 0

    def u(self, n):
        v = 0
        for _ in range(n):
            byte = self.b[self.at >> 3]
            v = (v << 1) | ((byte >> (7 - (self.at & 7))) & 1)
            self.at += 1
        return v

    def ue(self):
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
        return (1 << zeros) - 1 + self.u(zeros)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


def pps_facts(pps):
    """The PPS's initial QP and its chroma QP offsets."""
    r = Bits(rbsp_of(pps))
    r.ue()
    r.ue()
    r.u(1)
    r.u(1)
    r.u(3)
    r.u(1)
    r.u(1)
    r.ue()
    r.ue()
    init_qp = 26 + r.se()
    r.u(1)
    r.u(1)
    if r.u(1):
        r.ue()
    return dict(initQp=init_qp, cbQpOffset=r.se(), crQpOffset=r.se())


def rbsp_of(nal):
    """A NAL unit's payload without its emulation prevention bytes."""
    out = bytearray()
    zeros = 0
    for byte in nal[2:]:
        if zeros >= 2 and byte == 3:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def sps_facts(sps):
    rbsp = rbsp_of(sps)
    r = Bits(rbsp)
    r.u(4)
    sub_layers = r.u(3)
    r.u(1)
    ptl = bytes(rbsp[1:13])   # profile space .. general_level_idc: 12 bytes
    r.u(8 * 12)
    if sub_layers != 0:
        raise SystemExit("athv_read: an SPS with sub-layers is not described here")
    r.ue()                      # sps_seq_parameter_set_id
    chroma = r.ue()
    if chroma == 3:
        r.u(1)
    width, height = r.ue(), r.ue()
    if r.u(1):
        for _ in range(4):
            r.ue()
    depth_luma, depth_chroma = r.ue() + 8, r.ue() + 8
    return dict(ptl=ptl, profile=ptl[0] & 31, tier=(ptl[0] >> 5) & 1, level=ptl[11], chroma=chroma, width=width,
                height=height, depthLuma=depth_luma, depthChroma=depth_chroma)


def codec_string(ptl, sample_entry="hvc1"):
    """ISO/IEC 14496-15 Annex E: hvc1.<profile>.<compatibility, reversed>.<tier><level>.<constraints>."""
    space = ptl[0] >> 6
    tier = (ptl[0] >> 5) & 1
    profile = ptl[0] & 31
    compat, = struct.unpack(">I", ptl[1:5])
    reversed_compat = int("{:032b}".format(compat)[::-1], 2)
    constraints = list(ptl[5:11])
    while constraints and constraints[-1] == 0:
        constraints.pop()
    level = ptl[11]
    s = "%s.%s%d.%X.%s%d" % (sample_entry, "" if space == 0 else "ABC"[space - 1], profile, reversed_compat,
                            "H" if tier else "L", level)
    if constraints:
        s += "." + ".".join("%X" % c for c in constraints)
    return s


def hvcc(vps, sps, pps):
    """The HEVCDecoderConfigurationRecord a WebCodecs `description` takes (lengthSizeMinusOne 3)."""
    f = sps_facts(sps)
    ptl = f["ptl"]
    out = bytearray([1])
    out += ptl[0:12]                                 # profile space/tier/idc, compatibility, constraints, level
    out += struct.pack(">H", 0xF000)                 # min_spatial_segmentation_idc 0
    out += bytes([0xFC])                             # parallelismType 0
    out += bytes([0xFC | f["chroma"]])
    out += bytes([0xF8 | (f["depthLuma"] - 8)])
    out += bytes([0xF8 | (f["depthChroma"] - 8)])
    out += struct.pack(">H", 0)                      # avgFrameRate unknown
    out += bytes([(0 << 6) | (1 << 3) | (1 << 2) | 3])   # constant rate 0, one temporal layer, nested, 4-byte lengths
    arrays = [(32, vps), (33, sps), (34, pps)]
    out += bytes([len(arrays)])
    for nal_type, nal in arrays:
        out += bytes([0x80 | nal_type]) + struct.pack(">H", 1) + struct.pack(">H", len(nal)) + nal
    return bytes(out), f


def describe(data):
    """What a manifest says of a file: its layout, its pictures, its codec."""
    h = parse_header(data)
    lay = layout(h)
    records = walk(data)
    first = parse_record(data, records[0][0])
    frames = h["sourceFrames"] or len({parse_record(data, at)["source"] for at, _ in records})
    total = sum(size for _, size in records)
    per_frame = [struct.unpack_from("<I", data, at + 52)[0] for at, _ in records]
    gaussians = statistics.median(per_frame)
    degree = {0: 0, 3: 1, 8: 2, 15: 3}[h["restPerColour"]]
    out = dict(id=h["rendition"], gaussians=int(gaussians), maxGaussians=max(per_frame), shDegree=degree,
               quality=h["quality"], bits=h["bits"], restPerColour=h["restPerColour"], chunkSplats=h["chunkSplats"],
               columns=h["columns"], maxSplats=h["maxSplats"], hiInVideo=h["hiInVideo"],
               antialiased=h["antialiased"], positionBlock=h["positionBlock"],
               positionHighBits=h["positionHighBits"], chromaPlanes=h["chroma"], stableOrder=h["stableOrder"],
               relightable=h["relightable"], geometryLossless=h["geometry"], geometryKeyInterval=h["geometryInterval"],
               geometryStagger=h["geometryStagger"], geometryPositionBits=h["geometryPositionBits"],
               geometryScaleBits=h["geometryScaleBits"], geometryOrder=h["geometryOrder"],
               geometryRotationBits=h["geometryRotationBits"], picturesPerFrame=first["slices"],
               pictureWidth=lay["width"], pictureHeight=lay["height"], meanBytesPerFrame=round(total / frames),
               meanMbps=round(total / frames * 8 * h["rate"] / 1e6, 2), fullRange=True)
    if len(first["sets"]) == 3:
        record, f = hvcc(*first["sets"])
        out.update(codec=codec_string(f["ptl"], "hvc1"), codecAnnexB=codec_string(f["ptl"], "hev1"),
                   profile=f["profile"], tier=f["tier"], level=f["level"], hvcC=base64.b64encode(record).decode())
    return h, lay, records, first, out


def info(path, hvcc_path):
    with open(path, "rb") as fh:
        data = fh.read()
    h, lay, records, r, m = describe(data)
    print("header:", h)
    print("layout:", lay)
    sizes = [s for _, s in records]
    keys = sum(1 for at, _ in records if struct.unpack_from("<H", data, at + FLAGS_AT)[0] & 1)
    print("records: %d, %d key; bytes a record median %d, largest %d; %.2f MB in all; %d pictures a frame"
          % (len(records), keys, statistics.median(sizes), max(sizes), len(data) / 1e6, r["slices"]))
    print("a frame: %.4f MB, %.2f Mbit/s at %.3g frames/s (%.2f at 24)"
          % (m["meanBytesPerFrame"] / 1e6, m["meanMbps"], h["rate"], m["meanBytesPerFrame"] * 8 * 24 / 1e6))
    print("first record: rendition %d frame %d index %d source %d slice %d of %d, splats %d from %d (frame of %d) "
          "key %s time %.3f timecode %s; ranges %d floats; high parts %d bytes deflated to %d; %d parameter sets; "
          "access unit %d bytes"
          % (r["rendition"], r["frame"], r["index"], r["source"], r["slice"], r["slices"], r["splats"],
             r["firstSplat"], r["sourceSplats"], r["key"], r["time"], r["timecode"], len(r["ranges"]), r["highRaw"],
             r["highStored"], len(r["sets"]), len(r["unit"])))
    nals = []
    p = 0
    while p + 4 <= len(r["unit"]):
        n, = struct.unpack_from(">I", r["unit"], p)
        nals.append((r["unit"][p + 4] >> 1) & 63)
        p += 4 + n
    print("access unit NAL types:", nals)
    if len(r["sets"]) == 3:
        record, f = hvcc(*r["sets"])
        print("SPS: %dx%d, profile_idc %d, tier %d, level_idc %d, chroma_format_idc %d, %d-bit luma, %d-bit chroma"
              % (f["width"], f["height"], f["profile"], f["tier"], f["level"], f["chroma"], f["depthLuma"],
                 f["depthChroma"]))
        q = pps_facts(r["sets"][2])
        print("PPS: init_qp %d, cb_qp_offset %d, cr_qp_offset %d" % (q["initQp"], q["cbQpOffset"], q["crQpOffset"]))
        print("codec (hvcC description, length-prefixed units):", codec_string(f["ptl"], "hvc1"))
        print("codec (Annex B, parameter sets in band):", codec_string(f["ptl"], "hev1"))
        print("hvcC (%d bytes): %s" % (len(record), record.hex()))
        if hvcc_path:
            with open(hvcc_path, "wb") as out:
                out.write(record)


# --- a stream --------------------------------------------------------------

def read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def resync(stream, prefix):
    """Scans for the next "ATVF"; returns that record's length bytes and its first four, or None."""
    window = bytearray(prefix)
    while True:
        b = stream.read(1)
        if not b:
            return None
        window += b
        if len(window) >= 8 and window[-4:] == b"ATVF":
            return bytes(window[-8:-4]), b"ATVF"


def receive(seconds, rate_hint):
    stream = sys.stdin.buffer
    start = None
    per = {}           # rendition -> [records, bytes]
    frame_arrivals = []
    frame_stamps = []
    frames_seen = []
    manifests = 0
    ancillaries = 0
    broken = 0
    rate = rate_hint
    last_frame = None
    while True:
        prefix = read_exact(stream, 4)
        if prefix is None:
            break
        n, = struct.unpack("<I", prefix)
        head = b""
        if n < 28 or n > (1 << 30):
            broken += 1
            found = resync(stream, prefix)
            if found is None:
                break
            prefix, head = found
            n, = struct.unpack("<I", prefix)
        body = head + (read_exact(stream, n - len(head)) or b"")
        if len(body) < 28:
            break
        now = time.monotonic()
        if start is None:
            start = now
        if now - start > seconds:
            break
        magic = body[0:4]
        kind, rendition, frame, ptp, size = struct.unpack_from("<HHIQI", body, 4)
        if magic == b"ATVM" and kind == MANIFEST:
            manifests += 1
            jn, = struct.unpack_from("<I", body, 28)
            m = json.loads(body[32:32 + jn])
            if not rate:
                rate = m["fps"]["num"] / m["fps"]["den"]
            continue
        if magic == b"ATVA" and kind == ANCILLARY:
            ancillaries += 1
            jn, = struct.unpack_from("<I", body, 28)
            if ancillaries == 1:
                print("athv_read: ancillary data from frame %d: %s" % (frame, body[32:32 + jn].decode()))
            continue
        if magic != b"ATVF" or kind != PICTURE or size != len(body):
            broken += 1
            continue
        c = per.setdefault(rendition, [0, 0])
        c[0] += 1
        c[1] += 4 + len(body)
        if frame != last_frame:
            frame_arrivals.append(now)
            frame_stamps.append(ptp / 1e6)
            frames_seen.append(frame)
            last_frame = frame
    if len(frame_arrivals) < 3:
        print("athv_read: %d frames; too few to measure" % len(frame_arrivals))
        return
    span = frame_arrivals[-1] - frame_arrivals[0]
    gaps = [(b - a) * 1000.0 for a, b in zip(frame_arrivals, frame_arrivals[1:])]
    sent = [b - a for a, b in zip(frame_stamps, frame_stamps[1:])]
    period = 1000.0 / rate if rate else statistics.median(gaps)
    missing = (frames_seen[-1] - frames_seen[0] + 1) - len(frames_seen)
    total = sum(c[1] for c in per.values())
    print("athv_read: %d frames in %.2f s: %.2f frames/s, %.1f Mbit/s in all; %d manifests, %d ancillary, "
          "%d broken, %d frames missing" % (len(frames_seen), span, (len(frames_seen) - 1) / span,
                                            total * 8 / span / 1e6, manifests, ancillaries, broken, missing))
    for rid in sorted(per):
        c = per[rid]
        print("athv_read: rendition %d: %d records, %.2f records/s, %.2f Mbit/s"
              % (rid, c[0], c[0] / span, c[1] * 8 / span / 1e6))
    print("athv_read: frame arrivals: gaps median %.2f ms, shortest %.2f, longest %.2f; jitter (largest "
          "|gap - %.2f ms|) %.2f ms, std %.2f ms" % (statistics.median(gaps), min(gaps), max(gaps), period,
                                                     max(abs(g - period) for g in gaps), statistics.pstdev(gaps)))
    print("athv_read: sender's PTP stamps: gaps median %.2f ms, jitter %.2f ms"
          % (statistics.median(sent), max(abs(g - period) for g in sent)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--info", metavar="ATHV", help="describe a file instead of reading a stream")
    ap.add_argument("--hvcc", metavar="OUT", help="with --info: write the hvcC record here")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long a stream is read")
    ap.add_argument("--rate", type=float, default=0.0, help="the expected rate (default: the manifest's)")
    a = ap.parse_args()
    if a.info:
        info(a.info, a.hvcc)
    else:
        receive(a.seconds, a.rate)


if __name__ == "__main__":
    main()
