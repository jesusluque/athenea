// Copyright (c) 2026 jesus luque.
//
// A 3DGS PLY's header, read on the CPU: which property is where, and how
// each is encoded. Bookkeeping only -- names to indices, as
// io::readSplatPly does natively (modules/io/src/SplatReaders.cpp); the
// records themselves go to the GPU as the file holds them and webDecode
// (shaders/athenea/web/web_decode.slang) decodes them there.

const REQUIRED = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
  "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"];

const NO_FIELD = 0xffffffff;

/** Where "end_header\n" ends in `bytes`, or -1 if it has not arrived yet. */
export function headerEnd(bytes) {
  const tag = "end_header";
  const limit = Math.min(bytes.length, 1 << 16);
  outer: for (let i = 0; i + tag.length < limit; ++i) {
    for (let k = 0; k < tag.length; ++k) if (bytes[i + k] !== tag.charCodeAt(k)) continue outer;
    let at = i + tag.length;
    if (bytes[at] === 0x0d) at += 1;
    return bytes[at] === 0x0a ? at + 1 : -1;
  }
  return -1;
}

/**
 * The header, as what webDecode needs: `{count, stride (bytes), floats,
 * dataStart, decode: {DecodeParams field: value}, restPerColour, shWords}`.
 * Throws an Error that says why a file is not a cloud this module draws.
 */
export function parsePlyHeader(bytes) {
  const end = headerEnd(bytes);
  if (end < 0) throw new Error("PLY: no end_header in the first 64 KiB");
  const text = new TextDecoder("latin1").decode(bytes.subarray(0, end));
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines[0] !== "ply") throw new Error("not a PLY file");
  let format = "", vertexCount = -1, inVertex = false, beforeVertex = false;
  const props = [];
  for (const line of lines.slice(1)) {
    const w = line.split(/\s+/);
    if (w[0] === "format") format = w[1];
    else if (w[0] === "element") {
      inVertex = w[1] === "vertex";
      if (inVertex) vertexCount = Number(w[2]);
      else if (vertexCount < 0 && Number(w[2]) > 0) beforeVertex = true;
    } else if (w[0] === "property" && inVertex) {
      if (w[1] === "list") throw new Error("PLY: a list in the vertex element");
      props.push({ type: w[1], name: w[2] });
    }
  }
  if (format !== "binary_little_endian") {
    throw new Error(`PLY format '${format}'; splat clouds are binary_little_endian`);
  }
  if (beforeVertex) throw new Error("PLY: an element before the vertices");
  if (!(vertexCount > 0)) throw new Error("PLY: no vertices");
  if (!props.every((p) => p.type === "float" || p.type === "float32")) {
    // The native reader converts other types to float on the CPU; the web
    // module uploads the file's bytes as they are, so it takes float32 only.
    throw new Error("PLY: a property that is not float32 (this module reads float32 records only)");
  }
  const index = (name) => props.findIndex((p) => p.name === name);
  for (const name of REQUIRED) {
    if (index(name) < 0) throw new Error(`PLY: no '${name}' -- this does not look like a trained splat cloud`);
  }
  const restCount = props.filter((p) => p.name.startsWith("f_rest_")).length;
  let perColour = 0;
  for (const basis of [15, 8, 3]) if (restCount >= basis * 3) { perColour = basis; break; }
  if (perColour > 0 && restCount / 3 !== perColour) {
    throw new Error(`PLY: ${restCount} harmonic coefficients is not a whole degree`);
  }
  const restBase = perColour > 0 ? index("f_rest_0") : 0;
  for (let k = 0; k < perColour * 3; ++k) {
    if (index(`f_rest_${k}`) !== restBase + k) throw new Error("PLY: harmonics are not contiguous in the vertex element");
  }
  const floats = props.length;
  return {
    count: vertexCount,
    floats,
    stride: floats * 4,
    dataStart: end,
    restPerColour: perColour,
    // The engine's rows: SplatEncoding's defaults for a 3DGS PLY.
    decode: {
      stride: floats, keepPerColour: perColour,
      x: index("x"), y: index("y"), z: index("z"), opacity: index("opacity"),
      scale0: index("scale_0"), scale1: index("scale_1"), scale2: index("scale_2"),
      rotW: index("rot_0"), rotX: index("rot_1"), rotY: index("rot_2"), rotZ: index("rot_3"),
      dc0: index("f_dc_0"), dc1: index("f_dc_1"), dc2: index("f_dc_2"),
      restBase, filePerColour: perColour, restColourOuter: 1,
      opacityMode: 0, scaleMode: 0, colourMode: 0, rotationMode: 0, restMode: 0, flipYZ: 0,
      positionScale: 1.0,
      metallic: NO_FIELD, roughness: NO_FIELD, transmission: NO_FIELD, cryptoObject: NO_FIELD,
      transferBase: NO_FIELD, shadowBits: NO_FIELD, normal: NO_FIELD, emission: NO_FIELD, lobes: NO_FIELD,
    },
  };
}
