// Copyright (c) 2026 jesus luque.
//
// The camera's and the transform's bookkeeping: a handful of 4x4 affine
// matrices a frame, row-major [16]. Nothing of the cloud's data goes through
// here.

export const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export function mul(a, b) {
  const o = new Array(16).fill(0);
  for (let r = 0; r < 4; ++r) for (let c = 0; c < 4; ++c) for (let k = 0; k < 4; ++k) o[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
  return o;
}

export function inverseAffine(m) {
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const s = 1 / (a * A + b * B + c * C);
  const r = [
    A * s, -(b * i - c * h) * s, (b * f - c * e) * s,
    B * s, (a * i - c * g) * s, -(a * f - c * d) * s,
    C * s, -(a * h - b * g) * s, (a * e - b * d) * s,
  ];
  const t = [m[3], m[7], m[11]];
  return [
    r[0], r[1], r[2], -(r[0] * t[0] + r[1] * t[1] + r[2] * t[2]),
    r[3], r[4], r[5], -(r[3] * t[0] + r[4] * t[1] + r[5] * t[2]),
    r[6], r[7], r[8], -(r[6] * t[0] + r[7] * t[1] + r[8] * t[2]),
    0, 0, 0, 1,
  ];
}

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const normalize = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
export const apply = (m, p) => [0, 1, 2].map((r) => m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3]);

/** The largest of a matrix's three axis lengths: its scale, for a uniform one. */
export const scaleOf = (m) => Math.max(...[0, 1, 2].map((c) => Math.hypot(m[c], m[4 + c], m[8 + c])));

/**
 * A page's transform -- { position, rotation (degrees, three.js's XYZ Euler
 * order), scale } -- as an object-to-world matrix, or 16 numbers, row-major.
 */
export function transformMatrix(t) {
  if (!t) return identity();
  if (Array.isArray(t) && t.length === 16) return t.slice();
  const [x, y, z] = (t.rotation || [0, 0, 0]).map((v) => (v * Math.PI) / 180);
  const rx = [1, 0, 0, 0, 0, Math.cos(x), -Math.sin(x), 0, 0, Math.sin(x), Math.cos(x), 0, 0, 0, 0, 1];
  const ry = [Math.cos(y), 0, Math.sin(y), 0, 0, 1, 0, 0, -Math.sin(y), 0, Math.cos(y), 0, 0, 0, 0, 1];
  const rz = [Math.cos(z), -Math.sin(z), 0, 0, Math.sin(z), Math.cos(z), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const s = t.scale ?? 1;
  const p = t.position || [0, 0, 0];
  const m = mul(mul(rx, ry), rz);
  return [m[0] * s, m[1] * s, m[2] * s, p[0], m[4] * s, m[5] * s, m[6] * s, p[1], m[8] * s, m[9] * s, m[10] * s, p[2], 0, 0, 0, 1];
}

/** The view frame.slang draws in: +x right, +y up, +z forward, the -Z camera folded. */
export function worldToView({ position: e, target, up = [0, 1, 0] }) {
  const f = normalize(sub(target, e));
  const r = normalize(cross(f, up));
  const u = cross(r, f);
  return [...r, -dot(r, e), ...u, -dot(u, e), ...f, -dot(f, e), 0, 0, 0, 1];
}

/** A page's colour, "#rrggbb" or [r, g, b(, a)] in sRGB 0-1, kept as sRGB. */
export function parseColour(c) {
  if (Array.isArray(c)) return [c[0], c[1], c[2], c[3] ?? 1];
  const m = /^#?([0-9a-f]{6})$/i.exec(String(c));
  if (!m) return [0, 0, 0, 1];
  const v = parseInt(m[1], 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255, 1];
}

/** The sRGB transfer, inverted: for one colour of a uniform. */
export const srgbToLinear = (v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
