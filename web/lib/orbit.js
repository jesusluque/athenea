// Copyright (c) 2026 jesus luque.
//
// The orbit a page's canvas turns: left drag turns about the target, right
// drag or shift-drag pans, the wheel dollies. Camera bookkeeping, Y up.

/** A turntable about a target, Y up. */
export class Orbit {
  /** `onChange(camera)` is called with { position, target, up, fov } whenever it moves. */
  constructor(canvas, onChange) {
    this.canvas = canvas;
    this.onChange = onChange;
    this.target = [0, 0, 0];
    this.distance = 3;
    this.yaw = 0;
    this.pitch = 0;
    this.fov = 50;
    const down = (e) => {
      this.drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
      canvas.setPointerCapture?.(e.pointerId);
    };
    const move = (e) => {
      if (!this.drag) return;
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
      this.drag.x = e.clientX;
      this.drag.y = e.clientY;
      if (this.drag.pan) {
        const s = (2 * this.distance * Math.tan((this.fov * Math.PI) / 360)) / Math.max(canvas.clientHeight, 1);
        const [r, u] = this.axes();
        for (let k = 0; k < 3; ++k) this.target[k] += (-dx * r[k] + dy * u[k]) * s;
      } else {
        this.yaw -= dx * 0.005;
        this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * 0.005));
      }
      this.apply();
    };
    const up = () => { this.drag = null; };
    const wheel = (e) => {
      e.preventDefault();
      this.distance *= Math.exp(e.deltaY * 0.001);
      this.apply();
    };
    const menu = (e) => e.preventDefault();
    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("wheel", wheel, { passive: false });
    canvas.addEventListener("contextmenu", menu);
    this.dispose = () => {
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("wheel", wheel);
      canvas.removeEventListener("contextmenu", menu);
    };
  }

  eye() {
    const c = Math.cos(this.pitch);
    return [this.target[0] + this.distance * c * Math.sin(this.yaw), this.target[1] + this.distance * Math.sin(this.pitch),
      this.target[2] + this.distance * c * Math.cos(this.yaw)];
  }

  axes() {
    const e = this.eye();
    const f = [0, 1, 2].map((k) => this.target[k] - e[k]);
    const l = Math.hypot(...f);
    const fw = f.map((v) => v / l);
    const r = [-fw[2], 0, fw[0]];
    const rl = Math.hypot(...r) || 1;
    const right = r.map((v) => v / rl);
    const up = [right[1] * fw[2] - right[2] * fw[1], right[2] * fw[0] - right[0] * fw[2], right[0] * fw[1] - right[1] * fw[0]];
    return [right, up];
  }

  apply() {
    this.moving = performance.now();
    this.onChange({ position: this.eye(), target: this.target.slice(), up: [0, 1, 0], fov: this.fov });
  }

  set({ position, target, fov }) {
    if (fov) this.fov = fov;
    if (target) this.target = target.slice();
    if (position) {
      const d = [0, 1, 2].map((k) => position[k] - this.target[k]);
      this.distance = Math.hypot(...d) || 1;
      this.pitch = Math.asin(Math.max(-1, Math.min(1, d[1] / this.distance)));
      this.yaw = Math.atan2(d[0], d[2]);
    }
    this.apply();
  }

  frame({ min, max }) {
    const centre = [0, 1, 2].map((k) => (min[k] + max[k]) / 2);
    const radius = Math.hypot(...[0, 1, 2].map((k) => max[k] - min[k])) / 2 || 1;
    this.target = centre;
    this.distance = radius / Math.sin((this.fov * Math.PI) / 360) * 1.1;
    this.yaw = 0;
    this.pitch = 0.3;
    this.apply();
  }
}
