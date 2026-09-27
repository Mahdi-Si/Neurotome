// Pure helpers of the Input view (raw sample vs the model input it became).
import { bytes, dtypeShort, shape } from "./format";
import type { InputInfo, InputTime, RawInput, SignalRaw } from "./types";

/** `grace_hopper.jpg · JPEG · 512×600 px · 61.2 KB` */
export function rawCaption(info: InputInfo): string {
  const r = info.raw;
  if (isSignal(r)) {
    const n = Math.max(0, ...r.channels.map((c) => c.length));
    return [info.sample.name, `${r.channels.length} channel${r.channels.length === 1 ? "" : "s"}`, `${n} samples at ${r.fs} Hz`].join(" · ");
  }
  if (!r) return "";
  return [info.sample.name, r.format, `${r.width}×${r.height} px`, bytes(r.bytes)].filter(Boolean).join(" · ");
}

/** `1×3×224×224 · float32 · de-normalised for display` */
export function modelCaption(info: InputInfo): string {
  const how = !info.picture ? null : info.normalised ? "de-normalised for display" : "min–max stretched for display";
  return [shape(info.shape), dtypeShort(info.dtype), how].filter(Boolean).join(" · ");
}

/** Why there is no raw picture: the sample kind decides (text shows its text instead). */
export function rawNote(info: InputInfo): string {
  if (info.sample.kind === "random") return "No raw sample: the input is random";
  if (info.sample.kind === "ref") return "No raw signal: this dataset's load_sample returns none";
  if (info.sample.kind === "tensor") return info.steps ? "No raw picture: the uploaded tensor is the input"
    : "Created in-process (neurotome.open): preprocessing happened in your code";
  return "Not kept: this run was captured before 0.3.0 saved the raw input";
}

/** On-screen width of a W×H picture shown between `min` and `max` px tall (small ones scaled up, big ones down). */
export function pictureWidth(w: number, h: number, min = 224, max = 360): number {
  const tall = Math.min(max, Math.max(min, h));
  return Math.round((tall * w) / h);
}

/** (H, W) of an image-like model input (N, C, H, W), else null. */
export function pictureSize(info: InputInfo): [number, number] | null {
  return info.picture && info.shape.length === 4 ? [info.shape[2], info.shape[3]] : null;
}

// ---- dataset samples: a raw signal next to the (1, T, C) feature matrix it became ----

export const isSignal = (r: RawInput | null | undefined): r is SignalRaw => r?.kind === "signal";

/** A float (1, T, C) input: `/input/matrix` draws it as one time × column heatmap per block. */
export const hasMatrix = (info: InputInfo) =>
  info.shape.length === 3 && info.shape[0] === 1 && /float|half/.test(info.dtype ?? "");

type Span = [number, number];

/** Seconds on the display clock: minutes for a span of 10 min or more (`s to delivery` → `min to delivery`). */
export function clock(span: number, label: string): { div: number; label: string } {
  if (span < 600) return { div: 1, label };
  return { div: 60, label: /^s\b/.test(label) ? label.replace(/^s\b/, "min") : `${label} · min` };
}

/** Seconds the raw signal covers: sample i spans [t0 + i / fs, t0 + (i + 1) / fs). */
export const rawSpan = (r: SignalRaw): Span => [r.t0, r.t0 + Math.max(0, ...r.channels.map((c) => c.length)) / r.fs];

/** Seconds `cols` pooled columns of `stride` steps cover, each step a cell centred on t0 + i · step_s. */
export const stepSpan = (t: InputTime, cols: number, stride: number): Span =>
  [t.t0 - t.step_s / 2, t.t0 + (cols * stride - 0.5) * t.step_s];

export const union = (...spans: Span[]): Span => [Math.min(...spans.map((s) => s[0])), Math.max(...spans.map((s) => s[1]))];

/** Left / width in % of the part [a, b] of the domain [lo, hi], so charts on one clock line up. */
export function place([a, b]: Span, [lo, hi]: Span): { left: number; width: number } {
  const w = hi - lo || 1;
  return { left: ((a - lo) / w) * 100, width: ((b - a) / w) * 100 };
}

/** The row indices that get a label: every k-th so that at most `max` show. */
export function sparseRows(n: number, max = 8): number[] {
  const k = Math.max(1, Math.ceil(n / max));
  return Array.from({ length: Math.ceil(n / k) }, (_, i) => i * k);
}

/** [lo, hi] of the finite values of a min / max envelope, padded 5% (a flat signal gets ±1). */
export function envelopeDomain(lo: (number | null)[], hi: (number | null)[]): Span {
  const v = [...lo, ...hi].filter((x): x is number => x != null && Number.isFinite(x));
  if (!v.length) return [0, 1];
  let a = Math.min(...v), b = Math.max(...v);
  if (a === b) { a -= 1; b += 1; }
  const pad = (b - a) * 0.05;
  return [a - pad, b + pad];
}

/** SVG path of a line through the points i → (x(i), y(v)), a new subpath after each gap (null). */
export function linePath(v: (number | null)[], x: (i: number) => number, y: (v: number) => number): string {
  let d = "", pen = false;
  v.forEach((p, i) => {
    if (p == null) { pen = false; return; }
    d += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(p).toFixed(1)}`;
    pen = true;
  });
  return d;
}

/** SVG path of the band between lo and hi, one closed polygon per run without gaps. */
export function bandPath(lo: (number | null)[], hi: (number | null)[], x: (i: number) => number, y: (v: number) => number): string {
  let d = "", run: number[] = [];
  const close = () => {
    if (run.length) {
      const top = run.map((i, k) => `${k ? "L" : "M"}${x(i).toFixed(1)} ${y(hi[i]!).toFixed(1)}`).join("");
      const bottom = [...run].reverse().map((i) => `L${x(i).toFixed(1)} ${y(lo[i]!).toFixed(1)}`).join("");
      d += `${top}${bottom}Z`;
    }
    run = [];
  };
  lo.forEach((a, i) => (a == null || hi[i] == null ? close() : run.push(i)));
  close();
  return d;
}

/** Axis end texts padded to one length with figure spaces, so stacked charts get the same y gutter and line up. */
export const padEnd = (s: string, n: number) => s.padStart(n, " ");

/** Steps [a, b) of the input's time axis on its chart clock: seconds when `time` is known (step i is the cell centred
 * on t0 + i · step_s), else steps (cell i spans i ± ½). */
export const stepRange = (t: InputTime | null, a: number, b: number): Span =>
  t ? [t.t0 + (a - 0.5) * t.step_s, t.t0 + (b - 0.5) * t.step_s] : [a - 0.5, b - 0.5];
