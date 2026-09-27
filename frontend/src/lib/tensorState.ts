// Pure state and geometry for the tensor inspector (URL params, pooled-grid ↔ pixel, slices, compare).
import type { Cmap, GraphView, Norm, SliceResult, SliceValue, Stats } from "./types";

export interface TensorState { node: string | null; c: number | null; py: number | null; px: number | null; norm: Norm; cmap: Cmap; cmp: string | null;
  iv: string | null; vs: "clean" | null;   // vs = clean vs intervened, only with iv
  s?: string | null; t?: number | null }   // s: the Values panel's slice; t: an output step whose input windows the Input view draws

const int = (s: string | null) => (s != null && /^\d+$/.test(s.trim()) ? Number(s) : null);

export function parseTensorParams(params: URLSearchParams): TensorState {
  const py = int(params.get("py")), px = int(params.get("px"));
  const pair = py != null && px != null, iv = params.get("iv") || null;
  return {
    node: params.get("node") || null,
    c: int(params.get("c")),
    py: pair ? py : null,
    px: pair ? px : null,
    norm: params.get("norm") === "global" ? "global" : "channel",
    cmap: params.get("cmap") === "div" ? "div" : "seq",
    cmp: params.get("cmp") || null,
    iv, vs: iv && params.get("vs") === "clean" ? "clean" : null,
    s: params.get("s") || null, t: int(params.get("t")),
  };
}

export function tensorParams(s: TensorState): Record<string, string | null> {
  const pair = s.py != null && s.px != null;
  return {
    node: s.node, c: s.c == null ? null : String(s.c),
    py: pair ? String(s.py) : null, px: pair ? String(s.px) : null,
    norm: s.norm === "global" ? "global" : null, cmap: s.cmap === "div" ? "div" : null, cmp: s.cmp,
    iv: s.iv, vs: s.iv && s.vs ? s.vs : null,
    s: s.s ?? null, t: s.t == null ? null : String(s.t),
  };
}

// Pooled grids split `len` pixels into `n` cells like np.array_split: the first len % n cells are one wider.
export const cellStart = (len: number, n: number, i: number) => i * Math.floor(len / n) + Math.min(i, len % n);

export function cellOf(len: number, n: number, p: number): number {
  const q = Math.floor(len / n), r = len % n, x = Math.max(0, Math.min(len - 1, p));
  return x < (q + 1) * r ? Math.floor(x / (q + 1)) : r + Math.floor((x - (q + 1) * r) / q);
}

export const cellPixel = (len: number, n: number, i: number) =>
  Math.floor((cellStart(len, n, i) + cellStart(len, n, i + 1) - 1) / 2);

/** "4–7" for a pooled cell covering pixels 4..7, "4" for a single pixel. */
export function cellSpan(len: number, n: number, i: number): string {
  const a = cellStart(len, n, i), b = cellStart(len, n, i + 1) - 1;
  return a === b ? `${a}` : `${a}–${b}`;
}

/** A normalised chart value back to the tensor's units, given the range the server scaled with. */
export const unscale = (v: number, [lo, hi]: [number, number], cmap: Cmap) =>
  cmap === "div" ? v * Math.max(Math.abs(lo), Math.abs(hi)) : lo + v * (hi - lo);

/** Row of the source behind PCA point i of k sampled from n rows: mirrors np.linspace(0, n-1, k).astype(int). */
export const sampleIndex = (n: number, k: number, i: number) =>
  k <= 1 ? 0 : i === k - 1 ? n - 1 : Math.floor(i * ((n - 1) / (k - 1)));

/** Screen mapping for a w×h scatter with one scale on both axes (PCA distances stay true), centred, `pad` px inset. */
export function equalAspect(points: [number, number][], w: number, h: number, pad: number) {
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const k = Math.min((w - 2 * pad) / Math.max(x1 - x0, 1e-9), (h - 2 * pad) / Math.max(y1 - y0, 1e-9));
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const hw = w / 2 / k, hh = h / 2 / k;
  return {
    px: (v: number) => w / 2 + (v - cx) * k, py: (v: number) => h / 2 - (v - cy) * k,
    // the data values at the chart's edges (y from the top), for the axis ranges
    xRange: [cx - hw, cx + hw] as [number, number], yRange: [cy + hh, cy - hh] as [number, number],
  };
}

/** (H, W) of an image-like input shape (the last two axes of a rank ≥ 3 shape), else null. */
export const inputHW = (shape: number[] | null | undefined): [number, number] | null =>
  shape && shape.length >= 3 ? [shape[shape.length - 2], shape[shape.length - 1]] : null;

/** A line plot's y domain: always includes 0 and `domain`, never empty. */
export function lineDomain(values: number[], domain?: [number, number]): [number, number] {
  const lo = Math.min(0, domain?.[0] ?? 0, ...values);
  return [lo, Math.max(...values, domain?.[1] ?? -Infinity, lo + 1e-6)];
}

/** What the sequence heatmap was divided by: the shared scale's max(|lo|, |hi|), or the tensor's max |x| when the
 * map is not pooled (then max |pooled| = max |x|); null when pooled (block means, divisor not in the payload). */
export function seqFactor(t: { tokens: number; features: number; heatmap: number[][]; scale: [number, number] | null;
  stats: { min: number | null; max: number | null } }): number | null {
  if (t.scale) return Math.max(Math.abs(t.scale[0]), Math.abs(t.scale[1]));
  const exact = t.heatmap.length === t.tokens && (t.heatmap[0]?.length ?? 0) === t.features;
  return exact && t.stats.min != null && t.stats.max != null ? Math.max(Math.abs(t.stats.min), Math.abs(t.stats.max)) : null;
}

const win = (dim: number, size: number, centre = 0) => {
  const a = Math.max(0, Math.min(centre - Math.floor(size / 2), dim - size));
  return `${a}:${Math.min(dim, a + size)}`;
};

export function defaultSlice(shape: number[], o: { channel?: number | null; pixel?: [number, number] | null }): string {
  const n = shape.length;
  if (n === 0) return "[]";
  if (n === 1) return `[${win(shape[0], 64)}]`;
  if (n === 4 && shape[0] === 1 && o.channel != null && o.pixel)
    return `[0, ${o.channel}, ${win(shape[2], 16, o.pixel[0])}, ${win(shape[3], 16, o.pixel[1])}]`;
  const lead = shape.slice(0, -2).map(() => "0");
  const row = shape[n - 2] === 1 ? "0" : win(shape[n - 2], 16);
  return `[${[...lead, row, win(shape[n - 1], n === 2 ? 64 : 16)].join(", ")}]`;
}

const prod = (xs: number[]) => xs.reduce((a, b) => a * b, 1);

export const pageCount = (r: SliceResult) => prod(r.shape.slice(0, -2));

export function slicePage(r: SliceResult, page: number): { label: string; rows: number[]; cols: number[]; cells: SliceValue[][] } | null {
  const n = r.shape.length;
  if (n < 2 || r.values.length === 0) return null;
  const nr = r.shape[n - 2], nc = r.shape[n - 1], lead = r.shape.slice(0, -2);
  const at: number[] = [];
  for (let k = lead.length - 1, p = page; k >= 0; k--) { at[k] = r.indices[k][p % lead[k]]; p = Math.floor(p / lead[k]); }
  const off = page * nr * nc;
  return {
    label: `[${at.join(", ")}]`,
    rows: r.indices[n - 2], cols: r.indices[n - 1],
    cells: Array.from({ length: nr }, (_, i) => r.values.slice(off + i * nc, off + (i + 1) * nc)),
  };
}

/** Server error text without the "400 · " status prefix. */
export const errText = (e: unknown) => (e instanceof Error ? e.message : String(e)).replace(/^\d{3} · /, "");

export const parentNode = (view: GraphView, id: string) => view.edges.find((e) => e.target === id)?.source ?? null;

export const sameShape = (a: number[], b: number[]) => a.length === b.length && a.every((d, i) => d === b[i]);

export function sharedRange(a: Stats | null, b: Stats | null): [number, number] | null {
  if (a?.min == null || a.max == null || b?.min == null || b.max == null) return null;
  return [Math.min(a.min, b.min), Math.max(a.max, b.max)];
}
