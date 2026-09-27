// Pure Trace view state: URL params (defaults omitted), the attribution request, agreement cell tones, and the
// labels of a sequence input's channel × time map.
import { num } from "./format";
import { clock } from "./inputState";
import type { InputBlock, InputMap, InputTime, JobRequest, LayerMethod, Method, Objective, OutputFull, SeriesMap, TargetWindow, WindowKind } from "./types";

export interface TraceState {
  node: string | null;          // selected graph node
  m: Method;                    // input method, default "ig"
  t: number | null;             // logit target, null = top prediction
  obj: Objective | null;        // unit objective (URL obj + ch; both required, else null); overrides t
  steps: number;                // 1..512, default 32 (invalid → 32)
  layer: LayerMethod | null;    // layer panel method; null = panel closed
  lop: string | null;           // op for the layer / neuron panel (a node's output_op)
  nc: number | null;            // neuron-conductance channel within lop
  go: boolean;                  // deep link: run once on arrival, then drop go from the URL
  iv: string | null;            // open intervention (banner and Δ readouts; attribution stays on the clean pass)
}

const METHODS: Method[] = ["gxa", "saliency", "ig"];
const LAYERS: LayerMethod[] = ["layer_ig", "layer_cond"];
const int = (s: string | null) => (s != null && /^\d+$/.test(s.trim()) ? Number(s) : null);
const str = (n: number | null | undefined) => (n == null ? null : String(n));

export function parseTraceParams(params: URLSearchParams): TraceState {
  const m = params.get("m") as Method, layer = params.get("layer") as LayerMethod;
  const op = params.get("obj"), ch = int(params.get("ch")), steps = int(params.get("steps"));
  return {
    node: params.get("node") || null,
    m: METHODS.includes(m) ? m : "ig",
    t: int(params.get("t")),
    obj: op && ch != null ? { op, channel: ch } : null,
    steps: steps != null && steps >= 1 && steps <= 512 ? steps : 32,
    layer: LAYERS.includes(layer) ? layer : null,
    lop: params.get("lop") || null,
    nc: int(params.get("nc")),
    go: params.get("go") === "1",
    iv: params.get("iv") || null,
  };
}

/** URL params for href(); missing fields are defaults, and defaults are omitted. */
export function traceParams(s: Partial<TraceState>): Record<string, string | null> {
  const steps = s.steps ?? 32;
  return {
    node: s.node ?? null, m: s.m && s.m !== "ig" ? s.m : null, t: str(s.t),
    obj: s.obj?.op ?? null, ch: str(s.obj?.channel),
    steps: steps === 32 ? null : String(steps),
    layer: s.layer ?? null, lop: s.lop ?? null, nc: str(s.nc), go: s.go ? "1" : null, iv: s.iv ?? null,
  };
}

/** The layer panel's op: never the unit objective's own op (layer methods need a layer strictly before it). */
export const layerOp = (lop: string | null, obj: Objective | null) => (obj && lop === obj.op ? null : lop);

/** A tensor view's channel index is the op's axis-1 channel (what a unit objective means) only for (1, C, H, W). */
export const channelIsAxis1 = (shape: number[]) => shape.length === 4 && shape[0] === 1;

/** A logit objective cannot run when the model output has no logits; a unit objective still can. */
export const logitsBlocked = (s: Pick<TraceState, "obj">, outputError: string | null | undefined) => !s.obj && !!outputError;

export function attrRequest(s: TraceState): Extract<JobRequest, { kind: "attribution" }> {
  return { kind: "attribution", method: s.m, target: s.obj ? null : s.t, objective: s.obj, steps: s.steps };
}

/** Agreement matrix cell tone: jaccard/overlap in [0,1] → accent alpha v; spearman in [-1,1] → accent alpha v (v ≥ 0),
 * negative alpha |v| (v < 0); null → none. */
export function matrixTone(v: number | null): { color: "accent" | "negative" | null; alpha: number } {
  if (v == null || !Number.isFinite(v)) return { color: null, alpha: 0 };
  return { color: v < 0 ? "negative" : "accent", alpha: Math.min(1, Math.abs(v)) };
}

// ---- a sequence input's attribution map (rows = channels in block order, columns = time) ----

export const isSeries = (m: InputMap | null | undefined): m is SeriesMap => m?.kind === "series";

/** The display clock of a series map: seconds on the run's clock (minutes from a 10 min span), else steps.
 * `at(i)`: step i (fractional for a column centre) in display units. */
export function seriesClock(m: SeriesMap): { div: number; label: string; unit: string; at: (i: number) => number } {
  const t = m.time;
  if (!t) return { div: 1, label: "step", unit: "", at: (i) => i };
  const c = clock((m.steps - 1) * t.step_s, t.label);
  return { ...c, unit: c.div === 60 ? "min" : "s", at: (i) => (t.t0 + i * t.step_s) / c.div };
}

/** Row r: its block and channel label, `fhr_st · "ψ3 · 0.887 Hz"`; a one-channel block is its name; pooled rows `ch 160–162`. */
export function seriesRow(m: SeriesMap, r: number): string {
  const a = r * m.row_stride, z = Math.min(m.channels, a + m.row_stride) - 1;
  const b = m.blocks.find((k) => a >= k.start && a < k.stop);
  const label = a !== z ? `ch ${a}–${z}` : b?.labels?.[a - b.start] != null ? `"${b.labels[a - b.start]}"`
    : b && b.stop - b.start === 1 ? null : `ch ${a}`;
  return [b?.name ?? `ch ${a}`, label].filter(Boolean).join(" · ");
}

/** Column c: `t=610.00 s` at the centre of its steps, or `step 3–5` without a clock. */
export function seriesCol(m: SeriesMap, c: number): string {
  const a = c * m.stride, z = Math.min(m.steps, a + m.stride) - 1;
  if (!m.time) return `step ${a === z ? a : `${a}–${z}`}`;
  const k = seriesClock(m);
  return `t=${num(k.at((a + z) / 2))} ${k.unit}`;
}

/** Each block's rows as [top, bottom) in % of the map height (block separators and names beside the map). */
export function blockSpans(m: SeriesMap): { name: string; top: number; bottom: number }[] {
  const rows = Math.ceil(m.channels / m.row_stride) || 1;
  const pc = (ch: number) => (ch >= m.channels ? 100 : (Math.floor(ch / m.row_stride) / rows) * 100);
  return m.blocks.map((b) => ({ name: b.name, top: pc(b.start), bottom: pc(b.stop) }));
}

// ---- a picked output step: its input windows over the series map ----

/** The time axis (1 or 2) of a (1, A, B) input, as the server's `time_axis`: meta.input.time's, else 1 with blocks,
 * else the longer axis. */
export function seqTimeAxis(shape: number[], input?: { time?: InputTime | null; blocks?: InputBlock[] | null } | null): number {
  const a = input?.time?.axis;
  if (a === 1 || a === 2) return a;
  return input?.blocks?.length ? 1 : shape[1] >= shape[2] ? 1 : 2;
}

/** The channels [start, stop) that a window's named blocks span (null: every channel, or no block matched). */
export function blockChannels(blocks: InputBlock[], names: string[] | null): [number, number] | null {
  const hit = names ? blocks.filter((b) => names.includes(b.name)) : [];
  return hit.length ? [Math.min(...hit.map((b) => b.start)), Math.max(...hit.map((b) => b.stop))] : null;
}

/** The Values panel's slice of steps [a, b) and channels ch (null = all) of a (1, T, C) / (1, C, T) input:
 * `[0, 250:261, 80:116]`. */
export function sliceText(shape: number[], axis: number, [a, b]: [number, number], ch: [number, number] | null): string {
  return `[${shape.map((_, i) => (i === 0 ? "0" : i === axis ? `${a}:${b}` : ch ? `${ch[0]}:${ch[1]}` : ":")).join(", ")}]`;
}

/** Map rows [r0, r1) of a window's blocks (null blocks: every row). */
export function windowRows(m: SeriesMap, blocks: string[] | null): [number, number][] {
  if (!blocks) return [[0, m.map.length]];
  return m.blocks.filter((b) => blocks.includes(b.name)).map((b) => [Math.floor(b.start / m.row_stride), Math.ceil(b.stop / m.row_stride)]);
}

/** Map columns [c0, c1) that hold steps [a, b). */
export const windowCols = (m: SeriesMap, a: number, b: number): [number, number] => [Math.floor(a / m.stride), Math.ceil(b / m.stride)];

export interface WindowShare { label: string; kind: WindowKind | "outside"; start: number; stop: number; blocks: string[] | null; share: number; sum: number }

/** "Where the attribution is": each window's share of Σ|a| over the map and its signed Σ (map × scale), then the cells
 * outside every window. Windows may overlap (the step inside its context), so their shares need not add up. */
export function windowShares(m: SeriesMap, windows: TargetWindow[]): WindowShare[] {
  const rows = m.map.length, cols = m.map[0]?.length ?? 0, scale = m.scale ?? 1;
  const covered = m.map.map(() => new Uint8Array(cols));
  let whole = 0, outAbs = 0, outSum = 0;
  for (const r of m.map) for (const v of r) whole += Math.abs(v);
  const out: WindowShare[] = windows.map((w) => {
    const [c0, c1] = windowCols(m, w.start, w.stop);
    let abs = 0, sum = 0;
    for (const [r0, r1] of windowRows(m, w.blocks)) for (let r = r0; r < Math.min(rows, r1); r++) for (let c = c0; c < Math.min(cols, c1); c++) {
      const v = m.map[r][c];
      abs += Math.abs(v); sum += v; covered[r][c] = 1;
    }
    return { ...w, share: whole ? abs / whole : 0, sum: sum * scale };
  });
  m.map.forEach((r, i) => r.forEach((v, c) => { if (!covered[i][c]) { outAbs += Math.abs(v); outSum += v; } }));
  return [...out, { label: "outside all windows", kind: "outside", start: 0, stop: m.steps, blocks: null, share: whole ? outAbs / whole : 0, sum: outSum * scale }];
}

/** Steps [lo, hi) of the "Around step" zoom: the union of the windows and the step ± 15 % of it (≥ 2 steps), or the
 * step ± `pad` without windows; clipped to [0, steps). */
export function zoomSpan(steps: number, t: number, windows: TargetWindow[]): [number, number] {
  // the step and its target / horizon windows, plus a lead-in of 10 % of the input (2–30 steps); context windows
  // are left out, since one often spans the whole causal past and the zoom would then show everything
  const pad = Math.max(2, Math.min(30, Math.ceil(steps * 0.1)));
  const focus = windows.filter((w) => w.kind !== "context");
  const a = Math.min(t, ...focus.map((w) => w.start)), b = Math.max(t + 1, ...focus.map((w) => w.stop));
  return [Math.max(0, a - pad), Math.min(steps, b + pad)];
}

/** `t40 · +5:28 · value 0.412` for output element i (index form without a label; no value when pooled). */
export function stepText(out: OutputFull | null, i: number, what = "value"): string {
  const label = out?.labels?.[i], v = out && out.stride === 1 ? out.values[i] : null;
  return [label ?? `[${i}]`, v != null ? `${what} ${num(v, 4)}` : null].filter(Boolean).join(" · ");
}

/** `step 40` or `steps 28–39` for steps [a, b). */
export const stepsLabel = (a: number, b: number) => (b - a === 1 ? `step ${a}` : `steps ${a}–${b - 1}`);
