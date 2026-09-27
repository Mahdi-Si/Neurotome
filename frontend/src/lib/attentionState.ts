// Pure helpers for the attention view: the selected head's readout, pattern words, head-table columns, axis letters.
import { pct } from "./format";
import { spanLabel } from "./tokens";
import type { AttentionHead, AttentionTensorView, TensorView } from "./types";

/** Readout of cell (r, c) of the selected head's full map: `q #3 "cat" → k #1 "the" · 0.412`; v is the scaled
 * value (map / max), printed unscaled; a pooled cell says `mean`. `tokens` (the run's) label an axis only when it
 * has exactly as many positions (keys of lag attention are lags, not tokens). */
export function headReadout(t: AttentionTensorView, r: number, c: number, v: number, tokens: string[] | null): string {
  const rows = t.full.map.length, cols = t.full.map[0]?.length ?? 0;
  const fit = (n: number) => (tokens?.length === n ? tokens : null);
  const pooled = rows < t.queries || cols < t.keys;
  return `q ${spanLabel(fit(t.queries), t.queries, rows, r)} → k ${spanLabel(fit(t.keys), t.keys, cols, c)} · ${pooled ? "mean " : ""}${(v * t.full.max).toFixed(3)}`;
}

const PATTERN: Record<NonNullable<AttentionHead["pattern"]>, string> = { previous: "prev", first: "first", self: "self" };
export const patternLabel = (p: AttentionHead["pattern"]) => (p ? PATTERN[p] : "");

const f2 = (v: number | undefined) => (v == null ? "—" : v.toFixed(2));

/** Head-summary table columns after `Head`: token patterns for square maps, masked rows and normalised entropy
 * otherwise; zeros and the peak key for both. (`Pattern` is styled, so the view adds it itself for square maps.) */
type Column = [string, (h: AttentionHead) => string];
export function headColumns(square: boolean): Column[] {
  const mid: Column[] = square
    ? [["Prev", (h) => f2(h.prev)], ["First", (h) => f2(h.first)], ["Self", (h) => f2(h.self)]]
    : [["Entropy / log K", (h) => f2(h.entropy_norm)], ["Masked", (h) => pct(h.masked)]];
  return [
    ["Entropy (bits)", (h) => f2(h.entropy)],
    ...mid,
    ["Zeros", (h) => pct(h.sparsity)],
    ["Peak key", (h) => `k${h.argmax} · ${h.argmax_weight.toFixed(3)}`],
  ];
}

/** One letter group per axis: BCHW / BTD / BD for batch-1 tensors, BHQK (HQK without batch) for attention — BQHK /
 * QHK when heads sit in the middle — else d0, d1… */
export function axisLetters(shape: number[], kind: TensorView["kind"], layout: AttentionTensorView["layout"] = "hqk"): string[] {
  const n = shape.length, b = shape[0] === 1;
  const hqk = layout === "qhk" ? "QHK" : "HQK";
  const letters = kind === "attention" && n === 4 && b ? "B" + hqk : kind === "attention" && n === 3 ? hqk
    : b ? ({ 4: "BCHW", 3: "BTD", 2: "BD" } as Record<number, string>)[n] : undefined;
  return shape.map((_, i) => (letters ? letters[i] : `d${i}`));
}
