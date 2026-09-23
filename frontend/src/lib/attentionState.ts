// Pure helpers for the attention view: the selected head's readout, pattern words, axis letters.
import { spanLabel } from "./tokens";
import type { AttentionHead, AttentionTensorView, TensorView } from "./types";

/** Readout of cell (r, c) of the selected head's full map: `q #3 "cat" → k #1 "the" · 0.412`; v is the scaled
 * value (map / max), printed unscaled; a pooled cell says `mean`. */
export function headReadout(t: AttentionTensorView, r: number, c: number, v: number, tokens: string[] | null): string {
  const rows = t.full.map.length, cols = t.full.map[0]?.length ?? 0;
  const pooled = rows < t.tokens || cols < t.tokens;
  return `q ${spanLabel(tokens, t.tokens, rows, r)} → k ${spanLabel(tokens, t.tokens, cols, c)} · ${pooled ? "mean " : ""}${(v * t.full.max).toFixed(3)}`;
}

const PATTERN: Record<NonNullable<AttentionHead["pattern"]>, string> = { previous: "prev", first: "first", self: "self" };
export const patternLabel = (p: AttentionHead["pattern"]) => (p ? PATTERN[p] : "");

/** One letter group per axis: BCHW / BTD / BD for batch-1 tensors, BHQK (HQK without batch) for attention, else d0, d1… */
export function axisLetters(shape: number[], kind: TensorView["kind"]): string[] {
  const n = shape.length, b = shape[0] === 1;
  const letters = kind === "attention" && n === 4 && b ? "BHQK" : kind === "attention" && n === 3 ? "HQK"
    : b ? ({ 4: "BCHW", 3: "BTD", 2: "BD" } as Record<number, string>)[n] : undefined;
  return shape.map((_, i) => (letters ? letters[i] : `d${i}`));
}
