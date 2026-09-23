// Token labels for sequence views, attention maps, the lens grid and the token strip (run.sample.tokens).
import { signed as fmtSigned, pct } from "./format";
import { cellStart } from "./tensorState";
import type { RunMeta } from "./types";

const VISIBLE: Record<string, string> = { " ": "·", "\n": "↵", "\t": "→" };

/** A token made visible: " " → "·", "\n" → "↵", "\t" → "→", "" → "∅". */
export const showToken = (s: string): string => (s === "" ? "∅" : s.replace(/[ \n\t]/g, (c) => VISIBLE[c]));

/** run.sample.tokens when it has exactly n entries (the tensor's T), else null. */
export const tokenLabels = (run: RunMeta, n: number): string[] | null =>
  run.sample.tokens?.length === n ? run.sample.tokens : null;

/** Label of pooled cell i of n over len positions: `#3 "cat"` for one position, `#64–67 "a·b·c·d"` (first 4
 * tokens joined, "…" beyond) for a span (mirrors cellStart/cellSpan in tensorState). No tokens → `#3` / `#64–67`. */
export function spanLabel(tokens: string[] | null, len: number, n: number, i: number): string {
  const a = cellStart(len, n, i), b = cellStart(len, n, i + 1) - 1;
  const at = a === b ? `#${a}` : `#${a}–${b}`;
  if (!tokens) return at;
  return `${at} "${showToken(tokens.slice(a, Math.min(b + 1, a + 4)).join(""))}${b - a >= 4 ? "…" : ""}"`;
}

/** Token strip / cell fill: {color: "accent" | "negative", alpha: |v| / max ∈ [0, 1]}; unsigned → always accent. */
export function tokenTone(v: number, max: number, signed: boolean): { color: "accent" | "negative"; alpha: number } {
  return { color: signed && v < 0 ? "negative" : "accent", alpha: max > 0 ? Math.min(1, Math.abs(v) / max) : 0 };
}

/** Token strip readout: `#3 "cat" · +0.1234 · 21% of Σ|a|` (unsigned scores print without a sign). */
export function stripReadout(tokens: string[], scores: number[], signed: boolean, i: number): string {
  const sum = scores.reduce((a, v) => a + Math.abs(v), 0);
  const v = scores[i];
  return `#${i} "${showToken(tokens[i] ?? "")}" · ${signed ? fmtSigned(v, 4) : v.toFixed(4)} · ${pct(sum > 0 ? Math.abs(v) / sum : 0)} of Σ|a|`;
}
