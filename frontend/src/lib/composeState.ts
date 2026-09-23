// Pure helpers for composing interventions: pixel selection on pooled grids, continuations, source runs.
import { fixed } from "./format";
import { cellOf, cellStart } from "./tensorState";
import { showToken } from "./tokens";
import type { GenerationResult, GraphNode, RunMeta, TopK } from "./types";

type Px = [number, number];

const sortPx = (xs: Px[]) => xs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);

/** Toggle every pixel of pooled cell (r, c): all selected → removed, else added; sorted by (y, x). */
export function toggleCells(sel: Px[], H: number, W: number, rows: number, cols: number, r: number, c: number): Px[] {
  const cell: Px[] = [];
  for (let y = cellStart(H, rows, r); y < cellStart(H, rows, r + 1); y++)
    for (let x = cellStart(W, cols, c); x < cellStart(W, cols, c + 1); x++) cell.push([y, x]);
  const key = (p: Px) => `${p[0]},${p[1]}`;
  const have = new Set(sel.map(key));
  if (cell.every((p) => have.has(key(p)))) {
    const drop = new Set(cell.map(key));
    return sortPx(sel.filter((p) => !drop.has(key(p))));
  }
  return sortPx([...sel, ...cell.filter((p) => !have.has(key(p)))]);
}

/** Pooled cells that hold at least one selected pixel (DotMatrix `selected`). */
export const cellsOf = (sel: Px[], H: number, W: number, rows: number, cols: number): Px[] =>
  sortPx([...new Set(sel.map(([y, x]) => `${cellOf(H, rows, y)},${cellOf(W, cols, x)}`))].map((k) => k.split(",").map(Number) as Px));

/** First index where the two id lists differ (a shorter list differs where it ends); null when identical. */
export function divergeIndex(a: number[], b: number[]): number | null {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? null : n;
}

/** `step 3 · clean "a" 0.41 · "t" 0.12 · "e" 0.05 | intervened "c" 0.33 · …` (top-3 per side). */
export function genReadout(res: GenerationResult, i: number): string {
  const side = (top: GenerationResult["clean"]["top"]) =>
    (top[i] ?? []).slice(0, 3).map((t) => `"${showToken(t.token)}" ${fixed(t.prob)}`).join(" · ") || "—";
  return `step ${i} · clean ${side(res.clean.top)} | intervened ${side(res.steered.top)}`;
}

/** Runs of the same model spec (non-null) and weights, this run first. */
export function sameModelRuns(runs: RunMeta[], run: RunMeta): RunMeta[] {
  if (run.model.spec == null) return [];
  return [run, ...runs.filter((r) => r.id !== run.id && r.model.spec === run.model.spec && r.model.weights === run.model.weights)];
}

/** A node whose output can be edited: not an input / output, float dtype (unknown dtype on old runs passes). */
export const editable = (n: Pick<GraphNode, "is_input" | "is_output" | "dtype">) =>
  !n.is_input && !n.is_output && (!n.dtype || /float/.test(n.dtype));

/** Top-k hover readout: `before #2 "cat" 0.12 → after #1 0.41` for logit `index` ("—" on a side it is missing from). */
export function topReadout(before: TopK[], after: TopK[], index: number): string {
  const at = (xs: TopK[]) => {
    const k = xs.findIndex((t) => t.index === index);
    return k < 0 ? ["—", "—"] : [`#${k + 1}`, fixed(xs[k].prob)];
  };
  const t = before.find((x) => x.index === index) ?? after.find((x) => x.index === index);
  const [b, a] = [at(before), at(after)];
  return `before ${b[0]} "${t?.label != null ? showToken(t.label) : `[${index}]`}" ${b[1]} → after ${a[0]} ${a[1]}`;
}
