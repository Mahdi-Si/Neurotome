// Pure Lens view state: URL params (defaults omitted), cell text, the readout line, short layer names.
import { showToken } from "./tokens";
import type { LensResult } from "./types";

export type LensMode = "top" | "target";
export interface LensState {
  t: number | null;     // target token id, null = top prediction
  mode: LensMode;       // cell colour: top-1 prob (default) or target prob
  pos: number | null;   // selected position, null = last
}

const int = (s: string | null) => (s != null && /^\d+$/.test(s.trim()) ? Number(s) : null);

export const parseLensParams = (params: URLSearchParams): LensState =>
  ({ t: int(params.get("t")), mode: params.get("mode") === "target" ? "target" : "top", pos: int(params.get("pos")) });

export const lensParams = (s: LensState): Record<string, string | null> =>
  ({ t: s.t == null ? null : String(s.t), mode: s.mode === "target" ? "target" : null, pos: s.pos == null ? null : String(s.pos) });

/** `transformer.h.3` → `h.3`, `model.layers.12` → `layers.12`, `embed` → `embed`. */
export const shortLayer = (name: string) => name.split(".").slice(-2).join(".");

/** At most n characters, the last one "…" when cut. */
export const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

/** Column j (position pos0 + j) of the lens grid, as its index into the result (last column for null / out of range). */
export const lensColumn = (res: LensResult, pos: number | null) =>
  pos != null && pos >= res.pos0 && pos < res.pos0 + res.tokens.length ? pos - res.pos0 : res.tokens.length - 1;

/** Text in cell (layer l, column j): the top-1 token (6 chars) or the target's `#rank`. */
export const lensCellText = (res: LensResult, mode: LensMode, l: number, j: number) =>
  mode === "top" ? clip(showToken(res.top[l][j][0]?.token ?? ""), 6) : `#${res.target_rank[l][j]}`;

/** `h.3 · pos 5 "cat" → "dog" 0.41 · "the" 0.12 · "a" 0.05 · target "mat" #17 p 0.003` */
export function lensReadout(res: LensResult, l: number, j: number): string {
  const top = res.top[l][j].slice(0, 3).map((c) => `"${showToken(c.token)}" ${c.prob.toFixed(2)}`).join(" · ");
  return `${shortLayer(res.layers[l])} · pos ${res.pos0 + j} "${showToken(res.tokens[j])}" → ${top}`
    + ` · target "${showToken(res.target.token)}" #${res.target_rank[l][j]} p ${res.target_prob[l][j].toFixed(3)}`;
}
