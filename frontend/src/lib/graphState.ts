// Pure graph-workspace state: fold overrides in the URL, search ranking, colour modes.
import { num, pct, shape, shapeShort } from "./format";
import type { Baseline, ColorMode, Fold, GraphView, IndexEntry, NodeTone, PatchResult, Stats } from "./types";

const list = (s: string | null) => (s ?? "").split(",").filter(Boolean);
const uniq = (a: string[]) => [...new Set(a)];

export const parseFold = (params: URLSearchParams): Fold => ({ expand: list(params.get("expand")), collapse: list(params.get("collapse")) });

export const foldParams = (f: Fold) => ({ expand: f.expand.join(",") || null, collapse: f.collapse.join(",") || null });

export function toggleFold(f: Fold, call: string, open: boolean): Fold {
  const add = (a: string[]) => uniq([...a, call]);
  const drop = (a: string[]) => a.filter((c) => c !== call);
  return open ? { expand: add(f.expand), collapse: drop(f.collapse) } : { expand: drop(f.expand), collapse: add(f.collapse) };
}

/** Opens an entry's ancestors; a module entry is also folded, so it is a node that can be selected and centred. */
export function revealFold(f: Fold, entry: IndexEntry): Fold {
  const own = entry.kind === "module" ? [entry.id.slice(2)] : [];
  return {
    expand: uniq([...f.expand, ...entry.calls]).filter((c) => !own.includes(c)),
    collapse: uniq([...f.collapse.filter((c) => !entry.calls.includes(c)), ...own]),
  };
}

/** The index entry for a module address: its module call first, else any entry with that path. */
export const moduleEntry = (entries: IndexEntry[], address: string): IndexEntry | null =>
  entries.find((e) => e.kind === "module" && e.path === address) ?? entries.find((e) => e.path === address) ?? null;

/** Module address of a call or of an "m:"/"g:" id: "m:layer1.0:1" → "layer1.0". */
export const callAddress = (id: string) => id.replace(/^[mg]:/, "").replace(/:\d+$/, "");

/** True when group `gid` is `target` or nested inside it. */
export function withinGroup(view: GraphView, gid: string | null, target: string): boolean {
  const parent = new Map(view.groups.map((g) => [g.id, g.parent]));
  for (let g = gid; g; g = parent.get(g) ?? null) if (g === target) return true;
  return false;
}

const SHAPE_TOKEN = /^\d+([x×,]\d+)+$/;

export function searchIndex(entries: IndexEntry[], q: string, limit = 8): IndexEntry[] {
  const tokens = q.toLowerCase().split(/\s+/).filter(Boolean).map((t) => (SHAPE_TOKEN.test(t) ? t.replace(/[x,]/g, "×") : t));
  if (!tokens.length) return [];
  const whole = tokens.join(" ");
  const hits: { e: IndexEntry; rank: number; i: number }[] = [];
  entries.forEach((e, i) => {
    const label = e.label.toLowerCase();
    const hay = [label, e.path.toLowerCase(), e.type.toLowerCase(), shapeShort(e.shape), shape(e.shape)];
    if (!tokens.every((t) => hay.some((h) => h.includes(t)))) return;
    hits.push({ e, i, rank: (label === whole ? 0 : label.startsWith(whole) ? 1 : 2) * 2 + (e.kind === "module" ? 0 : 1) });
  });
  return hits.sort((a, b) => a.rank - b.rank || a.i - b.i).slice(0, limit).map((h) => h.e);
}

export function colorMode(params: URLSearchParams): ColorMode {
  const c = params.get("color");
  if (c === "attr" || c === "norm" || c === "sparsity" || c === "nan" || c === "patch") return c;
  return params.get("overlay") === "attr" ? "attr" : "plain";
}

export interface PatchState { op: string; baseline: Baseline; source: string | null }

/** URL patch=<op>&pb=zeros|mean|run&psrc=<run>; pb defaults to zeros, psrc is only read for run (and required). */
export function parsePatch(params: URLSearchParams): PatchState | null {
  const op = params.get("patch");
  const pb = params.get("pb");
  const baseline: Baseline = pb === "mean" || pb === "run" ? pb : "zeros";
  const source = baseline === "run" ? params.get("psrc") : null;
  return op && (baseline !== "run" || source) ? { op, baseline, source } : null;
}

export const patchParams = (p: PatchState | null) => ({
  patch: p?.op ?? null,
  pb: p && p.baseline !== "zeros" ? p.baseline : null,
  psrc: p?.baseline === "run" ? p.source : null,
});

/** RMS over finite values, exact from population mean and std. */
export const rms = (st: Stats | null | undefined) =>
  st && st.mean != null && st.std != null ? Math.sqrt(st.mean ** 2 + st.std ** 2) : null;

export interface Legend { title: string; lo: string; hi: string; note?: string }

export function nodeTones(view: GraphView, mode: ColorMode, patch?: PatchResult | null): { tones?: Map<string, NodeTone>; legend: Legend | null } {
  const tones = new Map<string, NodeTone>();
  if (mode === "patch") {
    if (!patch) return { legend: null };
    // normalise by the largest change elsewhere; the patched node itself would always be rel 1
    const own = view.nodes.find((n) => n.output_op === patch.op)?.id;
    const max = Math.max(0, ...view.nodes.filter((n) => n.id !== own).map((n) => patch.rel[n.output_op] ?? 0));
    for (const n of view.nodes) {
      const r = patch.rel[n.output_op];
      tones.set(n.id, n.id === own ? { v: 1, readout: `patched · ${patch.baseline}` }
        : r == null ? { v: null, readout: "‖Δ‖ —" }
        : { v: max > 0 ? Math.min(1, Math.max(0, r / max)) : 0, readout: `‖Δ‖ ${num(patch.deltas[n.output_op])} · rel ${pct(r)}` });
    }
    return { tones, legend: { title: `Patch Δ = ‖after − before‖ / ‖before‖ · ${patch.path} → ${patch.baseline}`, lo: "0", hi: pct(max) } };
  }
  if (mode === "attr") {
    for (const n of view.nodes) tones.set(n.id, { v: n.score_norm ?? null, readout: `|g×a| ${num(n.score, 3)}` });
    return { tones, legend: { title: "Node = |grad × act| · gxa · target top", lo: "0", hi: "max" } };
  }
  if (mode === "norm") {
    const r = new Map(view.nodes.map((n) => [n.id, rms(n.stats)]));
    const max = Math.max(0, ...[...r.values()].map((x) => x ?? 0));
    for (const n of view.nodes) {
      const x = r.get(n.id) ?? null;
      tones.set(n.id, { v: x == null ? null : max > 0 ? x / max : 0, readout: `RMS ${num(x)}` });
    }
    return { tones, legend: { title: "RMS = ‖x‖₂/√n (finite values)", lo: "0", hi: num(max || null) } };
  }
  if (mode === "sparsity") {
    for (const n of view.nodes) tones.set(n.id, { v: n.stats?.zero_frac ?? null, readout: `Zero ${pct(n.stats?.zero_frac)}` });
    return { tones, legend: { title: "Zero fraction", lo: "0%", hi: "100%" } };
  }
  if (mode === "nan") {
    let k = 0;
    let recorded = false;
    for (const n of view.nodes) {
      const nf = n.nonfinite ?? null;
      if (nf != null) recorded = true;
      const flag = (nf ?? 0) > 0 || (!!n.stats && n.stats.numel > 0 && n.stats.mean === null);
      if (flag) k++;
      tones.set(n.id, { v: null, flag, readout: `NaN/Inf ${nf ?? "—"}` });
    }
    const note = !recorded && !k ? "Not recorded for this run · re-capture to check" : undefined;
    return { tones, legend: { title: "Flag = NaN / Inf present", lo: "0", hi: `${k} flagged`, ...(note && { note }) } };
  }
  return { legend: null };
}
