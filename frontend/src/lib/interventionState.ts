// Pure intervention-set state: the `set` draft in the URL, index lists, scope axes, client edit labels.
import type { Edit, EditKind, InterventionSet } from "./types";

const KINDS: EditKind[] = ["zero", "mean", "resample", "constant", "scale", "steer"];

const nat = (x: unknown) => Number.isInteger(x) && (x as number) >= 0;
const opt = (x: unknown, ok: (v: unknown) => boolean) => x == null || ok(x);
const finiteNum = (x: unknown) => typeof x === "number" && Number.isFinite(x);
const isPositions = (p: unknown) => p == null || p === "last" || (Array.isArray(p)
  && (p.every(nat) || p.every((q) => Array.isArray(q) && q.length === 2 && q.every(nat))));
const isDirection = (d: unknown) => {
  const x = d as Record<string, unknown>;
  return x?.kind === "channel" ? nat(x.channel)
    : x?.kind === "contrast" && typeof x.run_a === "string" && typeof x.run_b === "string" && (x.pool === "mean" || x.pool === "last");
};

/** Every field the labels and the form read has its type: a hand-edited URL (`"alpha":"2"`) must not crash a view. */
const isEdit = (e: unknown): e is Edit => {
  const x = e as Record<string, unknown>;
  return !!e && typeof e === "object" && typeof x.op === "string" && KINDS.includes(x.kind as EditKind)
    && opt(x.channels, (c) => Array.isArray(c) && c.every(nat)) && isPositions(x.positions)
    && opt(x.over, (o) => o === "positions" || o === "runs") && opt(x.source_run, (s) => typeof s === "string")
    && opt(x.seed, nat) && opt(x.value, finiteNum) && opt(x.alpha, finiteNum) && opt(x.direction, isDirection);
};

/** The `set` URL param as a set; null when missing, malformed or without edits (never throws). */
export function parseSet(params: URLSearchParams): InterventionSet | null {
  const raw = params.get("set");
  if (!raw) return null;
  try {
    const s = JSON.parse(raw);
    if (!Array.isArray(s?.edits) || !s.edits.length || !s.edits.every(isEdit)) return null;
    const t = s.target;
    return { edits: s.edits, target: Number.isInteger(t) && t >= 0 ? t : null };
  } catch {
    return null;
  }
}

export const setParam = (s: InterventionSet | null): string | null => (s?.edits.length ? JSON.stringify(s) : null);

export const addEdit = (s: InterventionSet | null, e: Edit): InterventionSet => ({ edits: [...(s?.edits ?? []), e], target: s?.target ?? null });

export function removeEdit(s: InterventionSet, i: number): InterventionSet | null {
  const edits = s.edits.filter((_, k) => k !== i);
  return edits.length ? { ...s, edits } : null;
}

/** "" → []; "0, 3,7-9" → [0, 3, 7, 8, 9] (sorted, unique); a reversed range, an index ≥ `limit` or junk → null. */
export function parseIndexList(text: string, limit = 1 << 20): number[] | null {
  const out = new Set<number>();
  for (const part of text.split(",").map((p) => p.trim()).filter(Boolean)) {
    const m = /^(\d+)(?:\s*[-–]\s*(\d+))?$/.exec(part);
    if (!m) return null;
    const a = Number(m[1]), b = m[2] == null ? a : Number(m[2]);
    if (b < a || b >= limit) return null;   // checked before expanding: "0-999999999" must not loop
    for (let i = a; i <= b; i++) out.add(i);
  }
  return [...out].sort((x, y) => x - y);
}

/** [0, 3, 7, 8, 9] → "0,3,7–9": runs of three or more print as a range. */
export function formatIndexList(xs: number[] | null): string {
  if (!xs?.length) return "";
  const parts: string[] = [];
  for (let i = 0; i < xs.length; ) {
    let j = i;
    while (j + 1 < xs.length && xs[j + 1] === xs[j] + 1) j++;
    if (j - i >= 2) parts.push(`${xs[i]}–${xs[j]}`);
    else for (let k = i; k <= j; k++) parts.push(String(xs[k]));
    i = j + 1;
  }
  return parts.join(",");
}

export interface ScopeAxes { channels: number | null; positions: "pixels" | "tokens" | null; size: number[] }

/** Channel count and position axes of an op's output shape (batch axis included), spec §1.2. */
export function scopeAxes(shape: number[]): ScopeAxes {
  if (shape.length === 4) return { channels: shape[1], positions: "pixels", size: [shape[2], shape[3]] };
  if (shape.length === 3) return { channels: shape[2], positions: "tokens", size: [shape[1]] };
  if (shape.length === 2) return { channels: shape[1], positions: null, size: [] };
  return { channels: null, positions: null, size: [] };
}

const n3 = (x: number) => String(+x.toFixed(3));

function kindLabel(e: Edit): string {
  const a = e.alpha ?? 1;
  switch (e.kind) {
    case "zero": return "Zero";
    case "mean": return e.over === "runs" ? "Mean over runs" : "Mean over positions";
    case "resample": return e.source_run === "random" || !e.source_run ? `Resample ← random · seed ${e.seed ?? 0}` : `Replace ← ${e.source_run}`;
    case "constant": return `Constant ${n3(e.value ?? 0)}`;
    case "scale": return `Scale ×${n3(a)}`;
    case "steer": {
      const d = e.direction;
      const v = !d ? "?" : d.kind === "channel" ? `e_${d.channel}` : `(${d.run_b} − ${d.run_a}, ${d.pool})`;
      return `Steer ${a < 0 ? "−" : "+"}${n3(Math.abs(a))} × ${v}`;
    }
  }
}

function scopeLabel(e: Edit): string {
  const parts: string[] = [];
  if (e.channels?.length) parts.push(`ch ${formatIndexList(e.channels)}`);
  const p = e.positions;
  if (p === "last") parts.push("last token");
  else if (p?.length && Array.isArray(p[0])) {
    const px = p as [number, number][];
    parts.push(`px ${px.slice(0, 4).map(([y, x]) => `(${y},${x})`).join(" ")}${px.length > 4 ? ` +${px.length - 4}` : ""}`);
  } else if (p?.length) parts.push(`tokens ${formatIndexList(p as number[])}`);
  return parts.join(" · ") || "all";
}

/** Client wording of a draft edit: `<Kind> · <path> · <scope>` (spec §1.5). */
export const editLabel = (e: Edit, path: string) => `${kindLabel(e)} · ${path} · ${scopeLabel(e)}`;

/** Δ logit / Δ prob colour: supports the target (≥ 0) or hurts it. */
export const deltaColor = (v: number | null) => (v == null ? undefined : v >= 0 ? "var(--accent)" : "var(--negative-text)");
