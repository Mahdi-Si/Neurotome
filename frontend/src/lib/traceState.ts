// Pure Trace view state: URL params (defaults omitted), the attribution request, agreement cell tones.
import type { JobRequest, LayerMethod, Method, Objective } from "./types";

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
