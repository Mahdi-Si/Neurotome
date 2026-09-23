// Inspector section: replace a node's output and re-run the forward pass (activation patching).
import { useState } from "react";
import { api } from "../lib/api";
import { num, signed } from "../lib/format";
import type { PatchState } from "../lib/graphState";
import type { Baseline, GraphNode, Job, PatchResult } from "../lib/types";
import { useAsync } from "../lib/useAsync";
import { JobProgress } from "./JobProgress";

export interface PatchProps {
  spec: string | null;               // this run's model.spec; null = in-process run, no source runs
  weights: string | null;            // this run's model.weights; source runs must match both
  state: PatchState | null;          // the patch in the URL
  job: Job<PatchResult> | null;
  error: string | null;
  result: PatchResult | null;        // the result for `state`, null while running or stale
  onApply: (p: PatchState) => void;
  onCancel: () => void;
  onClear: () => void;
}

const BASELINES: [Baseline, string][] = [["zeros", "Zeros"], ["mean", "Spatial mean"], ["run", "Run"]];

/** Render keyed by node and URL patch state: the local choice resets per node and follows the URL. */
export function PatchPanel({ runId, node, patch }: { runId: string; node: GraphNode; patch: PatchProps }) {
  const { state, job, result } = patch;
  const mine = state?.op === node.output_op;
  const [base, setBase] = useState<Baseline>(mine ? state.baseline : "zeros");
  const [src, setSrc] = useState<string | null>(mine ? state.source : null);
  const runs = useAsync(() => (base === "run" && patch.spec ? api.runs() : null), [base === "run", patch.spec]);
  const others = (runs.data ?? []).filter((r) => r.id !== runId && r.model.spec === patch.spec && r.model.weights === patch.weights);
  const source = base === "run" ? (others.some((r) => r.id === src) ? src : others[0]?.id ?? null) : null;
  const r = result?.op === node.output_op ? result : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span className="lbl">Patch · replace this node's output</span>
        {state && <button type="button" className="lbl chart-export" onClick={patch.onClear}>Clear</button>}
      </div>
      <div className="seg lbl" role="group" aria-label="Patch baseline">
        {BASELINES.map(([id, text]) => <button key={id} type="button" aria-pressed={id === base} onClick={() => setBase(id)}>{text}</button>)}
      </div>
      {base === "run" && (
        <>
          <select className="field" aria-label="Source run" value={source ?? ""} disabled={!source} onChange={(e) => setSrc(e.target.value)}>
            {others.map((o) => <option key={o.id} value={o.id}>{o.id} · {o.sample.name}</option>)}
          </select>
          {!source && !runs.loading && <span className="lbl dim">No other run of this model</span>}
          {runs.error && <span className="err">{runs.error}</span>}
        </>
      )}
      <button type="button" className="btn solid lbl" disabled={base === "run" && !source}
        onClick={() => patch.onApply({ op: node.output_op, baseline: base, source })}>Patch & re-run</button>
      {mine && <JobProgress job={job} onCancel={patch.onCancel} />}
      {mine && patch.error && job?.status !== "error" && <span className="err">{patch.error}</span>}
      {r && (
        <div className="rule-top">
          {([
            ["Target", `logit[${r.target}] · ${r.target_label ?? "—"}`],
            ["Logit", <>{num(r.before.logit)} → {num(r.after.logit)}{" "}
              <span style={{ color: r.delta_logit >= 0 ? "var(--accent)" : "var(--negative-text)" }}>(Δ {signed(r.delta_logit, 2)})</span></>],
            ["Prob", `${num(r.before.prob)} → ${num(r.after.prob)} (Δ ${signed(r.delta_prob, 2)})`],
            ["Method", r.method_label],
          ] as const).map(([k, v]) => (
            <div key={k} className="kv"><span className="lbl dim">{k}</span><span className="mono" style={{ textAlign: "right" }}>{v}</span></div>
          ))}
          {r.top_after.slice(0, 5).map((t) => (
            <div key={t.index} className="kv dim"><span className="mono">{t.label ?? `[${t.index}]`}</span><span className="num">{num(t.prob)}</span></div>
          ))}
        </div>
      )}
    </div>
  );
}
