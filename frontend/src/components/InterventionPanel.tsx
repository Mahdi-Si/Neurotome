// Inspector section: this node under the open intervention (Δ row), an edit form that adds to the draft set, and Run set.
import { editable } from "../lib/composeState";
import { num, pct } from "../lib/format";
import { addEdit, setParam } from "../lib/interventionState";
import { href } from "../lib/router";
import type { GraphNode, InterventionResult, InterventionSet, Job, RunMeta } from "../lib/types";
import { EditForm } from "./EditForm";
import { JobProgress } from "./JobProgress";

export interface InterventionPanelProps {
  run: RunMeta; node: GraphNode;
  set: InterventionSet | null;                 // the draft (URL `set`)
  onSet: (s: InterventionSet | null) => void;  // write / clear the draft
  result: InterventionResult | null;           // the open intervention (URL `iv`)
  onRun: (s: InterventionSet) => void;         // run the draft as a job; the view opens its iv
  job: Job<InterventionResult> | null; error: string | null; onCancel: () => void;
}

const kv = (k: string, v: React.ReactNode) => (
  <div key={k} className="kv"><span className="lbl dim">{k}</span><span className="mono" style={{ textAlign: "right", overflowWrap: "anywhere" }}>{v}</span></div>
);

/** Δ rows of `op` under an intervention result: edited / unchanged / ‖Δ‖, rel, cosine, max |Δ|. */
function DeltaRows({ run, op, r }: { run: RunMeta; op: string; r: InterventionResult }) {
  if (!r.downstream.includes(op)) return <div className="rule-top">{kv("Unchanged", "not downstream")}</div>;
  const d = r.nodes[op];
  const labels = r.set.edits.flatMap((e, i) => (e.op === op ? [r.labels[i] ?? e.kind] : []));
  return (
    <div className="rule-top">
      {labels.length > 0 && kv("Edited", labels.join(" · "))}
      {d ? [kv("‖Δ‖", num(d.l2, 3)), kv("Rel Δ", pct(d.rel)), kv("Cosine", d.cos == null ? "—" : d.cos.toFixed(3)), kv("Max |Δ|", num(d.max_abs, 3)),
          ...(d.nonfinite ? [kv("To / from NaN, inf", String(d.nonfinite))] : [])]
        : kv("‖Δ‖", "—")}
      <div style={{ paddingTop: 7 }}>
        {r.saved.includes(op)
          ? <a className="lbl chart-export" href={href(run.id, "tensors", { node: op, iv: r.hash, vs: "clean" })}>Clean vs intervened →</a>
          : <span className="lbl dim">No intervened tensor · {r.truncated ? "the capture budget was reached before it" : "the clean run saved no tensor for it"}</span>}
      </div>
    </div>
  );
}

export function InterventionPanel({ run, node, set, onSet, result, onRun, job, error, onCancel }: InterventionPanelProps) {
  const n = set?.edits.length ?? 0;
  const op = node.output_op;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h2 className="lbl" style={{ fontWeight: 400 }}>Intervene · this node's output</h2>
        {set && <button type="button" className="lbl chart-export" onClick={() => onSet(null)}>Clear set</button>}
      </div>
      {result && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span className="lbl dim" style={{ overflowWrap: "anywhere" }}>{result.name}</span>
          <DeltaRows run={run} op={op} r={result} />
        </div>
      )}
      {editable(node)
        ? <EditForm key={op} run={run} op={{ id: op, shape: node.shape, path: node.path }} onAdd={(e) => onSet(addEdit(set, e))} />
        : <span className="lbl dim">{node.is_input || node.is_output ? "The model's input or output · pick an op inside the model" : "Only float tensors can be edited"}</span>}
      <div className="lbl dim" style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
        <span>{n} {n === 1 ? "edit" : "edits"} in set</span>
        {set && <a className="lbl chart-export" href={href(run.id, "interventions", { set: setParam(set), node: op, iv: result?.hash })}>Open in Interventions →</a>}
      </div>
      <button type="button" className="btn solid lbl" disabled={!set || job?.status === "running" || job?.status === "queued"}
        onClick={() => set && onRun(set)}>Run set ({n})</button>
      <JobProgress job={job} onCancel={onCancel} />
      {error && job?.status !== "error" && <span className="err">{error}</span>}
    </div>
  );
}
