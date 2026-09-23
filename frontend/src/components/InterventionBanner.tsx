// One quiet row under the TopBar while an intervention is open (URL `iv`): what was edited and how the target moved.
import { signed } from "../lib/format";
import { deltaColor } from "../lib/interventionState";
import type { InterventionResult } from "../lib/types";

export function InterventionBanner({ result, error, clearHref, note }: {
  result: InterventionResult | null; error: string | null; clearHref: string; note?: string;
}) {
  if (!result && !error) return null;
  const n = result?.set.edits.length ?? 0;
  return (
    <div className="panel" role="note" style={{ height: 32, flexShrink: 0, display: "flex", alignItems: "center", gap: 10, padding: "0 20px", borderBottom: "1px solid var(--line)", minWidth: 0 }}>
      <span className="lbl pill">Intervened</span>
      {result ? (
        <span className="lbl dim" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={result.labels.join("\n")}>
          {result.name} · {n} edit{n === 1 ? "" : "s"} · Δ logit <span style={{ color: deltaColor(result.delta_logit) }}>{signed(result.delta_logit, 2)}</span>
          {" "}· {result.saved.length}/{result.downstream.length} downstream tensors saved{result.truncated && " · budget reached"}{note && ` · ${note}`}
        </span>
      ) : <span className="err" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{error}</span>}
      <span style={{ flexGrow: 1 }} />
      <a className="lbl chart-export" href={clearHref} aria-label="Show the clean run">Clean</a>
    </div>
  );
}
