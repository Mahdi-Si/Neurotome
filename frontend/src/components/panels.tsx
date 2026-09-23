// Side panels shared by views: module tree (left) and node inspector (right).
import { useMemo, useState } from "react";
import { compact, displayName, dtypeShort, num, pct, shape, shapeShort } from "../lib/format";
import { rms } from "../lib/graphState";
import { href } from "../lib/router";
import type { GraphNode, GraphView, ModuleEntry } from "../lib/types";
import { Histogram } from "./charts";
import { Crosshair, IconSearch } from "./Icons";

export function ModuleTree({ modules, selectedPath, onPick }: {
  modules: ModuleEntry[]; selectedPath: string | null; onPick: (address: string) => void;
}) {
  const [q, setQ] = useState("");
  const shown = useMemo(() => {
    const s = q.trim().toLowerCase();
    return s ? modules.filter((m) => m.address.toLowerCase().includes(s) || m.class_name.toLowerCase().includes(s)) : modules;
  }, [modules, q]);
  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: 0, flexGrow: 1 }}>
      <div style={{ padding: "14px 16px 10px", display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span className="lbl">Modules</span><span className="lbl dim">{modules.length}</span>
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 8, height: 32, padding: "0 10px", border: "1px solid var(--line)", background: "var(--panel)", color: "var(--ink-2)" }}>
          <IconSearch />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter modules" aria-label="Filter modules"
            className="mono" style={{ border: 0, outline: 0, background: "transparent", width: "100%", color: "var(--ink)" }} />
        </label>
      </div>
      <div className="scroll" style={{ flexGrow: 1, padding: "0 8px 12px" }}>
        {shown.map((m) => {
          const on = selectedPath === m.address;
          const ancestor = !on && !!selectedPath && selectedPath.startsWith(m.address + ".");
          return (
            <button key={m.address} type="button" onClick={() => onPick(m.address)} className="mono"
              style={{
                width: "100%", display: "flex", alignItems: "center", height: 27, padding: "0 8px", border: 0,
                borderRadius: 4, cursor: "pointer", textAlign: "left",
                background: on ? "var(--accent-soft)" : "transparent",
                color: on ? "var(--accent)" : ancestor ? "var(--ink)" : "var(--ink)", fontWeight: ancestor ? 500 : 400,
              }}>
              <span style={{ width: (m.depth - 1) * 14, flexShrink: 0, alignSelf: "stretch", borderRight: m.depth > 1 ? "1px solid var(--line-soft)" : 0, marginRight: m.depth > 1 ? 8 : 0 }} />
              <span style={{ flexGrow: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.name}</span>
              <span className="lbl dim" style={{ fontSize: 9.5 }}>{m.class_name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function Inspector({ runId, view, node, total, linkFor = (id) => href(runId, "graph", { node: id }) }: {
  runId: string; view: GraphView; node: GraphNode | null; total: number; linkFor?: (id: string) => string;
}) {
  const rel = useMemo(() => {
    if (!node) return { from: [] as GraphNode[], to: [] as GraphNode[], rank: null as number | null };
    const byId = new Map(view.nodes.map((n) => [n.id, n]));
    const from = view.edges.filter((e) => e.target === node.id).map((e) => byId.get(e.source)!).filter(Boolean);
    const to = view.edges.filter((e) => e.source === node.id).map((e) => byId.get(e.target)!).filter(Boolean);
    const ranked = view.nodes.filter((n) => n.score != null && !n.is_output).sort((a, b) => Math.abs(b.score!) - Math.abs(a.score!));
    const idx = ranked.findIndex((n) => n.id === node.id);
    return { from, to, rank: idx >= 0 ? idx + 1 : null };
  }, [view, node]);

  if (!node) return <div className="empty lbl">Select a node</div>;
  const st = node.stats;
  const index = view.nodes.findIndex((n) => n.id === node.id) + 1;
  return (
    <div style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: 16, minHeight: 0 }} className="scroll">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Crosshair /><span className="lbl dim">Node {index} / {total}</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <span className="num" style={{ fontSize: 19, fontWeight: 500, letterSpacing: "-0.02em", wordBreak: "break-all" }}>{displayName(node)}</span>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <span className="lbl pill">{node.type}</span>
          {node.kind === "module" && <span className="lbl pill soft">{node.num_ops} ops · collapsed</span>}
        </div>
      </div>
      <div className="rule-top">
        {[
          ["Output", shape(node.shape)],
          ["Params", compact(node.num_params)],
          ["FLOPs", compact(node.flops)],
          ["Dtype", dtypeShort(node.dtype)],
          ["Device", node.device ?? "—"],
          ["Tensor op", node.output_op],
        ].map(([k, v]) => (
          <div key={k} className="kv"><span className="lbl dim">{k}</span><span className="mono" style={{ textAlign: "right", wordBreak: "break-all" }}>{v}</span></div>
        ))}
      </div>
      {st && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}><span className="lbl">Activation</span><span className="lbl dim">{shapeShort(node.shape)}</span></div>
          <Histogram counts={st.hist} range={st.hist_range} highlight={(i) => i === 0 && (st.zero_frac ?? 0) > 0.2} />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 8 }}>
            {[["Mean", num(st.mean)], ["Std", num(st.std)], ["RMS", num(rms(st))], ["Max", num(st.max)], ["Zero", pct(st.zero_frac)]].map(([k, v]) => (
              <div key={k} style={{ display: "flex", flexDirection: "column", gap: 2 }}><span className="lbl dim">{k}</span><span className="num">{v}</span></div>
            ))}
            {st.nan != null && (
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <span className="lbl dim">NaN / Inf</span>
                <span className="num" style={{ color: st.nan + (st.inf ?? 0) > 0 ? "var(--negative-text)" : undefined }}>{st.nan} / {st.inf ?? "—"}</span>
              </div>
            )}
          </div>
        </div>
      )}
      {node.score != null && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span className="lbl">Attribution (grad × act)</span>
            {rel.rank && <span className="lbl" style={{ color: "var(--accent)" }}>#{rel.rank}</span>}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ flexGrow: 1, height: 4, background: "var(--line-soft)", display: "flex" }}>
              <div style={{ width: `${Math.round((node.score_norm ?? 0) * 100)}%`, background: "var(--accent)" }} />
            </div>
            <span className="num">{num(node.score, 3)}</span>
          </div>
        </div>
      )}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12 }}>
        {([["From", rel.from], ["Feeds", rel.to]] as const).map(([k, list]) => (
          <div key={k} style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
            <span className="lbl dim">{k}</span>
            {list.length === 0 && <span className="mono dim">—</span>}
            {list.map((n) => (
              <a key={n.id} className="mono" href={linkFor(n.id)} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{displayName(n)}</a>
            ))}
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 8, marginTop: "auto" }}>
        <a href={href(runId, "tensors", { node: node.output_op })} className="btn solid lbl" style={{ height: 38 }}>Open tensor →</a>
        <a href={href(runId, "trace", { node: node.id })} className="btn lbl" style={{ height: 38 }}>Trace</a>
      </div>
    </div>
  );
}
