import { useMemo, useState } from "react";
import { DotMatrix } from "../components/charts";
import { GraphCanvas } from "../components/GraphCanvas";
import { Crosshair } from "../components/Icons";
import { InputOverlay, OverlayControls } from "../components/InputOverlay";
import { TopBar } from "../components/TopBar";
import { api } from "../lib/api";
import { fileName } from "../lib/exportPng";
import { displayName, num, pct, signed } from "../lib/format";
import { href, navigate, type Route } from "../lib/router";
import type { Attribution, GraphView, Method, RunMeta } from "../lib/types";

const METHODS: { id: Method; label: string }[] = [
  { id: "gxa", label: "Grad × input" },
  { id: "saliency", label: "Saliency" },
  { id: "ig", label: "Integrated grad" },
];

export function TraceViewPage({ run, route }: { run: RunMeta; route: Route }) {
  const [method, setMethod] = useState<Method>((route.params.get("m") as Method) || "ig");
  const [target, setTarget] = useState<number | null>(run.output[0]?.index ?? null);
  const [steps, setSteps] = useState(32);
  const [depth, setDepth] = useState(1);
  const [result, setResult] = useState<{ attr: Attribution; view: GraphView } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [overlay, setOverlay] = useState(false);
  const [alpha, setAlpha] = useState(0.6);
  const selected = route.params.get("node");

  const runIt = async (d = depth) => {
    setBusy(true); setError(null);
    try {
      const attr = await api.attribution(run.id, { method, target, steps });
      const view = await api.graph(run.id, d, { method, target, steps });
      setResult({ attr, view });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const view = result?.view;
  const byOp = useMemo(() => new Map((view?.nodes ?? []).map((n) => [n.output_op, n])), [view]);
  const pathNodes = useMemo(() => (view?.path ?? []).map((id) => view!.nodes.find((n) => n.id === id)!).filter(Boolean), [view]);
  const maxUnit = Math.max(1e-9, ...(result?.attr.top_units ?? []).map((u) => Math.abs(u.value)));

  return (
    <>
      <TopBar run={run} tab="trace" node={selected}>
        <button type="button" className="btn solid lbl" disabled={busy} onClick={() => runIt()}>{busy ? "Running…" : "Run attribution"}</button>
      </TopBar>
      <div style={{ height: 56, flexShrink: 0, display: "flex", alignItems: "center", gap: 24, padding: "0 20px", borderBottom: "1px solid var(--line)", flexWrap: "wrap" }}>
        <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="lbl dim">Target</span>
          <select className="field" value={target ?? ""} onChange={(e) => setTarget(e.target.value === "" ? null : Number(e.target.value))}>
            {run.output.map((o) => (
              <option key={o.index} value={o.index}>logit[{o.index}]{o.label ? ` · ${o.label}` : ""} · p {o.prob.toFixed(2)}</option>
            ))}
            <option value="">top prediction</option>
          </select>
        </label>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="lbl dim">Method</span>
          <div className="seg lbl" role="group" aria-label="Attribution method">
            {METHODS.map((m) => <button key={m.id} type="button" aria-pressed={m.id === method} onClick={() => setMethod(m.id)}>{m.label}</button>)}
          </div>
        </div>
        {method === "ig" && (
          <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="lbl dim">Steps</span>
            <input className="field" type="number" min={1} max={512} value={steps} onChange={(e) => setSteps(Number(e.target.value) || 32)} style={{ width: 72 }} />
          </label>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="lbl dim">Depth</span>
          <div className="seg lbl" role="group" aria-label="Graph depth">
            {[0, 1, 2, 3].map((d) => (
              <button key={d} type="button" aria-pressed={d === depth} onClick={() => { setDepth(d); if (result) runIt(d); }}>{d}</button>
            ))}
          </div>
        </div>
        <div style={{ flexGrow: 1 }} />
        {result?.attr.input?.convergence_delta != null && (
          <span className="lbl dim">Completeness Δ <span style={{ color: "var(--ink)" }}>{num(result.attr.input.convergence_delta, 4)}</span></span>
        )}
      </div>
      <div style={{ flexGrow: 1, minHeight: 0, display: "flex" }}>
        <aside className="scroll" style={{ width: 400, flexShrink: 0, borderRight: "1px solid var(--line)", padding: "20px 24px", display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}><Crosshair /><span className="lbl dim">Input · {run.input_shape?.join("×")}</span></div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
            <span className="num" style={{ fontSize: 26, fontWeight: 500, letterSpacing: "-0.03em" }}>SALIENCY</span>
            <span className="lbl">{result?.attr.method_label ?? "—"}</span>
          </div>
          {result?.attr.input && <OverlayControls run={run} on={overlay} onOn={setOverlay} alpha={alpha} onAlpha={setAlpha} />}
          <div className="rule-top" style={{ paddingTop: 12, borderBottom: "1px solid var(--line)" }}>
            {result?.attr.input
              ? overlay && run.preview
                ? <InputOverlay run={run} grid={result.attr.input.map} signed alpha={alpha} label="Input attribution over input"
                    exportName={fileName([run.id, result.attr.method, "overlay"], "png")}
                    readout={(r, c, v) => `cell (${r}, ${c}) · rel ${num(v, 3)}`} />
                : <DotMatrix grid={result.attr.input.map} pitch={14} signed label="Input attribution"
                    exportName={fileName([run.id, result.attr.method, "attr"], "png")} />
              : <div className="empty lbl" style={{ padding: "60px 0", textAlign: "center" }}>{busy ? "Computing…" : "Run attribution to see which inputs drove the target"}</div>}
          </div>
          <div className="lbl dim" style={{ display: "flex", gap: 16 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent)" }} />Supports</span>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 8, height: 8, borderRadius: "50%", border: "1px solid var(--negative)" }} />Opposes</span>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 8, height: 8, borderRadius: "50%", border: "1px solid var(--ink)" }} />Weak</span>
          </div>
          {result && (
            <div className="rule-top">
              {[
                ["Target", `logit[${result.attr.target}]${result.attr.target_label ? ` · ${result.attr.target_label}` : ""}`],
                ["Probability", result.attr.prob.toFixed(3)],
                ["Logit", num(result.attr.logit, 3)],
                ["Node scores", result.attr.node_method],
              ].map(([k, v]) => <div key={k} className="kv"><span className="lbl dim">{k}</span><span className="mono" style={{ textAlign: "right" }}>{v}</span></div>)}
            </div>
          )}
          {error && <div className="err mono">{error}</div>}
        </aside>
        <section style={{ flexGrow: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
          <div style={{ padding: "16px 20px 10px", display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <h1 style={{ fontSize: 26, whiteSpace: "nowrap" }}>Where does it come from?</h1>
            <span className="lbl dim" style={{ whiteSpace: "nowrap" }}>Edge width = flow · % = share at merges</span>
          </div>
          <div style={{ flexGrow: 1, minHeight: 0, borderTop: "1px solid var(--line)" }}>
            {view
              ? <GraphCanvas view={view} mode="flow" selected={selected} onSelect={(id) => navigate(href(run.id, "trace", { node: id }), true)}
                  onOpen={(id) => navigate(href(run.id, "tensors", { node: view.nodes.find((n) => n.id === id)?.output_op ?? id }))} />
              : <div className="empty lbl dotgrid" style={{ height: "100%" }}>The backward path appears here</div>}
          </div>
        </section>
        <aside className="panel scroll" style={{ width: 360, flexShrink: 0, borderLeft: "1px solid var(--line)", padding: 20, display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}><span className="lbl">Top contributors</span><span className="lbl dim">Channels → target</span></div>
            <div className="rule-top">
              {(result?.attr.top_units ?? []).map((u, i) => {
                const n = byOp.get(u.op);
                const w = (Math.abs(u.value) / maxUnit) * 50;
                return (
                  <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr 90px 60px", gap: 10, alignItems: "center", padding: "8px 0", borderBottom: "1px solid var(--line-soft)" }}>
                    <span style={{ display: "flex", flexDirection: "column", gap: 1, minWidth: 0 }}>
                      <a className="mono" href={href(run.id, "tensors", { node: n?.id ?? u.op, c: String(u.channel) })} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{u.path ?? (n ? displayName(n) : u.op)}</a>
                      <span className="lbl dim">ch {u.channel}</span>
                    </span>
                    <span style={{ height: 8, position: "relative", borderLeft: "1px solid var(--line)", borderRight: "1px solid var(--line)" }}>
                      <span style={{ position: "absolute", left: "50%", top: -2, bottom: -2, borderLeft: "1px solid var(--ink)" }} />
                      <span style={{ position: "absolute", top: 0, bottom: 0, [u.value >= 0 ? "left" : "right"]: "50%", width: `${w}%`, background: u.value >= 0 ? "var(--accent)" : "var(--negative)" }} />
                    </span>
                    <span className="num" style={{ fontSize: 12, textAlign: "right", color: u.value >= 0 ? "var(--accent)" : "var(--negative-text)" }}>{signed(u.value)}</span>
                  </div>
                );
              })}
              {!result && <div className="kv"><span className="lbl dim">No attribution yet</span></div>}
            </div>
          </div>
          {pathNodes.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}><span className="lbl">Dominant path</span><span className="lbl dim">{pathNodes.length} hops</span></div>
              <div className="rule-top">
                {pathNodes.map((n) => (
                  <a key={n.id} href={href(run.id, "trace", { node: n.id })} className="kv" style={{ color: n.id === selected ? "var(--accent)" : undefined }}>
                    <span className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{displayName(n)}</span>
                    <span className="num">{pct(n.score_norm)}</span>
                  </a>
                ))}
              </div>
            </div>
          )}
        </aside>
      </div>
    </>
  );
}
