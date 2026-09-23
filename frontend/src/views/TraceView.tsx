import { useEffect, useMemo, useRef, useState } from "react";
import { AgreementMatrix, type AgreementMetric } from "../components/AgreementMatrix";
import { DivergingBars, DotMatrix } from "../components/charts";
import { GraphCanvas } from "../components/GraphCanvas";
import { Crosshair } from "../components/Icons";
import { InputOverlay, OverlayControls } from "../components/InputOverlay";
import { JobProgress } from "../components/JobProgress";
import { TopBar } from "../components/TopBar";
import { api } from "../lib/api";
import { fileName } from "../lib/exportPng";
import { displayName, num, pct, signed } from "../lib/format";
import { useJob } from "../lib/jobs";
import { href, navigate, type Route } from "../lib/router";
import { errText } from "../lib/tensorState";
import { attrRequest, layerOp, parseTraceParams, traceParams, type TraceState } from "../lib/traceState";
import type { AgreementResult, Attribution, GraphView, Job, LayerMethod, LayerResult, Method, NeuronResult, RunMeta } from "../lib/types";

const METHODS: { id: Method; label: string }[] = [
  { id: "gxa", label: "Grad × input" },
  { id: "saliency", label: "Saliency" },
  { id: "ig", label: "Integrated grad" },
];
const LAYERS: { id: LayerMethod; label: string }[] = [{ id: "layer_ig", label: "Layer IG" }, { id: "layer_cond", label: "Conductance" }];
const NEURON_LABEL = "Neuron conductance (Σ over positions)";
const ELLIPSIS: React.CSSProperties = { minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" };

type AttrReq = ReturnType<typeof attrRequest>;
/** The objective part shared by every job request: target (null with a unit objective), objective, steps. */
const ctx = (st: TraceState) => {
  const { target, objective, steps } = attrRequest(st);
  return { target, objective, steps };
};

const Kv = ({ rows }: { rows: [string, string][] }) => (
  <>{rows.map(([k, v]) => <div key={k} className="kv"><span className="lbl dim">{k}</span><span className="mono" style={{ textAlign: "right" }}>{v}</span></div>)}</>
);

/** A POST failure (no job to carry it); a failed job shows its error in JobProgress. */
const JobError = ({ j }: { j: { job: Job<unknown> | null; error: string | null } }) =>
  j.error && j.job?.status !== "error" ? <div className="err mono">{j.error}</div> : null;

/** Signed contribution row: name + dim sub label, a bar around a centre rule, the signed value. */
function BarRow({ name, sub, to, value, max }: { name: string; sub: string; to: string; value: number; max: number }) {
  const w = (Math.abs(value) / Math.max(1e-9, max)) * 50;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 90px 60px", gap: 10, alignItems: "center", padding: "8px 0", borderBottom: "1px solid var(--line-soft)" }}>
      <span style={{ display: "flex", flexDirection: "column", gap: 1, minWidth: 0 }}>
        <a className="mono" href={to} style={ELLIPSIS}>{name}</a>
        <span className="lbl dim">{sub}</span>
      </span>
      <span style={{ height: 8, position: "relative", borderLeft: "1px solid var(--line)", borderRight: "1px solid var(--line)" }}>
        <span style={{ position: "absolute", left: "50%", top: -2, bottom: -2, borderLeft: "1px solid var(--ink)" }} />
        <span style={{ position: "absolute", top: 0, bottom: 0, [value >= 0 ? "left" : "right"]: "50%", width: `${w}%`, background: value >= 0 ? "var(--accent)" : "var(--negative)" }} />
      </span>
      <span className="num" style={{ fontSize: 12, textAlign: "right", color: value >= 0 ? "var(--accent)" : "var(--negative-text)" }}>{signed(value)}</span>
    </div>
  );
}

export function TraceViewPage({ run, route }: { run: RunMeta; route: Route }) {
  const s = parseTraceParams(route.params);
  const latest = useRef(s);
  latest.current = s;
  const set = (p: Partial<TraceState>) => navigate(href(run.id, "trace", traceParams({ ...latest.current, go: false, ...p })), true);

  const [depth, setDepth] = useState(1);
  const [overlay, setOverlay] = useState(false);
  const [alpha, setAlpha] = useState(0.6);
  const [metric, setMetric] = useState<AgreementMetric>("jaccard");
  const [pick, setPick] = useState<LayerMethod>(s.layer ?? "layer_ig");
  const [nonce, setNonce] = useState(0);
  const [res, setRes] = useState<{ attr: Attribution; req: AttrReq } | null>(null);
  const [view, setView] = useState<GraphView | null>(null);
  const [graphErr, setGraphErr] = useState<string | null>(null);
  const main = useJob<Attribution>();
  const layer = useJob<LayerResult>();
  const neuron = useJob<NeuronResult>();
  const agree = useJob<AgreementResult>();

  const loadGraph = (req: AttrReq, d: number) =>
    api.graph(run.id, d, { method: req.method, target: req.target, steps: req.steps, objective: req.objective })
      .then((v) => { setView(v); setGraphErr(null); }, (e) => setGraphErr(errText(e)));

  const runMain = async () => {
    const req = attrRequest(latest.current);
    const attr = await main.run(run.id, req);
    if (!attr) return false;
    setRes({ attr, req });
    await loadGraph(req, depth);
    return true;
  };

  const { target, objective, steps } = ctx(s);
  const objKey = JSON.stringify([target, objective, steps]);
  const layerReq = (st: TraceState) => st.layer && st.lop
    ? { kind: "layer" as const, method: st.layer, op: st.lop, ...ctx(st) } : null;
  const neuronReq = (st: TraceState) => st.nc != null && st.lop
    ? { kind: "neuron" as const, op: st.lop, channel: st.nc, ...ctx(st) } : null;

  // A change to layer, lop, the objective or steps re-runs the open layer panel / neuron map (not during a go deep link).
  useEffect(() => {
    const r = layerReq(s);
    if (!s.go && r) void layer.run(run.id, r);
  }, [s.layer, s.lop, objKey, nonce]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const r = neuronReq(s);
    if (s.nc == null) neuron.cancel();
    else if (!s.go && r) void neuron.run(run.id, r);
  }, [s.lop, s.nc, objKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // go=1 (on arrival, or a link while the view is open): main → layer → neuron once, then drop go from the URL.
  useEffect(() => {
    if (!s.go) return;
    let alive = true;
    (async () => {
      const ok = await runMain();
      const l = layerReq(s), n = neuronReq(s);
      if (ok && alive && l) await layer.run(run.id, l);
      if (ok && alive && n) await neuron.run(run.id, n);
      if (alive && latest.current.go) navigate(href(run.id, "trace", traceParams({ ...latest.current, go: false })), true);
    })();
    return () => { alive = false; };
  }, [s.go]); // eslint-disable-line react-hooks/exhaustive-deps

  const attr = res?.attr ?? null;
  const byOp = useMemo(() => new Map((view?.nodes ?? []).map((n) => [n.output_op, n])), [view]);
  const pathNodes = useMemo(() => (view?.path ?? []).map((id) => view!.nodes.find((n) => n.id === id)!).filter(Boolean), [view]);
  const maxUnit = Math.max(1e-9, ...(attr?.top_units ?? []).map((u) => Math.abs(u.value)));
  const selNode = view?.nodes.find((n) => n.id === s.node) ?? null;
  const lop = layerOp(s.lop ?? selNode?.output_op ?? (s.node && !s.node.startsWith("m:") ? s.node : null), s.obj);   // an op id from the Tensor view
  const lopNode = lop ? byOp.get(lop) ?? null : null;
  const lr = layer.job?.status === "done" && layer.job.result?.op === lop ? layer.job.result : null;
  const nr = s.nc != null && neuron.job?.status === "done" ? neuron.job.result : null;
  const ar = agree.job?.status === "done" ? agree.job.result : null;
  const approx = !!view?.edges.some((e) => e.share_method === "parent");
  const methodLabel = attr?.method_label ?? METHODS.find((m) => m.id === s.m)!.label;
  const objLabel = !s.obj ? null
    : attr?.objective.kind === "unit" && attr.objective.op === s.obj.op && attr.objective.channel === s.obj.channel
      ? attr.objective_label : `${s.obj.op} · ch ${s.obj.channel} · Σ spatial`;
  const neuronOn = s.nc != null;
  const map = neuronOn ? nr?.input.map : attr?.input?.map;
  const mapName = neuronOn ? fileName([run.id, "neuron", s.lop, `c${s.nc}`], "png") : fileName([run.id, attr?.method, "attr"], "png");
  const layerMethod = s.layer ?? pick;

  return (
    <>
      <TopBar run={run} tab="trace" node={s.node}>
        <button type="button" className="btn solid lbl" disabled={main.busy} onClick={() => void runMain()}>Run attribution</button>
      </TopBar>
      <div style={{ minHeight: 56, flexShrink: 0, display: "flex", alignItems: "center", gap: 24, padding: "8px 20px", borderBottom: "1px solid var(--line)", flexWrap: "wrap" }}>
        {s.obj ? (
          <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
            <span className="lbl dim">Objective</span>
            <span className="pill lbl" style={ELLIPSIS} title={objLabel ?? undefined}>Unit · {objLabel}</span>
            <button type="button" className="chart-export lbl" onClick={() => set({ obj: null, go: true })}>← Logits</button>
          </div>
        ) : (
          <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="lbl dim">Target</span>
            <select className="field" value={s.t ?? ""} onChange={(e) => set({ t: e.target.value === "" ? null : Number(e.target.value) })}>
              {run.output.map((o) => (
                <option key={o.index} value={o.index}>logit[{o.index}]{o.label ? ` · ${o.label}` : ""} · p {o.prob.toFixed(2)}</option>
              ))}
              {s.t != null && !run.output.some((o) => o.index === s.t) && <option value={s.t}>logit[{s.t}]</option>}
              <option value="">top prediction</option>
            </select>
          </label>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="lbl dim">Method</span>
          <div className="seg lbl" role="group" aria-label="Attribution method">
            {METHODS.map((m) => <button key={m.id} type="button" aria-pressed={m.id === s.m} onClick={() => set({ m: m.id })}>{m.label}</button>)}
          </div>
        </div>
        {(s.m === "ig" || s.layer || neuronOn) && (
          <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="lbl dim">Steps</span>
            <input className="field" type="number" min={1} max={512} value={s.steps} onChange={(e) => set({ steps: Number(e.target.value) || 32 })} style={{ width: 72 }} />
          </label>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="lbl dim">Depth</span>
          <div className="seg lbl" role="group" aria-label="Graph depth">
            {[0, 1, 2, 3].map((d) => (
              <button key={d} type="button" aria-pressed={d === depth} onClick={() => { setDepth(d); if (res) void loadGraph(res.req, d); }}>{d}</button>
            ))}
          </div>
        </div>
        <div style={{ flexGrow: 1 }} />
        {main.job && main.job.status !== "done" && <div style={{ width: 240 }}><JobProgress job={main.job} onCancel={main.cancel} /></div>}
        {attr?.input?.convergence_delta != null && (
          <span className="lbl dim">Completeness Δ <span style={{ color: "var(--ink)" }}>{num(attr.input.convergence_delta, 4)}</span></span>
        )}
      </div>
      <div style={{ flexGrow: 1, minHeight: 0, display: "flex" }}>
        <aside className="scroll" style={{ width: 400, flexShrink: 0, borderRight: "1px solid var(--line)", padding: "20px 24px", display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}><Crosshair /><span className="lbl dim">Input · {run.input_shape?.join("×")}</span></div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 12 }}>
            <span className="num" style={{ fontSize: 26, fontWeight: 500, letterSpacing: "-0.03em" }}>{neuronOn ? "NEURON" : "SALIENCY"}</span>
            <span className="lbl" style={{ textAlign: "right" }}>
              {neuronOn ? `${nr?.method_label ?? NEURON_LABEL} · ${nr?.path ?? s.lop ?? "—"} · ch ${s.nc}` : attr?.method_label ?? "—"}
            </span>
          </div>
          {neuronOn && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button type="button" className="chart-export lbl" style={{ alignSelf: "flex-start" }} onClick={() => set({ nc: null })}>← {methodLabel}</button>
              <JobProgress job={neuron.job} onCancel={neuron.cancel} />
              <JobError j={neuron} />
            </div>
          )}
          {map && <OverlayControls run={run} on={overlay} onOn={setOverlay} alpha={alpha} onAlpha={setAlpha} />}
          <div className="rule-top" style={{ paddingTop: 12, borderBottom: "1px solid var(--line)" }}>
            {map
              ? overlay && run.preview
                ? <InputOverlay run={run} grid={map} signed alpha={alpha} label={neuronOn ? "Neuron conductance over input" : "Input attribution over input"}
                    exportName={mapName.replace(/\.png$/, "_overlay.png")} readout={(r, c, v) => `cell (${r}, ${c}) · rel ${num(v, 3)}`} />
                : <DotMatrix grid={map} pitch={14} signed label={neuronOn ? "Neuron conductance" : "Input attribution"} exportName={mapName} />
              : <div className="empty lbl" style={{ padding: "60px 0", textAlign: "center" }}>
                  {(neuronOn ? neuron.busy : main.busy) ? "Computing…" : neuronOn ? "No map" : "Run attribution to see which inputs drove the objective"}
                </div>}
          </div>
          <div className="lbl dim" style={{ display: "flex", gap: 16 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent)" }} />Supports</span>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 8, height: 8, borderRadius: "50%", border: "1px solid var(--negative)" }} />Opposes</span>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 8, height: 8, borderRadius: "50%", border: "1px solid var(--ink)" }} />Weak</span>
          </div>
          {attr && (
            <div className="rule-top">
              <Kv rows={[
                ["Objective", attr.objective_label],
                ["Value", num(attr.value, 3)],
                ...(attr.prob != null ? [["Probability", attr.prob.toFixed(3)] as [string, string]] : []),
                ["Node scores", attr.node_method],
                ["Merge shares", attr.merge_method],
              ]} />
            </div>
          )}
          <JobError j={main} />
          {graphErr && <div className="err mono">{graphErr}</div>}
          <div style={{ display: "flex", flexDirection: "column", gap: 10, paddingTop: 6 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span className="lbl">Method agreement</span>
              <div className="seg lbl" role="group" aria-label="Agreement metric">
                <button type="button" aria-pressed={metric === "jaccard"} onClick={() => setMetric("jaccard")}>Jaccard</button>
                <button type="button" aria-pressed={metric === "spearman"} onClick={() => setMetric("spearman")}>Spearman</button>
              </div>
            </div>
            <button type="button" className="btn lbl" disabled={agree.busy}
              onClick={() => void agree.run(run.id, { kind: "agreement", op: lop, target, objective, steps })}>Compare methods</button>
            <JobProgress job={agree.job} onCancel={agree.cancel} />
            <JobError j={agree} />
            {ar && <span className="lbl dim">Objective · {ar.objective_label} · {ar.steps} steps</span>}
            {ar?.input && <AgreementMatrix block={ar.input} metric={metric} exportName={fileName([run.id, "agreement", "input"], "png")} />}
            {ar?.units && <AgreementMatrix block={ar.units} metric={metric} exportName={fileName([run.id, "agreement", ar.units.op], "png")} />}
            {ar?.units_error && <div className="mono dim">{ar.units_error}</div>}
          </div>
        </aside>
        <section style={{ flexGrow: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
          <div style={{ padding: "16px 20px 10px", display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 16 }}>
            <h1 style={{ fontSize: 26, whiteSpace: "nowrap" }}>Where does it come from?</h1>
            <span className="lbl dim" style={{ textAlign: "right" }}>
              Edge width = flow · % = share of merge output (grad × input){approx ? " · some merges approximate (parent g×a)" : ""}
            </span>
          </div>
          <div style={{ flexGrow: 1, minHeight: 0, borderTop: "1px solid var(--line)" }}>
            {view
              ? <GraphCanvas view={view} mode="flow" selected={s.node} onSelect={(id) => set({ node: id, ...(s.lop && { lop: view.nodes.find((n) => n.id === id)?.output_op ?? s.lop, nc: null }) })}
                  onOpen={(id) => navigate(href(run.id, "tensors", { node: view.nodes.find((n) => n.id === id)?.output_op ?? id }))} />
              : <div className="empty lbl dotgrid" style={{ height: "100%" }}>The backward path appears here</div>}
          </div>
        </section>
        <aside className="panel scroll" style={{ width: 360, flexShrink: 0, borderLeft: "1px solid var(--line)", padding: 20, display: "flex", flexDirection: "column", gap: 18 }}>
          {!lop && s.obj && <span className="lbl dim">Layer attribution · pick a node before the unit</span>}
          {lop && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                <span className="lbl" style={{ whiteSpace: "nowrap" }}>Layer attribution</span>
                <span className="lbl dim" style={ELLIPSIS}>{lr?.path ?? (lopNode ? displayName(lopNode) : lop)}</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div className="seg lbl" role="group" aria-label="Layer method">
                  {LAYERS.map((m) => (
                    <button key={m.id} type="button" aria-pressed={m.id === layerMethod}
                      onClick={() => (s.layer ? set({ layer: m.id }) : setPick(m.id))}>{m.label}</button>
                  ))}
                </div>
                <button type="button" className="btn lbl" onClick={() => { set({ layer: layerMethod, lop }); setNonce((k) => k + 1); }}>Run</button>
              </div>
              <JobProgress job={layer.job} onCancel={layer.cancel} />
              <JobError j={layer} />
              {lr && (
                <>
                  <DivergingBars values={lr.channels} highlight={s.lop === lr.op ? s.nc : null} xLabel="channel"
                    onPick={(c) => set({ nc: c, lop: lr.op })} exportName={fileName([run.id, lr.method, lr.op], "png")} />
                  <div className="rule-top">
                    <Kv rows={[
                      ["Method", lr.method_label],
                      ["Σ", num(lr.total, 4)],
                      ["Completeness Δ", num(lr.convergence_delta, 4)],
                      ["Objective", lr.objective_label],
                    ]} />
                    {lr.top.slice(0, 5).map((u) => (
                      <BarRow key={u.channel} name={`ch ${u.channel}`} sub={lr.path} value={u.value}
                        max={Math.max(1e-9, ...lr.top.slice(0, 5).map((t) => Math.abs(t.value)))}
                        to={href(run.id, "tensors", { node: lopNode?.id ?? lr.op, c: String(u.channel) })} />
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}><span className="lbl">Top contributors</span><span className="lbl dim">Channels → objective</span></div>
            <div className="rule-top">
              {(attr?.top_units ?? []).map((u, i) => {
                const n = byOp.get(u.op);
                return <BarRow key={i} name={u.path ?? (n ? displayName(n) : u.op)} sub={`ch ${u.channel}`} value={u.value} max={maxUnit}
                  to={href(run.id, "tensors", { node: n?.id ?? u.op, c: String(u.channel) })} />;
              })}
              {!attr && <div className="kv"><span className="lbl dim">No attribution yet</span></div>}
            </div>
          </div>
          {pathNodes.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}><span className="lbl">Dominant path</span><span className="lbl dim">{pathNodes.length} hops</span></div>
              <div className="rule-top">
                {pathNodes.map((n) => (
                  <a key={n.id} href={href(run.id, "trace", traceParams({ ...s, node: n.id, go: false }))} className="kv" style={{ color: n.id === s.node ? "var(--accent)" : undefined }}>
                    <span className="mono" style={ELLIPSIS}>{displayName(n)}</span>
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
