import { useMemo } from "react";
import { DivergingBars, DotMatrix, HeatCanvas, Histogram, LinePlot, Scatter } from "../components/charts";
import { Crosshair } from "../components/Icons";
import { TopBar } from "../components/TopBar";
import { api } from "../lib/api";
import { num, shape } from "../lib/format";
import { href, navigate, type Route } from "../lib/router";
import type { GraphNode, ImageTensorView, RunMeta, SequenceTensorView, TensorView, VectorTensorView } from "../lib/types";
import { useAsync } from "../lib/useAsync";

export function TensorViewPage({ run, route }: { run: RunMeta; route: Route }) {
  const graph = useAsync(() => api.graph(run.id, 12), [run.id]);
  const nodes = useMemo(() => (graph.data?.nodes ?? []).filter((n) => n.stats), [graph.data]);
  const nodeParam = route.params.get("node");
  const node: GraphNode | null = useMemo(
    () => nodes.find((n) => n.id === nodeParam) ?? nodes.find((n) => n.output_op === nodeParam)
      ?? nodes.find((n) => n.shape.length === 4 && !n.is_input) ?? nodes[0] ?? null,
    [nodes, nodeParam]);
  const chParam = route.params.get("c");
  const channel = chParam != null ? Number(chParam) : null;
  const tensor = useAsync(() => (node ? api.tensor(run.id, node.output_op, { channel }) : null), [run.id, node?.output_op, channel]);
  const t = tensor.data;

  const go = (p: Record<string, string | null>) =>
    navigate(href(run.id, "tensors", { node: node?.id ?? null, c: chParam, ...p }), true);

  return (
    <>
      <TopBar run={run} tab="tensors" node={node?.id} />
      <div style={{ flexGrow: 1, minHeight: 0, display: "flex" }}>
        <aside style={{ width: 300, flexShrink: 0, borderRight: "1px solid var(--line)", padding: 20, display: "flex", flexDirection: "column", gap: 20 }} className="scroll">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}><Crosshair /><span className="lbl dim">Tensor · output</span></div>
            <label className="lbl dim" htmlFor="node-pick">Node</label>
            <select id="node-pick" className="field" value={node?.id ?? ""} onChange={(e) => go({ node: e.target.value, c: null })}>
              {nodes.map((n) => <option key={n.id} value={n.id}>{n.path} · {n.type}</option>)}
            </select>
            {node && <span className="lbl dim">{node.type} · {node.output_op}</span>}
          </div>
          {t && (
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${t.shape.length}, minmax(0, 1fr))`, border: "1px solid var(--ink)" }}>
              {t.shape.map((d, i) => {
                const letters = t.shape[0] === 1 ? ({ 4: "BCHW", 3: "BTD", 2: "BD" } as Record<number, string>)[t.shape.length] : undefined;
                const axis = letters ? letters[i] : `d${i}`;
                const hot = t.kind === "image" && axis === "C";
                return (
                  <div key={i} style={{ padding: 8, display: "flex", flexDirection: "column", gap: 2, borderLeft: i ? "1px solid var(--ink)" : 0, background: hot ? "var(--accent-soft)" : undefined }}>
                    <span className="lbl" style={{ color: hot ? "var(--accent)" : "var(--ink-2)" }}>{axis}</span>
                    <span className="num" style={{ fontSize: 16, color: hot ? "var(--accent)" : undefined }}>{d}</span>
                  </div>
                );
              })}
            </div>
          )}
          {t?.kind === "image" && (
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="lbl dim">Channel (0–{t.channels - 1})</span>
              <input className="field" type="number" min={0} max={t.channels - 1} value={t.channel}
                onChange={(e) => go({ c: e.target.value })} />
            </label>
          )}
          {t && (
            <div className="rule-top">
              {[["View", t.kind], ["Scope", t.kind === "image" ? `channel c${t.channel}` : "whole tensor"], ["Mean", num(t.stats.mean)], ["Std", num(t.stats.std)], ["Min", num(t.stats.min)], ["Max", num(t.stats.max)]].map(([k, v]) => (
                <div key={k} className="kv"><span className="lbl dim">{k}</span><span className="num">{v}</span></div>
              ))}
            </div>
          )}
          <span className="lbl dim" style={{ marginTop: "auto" }}>Full shape {shape(t?.shape)}</span>
        </aside>
        {tensor.error && <div className="empty err">{tensor.error}</div>}
        {!t && !tensor.error && <div className="empty lbl">{tensor.loading ? "Loading tensor…" : "No tensor saved for this node"}</div>}
        {t && <TensorBody t={t} onChannel={(c) => go({ c: String(c) })} />}
      </div>
    </>
  );
}

function TensorBody({ t, onChannel }: { t: TensorView; onChannel: (c: number) => void }) {
  if (t.kind === "image") return <ImageBody t={t} onChannel={onChannel} />;
  if (t.kind === "sequence") return <SequenceBody t={t} />;
  return <VectorBody t={t} />;
}

const card: React.CSSProperties = { border: "1px solid var(--line)", background: "var(--panel)", padding: 14, display: "flex", flexDirection: "column", gap: 10, minWidth: 0 };

function Head({ left, right }: { left: string; right?: string }) {
  return <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}><span className="lbl">{left}</span>{right && <span className="lbl dim">{right}</span>}</div>;
}

function ImageBody({ t, onChannel }: { t: ImageTensorView; onChannel: (c: number) => void }) {
  return (
    <>
      <section className="scroll" style={{ flexGrow: 1, minWidth: 0, padding: "20px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <h1 style={{ fontSize: 26 }}>Channels</h1>
          <span className="lbl dim">Top {t.maps.length} of {t.channels} · sorted by mean · click to inspect</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(72px, 1fr))", gap: 10 }}>
          {t.maps.map((m) => {
            const on = m.channel === t.channel;
            return (
              <button key={m.channel} type="button" onClick={() => onChannel(m.channel)} aria-pressed={on}
                aria-label={`Channel ${m.channel}, mean ${num(m.mean)}`}
                style={{ padding: 0, border: 0, background: "transparent", cursor: "pointer", display: "flex", flexDirection: "column", gap: 4, textAlign: "left" }}>
                <span style={{ display: "block", border: on ? "2px solid var(--accent)" : "1px solid var(--line)" }}>
                  <HeatCanvas grid={m.map} style={{ aspectRatio: "1" }} label={`Channel ${m.channel} map`} />
                </span>
                <span style={{ display: "flex", justifyContent: "space-between" }}>
                  <span className="lbl" style={{ color: on ? "var(--accent)" : undefined }}>c{m.channel}</span>
                  <span className="lbl dim">{num(m.mean)}</span>
                </span>
              </button>
            );
          })}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 16 }}>
          <div style={card}>
            <Head left={`Channel vector @ (${t.vector.pixel.join(", ")})`} right={`1D · ${t.vector.values.length}`} />
            <DivergingBars values={t.vector.values} highlight={t.channel} onPick={onChannel} />
          </div>
          <div style={card}>
            <Head left={`PCA · ${t.height * t.width} positions`} right={`PC1 ${Math.round(t.pca.explained[0] * 100)}% · PC2 ${Math.round((t.pca.explained[1] ?? 0) * 100)}%`} />
            <Scatter points={t.pca.points} color={t.pca.color} height={150} />
          </div>
        </div>
      </section>
      <aside className="panel scroll" style={{ width: 420, flexShrink: 0, borderLeft: "1px solid var(--line)", padding: 20, display: "flex", flexDirection: "column", gap: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
          <span className="num" style={{ fontSize: 28, fontWeight: 500, letterSpacing: "-0.03em" }}>c{t.channel}</span>
          <span className="lbl dim">{t.height}×{t.width} → {t.dots.length}×{t.dots[0]?.length ?? 0} pooled</span>
        </div>
        <div className="rule-top" style={{ paddingTop: 10, borderBottom: "1px solid var(--line)" }}>
          <DotMatrix grid={t.dots} pitch={Math.min(22, Math.floor(360 / Math.max(1, t.dots[0]?.length ?? 1)))} label={`Channel ${t.channel} dot matrix`} />
        </div>
        <div className="lbl dim" style={{ display: "flex", justifyContent: "space-between" }}>
          <span>○ low</span><span>● high</span><span>Rank {t.channel_rank} / {t.channels}</span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Head left={`Row ${t.profile.row} profile`} right={`1D · ${t.profile.values.length}`} />
          <LinePlot values={t.profile.values} xLabel="col" />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Head left="Value distribution" right={`${t.hist.counts.length} bins`} />
          <Histogram counts={t.hist.counts} range={t.hist.range} height={70} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 8 }}>
          {[["Mean", num(t.stats.mean)], ["Std", num(t.stats.std)], ["Peak", `(${t.peak.join(",")})`], ["Max", num(t.stats.max)]].map(([k, v]) => (
            <div key={k} style={{ display: "flex", flexDirection: "column", gap: 2 }}><span className="lbl dim">{k}</span><span className="num">{v}</span></div>
          ))}
        </div>
      </aside>
    </>
  );
}

function SequenceBody({ t }: { t: SequenceTensorView }) {
  return (
    <>
      <section className="scroll" style={{ flexGrow: 1, minWidth: 0, padding: "20px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <h1 style={{ fontSize: 26 }}>Tokens × features</h1>
          <span className="lbl dim">{t.tokens} tokens · {t.features} features · signed, max-abs scaled</span>
        </div>
        <div style={{ border: "1px solid var(--line)" }}>
          <HeatCanvas grid={t.heatmap} signed style={{ height: 360, imageRendering: "pixelated" }} label="Token by feature heatmap" />
        </div>
        <div style={card}>
          <Head left="Per-token norm" right={`1D · ${t.token_norms.length}`} />
          <LinePlot values={t.token_norms} xLabel="token" />
        </div>
      </section>
      <aside className="panel scroll" style={{ width: 420, flexShrink: 0, borderLeft: "1px solid var(--line)", padding: 20, display: "flex", flexDirection: "column", gap: 18 }}>
        <Head left={`PCA · ${t.tokens} tokens`} right={`PC1 ${Math.round(t.pca.explained[0] * 100)}%`} />
        <Scatter points={t.pca.points} color={t.pca.color} />
        <Head left="Value distribution" />
        <Histogram counts={t.hist.counts} range={t.hist.range} height={70} />
      </aside>
    </>
  );
}

function VectorBody({ t }: { t: VectorTensorView }) {
  return (
    <>
      <section className="scroll" style={{ flexGrow: 1, minWidth: 0, padding: "20px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <h1 style={{ fontSize: 26 }}>Signal</h1>
          <span className="lbl dim">{t.length} values{t.values.length < t.length ? ` · pooled to ${t.values.length}` : ""}</span>
        </div>
        <div style={card}><DivergingBars values={t.values} height={200} xLabel="index" /></div>
      </section>
      <aside className="panel scroll" style={{ width: 420, flexShrink: 0, borderLeft: "1px solid var(--line)", padding: 20, display: "flex", flexDirection: "column", gap: 18 }}>
        <Head left="Largest |value|" />
        <div className="rule-top">
          {t.top.map((r) => (
            <div key={r.index} className="kv"><span className="mono">[{r.index}]</span>
              <span className="num" style={{ color: r.value >= 0 ? "var(--ink)" : "var(--negative-text)" }}>{num(r.value, 3)}</span></div>
          ))}
        </div>
        <Head left="Value distribution" />
        <Histogram counts={t.hist.counts} range={t.hist.range} height={70} />
      </aside>
    </>
  );
}
