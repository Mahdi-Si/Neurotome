import { useMemo, useRef, useState } from "react";
import { CaptureBanner } from "../components/CaptureBanner";
import { DivergingBars, DotMatrix, HeatCanvas, Histogram, LinePlot, Scatter } from "../components/charts";
import { ComparePanel, type CompareSide } from "../components/ComparePanel";
import { Crosshair } from "../components/Icons";
import { InputOverlay, OverlayControls } from "../components/InputOverlay";
import { InterventionBanner } from "../components/InterventionBanner";
import { TopBar } from "../components/TopBar";
import { ValuesTable } from "../components/ValuesTable";
import { api } from "../lib/api";
import { axisLetters, headReadout, patternLabel } from "../lib/attentionState";
import { hasTensor, unsavedText } from "../lib/capture";
import { exportTiles, fileName } from "../lib/exportPng";
import { num, shape } from "../lib/format";
import { href, navigate, type Route } from "../lib/router";
import { channelIsAxis1, traceParams } from "../lib/traceState";
import {
  cellOf, cellPixel, cellSpan, defaultSlice, parentNode, parseTensorParams, tensorParams, unscale, type TensorState,
} from "../lib/tensorState";
import { showToken, spanLabel, tokenLabels } from "../lib/tokens";
import type {
  AttentionTensorView, Cmap, GraphNode, ImageTensorView, Norm, RunMeta, SequenceTensorView, TensorView, VectorTensorView,
} from "../lib/types";
import { useAsync } from "../lib/useAsync";

const toInt = (v: string) => (/^\d+$/.test(v) ? Number(v) : null);
const textBtn = "lbl chart-export";

export function TensorViewPage({ run, route }: { run: RunMeta; route: Route }) {
  const graph = useAsync(() => api.graph(run.id, 12), [run.id]);
  const nodes = useMemo(() => (graph.data?.nodes ?? []).filter(hasTensor), [graph.data]);
  const s = parseTensorParams(route.params);
  // the URL's node without a saved tensor gets an empty state, never a silent fallback to another node
  const missing = useMemo(() => {
    const all = graph.data?.nodes ?? [];
    const n = all.find((x) => x.id === s.node) ?? all.find((x) => x.output_op === s.node);
    return n && !hasTensor(n) ? n : null;
  }, [graph.data, s.node]);
  const node: GraphNode | null = useMemo(
    () => missing ? null : nodes.find((n) => n.id === s.node) ?? nodes.find((n) => n.output_op === s.node)
      ?? nodes.find((n) => n.shape.length === 4 && !n.is_input) ?? nodes[0] ?? null,
    [nodes, s.node, missing]);
  const cmp = s.cmp ? nodes.find((n) => n.id === s.cmp) ?? null : null;
  const ivRes = useAsync(() => (s.iv ? api.intervention(run.id, s.iv) : null), [run.id, s.iv]);
  const ivResult = ivRes.data?.hash === s.iv ? ivRes.data : null;
  // the graph is the clean one (node list, clean stats); a side under the intervention takes its stats from the result
  const side = (n: GraphNode, iv: string | null, tag: string): CompareSide =>
    ({ node: n, iv, stats: (iv && ivResult?.stats[n.output_op]) || n.stats, tag });
  const tensor = useAsync(
    () => (node ? api.tensor(run.id, node.output_op, { channel: s.c, py: s.py, px: s.px, norm: s.norm, cmap: s.cmap, iv: s.iv })
      .then((v) => [node.output_op, v] as const) : null),
    [run.id, node?.output_op, s.c, s.py, s.px, s.norm, s.cmap, s.iv]);
  // the op travels with its view: while the next node loads, the stale view keeps its own op (the values table fetches by it)
  const [tOp, t] = tensor.data ?? [null, null];

  const go = (p: Partial<TensorState>) =>
    navigate(href(run.id, "tensors", tensorParams({ ...s, node: node?.id ?? null, ...p })), true);
  const toggleCompare = () => {
    if (cmp || !node) return go({ cmp: null });
    const par = graph.data && parentNode(graph.data, node.id);   // only nodes with a saved tensor can be compared
    go({ cmp: nodes.some((n) => n.id === par) ? par : nodes.find((n) => n.id !== node.id)?.id ?? null, vs: null });
  };
  const pixel = t?.kind === "image" ? t.vector.pixel : null;

  return (
    <>
      <TopBar run={run} tab="tensors" node={missing?.id ?? node?.id} carry={{ iv: s.iv }} />
      <CaptureBanner run={run} />
      <InterventionBanner result={ivResult} error={s.iv ? ivRes.error : null}
        clearHref={href(run.id, "tensors", tensorParams({ ...s, node: missing?.id ?? node?.id ?? null, iv: null, vs: null }))} />
      <div style={{ flexGrow: 1, minHeight: 0, display: "flex" }}>
        <aside style={{ width: 300, flexShrink: 0, borderRight: "1px solid var(--line)", padding: 20, display: "flex", flexDirection: "column", gap: 20 }} className="scroll">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}><Crosshair /><span className="lbl dim">Tensor · output</span></div>
            <label className="lbl dim" htmlFor="node-pick">Node</label>
            <select id="node-pick" className="field" value={missing?.id ?? node?.id ?? ""} onChange={(e) => go({ node: e.target.value, c: null, py: null, px: null, cmp: null })}>
              {missing && <option value={missing.id} disabled>{missing.path} · no tensor</option>}
              {nodes.map((n) => <option key={n.id} value={n.id}>{n.path} · {n.type}</option>)}
            </select>
            {node && <span className="lbl dim">{node.type} · {node.output_op}</span>}
            <button type="button" className="btn" aria-pressed={!!cmp} onClick={toggleCompare} disabled={nodes.length < 2}>
              {cmp ? "Exit compare" : "Compare"}
            </button>
            {s.iv && (
              <button type="button" className={s.vs ? "btn solid" : "btn"} aria-pressed={!!s.vs} disabled={!node}
                onClick={() => go({ vs: s.vs ? null : "clean", cmp: null })}>Clean vs intervened</button>
            )}
            {cmp && node && (
              <>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <label className="lbl dim" htmlFor="cmp-pick">Compare with</label>
                  <button type="button" className={textBtn} onClick={() => go({ node: cmp.id, cmp: node.id })}>Swap</button>
                </div>
                <select id="cmp-pick" className="field" value={cmp.id} onChange={(e) => go({ cmp: e.target.value })}>
                  {nodes.map((n) => <option key={n.id} value={n.id}>{n.path} · {n.type}</option>)}
                </select>
              </>
            )}
          </div>
          {t && (
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${t.shape.length}, minmax(0, 1fr))`, border: "1px solid var(--ink)" }}>
              {axisLetters(t.shape, t.kind).map((axis, i) => {
                const d = t.shape[i];
                const hot = (t.kind === "image" && axis === "C") || (t.kind === "attention" && axis === "H");
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
            <>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span className="lbl dim">Channel (0–{t.channels - 1})</span>
                <input className="field" type="number" min={0} max={t.channels - 1} value={s.c ?? t.channel}
                  onChange={(e) => { const c = toInt(e.target.value); if (c != null) go({ c }); }} />
              </label>
              {node && channelIsAxis1(t.shape) && (
                <a className="btn lbl" href={href(run.id, "trace", traceParams({ obj: { op: node.output_op, channel: t.channel }, node: node.id, go: true }))}>
                  Trace from c{t.channel}
                </a>
              )}
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span className="lbl dim">Position (y, x)</span>
                  <button type="button" className={textBtn} onClick={() => go({ py: null, px: null })} title="Jump to the channel's argmax">Peak</button>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  <input className="field" type="number" aria-label="Pixel y" min={0} max={t.height - 1} value={s.py ?? t.vector.pixel[0]}
                    onChange={(e) => { const y = toInt(e.target.value); if (y != null) go({ py: y, px: s.px ?? t.vector.pixel[1] }); }} />
                  <input className="field" type="number" aria-label="Pixel x" min={0} max={t.width - 1} value={s.px ?? t.vector.pixel[1]}
                    onChange={(e) => { const x = toInt(e.target.value); if (x != null) go({ py: s.py ?? t.vector.pixel[0], px: x }); }} />
                </div>
              </div>
              <Seg label="Scale" value={s.norm} onPick={(norm) => go({ norm })}
                options={[["channel", "Per channel"], ["global", "Global"]] as [Norm, string][]} />
              <Seg label="Colour" value={s.cmap} onPick={(cmap) => go({ cmap })}
                options={[["seq", "Sequential"], ["div", "Diverging"]] as [Cmap, string][]} />
            </>
          )}
          {t?.kind === "attention" && (
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="lbl dim">Head (0–{t.heads - 1})</span>
              <input className="field" type="number" min={0} max={t.heads - 1} value={s.c ?? t.head}
                onChange={(e) => { const c = toInt(e.target.value); if (c != null) go({ c }); }} />
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
        {node && (s.vs || cmp) ? (
          <ComparePanel run={run} a={s.vs ? side(node, null, "A · clean") : side(node, s.iv, "A")}
            b={s.vs ? side(node, s.iv, "B · intervened") : side(cmp!, s.iv, "B")} channel={s.c ?? (t?.kind === "image" ? t.channel : null)}
            pixel={s.py != null && s.px != null ? [s.py, s.px] : pixel} norm={s.norm} cmap={s.cmap} onPixel={(py, px) => go({ py, px })} />
        ) : (
          <>
            {missing && (
              <div className="empty" style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-start" }}>
                <span className="lbl">No tensor saved</span>
                <span className="lbl dim">{[missing.path, unsavedText(missing.unsaved)].filter(Boolean).join(" · ")}</span>
                <a className="lbl chart-export" href={href(run.id, "graph", { node: missing.id })}>← Graph</a>
              </div>
            )}
            {graph.error && <div className="empty err">{graph.error}</div>}
            {tensor.error && <div className="empty err">{tensor.error}</div>}
            {!missing && !graph.error && !t && !tensor.error && <div className="empty lbl">{tensor.loading || graph.loading ? "Loading tensor…" : "No tensor saved for this node"}</div>}
            {t && tOp && (
              <TensorBody run={run} op={tOp} iv={s.iv} t={t} onChannel={(c) => go({ c })} onPixel={(py, px) => go({ py, px })} />
            )}
          </>
        )}
      </div>
    </>
  );
}

function Seg<T extends string>({ label, value, options, onPick }: { label: string; value: T; options: [T, string][]; onPick: (v: T) => void }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span className="lbl dim">{label}</span>
      <div className="seg lbl" role="group" aria-label={label}>
        {options.map(([id, text]) => <button key={id} type="button" aria-pressed={id === value} onClick={() => onPick(id)}>{text}</button>)}
      </div>
    </div>
  );
}

interface BodyProps<V> { run: RunMeta; op: string; t: V; onChannel: (c: number) => void; onPixel: (py: number, px: number) => void }

/** Export name for a chart: run_op_c3_dots (charts append .png). */
const exName = (run: RunMeta, op: string, c: number | null, chart: string) =>
  fileName([run.id, op, c == null ? null : "c" + c, chart], "png").slice(0, -4);

function TensorBody({ iv, ...p }: BodyProps<TensorView> & { iv: string | null }) {
  const { t } = p;
  const values = <ValuesTable key={`${p.op}|${iv}`} runId={p.run.id} op={p.op} shape={t.shape} iv={iv}
    initial={defaultSlice(t.shape, t.kind === "image" ? { channel: t.channel, pixel: t.vector.pixel } : {})} />;
  if (t.kind === "image") return <ImageBody {...p} t={t} values={values} />;
  if (t.kind === "sequence") return <SequenceBody {...p} t={t} values={values} />;
  if (t.kind === "attention") return <AttentionBody {...p} t={t} values={values} />;
  return <VectorBody {...p} t={t} values={values} />;
}

const card: React.CSSProperties = { border: "1px solid var(--line)", background: "var(--panel)", padding: 14, display: "flex", flexDirection: "column", gap: 10, minWidth: 0 };
const main: React.CSSProperties = { flexGrow: 1, minWidth: 0, padding: "20px 24px", display: "flex", flexDirection: "column", gap: 16 };
const side: React.CSSProperties = { width: 420, flexShrink: 0, borderLeft: "1px solid var(--line)", padding: 20, display: "flex", flexDirection: "column", gap: 18 };

function Head({ left, right }: { left: string; right?: string }) {
  return <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}><span className="lbl">{left}</span>{right && <span className="lbl dim">{right}</span>}</div>;
}

function ImageBody({ run, op, t, onChannel, onPixel, values }: BodyProps<ImageTensorView> & { values: React.ReactNode }) {
  const [overlay, setOverlay] = useState(false);
  const [alpha, setAlpha] = useState(0.6);
  const gridRef = useRef<HTMLDivElement>(null);
  const ex = (chart: string) => exName(run, op, t.channel, chart);
  const signed = t.cmap === "div";
  const [py, px] = t.vector.pixel;
  const rows = t.dots.length, cols = t.dots[0]?.length ?? 0;
  const mark: [number, number] = [cellOf(t.height, rows, py), cellOf(t.width, cols, px)];
  const pick = (r: number, c: number) => onPixel(cellPixel(t.height, rows, r), cellPixel(t.width, cols, c));
  const readout = (r: number, c: number, v: number) =>
    `px (${cellSpan(t.height, rows, r)}, ${cellSpan(t.width, cols, c)}) · ${num(unscale(v, t.dots_range, t.cmap), 3)}`;
  const exportGrid = () => {
    const el = gridRef.current;
    if (!el) return;
    const n = getComputedStyle(el).gridTemplateColumns.split(" ").filter(Boolean).length;
    void exportTiles([...el.querySelectorAll("canvas")], Math.max(1, n), ex("grid") + ".png");
  };
  const scaleText = t.scale ? `Scale · global ${num(t.scale[0])} … ${num(t.scale[1])}` : "Scale · per channel";

  return (
    <>
      <section className="scroll" style={main}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <h1 style={{ fontSize: 26 }}>Channels</h1>
          <span style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
            <span className="lbl dim">Top {t.maps.length} of {t.channels} · sorted by mean · {t.norm === "global" ? "global" : "per-channel"} scale</span>
            <button type="button" className={textBtn} onClick={exportGrid} aria-label="Export channel grid as PNG">PNG</button>
          </span>
        </div>
        <div ref={gridRef} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(72px, 1fr))", gap: 10 }}>
          {t.maps.map((m) => {
            const on = m.channel === t.channel;
            const mr = m.map.length, mc = m.map[0]?.length ?? 0;
            return (
              <button key={m.channel} type="button" onClick={() => { if (!on) onChannel(m.channel); }} aria-pressed={on}
                aria-label={`Channel ${m.channel}, mean ${num(m.mean)}`}
                style={{ padding: 0, border: 0, background: "transparent", cursor: on ? "crosshair" : "pointer", display: "flex", flexDirection: "column", gap: 4, textAlign: "left" }}>
                <span style={{ display: "block", border: on ? "2px solid var(--accent)" : "1px solid var(--line)" }}>
                  <HeatCanvas grid={m.map} signed={signed} style={{ aspectRatio: "1" }} label={`Channel ${m.channel} map`}
                    {...(on ? {
                      mark: [cellOf(t.height, mr, py), cellOf(t.width, mc, px)] as [number, number],
                      onPick: (r: number, c: number) => onPixel(cellPixel(t.height, mr, r), cellPixel(t.width, mc, c)),
                    } : {})} />
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
            <Head left={`Channel vector @ (${py}, ${px})`} right={`1D · ${t.vector.values.length}`} />
            <DivergingBars values={t.vector.values} highlight={t.channel} onPick={onChannel} exportName={ex("vector")} />
          </div>
          <div style={card}>
            <Head left={`PCA · ${t.height * t.width} positions`} right={`PC1 ${Math.round(t.pca.explained[0] * 100)}% · PC2 ${Math.round((t.pca.explained[1] ?? 0) * 100)}%`} />
            <Scatter points={t.pca.points} color={t.pca.color} height={150} exportName={ex("pca")} />
          </div>
        </div>
        {values}
      </section>
      <aside className="panel scroll" style={side}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
          <span className="num" style={{ fontSize: 28, fontWeight: 500, letterSpacing: "-0.03em" }}>c{t.channel}</span>
          <span className="lbl dim">{t.height}×{t.width} → {rows}×{cols} pooled</span>
        </div>
        <OverlayControls run={run} on={overlay} onOn={setOverlay} alpha={alpha} onAlpha={setAlpha} />
        <div className="rule-top" style={{ paddingTop: 10, borderBottom: "1px solid var(--line)" }}>
          {overlay ? (
            <div style={{ width: 360, maxWidth: "100%", margin: "0 auto" }}>
              <InputOverlay run={run} grid={t.dots} signed={signed} alpha={alpha} label={`Channel ${t.channel} over input`}
                exportName={ex("overlay")} onPick={pick} mark={mark} readout={readout} />
            </div>
          ) : (
            <DotMatrix grid={t.dots} signed={signed} pitch={Math.min(22, Math.floor(360 / Math.max(1, cols)))} label={`Channel ${t.channel} dot matrix`}
              exportName={ex("dots")} onPick={pick} mark={mark} readout={readout} />
          )}
        </div>
        <div className="lbl dim" style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
          {signed ? (
            <span><span style={{ color: "var(--negative)" }}>●</span> neg · ○ 0 · <span style={{ color: "var(--accent)" }}>●</span> pos</span>
          ) : <span>○ low · ● high</span>}
          <span>Rank {t.channel_rank} / {t.channels}</span>
        </div>
        <span className="lbl dim" style={{ marginTop: -10 }}>{scaleText}</span>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Head left={`Row ${t.profile.row} profile`} right={`1D · ${t.profile.values.length}`} />
          <LinePlot values={t.profile.values} xLabel="col" exportName={ex("profile")} mark={cellOf(t.width, t.profile.values.length, px)} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Head left="Value distribution" right={`${t.hist.counts.length} bins`} />
          <Histogram counts={t.hist.counts} range={t.hist.range} height={70} exportName={ex("hist")} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 8 }}>
          {[["Mean", num(t.stats.mean)], ["Std", num(t.stats.std)], ["Max", num(t.stats.max)],
            ["Min", num(t.stats.min)], ["Peak (y,x)", `(${t.peak.join(",")})`], ["Pixel (y,x)", `(${py},${px})`]].map(([k, v]) => (
            <div key={k} style={{ display: "flex", flexDirection: "column", gap: 2 }}><span className="lbl dim">{k}</span><span className="num">{v}</span></div>
          ))}
        </div>
      </aside>
    </>
  );
}

/** Token labels as a column beside a map: one per row, right aligned, ellipsis at 10 chars (T ≤ 32). `tail` = the
 * readout line's height under the map, so rows line up with the map only. */
function TokenAxis({ tokens, tail = 0 }: { tokens: string[]; tail?: number }) {
  return (
    <div aria-hidden style={{ display: "grid", gridTemplateRows: `repeat(${tokens.length}, minmax(0, 1fr))${tail ? ` ${tail}px` : ""}`, flexShrink: 0 }}>
      {tokens.map((tk, i) => <span key={i} className="mono" style={{ ...tick, textAlign: "right", alignSelf: "center" }}>{showToken(tk)}</span>)}
    </div>
  );
}
const tick: React.CSSProperties = { fontSize: 10.5, color: "var(--ink-2)", maxWidth: "10ch", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", lineHeight: 1 };

function SequenceBody({ run, op, t, values }: BodyProps<SequenceTensorView> & { values: React.ReactNode }) {
  const ex = (chart: string) => exName(run, op, null, chart);
  const labels = tokenLabels(run, t.tokens);
  const rows = t.heatmap.length, cols = t.heatmap[0]?.length ?? 0;
  const axis = labels && t.tokens <= 32 && rows === t.tokens;
  const shown = (l: string[] | null) => l?.map(showToken);
  return (
    <>
      <section className="scroll" style={main}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <h1 style={{ fontSize: 26 }}>Tokens × features</h1>
          <span className="lbl dim">{t.tokens} tokens · {t.features} features · signed, max-abs scaled</span>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {axis && <TokenAxis tokens={labels} tail={14} />}
          <div style={{ flexGrow: 1, minWidth: 0 }}>
            <div style={{ border: "1px solid var(--line)" }}>
              <HeatCanvas grid={t.heatmap} signed style={{ height: 360, imageRendering: "pixelated" }} label="Token by feature heatmap"
                exportName={ex("heatmap")} readout={(r, c, v) => `${spanLabel(labels, t.tokens, rows, r)} · feature ${cellSpan(t.features, cols, c)} · ${num(v, 3)}`} />
            </div>
          </div>
        </div>
        <div style={card}>
          <Head left="Per-token norm" right={`1D · ${t.token_norms.length}`} />
          <LinePlot values={t.token_norms} xLabel="token" exportName={ex("norms")} labels={shown(tokenLabels(run, t.token_norms.length))} />
        </div>
        {values}
      </section>
      <aside className="panel scroll" style={side}>
        <Head left={`PCA · ${t.tokens} tokens`} right={`PC1 ${Math.round(t.pca.explained[0] * 100)}%`} />
        <Scatter points={t.pca.points} color={t.pca.color} exportName={ex("pca")}
          labels={t.tokens <= 400 ? shown(tokenLabels(run, t.pca.points.length)) : undefined} />
        <Head left="Value distribution" />
        <Histogram counts={t.hist.counts} range={t.hist.range} height={70} exportName={ex("hist")} />
      </aside>
    </>
  );
}

const th: React.CSSProperties = { padding: "6px 8px", textAlign: "right", fontWeight: 400, borderBottom: "1px solid var(--ink)" };
const td: React.CSSProperties = { padding: "5px 8px", textAlign: "right", borderBottom: "1px solid var(--line-soft)" };

function AttentionBody({ run, op, t, onChannel, values }: BodyProps<AttentionTensorView> & { values: React.ReactNode }) {
  const [mark, setMark] = useState<[number, number] | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const ex = (chart: string) => exName(run, op, t.head, chart);
  const labels = tokenLabels(run, t.tokens);
  const n = t.full.map.length;
  const ticks = labels && t.tokens <= 32 && n === t.tokens ? labels : null;
  const pick = (h: number) => { if (h !== t.head) onChannel(h); };
  const exportGrid = () => {
    const el = gridRef.current;
    if (!el) return;
    const cols = getComputedStyle(el).gridTemplateColumns.split(" ").filter(Boolean).length;
    void exportTiles([...el.querySelectorAll("canvas")], Math.max(1, cols), ex("heads") + ".png");
  };
  return (
    <>
      <section className="scroll" style={main}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <h1 style={{ fontSize: 26 }}>Attention</h1>
          <span style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
            <span className="lbl dim">{t.heads} heads · {t.tokens} tokens · each map scaled to its max</span>
            <button type="button" className={textBtn} onClick={exportGrid} aria-label="Export head grid as PNG">PNG</button>
          </span>
        </div>
        <div ref={gridRef} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(88px, 1fr))", gap: 10 }}>
          {t.maps.map((m) => {
            const on = m.head === t.head, sum = t.summary[m.head];
            return (
              <button key={m.head} type="button" onClick={() => pick(m.head)} aria-pressed={on}
                aria-label={`Head ${m.head}, entropy ${(sum?.entropy ?? 0).toFixed(2)} bits`}
                style={{ padding: 0, border: 0, background: "transparent", cursor: "pointer", display: "flex", flexDirection: "column", gap: 4, textAlign: "left" }}>
                <span style={{ display: "block", border: on ? "2px solid var(--accent)" : "1px solid var(--line)" }}>
                  <HeatCanvas grid={m.map} style={{ aspectRatio: "1", imageRendering: "pixelated" }} label={`Head ${m.head} map`} />
                </span>
                <span style={{ display: "flex", justifyContent: "space-between", gap: 4 }}>
                  <span className="lbl" style={{ color: on ? "var(--accent)" : undefined }}>h{m.head}</span>
                  <span className="lbl dim" style={{ textTransform: "none" }}>{sum ? `${sum.entropy.toFixed(2)} b` : ""}</span>
                  {sum?.pattern && <span className="lbl" style={{ color: "var(--accent)" }}>{patternLabel(sum.pattern)}</span>}
                </span>
              </button>
            );
          })}
        </div>
        <div style={card}>
          <Head left="Head summary" right="mean attention weight · entropy over keys" />
          <table aria-label="Head summary" className="num" style={{ borderCollapse: "collapse", fontSize: 12, width: "100%" }}>
            <thead className="lbl dim">
              <tr>{["Head", "Entropy (bits)", "Prev", "First", "Self", "Pattern"].map((h) => <th key={h} style={th}>{h}</th>)}</tr>
            </thead>
            <tbody>
              {t.summary.map((h) => (
                <tr key={h.head} onClick={() => pick(h.head)}
                  style={{ cursor: "pointer", background: h.head === t.head ? "var(--accent-soft)" : undefined }}>
                  <td style={td}>h{h.head}</td><td style={td}>{h.entropy.toFixed(2)}</td><td style={td}>{h.prev.toFixed(2)}</td>
                  <td style={td}>{h.first.toFixed(2)}</td><td style={td}>{h.self.toFixed(2)}</td>
                  <td style={{ ...td, color: "var(--accent)" }}>{patternLabel(h.pattern) || <span className="dim">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {values}
      </section>
      <aside className="panel scroll" style={side}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
          <span className="num" style={{ fontSize: 28, fontWeight: 500, letterSpacing: "-0.03em" }}>h{t.head}</span>
          <span className="lbl dim">{t.tokens}×{t.tokens}{n < t.tokens ? ` → ${n}×${n} pooled` : " · full resolution"}</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: ticks ? "auto minmax(0, 1fr)" : "minmax(0, 1fr)", columnGap: 6 }}>
          {ticks && <span />}
          {ticks && (
            <div aria-hidden style={{ display: "grid", gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))`, height: 52, alignItems: "end" }}>
              {ticks.map((tk, i) => (
                <span key={i} style={{ width: 0, height: 0, overflow: "visible", justifySelf: "center" }}>
                  <span className="mono" style={{ ...tick, display: "block", width: "max-content", transform: "rotate(-60deg)", transformOrigin: "left center", marginTop: -5 }}>{showToken(tk)}</span>
                </span>
              ))}
            </div>
          )}
          {ticks && <TokenAxis tokens={ticks} tail={14} />}
          <div style={{ border: "1px solid var(--line)" }}>
            <HeatCanvas key={t.head} grid={t.full.map} style={{ aspectRatio: "1", imageRendering: "pixelated" }} label={`Head ${t.head} attention`}
              exportName={ex("attention")} mark={mark} onPick={(r, c) => setMark([r, c])}
              readout={(r, c, v) => headReadout(t, r, c, v, labels)} />
          </div>
        </div>
        <span className="lbl dim" style={{ marginTop: -10 }}>Rows query · columns key · ○ 0 · ● the head's max ({num(t.full.max, 3)})</span>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Head left="Value distribution" right={`head ${t.head} · ${t.hist.counts.length} bins`} />
          <Histogram counts={t.hist.counts} range={t.hist.range} height={70} exportName={ex("hist")} />
        </div>
      </aside>
    </>
  );
}

function VectorBody({ run, op, t, values }: BodyProps<VectorTensorView> & { values: React.ReactNode }) {
  const ex = (chart: string) => exName(run, op, null, chart);
  return (
    <>
      <section className="scroll" style={main}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <h1 style={{ fontSize: 26 }}>Signal</h1>
          <span className="lbl dim">{t.length} values{t.values.length < t.length ? ` · pooled to ${t.values.length}` : ""}</span>
        </div>
        <div style={card}><DivergingBars values={t.values} height={200} xLabel="index" exportName={ex("signal")}
          labels={tokenLabels(run, t.values.length)?.map(showToken)} /></div>
        {values}
      </section>
      <aside className="panel scroll" style={side}>
        <Head left="Largest |value|" />
        <div className="rule-top">
          {t.top.map((r) => (
            <div key={r.index} className="kv"><span className="mono">[{r.index}]</span>
              <span className="num" style={{ color: r.value >= 0 ? "var(--ink)" : "var(--negative-text)" }}>{num(r.value, 3)}</span></div>
          ))}
        </div>
        <Head left="Value distribution" />
        <Histogram counts={t.hist.counts} range={t.hist.range} height={70} exportName={ex("hist")} />
      </aside>
    </>
  );
}
