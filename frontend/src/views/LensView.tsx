import { useEffect, useRef, useState } from "react";
import { LinePlot, Readout, pngName } from "../components/charts";
import { JobProgress } from "../components/JobProgress";
import { TopBar } from "../components/TopBar";
import { exportSvg, fileName } from "../lib/exportPng";
import { num } from "../lib/format";
import { useJob } from "../lib/jobs";
import { clip, lensCellText, lensColumn, lensParams, lensReadout, parseLensParams, shortLayer, type LensMode, type LensState } from "../lib/lensState";
import { href, navigate, type Route } from "../lib/router";
import { showToken } from "../lib/tokens";
import type { LensResult, RunMeta } from "../lib/types";

const HEAD_W = 120, COL = 56, ROW = 22, TOP = 24;
const MONO: React.CSSProperties = { fontFamily: "var(--font-mono)", fontSize: 10.5 };
const MODES: [LensMode, string][] = [["top", "Top-1"], ["target", "Target"]];
const card: React.CSSProperties = { border: "1px solid var(--line)", background: "var(--panel)", padding: 14, display: "flex", flexDirection: "column", gap: 10, minWidth: 0 };

export function LensViewPage({ run, route }: { run: RunMeta; route: Route }) {
  const s = parseLensParams(route.params);
  const set = (p: Partial<LensState>) => navigate(href(run.id, "lens", lensParams({ ...s, ...p })), true);
  const lens = useJob<LensResult>();
  const [res, setRes] = useState<LensResult | null>(null);
  const runLens = async () => { const r = await lens.run(run.id, { kind: "lens", target: s.t }); if (r) setRes(r); };
  // a new target re-runs an open lens (the server caches one result per target)
  useEffect(() => { if (res) void runLens(); }, [s.t]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!run.model.lm) {
    return (
      <>
        <TopBar run={run} tab="lens" />
        <div className="empty" style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-start" }}>
          <span className="lbl">Logit lens needs a causal language model</span>
          <a className="lbl chart-export" href={href(run.id, "graph")}>← Graph</a>
        </div>
      </>
    );
  }
  return (
    <>
      <TopBar run={run} tab="lens">
        <button type="button" className="btn solid lbl" disabled={lens.busy} onClick={() => void runLens()}>Run lens</button>
      </TopBar>
      <div style={{ minHeight: 56, flexShrink: 0, display: "flex", alignItems: "center", gap: 24, padding: "8px 20px", borderBottom: "1px solid var(--line)", flexWrap: "wrap" }}>
        <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="lbl dim">Target</span>
          <select className="field" value={s.t ?? ""} onChange={(e) => set({ t: e.target.value === "" ? null : Number(e.target.value) })}>
            <option value="">top prediction</option>
            {run.output.map((o) => (
              <option key={o.index} value={o.index}>{o.label != null ? showToken(o.label) : `#${o.index}`} · p {o.prob.toFixed(2)}</option>
            ))}
            {s.t != null && !run.output.some((o) => o.index === s.t) && <option value={s.t}>#{s.t}</option>}
          </select>
        </label>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="lbl dim">Colour</span>
          <div className="seg lbl" role="group" aria-label="Lens colour">
            {MODES.map(([m, label]) => <button key={m} type="button" aria-pressed={m === s.mode} onClick={() => set({ mode: m })}>{label}</button>)}
          </div>
        </div>
        <div style={{ flexGrow: 1 }} />
        {lens.job && lens.job.status !== "done"
          ? <div style={{ width: 240 }}><JobProgress job={lens.job} onCancel={lens.cancel} /></div>
          : res && <span className="lbl dim">{res.method_label}</span>}
      </div>
      {lens.error && lens.job?.status !== "error" && <div className="err mono" style={{ padding: "8px 20px", borderBottom: "1px solid var(--line)" }}>{lens.error}</div>}
      {res
        ? <LensBody run={run} res={res} s={s} set={set} />
        : <div className="empty lbl">{lens.busy ? "Computing…" : "Run lens to read every layer's prediction through the final norm and unembedding"}</div>}
    </>
  );
}

function LensBody({ run, res, s, set }: { run: RunMeta; res: LensResult; s: LensState; set: (p: Partial<LensState>) => void }) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [clicked, setClicked] = useState<number | null>(null);   // layer of the last clicked cell
  const rows = res.layers.length, cols = res.tokens.length;
  const j = lensColumn(res, s.pos);
  const [tl, tj] = hover ?? [clicked ?? rows - 1, j];
  const alpha = (l: number, c: number) => Math.min(1, Math.max(0, s.mode === "top" ? res.top[l][c][0]?.prob ?? 0 : res.target_prob[l][c]));
  const topk = res.top[tl][tj];
  const exportGrid = async () => { if (ref.current) await exportSvg(ref.current, pngName(fileName([run.id, "lens", s.mode], "png"), "Logit lens")); };

  return (
    <div style={{ flexGrow: 1, minHeight: 0, display: "flex" }}>
      <section className="scroll" style={{ flexGrow: 1, minWidth: 0, padding: "20px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <h1 style={{ fontSize: 26 }}>Logit lens</h1>
          <span className="lbl dim">
            {rows} layers × {cols} positions{res.pos0 ? ` · from pos ${res.pos0}` : ""} · {s.mode === "top" ? "fill = top-1 prob" : `fill = p("${showToken(res.target.token)}") · #rank`}
          </span>
        </div>
        <div style={card}>
          <div className="scroll" style={{ overflowX: "auto" }}>
            <svg ref={ref} width={HEAD_W + cols * COL + 1} height={TOP + rows * ROW + 1} role="img" aria-label="Logit lens"
              onMouseLeave={() => setHover(null)} style={{ display: "block" }}>
              {res.tokens.map((tk, c) => (
                <text key={c} x={HEAD_W + c * COL + COL / 2} y={TOP - 8} textAnchor="middle" fill={c === j ? "var(--accent)" : "var(--ink-2)"} style={MONO}>
                  {clip(showToken(tk), 8)}
                </text>
              ))}
              {res.layers.map((name, l) => (
                <text key={name} x={8} y={TOP + l * ROW + ROW / 2 + 4} fill="var(--ink-2)"
                  style={{ ...MONO, letterSpacing: "0.05em", textTransform: "uppercase" }}>{clip(shortLayer(name), 16)}</text>
              ))}
              {res.layers.map((_, l) => res.tokens.map((__, c) => {
                const x = HEAD_W + c * COL + 0.5, y = TOP + l * ROW + 0.5, a = alpha(l, c);
                return (
                  <g key={`${l}-${c}`} onMouseEnter={() => setHover([l, c])} onClick={() => { setClicked(l); set({ pos: res.pos0 + c }); }} style={{ cursor: "pointer" }}>
                    <rect x={x} y={y} width={COL} height={ROW} fill="var(--panel)" stroke="var(--line-soft)" />
                    <rect x={x + 0.5} y={y + 0.5} width={COL - 1} height={ROW - 1} fill="var(--accent)" fillOpacity={a} />
                    <text x={x + COL / 2} y={y + ROW / 2 + 4} textAnchor="middle" fill={a < 0.55 ? "var(--ink)" : "var(--panel)"} style={MONO}>
                      {lensCellText(res, s.mode, l, c)}
                    </text>
                  </g>
                );
              }))}
              <rect x={HEAD_W + j * COL + 0.5} y={TOP + 0.5} width={COL} height={rows * ROW} fill="none" stroke="var(--accent)" pointerEvents="none" />
            </svg>
          </div>
          <Readout text={hover ? lensReadout(res, hover[0], hover[1]) : null} label="Logit lens" onExport={exportGrid} />
        </div>
      </section>
      <aside className="panel scroll" style={{ width: 360, flexShrink: 0, borderLeft: "1px solid var(--line)", padding: 20, display: "flex", flexDirection: "column", gap: 18 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span className="num" style={{ fontSize: 24, fontWeight: 500, letterSpacing: "-0.03em" }}>Position {res.pos0 + j} · "{showToken(res.tokens[j])}"</span>
          <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="lbl dim">Position ({res.pos0}–{res.pos0 + cols - 1}) · or click a cell</span>
            <input className="field" type="number" min={res.pos0} max={res.pos0 + cols - 1} value={res.pos0 + j} style={{ width: 72 }}
              onChange={(e) => { const p = Number(e.target.value); if (Number.isInteger(p) && e.target.value !== "") set({ pos: p }); }} />
          </label>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
            <span className="lbl">Target across layers</span><span className="lbl dim">"{showToken(res.target.token)}" · #{res.target.id}</span>
          </div>
          <LinePlot values={res.target_prob.map((r) => r[j])} xLabel="layer" labels={res.layers.map(shortLayer)} mark={clicked}
            exportName={fileName([run.id, "lens", "target", `pos${res.pos0 + j}`], "png")} />
          <div className="rule-top">
            {res.layers.map((name, l) => (
              <div key={name} className="kv" style={{ color: l === tl && tj === j ? "var(--accent)" : undefined }}>
                <span className="mono">{shortLayer(name)}</span>
                <span className="num">#{res.target_rank[l][j]} · p {num(res.target_prob[l][j], 3)}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
            <span className="lbl">Top-{res.k} at {shortLayer(res.layers[tl])}</span><span className="lbl dim">pos {res.pos0 + tj}</span>
          </div>
          <div className="rule-top">
            {topk.map((c) => (
              <div key={c.id} style={{ display: "grid", gridTemplateColumns: "1fr 110px 48px", gap: 10, alignItems: "center", padding: "7px 0", borderBottom: "1px solid var(--line-soft)" }}>
                <span className="mono" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>"{showToken(c.token)}"</span>
                <span style={{ height: 8, borderLeft: "1px solid var(--ink)", position: "relative" }}>
                  <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${Math.min(1, c.prob) * 100}%`, background: "var(--ink)" }} />
                </span>
                <span className="num" style={{ fontSize: 12, textAlign: "right" }}>{c.prob.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
