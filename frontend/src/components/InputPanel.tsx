// The Input view: the raw sample next to the model input it became, the preprocessing between them in words,
// and the per-channel ranges of the values that entered the model. Shown in the Tensors tab for the input node.
// A dataset sample (a py: module's load_sample) shows its raw signal and the (1, T, C) feature matrix on one clock.
import { useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { exportSvg } from "../lib/exportPng";
import { num } from "../lib/format";
import {
  bandPath, clock, envelopeDomain, hasMatrix, isSignal, linePath, modelCaption, padEnd, pictureSize, pictureWidth, place,
  rawCaption, rawNote, rawSpan, sparseRows, stepRange, stepSpan, union,
} from "../lib/inputState";
import { showToken } from "../lib/tokens";
import type { InputInfo, MatrixView, RunMeta, SignalView } from "../lib/types";
import { useAsync } from "../lib/useAsync";
import { Axes, axisText, HeatCanvas, pngName, Readout } from "./charts";
import { Band, StepLine, WindowLegend, type StepMarks } from "./SeriesMap";

const card: React.CSSProperties = { border: "1px solid var(--line)", background: "var(--panel)", padding: 14, display: "flex", flexDirection: "column", gap: 12, minWidth: 0 };

function Head({ left, right }: { left: string; right?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
      <span className="lbl">{left}</span>{right && <span className="lbl dim" style={{ textAlign: "right" }}>{right}</span>}
    </div>
  );
}

/** A W×H picture with its pixel ranges on both axes (row 0 at the top), 224–360 px tall on screen. */
function Picture({ src, w, h, alt }: { src: string; w: number; h: number; alt: string }) {
  return (
    <div style={{ width: `min(100%, ${pictureWidth(w, h) + 44}px)`, margin: "0 auto" }}>
      <Axes x={{ from: 0, to: w - 1, label: "x · px" }} y={{ from: 0, to: h - 1, label: "y · px" }}>
        <img src={src} alt={alt} style={{ display: "block", width: "100%", height: "auto", border: "1px solid var(--line)",
          imageRendering: w < 112 || h < 112 ? "pixelated" : "auto" }} />
      </Axes>
    </div>
  );
}

function Tokens({ info }: { info: InputInfo }) {
  const toks = info.sample.tokens ?? [], ids = info.sample.token_ids ?? [];
  return (
    <Axes x={{ from: 0, to: Math.max(0, ids.length - 1), label: "position" }}>
      <div role="list" aria-label="Model input tokens" style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {ids.map((id, i) => (
          <span key={i} role="listitem" title={`#${i} · id ${id}`}
            style={{ display: "inline-flex", flexDirection: "column", alignItems: "center", padding: "2px 5px", border: "1px solid var(--line-soft)" }}>
            <span className="mono" style={{ whiteSpace: "pre" }}>{showToken(toks[i] ?? "")}</span>
            <span className="lbl dim" style={{ fontSize: 9 }}>{id}</span>
          </span>
        ))}
      </div>
    </Axes>
  );
}

// ---- signals: every chart shares the x domain (seconds on the raw clock), the y gutter width and a right-hand
// label column, so a raw sample and the model steps computed from it sit at the same horizontal position ----

type Span = [number, number];
interface Clock { domain: Span; div: number; label: string; unit: string; yw: number }
const LABEL_W = 112;

/** A chart plus the right-hand label column every chart of the view has (empty for line plots). */
function Row({ children, labels }: { children: React.ReactNode; labels?: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `minmax(0, 1fr) ${LABEL_W}px`, columnGap: 8 }}>
      <div style={{ minWidth: 0 }}>{children}</div><div style={{ position: "relative" }}>{labels}</div>
    </div>
  );
}

const xAxis = (c: Clock) => ({ from: c.domain[0] / c.div, to: c.domain[1] / c.div, label: c.label });

/** One raw channel: the min / max envelope of each pooled point as a soft band, the mean as a line. */
function SignalPlot({ ch, sig, clk, exportName, overlay }: { ch: SignalView["channels"][number]; sig: SignalView; clk: Clock; exportName: string; overlay?: React.ReactNode }) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const W = 600, H = 84, n = ch.mean.length;
  const [lo, hi] = envelopeDomain(ch.min, ch.max);
  const pos = place([sig.t0, sig.t0 + ch.length / sig.fs], clk.domain);
  const x = (i: number) => (Math.min(ch.length, (i + 0.5) * ch.stride) / ch.length) * W;
  const y = (v: number) => H - 2 - ((v - lo) / (hi - lo)) * (H - 4);
  const name = `${ch.name}${ch.unit ? ` · ${ch.unit}` : ""}`;
  const axes = { x: xAxis(clk), y: { from: padEnd(num(hi), clk.yw), to: padEnd(num(lo), clk.yw), label: name } };
  const t = (i: number) => (sig.t0 + ((i + 0.5) * ch.stride) / sig.fs) / clk.div;
  const u = ch.unit ? ` ${ch.unit}` : "";
  const text = hover == null || ch.mean[hover] == null ? null : ch.stride > 1
    ? `t ${num(t(hover))} ${clk.unit} · mean ${num(ch.mean[hover], 3)}${u} · ${num(ch.min[hover], 3)} … ${num(ch.max[hover], 3)}`
    : `t ${num(t(hover))} ${clk.unit} · ${num(ch.mean[hover], 3)}${u}`;
  return (
    <Row>
      <Axes {...axes}>
        <div style={{ marginLeft: `${pos.left}%`, width: `${pos.width}%` }}>
          <svg ref={ref} width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={`Raw ${ch.name}`}
            style={{ display: "block" }} onMouseLeave={() => setHover(null)}
            onMouseMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              setHover(Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * n))));
            }}>
            <path d={bandPath(ch.min, ch.max, x, y)} fill="var(--accent-soft)" />
            <path d={linePath(ch.mean, x, y)} fill="none" stroke="var(--accent)" strokeWidth="1.25" vectorEffect="non-scaling-stroke" />
            <line x1="0" x2={W} y1={H - 0.5} y2={H - 0.5} stroke="var(--ink)" vectorEffect="non-scaling-stroke" />
            {hover != null && <line x1={x(hover)} x2={x(hover)} y1="0" y2={H} stroke="var(--ink)" vectorEffect="non-scaling-stroke" />}
          </svg>
        </div>
        {overlay}
      </Axes>
      <Readout text={text ?? (ch.stride > 1 ? `${ch.length} → ${n} points · band = min … max, line = mean` : `${ch.length} samples`)}
        label={`Raw ${ch.name}`} onExport={async () => { if (ref.current) await exportSvg(ref.current, pngName(exportName, name), 2, axisText(axes)); }} />
    </Row>
  );
}

/** One block of a (1, T, C) input: rows = its columns (every n-th label on the right), x = time, diverging scale. */
function BlockMap({ b, mat, clk, exportName, overlay }: { b: MatrixView["blocks"][number]; mat: MatrixView; clk: Clock; exportName: string; overlay?: React.ReactNode }) {
  const rows = b.values.length, cols = b.values[0]?.length ?? 0;
  const m = Math.max(Math.abs(b.lo ?? 0), Math.abs(b.hi ?? 0)) || 1;
  const grid = useMemo(() => b.values.map((r) => r.map((v) => (v == null ? 0 : v / m))), [b, m]);
  const labels = b.labels ?? Array.from({ length: rows }, (_, i) => `col ${b.start + i}`);
  const time = mat.time?.axis === 1 ? mat.time : null;
  const pos = place(time ? stepSpan(time, cols, mat.stride) : [-0.5, cols * mat.stride - 0.5], clk.domain);
  const H = Math.min(200, Math.max(48, rows * 12));
  const name = `${b.name}${b.unit ? ` · ${b.unit}` : ""}`;
  const readout = (r: number, c: number) => {
    const a = c * mat.stride, z = Math.min(mat.steps, (c + 1) * mat.stride) - 1;
    const t = time ? `t ${num((time.t0 + ((a + z) / 2) * time.step_s) / clk.div)} ${clk.unit} · ` : "";
    return `${t}step ${a === z ? a : `${a}–${z}`} · "${labels[r]}" · ${num(b.values[r]?.[c], 3)}${b.unit ? ` ${b.unit}` : ""}`;
  };
  const tick = sparseRows(rows);
  return (
    <Row labels={
      <div className="axis" style={{ position: "absolute", top: 14, height: H, left: 0, right: 0 }}>
        {tick.map((i) => (
          <span key={i} title={labels[i]} style={{ position: "absolute", left: 0, right: 0, top: `${((i + 0.5) / rows) * 100}%`,
            transform: "translateY(-50%)", overflow: "hidden", textOverflow: "ellipsis" }}>{labels[i]}</span>
        ))}
      </div>
    }>
      <HeatCanvas grid={grid} signed label={b.name} exportName={exportName} readout={readout}
        x={xAxis(clk)} y={{ from: padEnd(String(b.start), clk.yw), to: padEnd(String(b.start + rows - 1), clk.yw), label: `${name} · column` }}
        style={{ height: H, width: `${pos.width}%`, marginLeft: `${pos.left}%`, imageRendering: "pixelated" }}>{overlay}</HeatCanvas>
    </Row>
  );
}

/** The raw signal card and the model input card of a dataset sample (or of any float (1, T, C) input). */
function SignalCards({ run, info, sig, mat, marks }: { run: RunMeta; info: InputInfo; sig: SignalView | null; mat: MatrixView | null; marks: StepMarks | null }) {
  const raw = isSignal(info.raw) ? info.raw : null;
  const time = mat?.time?.axis === 1 ? mat.time : null;
  const cols = mat?.blocks[0]?.values[0]?.length ?? 0;
  // one clock when the input's time axis is on the raw clock; else raw in seconds, the matrix in steps
  const rawDom = raw ? rawSpan(raw) : null;
  const matDom: Span | null = mat ? (time ? stepSpan(time, cols, mat.stride) : [-0.5, cols * mat.stride - 0.5]) : null;
  const shared = rawDom && matDom && time ? union(rawDom, matDom) : null;
  const yEnds = [...(sig?.channels.flatMap((c) => envelopeDomain(c.min, c.max).map((v) => num(v))) ?? []),
    ...(mat?.blocks.map((b) => String(b.start + b.values.length - 1)) ?? [])];
  const yw = Math.max(1, ...yEnds.map((s) => s.length));
  const clockOf = (dom: Span, seconds: boolean): Clock => {
    if (!seconds) return { domain: dom, div: 1, label: "step", unit: "", yw };
    const c = clock(dom[1] - dom[0], raw?.time_label ?? "s");
    return { domain: dom, ...c, unit: c.div === 60 ? "min" : "s", yw };
  };
  const rawClk = rawDom && clockOf(shared ?? rawDom, true);
  const matClk = matDom && clockOf(shared ?? matDom, time != null);
  const m = mat?.blocks.map((b) => Math.max(Math.abs(b.lo ?? 0), Math.abs(b.hi ?? 0)));
  // a picked output step (Trace → Show on input): its windows as bands on the maps of their blocks (and, on the raw
  // clock, over the raw time they cover), and the step as a rule
  const over = (dom: Span, block: string | null) => marks && (
    <>
      {marks.windows.filter((w) => block == null || !w.blocks || w.blocks.includes(block)).map((w, i) => {
        const p = place(stepRange(time, w.start, w.stop), dom);
        return <Band key={i} kind={w.kind} x={[p.left, p.left + p.width]} />;
      })}
      <StepLine x={place(stepRange(time, marks.step + 0.5, marks.step + 0.5), dom).left} />
    </>
  );
  return (
    <>
      <div style={card} role="group" aria-label="Raw input">
        <Head left="Raw input" right={rawCaption(info) || undefined} />
        {raw && sig && rawClk ? sig.channels.map((c) => <SignalPlot key={c.name} ch={c} sig={sig} clk={rawClk} exportName={`${run.id}_input_raw_${c.name}`}
          overlay={shared && over(rawClk.domain, null)} />)
          : raw ? <span className="lbl dim">Loading signal…</span> : <span className="lbl dim">{rawNote(info)}</span>}
        {raw && <a className="lbl chart-export" href={api.inputRawUrl(run.id, run.created)} download={raw.file}>Download raw (.npz) ↓</a>}
      </div>
      <div style={card} role="group" aria-label="Model input">
        <Head left="Model input" right={modelCaption(info)} />
        {marks && <WindowLegend marks={marks} />}
        {mat && matClk ? (
          <>
            {mat.blocks.map((b) => <BlockMap key={b.name} b={b} mat={mat} clk={matClk} exportName={`${run.id}_input_${b.name}`}
              overlay={over(matClk.domain, b.name)} />)}
            <span className="lbl dim">
              Diverging scale per block: ± max |value| ({m?.map((v) => num(v)).join(" · ")}) · accent &gt; 0 &gt; rust
              {mat.stride > 1 ? ` · ${mat.steps} → ${cols} columns, means of ${mat.stride} steps` : ""}
            </span>
          </>
        ) : <span className="lbl dim">Loading model input…</span>}
      </div>
    </>
  );
}

export function InputPanel({ run, step = null }: { run: RunMeta; step?: number | null }) {
  const res = useAsync(() => api.input(run.id), [run.id]);
  const win = useAsync(() => (step != null ? api.windows(run.id, step) : null), [run.id, step]);
  const marks = step != null && (win.data || win.error) ? { step, windows: win.data?.windows ?? [], error: win.data?.error ?? win.error } : null;
  const info = res.data;
  const signal = isSignal(info?.raw);
  const matrix = info != null && hasMatrix(info);
  const sig = useAsync(() => (signal ? api.inputSignal(run.id) : null), [run.id, signal]);
  const mat = useAsync(() => (matrix ? api.inputMatrix(run.id) : null), [run.id, matrix]);
  if (res.error) return <div className="empty err">{res.error}</div>;
  if (!info) return <div className="empty lbl">Loading input…</div>;
  const size = pictureSize(info);
  const text = info.sample.kind === "text" ? info.sample.text : null;
  const rows = Object.entries(info.info ?? {});
  const err = sig.error ?? mat.error;
  return (
    <section className="scroll" style={{ flexGrow: 1, minWidth: 0, padding: "20px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
        <h1 style={{ fontSize: 26 }}>Input</h1>
        <span className="lbl dim">The sample of this forward pass, and the tensor it became</span>
      </div>
      {err && <div className="err mono">{err}</div>}
      {signal || matrix ? <SignalCards run={run} info={info} sig={sig.data} mat={mat.data} marks={marks} /> : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 16 }}>
          <div style={card} role="group" aria-label="Raw input">
            <Head left="Raw input" right={rawCaption(info) || undefined} />
            {info.raw && !isSignal(info.raw) ? <Picture src={api.inputRawUrl(run.id, run.created)} w={info.raw.width} h={info.raw.height} alt="Raw input" />
              : text != null ? (
                <>
                  <pre className="mono" style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: 13 }}>{text}</pre>
                  <span className="lbl dim">{text.length} characters</span>
                </>
              ) : <span className="lbl dim">{rawNote(info)}</span>}
            {info.raw && <a className="lbl chart-export" href={api.inputRawUrl(run.id, run.created)} download={info.sample.name || info.raw.file}>Download raw ↓</a>}
          </div>
          <div style={card} role="group" aria-label="Model input">
            <Head left="Model input" right={modelCaption(info)} />
            {size ? <Picture src={api.inputModelUrl(run.id, run.created)} w={size[1]} h={size[0]} alt="Model input" />
              : info.sample.token_ids ? <Tokens info={info} />
              : <span className="lbl dim">Not a picture: its values are below and in the Tensor view</span>}
          </div>
        </div>
      )}
      {info.sample.kind === "ref" && (
        <div style={card} role="group" aria-label="Sample">
          <Head left="Sample" right={info.sample.name} />
          <div>
            {[["ref", info.sample.ref ?? ""], ...rows].map(([k, v]) => (
              <div key={k} className="kv"><span className="lbl dim">{k}</span><span className="mono" style={{ textAlign: "right", wordBreak: "break-word" }}>{v == null ? "—" : String(v)}</span></div>
            ))}
          </div>
        </div>
      )}
      <div style={card} role="group" aria-label="Preprocessing">
        <Head left="Preprocessing" right={info.steps ? `${info.steps.length} step${info.steps.length === 1 ? "" : "s"} · raw → model input` : undefined} />
        {info.steps ? (
          <ol className="mono" style={{ margin: 0, paddingLeft: 22, display: "flex", flexDirection: "column", gap: 6 }}>
            {info.steps.map((s, i) => <li key={i}>{s}</li>)}
          </ol>
        ) : <span className="lbl dim">Not recorded for this run</span>}
      </div>
      <div style={card}>
        <Head left="Model input values" right={info.channels[0]?.name != null ? "per block · after preprocessing"
          : info.channels[0]?.channel == null ? "whole tensor · after preprocessing" : "per channel · after preprocessing"} />
        <table aria-label="Model input values" className="num" style={{ borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr>{[info.channels[0]?.name != null ? "Block" : "Channel", "Mean", "Std", "Min", "Max"].map((h, j) => (
              <th key={h} className="lbl dim" style={{ textAlign: j === 0 ? "left" : "right", padding: "4px 8px", borderBottom: "1px solid var(--ink)" }}>{h}</th>
            ))}</tr>
          </thead>
          <tbody>
            {info.channels.map((c, i) => (
              <tr key={i} style={{ borderBottom: "1px solid var(--line-soft)" }}>
                <td style={{ padding: "4px 8px" }}>{c.name ?? (c.channel == null ? "all" : `c${c.channel}`)}</td>
                {[c.mean, c.std, c.min, c.max].map((v, j) => <td key={j} style={{ padding: "4px 8px", textAlign: "right" }}>{num(v, 4)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
