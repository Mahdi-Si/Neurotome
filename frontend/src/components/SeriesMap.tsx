// A sequence input's attribution in the Trace view: channels × time (block separators and names), the signed Σ over
// channels per time column under it on the same x, and each block's share of Σ|attribution|. With a picked output
// step: its input windows as bands, a zoom around them, and how much of the attribution lies in each window.
import { useMemo, useRef, useState } from "react";
import { exportSvg } from "../lib/exportPng";
import { num, pct } from "../lib/format";
import { linePath, padEnd } from "../lib/inputState";
import {
  blockChannels, blockSpans, seriesClock, seriesCol, seriesRow, stepsLabel, windowCols, windowRows, windowShares, zoomSpan,
} from "../lib/traceState";
import type { SeriesMap, TargetWindow, WindowKind } from "../lib/types";
import { Axes, axisText, HeatCanvas, pngName, Readout } from "./charts";

const LABEL_W = 112;
/** A signed sum, tiny ones in exponent form (`+1.2e-5`, `−0.034`). */
const sgn = (v: number) => `${v >= 0 ? "+" : "−"}${num(Math.abs(v), 3)}`;

/** A chart plus a right-hand label column (block names), so the map and the time line share one x. */
function Row({ children, labels }: { children: React.ReactNode; labels?: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `minmax(0, 1fr) ${LABEL_W}px`, columnGap: 6 }}>
      <div style={{ minWidth: 0 }}>{children}</div><div style={{ position: "relative" }}>{labels}</div>
    </div>
  );
}

/** What a picked output step adds to the map: its input windows (the module's `target_windows`), and the Input view. */
export interface StepMarks { step: number; windows: TargetWindow[]; error?: string | null; inputHref?: string | null }

const BAND: Record<WindowKind, React.CSSProperties> = {
  target: { background: "color-mix(in srgb, var(--accent) 22%, transparent)", borderLeft: "1.5px solid var(--accent)", borderRight: "1.5px solid var(--accent)" },
  // context windows are often the whole causal past: an outline only, so the map under them stays readable
  context: { border: "1px dashed var(--accent)" },
  horizon: { background: "color-mix(in srgb, var(--ink) 7%, transparent)", borderLeft: "1px dotted var(--ink)", borderRight: "1px dotted var(--ink)" },
};
// overlays keep their token colours under forced colours (Highlight / CanvasText), like the charts under them
const OVER: React.CSSProperties = { position: "absolute", boxSizing: "border-box", pointerEvents: "none", forcedColorAdjust: "none" };

/** A window's band over a chart, in % of it: x = [left, right], y = [top, bottom] (a block's rows; default all). */
export function Band({ kind, x, y = [0, 100] }: { kind: WindowKind; x: [number, number]; y?: [number, number] }) {
  const [a, b] = [Math.max(0, x[0]), Math.min(100, x[1])];
  if (b <= a) return null;
  return <div aria-hidden style={{ ...OVER, left: `${a}%`, width: `${b - a}%`, top: `${y[0]}%`, height: `${y[1] - y[0]}%`, ...BAND[kind] }} />;
}

/** The picked step: a 1.5 px `accent` rule at x % of a chart. */
export const StepLine = ({ x }: { x: number }) => (x < 0 || x > 100 ? null
  : <div aria-hidden style={{ ...OVER, left: `${x}%`, top: 0, bottom: 0, borderLeft: "1.5px solid var(--accent)", transform: "translateX(-0.75px)" }} />);

/** The legend of a step's windows: a swatch drawn like each band, its label, steps and blocks. */
export function WindowLegend({ marks }: { marks: StepMarks }) {
  const item: React.CSSProperties = { display: "flex", alignItems: "center", gap: 6 };
  return (
    <div role="list" aria-label="Input windows" className="lbl dim" style={{ display: "flex", flexWrap: "wrap", gap: "4px 14px" }}>
      <span role="listitem" style={item}><span style={{ ...OVER, position: "static", height: 10, borderLeft: "1.5px solid var(--accent)" }} />Step {marks.step}</span>
      {marks.windows.map((w, i) => (
        <span key={i} role="listitem" style={item}>
          <span style={{ ...OVER, position: "static", width: 16, height: 10, flexShrink: 0, ...BAND[w.kind] }} />
          {w.label} · {stepsLabel(w.start, w.stop)}{w.blocks ? ` · ${w.blocks.join(", ")}` : ""}
        </span>
      ))}
      {!marks.windows.length && <span role="listitem">{marks.error ?? "No input windows: the model module defines no target_windows"}</span>}
    </div>
  );
}

export function SeriesMapChart({ m, label, exportName, marks, valuesHref }: {
  m: SeriesMap; label: string; exportName: string; marks?: StepMarks | null;
  valuesHref?: (steps: [number, number], ch: [number, number] | null) => string | null;   // the Tensors view, Values prefilled
}) {
  const [hover, setHover] = useState<number | null>(null);
  // attribution often piles into a few cells; √ keeps the rest of the map visible (colour only; readouts stay linear)
  const [sqrt, setSqrt] = useState(true);
  const [zoom, setZoom] = useState(false);
  const [pick, setPick] = useState<[number, number] | null>(null);   // a clicked cell: full-map (row, column)
  const n = m.time_totals.length;
  const [c0, c1] = zoom && marks ? windowCols(m, ...zoomSpan(m.steps, marks.step, marks.windows)) : [0, n];
  const cols = Math.max(1, c1 - c0);
  const grid = useMemo(() => {
    const g = sqrt ? m.map.map((r) => r.map((v) => Math.sign(v) * Math.sqrt(Math.abs(v)))) : m.map;
    return c0 === 0 && c1 === n ? g : g.map((r) => r.slice(c0, c1));
  }, [m.map, sqrt, c0, c1, n]);
  const totals = c0 === 0 && c1 === n ? m.time_totals : m.time_totals.slice(c0, c1);
  const line = useRef<SVGSVGElement>(null);
  const clk = seriesClock(m), rows = m.map.length;
  const H = Math.min(300, Math.max(96, rows * 3)), W = 600, LH = 64;
  const top = Math.max(1e-12, ...totals.map(Math.abs));
  // one y gutter width for both charts (ends padded with figure spaces), so their x axes line up
  const ends = ["0", String(m.channels - 1), num(top), num(-top)], yw = Math.max(...ends.map((s) => s.length));
  const x = { from: clk.at(c0 * m.stride), to: clk.at(Math.min(m.steps, c1 * m.stride) - 1), label: clk.label };
  const lineAxes = { x, y: { from: padEnd(num(top), yw), to: padEnd(num(-top), yw), label: "Σ over channels" } };
  const spans = blockSpans(m);
  const peak = totals.reduce((k, v, i) => (Math.abs(v) > Math.abs(totals[k]) ? i : k), 0);
  const px = (i: number) => ((i + 0.5) / cols) * W, py = (v: number) => LH / 2 - (v / top) * (LH / 2 - 3);
  const xs = (step: number) => ((step / m.stride - c0) / cols) * 100;   // a step boundary in % of the (cropped) x
  const bands = (withRows: boolean) => marks && (
    <>
      {marks.windows.flatMap((w, i) => (withRows ? windowRows(m, w.blocks) : [[0, rows] as [number, number]]).map(([r0, r1], j) => (
        <Band key={`${i}.${j}`} kind={w.kind} x={[xs(w.start), xs(w.stop)]} y={[(r0 / rows) * 100, (Math.min(rows, r1) / rows) * 100]} />
      )))}
      <StepLine x={xs(marks.step + 0.5)} />
    </>
  );
  const cellBlock = pick && m.blocks.find((b) => pick[0] * m.row_stride >= b.start && pick[0] * m.row_stride < b.stop);
  const cellHref = pick && valuesHref?.([pick[1] * m.stride, Math.min(m.steps, (pick[1] + 1) * m.stride)], cellBlock ? [cellBlock.start, cellBlock.stop] : null);
  return (
    <div role="group" aria-label="Time-series attribution" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span className="lbl dim">Colour</span>
        <div className="seg lbl" role="group" aria-label="Colour scale">
          <button type="button" aria-pressed={!sqrt} onClick={() => setSqrt(false)}>Linear</button>
          <button type="button" aria-pressed={sqrt} onClick={() => setSqrt(true)}>√ |rel|</button>
        </div>
        {marks && (
          <>
            <span className="lbl dim">Zoom</span>
            <div className="seg lbl" role="group" aria-label="Zoom">
              <button type="button" aria-pressed={!zoom} onClick={() => setZoom(false)}>Full</button>
              <button type="button" aria-pressed={zoom} onClick={() => setZoom(true)}>Around step</button>
            </div>
          </>
        )}
      </div>
      <Row labels={
        <div className="axis" style={{ position: "absolute", top: 14, height: H, left: 0, right: 0 }}>
          {spans.map((b, i) => (
            <span key={i} title={b.name} style={{ position: "absolute", left: 0, right: 0, top: `${(b.top + b.bottom) / 2}%`,
              transform: "translateY(-50%)", overflow: "hidden", textOverflow: "ellipsis" }}>{b.name}</span>
          ))}
        </div>
      }>
        <HeatCanvas grid={grid} signed label={label} exportName={exportName}
          readout={(r, c) => `${seriesRow(m, r)} · ${seriesCol(m, c + c0)} · rel ${num(m.map[r][c + c0], 3)}`}
          onPick={valuesHref && ((r, c) => setPick([r, c + c0]))} mark={pick && pick[1] >= c0 && pick[1] < c1 ? [pick[0], pick[1] - c0] : null}
          x={x} y={{ from: padEnd("0", yw), to: padEnd(String(m.channels - 1), yw), label: "channel · by block" }}
          style={{ height: H, imageRendering: "pixelated" }}>
          {spans.slice(1).map((b, i) => (
            <div key={i} style={{ position: "absolute", left: 0, right: 0, top: `${b.top}%`, height: 1, background: "var(--ink)", pointerEvents: "none" }} />
          ))}
          {bands(true)}
        </HeatCanvas>
      </Row>
      {pick && (
        <div className="lbl dim" style={{ display: "flex", gap: 12, alignItems: "baseline", flexWrap: "wrap" }}>
          <span>Picked · {seriesRow(m, pick[0])} · {seriesCol(m, pick[1])}</span>
          {cellHref && <a className="lbl chart-export" href={cellHref}>Show input values →</a>}
          <button type="button" className="lbl chart-export" onClick={() => setPick(null)}>Clear</button>
        </div>
      )}
      <Row>
        <Axes {...lineAxes}>
          <svg ref={line} width="100%" height={LH} viewBox={`0 0 ${W} ${LH}`} preserveAspectRatio="none" role="img"
            aria-label="Attribution over time" style={{ display: "block" }} onMouseLeave={() => setHover(null)}
            onMouseMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              setHover(Math.min(cols - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * cols))));
            }}>
            <line x1="0" x2={W} y1={LH / 2} y2={LH / 2} stroke="var(--line)" strokeDasharray="2 3" vectorEffect="non-scaling-stroke" />
            <path d={linePath(totals, px, py)} fill="none" stroke="var(--accent)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
            <line x1="0" x2={W} y1={LH - 0.5} y2={LH - 0.5} stroke="var(--ink)" vectorEffect="non-scaling-stroke" />
            {hover != null && <line x1={px(hover)} x2={px(hover)} y1="0" y2={LH} stroke="var(--ink)" vectorEffect="non-scaling-stroke" />}
          </svg>
          {bands(false)}
        </Axes>
        <Readout text={hover != null ? `${seriesCol(m, hover + c0)} · Σ ${sgn(totals[hover])}`
          : cols ? `Σ per column${m.stride > 1 ? ` of ${m.stride} steps` : ""} · peak ${seriesCol(m, peak + c0)} · Σ ${sgn(totals[peak])}` : null}
          label="Attribution over time"
          onExport={async () => { if (line.current) await exportSvg(line.current, pngName(exportName.replace(/\.png$/, "_time"), "time"), 2, axisText(lineAxes)); }} />
      </Row>
      {marks && <WindowLegend marks={marks} />}
      {marks && marks.windows.length > 0 && <WhereTable m={m} marks={marks} valuesHref={valuesHref} />}
      {marks?.inputHref && <a className="lbl chart-export" style={{ alignSelf: "flex-start" }} href={marks.inputHref}>Show on input →</a>}
      <BlockShares m={m} exportName={exportName.replace(/\.png$/, "_blocks")} />
      <span className="lbl dim">
        {m.channels} channels × {m.steps} steps{m.stride > 1 || m.row_stride > 1 ? ` → ${rows} × ${n} cells, sums of ${m.row_stride} × ${m.stride}` : ""} · colour ± max |cell|{sqrt ? ", √ scaled" : ""}
        {zoom && marks ? ` · zoomed to ${stepsLabel(c0 * m.stride, Math.min(m.steps, c1 * m.stride))}` : ""}{valuesHref ? " · click a cell for its input values" : ""}
      </span>
    </div>
  );
}

/** "Where the attribution is": per window (and outside all of them) its share of Σ|attribution| and signed Σ; a
 * window's `Values →` opens the input values it covers. */
function WhereTable({ m, marks, valuesHref }: { m: SeriesMap; marks: StepMarks; valuesHref?: (steps: [number, number], ch: [number, number] | null) => string | null }) {
  const rows = useMemo(() => windowShares(m, marks.windows), [m, marks.windows]);
  const th: React.CSSProperties = { padding: "4px 8px", borderBottom: "1px solid var(--ink)", fontWeight: 400, whiteSpace: "nowrap" };
  const td: React.CSSProperties = { padding: "4px 8px", borderBottom: "1px solid var(--line-soft)" };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <span className="lbl">Where the attribution is</span>
      <table aria-label="Where the attribution is" className="num" style={{ borderCollapse: "collapse", fontSize: 12 }}>
        <thead><tr>{["Window", "Steps", "Share Σ|a|", "Σ a", ""].map((h, j) => (
          <th key={j} className="lbl dim" style={{ ...th, textAlign: j < 2 ? "left" : "right" }}>{h}</th>
        ))}</tr></thead>
        <tbody>{rows.map((r, i) => {
          const href = r.kind === "outside" ? null : valuesHref?.([r.start, r.stop], blockChannels(m.blocks, r.blocks));
          return (
            <tr key={i}>
              <td style={td}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  {r.kind !== "outside" && <span style={{ ...OVER, position: "static", display: "inline-block", width: 12, height: 9, ...BAND[r.kind] }} />}
                  {r.label}
                </span>
              </td>
              <td style={td}>{r.kind === "outside" ? "—" : `${stepsLabel(r.start, r.stop)}${r.blocks ? ` · ${r.blocks.join(", ")}` : ""}`}</td>
              <td style={{ ...td, textAlign: "right" }}>{pct(r.share)}</td>
              <td style={{ ...td, textAlign: "right", color: r.sum < 0 ? "var(--negative-text)" : undefined }}>{sgn(r.sum)}</td>
              <td style={{ ...td, textAlign: "right" }}>{href && <a className="lbl chart-export" style={{ whiteSpace: "nowrap" }} href={href} aria-label={`Show input values · ${r.label}`}>Values →</a>}</td>
            </tr>
          );
        })}</tbody>
      </table>
    </div>
  );
}

/** Each block's share of Σ|attribution|: one row per block, name · bar (0–100 %) · percentage; categorical, so no axes;
 * drawn at 1:1 (≤ 380 px wide) so its text keeps the axis size. */
function BlockShares({ m, exportName }: { m: SeriesMap; exportName: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const W = 380, RH = 18, NAME = 112, BAR = 208, H = m.blocks.length * RH;
  const whole = m.blocks.reduce((s, b) => s + b.abs, 0);
  const b = hover == null ? null : m.blocks[hover];
  return (
    <div>
      <svg ref={ref} width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Block shares" style={{ display: "block", maxWidth: "100%" }}
        onMouseLeave={() => setHover(null)}>
        {m.blocks.map((k, i) => (
          <g key={i} onMouseEnter={() => setHover(i)}>
            <rect x={0} y={i * RH} width={W} height={RH} fill="transparent" />
            <text x={0} y={i * RH + 12.5} fontSize="10" fill="var(--ink-2)" style={{ fontFamily: "var(--font-mono)" }}>
              {k.name.length > 18 ? `${k.name.slice(0, 17)}…` : k.name}
            </text>
            <rect x={NAME} y={i * RH + 5} width={BAR} height={8} fill="var(--line-soft)" />
            <rect x={NAME} y={i * RH + 5} width={Math.max(0.5, k.share * BAR)} height={8} fill={hover === i ? "var(--accent)" : "var(--ink)"} />
            <text x={W} y={i * RH + 12.5} fontSize="10" textAnchor="end" fill="var(--ink)" style={{ fontFamily: "var(--font-mono)" }}>{pct(k.share)}</text>
          </g>
        ))}
      </svg>
      <Readout text={b ? `${b.name} · ${pct(b.share)} of Σ|a| · Σ ${sgn(b.sum)} · ${b.stop - b.start} channel${b.stop - b.start === 1 ? "" : "s"}`
        : `share of Σ|attribution| per block · Σ|a| ${num(whole, 3)}`}
        label="Block shares" onExport={async () => { if (ref.current) await exportSvg(ref.current, pngName(exportName, "Block shares"), 2); }} />
    </div>
  );
}
