// Small, dependency-free chart primitives in the house style: thin marks, recessive axes,
// ink for magnitude, accent for the selected/positive, rust for negative. Every chart has
// a hover readout (caption line) so values are never color-only, with a PNG export button on its right.
import { useEffect, useRef, useState } from "react";
import { axisGutters, exportCanvas, exportSvg, fileName, type AxisPair } from "../lib/exportPng";
import { num } from "../lib/format";
import { cellOf, cellPixel, cellSpan, equalAspect, lineDomain } from "../lib/tensorState";
import { parseColor, useTokens } from "../lib/theme";

const INK = "var(--ink)";
const ACCENT = "var(--accent)";
const NEG = "var(--negative)";

type Cell = [number, number];

/** `<exportName ?? slug(label)>.png` */
export const pngName = (exportName: string | undefined, label: string) =>
  fileName([(exportName ?? label.toLowerCase()).replace(/\.png$/, "")], "png");

/** Quoted parts ("cat", token text) keep their case inside an uppercase .lbl line. */
const keepCase = (text: string) =>
  text.split(/("[^"]*")/).map((p, i) => (i % 2 ? <span key={i} style={{ textTransform: "none" }}>{p}</span> : p));

/** Readout line: hover text on the left, a PNG export button on the right. */
export function Readout({ text, label, onExport }: { text: string | null; label?: string; onExport?: () => Promise<void> }) {
  return (
    <div className="lbl dim" style={{ display: "flex", gap: 8, height: 14 }}>
      <span style={{ flex: 1, minWidth: 0, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {text == null ? " " : keepCase(text)}
      </span>
      {onExport && (
        <button type="button" className="chart-export" aria-label={`Export ${label ?? "chart"} as PNG`}
          onClick={() => void onExport().catch((e) => console.error("PNG export failed", e))}>PNG</button>
      )}
    </div>
  );
}

/** The grid cell under the pointer: r = floor(y/h·rows), c = floor(x/w·cols), clamped. */
export function cellAt(e: React.MouseEvent, rows: number, cols: number): Cell {
  const b = e.currentTarget.getBoundingClientRect();
  const k = (t: number, n: number) => Math.min(n - 1, Math.max(0, Math.floor((t || 0) * n)));
  return [k((e.clientY - b.top) / b.height, rows), k((e.clientX - b.left) / b.width, cols)];
}

export interface AxisRange { from: number | string; to: number | string; label: string }
type Ranges = { x?: AxisRange; y?: AxisRange };

/** Axis ranges as printed: integers (indices, counts) as they are, other numbers through num(), strings unchanged. */
export const axisText = ({ x, y }: Ranges): AxisPair => {
  const v = (n: number | string) => (typeof n === "string" ? n : Number.isInteger(n) && Math.abs(n) < 1e6 ? String(n) : num(n));
  const t = (a?: AxisRange) => a && { from: v(a.from), to: v(a.to), label: a.label };
  return { x: t(x), y: t(y) };
};

/** Axis ranges around a chart: the y label on its own row above (horizontal, left-aligned, so a short chart never clips
 * it), the y numbers in a left gutter (`y.from` at the TOP, `y.to` at the bottom), x on a row below (`x.from` left,
 * label centred, `x.to` right). Integers print as they are, other numbers through num(), strings unchanged. Mono
 * 10 px `ink-2`, 1 px `line` tick at each end, no gridlines. Either axis may be omitted. `fit`: shrink to the chart's
 * own width and centre (fixed-size charts such as the dot matrix). Gutter sizes come from axisGutters(), which the
 * PNG export shares, so an exported chart carries the same ranges in the same places. */
export function Axes({ x, y, children, fit }: Ranges & { children: React.ReactNode; fit?: boolean }) {
  const t = axisText({ x, y }), g = axisGutters(t);
  const tick = (o: React.CSSProperties): React.CSSProperties => ({ position: "absolute", background: "var(--line)", ...o });
  return (
    <div style={{ display: "grid", gridTemplateColumns: y ? `${g.left}px minmax(0, 1fr)` : "minmax(0, 1fr)",
      ...(fit ? { width: "fit-content", maxWidth: "100%", margin: "0 auto" } : {}) }}>
      {t.y && (
        <>
          <div className="axis" style={{ gridColumn: "1 / -1", height: g.top, overflow: "hidden", textOverflow: "ellipsis" }}>{t.y.label}</div>
          <div className="axis" style={{ position: "relative", display: "flex", flexDirection: "column", justifyContent: "space-between",
            alignItems: "flex-end", paddingRight: 8 }}>
            <span>{t.y.from}</span><span>{t.y.to}</span>
            <i style={tick({ right: 0, top: 0, width: 4, height: 1 })} /><i style={tick({ right: 0, bottom: 0, width: 4, height: 1 })} />
          </div>
        </>
      )}
      <div style={{ minWidth: 0, position: "relative" }}>{children}</div>
      {t.x && (
        <>
          {t.y && <span />}
          <div className="axis" style={{ position: "relative", height: g.bottom, paddingTop: 5, display: "grid",
            gridTemplateColumns: "auto minmax(0, 1fr) auto", columnGap: 6 }}>
            <span>{t.x.from}</span>
            <span style={{ textAlign: "center", overflow: "hidden", textOverflow: "ellipsis" }}>{t.x.label}</span>
            <span>{t.x.to}</span>
            <i style={tick({ left: 0, top: 0, width: 1, height: 4 })} /><i style={tick({ right: 0, top: 0, width: 1, height: 4 })} />
          </div>
        </>
      )}
    </div>
  );
}

const svgExport = (ref: React.RefObject<SVGSVGElement | null>, exportName: string | undefined, label: string, axes?: Ranges) =>
  async () => { if (ref.current) await exportSvg(ref.current, pngName(exportName, label), 2, axes && axisText(axes)); };

/** Histogram: one bar per bin, baseline rule, hover shows the bin range and count. Axes: x = the value range
 * (`xLabel` names the quantity), y = max count … 0 on the √count scale. */
export function Histogram({ counts, range, height = 64, highlight, exportName, xLabel = "value" }: {
  counts: number[]; range: [number, number]; height?: number; highlight?: (i: number) => boolean; exportName?: string;
  xLabel?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const max = Math.max(1, ...counts);
  const n = counts.length;
  const w = 100 / n;
  const lo = range[0], step = (range[1] - range[0]) / n;
  const axes = { x: { from: range[0], to: range[1], label: xLabel }, y: { from: Math.max(0, ...counts), to: 0, label: "√count" } };
  return (
    <div>
      <Axes {...axes}>
      <svg ref={ref} width="100%" height={height} viewBox={`0 0 100 ${height}`} preserveAspectRatio="none"
        role="img" aria-label="Value distribution" onMouseLeave={() => setHover(null)} style={{ display: "block" }}>
        {counts.map((c, i) => {
          const h = c === 0 ? 0 : Math.max(1, (Math.sqrt(c / max)) * (height - 2));
          return (
            <g key={i} onMouseEnter={() => setHover(i)}>
              <rect x={i * w} y={0} width={w} height={height} fill="transparent" />
              <rect x={i * w + w * 0.12} y={height - h} width={w * 0.76} height={h}
                fill={hover === i || highlight?.(i) ? ACCENT : INK} />
            </g>
          );
        })}
        <line x1="0" x2="100" y1={height - 0.5} y2={height - 0.5} stroke={INK} strokeWidth="1" vectorEffect="non-scaling-stroke" />
      </svg>
      </Axes>
      <Readout text={hover == null ? `${n} bins  ·  √count scale`
        : `[${num(lo + hover * step)}, ${num(lo + (hover + 1) * step)}${hover === n - 1 ? "]" : ")"}  ·  ${counts[hover]}`}
        label="Value distribution" onExport={svgExport(ref, exportName, "Value distribution", axes)} />
    </div>
  );
}

/** Dot matrix: circle size encodes magnitude. Unsigned grid in [0,1]; signed grid in [-1,1]. */
export function DotMatrix({ grid, pitch = 18, signed = false, label = "Dot matrix", exportName, onPick, mark, readout, selected, x, y }: {
  grid: number[][]; pitch?: number; signed?: boolean; label?: string; exportName?: string;
  onPick?: (r: number, c: number) => void; mark?: Cell | null; readout?: (r: number, c: number, v: number) => string;
  selected?: Cell[];   // cells framed by a 1px accent square (the position picker)
} & Ranges) {   // x / y: the real (unpooled) column / row ranges the grid covers
  const [hover, setHover] = useState<Cell | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const rows = grid.length, cols = grid[0]?.length ?? 0;
  const text = (r: number, c: number, v: number) => (readout ? readout(r, c, v) : `cell (${r}, ${c}) · ${num(v, 3)}`);
  const hv = hover ? grid[hover[0]]?.[hover[1]] : undefined;
  return (
    <div>
      <Axes x={x} y={y} fit>
      <svg ref={ref} width={cols * pitch} height={rows * pitch} role="img" aria-label={label}
        onMouseLeave={() => setHover(null)}
        style={{ display: "block", margin: "0 auto", cursor: onPick ? "crosshair" : undefined }}>
        {grid.map((row, r) => row.map((v, c) => {
          const a = Math.min(1, Math.abs(v));
          const d = 2.5 + a * (pitch - 5);
          let fill = "transparent", stroke = INK;
          if (signed) {
            if (v > 0.45) { fill = ACCENT; stroke = ACCENT; }
            else if (v < -0.3) { stroke = NEG; fill = v < -0.6 ? NEG : "transparent"; }
            else if (a < 0.08) stroke = "var(--line-faint)";
          } else if (v > 0.72) { fill = ACCENT; stroke = ACCENT; }
          else if (v > 0.4) fill = INK;
          return (
            <g key={`${r}-${c}`} onMouseEnter={() => setHover([r, c])} onClick={onPick && (() => onPick(r, c))}>
              <rect x={c * pitch} y={r * pitch} width={pitch} height={pitch} fill="transparent" />
              <circle cx={c * pitch + pitch / 2} cy={r * pitch + pitch / 2} r={d / 2 - 0.5}
                fill={fill} stroke={stroke} strokeWidth={hover && hover[0] === r && hover[1] === c ? 2 : 1} />
            </g>
          );
        }))}
        {selected?.map(([r, c]) => (
          <rect key={`s${r}-${c}`} x={c * pitch + 0.5} y={r * pitch + 0.5} width={pitch - 1} height={pitch - 1}
            fill="none" stroke={ACCENT} strokeWidth="1" pointerEvents="none" />
        ))}
        {mark && mark[0] >= 0 && mark[0] < rows && mark[1] >= 0 && mark[1] < cols && (
          <g pointerEvents="none" fill="none" stroke={ACCENT} strokeWidth="1">
            <line x1="0" x2={cols * pitch} y1={(mark[0] + 0.5) * pitch} y2={(mark[0] + 0.5) * pitch}
              strokeOpacity="0.5" shapeRendering="crispEdges" />
            <line x1={(mark[1] + 0.5) * pitch} x2={(mark[1] + 0.5) * pitch} y1="0" y2={rows * pitch}
              strokeOpacity="0.5" shapeRendering="crispEdges" />
            <circle cx={(mark[1] + 0.5) * pitch} cy={(mark[0] + 0.5) * pitch} r={pitch / 2 - 0.5} />
          </g>
        )}
      </svg>
      </Axes>
      <Readout text={hover && hv != null ? text(hover[0], hover[1], hv) : null}
        label={label} onExport={svgExport(ref, exportName, label, { x, y })} />
    </div>
  );
}

/** `"cat"` for point i when labels name it, else "" (readouts: `token 3 "cat" · 12.3`). */
const named = (labels: string[] | undefined, i: number) => (labels?.[i] != null ? ` "${labels[i]}"` : "");

/** `i`, or the span of source positions point i covers when `len` positions were pooled into n (`652–653`). */
const at = (len: number | undefined, n: number, i: number) => (len != null && len > n ? cellSpan(len, n, i) : `${i}`);

/** Line plot with a soft area fill, a hover crosshair and an optional accent rule at `mark`; `labels` name the x positions.
 * `len`: the unpooled length when `values` are block means, so the readout names source positions. */
export function LinePlot({ values, height = 96, xLabel = "index", yLabel = "value", exportName, mark, domain, labels, len }: {
  values: number[]; height?: number; xLabel?: string; yLabel?: string; exportName?: string; mark?: number | null;
  domain?: [number, number]; labels?: string[]; len?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const W = 380;
  const [lo, hi] = lineDomain(values, domain);
  const axes = values.length ? { x: { from: 0, to: (len ?? values.length) - 1, label: xLabel }, y: { from: hi, to: lo, label: yLabel } } : {};
  const x = (i: number) => (values.length <= 1 ? W / 2 : (i / (values.length - 1)) * W);
  const y = (v: number) => height - 3 - ((v - lo) / (hi - lo)) * (height - 8);
  const line = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join("");
  const area = values.length > 1 ? `${line}L${W} ${y(lo)}L0 ${y(lo)}Z` : "";   // a single point has no area: it gets a dot
  return (
    <div>
      <Axes {...axes}>
      <svg ref={ref} width="100%" height={height} viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" role="img"
        aria-label="Line plot" onMouseLeave={() => setHover(null)} style={{ display: "block" }}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          setHover(Math.round(((e.clientX - r.left) / r.width) * (values.length - 1)));
        }}>
        <path d={area} fill="var(--accent-soft)" />
        {lo < 0 && <line x1="0" x2={W} y1={y(0)} y2={y(0)} stroke="var(--line)" strokeDasharray="2 3" vectorEffect="non-scaling-stroke" />}
        <path d={line} fill="none" stroke={ACCENT} strokeWidth="2" vectorEffect="non-scaling-stroke" />
        {values.length === 1 && <line x1={W / 2} x2={W / 2} y1={y(values[0])} y2={y(values[0])} stroke={ACCENT} strokeWidth="6"
          strokeLinecap="round" vectorEffect="non-scaling-stroke" />}
        <line x1="0" x2={W} y1={height - 0.5} y2={height - 0.5} stroke={INK} vectorEffect="non-scaling-stroke" />
        {mark != null && mark >= 0 && mark < values.length && (
          <line x1={x(mark)} x2={x(mark)} y1="0" y2={height} stroke={ACCENT} strokeWidth="1" vectorEffect="non-scaling-stroke" />
        )}
        {hover != null && values[hover] != null && (
          <line x1={x(hover)} x2={x(hover)} y1="0" y2={height} stroke={INK} strokeWidth="1" vectorEffect="non-scaling-stroke" />
        )}
      </svg>
      </Axes>
      <Readout text={hover != null && values[hover] != null ? `${xLabel} ${at(len, values.length, hover)}${named(labels, hover)} · ${num(values[hover], 3)}`
        : values.length ? `data ${num(Math.min(...values), 3)} … ${num(Math.max(...values), 3)}` : null}
        label="Line plot" onExport={svgExport(ref, exportName, "Line plot", axes)} />
    </div>
  );
}

/** Diverging bars around a zero rule: ink/accent up, rust down. With `len` (> values.length) the bars are block means
 * of `len` source positions: `highlight` and `onPick` speak source indices and the readout names the span. */
export function DivergingBars({ values, height = 110, highlight, xLabel = "channel", yLabel = "value", onPick, exportName, max, labels, len }: {
  values: number[]; height?: number; highlight?: number | null; xLabel?: string; yLabel?: string; onPick?: (i: number) => void; exportName?: string;
  labels?: string[];   // names of the bars in the readout (token strings)
  max?: number;   // fixed |value| for full height (shared scale); default: the largest |value|
  len?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const m = Math.max(1e-9, max ?? 0, ...values.map(Math.abs));
  const n = values.length, w = 100 / n, mid = height / 2;
  const pooled = len != null && len > n;
  const hi = highlight == null ? null : pooled ? cellOf(len, n, highlight) : highlight;
  const axes = n ? { x: { from: 0, to: (len ?? n) - 1, label: xLabel }, y: { from: m, to: -m, label: yLabel } } : {};
  return (
    <div>
      <Axes {...axes}>
      <svg ref={ref} width="100%" height={height} viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" role="img"
        aria-label="Values per channel" onMouseLeave={() => setHover(null)} style={{ display: "block" }}>
        {values.map((v, i) => {
          const h = (Math.abs(v) / m) * (mid - 2);
          const color = i === hi ? ACCENT : v >= 0 ? INK : NEG;
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onClick={() => onPick?.(pooled ? cellPixel(len, n, i) : i)} style={{ cursor: onPick ? "pointer" : undefined }}>
              <rect x={i * w} y={0} width={w} height={height} fill="transparent" />
              <rect x={i * w + (n > 64 ? 0 : w * 0.1)} y={v >= 0 ? mid - h : mid} width={n > 64 ? w : w * 0.8}
                height={Math.max(h, 0.5)} fill={hover === i ? ACCENT : color} />
            </g>
          );
        })}
        <line x1="0" x2="100" y1={mid} y2={mid} stroke={INK} vectorEffect="non-scaling-stroke" />
      </svg>
      </Axes>
      <Readout text={hover != null ? `${xLabel} ${at(len, n, hover)}${named(labels, hover)} · ${num(values[hover], 3)}`
        : pooled ? `${len} → ${n} bars · block means` : null}
        label="Values per channel" onExport={svgExport(ref, exportName, "Values per channel", axes)} />
    </div>
  );
}

/** Scatter of 2D points on one scale for both axes (PCA distances stay true); color (0..1) drives opacity, top values
 * in accent. `name(i)` names point i in the readout (default `point i`); `caption` is the idle readout (what the shade means). */
export function Scatter({ points, color, height = 180, exportName, name, caption, xLabel = "PC1", yLabel = "PC2" }: {
  points: [number, number][]; color: number[]; height?: number; exportName?: string;
  name?: (i: number) => string; caption?: string; xLabel?: string; yLabel?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const W = 320;
  const { px, py, xRange, yRange } = equalAspect(points, W, height, 6);
  const axes = points.length ? { x: { from: xRange[0], to: xRange[1], label: xLabel }, y: { from: yRange[0], to: yRange[1], label: yLabel } } : {};
  return (
    <div>
      <Axes {...axes}>
      {/* the box keeps the viewBox's aspect, so one data unit is as long on x as on y and the axis ends sit on its edges */}
      <svg ref={ref} width="100%" viewBox={`0 0 ${W} ${height}`} role="img" aria-label="PCA projection"
        onMouseLeave={() => setHover(null)} style={{ display: "block", aspectRatio: `${W} / ${height}` }}>
        <path d={`M0.5 0V${height - 0.5}H${W}`} fill="none" stroke={INK} vectorEffect="non-scaling-stroke" />
        {points.map((p, i) => (
          <circle key={i} cx={px(p[0])} cy={py(p[1])} r={hover === i ? 5 : 3}
            fill={(color[i] ?? 0) > 0.6 ? ACCENT : INK} fillOpacity={0.18 + (color[i] ?? 0) * 0.82}
            stroke="var(--panel)" strokeWidth={hover === i ? 2 : 0} onMouseEnter={() => setHover(i)} />
        ))}
      </svg>
      </Axes>
      <Readout text={hover != null ? `${name ? name(hover) : `point ${hover}`} · PC (${num(points[hover][0])}, ${num(points[hover][1])})` : caption ?? null}
        label="PCA projection" onExport={svgExport(ref, exportName, "PCA projection", axes)} />
    </div>
  );
}

/** Crosshair over a positioned box: ring on the cell, half-opacity lines through its centre (1px divs). */
function CrossMarks({ mark: [r, c], rows, cols }: { mark: Cell; rows: number; cols: number }) {
  const box = (o: React.CSSProperties): React.CSSProperties => ({ position: "absolute", pointerEvents: "none", ...o });
  const cy = `calc(${((r + 0.5) / rows) * 100}% - 0.5px)`, cx = `calc(${((c + 0.5) / cols) * 100}% - 0.5px)`;
  return (
    <>
      <div style={box({ left: 0, right: 0, top: cy, height: 1, background: ACCENT, opacity: 0.5 })} />
      <div style={box({ top: 0, bottom: 0, left: cx, width: 1, background: ACCENT, opacity: 0.5 })} />
      <div style={box({ left: `${(c / cols) * 100}%`, top: `${(r / rows) * 100}%`, width: `${100 / cols}%`,
        height: `${100 / rows}%`, border: `1px solid ${ACCENT}`, borderRadius: "50%" })} />
    </>
  );
}

/** Canvas heatmap. Unsigned grid in [0,1] -> ink on panel; signed [-1,1] -> accent / rust (current theme's tokens).
 * `overlay`: transparent ground, accent (v ≥ 0) / negative (v < 0) at alpha |v|, drawn over the input image.
 * `children`: marks positioned over the map (e.g. block separators), not exported. */
export function HeatCanvas({ grid, signed = false, style, label, exportName, overlay = false, onPick, mark, readout, x, y, children }: {
  grid: number[][]; signed?: boolean; style?: React.CSSProperties; label?: string; exportName?: string;
  overlay?: boolean; onPick?: (r: number, c: number) => void; mark?: Cell | null;
  readout?: (r: number, c: number, v: number) => string; children?: React.ReactNode;
} & Ranges) {   // x / y: the real (unpooled) column / row ranges the map covers
  const ref = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<Cell | null>(null);
  const rows = grid.length, cols = grid[0]?.length ?? 0;
  const name = label ?? "Heatmap";
  const tk = useTokens();
  useEffect(() => {
    const c = ref.current;
    if (!c || !rows || !cols) return;
    const [PAPER, INK_RGB, ACCENT_RGB, NEG_RGB] = [tk.panel, tk.ink, tk.accent, tk.negative].map(parseColor);
    const h = grid.length, w = grid[0].length;
    c.width = w; c.height = h;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(w, h);
    for (let r = 0; r < h; r++) for (let q = 0; q < w; q++) {
      const v = grid[r][q], i = (r * w + q) * 4;
      let rgb: number[], alpha = 255;
      if (overlay) {
        rgb = v >= 0 ? ACCENT_RGB : NEG_RGB;
        alpha = Math.round(Math.min(1, Math.abs(v) || 0) * 255);
      } else if (signed) {
        const a = Math.min(1, Math.abs(v)), pole = v >= 0 ? ACCENT_RGB : NEG_RGB;
        rgb = PAPER.map((p, k) => p + (pole[k] - p) * a);
      } else {
        const a = Math.min(1, Math.max(0, v));
        rgb = PAPER.map((p, k) => p - (p - INK_RGB[k]) * a);
      }
      img.data[i] = rgb[0]; img.data[i + 1] = rgb[1]; img.data[i + 2] = rgb[2]; img.data[i + 3] = alpha;
    }
    ctx.putImageData(img, 0, 0);
  }, [grid, rows, cols, signed, overlay, tk]);
  const hv = hover ? grid[hover[0]]?.[hover[1]] : undefined;
  const map = (
    <div style={{ position: "relative", height: "100%" }}>
      <canvas ref={ref} role="img" aria-label={name}
        onMouseMove={readout && ((e) => setHover(cellAt(e, rows, cols)))}
        onMouseLeave={readout && (() => setHover(null))}
        onClick={onPick && ((e) => onPick(...cellAt(e, rows, cols)))}
        style={{ width: "100%", display: "block", cursor: onPick ? "crosshair" : undefined, ...style }} />
      {mark && rows > 0 && cols > 0 && <CrossMarks mark={mark} rows={rows} cols={cols} />}
      {children}
    </div>
  );
  const body = x || y ? <Axes x={x} y={y}>{map}</Axes> : map;
  if (!readout) return body;
  return (
    <div>
      {body}
      <Readout text={hover && hv != null ? readout(hover[0], hover[1], hv) : null} label={name}
        onExport={async () => { if (ref.current) await exportCanvas(ref.current, pngName(exportName, name), 480, axisText({ x, y })); }} />
    </div>
  );
}
