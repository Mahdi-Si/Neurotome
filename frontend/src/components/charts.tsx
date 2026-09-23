// Small, dependency-free chart primitives in the house style: thin marks, recessive axes,
// ink for magnitude, accent for the selected/positive, rust for negative. Every chart has
// a hover readout (caption line) so values are never color-only.
import { useEffect, useRef, useState } from "react";
import { num } from "../lib/format";

const INK = "var(--ink)";
const ACCENT = "var(--accent)";
const NEG = "var(--negative)";

function Readout({ text }: { text: string | null }) {
  return <div className="lbl dim" style={{ height: 14, whiteSpace: "nowrap", overflow: "hidden" }}>{text ?? " "}</div>;
}

/** Histogram: one bar per bin, baseline rule, hover shows the bin range and count. */
export function Histogram({ counts, range, height = 64, highlight }: {
  counts: number[]; range: [number, number]; height?: number; highlight?: (i: number) => boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(1, ...counts);
  const n = counts.length;
  const w = 100 / n;
  const lo = range[0], step = (range[1] - range[0]) / n;
  return (
    <div>
      <svg width="100%" height={height} viewBox={`0 0 100 ${height}`} preserveAspectRatio="none"
        role="img" aria-label="Value distribution" onMouseLeave={() => setHover(null)}>
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
      <Readout text={hover == null ? `${num(range[0])} … ${num(range[1])}  ·  √count scale`
        : `[${num(lo + hover * step)}, ${num(lo + (hover + 1) * step)})  ·  ${counts[hover]}`} />
    </div>
  );
}

/** Dot matrix: circle size encodes magnitude. Unsigned grid in [0,1]; signed grid in [-1,1]. */
export function DotMatrix({ grid, pitch = 18, signed = false, label = "Dot matrix" }: {
  grid: number[][]; pitch?: number; signed?: boolean; label?: string;
}) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const rows = grid.length, cols = grid[0]?.length ?? 0;
  return (
    <div>
      <svg width={cols * pitch} height={rows * pitch} role="img" aria-label={label}
        onMouseLeave={() => setHover(null)} style={{ display: "block", margin: "0 auto" }}>
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
            <g key={`${r}-${c}`} onMouseEnter={() => setHover([r, c])}>
              <rect x={c * pitch} y={r * pitch} width={pitch} height={pitch} fill="transparent" />
              <circle cx={c * pitch + pitch / 2} cy={r * pitch + pitch / 2} r={d / 2 - 0.5}
                fill={fill} stroke={stroke} strokeWidth={hover && hover[0] === r && hover[1] === c ? 2 : 1} />
            </g>
          );
        }))}
      </svg>
      <Readout text={hover ? `cell (${hover[0]}, ${hover[1]}) · ${num(grid[hover[0]][hover[1]], 3)}` : null} />
    </div>
  );
}

/** Line plot with a soft area fill and a hover crosshair. */
export function LinePlot({ values, height = 96, xLabel = "index" }: { values: number[]; height?: number; xLabel?: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 380;
  const lo = Math.min(0, ...values), hi = Math.max(...values, lo + 1e-6);
  const x = (i: number) => (values.length <= 1 ? 0 : (i / (values.length - 1)) * W);
  const y = (v: number) => height - 3 - ((v - lo) / (hi - lo)) * (height - 8);
  const line = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join("");
  const area = `${line}L${W} ${y(lo)}L0 ${y(lo)}Z`;
  return (
    <div>
      <svg width="100%" height={height} viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" role="img"
        aria-label="Line plot" onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          setHover(Math.round(((e.clientX - r.left) / r.width) * (values.length - 1)));
        }}>
        <path d={area} fill="var(--accent-soft)" />
        {lo < 0 && <line x1="0" x2={W} y1={y(0)} y2={y(0)} stroke="var(--line)" strokeDasharray="2 3" vectorEffect="non-scaling-stroke" />}
        <path d={line} fill="none" stroke={ACCENT} strokeWidth="2" vectorEffect="non-scaling-stroke" />
        <line x1="0" x2={W} y1={height - 0.5} y2={height - 0.5} stroke={INK} vectorEffect="non-scaling-stroke" />
        {hover != null && values[hover] != null && (
          <line x1={x(hover)} x2={x(hover)} y1="0" y2={height} stroke={INK} strokeWidth="1" vectorEffect="non-scaling-stroke" />
        )}
      </svg>
      <Readout text={hover != null && values[hover] != null ? `${xLabel} ${hover} · ${num(values[hover], 3)}` : null} />
    </div>
  );
}

/** Diverging bars around a zero rule: ink/accent up, rust down. */
export function DivergingBars({ values, height = 110, highlight, xLabel = "channel", onPick }: {
  values: number[]; height?: number; highlight?: number | null; xLabel?: string; onPick?: (i: number) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const m = Math.max(1e-9, ...values.map(Math.abs));
  const n = values.length, w = 100 / n, mid = height / 2;
  return (
    <div>
      <svg width="100%" height={height} viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" role="img"
        aria-label="Values per channel" onMouseLeave={() => setHover(null)}>
        {values.map((v, i) => {
          const h = (Math.abs(v) / m) * (mid - 2);
          const color = i === highlight ? ACCENT : v >= 0 ? INK : NEG;
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onClick={() => onPick?.(i)} style={{ cursor: onPick ? "pointer" : undefined }}>
              <rect x={i * w} y={0} width={w} height={height} fill="transparent" />
              <rect x={i * w + (n > 64 ? 0 : w * 0.1)} y={v >= 0 ? mid - h : mid} width={n > 64 ? w : w * 0.8}
                height={Math.max(h, 0.5)} fill={hover === i ? ACCENT : color} />
            </g>
          );
        })}
        <line x1="0" x2="100" y1={mid} y2={mid} stroke={INK} vectorEffect="non-scaling-stroke" />
      </svg>
      <Readout text={hover != null ? `${xLabel} ${hover} · ${num(values[hover], 3)}` : null} />
    </div>
  );
}

/** Scatter of 2D points; color (0..1) drives opacity, top values in accent. */
export function Scatter({ points, color, height = 180 }: { points: [number, number][]; color: number[]; height?: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 320;
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1e-9), y0 = Math.min(...ys), y1 = Math.max(...ys, y0 + 1e-9);
  const px = (v: number) => 6 + ((v - x0) / (x1 - x0)) * (W - 12);
  const py = (v: number) => height - 6 - ((v - y0) / (y1 - y0)) * (height - 12);
  return (
    <div>
      <svg width="100%" height={height} viewBox={`0 0 ${W} ${height}`} role="img" aria-label="PCA projection"
        onMouseLeave={() => setHover(null)} style={{ borderLeft: "1px solid var(--ink)", borderBottom: "1px solid var(--ink)" }}>
        {points.map((p, i) => (
          <circle key={i} cx={px(p[0])} cy={py(p[1])} r={hover === i ? 5 : 3}
            fill={(color[i] ?? 0) > 0.6 ? ACCENT : INK} fillOpacity={0.18 + (color[i] ?? 0) * 0.82}
            stroke="var(--panel)" strokeWidth={hover === i ? 2 : 0} onMouseEnter={() => setHover(i)} />
        ))}
      </svg>
      <Readout text={hover != null ? `point ${hover} · (${num(points[hover][0])}, ${num(points[hover][1])}) · value ${num(color[hover])}` : null} />
    </div>
  );
}

/** Canvas heatmap. Unsigned grid in [0,1] -> ink on paper; signed [-1,1] -> accent / rust. */
export function HeatCanvas({ grid, signed = false, style, label }: {
  grid: number[][]; signed?: boolean; style?: React.CSSProperties; label?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !grid.length) return;
    const h = grid.length, w = grid[0].length;
    c.width = w; c.height = h;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(w, h);
    for (let r = 0; r < h; r++) for (let q = 0; q < w; q++) {
      const v = grid[r][q], i = (r * w + q) * 4;
      let rgb: [number, number, number];
      if (signed) {
        const a = Math.min(1, Math.abs(v));
        const pole = v >= 0 ? [59, 75, 219] : [217, 83, 30];
        rgb = [250 + (pole[0] - 250) * a, 250 + (pole[1] - 250) * a, 248 + (pole[2] - 248) * a];
      } else {
        const a = Math.min(1, Math.max(0, v));
        rgb = [250 - (250 - 17) * a, 250 - (250 - 17) * a, 248 - (248 - 17) * a];
      }
      img.data[i] = rgb[0]; img.data[i + 1] = rgb[1]; img.data[i + 2] = rgb[2]; img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [grid, signed]);
  return <canvas ref={ref} role="img" aria-label={label ?? "Heatmap"} style={{ width: "100%", display: "block", ...style }} />;
}
