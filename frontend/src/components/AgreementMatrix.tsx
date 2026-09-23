// n×n method-agreement matrix: accent/negative tone by value, the value printed in each cell, hover readout, PNG export.
import { useRef, useState } from "react";
import { exportSvg } from "../lib/exportPng";
import { matrixTone } from "../lib/traceState";
import type { AgreementBlock } from "../lib/types";
import { Readout, pngName } from "./charts";

export type AgreementMetric = "jaccard" | "spearman";

const SHORT: Record<string, string> = { gxa: "GXA", saliency: "SAL", ig: "IG", layer_ig: "LIG", layer_cond: "LCOND" };
const C = 40, HEAD_W = 48, HEAD_H = 18;
const f = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(2));

export function AgreementMatrix({ block, metric, exportName }: { block: AgreementBlock; metric: AgreementMetric; exportName?: string }) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const n = block.methods.length;
  const head = (i: number) => SHORT[block.methods[i]] ?? block.methods[i].toUpperCase();
  const values = block[metric];
  const label = "Method agreement";
  const text = hover && `${head(hover[0])} × ${head(hover[1])} · Jaccard@${block.k} ${f(block.jaccard[hover[0]][hover[1]])}`
    + ` · overlap ${f(block.overlap[hover[0]][hover[1]])} · Spearman ${f(block.spearman[hover[0]][hover[1]])}`;
  const mono = { fontFamily: "var(--font-mono)", fontSize: 10.5 };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <span className="lbl dim">{block.basis} · top-k = {block.k} of {block.n}</span>
      <svg ref={ref} width={HEAD_W + n * C + 1} height={HEAD_H + n * C + 1} role="img" aria-label={`${label} · ${metric}`}
        onMouseLeave={() => setHover(null)} style={{ display: "block" }}>
        {block.methods.map((_, i) => (
          <g key={i} style={{ ...mono, letterSpacing: "0.05em" }} fill="var(--ink-2)">
            <text x={HEAD_W + i * C + C / 2} y={HEAD_H - 6} textAnchor="middle">{head(i)}</text>
            <text x={HEAD_W - 8} y={HEAD_H + i * C + C / 2 + 4} textAnchor="end">{head(i)}</text>
          </g>
        ))}
        {values.map((row, r) => row.map((v, c) => {
          const diag = r === c, t = matrixTone(diag ? null : v);
          const x = HEAD_W + c * C + 0.5, y = HEAD_H + r * C + 0.5;
          return (
            <g key={`${r}-${c}`} onMouseEnter={() => setHover(diag ? null : [r, c])}>
              <rect x={x} y={y} width={C} height={C} fill={diag ? "var(--line-soft)" : "var(--panel)"} stroke="var(--line)" />
              {t.color && <rect x={x + 0.5} y={y + 0.5} width={C - 1} height={C - 1} fill={`var(--${t.color})`} fillOpacity={t.alpha} />}
              <text x={x + C / 2} y={y + C / 2 + 4} textAnchor="middle" fill="var(--ink)" stroke="var(--panel)" strokeWidth={diag ? 0 : 2.5}
                paintOrder="stroke" style={mono}>{diag ? "—" : f(v)}</text>
            </g>
          );
        }))}
      </svg>
      {/* the full readout is too wide for one line of the aside, so it wraps above the export row */}
      <div className="lbl dim" style={{ minHeight: 28 }}>{text || " "}</div>
      <Readout text={null} label={label}
        onExport={async () => { if (ref.current) await exportSvg(ref.current, pngName(exportName, label)); }} />
    </div>
  );
}
