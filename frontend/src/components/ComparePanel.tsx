// Compare mode: two nodes side by side on a shared scale, plus element-wise Δ = B − A statistics.
import { api } from "../lib/api";
import { fileName } from "../lib/exportPng";
import { displayName, num, pct, shapeShort } from "../lib/format";
import { cellOf, cellPixel, cellSpan, sameShape, sharedRange, unscale } from "../lib/tensorState";
import type { Cmap, GraphNode, Norm, RunMeta, TensorView } from "../lib/types";
import { useAsync } from "../lib/useAsync";
import { DivergingBars, DotMatrix, HeatCanvas, Histogram, LinePlot } from "./charts";

export function ComparePanel({ run, a, b, channel, pixel, norm, cmap, onPixel }: {
  run: RunMeta; a: GraphNode; b: GraphNode;
  channel: number | null; pixel: [number, number] | null; norm: Norm; cmap: Cmap;
  onPixel: (py: number, px: number) => void;
}) {
  const range = sameShape(a.shape, b.shape) ? sharedRange(a.stats, b.stats) : null;
  const opts = { channel, py: pixel?.[0] ?? null, px: pixel?.[1] ?? null, cmap, ...(range ? { norm: "global" as const, lo: range[0], hi: range[1] } : { norm }) };
  const deps = [run.id, channel, pixel?.[0], pixel?.[1], norm, cmap, range?.[0], range?.[1]];
  const ta = useAsync(() => api.tensor(run.id, a.output_op, opts), [a.output_op, ...deps]);
  const tb = useAsync(() => api.tensor(run.id, b.output_op, opts), [b.output_op, ...deps]);
  const diff = useAsync(() => api.diff(run.id, a.output_op, b.output_op), [run.id, a.output_op, b.output_op]);
  const d = diff.data?.diff;
  const ex = (op: string, chart: string) => fileName([run.id, op, channel == null ? null : "c" + channel, chart], "png").slice(0, -4);
  const col: React.CSSProperties = { padding: "16px 18px", display: "flex", flexDirection: "column", gap: 12, minWidth: 0 };

  return (
    <section className="scroll" style={{ flexGrow: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "14px 18px", borderBottom: "1px solid var(--line)", display: "flex", justifyContent: "space-between", gap: 12 }}>
        <span className="lbl">Compare</span>
        <span className="lbl dim">{range ? `Shared scale · ${num(range[0])} … ${num(range[1])}` : `Scales differ · shapes ${shapeShort(a.shape)} vs ${shapeShort(b.shape)}`}</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", flexGrow: 1 }}>
        <Side tag="A" node={a} t={ta.data} error={ta.error} range={range} onPixel={onPixel} ex={(c) => ex(a.output_op, c)} style={col} />
        <Side tag="B" node={b} t={tb.data} error={tb.error} range={range} onPixel={onPixel} ex={(c) => ex(b.output_op, c)} style={{ ...col, borderLeft: "1px solid var(--line)" }} />
        <div style={{ ...col, borderLeft: "1px solid var(--line)" }}>
          <span className="lbl">Δ = B − A</span>
          {diff.error && <span className="err mono">{diff.error}</span>}
          {diff.loading && !diff.data && <span className="lbl dim">Loading…</span>}
          {diff.data && !d && (
            <div className="empty lbl" style={{ padding: 0 }}>
              Shapes differ ({shapeShort(diff.data.shape_a)} vs {shapeShort(diff.data.shape_b)}) · no element-wise difference
            </div>
          )}
          {d && (
            <>
              <div className="rule-top">
                {([["Mean Δ", num(d.stats.mean, 4)], ["Std Δ", num(d.stats.std, 4)], ["Min Δ", num(d.stats.min, 4)], ["Max Δ", num(d.stats.max, 4)],
                  ["Max |Δ|", num(d.max_abs, 4)], ["Mean |Δ|", num(d.mean_abs, 4)], ["Cosine", d.cosine == null ? "—" : d.cosine.toFixed(4)],
                  ["Rel ℓ2 ‖Δ‖/‖A‖", num(d.rel_l2, 4)]] as const).map(([k, v]) => (
                  <div key={k} className="kv"><span className="lbl dim">{k}</span><span className="num">{v}</span></div>
                ))}
              </div>
              <span className="lbl">Δ distribution</span>
              <Histogram counts={d.hist.counts} range={d.hist.range} height={70} exportName={fileName([run.id, a.output_op, b.output_op, "delta-hist"], "png").slice(0, -4)} />
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function Side({ tag, node, t, error, range, onPixel, ex, style }: {
  tag: string; node: GraphNode; t: TensorView | null; error: string | null; range: [number, number] | null;
  onPixel: (py: number, px: number) => void; ex: (chart: string) => string; style: React.CSSProperties;
}) {
  const st = node.stats;
  return (
    <div style={style}>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <span className="num" style={{ fontSize: 15, fontWeight: 500, wordBreak: "break-all" }}>{tag} · {displayName(node)}</span>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          <span className="lbl pill">{node.type}</span><span className="lbl dim">{shapeShort(node.shape)}</span>
        </div>
      </div>
      {error && <span className="err mono">{error}</span>}
      {!t && !error && <span className="lbl dim">Loading…</span>}
      {t?.kind === "image" && (() => {
        const rows = t.dots.length, cols = t.dots[0]?.length ?? 0, [py, px] = t.vector.pixel;
        return (
          <>
            <span className="lbl dim">Channel c{t.channel} · @ ({py}, {px})</span>
            <DotMatrix grid={t.dots} signed={t.cmap === "div"} pitch={Math.min(18, Math.floor(260 / Math.max(1, cols)))}
              label={`${tag} channel ${t.channel} dot matrix`} exportName={ex("dots")}
              mark={[cellOf(t.height, rows, py), cellOf(t.width, cols, px)]}
              onPick={(r, c) => onPixel(cellPixel(t.height, rows, r), cellPixel(t.width, cols, c))}
              readout={(r, c, v) => `px (${cellSpan(t.height, rows, r)}, ${cellSpan(t.width, cols, c)}) · ${num(unscale(v, t.dots_range, t.cmap), 3)}`} />
            <span className="lbl dim">Row {t.profile.row} profile</span>
            <LinePlot values={t.profile.values} xLabel="col" height={80} exportName={ex("profile")} domain={range ?? undefined}
              mark={cellOf(t.width, t.profile.values.length, px)} />
          </>
        );
      })()}
      {t?.kind === "sequence" && (
        <div style={{ border: "1px solid var(--line)" }}>
          <HeatCanvas grid={t.heatmap} signed style={{ height: 220, imageRendering: "pixelated" }} label={`${tag} token by feature heatmap`}
            exportName={ex("heatmap")} readout={(r, c, v) => `cell (${r}, ${c}) · ${num(t.scale ? unscale(v, t.scale, "div") : v, 3)}`} />
        </div>
      )}
      {t?.kind === "vector" && <DivergingBars values={t.values} height={140} xLabel="index" exportName={ex("signal")}
        max={range ? Math.max(Math.abs(range[0]), Math.abs(range[1])) : undefined} />}
      {st && (
        <>
          <span className="lbl">Whole tensor</span>
          <Histogram counts={st.hist} range={st.hist_range} height={60} exportName={ex("hist")} />
          <div className="rule-top">
            {[["Mean", num(st.mean)], ["Std", num(st.std)], ["Min", num(st.min)], ["Max", num(st.max)], ["Zero", pct(st.zero_frac)]].map(([k, v]) => (
              <div key={k} className="kv"><span className="lbl dim">{k}</span><span className="num">{v}</span></div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
