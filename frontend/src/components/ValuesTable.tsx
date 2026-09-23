// Raw values of a capped slice: expression input, paged mono table, capped .npy download.
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { download, fileName } from "../lib/exportPng";
import { shape as fmtShape } from "../lib/format";
import { errText, pageCount, slicePage } from "../lib/tensorState";
import type { SliceResult, SliceValue } from "../lib/types";

const cellStyle = (v: SliceValue): React.CSSProperties => ({
  fontSize: 11, textAlign: "right", padding: "3px 8px", borderBottom: "1px solid var(--line-soft)", whiteSpace: "nowrap",
  color: typeof v === "string" || v < 0 ? "var(--negative-text)" : undefined,
});
const head: React.CSSProperties = { position: "sticky", background: "var(--panel)", padding: "3px 8px", textAlign: "right", fontWeight: 400 };

export function ValuesTable({ runId, op, shape, initial, iv = null }: { runId: string; op: string; shape: number[]; initial: string; iv?: string | null }) {
  const [expr, setExpr] = useState(initial);
  const [res, setRes] = useState<SliceResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [page, setPage] = useState(0);

  const run = <T,>(p: Promise<T>, ok: (v: T) => void) => {
    setBusy(true); setErr(null);
    p.then(ok, (e) => setErr(errText(e))).finally(() => setBusy(false));
  };
  const show = () => run(api.slice(runId, op, expr, iv), (r) => { setRes(r); setPage(0); });
  const npy = () => run(api.sliceNpy(runId, op, expr, iv), (b) => download(b, fileName([runId, iv, op, expr], "npy")));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(show, []);

  const pages = res ? pageCount(res) : 0;
  const pg = res ? slicePage(res, Math.min(page, pages - 1)) : null;
  return (
    <div style={{ border: "1px solid var(--line)", background: "var(--panel)", padding: 14, display: "flex", flexDirection: "column", gap: 10, minWidth: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
        <span className="lbl">Values</span><span className="lbl dim">Tensor {fmtShape(shape)}</span>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <input className="field mono" style={{ width: "100%" }} value={expr} placeholder="[0, 12, 20:30, :]" aria-label="Slice expression"
          onChange={(e) => setExpr(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !busy) show(); }} />
        <button type="button" className="btn solid" onClick={show} disabled={busy}>Show</button>
        <button type="button" className="btn" style={{ whiteSpace: "nowrap" }} onClick={npy} disabled={busy} aria-label="Download slice as .npy">.npy ↓</button>
      </div>
      {err && <span className="err mono">{err}</span>}
      {busy && <span className="lbl dim">Loading…</span>}
      {res && (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
            <span className="lbl dim">{res.key} · shape {res.shape.length ? res.shape.join("×") : "scalar"}{res.axes.length ? ` · axes ${res.axes.join(", ")}` : ""}</span>
            {pg && pages > 1 && (
              <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span className="lbl dim">Page</span>
                <select className="field mono" value={Math.min(page, pages - 1)} onChange={(e) => setPage(Number(e.target.value))}>
                  {Array.from({ length: pages }, (_, i) => <option key={i} value={i}>{slicePage(res, i)!.label}</option>)}
                </select>
              </label>
            )}
          </div>
          {res.truncated && (
            <span className="lbl dim">Showing {res.shape.join("×")} of {res.full_shape.join("×")} · narrow the slice to see the rest</span>
          )}
          <div className="scroll" style={{ maxHeight: 320, overflow: "auto", border: "1px solid var(--line-soft)" }}>
            {res.values.length === 0 && <div className="lbl dim" style={{ padding: 8 }}>Empty slice · shape {res.shape.join("×")}</div>}
            {res.shape.length === 0 && <div className="num" style={{ padding: 8 }}>{String(res.values[0])}</div>}
            {res.shape.length === 1 && res.values.length > 0 && (
              <table className="num" style={{ borderCollapse: "collapse" }}>
                <thead><tr><th className="lbl dim" style={{ ...head, top: 0 }}>index</th><th className="lbl dim" style={{ ...head, top: 0 }}>value</th></tr></thead>
                <tbody>{res.values.map((v, i) => (
                  <tr key={i}><th className="lbl dim" style={{ ...head, left: 0 }}>{res.indices[0][i]}</th><td style={cellStyle(v)}>{String(v)}</td></tr>
                ))}</tbody>
              </table>
            )}
            {pg && (
              <table className="num" style={{ borderCollapse: "collapse" }}>
                <thead><tr>
                  <th style={{ ...head, top: 0, left: 0, zIndex: 2 }} />
                  {pg.cols.map((c) => <th key={c} className="lbl dim" style={{ ...head, top: 0, zIndex: 1 }}>{c}</th>)}
                </tr></thead>
                <tbody>{pg.cells.map((row, i) => (
                  <tr key={pg.rows[i]}>
                    <th className="lbl dim" style={{ ...head, left: 0, zIndex: 1 }}>{pg.rows[i]}</th>
                    {row.map((v, j) => <td key={j} style={cellStyle(v)}>{String(v)}</td>)}
                  </tr>
                ))}</tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  );
}
