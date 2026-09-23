// Compose one edit of an op's output: kind + parameters + channel / position scope, then "Add to set".
// Used by the graph inspector and the Interventions tab; render with key={op.id} so the draft resets per op.
import { useState } from "react";
import { api } from "../lib/api";
import { cellsOf, sameModelRuns, toggleCells } from "../lib/composeState";
import { num } from "../lib/format";
import { editLabel, parseIndexList, scopeAxes } from "../lib/interventionState";
import { cellSpan, unscale } from "../lib/tensorState";
import { tokenLabels } from "../lib/tokens";
import type { Edit, EditKind, RunMeta } from "../lib/types";
import { useAsync } from "../lib/useAsync";
import { DotMatrix } from "./charts";
import { TokenStrip } from "./TokenStrip";

const KINDS: [EditKind, string][] = [["zero", "Zero"], ["mean", "Mean"], ["resample", "Resample"], ["constant", "Constant"], ["scale", "Scale"], ["steer", "Steer"]];

/** A seg control named `name`: one aria-pressed button per [value, text]. */
function Seg<T extends string>({ name, value, options, onChange, disabled }: {
  name: string; value: T; options: [T, string][]; onChange: (v: T) => void; disabled?: (v: T) => string | null;
}) {
  return (
    <div className="seg lbl" role="group" aria-label={name} style={{ alignSelf: "flex-start", flexWrap: "wrap" }}>
      {options.map(([v, text]) => (
        <button key={v} type="button" aria-pressed={v === value} disabled={!!disabled?.(v)} title={disabled?.(v) ?? undefined}
          onClick={() => onChange(v)}>{text}</button>
      ))}
    </div>
  );
}

/** A dim label on the left, the control on the right. */
const Row = ({ k, children }: { k: string; children: React.ReactNode }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
    <span className="lbl dim" style={{ width: 72, flexShrink: 0 }}>{k}</span>{children}
  </div>
);

const numField = (name: string, value: string, set: (v: string) => void, o: { min?: number; max?: number; step?: string } = {}) => (
  <input className="field" type="number" aria-label={name} value={value} min={o.min} max={o.max} step={o.step ?? "any"}
    onChange={(e) => set(e.target.value)} style={{ width: 96 }} />
);

const finite = (s: string) => (s.trim() !== "" && Number.isFinite(Number(s)) ? Number(s) : null);
const index = (s: string, n: number) => { const v = finite(s); return v != null && Number.isInteger(v) && v >= 0 && v < n ? v : null; };

export function EditForm({ run, op, onAdd }: { run: RunMeta; op: { id: string; shape: number[]; path: string }; onAdd: (e: Edit) => void }) {
  const axes = scopeAxes(op.shape);
  const [kind, setKind] = useState<EditKind>("zero");
  const [over, setOver] = useState<"positions" | "runs">("positions");
  const [source, setSource] = useState("random");
  const [f, setF] = useState({ seed: "0", value: "0", alpha: "1", channel: "0" });
  const field = (k: keyof typeof f) => (v: string) => setF((s) => ({ ...s, [k]: v }));
  const [dir, setDir] = useState<"contrast" | "channel">("contrast");
  const [ab, setAb] = useState<{ a: string | null; b: string | null }>({ a: null, b: null });
  const [pool, setPool] = useState<"mean" | "last">("mean");
  const [chText, setChText] = useState("");
  const [posMode, setPosMode] = useState<"all" | "last" | "selected">("all");
  const [pixels, setPixels] = useState<[number, number][]>([]);
  const [toks, setToks] = useState<number[]>([]);

  const inProcess = run.model.spec == null;
  const needRuns = !inProcess && (kind === "resample" || (kind === "steer" && dir === "contrast"));
  const runs = useAsync(() => (needRuns ? api.runs() : null), [needRuns]);
  const same = inProcess ? [run] : sameModelRuns(runs.data ?? [run], run);
  const runName = (r: RunMeta) => `${r.id} · ${r.id === run.id ? "this run" : r.sample.name}`;
  const runA = ab.a ?? same[0]?.id ?? run.id, runB = ab.b ?? same[1]?.id ?? runA;

  const n = axes.channels;
  const channels = parseIndexList(chText, n ?? undefined);
  const chBad = n != null && channels == null;
  const firstCh = channels?.length && !chBad ? channels[0] : null;
  const [H, W] = axes.size;
  const T = axes.positions === "tokens" ? axes.size[0] : 0;
  const tensor = useAsync(() => (axes.positions === "pixels" ? api.tensor(run.id, op.id, { channel: firstCh }) : null), [op.id, firstCh]);
  const img = tensor.data?.kind === "image" ? tensor.data : null;

  const positions: Edit["positions"] = axes.positions === "pixels" ? (pixels.length ? pixels : null)
    : axes.positions === "tokens" ? (posMode === "last" ? "last" : posMode === "selected" && toks.length ? [...toks].sort((a, b) => a - b) : null)
    : null;
  const params = ((): Partial<Edit> | null => {
    switch (kind) {
      case "zero": return {};
      case "mean": return { over };
      case "resample": return inProcess ? null : source === "random" ? (index(f.seed, 2 ** 31) == null ? null : { source_run: "random", seed: Number(f.seed) }) : { source_run: source };
      case "constant": return finite(f.value) == null ? null : { value: Number(f.value) };
      case "scale": return finite(f.alpha) == null ? null : { alpha: Number(f.alpha) };
      case "steer": {
        if (finite(f.alpha) == null || n == null) return null;
        if (dir === "channel") return index(f.channel, n) == null ? null : { alpha: Number(f.alpha), direction: { kind: "channel", channel: Number(f.channel) } };
        return { alpha: Number(f.alpha), direction: { kind: "contrast", run_a: runA, run_b: runB, pool: T ? pool : "mean" } };
      }
    }
  })();
  const edit: Edit | null = params && !chBad && !(posMode === "selected" && T && !toks.length)
    ? { op: op.id, kind, channels: channels?.length ? channels : null, positions, ...params } : null;

  const runSelect = (name: string, value: string, set: (v: string) => void) => (
    <select className="field" aria-label={name} value={value} onChange={(e) => set(e.target.value)} disabled={inProcess} style={{ minWidth: 0, flex: 1 }}>
      {same.map((r) => <option key={r.id} value={r.id}>{runName(r)}</option>)}
    </select>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <Seg name="Edit kind" value={kind} options={KINDS} onChange={setKind}
        disabled={(k) => (k === "steer" && n == null ? "Steering needs a (B, C, H, W), (B, T, D) or (B, D) tensor" : null)} />
      {kind === "mean" && <Row k="Over"><Seg name="Over" value={over} options={[["positions", "Positions"], ["runs", "Runs"]]} onChange={setOver} /></Row>}
      {kind === "resample" && (
        <>
          <Row k="Source">
            <select className="field" aria-label="Source run" value={source} disabled={inProcess} onChange={(e) => setSource(e.target.value)} style={{ minWidth: 0, flex: 1 }}>
              <option value="random">Random run</option>
              {same.map((r) => <option key={r.id} value={r.id}>{runName(r)}</option>)}
            </select>
          </Row>
          {inProcess && <span className="lbl dim">In-process run · no source runs</span>}
          {source === "random" && !inProcess && <Row k="Seed">{numField("Seed", f.seed, field("seed"), { min: 0, step: "1" })}</Row>}
        </>
      )}
      {kind === "constant" && <Row k="Value">{numField("Value", f.value, field("value"))}</Row>}
      {kind === "steer" && (
        <>
          <Row k="Direction"><Seg name="Direction" value={dir} options={[["contrast", "Contrast"], ["channel", "Channel"]]} onChange={setDir} /></Row>
          {dir === "contrast" && (
            <>
              <Row k="Run A">{runSelect("Run A", runA, (a) => setAb((s) => ({ ...s, a })))}</Row>
              <Row k="Run B">{runSelect("Run B", runB, (b) => setAb((s) => ({ ...s, b })))}</Row>
              {T > 0 && <Row k="Pool"><Seg name="Pool" value={pool} options={[["mean", "Mean"], ["last", "Last"]]} onChange={setPool} /></Row>}
              <span className="lbl dim">v = pool(B) − pool(A) at this op</span>
            </>
          )}
          {dir === "channel" && n != null && <Row k="Channel">{numField("Channel", f.channel, field("channel"), { min: 0, max: n - 1, step: "1" })}</Row>}
        </>
      )}
      {(kind === "scale" || kind === "steer") && <Row k="Alpha">{numField("Alpha", f.alpha, field("alpha"))}</Row>}
      {runs.error && <span className="err">{runs.error}</span>}
      {n != null && (
        <Row k="Channels">
          <input className="field" aria-label="Channels" placeholder="all · e.g. 0,3,7-9" value={chText}
            onChange={(e) => setChText(e.target.value)} style={{ minWidth: 0, flex: 1 }} />
        </Row>
      )}
      {chBad && <span className="err">Channels are 0–{n! - 1}</span>}
      {axes.positions === "pixels" && (
        img ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <DotMatrix grid={img.dots} label="Position picker" pitch={Math.min(18, Math.floor(300 / Math.max(1, img.dots[0]?.length ?? 1)))}
              selected={cellsOf(pixels, H, W, img.dots.length, img.dots[0]?.length ?? 0)}
              onPick={(r, c) => setPixels((p) => toggleCells(p, H, W, img.dots.length, img.dots[0]?.length ?? 0, r, c))}
              readout={(r, c, v) => `px (${cellSpan(H, img.dots.length, r)}, ${cellSpan(W, img.dots[0].length, c)}) · ${num(unscale(v, img.dots_range, img.cmap), 3)}`} />
            <div className="lbl dim" style={{ display: "flex", gap: 6 }}>
              <span>{pixels.length ? `${pixels.length} px selected ·` : "All positions · click cells to scope"}</span>
              {pixels.length > 0 && <button type="button" className="chart-export lbl" onClick={() => setPixels([])}>Clear</button>}
            </div>
          </div>
        ) : <span className="lbl dim">{tensor.loading ? "Loading positions…" : `No position picker${tensor.error ? ` · ${tensor.error}` : ""} · edits every position`}</span>
      )}
      {T > 0 && (
        <>
          <Row k="Positions"><Seg name="Positions" value={posMode} options={[["all", "All"], ["last", "Last"], ["selected", "Selected"]]} onChange={setPosMode} /></Row>
          {posMode === "selected" && (
            <TokenStrip label="Token positions" tokens={tokenLabels(run, T) ?? Array.from({ length: T }, (_, i) => `#${i}`)}
              scores={Array(T).fill(0)} signed={false} selected={toks}
              onToggle={(i) => setToks((s) => (s.includes(i) ? s.filter((x) => x !== i) : [...s, i]))} />
          )}
        </>
      )}
      {edit && <span className="lbl dim" style={{ overflowWrap: "anywhere" }}>{editLabel(edit, op.path)}</span>}
      <button type="button" className="btn lbl" disabled={!edit} onClick={() => edit && onAdd(edit)}>Add to set</button>
    </div>
  );
}
