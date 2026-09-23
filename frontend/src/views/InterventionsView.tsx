// Interventions tab: saved interventions (left), the draft set + an edit form, the open result, and generation for LMs.
import { useState } from "react";
import { Readout } from "../components/charts";
import { EditForm } from "../components/EditForm";
import { JobProgress } from "../components/JobProgress";
import { TopBar } from "../components/TopBar";
import { api } from "../lib/api";
import { editable, genReadout, topReadout } from "../lib/composeState";
import { displayName, fixed, num, shapeShort, signed } from "../lib/format";
import { addEdit, deltaColor, editLabel, parseSet, removeEdit, setParam } from "../lib/interventionState";
import { useJob } from "../lib/jobs";
import { href, navigate, parseHash, type Route } from "../lib/router";
import { errText } from "../lib/tensorState";
import { showToken } from "../lib/tokens";
import type { Continuation, GenerationResult, InterventionResult, InterventionSet, InterventionSummary, RunMeta, TopK } from "../lib/types";
import { useAsync } from "../lib/useAsync";

const card: React.CSSProperties = { border: "1px solid var(--line)", background: "var(--panel)", padding: 14, display: "flex", flexDirection: "column", gap: 10, minWidth: 0 };
const th: React.CSSProperties = { padding: "6px 8px", textAlign: "left", fontWeight: 400, borderBottom: "1px solid var(--ink)" };
const td: React.CSSProperties = { padding: "5px 8px", borderBottom: "1px solid var(--line-soft)", whiteSpace: "pre" };
const ellipsis: React.CSSProperties = { minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" };
const tokLabel = (t: { label: string | null; index: number }) => (t.label != null ? showToken(t.label) : `[${t.index}]`);

export function InterventionsViewPage({ run, route }: { run: RunMeta; route: Route }) {
  const p = route.params;
  const iv = p.get("iv") || null, nodeParam = p.get("node") || null;
  const set = parseSet(p);
  // merge over the current URL (async callers may run after the route changed)
  const nav = (o: Record<string, string | null>) =>
    navigate(href(run.id, "interventions", { ...Object.fromEntries(parseHash(location.hash).params), ...o }), true);

  const [tick, setTick] = useState(0);
  const refresh = () => setTick((t) => t + 1);
  const graph = useAsync(() => api.graph(run.id, 12), [run.id]);
  const saved = useAsync(() => api.interventions(run.id), [run.id, tick]);
  const res = useAsync(() => (iv ? api.intervention(run.id, iv) : null), [run.id, iv, tick]);
  const r = res.data?.hash === iv ? res.data : null;
  const job = useJob<InterventionResult>();
  const runSet = async () => {
    if (!set) return;
    const out = await job.run(run.id, { kind: "intervene", interventions: set });
    if (out) { nav({ iv: out.hash }); refresh(); }
  };

  const nodes = (graph.data?.nodes ?? []).filter(editable);
  const pick = nodes.find((n) => n.output_op === nodeParam || n.id === nodeParam) ?? nodes[0] ?? null;
  const pathOf = (op: string) => graph.data?.nodes.find((n) => n.output_op === op)?.path ?? op;
  const genSet = r?.set ?? set;

  return (
    <>
      <TopBar run={run} tab="interventions" node={nodeParam} carry={{ iv, set: setParam(set) }}>
        <button type="button" className="btn solid lbl" disabled={!set || job.busy} onClick={() => void runSet()}>Run interventions</button>
      </TopBar>
      <div style={{ flexGrow: 1, minHeight: 0, display: "flex" }}>
        <aside className="panel scroll" style={{ width: 300, flexShrink: 0, borderRight: "1px solid var(--line)", padding: 20, display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span className="lbl">Saved</span><span className="lbl dim">{saved.data?.length ?? "—"}</span>
          </div>
          {saved.error && <span className="err">{saved.error}</span>}
          {saved.data && (
            <SavedList run={run} items={saved.data} open={iv} onOpen={(h) => nav({ iv: h })} onChanged={refresh} />
          )}
        </aside>
        <section className="scroll" style={{ flexGrow: 1, minWidth: 0, padding: "20px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 16, alignItems: "start" }}>
            <div style={card}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <span className="lbl">Set</span>
                {set && <button type="button" className="lbl chart-export" onClick={() => nav({ set: null })}>Clear set</button>}
              </div>
              <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span className="lbl dim" style={{ width: 72, flexShrink: 0 }}>Target</span>
                <select className="field" aria-label="Target" value={set?.target ?? ""} disabled={!set} style={{ minWidth: 0, flex: 1 }}
                  onChange={(e) => set && nav({ set: setParam({ ...set, target: e.target.value === "" ? null : Number(e.target.value) }) })}>
                  <option value="">top prediction</option>
                  {run.output.map((o) => <option key={o.index} value={o.index}>{tokLabel(o)} · p {fixed(o.prob)}</option>)}
                  {set?.target != null && !run.output.some((o) => o.index === set.target) && <option value={set.target}>#{set.target}</option>}
                </select>
              </label>
              <div className="rule-top">
                {set?.edits.map((e, i) => (
                  <div key={i} className="kv" style={{ alignItems: "baseline" }}>
                    <span className="mono" style={{ overflowWrap: "anywhere" }}><span className="dim">{i + 1} · </span>{editLabel(e, pathOf(e.op))}</span>
                    <button type="button" className="chart-export" aria-label={`Remove edit ${i + 1}`} onClick={() => nav({ set: setParam(removeEdit(set, i)) })}>×</button>
                  </div>
                )) ?? <div className="lbl dim" style={{ paddingTop: 7 }}>No edits · add one below or from the graph inspector</div>}
              </div>
              <JobProgress job={job.job} onCancel={job.cancel} />
              {job.error && job.job?.status !== "error" && <span className="err">{job.error}</span>}
            </div>
            <div style={card}>
              <span className="lbl">Add edit</span>
              {graph.error && <span className="err">{graph.error}</span>}
              {graph.loading && !graph.data && <span className="lbl dim">Loading graph…</span>}
              {graph.data && (
                <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span className="lbl dim" style={{ width: 72, flexShrink: 0 }}>Node</span>
                  <select className="field" aria-label="Node" value={pick?.output_op ?? ""} style={{ minWidth: 0, flex: 1 }}
                    onChange={(e) => nav({ node: e.target.value })}>
                    {nodes.map((n) => <option key={n.id} value={n.output_op}>{displayName(n)} · {shapeShort(n.shape)}</option>)}
                  </select>
                </label>
              )}
              {pick && <EditForm key={pick.output_op} run={run} op={{ id: pick.output_op, shape: pick.shape, path: pick.path }}
                onAdd={(e) => nav({ set: setParam(addEdit(set, e)) })} />}
            </div>
          </div>
          {r ? <ResultCard run={run} r={r} set={set} />
            : iv && res.error ? <span className="err">{res.error}</span>
            : <span className="lbl dim">{iv && res.loading ? "Loading…" : "Compose a set and run it; results are saved with this run"}</span>}
          {run.model.lm && <GenerationCard key={setParam(genSet) ?? ""} run={run} set={genSet} />}
        </section>
      </div>
    </>
  );
}

function SavedList({ run, items, open, onOpen, onChanged }: {
  run: RunMeta; items: InterventionSummary[]; open: string | null; onOpen: (hash: string | null) => void; onChanged: () => void;
}) {
  const [renaming, setRenaming] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const act = (p: Promise<unknown>, after: () => void) =>
    p.then(() => { setErr(null); after(); onChanged(); }, (e) => setErr(errText(e)));
  if (!items.length) return <span className="lbl dim">No saved interventions · run a set to save one</span>;
  return (
    <>
      {err && <span className="err">{err}</span>}
      <ul aria-label="Saved interventions" style={{ listStyle: "none", margin: 0, padding: 0 }} className="rule-top">
        {items.map((s) => (
          <li key={s.hash} style={{ display: "flex", flexDirection: "column", gap: 4, padding: "8px 0", borderBottom: "1px solid var(--line-soft)" }}>
            {renaming === s.hash ? (
              <input className="field" aria-label="Intervention name" autoFocus defaultValue={s.name} maxLength={80}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void act(api.renameIntervention(run.id, s.hash, e.currentTarget.value), () => setRenaming(null));
                  if (e.key === "Escape") setRenaming(null);
                }} />
            ) : (
              <button type="button" aria-pressed={open === s.hash} onClick={() => onOpen(s.hash)} className="mono"
                style={{ border: 0, padding: "2px 4px", textAlign: "left", cursor: "pointer", overflowWrap: "anywhere",
                  background: open === s.hash ? "var(--accent-soft)" : "transparent", color: open === s.hash ? "var(--accent)" : "var(--ink)" }}>{s.name}</button>
            )}
            <span className="lbl dim">
              {s.labels.length} {s.labels.length === 1 ? "edit" : "edits"} · Δ logit <span style={{ color: deltaColor(s.delta_logit) }}>{signed(s.delta_logit, 2)}</span>
              {s.truncated && " · budget reached"}
            </span>
            <div style={{ display: "flex", gap: 12 }}>
              <button type="button" className="chart-export lbl" onClick={() => { setRenaming(s.hash); setConfirm(null); }}>Rename</button>
              {confirm === s.hash
                ? <button type="button" className="chart-export lbl" style={{ color: "var(--negative-text)" }}
                    onClick={() => void act(api.deleteIntervention(run.id, s.hash), () => { setConfirm(null); if (open === s.hash) onOpen(null); })}>Confirm delete</button>
                : <button type="button" className="chart-export lbl" onClick={() => setConfirm(s.hash)}>Delete</button>}
            </div>
          </li>
        ))}
      </ul>
    </>
  );
}

function TopList({ title, xs, target, onHover }: { title: string; xs: TopK[]; target: number; onHover: (i: number) => void }) {
  return (
    <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
      <span className="lbl dim" style={{ paddingBottom: 6 }}>{title}</span>
      <div className="rule-top">
        {xs.slice(0, 5).map((t) => (
          <div key={t.index} onMouseEnter={() => onHover(t.index)}
            style={{ display: "grid", gridTemplateColumns: "1fr 80px 44px", gap: 10, alignItems: "center", padding: "6px 0", borderBottom: "1px solid var(--line-soft)" }}>
            <span className="mono" style={{ ...ellipsis, color: t.index === target ? "var(--accent)" : undefined }}>{tokLabel(t)}</span>
            <span style={{ height: 4, background: "var(--line-soft)" }}>
              <span style={{ display: "block", height: "100%", width: `${Math.min(1, t.prob ?? 0) * 100}%`, background: "var(--ink)" }} />
            </span>
            <span className="num" style={{ fontSize: 12, textAlign: "right" }}>{fixed(t.prob)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultCard({ run, r, set }: { run: RunMeta; r: InterventionResult; set: InterventionSet | null }) {
  const [hover, setHover] = useState<number | null>(null);
  const op = r.edited[0];
  const delta = (a: number | null, b: number | null, d: number | null) => <>{num(a, 3)} → {num(b, 3)} <span style={{ color: deltaColor(d) }}>(Δ {signed(d, 3)})</span></>;
  return (
    <section aria-label="Intervention result" style={card}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
        <span className="lbl" style={{ overflowWrap: "anywhere" }}>Result · {r.name}</span>
        <span className="lbl dim" style={{ textAlign: "right" }}>{r.method_label}</span>
      </div>
      <div className="rule-top">
        {([
          ["Target", `logit[${r.target}] · ${r.target_label != null ? showToken(r.target_label) : "—"}`],
          ["Logit", delta(r.before.logit, r.after.logit, r.delta_logit)],
          ["Prob", delta(r.before.prob, r.after.prob, r.delta_prob)],
        ] as const).map(([k, v]) => (
          <div key={k} className="kv"><span className="lbl dim">{k}</span><span className="mono" style={{ textAlign: "right" }}>{v}</span></div>
        ))}
      </div>
      <div onMouseLeave={() => setHover(null)} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <div style={{ display: "flex", gap: 24 }}>
          <TopList title="Top-k before" xs={r.top_before} target={r.target} onHover={setHover} />
          <TopList title="Top-k after" xs={r.top_after} target={r.target} onHover={setHover} />
        </div>
        <Readout text={hover != null ? topReadout(r.top_before, r.top_after, hover) : null} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span className="lbl dim">Edits</span>
        {r.labels.map((l, i) => <span key={i} className="mono" style={{ overflowWrap: "anywhere" }}><span className="dim">{i + 1} · </span>{l}</span>)}
      </div>
      {r.truncated && <span className="lbl dim">Budget reached · {r.saved.length}/{r.downstream.length} downstream tensors saved; the graph still shows every Δ</span>}
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        <a className="lbl chart-export" href={href(run.id, "graph", { iv: r.hash, color: "iv", set: setParam(set) })}>Graph · Intervention Δ →</a>
        {op && r.saved.includes(op)
          ? <a className="lbl chart-export" href={href(run.id, "tensors", { node: op, iv: r.hash, vs: "clean" })}>Clean vs intervened →</a>
          : <span className="lbl dim">No intervened tensor saved for {op}</span>}
      </div>
    </section>
  );
}

function GenerationCard({ run, set }: { run: RunMeta; set: InterventionSet | null }) {
  const [n, setN] = useState("16");
  const gen = useJob<GenerationResult>();
  const [res, setRes] = useState<GenerationResult | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const k = Number(n), ok = Number.isInteger(k) && k >= 1 && k <= 64;
  const go = async () => {
    if (!set || !ok) return;
    const out = await gen.run(run.id, { kind: "generate", interventions: set, max_new_tokens: k });
    if (out) setRes(out);
  };
  const d = res?.diverge ?? null;
  const side = (name: string, c: Continuation) => (
    <div role="group" aria-label={name} style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 6 }}>
      <span className="lbl dim">{name.split(" ")[0]}</span>
      <div className="mono" style={{ display: "flex", flexWrap: "wrap", rowGap: 4, lineHeight: 1.6 }}>
        {res!.prompt.map((t, i) => <span key={`p${i}`} className="dim" style={{ whiteSpace: "pre" }}>{showToken(t)}</span>)}
        {c.tokens.map((t, i) => (
          <span key={i} tabIndex={0} onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} onBlur={() => setHover(null)}
            style={{ whiteSpace: "pre", borderBottom: `1px solid ${d != null && i >= d ? "var(--ink)" : "transparent"}`,
              background: hover === i ? "var(--accent-soft)" : undefined }}>{showToken(t)}</span>
        ))}
      </div>
    </div>
  );
  return (
    <div style={card}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span className="lbl" style={{ flexGrow: 1 }}>Generation</span>
        <span className="lbl dim">Tokens</span>
        <input className="field" type="number" aria-label="Tokens" min={1} max={64} step={1} value={n} onChange={(e) => setN(e.target.value)} style={{ width: 72 }} />
        <button type="button" className="btn lbl" disabled={!set || !ok || gen.busy} onClick={() => void go()}>Generate</button>
      </div>
      {!set && <span className="lbl dim">Open an intervention or compose a set to generate with it</span>}
      {!ok && <span className="err">Tokens are 1–64</span>}
      <JobProgress job={gen.job} onCancel={gen.cancel} />
      {gen.error && gen.job?.status !== "error" && <span className="err">{gen.error}</span>}
      {res && (
        <>
          <span className="lbl dim">{res.method_label}{d == null ? " · identical continuations" : ` · diverges at step ${d}`}</span>
          <div onMouseLeave={() => setHover(null)} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div style={{ display: "flex", gap: 24 }}>
              {side("Clean continuation", res.clean)}
              {side("Intervened continuation", res.steered)}
            </div>
            <Readout text={hover != null ? genReadout(res, hover) : null} />
          </div>
          <table aria-label="Per-step top tokens" className="num" style={{ borderCollapse: "collapse", fontSize: 12, width: "100%" }}>
            <thead className="lbl dim"><tr>{["Step", "Clean", "p", "Intervened", "p"].map((h, i) => <th key={i} style={th}>{h}</th>)}</tr></thead>
            <tbody>
              {res.clean.top.map((top, i) => {
                const a = top[0], b = res.steered.top[i]?.[0];
                return (
                  <tr key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                    style={{ background: d != null && i >= d ? "var(--accent-soft)" : undefined }}>
                    <td style={td}>{i}</td>
                    <td style={td}>{a ? `"${showToken(a.token)}"` : "—"}</td><td style={td}>{fixed(a?.prob, 3)}</td>
                    <td style={td}>{b ? `"${showToken(b.token)}"` : "—"}</td><td style={td}>{fixed(b?.prob, 3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
