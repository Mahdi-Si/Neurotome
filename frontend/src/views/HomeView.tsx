import { useEffect, useRef, useState } from "react";
import { IconArrow, IconGraph, IconHome, IconTensor, IconTrace, Logo } from "../components/Icons";
import { JobProgress } from "../components/JobProgress";
import { ThemeToggle } from "../components/TopBar";
import { api } from "../lib/api";
import { isPySpec, isTextSpec, parseBudget, parseModules, sampleGroups } from "../lib/capture";
import { bytes, outScore } from "../lib/format";
import { followJob } from "../lib/jobs";
import { href, navigate, parseHash } from "../lib/router";
import { errText } from "../lib/tensorState";
import type { Backend, Job, RunMeta, SampleRef } from "../lib/types";
import { useAsync } from "../lib/useAsync";

export function HomeView() {
  const [reload, setReload] = useState(0);
  const runs = useAsync(() => api.runs(), [reload]);
  const health = useAsync(() => api.health(), []);
  const models = useAsync(() => api.models(), []);

  const [model, setModel] = useState("torchvision:resnet18");
  const [weights, setWeights] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [picked, setPicked] = useState<SampleKind | null>(null);   // sticky once the user picks a sample kind
  const [samples, setSamples] = useState<{ list: SampleRef[]; error: string | null } | null>(null);   // a py: module's dataset
  const [filter, setFilter] = useState("");
  const [ref, setRef] = useState("");
  const [text, setText] = useState("");
  const [backend, setBackend] = useState<Backend | "">("");
  const [mods, setMods] = useState("");
  const [budget, setBudget] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Job<RunMeta> | null>(null);
  const ctl = useRef<AbortController | null>(null);
  const last = useRef<Job<RunMeta> | null>(null);
  const track = (j: Job<RunMeta> | null) => { last.current = j; setJob(j); };

  // a py: spec lists its dataset (list_samples) once typing pauses; a module with samples defaults to Dataset
  useEffect(() => {
    if (!isPySpec(model)) { setSamples(null); return; }
    let alive = true;
    const t = setTimeout(() => api.samples(model.trim()).then(
      (r) => alive && setSamples({ list: r.samples, error: r.error }),
      (e) => alive && setSamples({ list: [], error: errText(e) })), 350);
    return () => { alive = false; clearTimeout(t); };
  }, [model]);
  const data = samples?.list ?? [];
  const auto: SampleKind = data.length ? "dataset" : isTextSpec(model, models.data?.text ?? []) ? "text" : "file";
  const kind = picked === "dataset" && !data.length ? auto : picked ?? auto;
  const view = sampleGroups(data, filter);
  const chosen = data.some((s) => s.ref === ref) ? ref : view.groups[0]?.[1][0]?.ref ?? "";
  const nnsight = models.data?.backends?.nnsight ?? false;

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    if (kind === "text" && !text.trim()) { setError("Type some text, or pick File"); return; }
    if (kind === "dataset" && !chosen) { setError("Pick a dataset sample"); return; }
    const max_capture_mb = parseBudget(budget);
    if (Number.isNaN(max_capture_mb)) { setError("Budget must be a positive number of MB, or empty for the server default"); return; }
    const c = (ctl.current = new AbortController());
    setBusy(true); setError(null); track(null);
    try {
      let sample: Record<string, unknown> = { kind: "random" };
      if (kind === "text") sample = { kind: "text", text };
      else if (kind === "dataset") sample = { kind: "ref", ref: chosen };
      else if (file) {
        const up = await api.upload(file);
        sample = { kind: "upload", upload_id: up.upload_id, name: file.name };
      }
      const meta = await followJob<RunMeta>(
        () => api.createRun({ model, weights: weights || null, sample, modules: parseModules(mods), max_capture_mb, backend: backend || null }),
        track, c.signal);
      if (parseHash(location.hash).runId == null) navigate(href(meta.id, "graph"));   // still on Home: never yank a user who moved on
    } catch (err) {
      const msg = errText(err);
      if (msg === "Cancelled" && last.current) track({ ...last.current, status: "cancelled" });
      else if (last.current?.status !== "error") setError(msg);   // a failed job shows its own message in JobProgress
    } finally {
      setBusy(false);
    }
  };

  const latest = runs.data?.[0]?.id ?? null;
  return (
    <div style={{ flexGrow: 1, minHeight: 0, display: "flex" }}>
      <aside style={{ width: 232, flexShrink: 0, borderRight: "1px solid var(--line)", padding: "18px 12px", display: "flex", flexDirection: "column", gap: 24 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 8px" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 9 }}><Logo /><span className="lbl" style={{ fontSize: 12, fontWeight: 500, letterSpacing: ".14em" }}>Neurotome</span></span>
          <span className="lbl dim">{health.data ? `v${health.data.version}` : ""}</span>
        </div>
        <nav style={{ display: "flex", flexDirection: "column", gap: 2 }} aria-label="Primary">
          {[
            { to: "#/", label: "Home", icon: <IconHome />, on: true },
            { to: href(latest, "graph"), label: "Graph", icon: <IconGraph /> },
            { to: href(latest, "tensors"), label: "Tensors", icon: <IconTensor /> },
            { to: href(latest, "trace"), label: "Trace", icon: <IconTrace /> },
          ].map((l) => (
            <a key={l.label} href={l.to} aria-current={l.on ? "page" : undefined}
              style={{ display: "flex", alignItems: "center", gap: 10, height: 36, padding: "0 10px", borderRadius: 6, background: l.on ? "var(--muted-fill)" : undefined, opacity: !l.on && !latest ? 0.45 : 1 }}>
              {l.icon}{l.label}
            </a>
          ))}
        </nav>
        <div style={{ flexGrow: 1 }} />
        <div className="panel" style={{ border: "1px solid var(--line)", padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="lbl dim">Local engine</span>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: health.data ? "var(--accent)" : health.error ? "var(--negative)" : "var(--line)" }} />
          </div>
          <span className="mono">{location.host}</span>
          <span className="lbl dim">{health.data ? `${health.data.device} · torch ${health.data.torch}` : health.error ? "offline" : "…"}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 8px" }}>
          <span className="lbl dim">Theme</span><ThemeToggle />
        </div>
      </aside>

      <main className="scroll" style={{ flexGrow: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
        <section style={{ display: "flex", borderBottom: "1px solid var(--line)", minHeight: 320, flexShrink: 0 }}>
          <div style={{ flexGrow: 1, padding: "28px 40px", display: "flex", flexDirection: "column" }}>
            <span className="lbl">Model dissection workbench</span>
            <h1 style={{ margin: "22px 0 0", fontSize: 66, letterSpacing: "-0.055em", lineHeight: 0.9 }}>Open the model.<br />See every signal.</h1>
            <p className="dim" style={{ maxWidth: 520, marginTop: 20, lineHeight: 1.5 }}>
              Capture every operation of one forward pass, browse the computation graph, open any tensor as
              maps, signals or projections, and trace which parts of the model and the input drove an output.
            </p>
          </div>
          <form onSubmit={create} className="panel" style={{ width: 420, flexShrink: 0, borderLeft: "1px solid var(--line)", padding: "24px 28px", display: "flex", flexDirection: "column", gap: 14 }}>
            <span className="lbl">New run</span>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="lbl dim">Model</span>
              <input className="field" list="model-suggestions" value={model} onChange={(e) => setModel(e.target.value)} required />
              <datalist id="model-suggestions">{models.data?.suggested.map((m) => <option key={m} value={m} />)}</datalist>
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="lbl dim">Weights</span>
              <select className="field" value={weights} onChange={(e) => setWeights(e.target.value)}>
                <option value="">Random init (no download)</option>
                <option value="DEFAULT">Pretrained · DEFAULT</option>
              </select>
            </label>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span className="lbl dim">Sample</span>
                <div className="seg lbl" role="group" aria-label="Sample kind">
                  {KINDS.filter((k) => k !== "dataset" || data.length).map((k) => (
                    <button key={k} type="button" aria-pressed={kind === k} onClick={() => setPicked(k)}>{KIND_LABEL[k]}</button>
                  ))}
                </div>
              </div>
              {kind === "dataset" ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <input className="field" aria-label="Filter samples" placeholder="filter · ref, label or group" value={filter} onChange={(e) => setFilter(e.target.value)} />
                  <select className="field" aria-label="Dataset sample" size={6} value={chosen} onChange={(e) => setRef(e.target.value)}
                    style={{ height: "auto", padding: 4 }}>
                    {view.groups.map(([g, rows]) => {
                      const opts = rows.map((s) => <option key={s.ref} value={s.ref} title={s.ref}>{s.label}</option>);
                      return g ? <optgroup key={g} label={g}>{opts}</optgroup> : opts;
                    })}
                  </select>
                  <span className="lbl dim">{view.matched} of {data.length} samples{view.shown < view.matched ? ` · first ${view.shown} shown` : ""}{chosen ? ` · ${chosen}` : ""}</span>
                </div>
              ) : kind === "text" ? (
                <textarea className="field" rows={3} aria-label="Sample text" placeholder="The cat sat on the" value={text}
                  onChange={(e) => setText(e.target.value)} style={{ height: "auto", padding: "8px 10px", resize: "vertical" }} />
              ) : (
                <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <span className="lbl dim">image, .npy or .pt · empty = random tensor / random tokens</span>
                  <input type="file" accept="image/*,.npy,.pt,.pth" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="mono" />
                </label>
              )}
              {samples?.error && <span className="lbl dim" title={samples.error} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>No dataset · {samples.error}</span>}
            </div>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="lbl dim" style={{ display: "flex", justifyContent: "space-between" }}>
                <span>Capture</span>{!backend && models.data?.nnsight_params != null && <span>NNsight above {models.data.nnsight_params / 1e9} B params</span>}
              </span>
              <select className="field" aria-label="Capture backend" value={backend} onChange={(e) => setBackend(e.target.value as Backend | "")}>
                <option value="">Auto</option>
                <option value="torchlens">TorchLens · every op</option>
                <option value="nnsight" disabled={!nnsight} title={nnsight ? undefined : "pip install 'neurotome[nnsight]'"}>NNsight · module outputs</option>
              </select>
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="lbl dim">Modules · comma separated, empty = every op</span>
              <input className="field" aria-label="Capture modules" placeholder="layer3, layer4" value={mods} onChange={(e) => setMods(e.target.value)} />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="lbl dim" style={{ display: "flex", justifyContent: "space-between" }}>
                <span>Budget · MB</span>{health.data?.max_capture_mb != null && <span>default {health.data.max_capture_mb}</span>}
              </span>
              <input className="field" type="number" min={1} step="any" aria-label="Capture budget (MB)" value={budget}
                placeholder={health.data?.max_capture_mb != null ? String(health.data.max_capture_mb) : undefined} onChange={(e) => setBudget(e.target.value)} />
            </label>
            <button type="submit" className="btn solid" disabled={busy} style={{ height: 44, justifyContent: "space-between", padding: "0 18px", fontWeight: 500 }}>
              {busy ? "Capturing…" : "Capture forward pass"} <IconArrow />
            </button>
            {job && <JobProgress job={job} onCancel={() => ctl.current?.abort()} />}
            {error && <span className="err mono">{error}</span>}
          </form>
        </section>

        <section style={{ display: "flex", flexGrow: 1 }}>
          <div style={{ flexGrow: 1, minWidth: 0, padding: "32px 40px", display: "flex", flexDirection: "column", gap: 18 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <h2 style={{ fontSize: 30 }}>Recent runs</h2>
              <span className="lbl dim">{runs.data?.length ?? 0} runs · {bytes((runs.data ?? []).reduce((a, r) => a + r.model.activation_bytes, 0))} captured</span>
            </div>
            <div className="rule-top">
              <div className="lbl dim" style={rowGrid}>
                <span>Model</span><span>Sample</span><span>Ops</span><span>Captured</span><span>Top output</span><span />
              </div>
              {runs.error && <div className="kv err mono">{runs.error}</div>}
              {runs.data?.length === 0 && <div className="kv"><span className="dim">No runs yet. Capture one on the right.</span></div>}
              {runs.data?.map((r) => (
                <div key={r.id} style={{ ...rowGrid, fontSize: 14 }}>
                  <a href={href(r.id, "graph")} style={{ fontWeight: 500 }}>{r.model.spec?.split(":").pop() ?? r.model.class_name}<span className="lbl dim"> · {r.id}</span></a>
                  <span className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.sample.name}</span>
                  <span className="mono">{r.model.saved_ops != null && r.model.saved_ops < r.model.num_ops ? `${r.model.saved_ops}/${r.model.num_ops}` : r.model.num_ops}</span>
                  <span className="mono">{bytes(r.model.activation_bytes)}</span>
                  <span className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {r.output[0] ? `${r.output[0].label ?? `#${r.output[0].index}`} · ${outScore(r.output[0])}` : "—"}
                  </span>
                  <button type="button" className="lbl dim" aria-label={`Delete ${r.id}`} style={{ border: 0, background: "transparent", cursor: "pointer", justifySelf: "end" }}
                    onClick={() => api.deleteRun(r.id).then(() => setReload((x) => x + 1), (err) => setError(errText(err)))}>Delete</button>
                </div>
              ))}
            </div>
          </div>
          <div style={{ width: 420, flexShrink: 0, borderLeft: "1px solid var(--line)", padding: "32px 28px", display: "flex", flexDirection: "column", gap: 16 }}>
            <h2 style={{ fontSize: 30 }}>From Python</h2>
            <pre className="panel mono" style={{ margin: 0, padding: "14px 16px", border: "1px solid var(--line)", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
{`import neurotome
neurotome.open(model, x)
`}<span className="dim"># captures, then serves this UI</span>{`

$ neurotome capture torchvision:resnet18 \\
    --weights DEFAULT --image cat.jpg`}
            </pre>
            <span className="dim" style={{ fontSize: 12 }}><span className="lbl">Runs directory</span><br /><span className="mono" style={{ wordBreak: "break-all" }}>{health.data?.runs_dir ?? "…"}</span></span>
          </div>
        </section>
      </main>
    </div>
  );
}

type SampleKind = "text" | "file" | "dataset";
const KINDS: SampleKind[] = ["text", "file", "dataset"];
const KIND_LABEL: Record<SampleKind, string> = { text: "Text", file: "File", dataset: "Dataset" };

const rowGrid: React.CSSProperties = {
  display: "grid", gridTemplateColumns: "1.4fr 1.4fr 0.5fr 0.8fr 1.3fr 0.5fr", gap: 12, padding: "12px 0",
  borderBottom: "1px solid var(--line)", alignItems: "center",
};
