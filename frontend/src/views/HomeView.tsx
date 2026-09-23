import { useState } from "react";
import { IconArrow, IconGraph, IconHome, IconTensor, IconTrace, Logo } from "../components/Icons";
import { api } from "../lib/api";
import { bytes } from "../lib/format";
import { href, navigate } from "../lib/router";
import { useAsync } from "../lib/useAsync";

export function HomeView() {
  const [reload, setReload] = useState(0);
  const runs = useAsync(() => api.runs(), [reload]);
  const health = useAsync(() => api.health(), []);
  const models = useAsync(() => api.models(), []);

  const [model, setModel] = useState("torchvision:resnet18");
  const [weights, setWeights] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      let sample: Record<string, unknown> = { kind: "random" };
      if (file) {
        const up = await api.upload(file);
        sample = { kind: "upload", upload_id: up.upload_id, name: file.name };
      }
      const meta = await api.createRun({ model, weights: weights || null, sample });
      navigate(href(meta.id, "graph"));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const latest = runs.data?.[0]?.id ?? null;
  return (
    <div style={{ flexGrow: 1, minHeight: 0, display: "flex" }}>
      <aside style={{ width: 232, flexShrink: 0, borderRight: "1px solid var(--line)", padding: "18px 12px", display: "flex", flexDirection: "column", gap: 24 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 8px" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 9 }}><Logo /><span className="lbl" style={{ fontSize: 12, fontWeight: 500, letterSpacing: ".14em" }}>Dissect</span></span>
          <span className="lbl dim">v0.1</span>
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
      </aside>

      <main className="scroll" style={{ flexGrow: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
        <section style={{ display: "flex", borderBottom: "1px solid var(--line)", minHeight: 320 }}>
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
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="lbl dim">Sample · image, .npy or .pt (empty = random tensor)</span>
              <input type="file" accept="image/*,.npy,.pt,.pth" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="mono" />
            </label>
            <button type="submit" className="btn solid" disabled={busy} style={{ height: 44, justifyContent: "space-between", padding: "0 18px", fontWeight: 500 }}>
              {busy ? "Capturing…" : "Capture forward pass"} <IconArrow />
            </button>
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
                  <span className="mono">{r.model.num_ops}</span>
                  <span className="mono">{bytes(r.model.activation_bytes)}</span>
                  <span className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {r.output[0] ? `${r.output[0].label ?? `#${r.output[0].index}`} · ${r.output[0].prob.toFixed(2)}` : "—"}
                  </span>
                  <button type="button" className="lbl dim" aria-label={`Delete ${r.id}`} style={{ border: 0, background: "transparent", cursor: "pointer", justifySelf: "end" }}
                    onClick={async () => { await api.deleteRun(r.id); setReload((x) => x + 1); }}>Delete</button>
                </div>
              ))}
            </div>
          </div>
          <div style={{ width: 420, flexShrink: 0, borderLeft: "1px solid var(--line)", padding: "32px 28px", display: "flex", flexDirection: "column", gap: 16 }}>
            <h2 style={{ fontSize: 30 }}>From Python</h2>
            <pre className="panel mono" style={{ margin: 0, padding: "14px 16px", border: "1px solid var(--line)", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
{`import dissect
dissect.open(model, x)
`}<span className="dim"># captures, then serves this UI</span>{`

$ dissect capture torchvision:resnet18 \\
    --weights DEFAULT --image cat.jpg`}
            </pre>
            <span className="dim" style={{ fontSize: 12 }}><span className="lbl">Runs directory</span><br /><span className="mono" style={{ wordBreak: "break-all" }}>{health.data?.runs_dir ?? "…"}</span></span>
          </div>
        </section>
      </main>
    </div>
  );
}

const rowGrid: React.CSSProperties = {
  display: "grid", gridTemplateColumns: "1.4fr 1.4fr 0.5fr 0.8fr 1.3fr 0.5fr", gap: 12, padding: "12px 0",
  borderBottom: "1px solid var(--line)", alignItems: "center",
};
