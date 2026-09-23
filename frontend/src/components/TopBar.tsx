import { href, type Tab } from "../lib/router";
import { bytes } from "../lib/format";
import type { RunMeta } from "../lib/types";
import { Logo } from "./Icons";

const TABS: { tab: Tab; label: string }[] = [
  { tab: "graph", label: "Graph" },
  { tab: "tensors", label: "Tensors" },
  { tab: "trace", label: "Trace" },
];

export function TopBar({ run, tab, node, children }: {
  run: RunMeta | null; tab: Tab; node?: string | null; children?: React.ReactNode;
}) {
  const model = run?.model.spec?.split(":").pop() ?? run?.model.class_name ?? "";
  return (
    <header style={{
      height: "var(--topbar-h)", flexShrink: 0, display: "flex", alignItems: "center", gap: 18,
      padding: "0 20px", borderBottom: "1px solid var(--line)",
    }}>
      <a href="#/" style={{ display: "flex", alignItems: "center", gap: 9 }} aria-label="Neurotome home">
        <Logo />
        <span className="lbl" style={{ fontSize: 12, fontWeight: 500, letterSpacing: ".14em" }}>Neurotome</span>
      </a>
      <span style={{ width: 1, height: 20, background: "var(--line)" }} />
      {run && (
        <div className="mono dim" style={{ display: "flex", gap: 8, whiteSpace: "nowrap" }}>
          <span>{model}</span><span>/</span><span>{run.id}</span><span>/</span>
          <span style={{ color: "var(--ink)" }}>{run.sample.name}</span>
        </div>
      )}
      {run && (
        <nav style={{ alignSelf: "stretch", display: "flex", marginLeft: 16 }} aria-label="Views">
          {TABS.map((t) => (
            <a key={t.tab} href={href(run.id, t.tab, { node })} className="lbl"
              aria-current={t.tab === tab ? "page" : undefined}
              style={{
                display: "flex", alignItems: "center", padding: "0 14px",
                color: t.tab === tab ? "var(--ink)" : "var(--ink-2)",
                borderBottom: t.tab === tab ? "1.5px solid var(--ink)" : "1.5px solid transparent",
                marginBottom: -1,
              }}>{t.label}</a>
          ))}
        </nav>
      )}
      <div style={{ flexGrow: 1 }} />
      {run && (
        <span className="lbl dim" style={{ whiteSpace: "nowrap" }}>
          Fwd · {run.model.num_ops} ops · {bytes(run.model.activation_bytes)}
        </span>
      )}
      {children}
    </header>
  );
}
