import { href, type Tab } from "../lib/router";
import { bytes } from "../lib/format";
import { useTheme } from "../lib/theme";
import type { RunMeta } from "../lib/types";
import { Logo } from "./Icons";

const TABS: { tab: Tab; label: string }[] = [
  { tab: "graph", label: "Graph" },
  { tab: "tensors", label: "Tensors" },
  { tab: "trace", label: "Trace" },
  { tab: "lens", label: "Lens" },
];
const tabStyle: React.CSSProperties = { display: "flex", alignItems: "center", padding: "0 14px", marginBottom: -1 };

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
      <span style={divider} />
      {run && (
        <div className="mono dim" title={`${model} / ${run.id} / ${run.sample.name}`}
          style={{ display: "flex", gap: 8, whiteSpace: "nowrap", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>
          <span>{model}</span><span>/</span><span>{run.id}</span><span>/</span>
          <span style={{ color: "var(--ink)", overflow: "hidden", textOverflow: "ellipsis" }}>{run.sample.name}</span>
        </div>
      )}
      {run && (
        <nav style={{ alignSelf: "stretch", display: "flex", marginLeft: 16, flexShrink: 0 }} aria-label="Views">
          {TABS.map((t) => t.tab === "lens" && !run.model.lm ? (
            <span key={t.tab} className="lbl" aria-disabled="true" title="Logit lens needs a causal language model"
              style={{ ...tabStyle, color: "var(--ink-2)", opacity: 0.45, borderBottom: "1.5px solid transparent" }}>{t.label}</span>
          ) : (
            <a key={t.tab} href={href(run.id, t.tab, t.tab === "lens" ? {} : { node })} className="lbl"
              aria-current={t.tab === tab ? "page" : undefined}
              style={{
                ...tabStyle,
                color: t.tab === tab ? "var(--ink)" : "var(--ink-2)",
                borderBottom: t.tab === tab ? "1.5px solid var(--ink)" : "1.5px solid transparent",
              }}>{t.label}</a>
          ))}
        </nav>
      )}
      <div style={{ flexGrow: 1 }} />
      {run && (
        <span className="lbl dim" style={{ whiteSpace: "nowrap", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>
          Fwd · {run.model.num_ops} ops · {bytes(run.model.activation_bytes)}
        </span>
      )}
      {children}
      <span style={divider} />
      <ThemeToggle />
    </header>
  );
}

const divider: React.CSSProperties = { width: 1, height: 20, flexShrink: 0, background: "var(--line)" };

/** Text button naming the theme it switches to. */
export function ThemeToggle() {
  const [theme, toggle] = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button type="button" className="chart-export lbl" onClick={toggle} aria-label={`Switch to ${next} theme`}>
      {next === "dark" ? "Dark" : "Light"}
    </button>
  );
}
