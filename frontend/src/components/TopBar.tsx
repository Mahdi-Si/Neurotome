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
  { tab: "interventions", label: "Interventions" },
];
const tabStyle: React.CSSProperties = { display: "flex", alignItems: "center", padding: "0 11px", marginBottom: -1 };

/** Params a tab link keeps: none to Lens; `set` (the draft) only to the tabs that compose it. */
const linkParams = (tab: Tab, node: string | null | undefined, carry: Record<string, string | null> = {}) =>
  tab === "lens" ? {} : { node, ...carry, ...(tab !== "graph" && tab !== "interventions" && { set: null }) };

export function TopBar({ run, tab, node, carry, children }: {
  run: RunMeta | null; tab: Tab; node?: string | null; carry?: Record<string, string | null>; children?: React.ReactNode;
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
          style={{ display: "flex", gap: 8, whiteSpace: "nowrap", minWidth: 76, overflow: "hidden" }}>
          <span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>{model} /</span>
          <span style={{ flexShrink: 0 }}>{run.id}</span>
          <span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>/ <span style={{ color: "var(--ink)" }}>{run.sample.name}</span></span>
        </div>
      )}
      {run && (
        <nav style={{ alignSelf: "stretch", display: "flex", marginLeft: 8, flexShrink: 0 }} aria-label="Views">
          {TABS.map((t) => t.tab === "lens" && !run.model.lm ? (
            <span key={t.tab} className="lbl" aria-disabled="true" title="Logit lens needs a causal language model"
              style={{ ...tabStyle, color: "var(--ink-2)", opacity: 0.45, borderBottom: "1.5px solid transparent" }}>{t.label}</span>
          ) : (
            <a key={t.tab} href={href(run.id, t.tab, linkParams(t.tab, node, carry))} className="lbl"
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
        <span className="lbl dim" style={{ whiteSpace: "nowrap", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", flexShrink: 100 }}>
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
