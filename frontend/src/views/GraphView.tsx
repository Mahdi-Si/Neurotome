import { useMemo } from "react";
import { GraphCanvas } from "../components/GraphCanvas";
import { Inspector, ModuleTree } from "../components/panels";
import { TopBar } from "../components/TopBar";
import { api } from "../lib/api";
import { href, navigate, type Route } from "../lib/router";
import type { GraphView as GV, RunMeta } from "../lib/types";
import { useAsync } from "../lib/useAsync";

export function pickNodeForModule(view: GV, address: string): string | null {
  const exact = view.nodes.find((n) => n.path === address);
  if (exact) return exact.id;
  const group = view.groups.find((g) => g.address === address);
  if (group) {
    const inside = view.nodes.find((n) => n.group === group.id);
    if (inside) return inside.id;
  }
  const child = view.nodes.find((n) => n.path.startsWith(address + "."));
  if (child) return child.id;
  // module folded into a collapsed ancestor
  const anc = view.nodes.find((n) => n.kind === "module" && address.startsWith(n.path + "."));
  return anc?.id ?? null;
}

export function GraphViewPage({ run, route }: { run: RunMeta; route: Route }) {
  const depth = Number(route.params.get("depth") ?? 2);
  const overlay = route.params.get("overlay") === "attr";
  const graph = useAsync(() => api.graph(run.id, depth, overlay ? { method: "gxa" } : undefined), [run.id, depth, overlay]);
  const modules = useAsync(() => api.modules(run.id), [run.id]);
  const view = graph.data;

  const selectedId = route.params.get("node");
  const selected = useMemo(() => {
    if (!view) return null;
    return view.nodes.find((n) => n.id === selectedId)
      ?? view.nodes.find((n) => n.output_op === selectedId)
      ?? view.nodes.find((n) => !n.is_input) ?? null;
  }, [view, selectedId]);

  const go = (params: Record<string, string | null>) =>
    navigate(href(run.id, "graph", { node: selected?.id ?? null, depth: String(depth), overlay: overlay ? "attr" : null, ...params }), true);

  return (
    <>
      <TopBar run={run} tab="graph" node={selected?.id}>
        <span className="lbl dim">Depth</span>
        <div className="seg lbl" role="group" aria-label="Nesting depth">
          {[0, 1, 2, 3, 4].map((d) => (
            <button key={d} type="button" aria-pressed={d === depth} onClick={() => go({ depth: String(d) })}>{d}</button>
          ))}
          <button type="button" aria-pressed={depth >= 12} onClick={() => go({ depth: "12" })}>All</button>
        </div>
        <div className="seg lbl" role="group" aria-label="Overlay">
          <button type="button" aria-pressed={!overlay} onClick={() => go({ overlay: null })}>Plain</button>
          <button type="button" aria-pressed={overlay} onClick={() => go({ overlay: "attr" })}>Attribution</button>
        </div>
      </TopBar>
      <div style={{ flexGrow: 1, minHeight: 0, display: "flex" }}>
        <aside style={{ width: 248, flexShrink: 0, borderRight: "1px solid var(--line)", display: "flex", flexDirection: "column", minHeight: 0 }}>
          {modules.data && (
            <ModuleTree modules={modules.data} selectedPath={selected?.path ?? null}
              onPick={(addr) => { const id = view && pickNodeForModule(view, addr); if (id) go({ node: id }); }} />
          )}
        </aside>
        <section style={{ flexGrow: 1, minWidth: 0, position: "relative" }}>
          {graph.error && <div className="empty err">{graph.error}</div>}
          {view && (
            <>
              <GraphCanvas view={view} selected={selected?.id ?? null} focus={selected?.id} onSelect={(id) => go({ node: id })} />
              <div style={{ position: "absolute", left: 16, top: 14, display: "flex", flexDirection: "column", gap: 3, pointerEvents: "none" }}>
                <span className="lbl">Forward trace</span>
                <span className="lbl dim">{run.sample.name} · {view.nodes.length} nodes · depth {depth >= 12 ? "all" : depth}</span>
              </div>
              {overlay && (
                <div className="panel" style={{ position: "absolute", right: 16, top: 14, width: 176, padding: 10, border: "1px solid var(--line)", display: "flex", flexDirection: "column", gap: 6 }}>
                  <span className="lbl">Node bar = |grad × act|</span>
                  <div style={{ height: 4, background: "linear-gradient(90deg, var(--accent-soft), var(--accent))" }} />
                  <div className="lbl dim" style={{ display: "flex", justifyContent: "space-between" }}><span>0</span><span>max</span></div>
                </div>
              )}
            </>
          )}
          {graph.loading && !view && <div className="empty lbl">Loading graph…</div>}
        </section>
        <aside className="panel" style={{ width: 360, flexShrink: 0, borderLeft: "1px solid var(--line)", display: "flex", flexDirection: "column", minHeight: 0 }}>
          {view && <Inspector runId={run.id} view={view} node={selected} total={view.nodes.length} />}
        </aside>
      </div>
    </>
  );
}
