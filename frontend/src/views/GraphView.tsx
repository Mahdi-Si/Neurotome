import { useMemo, useState } from "react";
import { GraphCanvas } from "../components/GraphCanvas";
import { GraphSearch } from "../components/GraphSearch";
import { Inspector, ModuleTree } from "../components/panels";
import { TopBar } from "../components/TopBar";
import { api } from "../lib/api";
import { callAddress, colorMode, foldParams, moduleEntry, nodeTones, parseFold, revealFold, toggleFold, withinGroup } from "../lib/graphState";
import { href, navigate, type Route } from "../lib/router";
import type { ColorMode, GraphView as GV, IndexEntry, RunMeta } from "../lib/types";
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

const COLOURS: { id: ColorMode; label: string }[] = [
  { id: "plain", label: "Plain" }, { id: "attr", label: "Attr" }, { id: "norm", label: "Norm" },
  { id: "sparsity", label: "Sparsity" }, { id: "nan", label: "NaN/Inf" },
];

export function GraphViewPage({ run, route }: { run: RunMeta; route: Route }) {
  const depth = Number(route.params.get("depth") ?? 2);
  const color = colorMode(route.params);
  const attr = color === "attr";
  const fold = parseFold(route.params);
  const fp = foldParams(fold);
  const graph = useAsync(() => api.graph(run.id, depth, attr ? { method: "gxa" } : undefined, fold), [run.id, depth, attr, fp.expand, fp.collapse]);
  const modules = useAsync(() => api.modules(run.id), [run.id]);
  const index = useAsync(() => api.index(run.id), [run.id]);
  const view = graph.data;
  const [query, setQuery] = useState("");
  const [centre, setCentre] = useState(0);

  const selectedId = route.params.get("node");
  const selected = useMemo(() => {
    if (!view) return null;
    const byId = (id: string | null) => (id ? view.nodes.find((n) => n.id === id) : undefined);
    return byId(selectedId)
      ?? view.nodes.find((n) => n.output_op === selectedId)
      ?? (selectedId?.startsWith("m:") ? byId(pickNodeForModule(view, callAddress(selectedId))) : undefined)
      ?? view.nodes.find((n) => !n.is_input) ?? null;
  }, [view, selectedId]);
  // focus (and search centring) waits until the requested node is really in the view, not a fallback
  const exact = !!selected && (!selectedId || selected.id === selectedId || selected.output_op === selectedId);
  const { tones, legend } = useMemo(() => (view ? nodeTones(view, color) : { tones: undefined, legend: null }), [view, color]);

  const state = (params: Record<string, string | null>) =>
    ({ node: selected?.id ?? null, depth: String(depth), ...fp, color: color === "plain" ? null : color, ...params });
  const go = (params: Record<string, string | null>) => navigate(href(run.id, "graph", state(params)), true);

  const onToggle = (call: string, open: boolean) => {
    // closing a module that holds the selection selects the folded module node instead
    const folds = !open && !!selected && withinGroup(view!, selected.group, "g:" + call);
    go({ ...foldParams(toggleFold(fold, call, open)), ...(folds && { node: "m:" + call }) });
  };
  const onOpen = (id: string) => {
    const n = view?.nodes.find((x) => x.id === id);
    if (n) navigate(href(run.id, "tensors", { node: n.output_op }));
  };
  const onPick = (e: IndexEntry) => {
    go({ ...foldParams(revealFold(fold, e)), node: e.id });
    setCentre((c) => c + 1);
  };
  // tree pick: a visible node for the module itself, else reveal it like search, else the old heuristic while the index loads
  const onTreePick = (addr: string) => {
    if (!view) return;
    const own = view.nodes.find((n) => n.path === addr);
    if (own) { go({ node: own.id }); setCentre((c) => c + 1); return; }
    const e = index.data && moduleEntry(index.data, addr);
    if (e) return onPick(e);
    const id = pickNodeForModule(view, addr);
    if (id) go({ node: id });
  };

  return (
    <>
      <TopBar run={run} tab="graph" node={selected?.id}>
        <span className="lbl dim">Depth</span>
        <div className="seg lbl" role="group" aria-label="Nesting depth">
          {[0, 1, 2, 3, 4].map((d) => (
            <button key={d} type="button" aria-pressed={d === depth} onClick={() => go({ depth: String(d), expand: null, collapse: null })}>{d}</button>
          ))}
          <button type="button" aria-pressed={depth >= 12} onClick={() => go({ depth: "12", expand: null, collapse: null })}>All</button>
        </div>
        <span className="lbl dim">Colour</span>
        <div className="seg lbl" role="group" aria-label="Node colour">
          {COLOURS.map((c) => (
            <button key={c.id} type="button" aria-pressed={c.id === color} onClick={() => go({ color: c.id === "plain" ? null : c.id })}>{c.label}</button>
          ))}
        </div>
      </TopBar>
      <div style={{ flexGrow: 1, minHeight: 0, display: "flex" }}>
        <aside style={{ width: 248, flexShrink: 0, borderRight: "1px solid var(--line)", display: "flex", flexDirection: "column", minHeight: 0 }}>
          {modules.data && (
            <ModuleTree modules={modules.data} selectedPath={selected?.path ?? null}
              onPick={onTreePick} />
          )}
        </aside>
        <section style={{ flexGrow: 1, minWidth: 0, position: "relative" }}>
          {graph.error && <div className="empty err">{graph.error}</div>}
          {view && (
            <>
              <GraphCanvas view={view} selected={selected?.id ?? null} focus={exact ? selected.id : null} centre={centre} tones={tones}
                onSelect={(id) => go({ node: id })} onToggle={onToggle} onOpen={onOpen} onEscape={() => setQuery("")} />
              <div style={{ position: "absolute", left: 16, top: 14, display: "flex", flexDirection: "column", gap: 3, pointerEvents: "none" }}>
                <span className="lbl">Forward trace</span>
                <span className="lbl dim">{run.sample.name} · {view.nodes.length} nodes · depth {depth >= 12 ? "all" : depth}</span>
              </div>
              {legend && (
                <div className="panel" style={{ position: "absolute", right: 16, top: 14, width: 200, padding: 10, border: "1px solid var(--line)", display: "flex", flexDirection: "column", gap: 6 }}>
                  <span className="lbl">{legend.title}</span>
                  {color === "nan"
                    ? <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ width: 22, height: 12, border: "1.5px solid var(--negative)", background: "var(--panel)" }} />
                        <span className="lbl" style={{ color: "var(--negative-text)" }}>{legend.hi}</span>
                      </div>
                    : <>
                        <div style={{ height: 4, background: "linear-gradient(90deg, var(--accent-soft), var(--accent))" }} />
                        <div className="lbl dim" style={{ display: "flex", justifyContent: "space-between" }}><span>{legend.lo}</span><span>{legend.hi}</span></div>
                      </>}
                  {color !== "nan" && <span className="lbl dim">Module = its output tensor</span>}
                  {legend.note && <span className="lbl dim">{legend.note}</span>}
                </div>
              )}
            </>
          )}
          <GraphSearch entries={index.data} error={index.error} query={query} onQuery={setQuery} onPick={onPick} />
          {graph.loading && !view && <div className="empty lbl">Loading graph…</div>}
        </section>
        <aside className="panel" style={{ width: 360, flexShrink: 0, borderLeft: "1px solid var(--line)", display: "flex", flexDirection: "column", minHeight: 0 }}>
          {view && <Inspector runId={run.id} view={view} node={selected} total={view.nodes.length}
            linkFor={(id) => href(run.id, "graph", state({ node: id }))} />}
        </aside>
      </div>
    </>
  );
}
