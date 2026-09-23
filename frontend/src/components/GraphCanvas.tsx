// Pan/zoom computation graph. ELK does the layout; nodes are real <button>s over an SVG edge layer.
// mode "plain": thin ink edges, attribution bar on each node (if scores are present).
// mode "flow":  edge width = strength of attribution flowing through it, dominant path in accent.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { compact, pct, shapeShort } from "../lib/format";
import { layoutGraph, type Placed } from "../lib/layout";
import type { GraphNode, GraphView } from "../lib/types";

interface Props {
  view: GraphView;
  selected: string | null;
  onSelect: (id: string) => void;
  mode?: "plain" | "flow";
  focus?: string | null; // node to scroll into view when it changes
}

interface Tf { x: number; y: number; k: number }

export function GraphCanvas({ view, selected, onSelect, mode = "plain", focus }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [placed, setPlaced] = useState<Placed | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tf, setTf] = useState<Tf>({ x: 0, y: 0, k: 1 });
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  useEffect(() => {
    let alive = true;
    setError(null);
    layoutGraph(view).then((p) => alive && setPlaced(p)).catch((e) => alive && setError(String(e)));
    return () => { alive = false; };
  }, [view]);

  const fit = useCallback((full: boolean) => {
    const el = wrap.current;
    if (!el || !placed) return;
    const W = el.clientWidth, H = el.clientHeight;
    const k = full
      ? Math.min(1, (W - 40) / placed.width, (H - 40) / placed.height)
      : Math.min(1, (W - 40) / placed.width);
    setTf({ k, x: (W - placed.width * k) / 2, y: full ? (H - placed.height * k) / 2 : 56 });
  }, [placed]);

  useLayoutEffect(() => { fit(false); }, [fit]);

  // bring the focused node into view
  useEffect(() => {
    const el = wrap.current;
    const box = focus && placed?.nodes.get(focus);
    if (!el || !box) return;
    setTf((t) => {
      const cx = box.x * t.k + t.x, cy = box.y * t.k + t.y;
      const inside = cx > 20 && cy > 20 && cx + box.w * t.k < el.clientWidth - 20 && cy + box.h * t.k < el.clientHeight - 20;
      if (inside) return t;
      return { ...t, x: el.clientWidth / 2 - (box.x + box.w / 2) * t.k, y: el.clientHeight / 2 - (box.y + box.h / 2) * t.k };
    });
  }, [focus, placed]);

  // wheel: pan; ctrl/cmd + wheel (and trackpad pinch): zoom around the cursor
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const r = el.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        setTf((t) => {
          const k = Math.min(2.5, Math.max(0.08, t.k * Math.exp(-e.deltaY * 0.01)));
          return { k, x: mx - ((mx - t.x) / t.k) * k, y: my - ((my - t.y) / t.k) * k };
        });
      } else {
        setTf((t) => ({ ...t, x: t.x - e.deltaX, y: t.y - e.deltaY }));
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const zoom = (f: number) => {
    const el = wrap.current;
    if (!el) return;
    const mx = el.clientWidth / 2, my = el.clientHeight / 2;
    setTf((t) => {
      const k = Math.min(2.5, Math.max(0.08, t.k * f));
      return { k, x: mx - ((mx - t.x) / t.k) * k, y: my - ((my - t.y) / t.k) * k };
    });
  };

  const nodes = useMemo(() => new Map(view.nodes.map((n) => [n.id, n])), [view]);
  const groups = useMemo(() => new Map(view.groups.map((g) => [g.id, g])), [view]);
  const pathSet = useMemo(() => new Set(view.path ?? []), [view]);
  const hasScores = view.nodes.some((n) => n.score_norm != null);
  const parentCount = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of view.edges) m.set(e.target, (m.get(e.target) ?? 0) + 1);
    return m;
  }, [view]);

  return (
    <div ref={wrap} className="dotgrid" style={{ position: "relative", overflow: "hidden", width: "100%", height: "100%", cursor: drag.current ? "grabbing" : "grab", touchAction: "none" }}
      onPointerDown={(e) => {
        if ((e.target as HTMLElement).closest("button")) return;
        drag.current = { x: e.clientX, y: e.clientY, tx: tf.x, ty: tf.y };
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (d) setTf((t) => ({ ...t, x: d.tx + e.clientX - d.x, y: d.ty + e.clientY - d.y }));
      }}
      onPointerUp={() => { drag.current = null; }}
      aria-label="Computation graph">
      {error && <div className="empty err">Layout failed: {error}</div>}
      {!placed && !error && <div className="empty lbl">Laying out {view.nodes.length} nodes…</div>}
      {placed && (
        <div style={{ position: "absolute", left: 0, top: 0, transformOrigin: "0 0", transform: `translate(${tf.x}px, ${tf.y}px) scale(${tf.k})` }}>
          <svg width={placed.width} height={placed.height} style={{ position: "absolute", left: 0, top: 0, overflow: "visible" }} aria-hidden="true">
            <defs>
              <marker id="ah" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0 0.5 5 3 0 5.5" stroke="var(--ink)" strokeWidth="1" fill="none" />
              </marker>
              <marker id="ah-acc" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">
                <path d="M0 0.5 5 3 0 5.5" stroke="var(--accent)" strokeWidth="1.2" fill="none" />
              </marker>
            </defs>
            {[...placed.groups.entries()].map(([id, b]) => (
              <rect key={id} x={b.x + 0.5} y={b.y + 0.5} width={b.w} height={b.h} fill="rgba(250,250,248,.55)" stroke="var(--ink)" strokeDasharray="3 3" />
            ))}
            {placed.edges.map(({ edge, points }, i) => {
              const d = points.map((p, j) => `${j ? "L" : "M"}${p[0]} ${p[1]}`).join("");
              if (mode === "flow") {
                const src = nodes.get(edge.source);
                const strength = src?.score_norm ?? 0;
                const on = !!edge.on_path;
                const merge = (parentCount.get(edge.target) ?? 0) > 1 && edge.share != null;
                const end = points[points.length - 1];
                return (
                  <g key={i}>
                    <path d={d} fill="none" stroke={on ? "var(--accent)" : "var(--line-faint)"} strokeWidth={1 + strength * 4}
                      strokeLinejoin="round" markerEnd={on ? "url(#ah-acc)" : undefined} />
                    {merge && (
                      <text x={end[0] + 6} y={end[1] - 8} fontFamily="var(--font-mono)" fontSize="10"
                        fill={on ? "var(--accent)" : "var(--ink-2)"}>{pct(edge.share)}</text>
                    )}
                  </g>
                );
              }
              return <path key={i} d={d} fill="none" stroke="var(--ink)" strokeWidth="1" markerEnd="url(#ah)" />;
            })}
            {[...placed.groups.entries()].map(([id, b]) => {
              const g = groups.get(id);
              const right = `${g?.num_ops ?? 0} OPS · ${compact(g?.num_params ?? 0)}`;
              const budget = Math.max(6, Math.floor((b.w - 32) / 6.4) - right.length);
              const full = `${g?.address ?? id} · ${g?.class_name ?? ""}`.toUpperCase();
              const left = full.length > budget ? `${full.slice(0, budget - 1)}…` : full;
              const halo = { paintOrder: "stroke", stroke: "var(--paper)", strokeWidth: 5, strokeLinejoin: "round" } as const;
              return (
                <g key={`l-${id}`}>
                  <text x={b.x + 10} y={b.y + 18} fontFamily="var(--font-mono)" fontSize="10" letterSpacing=".05em" fill="var(--ink)" {...halo}>
                    <title>{full}</title>{left}
                  </text>
                  <text x={b.x + b.w - 10} y={b.y + 18} textAnchor="end" fontFamily="var(--font-mono)" fontSize="10" fill="var(--ink-2)" {...halo}>
                    {right}
                  </text>
                </g>
              );
            })}
          </svg>
          {[...placed.nodes.entries()].map(([id, b]) => {
            const n = nodes.get(id);
            return n ? (
              <NodeCard key={id} n={n} x={b.x} y={b.y} w={b.w} h={b.h} selected={selected === id}
                onPath={mode === "flow" && pathSet.has(id)} showBar={hasScores} onClick={() => onSelect(id)} />
            ) : null;
          })}
        </div>
      )}
      <div title="Drag or scroll to pan · ⌘/Ctrl + scroll to zoom" style={{ position: "absolute", right: 16, bottom: 16, display: "flex", border: "1px solid var(--ink)", background: "var(--panel)" }}>
        <button type="button" aria-label="Zoom out" onClick={() => zoom(1 / 1.25)} style={zbtn}>−</button>
        <span className="lbl" style={{ width: 52, display: "flex", alignItems: "center", justifyContent: "center", borderRight: "1px solid var(--ink)" }}>
          {Math.round(tf.k * 100)}%
        </span>
        <button type="button" aria-label="Zoom in" onClick={() => zoom(1.25)} style={zbtn}>+</button>
        <button type="button" className="lbl" onClick={() => fit(true)} style={{ ...zbtn, width: "auto", padding: "0 12px", borderRight: 0 }}>Fit</button>
      </div>
    </div>
  );
}

const zbtn: React.CSSProperties = {
  width: 32, height: 30, background: "transparent", border: 0, borderRight: "1px solid var(--ink)",
  cursor: "pointer", fontFamily: "var(--font-mono)",
};

function NodeCard({ n, x, y, w, h, selected, onPath, showBar, onClick }: {
  n: GraphNode; x: number; y: number; w: number; h: number; selected: boolean; onPath: boolean; showBar: boolean; onClick: () => void;
}) {
  const hot = selected || onPath;
  return (
    <button type="button" onClick={onClick} title={`${n.path}\n${n.type} · ${shapeShort(n.shape)}`}
      aria-pressed={selected}
      style={{
        position: "absolute", left: x, top: y, width: w, height: h, padding: "0 10px", display: "flex",
        flexDirection: "column", justifyContent: "center", alignItems: "stretch", gap: 1, textAlign: "left",
        cursor: "pointer", overflow: "hidden", borderRadius: "var(--radius)",
        background: n.is_output ? "var(--ink)" : hot ? "var(--accent-soft)" : "var(--panel)",
        color: n.is_output ? "var(--panel)" : "var(--ink)",
        border: selected ? "1.5px solid var(--accent)" : n.is_input ? "1px dashed var(--ink)" : onPath ? "1px solid var(--accent)" : "1px solid var(--ink)",
        boxShadow: n.kind === "module" ? "4px 4px 0 -1px var(--paper), 4px 4px 0 0 var(--ink)" : "none",
      }}>
      <span style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
        <span className="lbl" style={{ fontSize: 9.5, color: n.is_output ? "inherit" : "var(--ink-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {n.kind === "module" ? `${n.type} · ${n.num_ops} ops` : n.type}
        </span>
        <span className="num" style={{ fontSize: 9.5, color: n.is_output ? "inherit" : "var(--ink-2)", whiteSpace: "nowrap" }}>{shapeShort(n.shape)}</span>
      </span>
      <span style={{ fontSize: 13, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{n.label}</span>
      {showBar && n.score_norm != null && !n.is_output && (
        <span style={{ position: "absolute", left: 0, bottom: 0, height: 3, width: `${Math.round(n.score_norm * 100)}%`, background: "var(--accent)", opacity: 0.35 + n.score_norm * 0.65 }} />
      )}
    </button>
  );
}
