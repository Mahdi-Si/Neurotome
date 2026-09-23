// Pan/zoom computation graph. ELK does the layout (in a worker, cached); edges, group boxes and far-zoom
// nodes are drawn on one canvas; nodes in view at k >= DOM_MIN_ZOOM are real <button>s on top.
// mode "plain": thin ink edges, tone tint / attribution bar on each node.
// mode "flow":  edge width = strength of attribution flowing through it, dominant path in accent.
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { unsavedText } from "../lib/capture";
import { compact, displayName, pct, shapeShort } from "../lib/format";
import { layoutGraph, type Box, type Placed } from "../lib/layout";
import { useTokens, type Tokens } from "../lib/theme";
import type { GraphNode, GraphView, NodeTone } from "../lib/types";
import {
  DOM_MIN_ZOOM, edgeReadout, edgesInView, hitEdge, hitNode, mix, overlaps, step, toGraph, toScreen, viewRect,
  visibleBoxes, visibleNodes, type Seg, type Tf,
} from "../lib/viewport";
import { Minimap } from "./Minimap";

interface Props {
  view: GraphView;
  selected: string | null;
  onSelect: (id: string) => void;
  mode?: "plain" | "flow";
  focus?: string | null; // keep this node in view when it changes
  centre?: number; // when this number changes, centre `focus` once it is placed
  tones?: Map<string, NodeTone>; // colour mode; absent → score bar when scores exist
  onToggle?: (call: string, open: boolean) => void; // "+" on module node (open), "−" on group (close)
  onOpen?: (id: string) => void; // Enter on the selected node
  onEscape?: () => void; // Escape while the canvas has focus
}

const tint = (c: Tokens, v: number) => mix(c.panel, c.accent, 0.08 + 0.32 * v);
const ZMIN = 0.02, ZMAX = 2.5;
const clampK = (k: number) => Math.min(ZMAX, Math.max(ZMIN, k));
type Hover = { node?: string; edge?: number } | null;

function trace(ctx: CanvasRenderingContext2D, pts: [number, number][], ah: number) {
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let j = 1; j < pts.length; j++) ctx.lineTo(pts[j][0], pts[j][1]);
  if (!ah || pts.length < 2) return;
  const [ex, ey] = pts[pts.length - 1], [px, py] = pts[pts.length - 2];
  const len = Math.hypot(ex - px, ey - py) || 1, ux = (ex - px) / len, uy = (ey - py) / len;
  ctx.moveTo(ex - ah * ux - 0.6 * ah * uy, ey - ah * uy + 0.6 * ah * ux);
  ctx.lineTo(ex, ey);
  ctx.lineTo(ex - ah * ux + 0.6 * ah * uy, ey - ah * uy - 0.6 * ah * ux);
}

export function GraphCanvas({ view, selected, onSelect, mode = "plain", focus, centre, tones, onToggle, onOpen, onEscape }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [laid, setLaid] = useState<{ view: GraphView; placed: Placed } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tf, setTf] = useState<Tf>({ x: 0, y: 0, k: 1 });
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState<Hover>(null);
  const drag = useRef<{ x: number; y: number; tx: number; ty: number; moved: boolean } | null>(null);
  const anchor = useRef<{ id: string; sx: number; sy: number } | null>(null);
  const fitted = useRef<number | null>(null);
  const seen = useRef<{ focus?: string | null; centre?: number }>({ centre });
  const ptr = useRef<{ x: number; y: number } | null>(null);
  const hoverRaf = useRef(0);
  const c = useTokens();
  const live = useRef({ tf, laid, onSelect, onToggle });
  live.current = { tf, laid, onSelect, onToggle };

  useEffect(() => {
    let alive = true;
    setError(null);
    layoutGraph(view).then((placed) => alive && setLaid({ view, placed })).catch((e) => alive && setError(String(e)));
    return () => { alive = false; };
  }, [view]);

  useLayoutEffect(() => {
    const el = wrap.current!;
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const fit = useCallback((full: boolean) => {
    const el = wrap.current, p = live.current.laid?.placed;
    if (!el || !p) return;
    const W = el.clientWidth, H = el.clientHeight;
    // the first view keeps labelled DOM nodes (k >= DOM_MIN_ZOOM); "Fit" shows the whole graph however small
    const k = full ? Math.min(1, (W - 40) / p.width, (H - 40) / p.height) : Math.min(1, Math.max(DOM_MIN_ZOOM, (W - 40) / p.width));
    setTf({ k, x: full ? (W - p.width * k) / 2 : Math.max(20, (W - p.width * k) / 2), y: full ? (H - p.height * k) / 2 : 56 });
  }, []);

  // refit on first placement or a depth change; after a fold toggle keep the toggled element where it was
  useLayoutEffect(() => {
    if (!laid) return;
    const a = anchor.current;
    anchor.current = null;
    if (fitted.current !== laid.view.depth) { fitted.current = laid.view.depth; fit(false); return; }
    const b = a && (laid.placed.nodes.get(a.id) ?? laid.placed.groups.get(a.id));
    if (a && b) setTf((t) => ({ ...t, x: a.sx - b.x * t.k, y: a.sy - b.y * t.k }));
  }, [laid, fit]);

  const reveal = useCallback((b: Box, force: boolean) => {
    const el = wrap.current;
    if (!el) return;
    const W = el.clientWidth, H = el.clientHeight;
    setTf((t) => {
      const [sx, sy] = toScreen(t, b.x, b.y);
      const inside = sx > 20 && sy > 20 && sx + b.w * t.k < W - 20 && sy + b.h * t.k < H - 20;
      return inside && !force ? t : { ...t, x: W / 2 - (b.x + b.w / 2) * t.k, y: H / 2 - (b.y + b.h / 2) * t.k };
    });
  }, []);

  // bring `focus` into view when it changes; centre it when `centre` changes
  useEffect(() => {
    const box = focus && laid?.view === view ? laid.placed.nodes.get(focus) : undefined;
    if (!box) return;
    const force = centre !== seen.current.centre;
    if (!force && focus === seen.current.focus) return;
    seen.current = { focus, centre };
    reveal(box, force);
  }, [focus, centre, laid, view, reveal]);

  // wheel: pan; ctrl/cmd + wheel (and trackpad pinch): zoom around the cursor
  useEffect(() => {
    const el = wrap.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const r = el.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        setTf((t) => {
          const k = clampK(t.k * Math.exp(-e.deltaY * 0.01));
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
    const el = wrap.current!;
    const mx = el.clientWidth / 2, my = el.clientHeight / 2;
    setTf((t) => {
      const k = clampK(t.k * f);
      return { k, x: mx - ((mx - t.x) / t.k) * k, y: my - ((my - t.y) / t.k) * k };
    });
  };

  const pick = useCallback((id: string) => {
    live.current.onSelect(id);
    wrap.current?.focus({ preventScroll: true });
  }, []);

  // id is "m:<call>" (open) or "g:<call>" (close); remember where it sits so the counterpart lands there
  const toggle = useCallback((id: string, open: boolean) => {
    const { tf: t, laid: l, onToggle: cb } = live.current;
    const b = l && (open ? l.placed.nodes.get(id) : l.placed.groups.get(id));
    const call = id.slice(2);
    if (b) anchor.current = { id: (open ? "g:" : "m:") + call, sx: b.x * t.k + t.x, sy: b.y * t.k + t.y };
    cb?.(call, open);
    wrap.current?.focus({ preventScroll: true });
  }, []);

  const probe = () => {
    hoverRaf.current = 0;
    const p = live.current.laid?.placed, q = ptr.current, t = live.current.tf;
    if (!p || !q || drag.current) { setHover(null); return; }
    const [gx, gy] = toGraph(t, q.x, q.y);
    const n = hitNode(p, gx, gy);
    if (n) { setHover((h) => (h?.node === n ? h : { node: n })); return; }
    const e = hitEdge(p, gx, gy, 5 / t.k);
    setHover((h) => (e == null ? null : h?.edge === e ? h : { edge: e }));
  };

  const nodes = useMemo(() => new Map((laid?.view.nodes ?? []).map((n) => [n.id, n])), [laid]);
  const groups = useMemo(() => new Map((laid?.view.groups ?? []).map((g) => [g.id, g])), [laid]);
  const pathSet = useMemo(() => new Set(laid?.view.path ?? []), [laid]);
  const hasScores = useMemo(() => !!laid?.view.nodes.some((n) => n.score_norm != null), [laid]);
  const parentCount = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of laid?.view.edges ?? []) m.set(e.target, (m.get(e.target) ?? 0) + 1);
    return m;
  }, [laid]);

  const dom = !!laid && tf.k >= DOM_MIN_ZOOM;
  const shownNodes = useMemo(() => (dom ? visibleNodes(laid!.placed, tf, size.w, size.h, 200) : []), [dom, laid, tf, size]);
  const shownGroups = useMemo(() => (dom ? visibleBoxes(laid!.placed.groups, tf, size.w, size.h, 200) : []), [dom, laid, tf, size]);

  // one canvas for groups, edges and (below DOM_MIN_ZOOM) nodes; redrawn at most once per frame
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      const cv = canvas.current;
      if (!cv) return;
      const dpr = window.devicePixelRatio || 1;
      const cw = Math.round(size.w * dpr), ch = Math.round(size.h * dpr);
      if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
      const ctx = cv.getContext("2d")!;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, cw, ch);
      if (!laid) return;
      const { placed, view: lv } = laid;
      const { k } = tf, px = 1 / k, r = viewRect(tf, size.w, size.h);
      ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * tf.x, dpr * tf.y);

      ctx.lineWidth = px;
      ctx.setLineDash([3 * px, 3 * px]);
      ctx.strokeStyle = c.ink;
      for (const b of placed.groups.values()) {
        if (!overlaps(r, b)) continue;
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = c.panel;
        ctx.fillRect(b.x, b.y, b.w, b.h);
        ctx.globalAlpha = 1;
        ctx.strokeRect(b.x + px / 2, b.y + px / 2, b.w, b.h);
      }
      ctx.setLineDash([]);

      const vis = edgesInView(placed, tf, size.w, size.h);
      const ah = k >= 0.25 ? 5 * px : 0;
      if (mode === "flow") {
        const buckets = new Map<string, number[]>();
        for (const i of vis) {
          const e = lv.edges[placed.edges[i].i];
          const w = Math.round(1 + (nodes.get(e.source)?.score_norm ?? 0) * 4);
          const key = `${e.on_path ? "a" : "f"}${w}`;
          const b = buckets.get(key);
          if (b) b.push(i); else buckets.set(key, [i]);
        }
        for (const [key, idx] of buckets) {
          const on = key[0] === "a";
          ctx.beginPath();
          for (const i of idx) trace(ctx, placed.edges[i].points, on ? ah * 1.4 : 0);
          ctx.strokeStyle = on ? c.accent : c.lineFaint;
          ctx.lineWidth = Number(key.slice(1)) * px;
          ctx.lineJoin = "round";
          ctx.stroke();
        }
        if (k >= DOM_MIN_ZOOM) {
          ctx.font = `10px ${c.fontMono}`;
          for (const i of vis) {
            const pe = placed.edges[i], e = lv.edges[pe.i];
            if (e.share == null || (parentCount.get(e.target) ?? 0) < 2) continue;
            const [ex, ey] = pe.points[pe.points.length - 1];
            ctx.fillStyle = e.on_path ? c.accent : c.ink2;
            ctx.fillText(pct(e.share), ex + 6, ey - 8);
          }
        }
      } else {
        ctx.beginPath();
        for (const i of vis) trace(ctx, placed.edges[i].points, ah);
        ctx.strokeStyle = c.ink;
        ctx.lineWidth = px;
        ctx.stroke();
      }
      if (hover?.edge != null && placed.edges[hover.edge]) {
        ctx.beginPath();
        trace(ctx, placed.edges[hover.edge].points, ah);
        ctx.strokeStyle = c.accent;
        ctx.lineWidth = 1.5 * px;
        ctx.stroke();
      }

      if (k >= DOM_MIN_ZOOM) return;
      for (const id of visibleNodes(placed, tf, size.w, size.h)) {
        const n = nodes.get(id), b = placed.nodes.get(id)!, t = tones?.get(id);
        if (!n) continue;
        const hot = id === selected || (mode === "flow" && pathSet.has(id));
        ctx.fillStyle = n.is_output ? c.ink : t?.v != null ? tint(c, t.v) : hot ? c.accentSoft : c.panel;
        ctx.fillRect(b.x, b.y, b.w, b.h);
        const strong = id === selected || t?.flag;
        ctx.strokeStyle = id === selected ? c.accent : t?.flag ? c.negative : c.ink;
        ctx.lineWidth = (strong ? 1.5 : 1) * px;
        ctx.setLineDash(n.unsaved && !strong ? [1.5 * px, 1.5 * px] : []);
        ctx.strokeRect(b.x, b.y, b.w, b.h);
      }
      ctx.setLineDash([]);
    });
    return () => cancelAnimationFrame(id);
  }, [laid, tf, size, tones, hover, selected, mode, c, nodes, pathSet, parentCount]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.target !== e.currentTarget || !laid || e.ctrlKey || e.metaKey || e.altKey) return;
    const key = e.key;
    if (key.startsWith("Arrow")) {
      e.preventDefault();
      const next = step(laid.view, laid.placed, selected, key);
      if (next) { onSelect(next); reveal(laid.placed.nodes.get(next)!, false); }
    } else if (key === "Enter") {
      e.preventDefault();
      if (selected) onOpen?.(selected);
    } else if (key === "Escape") {
      e.preventDefault();
      onEscape?.();
    } else if (key === "+" || key === "=" || key === "-") {
      e.preventDefault();
      const n = selected ? nodes.get(selected) : undefined;
      if (!onToggle || !n) return;
      if (key !== "-" && n.kind === "module") toggle(n.id, true);
      if (key === "-" && n.group && laid.placed.groups.has(n.group)) toggle(n.group, false);
    }
  };

  let readout: Seg[] | null = null;
  if (laid && hover?.node) {
    const n = nodes.get(hover.node), t = tones?.get(hover.node);
    if (n) readout = [{ t: displayName(n) }, { t: n.type }, { t: shapeShort(n.shape) }, ...(t ? [{ t: t.readout }] : []),
      ...(n.unsaved ? [{ t: `no tensor · ${unsavedText(n.unsaved)}` }] : [])];
  } else if (laid && hover?.edge != null) {
    const e = laid.view.edges[laid.placed.edges[hover.edge]?.i];
    if (e) readout = edgeReadout(e, nodes.get(e.source), nodes.get(e.target));
  }

  return (
    <div ref={wrap} className="dotgrid graph-canvas" tabIndex={0} role="application"
      aria-label="Computation graph. Arrow keys follow edges, Enter opens tensor"
      style={{ position: "relative", overflow: "hidden", width: "100%", height: "100%", cursor: "grab", touchAction: "none" }}
      onKeyDown={onKeyDown}
      onPointerDown={(e) => {
        if ((e.target as HTMLElement).closest("button")) return;
        drag.current = { x: e.clientX, y: e.clientY, tx: tf.x, ty: tf.y, moved: false };
        e.currentTarget.setPointerCapture(e.pointerId);
        setHover(null);
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (d) {
          if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 3) d.moved = true;
          setTf((t) => ({ ...t, x: d.tx + e.clientX - d.x, y: d.ty + e.clientY - d.y }));
          return;
        }
        const r = e.currentTarget.getBoundingClientRect();
        ptr.current = { x: e.clientX - r.left, y: e.clientY - r.top };
        hoverRaf.current ||= requestAnimationFrame(probe);
      }}
      onPointerUp={(e) => {
        const d = drag.current;
        drag.current = null;
        if (!d || d.moved || !laid || tf.k >= DOM_MIN_ZOOM) return;
        const r = e.currentTarget.getBoundingClientRect();
        const id = hitNode(laid.placed, ...toGraph(tf, e.clientX - r.left, e.clientY - r.top));
        if (id) pick(id);
      }}
      onPointerLeave={() => { ptr.current = null; setHover(null); }}>
      <canvas ref={canvas} aria-hidden="true" style={{ position: "absolute", left: 0, top: 0, width: size.w, height: size.h, pointerEvents: "none" }} />
      {error && <div className="empty err">Layout failed: {error}</div>}
      {!laid && !error && <div className="empty lbl">Laying out {view.nodes.length} nodes…</div>}
      {dom && (
        <div style={{ position: "absolute", left: 0, top: 0, transformOrigin: "0 0", transform: `translate(${tf.x}px, ${tf.y}px) scale(${tf.k})` }}>
          {shownGroups.map((id) => {
            const b = laid!.placed.groups.get(id)!, g = groups.get(id);
            const full = `${g?.address ?? id} · ${g?.class_name ?? ""}`.toUpperCase();
            return (
              <div key={id} style={{ position: "absolute", left: b.x + 8, top: b.y + 9, width: b.w - 16, display: "flex", alignItems: "center", gap: 6, fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: ".05em", lineHeight: "16px" }}>
                {onToggle && (
                  <button type="button" tabIndex={-1} aria-label={`Collapse ${g?.address ?? id}`} title="Collapse"
                    onClick={() => toggle(id, false)} style={toggleStyle}>−</button>
                )}
                <span title={full} style={{ ...halo, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{full}</span>
                <span style={{ ...halo, marginLeft: "auto", color: "var(--ink-2)", whiteSpace: "nowrap" }}>
                  {`${g?.num_ops ?? 0} OPS · ${compact(g?.num_params ?? 0)}`}
                </span>
              </div>
            );
          })}
          {shownNodes.map((id) => {
            const n = nodes.get(id), b = laid!.placed.nodes.get(id)!;
            if (!n) return null;
            return (
              <NodeCard key={id} n={n} b={b} selected={selected === id} onPath={mode === "flow" && pathSet.has(id)}
                showBar={hasScores} tone={tones?.get(id)} toned={!!tones} colors={c} onPick={pick}
                onExpand={onToggle && n.kind === "module" ? toggle : undefined} />
            );
          })}
        </div>
      )}
      {laid && (
        <Minimap placed={laid.placed} tones={tones} selected={selected} tf={tf} w={size.w} h={size.h} colors={c}
          onCentre={(gx, gy) => setTf((t) => ({ ...t, x: size.w / 2 - gx * t.k, y: size.h / 2 - gy * t.k }))} />
      )}
      {readout && (
        <div className="lbl" style={{ position: "absolute", left: 232, right: 200, bottom: 22, pointerEvents: "none", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {readout.map((s, i) => (
            <span key={i}>{i ? " · " : ""}<span style={s.hi ? { color: "var(--accent)" } : undefined}>{s.t}</span></span>
          ))}
        </div>
      )}
      <div title="Drag or scroll to pan · ⌘/Ctrl + scroll to zoom" onClick={() => wrap.current?.focus({ preventScroll: true })} style={{ position: "absolute", right: 16, bottom: 16, display: "flex", border: "1px solid var(--ink)", background: "var(--panel)" }}>
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

const toggleStyle: React.CSSProperties = {
  width: 16, height: 16, flexShrink: 0, padding: 0, display: "flex", alignItems: "center", justifyContent: "center",
  fontFamily: "var(--font-mono)", fontSize: 11, lineHeight: 1, border: "1px solid var(--ink)", background: "var(--panel)",
  borderRadius: "var(--radius)", cursor: "pointer",
};

const halo: React.CSSProperties = { background: "var(--paper)", padding: "0 3px" };

const NodeCard = memo(function NodeCard({ n, b, selected, onPath, showBar, tone, toned, colors, onPick, onExpand }: {
  n: GraphNode; b: Box; selected: boolean; onPath: boolean; showBar: boolean; tone?: NodeTone; toned: boolean;
  colors: Tokens; onPick: (id: string) => void; onExpand?: (id: string, open: boolean) => void;
}) {
  const hot = selected || onPath;
  const bar = toned ? tone?.v : showBar ? n.score_norm : null;
  const fill = !n.is_output && tone?.v != null ? tint(colors, tone.v) : null;
  const title = `${n.path}\n${n.type} · ${shapeShort(n.shape)}${tone ? `\n${tone.readout}` : ""}`
    + (n.unsaved ? `\nno tensor · ${unsavedText(n.unsaved)}` : "");
  return (
    <>
      <button type="button" tabIndex={-1} onClick={() => onPick(n.id)} title={title} aria-pressed={selected}
        style={{
          position: "absolute", left: b.x, top: b.y, width: b.w, height: b.h, padding: "0 10px", display: "flex",
          flexDirection: "column", justifyContent: "center", alignItems: "stretch", gap: 1, textAlign: "left",
          cursor: "pointer", overflow: "hidden", borderRadius: "var(--radius)",
          background: n.is_output ? "var(--ink)" : fill ?? (hot ? "var(--accent-soft)" : "var(--panel)"),
          color: n.is_output ? "var(--panel)" : "var(--ink)",
          border: selected ? "1.5px solid var(--accent)" : tone?.flag ? "1.5px solid var(--negative)"
            : n.is_input ? "1px dashed var(--ink)" : onPath ? "1px solid var(--accent)" : n.unsaved ? "1px dotted var(--ink)" : "1px solid var(--ink)",
          boxShadow: n.kind === "module" ? "4px 4px 0 -1px var(--paper), 4px 4px 0 0 var(--ink)" : "none",
        }}>
        <span style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
          <span className="lbl" style={{ fontSize: 9.5, color: n.is_output ? "inherit" : "var(--ink-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {n.kind === "module" ? `${n.type} · ${n.num_ops} ops` : n.type}
          </span>
          {tone?.flag
            ? <span className="lbl" style={{ fontSize: 9.5, color: "var(--negative-text)", whiteSpace: "nowrap" }}>NAN/INF</span>
            : <span className="num" style={{ fontSize: 9.5, color: n.is_output ? "inherit" : "var(--ink-2)", whiteSpace: "nowrap", marginRight: onExpand ? 8 : 0 }}>{n.unsaved ? `∅ ${shapeShort(n.shape)}` : shapeShort(n.shape)}</span>}
        </span>
        <span style={{ fontSize: 13, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{n.label}</span>
        {bar != null && !n.is_output && (
          <span style={{ position: "absolute", left: 0, bottom: 0, height: 3, width: `${Math.round(bar * 100)}%`, background: "var(--accent)", opacity: toned ? 1 : 0.35 + bar * 0.65 }} />
        )}
      </button>
      {onExpand && (
        <button type="button" tabIndex={-1} aria-label={`Expand ${n.path}`} title="Expand" onClick={() => onExpand(n.id, true)}
          style={{ ...toggleStyle, position: "absolute", left: b.x + b.w - 9, top: b.y - 7 }}>+</button>
      )}
    </>
  );
});
