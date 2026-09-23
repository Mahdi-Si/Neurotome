// Pure viewport maths for GraphCanvas and Minimap: culling, hit testing, keyboard steps, colour mixing.
// ponytail: linear scans per frame are fine to ~20k nodes; add a grid index if the benchmark says otherwise.
import { displayName, shapeShort } from "./format";
import { parseColor } from "./theme";
import type { Box, Placed } from "./layout";
import type { GraphEdge, GraphNode, GraphView } from "./types";

export interface Tf { x: number; y: number; k: number }
export const DOM_MIN_ZOOM = 0.4;

export const toScreen = (tf: Tf, gx: number, gy: number): [number, number] => [gx * tf.k + tf.x, gy * tf.k + tf.y];
export const toGraph = (tf: Tf, sx: number, sy: number): [number, number] => [(sx - tf.x) / tf.k, (sy - tf.y) / tf.k];

/** The graph-space rectangle visible in a w×h viewport, inflated by `margin` screen px. */
export function viewRect(tf: Tf, w: number, h: number, margin = 0): Box {
  const [x0, y0] = toGraph(tf, -margin, -margin);
  return { x: x0, y: y0, w: (w + 2 * margin) / tf.k, h: (h + 2 * margin) / tf.k };
}

export const overlaps = (a: Box, b: Box) => a.x <= b.x + b.w && b.x <= a.x + a.w && a.y <= b.y + b.h && b.y <= a.y + a.h;

export function visibleBoxes(boxes: Map<string, Box>, tf: Tf, w: number, h: number, margin = 0): string[] {
  const r = viewRect(tf, w, h, margin);
  const out: string[] = [];
  for (const [id, b] of boxes) if (overlaps(r, b)) out.push(id);
  return out;
}

export const visibleNodes = (placed: Placed, tf: Tf, w: number, h: number, margin = 0) =>
  visibleBoxes(placed.nodes, tf, w, h, margin);

/** Indices into placed.edges whose bounding box meets the viewport. */
export function edgesInView(placed: Placed, tf: Tf, w: number, h: number): number[] {
  const r = viewRect(tf, w, h);
  const out: number[] = [];
  placed.edges.forEach((e, i) => { if (overlaps(r, e.bbox)) out.push(i); });
  return out;
}

export function hitNode(placed: Placed, gx: number, gy: number): string | null {
  for (const [id, b] of placed.nodes) if (gx >= b.x && gx <= b.x + b.w && gy >= b.y && gy <= b.y + b.h) return id;
  return null;
}

function segDist(px: number, py: number, [ax, ay]: [number, number], [bx, by]: [number, number]): number {
  const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy;
  const t = len ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len)) : 0;
  return Math.hypot(px - ax - t * dx, py - ay - t * dy);
}

/** Index into placed.edges of the nearest polyline within `tol` graph units, or null. */
export function hitEdge(placed: Placed, gx: number, gy: number, tol: number): number | null {
  let best: number | null = null, bestD = tol;
  placed.edges.forEach(({ points, bbox: b }, i) => {
    if (gx < b.x - tol || gx > b.x + b.w + tol || gy < b.y - tol || gy > b.y + b.h + tol) return;
    for (let j = 1; j < points.length; j++) {
      const d = segDist(gx, gy, points[j - 1], points[j]);
      if (d <= bestD) { bestD = d; best = i; }
    }
  });
  return best;
}

/** Keyboard move from `current`: ↓ child, ↑ parent, ←/→ sibling, each nearest in x. Null = no move. */
export function step(view: GraphView, placed: Placed, current: string | null, key: string): string | null {
  if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(key)) return null;
  const cur = current ? placed.nodes.get(current) : undefined;
  if (!current || !cur) {
    const first = view.nodes.find((n) => n.is_input && placed.nodes.has(n.id)) ?? view.nodes.find((n) => placed.nodes.has(n.id));
    return first?.id ?? null;
  }
  const cx = (id: string) => { const b = placed.nodes.get(id)!; return b.x + b.w / 2; };
  const kids = (id: string) => view.edges.filter((e) => e.source === id).map((e) => e.target);
  const parents = (id: string) => view.edges.filter((e) => e.target === id).map((e) => e.source);
  const x = cx(current);
  let pool: string[];
  if (key === "ArrowDown") pool = kids(current);
  else if (key === "ArrowUp") pool = parents(current);
  else {
    pool = [...parents(current).flatMap(kids), ...kids(current).flatMap(parents)]
      .filter((id) => id !== current && placed.nodes.has(id) && (key === "ArrowLeft" ? cx(id) < x : cx(id) > x));
  }
  let best: string | null = null, bestD = Infinity;
  for (const id of pool) {
    if (!placed.nodes.has(id)) continue;
    const d = Math.abs(cx(id) - x);
    if (d < bestD) { bestD = d; best = id; }
  }
  return best;
}

/** Linear blend of two colours ("#rgb", "#rrggbb", "rgb()"), as "#rrggbb". */
export function mix(a: string, b: string, t: number): string {
  const x = parseColor(a), y = parseColor(b);
  return "#" + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

export interface Seg { t: string; hi?: boolean }
const bare = (s: string | null | undefined) => (s ? s.replace(/^torch\./, "") : null);

function pair(a: string | null | undefined, b: string | null | undefined): Seg | null {
  const x = bare(a), y = bare(b);
  if (x && y && x !== y) return { t: `${x} → ${y}`, hi: true };
  return x || y ? { t: (x || y)! } : null;
}

/** Hover readout for an edge: "src → tgt · shape · dtype · device"; changes across the edge are highlighted. */
export function edgeReadout(e: GraphEdge, src: GraphNode | undefined, tgt: GraphNode | undefined): Seg[] {
  const name = (n: GraphNode | undefined, id: string) => (n ? displayName(n) : id);
  const segs: (Seg | null)[] = [
    { t: `${name(src, e.source)} → ${name(tgt, e.target)}` },
    e.shape?.length ? { t: shapeShort(e.shape) } : null,
    pair(e.dtype, e.target_dtype),
    pair(e.device, e.target_device),
  ];
  return segs.filter((s): s is Seg => s !== null);
}
