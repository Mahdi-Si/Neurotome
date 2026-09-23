// Graph layout with ELK (layered, top-down, orthogonal edges, nested module groups).
// ELK runs in a Web Worker when the browser has one; results are cached by graph structure.
import type { ELK as ElkApi, ElkExtendedEdge, ElkNode } from "elkjs/lib/elk-api";
import type { GraphView } from "./types";

export const NODE_W = 196;
export const NODE_H = 46;

export interface Box { x: number; y: number; w: number; h: number }
/** `i` indexes `view.edges` of any view with the same layoutKey; `bbox` bounds the polyline. */
export interface PlacedEdge { i: number; points: [number, number][]; bbox: Box }
export interface Placed {
  nodes: Map<string, Box>;
  groups: Map<string, Box>;
  edges: PlacedEdge[];
  width: number;
  height: number;
}

// ELK is ~1.5 MB; load it on first layout so the home page stays light.
const bundled = () => import("elkjs/lib/elk.bundled.js").then((m) => new m.default());

async function workerElk(): Promise<ElkApi & { broken: Promise<never> }> {
  const [{ default: Elk }, { default: workerUrl }] = await Promise.all([
    import("elkjs/lib/elk-api.js"), import("elkjs/lib/elk-worker.min.js?url"),
  ]);
  let fail!: (e: unknown) => void;
  const broken = new Promise<never>((_, rej) => { fail = rej; });
  broken.catch(() => {});
  const elk = new Elk({
    workerUrl,
    workerFactory: (url) => {
      const w = new Worker(url!);
      w.addEventListener("error", (e) => fail(new Error(e.message || "ELK worker failed")));
      return w;
    },
  });
  return Object.assign(elk, { broken });
}

type Elk = ElkApi & { broken?: Promise<never> };
let elkPromise: Promise<Elk> | null = null;
const getElk = (): Promise<Elk> => (elkPromise ??= typeof Worker === "undefined" ? bundled() : workerElk().catch(bundled));

async function runElk(root: ElkNode): Promise<ElkNode> {
  const elk = await getElk();
  if (!elk.broken) return elk.layout(root);
  try {
    return await Promise.race([elk.layout(root), elk.broken]);
  } catch {
    elk.terminateWorker();
    elkPromise = bundled();
    return (await elkPromise).layout(root);
  }
}

/** Content-derived key: same nodes, groups and edges → same layout, whatever the scores or colours. */
export function layoutKey(view: GraphView): string {
  const nodes = view.nodes.map((n) => `${n.id}@${n.group ?? ""}`).sort();
  const groups = view.groups.map((g) => `${g.id}@${g.parent ?? ""}`).sort();
  const edges = view.edges.map((e) => `${e.source}>${e.target}`);
  return `${nodes.join("|")}#${groups.join("|")}#${edges.join("|")}`;
}

const CACHE_MAX = 24;
const cache = new Map<string, Promise<Placed>>();

export function layoutGraph(view: GraphView): Promise<Placed> {
  const key = layoutKey(view);
  let p = cache.get(key);
  if (!p) {
    p = computeLayout(view);
    cache.set(key, p);
    if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value!);
    p.catch(() => cache.delete(key));
  }
  return p;
}

const ROOT_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.hierarchyHandling": "INCLUDE_CHILDREN",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.spacing.nodeNodeBetweenLayers": "28",
  "elk.spacing.nodeNode": "24",
  "elk.spacing.edgeNode": "14",
  "elk.layered.spacing.edgeNodeBetweenLayers": "12",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.padding": "[top=24,left=24,bottom=24,right=24]",
};

const GROUP_OPTIONS: Record<string, string> = {
  "elk.padding": "[top=34,left=18,bottom=18,right=18]",
  "elk.direction": "DOWN",
};

async function computeLayout(view: GraphView): Promise<Placed> {
  const root: ElkNode = { id: "__root", children: [], edges: [], layoutOptions: ROOT_OPTIONS };
  const groups = new Map<string, ElkNode>();
  const groupParent = new Map<string, string | null>();
  for (const g of view.groups) groups.set(g.id, { id: g.id, children: [], edges: [], layoutOptions: GROUP_OPTIONS });
  for (const g of view.groups) groupParent.set(g.id, g.parent && groups.has(g.parent) ? g.parent : null);
  for (const g of view.groups) {
    const parent = groupParent.get(g.id);
    (parent ? groups.get(parent)! : root).children!.push(groups.get(g.id)!);
  }
  const nodeGroup = new Map<string, string | null>();
  for (const n of view.nodes) {
    const gid = n.group && groups.has(n.group) ? n.group : null;
    nodeGroup.set(n.id, gid);
    (gid ? groups.get(gid)! : root).children!.push({ id: n.id, width: NODE_W, height: NODE_H });
  }

  // chain of containers from the root down to a node's group
  const chain = (nodeId: string): string[] => {
    const out: string[] = [];
    let g = nodeGroup.get(nodeId) ?? null;
    while (g) { out.unshift(g); g = groupParent.get(g) ?? null; }
    return ["__root", ...out];
  };

  view.edges.forEach((e, i) => {
    const a = chain(e.source), b = chain(e.target);
    let lca = "__root";
    for (let k = 0; k < Math.min(a.length, b.length) && a[k] === b[k]; k++) lca = a[k];
    const container = lca === "__root" ? root : groups.get(lca)!;
    container.edges!.push({ id: `e${i}`, sources: [e.source], targets: [e.target] } as ElkExtendedEdge);
  });

  const res = await runElk(root);

  const placed: Placed = { nodes: new Map(), groups: new Map(), edges: [], width: res.width ?? 0, height: res.height ?? 0 };
  const walk = (n: ElkNode, ox: number, oy: number) => {
    for (const e of (n.edges ?? []) as ElkExtendedEdge[]) {
      const i = Number(e.id.slice(1));
      for (const s of e.sections ?? []) {
        const pts: [number, number][] = [[s.startPoint.x + ox, s.startPoint.y + oy]];
        for (const b of s.bendPoints ?? []) pts.push([b.x + ox, b.y + oy]);
        pts.push([s.endPoint.x + ox, s.endPoint.y + oy]);
        placed.edges.push({ i, points: pts, bbox: bounds(pts) });
      }
    }
    for (const c of n.children ?? []) {
      const x = ox + (c.x ?? 0), y = oy + (c.y ?? 0);
      const box = { x, y, w: c.width ?? 0, h: c.height ?? 0 };
      if (groups.has(c.id)) {
        placed.groups.set(c.id, box);
        walk(c, x, y);
      } else {
        placed.nodes.set(c.id, box);
      }
    }
  };
  walk(res, 0, 0);
  return placed;
}

export function bounds(pts: [number, number][]): Box {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}
