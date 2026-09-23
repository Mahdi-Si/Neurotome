// Graph layout with ELK (layered, top-down, orthogonal edges, nested module groups).
import type { ELK as ElkApi, ElkExtendedEdge, ElkNode } from "elkjs/lib/elk-api";
import type { GraphEdge, GraphView } from "./types";

export const NODE_W = 196;
export const NODE_H = 46;

export interface Box { x: number; y: number; w: number; h: number }
export interface PlacedEdge { edge: GraphEdge; points: [number, number][] }
export interface Placed {
  nodes: Map<string, Box>;
  groups: Map<string, Box>;
  edges: PlacedEdge[];
  width: number;
  height: number;
}

// ELK is ~1.5 MB; load it on first layout so the home page stays light.
let elkPromise: Promise<ElkApi> | null = null;
const getElk = () => (elkPromise ??= import("elkjs/lib/elk.bundled.js").then((m) => new m.default()));

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

export async function layoutGraph(view: GraphView): Promise<Placed> {
  const root: ElkNode = { id: "__root", children: [], edges: [], layoutOptions: ROOT_OPTIONS };
  const groups = new Map<string, ElkNode>();
  const groupParent = new Map<string, string | null>();
  for (const g of view.groups) {
    groups.set(g.id, { id: g.id, children: [], edges: [], layoutOptions: GROUP_OPTIONS });
    groupParent.set(g.id, g.parent && view.groups.some((x) => x.id === g.parent) ? g.parent : null);
  }
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

  const byEdgeId = new Map<string, GraphEdge>();
  view.edges.forEach((e, i) => {
    const a = chain(e.source), b = chain(e.target);
    let lca = "__root";
    for (let k = 0; k < Math.min(a.length, b.length) && a[k] === b[k]; k++) lca = a[k];
    const container = lca === "__root" ? root : groups.get(lca)!;
    const id = `e${i}`;
    byEdgeId.set(id, e);
    container.edges!.push({ id, sources: [e.source], targets: [e.target] } as ElkExtendedEdge);
  });

  const res = await (await getElk()).layout(root);

  const placed: Placed = { nodes: new Map(), groups: new Map(), edges: [], width: res.width ?? 0, height: res.height ?? 0 };
  const walk = (n: ElkNode, ox: number, oy: number) => {
    for (const e of (n.edges ?? []) as ElkExtendedEdge[]) {
      const edge = byEdgeId.get(e.id);
      if (!edge) continue;
      for (const s of e.sections ?? []) {
        const pts: [number, number][] = [[s.startPoint.x + ox, s.startPoint.y + oy]];
        for (const b of s.bendPoints ?? []) pts.push([b.x + ox, b.y + oy]);
        pts.push([s.endPoint.x + ox, s.endPoint.y + oy]);
        placed.edges.push({ edge, points: pts });
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
