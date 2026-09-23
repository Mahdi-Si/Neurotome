// Canvas-engine benchmark: layout time, cache hits and per-frame culling cost on a 2000-node graph.
import { describe, expect, it } from "vitest";
import { layoutGraph } from "./layout";
import type { GraphEdge, GraphGroup, GraphNode, GraphView } from "./types";
import { edgesInView, visibleNodes } from "./viewport";

/** `n` op nodes in blocks of 8: a chain per block, a residual from each block's input to its last node, blocks chained. */
export function syntheticView(n: number): GraphView {
  const nodes: GraphNode[] = [], edges: GraphEdge[] = [], groups: GraphGroup[] = [];
  const blocks = Math.ceil(n / 8);
  const e = (source: string, target: string) => edges.push({ source, target, shape: [1, 64, 56, 56] });
  for (let b = 0; b < blocks; b++) {
    const gid = `g:block${b}:1`;
    groups.push({ id: gid, address: `block${b}`, class_name: "Block", num_params: 0, parent: null, num_ops: 8 });
    for (let j = 0; j < 8; j++) {
      const id = `op_${b}_${j}`;
      nodes.push({
        id, kind: "op", label: `op${j}`, path: `block${b}.op${j}`, type: "conv2d", num_ops: 1, num_params: 0,
        output_op: id, shape: [1, 64, 56, 56], flops: 0, stats: null, group: gid,
        is_input: b === 0 && j === 0, is_output: b === blocks - 1 && j === 7,
      });
      if (j) e(`op_${b}_${j - 1}`, id);
    }
    e(`op_${b}_0`, `op_${b}_7`);
    if (b) e(`op_${b - 1}_7`, `op_${b}_0`);
  }
  return { depth: 1, nodes, edges, groups };
}

function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

describe("canvas engine performance", { timeout: 30_000 }, () => {
  it("lays out, caches and culls 2000 nodes within budget", async () => {
    let t = performance.now();
    const placed = await layoutGraph(syntheticView(2000));
    const layoutMs = performance.now() - t;
    expect(placed.nodes.size).toBe(2000);
    expect(layoutMs).toBeLessThan(6000);

    t = performance.now();
    const again = await layoutGraph(syntheticView(2000));
    const hitMs = performance.now() - t;
    expect(again).toBe(placed);
    expect(hitMs).toBeLessThan(20);

    const W = 1600, H = 900, rand = rng(42);
    const tfs = Array.from({ length: 500 }, () => {
      const k = 0.05 + rand() * 2.45;
      return { k, x: W / 2 - rand() * placed.width * k, y: H / 2 - rand() * placed.height * k };
    });
    t = performance.now();
    for (const tf of tfs) { visibleNodes(placed, tf, W, H, 200); edgesInView(placed, tf, W, H); }
    const frameMs = (performance.now() - t) / tfs.length;
    expect(frameMs).toBeLessThan(1);

    const most = (k: number) => Math.max(...[...placed.nodes.values()].map((b) =>
      visibleNodes(placed, { k, x: W / 2 - b.x * k, y: H / 2 - b.y * k }, W, H, 200).length));
    const at1 = most(1), at04 = most(0.4);
    expect(at1).toBeLessThanOrEqual(150);
    expect(at04).toBeLessThanOrEqual(400);

    console.info(
      `perf · layout ${layoutMs.toFixed(0)} ms · cache hit ${hitMs.toFixed(2)} ms · cull ${frameMs.toFixed(3)} ms/frame`
      + ` · max DOM nodes ${at1} @k=1, ${at04} @k=0.4 · ${placed.width.toFixed(0)}×${placed.height.toFixed(0)}`,
    );
  });
});
