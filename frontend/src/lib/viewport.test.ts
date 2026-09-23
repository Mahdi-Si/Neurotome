import { describe, expect, it } from "vitest";
import { bounds, type Placed } from "./layout";
import type { GraphNode, GraphView } from "./types";
import { edgeReadout, edgesInView, hitEdge, hitNode, mix, step, toGraph, toScreen, visibleNodes } from "./viewport";

const node = (id: string, extra: Partial<GraphNode> = {}): GraphNode => ({
  id, kind: "op", label: id, path: id, type: "relu", num_ops: 1, num_params: 0, output_op: id, shape: [1, 4],
  flops: 0, stats: null, group: null, is_input: false, is_output: false, ...extra,
});

const box = (x: number, y: number) => ({ x, y, w: 40, h: 20 });
const placedEdge = (i: number, points: [number, number][]) => ({ i, points, bbox: bounds(points) });

// in → a, in → b, a → out, b → out; a left of b, out slightly right of centre (nearer b)
const diamond: GraphView = {
  depth: 0,
  nodes: [node("in", { is_input: true }), node("a"), node("b"), node("out", { is_output: true })],
  edges: [["in", "a"], ["in", "b"], ["a", "out"], ["b", "out"]].map(([source, target]) => ({ source, target, shape: [1, 4] })),
  groups: [],
};
const placed: Placed = {
  nodes: new Map([["in", box(50, 0)], ["a", box(0, 50)], ["b", box(100, 50)], ["out", box(70, 100)]]),
  groups: new Map(),
  edges: [],
  width: 140, height: 120,
};

describe("step", () => {
  it("follows edges on a diamond", () => {
    expect(step(diamond, placed, "in", "ArrowDown")).toBe("a");
    expect(step(diamond, placed, "a", "ArrowRight")).toBe("b");
    expect(step(diamond, placed, "b", "ArrowLeft")).toBe("a");
    expect(step(diamond, placed, "out", "ArrowUp")).toBe("b");
    expect(step(diamond, placed, "a", "ArrowLeft")).toBeNull();
    expect(step(diamond, placed, "out", "ArrowDown")).toBeNull();
  });
  it("selects the input when nothing is selected", () => {
    expect(step(diamond, placed, null, "ArrowDown")).toBe("in");
    expect(step(diamond, placed, null, "Enter")).toBeNull();
  });
});

describe("hit testing", () => {
  const two: Placed = {
    ...placed,
    edges: [placedEdge(0, [[0, 0], [100, 0]]), placedEdge(1, [[0, 8], [100, 8]])],
  };
  it("hits a horizontal segment within tolerance", () => {
    expect(hitEdge(two, 50, -3, 5)).toBe(0);
    expect(hitEdge(two, 50, -10, 5)).toBeNull();
  });
  it("picks the nearer of two edges", () => {
    expect(hitEdge(two, 50, 5, 5)).toBe(1);
    expect(hitEdge(two, 50, 3, 5)).toBe(0);
  });
  it("finds nodes by point", () => {
    expect(hitNode(placed, 10, 60)).toBe("a");
    expect(hitNode(placed, 60, 60)).toBeNull();
  });
});

describe("culling", () => {
  it("returns only boxes in the viewport", () => {
    const tf = { x: 0, y: 0, k: 1 };
    expect(visibleNodes(placed, tf, 100, 45)).toEqual(["in"]);
    expect(visibleNodes(placed, tf, 45, 45)).toEqual([]);
    expect(visibleNodes(placed, tf, 100, 45, 10).sort()).toEqual(["a", "b", "in"]);
    expect(visibleNodes(placed, { x: 0, y: 0, k: 0.5 }, 100, 60).sort()).toEqual(["a", "b", "in", "out"]);
    expect(visibleNodes(placed, { x: -1000, y: 0, k: 1 }, 100, 100)).toEqual([]);
    const withEdge: Placed = { ...placed, edges: [placedEdge(0, [[500, 0], [500, 50]])] };
    expect(edgesInView(withEdge, tf, 100, 100)).toEqual([]);
    expect(edgesInView(withEdge, { x: -450, y: 0, k: 1 }, 100, 100)).toEqual([0]);
  });
  it("converts between screen and graph space", () => {
    const tf = { x: 10, y: -20, k: 2 };
    expect(toGraph(tf, ...toScreen(tf, 3, 4))).toEqual([3, 4]);
  });
});

describe("mix and readout", () => {
  it("lerps hex colours", () => {
    expect(mix("#000000", "#ffffff", 0.5)).toBe("#808080");
    expect(mix("#fafaf8", "#3b4bdb", 0)).toBe("#fafaf8");
    expect(mix("#fafaf8", "#3b4bdb", 1)).toBe("#3b4bdb");
  });
  it("mixes rgb() tokens as read from the DOM", () => {
    expect(mix("rgb(0, 0, 0)", "rgb(255, 255, 255)", 0.5)).toBe("#808080");
    expect(mix("rgb(28, 28, 26)", "#8b97ff", 0)).toBe("#1c1c1a");
    expect(mix("rgba(250, 250, 248, 1)", "rgb(59, 75, 219)", 1)).toBe("#3b4bdb");
  });
  it("describes an edge and flags dtype/device changes", () => {
    const e = { source: "a", target: "b", shape: [1, 64, 56, 56], dtype: "torch.float32", device: "cuda:0", target_dtype: "torch.float16", target_device: "cuda:0" };
    expect(edgeReadout(e, node("a"), node("b"))).toEqual([
      { t: "a → b" }, { t: "64×56×56" }, { t: "float32 → float16", hi: true }, { t: "cuda:0" },
    ]);
    expect(edgeReadout({ source: "a", target: "b", shape: [1, 4] }, node("a"), undefined).map((s) => s.t)).toEqual(["a → b", "4"]);
  });
});
