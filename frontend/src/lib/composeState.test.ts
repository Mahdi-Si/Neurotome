import { describe, expect, it } from "vitest";
import { cellsOf, divergeIndex, editable, genReadout, moduleSites, sameModelRuns, toggleCells, topReadout } from "./composeState";
import type { GenerationResult, GraphNode, GraphView, RunMeta, TopK } from "./types";

describe("pixel selection on pooled cells", () => {
  it("adds every pixel of a cell, then removes them", () => {
    const a = toggleCells([], 4, 4, 2, 2, 1, 0);
    expect(a).toEqual([[2, 0], [2, 1], [3, 0], [3, 1]]);
    expect(toggleCells(a, 4, 4, 2, 2, 1, 0)).toEqual([]);
  });
  it("completes a partly selected cell and keeps the order sorted", () => {
    expect(toggleCells([[3, 3], [0, 0]], 4, 4, 2, 2, 0, 0)).toEqual([[0, 0], [0, 1], [1, 0], [1, 1], [3, 3]]);
  });
  it("uses np.array_split spans on uneven pooling", () => {
    // 5 pixels in 2 cells: [0, 1, 2] and [3, 4]
    expect(toggleCells([], 5, 1, 2, 1, 0, 0)).toEqual([[0, 0], [1, 0], [2, 0]]);
    expect(toggleCells([], 5, 1, 2, 1, 1, 0)).toEqual([[3, 0], [4, 0]]);
  });
  it("maps selected pixels to the cells they touch", () => {
    expect(cellsOf([[0, 0], [1, 1], [4, 0], [3, 0]], 5, 2, 2, 1)).toEqual([[0, 0], [1, 0]]);
    expect(cellsOf([], 4, 4, 2, 2)).toEqual([]);
  });
});

describe("continuations", () => {
  it("finds where two continuations diverge", () => {
    expect(divergeIndex([1, 2, 3], [1, 2, 3])).toBeNull();
    expect(divergeIndex([1, 2, 3], [1, 5, 3])).toBe(1);
    expect(divergeIndex([1, 2], [1, 2, 3])).toBe(2);
    expect(divergeIndex([], [])).toBeNull();
  });
  it("prints the top-3 of both sides at a step", () => {
    const top = (s: string) => [{ id: 0, token: s, prob: 0.5 }, { id: 1, token: " ", prob: 0.25 }, { id: 2, token: "e", prob: 0.125 }, { id: 3, token: "x", prob: 0.1 }];
    const side = { ids: [0], tokens: ["a"], text: "a", top: [top("a")] };
    const res: GenerationResult = { hash: "0".repeat(12), n: 1, labels: [], method_label: "", prompt: [], clean: side,
      steered: { ...side, top: [top("c")] }, diverge: null };
    expect(genReadout(res, 0)).toBe('step 0 · clean "a" 0.50 · "·" 0.25 · "e" 0.13 | intervened "c" 0.50 · "·" 0.25 · "e" 0.13');
    expect(genReadout(res, 5)).toBe("step 5 · clean — | intervened —");
  });
});

const meta = (id: string, spec: string | null, weights: string | null = "DEFAULT"): RunMeta => ({
  id, created: 0, input_shape: null, output: [], categories: null, reloadable: true, status: "ready", sample: { kind: "random", name: id },
  model: { spec, weights, class_name: "M", num_params: 0, num_ops: 1, flops: 0, activation_bytes: 0, truncated: false },
});

describe("source runs", () => {
  it("keeps runs of the same spec and weights, this run first", () => {
    const me = meta("run_0002", "py:m:build");
    const runs = [meta("run_0001", "py:m:build"), me, meta("run_0003", "py:other:build"), meta("run_0004", "py:m:build", "RANDOM")];
    expect(sameModelRuns(runs, me).map((r) => r.id)).toEqual(["run_0002", "run_0001"]);
  });
  it("gives an in-process run (spec null) no source runs", () => {
    expect(sameModelRuns([meta("run_0001", null)], meta("run_0001", null))).toEqual([]);
  });
});

describe("editable nodes and top-k readout", () => {
  it("refuses input, output and non-float nodes", () => {
    expect(editable({ is_input: false, is_output: false, dtype: "torch.float32" })).toBe(true);
    expect(editable({ is_input: false, is_output: false, dtype: null })).toBe(true);
    expect(editable({ is_input: false, is_output: false, dtype: "torch.int64" })).toBe(false);
    expect(editable({ is_input: true, is_output: false, dtype: "torch.float32" })).toBe(false);
    expect(editable({ is_input: false, is_output: true, dtype: "torch.float32" })).toBe(false);
  });
  it("names the rank and prob on both sides", () => {
    const k = (index: number, prob: number, label: string | null = null): TopK => ({ index, prob, logit: 0, label });
    const before = [k(1, 0.5, "dog"), k(3, 0.12, "cat")], after = [k(3, 0.41, "cat"), k(1, 0.2, "dog")];
    expect(topReadout(before, after, 3)).toBe('before #2 "cat" 0.12 → after #1 0.41');
    expect(topReadout(before, [k(7, 0.9)], 7)).toBe('before — "[7]" — → after #1 0.90');
    expect(topReadout(before, [{ index: 3, prob: null, logit: null, label: "cat" }], 3)).toBe('before #2 "cat" 0.12 → after #1 —');
  });
});

describe("module sites (the server's edit label paths)", () => {
  // a BasicBlock at full depth: conv (leaf) → relu (leaf, call 1) → conv2 (leaf) → add → relu (leaf, call 2) → head (leaf)
  const n = (id: string, path: string, group: string | null): GraphNode => ({
    id, kind: "op", label: id, path, type: id, num_ops: 1, num_params: 0, output_op: id, shape: [1, 4], flops: 0,
    stats: null, group, is_input: false, is_output: false,
  });
  const g = "g:layer.0:1";
  const view: GraphView = {
    depth: 12, groups: [{ id: g, address: "layer.0", class_name: "BasicBlock", num_params: 0, parent: null, num_ops: 5 }],
    nodes: [n("conv_1", "layer.0.conv1", g), n("relu_1", "layer.0.relu", g), n("conv_2", "layer.0.conv2", g),
      n("add_1", "add_1", g), n("relu_2", "layer.0.relu", g), n("mean_1", "mean_1", null)],
    edges: [["conv_1", "relu_1"], ["relu_1", "conv_2"], ["conv_2", "add_1"], ["add_1", "relu_2"], ["relu_2", "mean_1"]]
      .map(([source, target]) => ({ source, target, shape: [1, 4] })),
  };
  it("names leaf calls (@n after the first), then the innermost group an op leaves", () => {
    const { path, op } = moduleSites(view);
    expect(path.get("relu_1")).toBe("layer.0.relu");
    expect(path.get("relu_2")).toBe("layer.0.relu@2");   // also the block's output: the leaf is innermost
    expect(path.get("add_1")).toBeUndefined();
    expect(path.get("mean_1")).toBeUndefined();
    expect(op.get("m:layer.0:1")).toBe("relu_2");
    expect(op.get("m:layer.0.relu:2")).toBe("relu_2");
    // without the leaf relu, the add is the block's output and takes the block's name
    const noRelu: GraphView = { ...view, nodes: view.nodes.filter((x) => x.id !== "relu_2"),
      edges: [...view.edges.filter((e) => e.target !== "relu_2" && e.source !== "relu_2"), { source: "add_1", target: "mean_1", shape: [1, 4] }] };
    expect(moduleSites(noRelu).path.get("add_1")).toBe("layer.0");
  });
});
