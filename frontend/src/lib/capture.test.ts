import { describe, expect, it } from "vitest";
import { captureNote, hasTensor, parseBudget, parseModules, unsavedText } from "./capture";
import type { GraphNode, RunMeta, Stats } from "./types";

const meta = (model: Partial<RunMeta["model"]>): RunMeta => ({
  id: "run_0001", created: 0, sample: { kind: "random", name: "random" }, input_shape: [1, 3, 8, 8], output: [],
  categories: null, reloadable: true, status: "ready",
  model: { spec: null, weights: null, class_name: "Net", num_params: 0, num_ops: 1210, flops: 0, activation_bytes: 0, truncated: false, ...model },
});
const stats: Stats = { mean: 0, std: 0, min: 0, max: 0, zero_frac: 0, hist: [], hist_range: [0, 0], numel: 4 };
const node = (o: Partial<GraphNode>): GraphNode => ({
  id: "a", kind: "op", label: "a", path: "a", type: "relu", num_ops: 1, num_params: 0, output_op: "a", shape: [1, 4],
  flops: 0, stats, group: null, is_input: false, is_output: false, ...o,
});

describe("capture form", () => {
  it("parses module lists", () => {
    expect(parseModules("layer3, layer4.1  layer3")).toEqual(["layer3", "layer4.1"]);
    expect(parseModules(" a,b ,, c\n")).toEqual(["a", "b", "c"]);
    expect(parseModules("")).toBeNull();
    expect(parseModules(" , ")).toBeNull();
  });
  it("parses the budget", () => {
    expect(parseBudget("")).toBeNull();
    expect(parseBudget("  ")).toBeNull();
    expect(parseBudget("512")).toBe(512);
    expect(parseBudget("0.5")).toBe(0.5);
    for (const bad of ["0", "-1", "x", "Infinity"]) expect(parseBudget(bad)).toBeNaN();
  });
});

describe("partial captures", () => {
  it("names why a node has no tensor", () => {
    expect(unsavedText("budget")).toBe("capture budget");
    expect(unsavedText("modules")).toBe("outside the captured modules");
    expect(unsavedText("no_tensor")).toBe("no tensor output");
    expect(unsavedText(null)).toBe("");
    expect(unsavedText(undefined)).toBe("");
  });
  it("knows which nodes have a tensor", () => {
    expect(hasTensor(node({}))).toBe(true);
    expect(hasTensor(node({ unsaved: null }))).toBe(true);
    expect(hasTensor(node({ unsaved: "budget" }))).toBe(false);   // stats alone do not mean a saved tensor
    expect(hasTensor(node({ stats: null }))).toBe(false);          // old server: no stats → no tensor
  });
  it("writes the banner", () => {
    const G = 2 * 1024 ** 3;
    expect(captureNote(meta({ truncated: true, budget_bytes: G, saved_ops: 412 }))).toEqual(
      { tag: "Truncated", text: "Capture budget 2.0 GB reached · 412/1210 ops saved · ∅ nodes have no tensor" });
    expect(captureNote(meta({ truncated: true, budget_bytes: G, saved_ops: 100, capture_modules: ["layer3", "layer4"] }))).toEqual(
      { tag: "Truncated", text: "Capture budget 2.0 GB reached · 100/1210 ops saved · ∅ nodes have no tensor · modules layer3, layer4" });
    expect(captureNote(meta({ truncated: true }))).toEqual({ tag: "Truncated", text: "Capture budget reached · some ops have no tensor" });
    expect(captureNote(meta({ capture_modules: ["layer3", "layer4"], saved_ops: 212, budget_bytes: G }))).toEqual(
      { tag: "Selective", text: "Tensors saved inside layer3, layer4 · 212/1210 ops saved · ∅ nodes have no tensor" });
    expect(captureNote(meta({ saved_ops: 1210, budget_bytes: G, capture_modules: null }))).toBeNull();
    expect(captureNote(meta({}))).toBeNull();
  });
});
