import { describe, expect, it } from "vitest";
import { dtypeShort } from "./format";
import {
  callAddress, colorMode, foldParams, moduleEntry, nodeTones, parseFold, revealFold, searchIndex, toggleFold, withinGroup,
} from "./graphState";
import type { GraphNode, GraphView, IndexEntry, InterventionResult, Stats } from "./types";

const st = (o: Partial<Stats>): Stats =>
  ({ mean: 0, std: 0, min: 0, max: 0, zero_frac: 0, hist: [], hist_range: [0, 0], numel: 10, ...o });
const node = (id: string, o: Partial<GraphNode> = {}): GraphNode => ({
  id, kind: "op", label: id, path: id, type: "relu", num_ops: 1, num_params: 0, output_op: id, shape: [1, 4],
  flops: 0, stats: null, group: null, is_input: false, is_output: false, ...o,
});
const view = (nodes: GraphNode[]): GraphView => ({ depth: 2, nodes, edges: [], groups: [] });
const entry = (id: string, o: Partial<IndexEntry> = {}): IndexEntry =>
  ({ id, kind: "op", label: id, path: id, type: "relu", shape: [1, 4], calls: [], ...o });

describe("fold state", () => {
  it("toggles without duplicates, open then close gives collapse only", () => {
    let f = toggleFold({ expand: [], collapse: ["a:1"] }, "a:1", true);
    expect(f).toEqual({ expand: ["a:1"], collapse: [] });
    expect(toggleFold(f, "a:1", true)).toEqual(f);
    f = toggleFold(f, "a:1", false);
    expect(f).toEqual({ expand: [], collapse: ["a:1"] });
    expect(toggleFold(f, "a:1", false)).toEqual(f);
  });
  it("round-trips through URL params", () => {
    const f = { expand: ["layer1.0:1", "layer2:1"], collapse: ["layer3:1"] };
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(foldParams(f))) if (v) q.set(k, v);
    expect(parseFold(new URLSearchParams(q.toString()))).toEqual(f);
    expect(foldParams({ expand: [], collapse: [] })).toEqual({ expand: null, collapse: null });
    expect(parseFold(new URLSearchParams(""))).toEqual({ expand: [], collapse: [] });
  });
  it("reveals an entry: expands its calls and removes them from collapse", () => {
    const f = revealFold({ expand: ["x:1"], collapse: ["layer1:1", "z:1"] }, entry("conv", { calls: ["layer1:1", "layer1.0:1"] }));
    expect(f).toEqual({ expand: ["x:1", "layer1:1", "layer1.0:1"], collapse: ["z:1"] });
  });
  it("reveals a module entry as a folded node, even when it was expanded", () => {
    const m = entry("m:layer1.0:1", { kind: "module", calls: ["layer1:1"] });
    expect(revealFold({ expand: ["layer1.0:1"], collapse: ["layer1:1"] }, m)).toEqual({ expand: ["layer1:1"], collapse: ["layer1.0:1"] });
  });
  it("finds the module entry for a tree address, preferring it over an op with the same path", () => {
    const op = entry("linear_1", { path: "fc" });
    const m = entry("m:layer1.0:1", { kind: "module", path: "layer1.0", calls: ["layer1:1"] });
    const fc = entry("m:fc:1", { kind: "module", path: "fc" });
    expect(moduleEntry([op, m, fc], "fc")).toBe(fc);
    expect(moduleEntry([op, m], "fc")).toBe(op);
    expect(moduleEntry([op, m], "nope")).toBeNull();
    expect(revealFold({ expand: ["layer1.0:1"], collapse: [] }, moduleEntry([op, m], "layer1.0")!).collapse).toContain("layer1.0:1");
  });
  it("resolves addresses and nested groups", () => {
    expect(callAddress("m:layer1.0:1")).toBe("layer1.0");
    const v: GraphView = { ...view([]), groups: [
      { id: "g:a:1", address: "a", class_name: "", num_params: 0, parent: null, num_ops: 2 },
      { id: "g:a.b:1", address: "a.b", class_name: "", num_params: 0, parent: "g:a:1", num_ops: 2 },
    ] };
    expect(withinGroup(v, "g:a.b:1", "g:a:1")).toBe(true);
    expect(withinGroup(v, "g:a:1", "g:a.b:1")).toBe(false);
    expect(withinGroup(v, null, "g:a:1")).toBe(false);
  });
});

describe("searchIndex", () => {
  const entries = [
    entry("conv2d_1_1", { label: "conv1", path: "layer1.0.conv1", type: "conv2d", shape: [1, 64, 56, 56], calls: ["layer1:1", "layer1.0:1"] }),
    entry("relu_2_2", { label: "relu", path: "layer1.0.relu", type: "relu", shape: [1, 64, 56, 56] }),
    entry("linear_9_9", { label: "fc", path: "fc", type: "linear", shape: [1, 10] }),
    entry("m:layer1.0:1", { kind: "module", label: "0", path: "layer1.0", type: "BasicBlock", shape: [1, 64, 56, 56] }),
    entry("m:layer1:1", { kind: "module", label: "layer1", path: "layer1", type: "Sequential", shape: [1, 64, 56, 56] }),
  ];
  it("ranks a module by name before path-only op matches", () => {
    const r = searchIndex(entries, "layer1").map((e) => e.id);
    expect(r[0]).toBe("m:layer1:1");
    expect(r.indexOf("m:layer1.0:1")).toBeLessThan(r.indexOf("conv2d_1_1"));
    expect(r).toContain("relu_2_2");
  });
  it("finds by op type, shape and multi-token AND", () => {
    expect(searchIndex(entries, "Linear").map((e) => e.id)).toEqual(["linear_9_9"]);
    expect(searchIndex(entries, "64x56x56")).toHaveLength(4);
    expect(searchIndex(entries, "64,56,56")).toHaveLength(4);
    expect(searchIndex(entries, "1x64x56x56")).toHaveLength(4);
    expect(searchIndex(entries, "layer1.0 conv").map((e) => e.id)).toEqual(["conv2d_1_1"]);
    expect(searchIndex(entries, "fc 64")).toEqual([]);
  });
  it("puts exact and prefix label matches first and caps results", () => {
    expect(searchIndex(entries, "conv")[0].id).toBe("conv2d_1_1");
    const many = Array.from({ length: 20 }, (_, i) => entry(`relu_${i}`));
    expect(searchIndex(many, "relu")).toHaveLength(8);
    expect(searchIndex(many, "relu", 3)).toHaveLength(3);
    expect(searchIndex(many, "   ")).toEqual([]);
  });
});

describe("colour modes", () => {
  it("parses color with the legacy overlay", () => {
    expect(colorMode(new URLSearchParams("color=norm"))).toBe("norm");
    expect(colorMode(new URLSearchParams("overlay=attr"))).toBe("attr");
    expect(colorMode(new URLSearchParams("color=bogus"))).toBe("plain");
    expect(colorMode(new URLSearchParams("color=iv"))).toBe("iv");
    expect(colorMode(new URLSearchParams("color=patch"))).toBe("iv");
  });
  it("plain has no tones or legend", () => {
    expect(nodeTones(view([node("a")]), "plain")).toEqual({ legend: null });
  });
  it("attr passes score_norm through and names the method", () => {
    const { tones, legend } = nodeTones(view([node("a", { score: 0.5, score_norm: 0.25 }), node("b")]), "attr");
    expect(tones!.get("a")).toEqual({ v: 0.25, readout: "|g×a| 0.500" });
    expect(tones!.get("b")!.v).toBeNull();
    expect(legend!.title).toContain("gxa");
  });
  it("norm derives RMS from mean and std, max node gets v = 1", () => {
    const v = view([node("a", { stats: st({ mean: 3, std: 4 }) }), node("b", { stats: st({ mean: 0, std: 2.5 }) }), node("c")]);
    const { tones, legend } = nodeTones(v, "norm");
    expect(tones!.get("a")).toEqual({ v: 1, readout: "RMS 5.00" });
    expect(tones!.get("b")!.v).toBeCloseTo(0.5);
    expect(tones!.get("c")!.v).toBeNull();
    expect(legend!.hi).toBe("5.00");
  });
  it("sparsity equals zero_frac", () => {
    const { tones } = nodeTones(view([node("a", { stats: st({ zero_frac: 0.42 }) })]), "sparsity");
    expect(tones!.get("a")).toEqual({ v: 0.42, readout: "Zero 42%" });
  });
  it("nan flags nonfinite > 0 and all-non-finite stats", () => {
    const v = view([
      node("a", { nonfinite: 3 }), node("b", { nonfinite: 0 }),
      node("c", { nonfinite: 0, stats: st({ mean: null, std: null }) }),
    ]);
    const { tones, legend } = nodeTones(v, "nan");
    expect(tones!.get("a")).toEqual({ v: null, flag: true, readout: "NaN/Inf 3" });
    expect(tones!.get("b")!.flag).toBe(false);
    expect(tones!.get("c")!.flag).toBe(true);
    expect(legend).toMatchObject({ hi: "2 flagged" });
    expect(legend!.note).toBeUndefined();
  });
  it("old runs without nonfinite get the not-recorded note", () => {
    const { tones, legend } = nodeTones(view([node("a", { stats: st({}) }), node("b")]), "nan");
    expect(tones!.get("a")).toEqual({ v: null, flag: false, readout: "NaN/Inf —" });
    expect(legend!.note).toBe("Not recorded for this run · re-capture to check");
  });
  it("strips the torch prefix from dtypes", () => {
    expect(dtypeShort("torch.float32")).toBe("float32");
    expect(dtypeShort(null)).toBe("—");
  });
});

describe("intervention Δ", () => {
  const row = (l2: number, rel: number, cos: number | null = 0.9) => ({ l2, rel, cos, max_abs: l2 / 2 });
  const result = (o: Partial<InterventionResult> = {}): InterventionResult => ({
    hash: "3f9a0c1d2e4b", run_id: "run_0001", name: "Zero · layer1.conv · all (+1)", created: 0,
    set: { edits: [{ op: "b", kind: "zero", channels: null, positions: null }, { op: "b", kind: "scale", channels: null, positions: null, alpha: 2 },
      { op: "x", kind: "zero", channels: null, positions: null }], target: null },
    labels: ["Zero · layer1.conv · all", "Scale ×2 · layer1.conv · all", "Zero · x · all"],
    method_label: "Intervention · TorchLens capture-time hooks (every op)", target: 3, target_label: "cat",
    before: { logit: 2, prob: 0.8 }, after: { logit: 1, prob: 0.4 }, delta_logit: -1, delta_prob: -0.4, top_before: [], top_after: [],
    edited: ["b", "x"], downstream: ["b", "c", "d", "e", "x"], saved: ["b", "c"], truncated: false, budget_bytes: null, bytes: 0,
    nodes: { b: row(9, 1), c: row(2, 0.5, null), d: row(1, 0.25) }, stats: {}, ...o,
  });
  it("edited ops v 1 with their labels, normalised by the max rel outside them", () => {
    const v = view([node("a"), node("m:b", { kind: "module", output_op: "b" }), node("c"), node("d"), node("e")]);
    const { tones, legend } = nodeTones(v, "iv", result());
    expect(tones!.get("m:b")).toEqual({ v: 1, readout: "edited · Zero · layer1.conv · all · Scale ×2 · layer1.conv · all" });
    expect(tones!.get("c")).toEqual({ v: 1, readout: "‖Δ‖ 2.00 · rel 50% · cos — · max|Δ| 1.00" });
    expect(tones!.get("d")!.v).toBeCloseTo(0.5);
    expect(tones!.get("d")!.readout).toContain("cos 0.900");
    expect(tones!.get("a")).toEqual({ v: 0, readout: "unchanged · not downstream" });
    expect(tones!.get("e")).toEqual({ v: null, readout: "‖Δ‖ —" });
    expect(legend).toEqual({ title: "Intervention Δ = ‖after − before‖ / ‖before‖ · Zero · layer1.conv · all (+1)", lo: "0", hi: "50%" });
  });
  it("no downstream change gives v 0, a missing result gives no legend", () => {
    const { tones } = nodeTones(view([node("c")]), "iv", result({ nodes: { c: row(0, 0) } }));
    expect(tones!.get("c")!.v).toBe(0);
    expect(nodeTones(view([node("a")]), "iv", null)).toEqual({ legend: null });
    expect(nodeTones(view([node("a")]), "iv")).toEqual({ legend: null });
  });
  it("values that became NaN / inf are a maximal Δ even when the finite part is 0", () => {
    const { tones } = nodeTones(view([node("c"), node("d")]), "iv", result({ nodes: { c: { ...row(0, 0), nonfinite: 12 }, d: row(1, 0.25) } }));
    expect(tones!.get("c")).toEqual({ v: 1, readout: "12 values changed to or from NaN/inf · finite: ‖Δ‖ 0.00 · rel 0% · cos 0.900 · max|Δ| 0.00" });
    expect(tones!.get("d")!.v).toBe(1);
  });
});
