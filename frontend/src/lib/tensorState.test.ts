import { describe, expect, it } from "vitest";
import {
  cellOf, cellPixel, cellSpan, cellStart, defaultSlice, equalAspect, errText, inputHW, lineDomain, pageCount, parentNode, parseTensorParams, sameShape,
  sampleIndex, seqFactor, sharedRange, slicePage, tensorParams, unscale, type TensorState,
} from "./tensorState";
import type { GraphView, SliceResult, Stats } from "./types";

const q = (s: string) => new URLSearchParams(s);

describe("tensor URL state", () => {
  it("round-trips through URL params", () => {
    const s: TensorState = { node: "m:layer1.0:1", c: 3, py: 12, px: 40, norm: "global", cmap: "div", cmp: "relu_2_5", iv: "3f9a0c1d2e4b", vs: "clean",
      s: "[0, 28:41, 5:10]", t: 40 };
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(tensorParams(s))) if (v != null) p.set(k, v);
    expect(parseTensorParams(q(p.toString()))).toEqual(s);
  });
  it("omits defaults", () => {
    const s = parseTensorParams(q(""));
    expect(s).toEqual({ node: null, c: null, py: null, px: null, norm: "channel", cmap: "seq", cmp: null, iv: null, vs: null, s: null, t: null });
    expect(Object.values(tensorParams(s)).every((v) => v == null)).toBe(true);
  });
  it("falls back on bad values and drops a lone py/px", () => {
    const s = parseTensorParams(q("c=1.5&py=3&norm=weird&cmap=rainbow"));
    expect(s).toMatchObject({ c: null, py: null, px: null, norm: "channel", cmap: "seq" });
    expect(parseTensorParams(q("c=x&px=4")).px).toBeNull();
    expect(parseTensorParams(q("py=-1&px=2")).py).toBeNull();
  });
  it("keeps vs only with iv", () => {
    expect(parseTensorParams(q("vs=clean")).vs).toBeNull();
    expect(parseTensorParams(q("iv=abc&vs=other")).vs).toBeNull();
    expect(parseTensorParams(q("iv=abc&vs=clean"))).toMatchObject({ iv: "abc", vs: "clean" });
    expect(tensorParams({ ...parseTensorParams(q("")), vs: "clean" }).vs).toBeNull();
    expect(tensorParams(parseTensorParams(q("iv=abc")))).toMatchObject({ iv: "abc", vs: null });
  });
});

describe("pooled cells mirror np.array_split", () => {
  it("starts cells like numpy for 56 into 16", () => {
    const starts = Array.from({ length: 17 }, (_, i) => cellStart(56, 16, i));
    const sizes = starts.slice(1).map((s, i) => s - starts[i]);
    expect(sizes).toEqual([...Array(8).fill(4), ...Array(8).fill(3)]);
    expect(starts.slice(0, 10)).toEqual([0, 4, 8, 12, 16, 20, 24, 28, 32, 35]);
  });
  it("inverts cellPixel and clamps", () => {
    expect(cellOf(56, 16, 33)).toBe(8);
    for (let i = 0; i < 16; i++) expect(cellOf(56, 16, cellPixel(56, 16, i))).toBe(i);
    for (let p = 0; p < 56; p++) { const c = cellOf(56, 16, p); expect(cellStart(56, 16, c) <= p && p < cellStart(56, 16, c + 1)).toBe(true); }
    expect(cellOf(56, 16, -5)).toBe(0);
    expect(cellOf(56, 16, 99)).toBe(15);
    expect(cellSpan(56, 16, 0)).toBe("0–3");
  });
  it("is the identity when n == len", () => {
    for (let i = 0; i < 7; i++) {
      expect(cellStart(7, 7, i)).toBe(i);
      expect(cellOf(7, 7, i)).toBe(i);
      expect(cellPixel(7, 7, i)).toBe(i);
      expect(cellSpan(7, 7, i)).toBe(`${i}`);
    }
  });
  it("unscale round-trips both colormaps", () => {
    const seq = (x: number, lo: number, hi: number) => (x - lo) / (hi - lo);
    expect(unscale(seq(1.5, -2, 4), [-2, 4], "seq")).toBeCloseTo(1.5);
    expect(unscale(-1.5 / 4, [-2, 4], "div")).toBeCloseTo(-1.5);
    expect(unscale(0, [-2, 4], "div")).toBe(0);
  });
});

describe("PCA points and sequence scale", () => {
  it("maps a PCA point back to its row like np.linspace(0, n - 1, min(n, 400)).astype(int)", () => {
    // expected values from numpy
    expect([0, 1, 2, 137, 398, 399].map((i) => sampleIndex(12544, 400, i))).toEqual([0, 31, 62, 4306, 12511, 12543]);
    expect([0, 1, 2, 137, 398, 399].map((i) => sampleIndex(401, 400, i))).toEqual([0, 1, 2, 137, 398, 400]);
    expect([0, 1, 2, 137, 398, 399].map((i) => sampleIndex(1000, 400, i))).toEqual([0, 2, 5, 343, 996, 999]);
    for (let i = 0; i < 50; i++) expect(sampleIndex(50, 50, i)).toBe(i);
    expect(sampleIndex(1, 1, 0)).toBe(0);
  });
  it("scales both PCA axes by one factor, so the scatter keeps the projection's aspect", () => {
    const { px, py } = equalAspect([[0, 0], [10, 1]], 320, 180, 6);
    expect(px(10) - px(0)).toBeCloseTo(10 * (py(0) - py(1)));   // 1 unit of x = 1 unit of y on screen
    expect(px(0)).toBeCloseTo(6);
    expect(px(10)).toBeCloseTo(314);
    expect((py(0) + py(1)) / 2).toBeCloseTo(90);                  // centred vertically
    // the axis ranges are the data values at the chart's edges, y from the top
    const f = equalAspect([[0, 0], [10, 1]], 320, 180, 6);
    expect(f.px(f.xRange[0])).toBeCloseTo(0); expect(f.px(f.xRange[1])).toBeCloseTo(320);
    expect(f.py(f.yRange[0])).toBeCloseTo(0); expect(f.py(f.yRange[1])).toBeCloseTo(180);
    expect(f.xRange[1] - f.xRange[0]).toBeCloseTo((f.yRange[0] - f.yRange[1]) * 320 / 180);
    const one = equalAspect([[2, 3]], 320, 180, 6);
    expect([one.px(2), one.py(3)]).toEqual([160, 90]);
  });
  it("reads an input's (H, W) from its shape", () => {
    expect(inputHW([1, 3, 224, 200])).toEqual([224, 200]);
    expect(inputHW([3, 28, 28])).toEqual([28, 28]);
    expect(inputHW([1, 22])).toBeNull();
    expect(inputHW(null)).toBeNull();
  });
  it("gives a line plot a y domain with 0 in it, the fixed domain honoured, never empty", () => {
    expect(lineDomain([1, 5, 3])).toEqual([0, 5]);
    expect(lineDomain([-2, 1])).toEqual([-2, 1]);
    expect(lineDomain([0.04, 0.05], [0, 1])).toEqual([0, 1]);
    const [lo, hi] = lineDomain([0, 0]);
    expect(lo).toBe(0); expect(hi).toBeGreaterThan(0);
  });
  it("knows the sequence heatmap's divisor only when it is exact", () => {
    const t = { tokens: 6, features: 32, heatmap: Array(6).fill(Array(32).fill(0)), scale: null, stats: { mean: 0, std: 1, min: -2.5, max: 1.5 } };
    expect(seqFactor(t)).toBe(2.5);
    expect(seqFactor({ ...t, scale: [-4, 3] })).toBe(4);
    expect(seqFactor({ ...t, tokens: 100 })).toBeNull();   // pooled: the max |mean| of a block is unknown
  });
});

describe("values table", () => {
  it("builds default slices", () => {
    expect(defaultSlice([1, 64, 56, 56], { channel: 3, pixel: [2, 50] })).toBe("[0, 3, 0:16, 40:56]");
    expect(defaultSlice([1, 64, 56, 56], { channel: 3, pixel: [30, 20] })).toBe("[0, 3, 22:38, 12:28]");
    expect(defaultSlice([1, 4, 8, 8], { channel: 0, pixel: [7, 7] })).toBe("[0, 0, 0:8, 0:8]");
    expect(defaultSlice([1, 64, 56, 56], {})).toBe("[0, 0, 0:16, 0:16]");
    expect(defaultSlice([1, 197, 768], {})).toBe("[0, 0:16, 0:16]");
    expect(defaultSlice([1, 1000], {})).toBe("[0, 0:64]");
    expect(defaultSlice([10], {})).toBe("[0:10]");
  });
  const res = (o: Partial<SliceResult>): SliceResult => ({
    op: "x", expr: "", key: "", shape: [], full_shape: [], truncated: false, axes: [], indices: [], values: [], ...o,
  });
  it("pages over leading axes", () => {
    const r = res({ shape: [2, 2, 3], axes: [1, 2, 3], indices: [[5, 9], [0, 1], [10, 11, 12]], values: Array.from({ length: 12 }, (_, i) => i) });
    expect(pageCount(r)).toBe(2);
    const p = slicePage(r, 1)!;
    expect(p.label).toBe("[9]");
    expect(p.cells).toEqual([[6, 7, 8], [9, 10, 11]]);
    expect(p.rows).toEqual([0, 1]);
    expect(p.cols).toEqual([10, 11, 12]);
    const r4 = res({ shape: [2, 3, 1, 1], indices: [[0, 1], [4, 5, 6], [0], [0]], values: [0, 1, 2, 3, 4, "nan"] });
    expect(slicePage(r4, 4)!.label).toBe("[1, 5]");
    expect(slicePage(r4, 5)!.cells).toEqual([["nan"]]);
    expect(slicePage(res({ shape: [3], indices: [[0, 1, 2]], values: [1, 2, 3] }), 0)).toBeNull();
    expect(slicePage(res({ shape: [0, 2, 2], indices: [[], [0, 1], [0, 1]], values: [] }), -1)).toBeNull();
  });
  it("strips the status prefix from server errors", () => {
    expect(errText(new Error("400 · Index 70 is out of range for axis 1 (size 64)."))).toBe("Index 70 is out of range for axis 1 (size 64).");
    expect(errText(new Error("network down"))).toBe("network down");
  });
});

describe("compare", () => {
  const st = (min: number | null, max: number | null): Stats =>
    ({ mean: 0, std: 0, min, max, zero_frac: 0, hist: [], hist_range: [0, 0], numel: 1 });
  it("finds the parent through the first incoming edge", () => {
    const v = { depth: 12, nodes: [], groups: [], edges: [
      { source: "input_1", target: "conv_1", shape: [] }, { source: "conv_1", target: "relu_1", shape: [] },
      { source: "x", target: "relu_1", shape: [] },
    ] } as GraphView;
    expect(parentNode(v, "relu_1")).toBe("conv_1");
    expect(parentNode(v, "input_1")).toBeNull();
  });
  it("unions the stats ranges", () => {
    expect(sharedRange(st(-1, 2), st(0, 5))).toEqual([-1, 5]);
    expect(sharedRange(st(-1, 2), null)).toBeNull();
    expect(sharedRange(st(null, 2), st(0, 1))).toBeNull();
    expect(sameShape([1, 4, 8], [1, 4, 8])).toBe(true);
    expect(sameShape([1, 4, 8], [1, 4])).toBe(false);
  });
});
