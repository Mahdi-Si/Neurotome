import { describe, expect, it } from "vitest";
import { axisLetters, headColumns, headReadout, patternLabel } from "./attentionState";
import type { AttentionHead, AttentionTensorView } from "./types";

const view = (tokens: number, n: number, max = 0.5, keys = tokens, nk = n): AttentionTensorView => ({
  kind: "attention", shape: [1, 2, tokens, keys], layout: "hqk", square: tokens === keys, heads: 2, tokens, head: 0,
  queries: tokens, keys, maps: [], full: { map: Array.from({ length: n }, () => Array(nk).fill(0)), max }, summary: [],
  key_profile: [], stats: { mean: 0, std: 0, min: 0, max: 1 }, hist: { counts: [], range: [0, 1] },
});

const head: AttentionHead = { head: 1, entropy: 2.5, entropy_norm: 0.4, sparsity: 0.25, masked: 0.05, argmax: 3, argmax_weight: 0.1234 };

describe("attention view", () => {
  it("reads single cells with tokens, unscaled", () => {
    expect(headReadout(view(3, 3), 2, 1, 0.5, ["the", " cat", " sat"])).toBe('q #2 "·sat" → k #1 "·cat" · 0.250');
    expect(headReadout(view(3, 3), 0, 0, 1, null)).toBe("q #0 → k #0 · 0.500");
  });
  it("says mean for pooled cells", () => {
    expect(headReadout(view(200, 128, 1), 0, 127, 0.25, null)).toBe("q #0–1 → k #199 · mean 0.250");
  });
  it("labels only the axis whose length matches the tokens", () => {
    // 3 queries = 3 tokens, 5 keys (lags): the key axis stays numeric
    expect(headReadout(view(3, 3, 1, 5, 5), 1, 4, 0.5, ["a", "b", "c"])).toBe('q #1 "b" → k #4 · 0.500');
    expect(headReadout(view(300, 128, 1, 91, 91), 127, 90, 1, null)).toBe("q #298–299 → k #90 · mean 1.000");
  });
  it("names patterns", () => {
    expect(patternLabel("previous")).toBe("prev");
    expect(patternLabel("first")).toBe("first");
    expect(patternLabel("self")).toBe("self");
    expect(patternLabel(null)).toBe("");
    expect(patternLabel(undefined)).toBe("");
  });
  it("adapts the head-table columns", () => {
    const sq = headColumns(true), ns = headColumns(false);
    expect(sq.map(([c]) => c)).toEqual(["Entropy (bits)", "Prev", "First", "Self", "Zeros", "Peak key"]);
    expect(ns.map(([c]) => c)).toEqual(["Entropy (bits)", "Entropy / log K", "Masked", "Zeros", "Peak key"]);
    expect(ns.map(([, f]) => f(head))).toEqual(["2.50", "0.40", "5.0%", "25%", "k3 · 0.123"]);
    expect(sq[1][1](head)).toBe("—");   // no prev on a non-square head
    expect(sq[1][1]({ ...head, prev: 0.9 })).toBe("0.90");
  });
  it("letters the axes", () => {
    expect(axisLetters([1, 12, 7, 7], "attention")).toEqual(["B", "H", "Q", "K"]);
    expect(axisLetters([12, 7, 7], "attention")).toEqual(["H", "Q", "K"]);
    expect(axisLetters([1, 300, 4, 91], "attention", "qhk")).toEqual(["B", "Q", "H", "K"]);
    expect(axisLetters([300, 4, 91], "attention", "qhk")).toEqual(["Q", "H", "K"]);
    expect(axisLetters([1, 64, 8, 8], "image")).toEqual(["B", "C", "H", "W"]);
    expect(axisLetters([1, 7, 32], "sequence")).toEqual(["B", "T", "D"]);
    expect(axisLetters([2, 7], "vector")).toEqual(["d0", "d1"]);
  });
});
