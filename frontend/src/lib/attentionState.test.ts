import { describe, expect, it } from "vitest";
import { axisLetters, headReadout, patternLabel } from "./attentionState";
import type { AttentionTensorView } from "./types";

const view = (tokens: number, n: number, max = 0.5): AttentionTensorView => ({
  kind: "attention", shape: [1, 2, tokens, tokens], heads: 2, tokens, head: 0,
  maps: [], full: { map: Array.from({ length: n }, () => Array(n).fill(0)), max }, summary: [],
  stats: { mean: 0, std: 0, min: 0, max: 1 }, hist: { counts: [], range: [0, 1] },
});

describe("attention view", () => {
  it("reads single cells with tokens, unscaled", () => {
    expect(headReadout(view(3, 3), 2, 1, 0.5, ["the", " cat", " sat"])).toBe('q #2 "·sat" → k #1 "·cat" · 0.250');
    expect(headReadout(view(3, 3), 0, 0, 1, null)).toBe("q #0 → k #0 · 0.500");
  });
  it("says mean for pooled cells", () => {
    expect(headReadout(view(200, 128, 1), 0, 127, 0.25, null)).toBe("q #0–1 → k #199 · mean 0.250");
  });
  it("names patterns", () => {
    expect(patternLabel("previous")).toBe("prev");
    expect(patternLabel("first")).toBe("first");
    expect(patternLabel("self")).toBe("self");
    expect(patternLabel(null)).toBe("");
  });
  it("letters the axes", () => {
    expect(axisLetters([1, 12, 7, 7], "attention")).toEqual(["B", "H", "Q", "K"]);
    expect(axisLetters([12, 7, 7], "attention")).toEqual(["H", "Q", "K"]);
    expect(axisLetters([1, 64, 8, 8], "image")).toEqual(["B", "C", "H", "W"]);
    expect(axisLetters([1, 7, 32], "sequence")).toEqual(["B", "T", "D"]);
    expect(axisLetters([2, 7], "vector")).toEqual(["d0", "d1"]);
  });
});
