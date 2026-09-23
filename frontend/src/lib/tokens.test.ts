import { describe, expect, it } from "vitest";
import { showToken, spanLabel, stripReadout, tokenLabels, tokenTone } from "./tokens";
import type { RunMeta } from "./types";

const run = (tokens?: string[]): RunMeta => ({
  id: "run_0001", created: 0, input_shape: [1, 3], output: [], categories: null, reloadable: true, status: "ready",
  sample: { kind: "text", name: "the cat", tokens },
  model: { spec: null, weights: null, class_name: "LM", num_params: 0, num_ops: 1, flops: 0, activation_bytes: 0, truncated: false },
});

describe("token labels", () => {
  it("makes whitespace visible", () => {
    expect(showToken(" cat")).toBe("·cat");
    expect(showToken("a\nb\tc")).toBe("a↵b→c");
    expect(showToken("")).toBe("∅");
    expect(showToken("cat")).toBe("cat");
  });
  it("uses the sample tokens only when their count matches the tensor", () => {
    expect(tokenLabels(run(["the", " cat", " sat"]), 3)).toEqual(["the", " cat", " sat"]);
    expect(tokenLabels(run(["the", " cat", " sat"]), 4)).toBeNull();
    expect(tokenLabels(run(), 3)).toBeNull();
  });
  it("labels single positions and pooled spans", () => {
    const toks = Array.from({ length: 10 }, (_, i) => String.fromCharCode(97 + i));   // a..j
    expect(spanLabel(toks, 10, 10, 3)).toBe('#3 "d"');
    expect(spanLabel(toks, 10, 5, 1)).toBe('#2–3 "cd"');
    expect(spanLabel(toks, 10, 2, 0)).toBe('#0–4 "abcd…"');
    expect(spanLabel([" a", " b", " c", " d"], 4, 1, 0)).toBe('#0–3 "·a·b·c·d"');
    expect(spanLabel(null, 10, 10, 3)).toBe("#3");
    expect(spanLabel(null, 68, 17, 16)).toBe("#64–67");
  });
  it("tones signed and unsigned scores", () => {
    expect(tokenTone(0.5, 2, true)).toEqual({ color: "accent", alpha: 0.25 });
    expect(tokenTone(-2, 2, true)).toEqual({ color: "negative", alpha: 1 });
    expect(tokenTone(-2, 2, false)).toEqual({ color: "accent", alpha: 1 });
    expect(tokenTone(1, 0, true)).toEqual({ color: "accent", alpha: 0 });
  });
  it("reads a strip token out", () => {
    expect(stripReadout(["the", " cat"], [0.1234, -0.4], true, 0)).toBe('#0 "the" · +0.1234 · 24% of Σ|a|');
    expect(stripReadout(["the", " cat"], [0.1234, -0.4], true, 1)).toBe('#1 "·cat" · −0.4000 · 76% of Σ|a|');
    expect(stripReadout(["a"], [0.5], false, 0)).toBe('#0 "a" · 0.5000 · 100% of Σ|a|');
    expect(stripReadout(["a"], [0], false, 0)).toBe('#0 "a" · 0.0000 · 0% of Σ|a|');
  });
});
