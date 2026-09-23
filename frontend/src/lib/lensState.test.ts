import { describe, expect, it } from "vitest";
import { clip, lensCellText, lensColumn, lensParams, lensReadout, parseLensParams, shortLayer, type LensState } from "./lensState";
import { href, parseHash } from "./router";
import type { LensResult } from "./types";

const cell = (token: string, prob: number, id = 0) => ({ id, token, prob });
const res: LensResult = {
  method_label: "Logit lens · transformer.ln_f + lm_head", layers: ["embed", "transformer.h.0", "transformer.h.1"],
  pos0: 4, tokens: [" the", " cat"], k: 3,
  top: [
    [[cell(" the", 0.2), cell("a", 0.1), cell("b", 0.05)], [cell(" catalogue", 0.3), cell("x", 0.2), cell("y", 0.1)]],
    [[cell(" dog", 0.41), cell(" the", 0.12), cell(" a", 0.05)], [cell("s", 0.5), cell("x", 0.2), cell("y", 0.1)]],
    [[cell(" mat", 0.9), cell(" the", 0.05), cell(" a", 0.01)], [cell(" sat", 0.8), cell("x", 0.1), cell("y", 0.05)]],
  ],
  target: { id: 9, token: " mat" },
  target_prob: [[0.001, 0.002], [0.003, 0.02], [0.9, 0.3]],
  target_rank: [[40, 30], [17, 5], [1, 2]],
};

describe("lens URL state", () => {
  it("omits defaults", () => {
    const s = parseLensParams(new URLSearchParams(""));
    expect(s).toEqual({ t: null, mode: "top", pos: null });
    expect(href("r", "lens", lensParams(s))).toBe("#/runs/r/lens");
  });
  it("round-trips every field", () => {
    const s: LensState = { t: 17, mode: "target", pos: 5 };
    expect(parseLensParams(parseHash(href("r", "lens", lensParams(s))).params)).toEqual(s);
    expect(parseLensParams(new URLSearchParams("t=x&mode=odd&pos=-1"))).toEqual({ t: null, mode: "top", pos: null });
  });
});

describe("lens grid", () => {
  it("shortens layer names", () => {
    expect(shortLayer("transformer.h.3")).toBe("h.3");
    expect(shortLayer("model.layers.12")).toBe("layers.12");
    expect(shortLayer("embed")).toBe("embed");
  });
  it("clips text", () => {
    expect(clip("abcdef", 6)).toBe("abcdef");
    expect(clip("abcdefg", 6)).toBe("abcde…");
  });
  it("maps positions to columns", () => {
    expect(lensColumn(res, 5)).toBe(1);
    expect(lensColumn(res, 4)).toBe(0);
    expect(lensColumn(res, null)).toBe(1);
    expect(lensColumn(res, 2)).toBe(1);
    expect(lensColumn(res, 99)).toBe(1);
  });
  it("writes cell text in both modes", () => {
    expect(lensCellText(res, "top", 1, 0)).toBe("·dog");
    expect(lensCellText(res, "top", 0, 1)).toBe("·cata…");
    expect(lensCellText(res, "target", 1, 0)).toBe("#17");
    expect(lensCellText(res, "target", 2, 0)).toBe("#1");
  });
  it("reads a cell out", () => {
    expect(lensReadout(res, 1, 1)).toBe('h.0 · pos 5 "·cat" → "s" 0.50 · "x" 0.20 · "y" 0.10 · target "·mat" #5 p 0.020');
    expect(lensReadout(res, 0, 0)).toBe('embed · pos 4 "·the" → "·the" 0.20 · "a" 0.10 · "b" 0.05 · target "·mat" #40 p 0.001');
  });
});
