import { describe, expect, it } from "vitest";
import { href, parseHash } from "./router";
import { attrRequest, channelIsAxis1, layerOp, logitsBlocked, matrixTone, parseTraceParams, traceParams, type TraceState } from "./traceState";

const q = (s: string) => new URLSearchParams(s);
const roundTrip = (s: Partial<TraceState>) => parseTraceParams(parseHash(href("run_0001", "trace", traceParams(s))).params);

describe("trace URL state", () => {
  it("omits defaults", () => {
    const s = parseTraceParams(q(""));
    expect(s).toEqual({ node: null, m: "ig", t: null, obj: null, steps: 32, layer: null, lop: null, nc: null, go: false, iv: null });
    expect(Object.values(traceParams(s)).every((v) => v == null)).toBe(true);
    expect(href("r", "trace", traceParams({}))).toBe("#/runs/r/trace");
  });
  it("round-trips every field", () => {
    const s: TraceState = {
      node: "m:layer1.0:1", m: "gxa", t: 207, obj: { op: "relu_2_4", channel: 3 }, steps: 64,
      layer: "layer_cond", lop: "relu_2_4", nc: 5, go: true, iv: "3f9a0c1d2e4b",
    };
    expect(roundTrip(s)).toEqual(s);
    expect(traceParams(s)).toMatchObject({ obj: "relu_2_4", ch: "3", go: "1", steps: "64", m: "gxa" });
    expect(traceParams({ node: "x", layer: "layer_ig", lop: "conv", go: true })).toEqual({
      node: "x", m: null, t: null, obj: null, ch: null, steps: null, layer: "layer_ig", lop: "conv", nc: null, go: "1", iv: null,
    });
  });
  it("needs both obj and ch", () => {
    expect(parseTraceParams(q("obj=relu_2_4")).obj).toBeNull();
    expect(parseTraceParams(q("ch=3")).obj).toBeNull();
    expect(parseTraceParams(q("obj=relu_2_4&ch=x")).obj).toBeNull();
    expect(parseTraceParams(q("obj=relu_2_4&ch=0")).obj).toEqual({ op: "relu_2_4", channel: 0 });
  });
  it("falls back on bad values", () => {
    const s = parseTraceParams(q("m=nope&t=-1&steps=0&nc=1.5&layer=weird&go=true"));
    expect(s).toMatchObject({ m: "ig", t: null, steps: 32, nc: null, layer: null, go: false });
    expect(parseTraceParams(q("steps=513")).steps).toBe(32);
    expect(parseTraceParams(q("steps=512")).steps).toBe(512);
    expect(parseTraceParams(q("steps=abc")).steps).toBe(32);
  });
});

describe("attrRequest", () => {
  it("nulls the target when a unit objective is set", () => {
    const s = parseTraceParams(q("t=207&m=saliency&steps=8"));
    expect(attrRequest(s)).toEqual({ kind: "attribution", method: "saliency", target: 207, objective: null, steps: 8 });
    expect(attrRequest({ ...s, obj: { op: "relu_2_4", channel: 3 } })).toEqual({
      kind: "attribution", method: "saliency", target: null, objective: { op: "relu_2_4", channel: 3 }, steps: 8,
    });
  });
});

describe("matrixTone", () => {
  it("maps sign to colour and |v| to alpha", () => {
    expect(matrixTone(null)).toEqual({ color: null, alpha: 0 });
    expect(matrixTone(0)).toEqual({ color: "accent", alpha: 0 });
    expect(matrixTone(1)).toEqual({ color: "accent", alpha: 1 });
    expect(matrixTone(0.41)).toEqual({ color: "accent", alpha: 0.41 });
    expect(matrixTone(-0.5)).toEqual({ color: "negative", alpha: 0.5 });
    expect(matrixTone(-1)).toEqual({ color: "negative", alpha: 1 });
  });
});

describe("trace from a unit", () => {
  it("never attributes a layer to its own unit objective", () => {
    const obj = { op: "conv2d_1_1", channel: 3 };
    expect(layerOp("conv2d_1_1", obj)).toBeNull();
    expect(layerOp("relu_1_2", obj)).toBe("relu_1_2");
    expect(layerOp("conv2d_1_1", null)).toBe("conv2d_1_1");
  });
  it("offers a unit only when the view's channel is axis 1", () => {
    expect(channelIsAxis1([1, 64, 8, 8])).toBe(true);
    expect(channelIsAxis1([2, 64, 8, 8])).toBe(false);
    expect(channelIsAxis1([1, 16, 4, 8, 8])).toBe(false);
    expect(channelIsAxis1([64, 8, 8])).toBe(false);
  });
});

describe("logitsBlocked", () => {
  it("blocks only logit objectives on a run whose output has no logits", () => {
    const obj = { op: "conv2d_1_1", channel: 1 };
    expect(logitsBlocked({ obj: null }, "no logits")).toBe(true);
    expect(logitsBlocked({ obj }, "no logits")).toBe(false);
    expect(logitsBlocked({ obj: null }, null)).toBe(false);
  });
});
