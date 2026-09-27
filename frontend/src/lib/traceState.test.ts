import { describe, expect, it } from "vitest";
import { href, parseHash } from "./router";
import {
  attrRequest, blockChannels, blockSpans, channelIsAxis1, isSeries, layerOp, logitsBlocked, matrixTone, parseTraceParams, seqTimeAxis, seriesClock,
  seriesCol, seriesRow, sliceText, stepsLabel, stepText, traceParams, windowShares, zoomSpan, type TraceState,
} from "./traceState";
import type { SeriesMap, TargetWindow } from "./types";

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

describe("time-series attribution map", () => {
  const blk = (name: string, start: number, stop: number, labels: string[] | null = null) =>
    ({ name, start, stop, labels, unit: null, sum: 0, abs: 1, share: 0.5 });
  // (1, 300, 126)-like: 4 s steps from −1200 s, 3 steps a column, two blocks of labelled channels
  const m: SeriesMap = {
    kind: "series", map: [[0]], total: 0, convergence_delta: null, channels: 6, row_stride: 1, steps: 300, stride: 3,
    time: { t0: -1200, step_s: 4, label: "s to delivery" },
    blocks: [blk("fhr_st", 0, 4, ["ψ0", "ψ1", "ψ2", "ψ3 · 0.887 Hz"]), blk("up_st", 4, 6)], time_totals: [],
  };
  it("labels rows by block and channel", () => {
    expect(isSeries(m)).toBe(true);
    expect(isSeries({ map: [[0]], total: 0, convergence_delta: null })).toBe(false);
    expect(seriesRow(m, 3)).toBe('fhr_st · "ψ3 · 0.887 Hz"');
    expect(seriesRow(m, 5)).toBe("up_st · ch 5");
    expect(seriesRow({ ...m, blocks: [blk("fhr", 0, 1), blk("up", 1, 2)], channels: 2 }, 1)).toBe("up");
    expect(seriesRow({ ...m, row_stride: 4, channels: 400 }, 1)).toBe("up_st · ch 4–7");
  });
  it("puts columns on the run's clock, minutes from a 10 min span", () => {
    const k = seriesClock(m);
    expect([k.div, k.label, k.unit, k.at(0), k.at(299)]).toEqual([60, "min to delivery", "min", -20, (-1200 + 299 * 4) / 60]);
    expect(seriesCol(m, 0)).toBe("t=-19.93 min");   // steps 0–2, centre step 1 = −1196 s
    expect(seriesCol({ ...m, time: null }, 2)).toBe("step 6–8");
    expect(seriesCol({ ...m, time: null, stride: 1 }, 2)).toBe("step 2");
    expect(seriesClock({ ...m, time: { t0: 608, step_s: 8, label: "s from recording start" }, steps: 74 }).label).toBe("s from recording start");
  });
  it("places block separators on row boundaries", () => {
    expect(blockSpans(m)).toEqual([{ name: "fhr_st", top: 0, bottom: (4 / 6) * 100 }, { name: "up_st", top: (4 / 6) * 100, bottom: 100 }]);
    const pooled = blockSpans({ ...m, channels: 400, row_stride: 3, blocks: [blk("a", 0, 200), blk("b", 200, 400)] });
    expect(pooled[1].top).toBeCloseTo((66 / 134) * 100);
    expect(pooled[1].bottom).toBe(100);
  });
});

describe("a picked step's input windows", () => {
  const blk = (name: string, start: number, stop: number) => ({ name, start, stop, labels: null, unit: null, sum: 0, abs: 1, share: 0.5 });
  // 4 channels (2 blocks) × 6 steps, unpooled; scale 2: map × 2 = Σ a of a cell
  const m: SeriesMap = {
    kind: "series", total: 0, convergence_delta: null, channels: 4, row_stride: 1, steps: 6, stride: 1, time: null, scale: 2,
    map: [
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, -1, 0, 0, 0, 0],
      [0, 0, 0.5, 0, 0, 0.5],
    ],
    blocks: [blk("hr", 0, 2), blk("up", 2, 4)], time_totals: [0, -2, 3, 0, 0, 1],
  };
  const w = (kind: TargetWindow["kind"], start: number, stop: number, blocks: string[] | null = null): TargetWindow =>
    ({ label: `${kind} ${start}`, kind, start, stop, blocks });
  it("splits Σ|a| over windows, and what lies outside all of them", () => {
    const rows = windowShares(m, [w("target", 2, 3), w("context", 0, 2, ["up"])]);
    expect(rows.map((r) => [r.kind, r.share, r.sum])).toEqual([
      ["target", 1.5 / 3, 3],        // step 2 over every channel: 1 + 0.5
      ["context", 1 / 3, -2],        // steps 0–1 of block up: −1
      ["outside", 0.5 / 3, 1],       // step 5 of up
    ]);
    // overlapping windows each count their own cells; outside counts a cell once
    const overlap = windowShares(m, [w("context", 0, 6), w("target", 2, 3)]);
    expect(overlap.map((r) => r.share)).toEqual([1, 0.5, 0]);
    expect(windowShares({ ...m, map: m.map.map((r) => r.map(() => 0)) }, [])[0].share).toBe(0);
  });
  it("pools windows onto pooled columns and rows", () => {
    const pooled = { ...m, stride: 3, row_stride: 4, steps: 18, channels: 8, map: [[1, 0], [0, 3]], blocks: [blk("hr", 0, 4), blk("up", 4, 8)] };
    // steps 4–5 are in column 1; block up is row 1
    expect(windowShares(pooled, [w("horizon", 4, 6, ["up"])])[0].share).toBe(0.75);
  });
  it("zooms to the step and its target / horizon windows ± 10 % of the input (2–30 steps), ignoring context", () => {
    expect(zoomSpan(300, 123, [w("target", 123, 124), w("context", 0, 124), w("horizon", 124, 134)])).toEqual([93, 164]);
    expect(zoomSpan(300, 5, [w("target", 5, 6)])).toEqual([0, 36]);
    expect(zoomSpan(74, 40, [w("target", 40, 41), w("context", 28, 40)])).toEqual([32, 49]);
    expect(zoomSpan(300, 10, [])).toEqual([0, 41]);
    expect(zoomSpan(300, 290, [])).toEqual([260, 300]);
  });
  it("writes the Values slice of a window", () => {
    expect(sliceText([1, 300, 126], 1, [250, 261], [80, 116])).toBe("[0, 250:261, 80:116]");
    expect(sliceText([1, 2, 5280], 2, [0, 40], null)).toBe("[0, :, 0:40]");
    expect(blockChannels(m.blocks, ["up"])).toEqual([2, 4]);
    expect(blockChannels(m.blocks, ["hr", "up"])).toEqual([0, 4]);
    expect(blockChannels(m.blocks, null)).toBeNull();
    expect(blockChannels(m.blocks, ["nope"])).toBeNull();
    expect(seqTimeAxis([1, 300, 126], { time: { axis: 1, step_s: 4, t0: 0 } })).toBe(1);
    expect(seqTimeAxis([1, 2, 5280], null)).toBe(2);
    expect(seqTimeAxis([1, 20, 30], { blocks: [blk("a", 0, 30)] })).toBe(1);
  });
  it("names a step by its label and value", () => {
    const out = { n: 3, stride: 1, values: [0.5, null, 2], labels: ["t0 · +0:08", null, "t2 · +0:24"] };
    expect(stepText(out, 2)).toBe("t2 · +0:24 · value 2.0000");
    expect(stepText(out, 1)).toBe("[1]");
    expect(stepText({ ...out, stride: 4, labels: null }, 0)).toBe("[0]");
    expect([stepsLabel(40, 41), stepsLabel(28, 40)]).toEqual(["step 40", "steps 28–39"]);
  });
});
