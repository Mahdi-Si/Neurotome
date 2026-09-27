import { describe, expect, it } from "vitest";
import {
  bandPath, clock, envelopeDomain, hasMatrix, isSignal, linePath, modelCaption, padEnd, pictureSize, pictureWidth, place, rawCaption,
  rawNote, rawSpan, sparseRows, stepRange, stepSpan, union,
} from "./inputState";
import type { InputInfo, SignalRaw } from "./types";

const info = (o: Partial<InputInfo> = {}): InputInfo => ({
  op: "input_1", shape: [1, 3, 224, 224], dtype: "torch.float32",
  sample: { kind: "image", name: "cat.jpg" }, raw: { file: "input_raw.jpg", width: 512, height: 600, format: "JPEG", bytes: 62_700 },
  steps: ["Convert to RGB"], normalised: true, picture: true, channels: [], ...o,
});

describe("input view", () => {
  it("captions the raw sample and the model input", () => {
    expect(rawCaption(info())).toMatch(/^cat\.jpg · JPEG · 512×600 px · /);
    expect(rawCaption(info({ raw: null }))).toBe("");
    expect(modelCaption(info())).toBe("1×3×224×224 · float32 · de-normalised for display");
    expect(modelCaption(info({ normalised: false }))).toBe("1×3×224×224 · float32 · min–max stretched for display");
    expect(modelCaption(info({ picture: false, shape: [1, 7], dtype: "torch.int64" }))).toBe("1×7 · int64");
  });
  it("says why there is no raw picture", () => {
    expect(rawNote(info({ sample: { kind: "random", name: "randn" } }))).toMatch(/random/);
    expect(rawNote(info({ sample: { kind: "tensor", name: "x.npy" } }))).toMatch(/uploaded tensor/);
    expect(rawNote(info({ sample: { kind: "tensor", name: "in-process input" }, steps: null }))).toMatch(/neurotome\.open/);
    expect(rawNote(info({ raw: null, steps: null }))).toMatch(/before 0\.3\.0/);
  });
  it("sizes pictures between 224 and 360 px tall, keeping the aspect", () => {
    expect(pictureWidth(28, 28)).toBe(224);          // small: scaled up
    expect(pictureWidth(512, 600)).toBe(307);        // big: 360 tall
    expect(pictureWidth(300, 200)).toBe(336);        // in between: 224 tall (its own 200 is below the floor)
    expect(pictureSize(info())).toEqual([224, 224]);
    expect(pictureSize(info({ picture: false }))).toBeNull();
  });
});

describe("signal input view", () => {
  const raw: SignalRaw = { kind: "signal", file: "input_raw.npz", fs: 4, t0: 600, time_label: "s from recording start",
    channels: [{ name: "hr", unit: "bpm", length: 2400 }, { name: "pressure", unit: "mmHg", length: 2400 }] };
  const sig = info({ raw, shape: [1, 74, 10], sample: { kind: "ref", name: "rec02 · segment 1", ref: "rec02:1" }, picture: false });
  it("recognises a signal run and a (1, T, C) float input", () => {
    expect(isSignal(sig.raw)).toBe(true);
    expect(isSignal(info().raw)).toBe(false);
    expect(hasMatrix(sig)).toBe(true);
    expect(hasMatrix(info())).toBe(false);
    expect(hasMatrix(info({ shape: [1, 7, 4], dtype: "torch.int64" }))).toBe(false);
    expect(rawCaption(sig)).toBe("rec02 · segment 1 · 2 channels · 2400 samples at 4 Hz");
    expect(rawNote(info({ raw: null, sample: { kind: "ref", name: "r" } }))).toMatch(/load_sample/);
  });
  it("puts the raw signal and the model steps on one clock", () => {
    expect(rawSpan(raw)).toEqual([600, 1200]);
    const steps = stepSpan({ axis: 1, step_s: 8, t0: 608 }, 74, 1);   // cells centred on 608 + 8i
    expect(steps).toEqual([604, 1196]);
    const dom = union(rawSpan(raw), steps);
    expect(dom).toEqual([600, 1200]);
    expect(place(steps, dom)).toEqual({ left: (4 / 600) * 100, width: (592 / 600) * 100 });
    expect(place([0, 1], [0, 0])).toEqual({ left: 0, width: 100 });   // an empty domain never divides by 0
    expect(clock(600, "s from recording start")).toEqual({ div: 60, label: "min from recording start" });
    expect(clock(120, "s")).toEqual({ div: 1, label: "s" });
    expect(clock(3600, "hours")).toEqual({ div: 60, label: "hours · min" });
  });
  it("labels every k-th row and pads axis ends", () => {
    expect(sparseRows(5)).toEqual([0, 1, 2, 3, 4]);
    expect(sparseRows(36)).toEqual([0, 5, 10, 15, 20, 25, 30, 35]);
    expect(padEnd("12", 4)).toBe("  12");
  });
  it("draws envelopes with gaps", () => {
    const x = (i: number) => i * 10, y = (v: number) => v;
    expect(linePath([1, 2, null, 4], x, y)).toBe("M0.0 1.0L10.0 2.0M30.0 4.0");
    expect(bandPath([0, 1, null], [2, 3, null], x, y)).toBe("M0.0 2.0L10.0 3.0L10.0 1.0L0.0 0.0Z");
    expect(envelopeDomain([0, null], [10, NaN])).toEqual([-0.5, 10.5]);
    expect(envelopeDomain([5], [5])).toEqual([3.9, 6.1]);
    expect(envelopeDomain([null], [null])).toEqual([0, 1]);
  });
});

describe("a step range on the chart clock", () => {
  it("is seconds with a time axis, else steps", () => {
    expect(stepRange({ axis: 1, step_s: 8, t0: 608 }, 28, 40)).toEqual([828, 924]);   // cells centred on 608 + 8i
    expect(stepRange(null, 28, 40)).toEqual([27.5, 39.5]);
  });
});
