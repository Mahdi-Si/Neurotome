import { describe, expect, it } from "vitest";
import { api } from "./api";
import { compact, deltaLarge, fixed, pct, shapeShort, signed } from "./format";
import { href, parseHash } from "./router";

describe("router", () => {
  it("round-trips run routes", () => {
    const h = href("run_0003", "tensors", { node: "conv2d_3_8", c: "12" });
    const r = parseHash(h);
    expect(r.runId).toBe("run_0003");
    expect(r.tab).toBe("tensors");
    expect(r.params.get("node")).toBe("conv2d_3_8");
    expect(r.params.get("c")).toBe("12");
  });
  it("falls back to home and graph", () => {
    expect(parseHash("#/").runId).toBeNull();
    expect(parseHash("#/runs/run_0001/nope").tab).toBe("graph");
  });
  it("routes the lens tab", () => {
    const r = parseHash(href("run_0002", "lens", { t: "17", mode: "target", pos: null }));
    expect(r.tab).toBe("lens");
    expect(r.params.get("t")).toBe("17");
    expect(r.params.has("pos")).toBe(false);
  });
  it("routes the interventions tab with iv and set", () => {
    const set = JSON.stringify({ edits: [{ op: "conv2d_1_1", kind: "zero", channels: null, positions: null }], target: null });
    const h = href("run_0003", "interventions", { node: "conv2d_1_1", iv: "3f9a0c1d2e4b", set });
    expect(h.startsWith("#/runs/run_0003/interventions?")).toBe(true);
    const r = parseHash(h);
    expect(r.tab).toBe("interventions");
    expect(r.params.get("iv")).toBe("3f9a0c1d2e4b");
    expect(JSON.parse(r.params.get("set")!).edits[0].op).toBe("conv2d_1_1");
  });
});

describe("format", () => {
  it("formats numbers compactly", () => {
    expect(compact(11_689_512)).toBe("11.7 M");
    expect(shapeShort([1, 64, 56, 56])).toBe("64×56×56");
    expect(signed(-0.0514)).toBe("−0.051");
    expect([signed(null), signed(NaN), fixed(null), fixed(0.456), fixed(Infinity, 3)]).toEqual(["—", "—", "—", "0.46", "—"]);
    expect([pct(0), pct(0.0042), pct(0.64), pct(null)]).toEqual(["0%", "0.4%", "64%", "—"]);
  });
  it("flags a large IG completeness Δ relative to Σ attributions", () => {
    expect(deltaLarge(28.6, 3.1)).toBe(true);
    expect(deltaLarge(0.0008, -3.1)).toBe(false);
    expect(deltaLarge(null, 3.1)).toBe(false);
    expect(deltaLarge(0.01, 0)).toBe(true);
  });
});

describe("api", () => {
  it("versions the input preview by capture time, so a reused run id never shows a stale image", () => {
    expect(api.inputPreviewUrl("run_0007", 1700000000.5)).toBe("/api/runs/run_0007/input.png?v=1700000000.5");
  });
});
