import { describe, expect, it } from "vitest";
import { compact, shapeShort, signed } from "./format";
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
});

describe("format", () => {
  it("formats numbers compactly", () => {
    expect(compact(11_689_512)).toBe("11.7 M");
    expect(shapeShort([1, 64, 56, 56])).toBe("64×56×56");
    expect(signed(-0.0514)).toBe("−0.051");
  });
});
