import { describe, expect, it } from "vitest";
import { fileName, resolveCssVars } from "./exportPng";

describe("resolveCssVars", () => {
  const look = (n: string) => ({ "--ink": "#111111", "--panel": " #fafaf8 " } as Record<string, string>)[n] ?? "";

  it("replaces known variables and trims the value", () => {
    expect(resolveCssVars('<rect fill="var(--ink)" stroke="var( --panel )"/>', look))
      .toBe('<rect fill="#111111" stroke="#fafaf8"/>');
  });
  it("uses the fallback when the lookup is empty, else empty", () => {
    expect(resolveCssVars("fill: var(--x, #fff); stroke: var(--y)", look)).toBe("fill: #fff; stroke: ");
    expect(resolveCssVars("fill: var(--ink, #fff)", look)).toBe("fill: #111111");
  });
  it("leaves other text alone", () => {
    const s = '<svg viewBox="0 0 10 10"><circle r="2" fill="transparent"/></svg>';
    expect(resolveCssVars(s, look)).toBe(s);
  });
});

describe("fileName", () => {
  it("sanitises each part and joins with _", () => {
    expect(fileName(["run_0001", "conv2d_3_8", "[0, 1:3]"], "npy")).toBe("run_0001_conv2d_3_8_0-1-3.npy");
  });
  it("drops null, undefined and empty parts", () => {
    expect(fileName(["run_0001", null, "", undefined, "c", 3, "dots"], "png")).toBe("run_0001_c_3_dots.png");
    expect(fileName(["a/b", "[:]"], "png")).toBe("a-b.png");
  });
});
