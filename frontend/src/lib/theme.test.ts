import { describe, expect, it } from "vitest";
import tokens from "../../../design/tokens.json";
import raw from "../styles/tokens.css?raw";
import { contrast, parseColor, resolveTheme, storedFor } from "./theme";

const css = raw.replace(/\/\*[\s\S]*?\*\//g, "");

/** `prop: value;` pairs of the first block opened by `selector {`. */
function block(selector: string): Record<string, string> {
  const at = css.indexOf(`${selector} {`);
  expect(at, selector).toBeGreaterThanOrEqual(0);
  const body = css.slice(css.indexOf("{", at) + 1, css.indexOf("}", at));
  return Object.fromEntries([...body.matchAll(/([\w-]+):\s*([^;{}]+);/g)].map((m) => [m[1], m[2].trim()]));
}
const light = block(":root");
const dark = block(':root[data-theme="dark"]');

describe("theme choice", () => {
  it("resolves stored over system", () => {
    expect(resolveTheme("dark", false)).toBe("dark");
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme(null, true)).toBe("dark");
    expect(resolveTheme(null, false)).toBe("light");
    expect(resolveTheme("sepia", true)).toBe("dark");
  });
  it("stores only a choice that differs from the system", () => {
    expect(storedFor("dark", "light")).toBe("dark");
    expect(storedFor("light", "dark")).toBe("light");
    expect(storedFor("dark", "dark")).toBeNull();
    expect(storedFor("light", "light")).toBeNull();
  });
});

describe("colours", () => {
  it("parses hex and rgb()", () => {
    expect(parseColor("#fff")).toEqual([255, 255, 255]);
    expect(parseColor("#3B4BDB")).toEqual([59, 75, 219]);
    expect(parseColor("rgb(59, 75, 219)")).toEqual([59, 75, 219]);
    expect(parseColor("rgba(1, 2, 3, 0.5)")).toEqual([1, 2, 3]);
    expect(parseColor("rgb(1 2 3 / 50%)")).toEqual([1, 2, 3]);
    expect(() => parseColor("Canvas")).toThrow();
  });
  it("computes WCAG contrast", () => {
    expect(contrast("#000", "#fff")).toBeCloseTo(21, 5);
    expect(contrast("#fff", "#000")).toBeCloseTo(21, 5);
    expect(contrast("#777", "#777")).toBe(1);
  });
});

describe("tokens.css", () => {
  for (const [name, t] of [["light", light], ["dark", dark]] as const) {
    it(`${name} tokens meet the contrast floors`, () => {
      for (const ground of [t["--paper"], t["--panel"]]) {
        expect(contrast(t["--ink"], ground)).toBeGreaterThanOrEqual(4.5);
        expect(contrast(t["--ink-2"], ground)).toBeGreaterThanOrEqual(4.5);
      }
      expect(contrast(t["--accent"], t["--panel"])).toBeGreaterThanOrEqual(4.5);
      expect(contrast(t["--negative"], t["--panel"])).toBeGreaterThanOrEqual(3);
      expect(contrast(t["--negative-text"], t["--panel"])).toBeGreaterThanOrEqual(4.5);
      expect(contrast(t["--accent"], t["--accent-soft"])).toBeGreaterThanOrEqual(4.5);
    });
  }
  it("declares color-scheme per theme and keeps the two dark blocks identical", () => {
    expect(light["color-scheme"]).toBe("light");
    expect(dark["color-scheme"]).toBe("dark");
    expect(block(':root:not([data-theme="light"])')).toEqual(dark);
  });
  it("maps forced colours to system colours only", () => {
    const forced = block(":root:root");
    expect(Object.keys(forced).sort()).toEqual(Object.keys(dark).filter((k) => k !== "color-scheme").sort());
    for (const v of Object.values(forced)) expect(v).toMatch(/^[A-Za-z]+$/);
  });
  it("matches design/tokens.json", () => {
    const pick = (t: Record<string, string>) =>
      Object.fromEntries(Object.entries(t).filter(([k]) => k.startsWith("--")).map(([k, v]) => [k.slice(2), v.toLowerCase()]));
    const lower = (o: Record<string, string>) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, v.toLowerCase()]));
    const lightColours = Object.fromEntries(Object.entries(pick(light)).filter(([k]) => k in tokens.color));
    expect(lightColours).toEqual(lower(tokens.color));
    expect(pick(dark)).toEqual(lower(tokens["color-dark"]));
  });
});
