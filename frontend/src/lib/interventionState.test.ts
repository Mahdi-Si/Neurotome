import { describe, expect, it } from "vitest";
import {
  addEdit, deltaColor, editLabel, formatIndexList, parseIndexList, parseSet, removeEdit, scopeAxes, setParam,
} from "./interventionState";
import type { Edit, InterventionSet } from "./types";

const edit = (o: Partial<Edit> = {}): Edit => ({ op: "conv2d_1_1", kind: "zero", channels: null, positions: null, ...o });
const q = (set: string | null) => new URLSearchParams(set == null ? "" : { set });

describe("set in the URL", () => {
  it("round-trips through setParam / parseSet", () => {
    const s: InterventionSet = { edits: [edit({ channels: [0, 3] }), edit({ kind: "scale", alpha: 2, positions: "last" })], target: 3 };
    expect(parseSet(q(setParam(s)))).toEqual(s);
    expect(parseSet(q(setParam({ ...s, target: null })))!.target).toBeNull();
  });
  it("is null when missing, malformed or empty", () => {
    for (const raw of [null, "", "{", "[]", '{"edits":[]}', '{"edits":[{"op":"a","kind":"bogus"}]}', '{"edits":[{"kind":"zero"}]}', "null"])
      expect(parseSet(q(raw))).toBeNull();
    expect(parseSet(q('{"edits":[{"op":"a","kind":"zero"}],"target":-1}'))!.target).toBeNull();
    expect(setParam(null)).toBeNull();
  });
  it("rejects edits whose fields have the wrong type (they used to crash editLabel)", () => {
    for (const bad of [{ kind: "scale", alpha: "2" }, { kind: "constant", value: null, alpha: [1] }, { channels: "0,3" }, { channels: [-1] },
      { positions: "first" }, { positions: [[1, 2, 3]] }, { kind: "steer", alpha: 1, direction: { kind: "channel", channel: "3" } },
      { kind: "steer", alpha: 1, direction: { kind: "contrast", run_a: "a" } }, { kind: "resample", seed: 1.5 }]) {
      const raw = JSON.stringify({ edits: [edit(bad as Partial<Edit>)] });
      expect(parseSet(q(raw)), raw).toBeNull();
    }
    const ok = edit({ kind: "steer", alpha: -1, positions: [[0, 1]], direction: { kind: "contrast", run_a: "a", run_b: "b", pool: "last" } });
    expect(parseSet(q(JSON.stringify({ edits: [ok] })))!.edits[0]).toEqual(ok);
    expect(parseSet(q('{"edits":[{"op":"a","kind":"zero"}]}'))!.edits).toHaveLength(1);   // missing scope keys = all
    expect(setParam({ edits: [], target: 1 })).toBeNull();
  });
  it("adds and removes edits; the last one out gives null", () => {
    const a = addEdit(null, edit());
    expect(a).toEqual({ edits: [edit()], target: null });
    const b = addEdit({ ...a, target: 2 }, edit({ op: "b" }));
    expect(b.edits.map((e) => e.op)).toEqual(["conv2d_1_1", "b"]);
    expect(b.target).toBe(2);
    expect(removeEdit(b, 0)).toEqual({ edits: [edit({ op: "b" })], target: 2 });
    expect(removeEdit(a, 0)).toBeNull();
  });
});

describe("index lists", () => {
  it("parses lists and ranges, sorted and unique", () => {
    expect(parseIndexList("")).toEqual([]);
    expect(parseIndexList("   ")).toEqual([]);
    expect(parseIndexList("0, 3,7-9")).toEqual([0, 3, 7, 8, 9]);
    expect(parseIndexList(" 9 , 2 – 3, 2 ")).toEqual([2, 3, 9]);
  });
  it("rejects reversed ranges and junk", () => {
    for (const t of ["3-1", "x", "-1", "1.5", "1-", "2,a"]) expect(parseIndexList(t)).toBeNull();
    expect(parseIndexList("0-999999999")).toBeNull();   // rejected before expanding
    expect(parseIndexList("0-3", 4)).toEqual([0, 1, 2, 3]);
    expect(parseIndexList("2, 4", 4)).toBeNull();
  });
  it("formats runs of three or more as ranges", () => {
    expect(formatIndexList(null)).toBe("");
    expect(formatIndexList([])).toBe("");
    expect(formatIndexList([0, 3, 7, 8, 9])).toBe("0,3,7–9");
    expect(formatIndexList([1, 2, 5])).toBe("1,2,5");
    expect(parseIndexList(formatIndexList([0, 1, 2, 3, 10]))).toEqual([0, 1, 2, 3, 10]);
  });
});

describe("scope axes", () => {
  it("follows the rank", () => {
    expect(scopeAxes([1, 8, 6, 5])).toEqual({ channels: 8, positions: "pixels", size: [6, 5] });
    expect(scopeAxes([1, 7, 32])).toEqual({ channels: 32, positions: "tokens", size: [7] });
    expect(scopeAxes([1, 10])).toEqual({ channels: 10, positions: null, size: [] });
    expect(scopeAxes([10])).toEqual({ channels: null, positions: null, size: [] });
  });
});

describe("edit labels", () => {
  const L = (o: Partial<Edit>) => editLabel(edit(o), "block.conv2");
  it("names every kind", () => {
    expect(L({})).toBe("Zero · block.conv2 · all");
    expect(L({ kind: "mean", over: "positions" })).toBe("Mean over positions · block.conv2 · all");
    expect(L({ kind: "mean" })).toBe("Mean over positions · block.conv2 · all");
    expect(L({ kind: "mean", over: "runs" })).toBe("Mean over runs · block.conv2 · all");
    expect(L({ kind: "resample", source_run: "run_0004" })).toBe("Replace ← run_0004 · block.conv2 · all");
    expect(L({ kind: "resample", source_run: "random", seed: 0 })).toBe("Resample ← random · seed 0 · block.conv2 · all");
    expect(L({ kind: "constant", value: 0.5 })).toBe("Constant 0.5 · block.conv2 · all");
    expect(L({ kind: "scale", alpha: 2 })).toBe("Scale ×2 · block.conv2 · all");
    expect(L({ kind: "scale", alpha: 1.25 })).toBe("Scale ×1.25 · block.conv2 · all");
    expect(L({ kind: "scale", alpha: 1 / 3 })).toBe("Scale ×0.333 · block.conv2 · all");
    expect(L({ kind: "steer", alpha: 4, direction: { kind: "contrast", run_a: "run_0005", run_b: "run_0006", pool: "last" } }))
      .toBe("Steer +4 × (run_0006 − run_0005, last) · block.conv2 · all");
    expect(L({ kind: "steer", alpha: 4, direction: { kind: "channel", channel: 3 } })).toBe("Steer +4 × e_3 · block.conv2 · all");
    expect(L({ kind: "steer", alpha: -0.5, direction: { kind: "channel", channel: 3 } })).toBe("Steer −0.5 × e_3 · block.conv2 · all");
  });
  it("names every scope form", () => {
    expect(L({ channels: [0, 3, 7, 8, 9] })).toBe("Zero · block.conv2 · ch 0,3,7–9");
    expect(L({ positions: [[4, 7], [4, 8]] })).toBe("Zero · block.conv2 · px (4,7) (4,8)");
    expect(L({ positions: [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1]] })).toBe("Zero · block.conv2 · px (0,0) (0,1) (0,2) (0,3) +2");
    expect(L({ positions: "last" })).toBe("Zero · block.conv2 · last token");
    expect(L({ positions: [2, 5] })).toBe("Zero · block.conv2 · tokens 2,5");
    expect(L({ channels: [1], positions: [2, 3, 4] })).toBe("Zero · block.conv2 · ch 1 · tokens 2–4");
  });
});

it("colours Δ by sign", () => {
  expect(deltaColor(0)).toBe("var(--accent)");
  expect(deltaColor(-0.1)).toBe("var(--negative-text)");
  expect(deltaColor(null)).toBeUndefined();
});
