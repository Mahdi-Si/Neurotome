// Capture options (Home form) and partial-capture wording (banner, graph nodes, inspector, tensor view).
import { bytes } from "./format";
import type { GraphNode, RunMeta, Unsaved } from "./types";

/** The node's output tensor is on disk (old servers send no `unsaved`: saved iff it has stats). */
export const hasTensor = (n: GraphNode) => !n.unsaved && n.stats != null;

/** "layer3, layer4.1  layer3" → ["layer3", "layer4.1"] (split on commas/whitespace, trimmed, deduped); nothing → null */
export function parseModules(text: string): string[] | null {
  const mods = [...new Set(text.split(/[\s,]+/).filter(Boolean))];
  return mods.length ? mods : null;
}

/** "" → null; a positive finite number → it; anything else → NaN (the form refuses NaN) */
export function parseBudget(text: string): number | null {
  const t = text.trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) && n > 0 ? n : NaN;
}

const UNSAVED: Record<Unsaved, string> = {
  budget: "capture budget", modules: "outside the captured modules", no_tensor: "no tensor output",
};
export const unsavedText = (u: Unsaved | null | undefined) => (u ? UNSAVED[u] : "");

/** The banner for a partial capture, or null for a complete one. */
export function captureNote(meta: RunMeta): { tag: "Truncated" | "Selective"; text: string } | null {
  const { truncated, budget_bytes, capture_modules: mods, saved_ops, num_ops } = meta.model;
  const count = saved_ops != null ? ` · ${saved_ops}/${num_ops} ops saved · ∅ nodes have no tensor` : "";
  if (truncated) {
    if (saved_ops == null) return { tag: "Truncated", text: "Capture budget reached · some ops have no tensor" };
    const budget = budget_bytes != null ? ` ${bytes(budget_bytes)}` : "";
    return { tag: "Truncated", text: `Capture budget${budget} reached${count}${mods?.length ? ` · modules ${mods.join(", ")}` : ""}` };
  }
  if (mods?.length) return { tag: "Selective", text: `Tensors saved inside ${mods.join(", ")}${count}` };
  return null;
}
