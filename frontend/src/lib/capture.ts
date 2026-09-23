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

const VISION = /vit|deit|beit|swin|convnext|resnet|dino|detr|clip|siglip|image|segformer|mobilenet|efficientnet|regnet|yolo/i;

/** The Home form defaults to a text sample for a known text model or any hf: spec that does not look like vision. */
export const isTextSpec = (spec: string, text: string[]) => text.includes(spec) || (/^hf:/i.test(spec) && !VISION.test(spec));

/** The banner for a partial capture or an NNsight (module-level) run, or null for a complete TorchLens one. */
export function captureNote(meta: RunMeta): { tag: "Truncated" | "Selective" | "Modules"; text: string } | null {
  const { truncated, budget_bytes, capture_modules: mods, saved_ops, num_ops, backend } = meta.model;
  const count = saved_ops != null ? ` · ${saved_ops}/${num_ops} ops saved · ∅ nodes have no tensor` : "";
  if (truncated) {
    if (saved_ops == null) return { tag: "Truncated", text: "Capture budget reached · some ops have no tensor" };
    const budget = budget_bytes != null ? ` ${bytes(budget_bytes)}` : "";
    return { tag: "Truncated", text: `Capture budget${budget} reached${count}${mods?.length ? ` · modules ${mods.join(", ")}` : ""}` };
  }
  if (mods?.length) return { tag: "Selective", text: `Tensors saved inside ${mods.join(", ")}${count}` };
  if (backend === "nnsight") return { tag: "Modules", text: `NNsight capture · module outputs only · ${num_ops} module calls` };
  return null;
}
