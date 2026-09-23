// Overview of the whole graph, bottom-left of GraphCanvas. Click or drag to centre the main view there.
import { useEffect, useRef } from "react";
import type { Placed } from "../lib/layout";
import type { NodeTone } from "../lib/types";
import { mix, toScreen, viewRect, type Tf } from "../lib/viewport";
import type { Tokens } from "./GraphCanvas";

const MM_W = 200, MM_H = 140;

interface Props {
  placed: Placed;
  tones?: Map<string, NodeTone>;
  selected: string | null;
  tf: Tf;
  w: number;
  h: number;
  colors: Tokens;
  onCentre: (gx: number, gy: number) => void;
}

export function Minimap({ placed, tones, selected, tf, w, h, colors: c, onCentre }: Props) {
  const cv = useRef<HTMLCanvasElement>(null);
  const W = Math.max(1, placed.width), H = Math.max(1, placed.height);
  const mh = Math.max(24, Math.min(MM_H, (H * MM_W) / W));
  const s = Math.min(MM_W / W, mh / H);
  const ox = (MM_W - W * s) / 2, oy = (mh - H * s) / 2;
  const [x0, y0] = toScreen(tf, 0, 0), [x1, y1] = toScreen(tf, W, H);
  const fits = x0 >= 0 && y0 >= 0 && x1 <= w && y1 <= h;

  useEffect(() => {
    const el = cv.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = Math.round(MM_W * dpr);
    el.height = Math.round(mh * dpr);
    const ctx = el.getContext("2d")!;
    ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * ox, dpr * oy);
    const px = 1 / s;
    ctx.lineWidth = px;
    ctx.strokeStyle = c.lineFaint;
    for (const b of placed.groups.values()) ctx.strokeRect(b.x, b.y, b.w, b.h);
    const dot = (b: { x: number; y: number; w: number; h: number }, min: number) =>
      ctx.fillRect(b.x, b.y, Math.max(b.w, min * px), Math.max(b.h, min * px));
    for (const [id, b] of placed.nodes) {
      const t = tones?.get(id);
      ctx.globalAlpha = t?.flag || t?.v != null ? 1 : 0.55;
      ctx.fillStyle = t?.flag ? c.negative : t?.v != null ? mix(c.panel, c.accent, 0.08 + 0.32 * t.v) : c.ink;
      dot(b, 1);
    }
    ctx.globalAlpha = 1;
    const sel = selected ? placed.nodes.get(selected) : undefined;
    if (sel) { ctx.fillStyle = c.accent; dot(sel, 3); }
  }, [placed, tones, selected, fits, mh, s, ox, oy, c]);

  if (fits) return null;
  const vr = viewRect(tf, w, h);
  const centreAt = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onCentre((e.clientX - r.left - ox) / s, (e.clientY - r.top - oy) / s);
  };
  return (
    <div className="panel" style={{ position: "absolute", left: 16, bottom: 16, width: MM_W + 2, height: mh + 2, border: "1px solid var(--line)", overflow: "hidden" }}>
      <canvas ref={cv} aria-label="Minimap" style={{ display: "block", width: MM_W, height: mh, cursor: "crosshair", touchAction: "none" }}
        onPointerDown={(e) => { e.stopPropagation(); e.currentTarget.setPointerCapture(e.pointerId); centreAt(e); }}
        onPointerMove={(e) => { e.stopPropagation(); if (e.currentTarget.hasPointerCapture(e.pointerId)) centreAt(e); }} />
      <div style={{
        position: "absolute", pointerEvents: "none", border: "1px solid var(--accent)",
        left: ox + vr.x * s, top: oy + vr.y * s, width: Math.max(2, vr.w * s), height: Math.max(2, vr.h * s),
      }} />
    </div>
  );
}
