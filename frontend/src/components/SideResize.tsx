// A draggable left edge for a view's right panel: drag it, or focus it and press ←/→ (Home or a double-click resets).
// The width is remembered per view in localStorage.
import { useRef, useState } from "react";

const MIN = 320;
const STEP = 24;

/** At least MIN px, at most 60% of the window (never below MIN). */
export const clampWidth = (w: number, viewport: number) => Math.round(Math.max(MIN, Math.min(w, viewport * 0.6)));

const read = (key: string, def: number) => {
  try { const v = Number(localStorage.getItem(key)); return v > 0 ? v : def; } catch { return def; }
};
const save = (key: string, v: number | null) => {
  try { if (v == null) localStorage.removeItem(key); else localStorage.setItem(key, String(v)); } catch { /* session only */ }
};

/** [width, handle]: render the handle just before the panel, in the same flex row, and give the panel `width`. */
export function useSideWidth(view: string, def: number): [number, React.ReactNode] {
  const key = `neurotome.side.${view}`;
  const [w, setW] = useState(() => clampWidth(read(key, def), window.innerWidth));
  const drag = useRef<{ x: number; w: number } | null>(null);
  const set = (v: number) => {
    const c = clampWidth(v, window.innerWidth);
    setW(c);
    save(key, c === def ? null : c);
  };
  const handle = (
    <div style={{ position: "relative", width: 0, flexShrink: 0, zIndex: 5 }}>
      <div role="separator" aria-orientation="vertical" aria-label="Resize panel" aria-valuenow={w} aria-valuemin={MIN}
        tabIndex={0} className="side-resize" title="Drag to resize · double-click to reset"
        onPointerDown={(e) => { e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId); drag.current = { x: e.clientX, w }; }}
        onPointerMove={(e) => { if (drag.current) set(drag.current.w + drag.current.x - e.clientX); }}
        onPointerUp={() => { drag.current = null; }}
        onPointerCancel={() => { drag.current = null; }}
        onDoubleClick={() => set(def)}
        onKeyDown={(e) => {
          const next = e.key === "ArrowLeft" ? w + STEP : e.key === "ArrowRight" ? w - STEP : e.key === "Home" ? def : null;
          if (next != null) { e.preventDefault(); set(next); }
        }} />
    </div>
  );
  return [w, handle];
}
