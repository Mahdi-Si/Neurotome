// A map drawn over the captured input preview (input_preview.png), plus its stateless controls.
import { useRef, useState } from "react";
import { api } from "../lib/api";
import { exportLayers } from "../lib/exportPng";
import { num } from "../lib/format";
import type { RunMeta } from "../lib/types";
import { Axes, HeatCanvas, Readout, axisText, cellAt, pngName, type AxisRange } from "./charts";

export function InputOverlay({ run, grid, signed, alpha, label, exportName, onPick, mark, readout, x, y }: {
  run: RunMeta;
  grid: number[][];
  signed: boolean;
  alpha: number;
  label: string;
  exportName?: string;
  onPick?: (r: number, c: number) => void;
  mark?: [number, number] | null;
  readout?: (r: number, c: number, v: number) => string;
  x?: AxisRange; y?: AxisRange;   // the map's real column / row ranges
}) {
  const img = useRef<HTMLImageElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [small, setSmall] = useState(false);
  const rows = grid.length, cols = grid[0]?.length ?? 0;
  const hv = hover ? grid[hover[0]]?.[hover[1]] : undefined;
  const text = (r: number, c: number, v: number) => (readout ? readout(r, c, v) : `cell (${r}, ${c}) · ${num(v, 3)}`);
  const onExport = async () => {
    const map = box.current?.querySelector("canvas");
    if (img.current && map) await exportLayers([{ el: img.current, alpha: 1 }, { el: map, alpha }], pngName(exportName, label), axisText({ x, y }));
  };
  return (
    <div>
      <Axes x={x} y={y}>
      <div ref={box} style={{ position: "relative" }}
        onMouseMove={(e) => rows && cols && setHover(cellAt(e, rows, cols))} onMouseLeave={() => setHover(null)}>
        <img ref={img} src={api.inputPreviewUrl(run.id, run.created)} alt="Input preview"
          onLoad={(e) => setSmall(e.currentTarget.naturalWidth < 112)}
          style={{ width: "100%", display: "block", imageRendering: small ? "pixelated" : "auto" }} />
        <div style={{ position: "absolute", inset: 0 }}>
          <HeatCanvas grid={grid} signed={signed} overlay label={label} onPick={onPick} mark={mark}
            style={{ height: "100%", opacity: alpha, imageRendering: "auto" }} />
        </div>
      </div>
      </Axes>
      <Readout text={hover && hv != null ? text(hover[0], hover[1], hv) : null} label={label} onExport={onExport} />
    </div>
  );
}

export function OverlayControls({ run, on, onOn, alpha, onAlpha }: {
  run: RunMeta; on: boolean; onOn: (on: boolean) => void; alpha: number; onAlpha: (a: number) => void;
}) {
  const has = !!run.preview;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
      <div className="seg lbl" role="group" aria-label="Map display">
        <button type="button" aria-pressed={!on} onClick={() => onOn(false)}>Dots</button>
        <button type="button" aria-pressed={on} disabled={!has} onClick={() => onOn(true)}
          title={has ? undefined : "No input preview for this run · re-capture to add one"}>Over input</button>
      </div>
      {on && (
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="lbl dim">Opacity</span>
          <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => onAlpha(Number(e.target.value))}
            style={{ accentColor: "var(--accent)", width: 96 }} />
          <span className="num" style={{ fontSize: 11, width: 32, textAlign: "right" }}>{Math.round(alpha * 100)}%</span>
        </label>
      )}
    </div>
  );
}
