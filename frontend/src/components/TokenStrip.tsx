// Token attribution as text: one mono span per input token, shaded accent / negative by |score|, hover / focus readout.
// With `onToggle` the tokens are toggle buttons (selection mode: accent-soft ground + accent border when selected).
// No PNG export: it is text (DESIGN.md notes the exception).
import { useState } from "react";
import { showToken, stripReadout, tokenTone } from "../lib/tokens";
import { Readout } from "./charts";

export function TokenStrip({ tokens, scores, signed, selected, onToggle, label = "Token attribution" }: {
  tokens: string[]; scores: number[]; signed: boolean;
  selected?: number[]; onToggle?: (i: number) => void; label?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(0, ...scores.map(Math.abs));
  const on = (i: number) => !!selected?.includes(i);
  const text = (i: number) => onToggle
    ? `#${i} "${showToken(tokens[i] ?? "")}"${on(i) ? " · selected" : ""}`
    : stripReadout(tokens, scores, signed, i);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div role="group" aria-label={label} onMouseLeave={() => setHover(null)}
        style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
        {scores.map((v, i) => {
          const { color, alpha } = tokenTone(v, max, signed);
          const props = {
            className: "mono", "aria-label": text(i),
            onMouseEnter: () => setHover(i), onFocus: () => setHover(i), onBlur: () => setHover(null),
            style: {
              fontSize: 13, padding: "2px 3px", whiteSpace: "pre",
              border: `1px solid ${on(i) ? "var(--accent)" : hover === i ? "var(--ink)" : "var(--line-soft)"}`,
              background: on(i) ? "var(--accent-soft)" : `color-mix(in srgb, var(--${color}) ${Math.round(alpha * 100)}%, transparent)`,
            } as React.CSSProperties,
          };
          return onToggle
            ? <button key={i} type="button" aria-pressed={on(i)} onClick={() => onToggle(i)} {...props} style={{ ...props.style, cursor: "pointer" }}>{showToken(tokens[i] ?? "")}</button>
            : <span key={i} tabIndex={0} {...props}>{showToken(tokens[i] ?? "")}</span>;
        })}
      </div>
      <Readout text={hover != null ? text(hover) : null} />
    </div>
  );
}
