// Token attribution as text: one mono span per input token, shaded accent / negative by |score|, hover / focus readout.
// No PNG export: it is text (DESIGN.md notes the exception).
import { useState } from "react";
import { showToken, stripReadout, tokenTone } from "../lib/tokens";
import { Readout } from "./charts";

export function TokenStrip({ tokens, scores, signed }: { tokens: string[]; scores: number[]; signed: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(0, ...scores.map(Math.abs));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div role="group" aria-label="Token attribution" onMouseLeave={() => setHover(null)}
        style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
        {scores.map((v, i) => {
          const { color, alpha } = tokenTone(v, max, signed);
          return (
            <span key={i} className="mono" tabIndex={0} aria-label={stripReadout(tokens, scores, signed, i)}
              onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} onBlur={() => setHover(null)}
              style={{
                fontSize: 13, padding: "2px 3px", whiteSpace: "pre",
                border: `1px solid ${hover === i ? "var(--ink)" : "var(--line-soft)"}`,
                background: `color-mix(in srgb, var(--${color}) ${Math.round(alpha * 100)}%, transparent)`,
              }}>{showToken(tokens[i] ?? "")}</span>
          );
        })}
      </div>
      <Readout text={hover != null ? stripReadout(tokens, scores, signed, hover) : null} />
    </div>
  );
}
