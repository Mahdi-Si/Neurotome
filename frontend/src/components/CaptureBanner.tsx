// One quiet row under the TopBar when a run's tensors were only partly saved (budget or module selection).
import { captureNote } from "../lib/capture";
import type { RunMeta } from "../lib/types";

export function CaptureBanner({ run }: { run: RunMeta }) {
  const note = captureNote(run);
  if (!note) return null;
  return (
    <div className="panel" role="note" style={{ height: 32, flexShrink: 0, display: "flex", alignItems: "center", gap: 10, padding: "0 20px", borderBottom: "1px solid var(--line)", minWidth: 0 }}>
      <span className="lbl pill">{note.tag}</span>
      <span className="lbl dim" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={note.text}>{note.text}</span>
    </div>
  );
}
