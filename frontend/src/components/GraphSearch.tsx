// Graph search box: name, type or shape over the run's search index (ops + multi-op module calls).
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { shapeShort } from "../lib/format";
import { searchIndex } from "../lib/graphState";
import type { IndexEntry } from "../lib/types";
import { useAsync } from "../lib/useAsync";
import { IconSearch } from "./Icons";

const typing = (el: Element | null) =>
  !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || (el as HTMLElement).isContentEditable);

export function GraphSearch({ runId, query, onQuery, onPick }: {
  runId: string; query: string; onQuery: (q: string) => void; onPick: (e: IndexEntry) => void;
}) {
  const index = useAsync(() => api.index(runId), [runId]);
  const input = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const results = useMemo(() => searchIndex(index.data ?? [], query), [index.data, query]);

  useEffect(() => setActive(0), [query]);
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const k = e.key === "/" || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k");
      if (k && !typing(document.activeElement)) { e.preventDefault(); input.current?.focus(); }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, []);

  const pick = (e: IndexEntry | undefined) => {
    if (!e) return;
    setOpen(false);
    input.current?.blur();
    onPick(e);
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setOpen(true);
      const n = results.length || 1;
      setActive((a) => (a + (e.key === "ArrowDown" ? 1 : n - 1)) % n);
    } else if (e.key === "Enter") {
      e.preventDefault();
      pick(results[active]);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onQuery("");
      input.current?.blur();
    }
  };

  const shown = open && query.trim() !== "";
  return (
    <div style={{ position: "absolute", left: 16, top: 52, width: 280, zIndex: 3 }}>
      <label className="field" style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--ink-2)", borderColor: open ? "var(--ink)" : undefined }}>
        <IconSearch />
        <input ref={input} value={query} placeholder="Search name, type or shape  /" aria-label="Search graph"
          role="combobox" aria-expanded={shown} aria-controls="graph-search-results" aria-autocomplete="list"
          aria-activedescendant={shown && results[active] ? `gs-${active}` : undefined}
          onChange={(e) => { onQuery(e.target.value); setOpen(true); }} onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)} onKeyDown={onKey}
          className="mono" style={{ border: 0, outline: 0, background: "transparent", width: "100%", color: "var(--ink)", padding: 0 }} />
      </label>
      {shown && (
        <div id="graph-search-results" role="listbox" className="panel" style={{ border: "1px solid var(--line)", borderTop: 0 }}>
          {index.error && <div className="lbl err" style={{ padding: "8px 10px" }}>{index.error}</div>}
          {!index.error && results.length === 0 && (
            <div className="lbl dim" style={{ padding: "8px 10px" }}>{index.loading ? "Loading index…" : "No matches"}</div>
          )}
          {results.map((r, i) => (
            <div key={r.id} id={`gs-${i}`} role="option" aria-selected={i === active}
              onMouseDown={(e) => e.preventDefault()} onClick={() => pick(r)} onMouseEnter={() => setActive(i)}
              style={{ padding: "6px 10px", cursor: "pointer", borderBottom: "1px solid var(--line-soft)", background: i === active ? "var(--accent-soft)" : undefined }}>
              <div style={{ display: "flex", gap: 8, alignItems: "baseline", minWidth: 0 }}>
                <span className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.label}</span>
                <span className="lbl dim" style={{ flexGrow: 1, whiteSpace: "nowrap" }}>{r.type}</span>
                <span className="mono dim" style={{ whiteSpace: "nowrap" }}>{shapeShort(r.shape)}</span>
              </div>
              <div className="mono dim" style={{ fontSize: 11, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.path}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
