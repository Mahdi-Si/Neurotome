import { useEffect, useState } from "react";

// Tiny hash router. Routes:
//   #/                               home
//   #/runs/<id>/<tab>?node=<nodeId>  tab = graph | tensors | trace | lens | interventions
export type Tab = "graph" | "tensors" | "trace" | "lens" | "interventions";

export interface Route {
  runId: string | null;
  tab: Tab;
  params: URLSearchParams;
}

/** decodeURIComponent that keeps a malformed escape (`%E0%A4%A`) as typed: it throws, and parseHash must not. */
const decode = (s: string) => { try { return decodeURIComponent(s); } catch { return s; } };

export function parseHash(hash: string): Route {
  const [path, query = ""] = hash.replace(/^#/, "").split("?");
  const parts = path.split("/").filter(Boolean);
  const params = new URLSearchParams(query);
  if (parts[0] === "runs" && parts[1]) {
    const tab = (["graph", "tensors", "trace", "lens", "interventions"].includes(parts[2]) ? parts[2] : "graph") as Tab;
    return { runId: decode(parts[1]), tab, params };
  }
  return { runId: null, tab: "graph", params };
}

export function href(runId: string | null, tab: Tab = "graph", params: Record<string, string | null | undefined> = {}): string {
  if (!runId) return "#/";
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v != null && v !== "") q.set(k, v);
  const qs = q.toString();
  return `#/runs/${encodeURIComponent(runId)}/${tab}${qs ? `?${qs}` : ""}`;
}

export function navigate(to: string, replace = false): void {
  if (replace) history.replaceState(null, "", to);
  else history.pushState(null, "", to);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(location.hash));
    window.addEventListener("hashchange", on);
    window.addEventListener("popstate", on);
    return () => {
      window.removeEventListener("hashchange", on);
      window.removeEventListener("popstate", on);
    };
  }, []);
  return route;
}
