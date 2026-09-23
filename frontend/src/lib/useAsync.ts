import { useEffect, useState } from "react";
import { errText } from "./tensorState";

export interface Async<T> { data: T | null; error: string | null; loading: boolean }

/** Run an async loader whenever deps change; stale results are dropped. */
export function useAsync<T>(fn: () => Promise<T> | null, deps: unknown[]): Async<T> {
  const [state, setState] = useState<Async<T>>({ data: null, error: null, loading: true });
  useEffect(() => {
    let alive = true;
    const p = fn();
    if (!p) { setState({ data: null, error: null, loading: false }); return; }
    setState((s) => ({ data: s.data, error: null, loading: true }));
    p.then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e) => alive && setState({ data: null, error: errText(e), loading: false }));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
