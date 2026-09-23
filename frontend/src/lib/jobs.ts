// Client side of the in-process job runner: follow one job to its end (SSE, else polling), one job slot per panel.
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { errText } from "./tensorState";
import type { Job, JobRequest, JobStatus } from "./types";

export const isTerminal = (s: JobStatus) => s === "done" || s === "error" || s === "cancelled";

/** "Queued" | "42% · IG batch 3/8" | "Done" | "Cancelled" | "Error · <message>" */
export function progressText(job: Job): string {
  switch (job.status) {
    case "queued": return "Queued";
    case "running": return `${Math.round(job.progress * 100)}%${job.message ? ` · ${job.message}` : ""}`;
    case "done": return "Done";
    case "cancelled": return "Cancelled";
    case "error": return `Error · ${job.error || "failed"}`;
  }
}

const POLL_MS = 400;

/** Start a job and follow it to the end: POST, then SSE (EventSource) or, without EventSource or after an SSE
 * error, polling GET every 400 ms. Resolves the result when done; rejects with Error(job.error) on error and
 * Error("Cancelled") when cancelled. Aborting `signal` cancels the job server-side and rejects with Error("Cancelled"). */
export function runJob<R>(runId: string, body: JobRequest, onUpdate?: (job: Job<R>) => void, signal?: AbortSignal): Promise<R> {
  return new Promise<R>((resolve, reject) => {
    if (signal?.aborted) { reject(new Error("Cancelled")); return; }
    let id: string | null = null, es: EventSource | null = null, over = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const end = () => { over = true; es?.close(); clearTimeout(timer); signal?.removeEventListener("abort", abort); };
    const fail = (e: unknown) => { if (!over) { end(); reject(e instanceof Error ? e : new Error(String(e))); } };
    const abort = () => { if (id) api.cancelJob(id).catch(() => {}); fail(new Error("Cancelled")); };
    const update = (j: Job<R>) => {
      if (over) return;
      onUpdate?.(j);
      if (!isTerminal(j.status)) return;
      end();
      if (j.status === "done") resolve(j.result as R);
      else reject(new Error(j.status === "cancelled" ? "Cancelled" : j.error || "Job failed"));
    };
    const poll = () => {
      timer = setTimeout(() => api.job(id!).then((j) => { update(j as Job<R>); if (!over) poll(); }, fail), POLL_MS);
    };
    signal?.addEventListener("abort", abort);
    api.startJob(runId, body).then((j) => {
      if (over) { api.cancelJob(j.id).catch(() => {}); return; }   // aborted while the POST was in flight
      id = j.id;
      update(j as Job<R>);
      if (over) return;
      if (typeof EventSource === "undefined") { poll(); return; }
      es = new EventSource(api.jobEventsUrl(j.id));
      es.onmessage = (e) => update(JSON.parse(e.data) as Job<R>);
      es.onerror = () => { es?.close(); es = null; if (!over) poll(); };
    }, fail);
  });
}

/** One job slot for a panel: run() cancels a job still running in this slot (superseded), then starts the new one.
 * run() resolves null when cancelled or superseded; `error` holds the message of a failed job.
 * An identical request while one is in flight joins it instead (the server would dedup it anyway). */
export function useJob<R>(): {
  job: Job<R> | null; busy: boolean; error: string | null;
  run: (runId: string, body: JobRequest) => Promise<R | null>;
  cancel: () => void;
} {
  const [job, setJob] = useState<Job<R> | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const slot = useRef<{ key: string; ctl: AbortController; p: Promise<R | null> } | null>(null);
  const live = useRef(true);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
      const s = slot.current;
      setTimeout(() => { if (!live.current) s?.ctl.abort(); });   // a real unmount, not StrictMode's remount
    };
  }, []);

  const run = useCallback((runId: string, body: JobRequest): Promise<R | null> => {
    const key = JSON.stringify([runId, body]);
    if (slot.current?.key === key && !slot.current.ctl.signal.aborted) return slot.current.p;
    slot.current?.ctl.abort();
    const ctl = new AbortController();
    const mine = () => slot.current?.ctl === ctl;
    setJob(null); setError(null); setBusy(true);
    const p = runJob<R>(runId, body, (j) => { if (mine()) setJob(j); }, ctl.signal)
      .catch((e: unknown) => {
        const msg = errText(e);
        if (mine()) {
          if (msg === "Cancelled") setJob((j) => j && { ...j, status: "cancelled" });
          else setError(msg);
        }
        return null;
      })
      .finally(() => { if (mine()) { slot.current = null; setBusy(false); } });
    slot.current = { key, ctl, p };
    return p;
  }, []);

  const cancel = useCallback(() => slot.current?.ctl.abort(), []);
  return { job, busy, error, run, cancel };
}
