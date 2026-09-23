import { afterEach, describe, expect, it, vi } from "vitest";
import { progressText, runJob } from "./jobs";
import type { Job, JobRequest } from "./types";

const job = (o: Partial<Job>): Job => ({
  id: "job_000001", kind: "attribution", run_id: "run_0001", status: "queued", progress: 0, message: "",
  result: null, error: null, created: 0, ...o,
});
const body: JobRequest = { kind: "attribution", method: "gxa", target: null, objective: null, steps: 32 };

/** fetch stub: POST /jobs → first, GET /jobs/<id> → each of `gets` in turn; records every call. */
function stub(first: Job, gets: Job[] = []) {
  const calls: { url: string; method: string }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    calls.push({ url, method });
    const j = url.endsWith("/cancel") ? job({ status: "cancelled" }) : method === "POST" ? first : gets.length > 1 ? gets.shift() : gets[0];
    return new Response(JSON.stringify(j), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
  return calls;
}

afterEach(() => { vi.unstubAllGlobals(); });

describe("progressText", () => {
  it("covers every status", () => {
    expect(progressText(job({ status: "queued" }))).toBe("Queued");
    expect(progressText(job({ status: "running", progress: 0.42, message: "IG batch 3/8" }))).toBe("42% · IG batch 3/8");
    expect(progressText(job({ status: "running", progress: 0.1 }))).toBe("10%");
    expect(progressText(job({ status: "done", progress: 1 }))).toBe("Done");
    expect(progressText(job({ status: "cancelled" }))).toBe("Cancelled");
    expect(progressText(job({ status: "error", error: "boom" }))).toBe("Error · boom");
  });
});

describe("runJob without EventSource", () => {
  it("polls until done and resolves the result", async () => {
    expect(typeof EventSource).toBe("undefined");
    const calls = stub(job({ status: "running", progress: 0.3 }), [job({ status: "done", progress: 1, result: { ok: 1 } })]);
    const seen: string[] = [];
    await expect(runJob("run_0001", body, (j) => seen.push(j.status))).resolves.toEqual({ ok: 1 });
    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual(["POST /api/runs/run_0001/jobs", "GET /api/jobs/job_000001"]);
    expect(seen).toEqual(["running", "done"]);
  });
  it("rejects with the job's error", async () => {
    stub(job({ status: "running" }), [job({ status: "error", error: "layer1.0.relu is called 2 times" })]);
    await expect(runJob("run_0001", body)).rejects.toThrow("layer1.0.relu is called 2 times");
  });
  it("resolves a job born done without a GET", async () => {
    const calls = stub(job({ status: "done", progress: 1, result: 7 }));
    await expect(runJob("run_0001", body)).resolves.toBe(7);
    expect(calls).toHaveLength(1);
  });
  it("cancels server-side when aborted", async () => {
    const calls = stub(job({ status: "running" }), [job({ status: "running" })]);
    const ctl = new AbortController();
    const p = runJob("run_0001", body, () => ctl.abort(), ctl.signal);
    await expect(p).rejects.toThrow("Cancelled");
    await vi.waitFor(() => expect(calls.at(-1)).toEqual({ url: "/api/jobs/job_000001/cancel", method: "POST" }));
    expect(calls.some((c) => c.method === "GET")).toBe(false);
  });
});
