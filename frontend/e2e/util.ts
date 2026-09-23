import { expect, type APIRequestContext } from "@playwright/test";
import type { Job, RunMeta } from "../src/lib/types";

export const MODEL = "py:examples.custom_model:build";   // 1×1×28×28 input, modules stem · blocks.0–2 · head

const base = () => {
  const b = process.env.NEUROTOME_E2E_URL;
  if (!b) throw new Error("NEUROTOME_E2E_URL is not set; run through playwright.config.ts (global setup)");
  return b;
};

/** The app at a hash route, e.g. url("#/runs/run_0001/graph?depth=1"). */
export const url = (hash = "#/") => `${base()}/${hash}`;

/** Capture a run through the API (POST /api/runs → capture job → poll to done) and return its meta. */
export async function createRun(request: APIRequestContext, body: Record<string, unknown> = {}): Promise<RunMeta> {
  const r = await request.post(`${base()}/api/runs`, { data: { model: MODEL, sample: { kind: "random" }, ...body } });
  expect(r.status(), await r.text()).toBe(202);
  let job = (await r.json()) as Job<RunMeta>;
  const deadline = Date.now() + 45_000;
  while (job.status === "queued" || job.status === "running") {
    if (Date.now() > deadline) throw new Error(`capture job ${job.id} still ${job.status}`);
    await new Promise((res) => setTimeout(res, 200));
    job = await (await request.get(`${base()}/api/jobs/${job.id}`)).json();
  }
  if (job.status !== "done" || !job.result) throw new Error(`capture job ${job.id} ${job.status}: ${job.error}`);
  return job.result;
}

/** GET a JSON API path (e.g. "/api/runs/<id>/graph?depth=12"). */
export async function getJson<T>(request: APIRequestContext, path: string): Promise<T> {
  const r = await request.get(`${base()}${path}`);
  expect(r.ok(), await r.text()).toBe(true);
  return r.json();
}
