import type {
  Attribution, CreateRun, DiffResult, Fold, GraphView, Health, IndexEntry, Job, JobRequest, Method, ModelsInfo, ModuleEntry, Objective, RunMeta,
  SliceResult,
  TensorOpts, TensorView,
} from "./types";

async function ok(res: Response): Promise<Response> {
  if (res.ok) return res;
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch { /* not json */ }
  throw new Error(`${res.status} · ${detail}`);
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  return (await ok(await fetch(path, init))).json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const enc = encodeURIComponent;

export const api = {
  health: () => req<Health>("/api/health"),
  models: () => req<ModelsInfo>("/api/models"),
  runs: () => req<RunMeta[]>("/api/runs"),
  run: (id: string) => req<RunMeta>(`/api/runs/${enc(id)}`),
  createRun: (body: CreateRun) => req<Job<RunMeta>>("/api/runs", json(body)),   // 202 + capture job
  deleteRun: (id: string) => req<{ deleted: boolean }>(`/api/runs/${enc(id)}`, { method: "DELETE" }),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<{ upload_id: string; name: string }>("/api/uploads", { method: "POST", body: fd });
  },
  modules: (id: string) => req<ModuleEntry[]>(`/api/runs/${enc(id)}/modules`),
  graph: (id: string, depth: number, attr?: { method: Method; target?: number | null; steps?: number; objective?: Objective | null }, fold?: Fold) => {
    const q = new URLSearchParams({ depth: String(depth) });
    if (attr) {
      q.set("method", attr.method);
      if (attr.target != null) q.set("target", String(attr.target));
      if (attr.steps) q.set("steps", String(attr.steps));
      if (attr.objective) { q.set("obj_op", attr.objective.op); q.set("obj_ch", String(attr.objective.channel)); }
    }
    if (fold?.expand.length) q.set("expand", fold.expand.join(","));
    if (fold?.collapse.length) q.set("collapse", fold.collapse.join(","));
    return req<GraphView>(`/api/runs/${enc(id)}/graph?${q}`);
  },
  index: (id: string) => req<IndexEntry[]>(`/api/runs/${enc(id)}/index`),
  tensor: (id: string, op: string, opts: TensorOpts = {}) => {
    const q = new URLSearchParams();
    if (opts.channel != null) q.set("channel", String(opts.channel));
    if (opts.py != null && opts.px != null) { q.set("py", String(opts.py)); q.set("px", String(opts.px)); }
    if (opts.norm === "global") q.set("norm", "global");
    if (opts.cmap === "div") q.set("cmap", "div");
    if (opts.lo != null && opts.hi != null) { q.set("lo", String(opts.lo)); q.set("hi", String(opts.hi)); }
    return req<TensorView>(`/api/runs/${enc(id)}/ops/${enc(op)}/tensor?${q}`);
  },
  inputPreviewUrl: (id: string, created: number) => `/api/runs/${enc(id)}/input.png?v=${created}`,
  slice: (id: string, op: string, s: string) =>
    req<SliceResult>(`/api/runs/${enc(id)}/ops/${enc(op)}/slice?${new URLSearchParams({ s })}`),
  sliceNpy: async (id: string, op: string, s: string) =>
    (await ok(await fetch(`/api/runs/${enc(id)}/ops/${enc(op)}/slice.npy?${new URLSearchParams({ s })}`))).blob(),
  diff: (id: string, a: string, b: string) => req<DiffResult>(`/api/runs/${enc(id)}/diff?${new URLSearchParams({ a, b })}`),
  attribution: (id: string, body: { method: Method; target: number | null; steps: number; objective?: Objective | null }) =>
    req<Attribution>(`/api/runs/${enc(id)}/attribution`, json(body)),
  startJob: (runId: string, body: JobRequest) => req<Job>(`/api/runs/${enc(runId)}/jobs`, json(body)),
  job: (jobId: string) => req<Job>(`/api/jobs/${enc(jobId)}`),
  cancelJob: (jobId: string) => req<Job>(`/api/jobs/${enc(jobId)}/cancel`, { method: "POST" }),
  jobEventsUrl: (jobId: string) => `/api/jobs/${enc(jobId)}/events`,
};
