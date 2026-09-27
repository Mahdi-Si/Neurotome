import type {
  Attribution, CreateRun, DiffResult, Fold, GraphView, Health, IndexEntry, InputInfo, InterventionResult, InterventionSummary, Job, JobRequest, Method,
  MatrixView, ModelsInfo, ModuleEntry, Objective, OutputFull, RunMeta, SampleRef, SignalView, SliceResult, TargetWindow,
  TensorOpts, TensorView,
} from "./types";

/** A `detail` as one line: the string, or a 422's `[{loc, msg}]` as `steps: Input should be …` (never a JSON dump). */
export function detailText(d: unknown): string {
  if (typeof d === "string") return d;
  if (Array.isArray(d) && d.every((e) => typeof e?.msg === "string"))
    return d.map((e) => { const loc = Array.isArray(e.loc) ? e.loc.filter((x: unknown) => x !== "body").join(".") : ""; return loc ? `${loc}: ${e.msg}` : e.msg; }).join(" · ");
  return JSON.stringify(d);
}

async function ok(res: Response): Promise<Response> {
  if (res.ok) return res;
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = detailText(body.detail);
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
/** `?key=value` params, dropping null / undefined values. */
const qs = (o: Record<string, string | null | undefined>) =>
  new URLSearchParams(Object.entries(o).filter((e): e is [string, string] => e[1] != null));

export const api = {
  health: () => req<Health>("/api/health"),
  models: () => req<ModelsInfo>("/api/models"),
  runs: () => req<RunMeta[]>("/api/runs"),
  samples: (model: string) => req<{ samples: SampleRef[]; error: string | null }>(`/api/samples?${qs({ model })}`),
  run: (id: string) => req<RunMeta>(`/api/runs/${enc(id)}`),
  createRun: (body: CreateRun) => req<Job<RunMeta>>("/api/runs", json(body)),   // 202 + capture job
  deleteRun: (id: string) => req<{ deleted: boolean }>(`/api/runs/${enc(id)}`, { method: "DELETE" }),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<{ upload_id: string; name: string }>("/api/uploads", { method: "POST", body: fd });
  },
  modules: (id: string) => req<ModuleEntry[]>(`/api/runs/${enc(id)}/modules`),
  graph: (id: string, depth: number, attr?: { method: Method; target?: number | null; steps?: number; objective?: Objective | null }, fold?: Fold, iv?: string | null) => {
    const q = new URLSearchParams({ depth: String(depth) });
    if (attr) {
      q.set("method", attr.method);
      if (attr.target != null) q.set("target", String(attr.target));
      if (attr.steps) q.set("steps", String(attr.steps));
      if (attr.objective) { q.set("obj_op", attr.objective.op); q.set("obj_ch", String(attr.objective.channel)); }
    }
    if (fold?.expand.length) q.set("expand", fold.expand.join(","));
    if (fold?.collapse.length) q.set("collapse", fold.collapse.join(","));
    if (iv) q.set("iv", iv);
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
    if (opts.iv) q.set("iv", opts.iv);
    return req<TensorView>(`/api/runs/${enc(id)}/ops/${enc(op)}/tensor?${q}`);
  },
  inputPreviewUrl: (id: string, created: number) => `/api/runs/${enc(id)}/input.png?v=${created}`,
  input: (id: string) => req<InputInfo>(`/api/runs/${enc(id)}/input`),
  inputRawUrl: (id: string, created: number) => `/api/runs/${enc(id)}/input/raw?v=${created}`,
  inputModelUrl: (id: string, created: number) => `/api/runs/${enc(id)}/input/model.png?v=${created}`,
  inputSignal: (id: string, points = 1500) => req<SignalView>(`/api/runs/${enc(id)}/input/signal?points=${points}`),
  inputMatrix: (id: string) => req<MatrixView>(`/api/runs/${enc(id)}/input/matrix`),
  output: (id: string) => req<OutputFull>(`/api/runs/${enc(id)}/output`),
  windows: (id: string, t: number) => req<{ windows: TargetWindow[]; error: string | null }>(`/api/runs/${enc(id)}/windows?t=${t}`),
  slice: (id: string, op: string, s: string, iv?: string | null) =>
    req<SliceResult>(`/api/runs/${enc(id)}/ops/${enc(op)}/slice?${qs({ s, iv })}`),
  sliceNpy: async (id: string, op: string, s: string, iv?: string | null) =>
    (await ok(await fetch(`/api/runs/${enc(id)}/ops/${enc(op)}/slice.npy?${qs({ s, iv })}`))).blob(),
  diff: (id: string, a: string, b: string, aIv?: string | null, bIv?: string | null) =>
    req<DiffResult>(`/api/runs/${enc(id)}/diff?${qs({ a, b, a_iv: aIv, b_iv: bIv })}`),
  interventions: (id: string) => req<InterventionSummary[]>(`/api/runs/${enc(id)}/interventions`),
  intervention: (id: string, hash: string) => req<InterventionResult>(`/api/runs/${enc(id)}/interventions/${enc(hash)}`),
  renameIntervention: (id: string, hash: string, name: string) =>
    req<InterventionResult>(`/api/runs/${enc(id)}/interventions/${enc(hash)}`, { ...json({ name }), method: "PATCH" }),
  deleteIntervention: (id: string, hash: string) =>
    req<{ deleted: boolean }>(`/api/runs/${enc(id)}/interventions/${enc(hash)}`, { method: "DELETE" }),
  attribution: (id: string, body: { method: Method; target: number | null; steps: number; objective?: Objective | null }) =>
    req<Attribution>(`/api/runs/${enc(id)}/attribution`, json(body)),
  startJob: (runId: string, body: JobRequest) => req<Job>(`/api/runs/${enc(runId)}/jobs`, json(body)),
  job: (jobId: string) => req<Job>(`/api/jobs/${enc(jobId)}`),
  cancelJob: (jobId: string) => req<Job>(`/api/jobs/${enc(jobId)}/cancel`, { method: "POST" }),
  jobEventsUrl: (jobId: string) => `/api/jobs/${enc(jobId)}/events`,
};
