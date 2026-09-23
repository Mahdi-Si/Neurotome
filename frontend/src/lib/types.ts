// Types mirroring the Python API (backend/neurotome/api.py). Keep in sync with docs/API.md.

export interface Stats {
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  zero_frac: number | null;
  hist: number[];
  hist_range: [number, number];
  numel: number;
  nan?: number; // absent on pre-M1 runs
  inf?: number;
}

export interface TopK { index: number; prob: number; logit: number; label: string | null }

export type Unsaved = "budget" | "modules" | "no_tensor";

export interface RunMeta {
  id: string;
  created: number;
  model: {
    spec: string | null; weights: string | null; class_name: string; num_params: number;
    num_ops: number; flops: number; activation_bytes: number; truncated: boolean;
    budget_bytes?: number | null; capture_modules?: string[] | null; saved_ops?: number; // absent before M4
  };
  sample: { kind: string; name: string };
  input_shape: number[] | null;
  output: TopK[];
  categories: string[] | null;
  reloadable: boolean;
  status: string;
  preview?: { width: number; height: number } | null; // absent on pre-M2 runs
  output_error?: string | null; // absent before M4
}

export interface GraphNode {
  id: string;
  kind: "op" | "module";
  label: string;
  path: string;
  type: string;
  num_ops: number;
  num_params: number;
  output_op: string;
  shape: number[];
  flops: number;
  stats: Stats | null;
  group: string | null;
  is_input: boolean;
  is_output: boolean;
  score?: number | null;
  score_norm?: number | null;
  dtype?: string | null;
  device?: string | null;
  nonfinite?: number | null;
  unsaved?: Unsaved | null; // null = tensor saved; absent (old server) → saved iff stats
}

export interface GraphEdge {
  source: string; target: string; shape: number[];
  dtype?: string | null; device?: string | null; target_dtype?: string | null; target_device?: string | null;
  share?: number; on_path?: boolean; share_method?: "merge" | "parent";
}

export interface IndexEntry {
  id: string; kind: "op" | "module"; label: string; path: string; type: string; shape: number[]; calls: string[];
}

/** Per-module fold overrides, as module calls, e.g. "layer1.0:1". */
export interface Fold { expand: string[]; collapse: string[] }

export type ColorMode = "plain" | "attr" | "norm" | "sparsity" | "nan" | "patch";

/** Per-node colour-mode value, computed by the view and drawn by GraphCanvas. */
export interface NodeTone { v: number | null; flag?: boolean; readout: string } // v in [0,1] → tint + bar; flag → negative outline + badge

export interface GraphGroup {
  id: string; address: string; class_name: string; num_params: number; parent: string | null; num_ops: number;
}

export interface GraphView {
  depth: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  groups: GraphGroup[];
  path?: string[];
}

export interface ModuleEntry { address: string; name: string; class_name: string; depth: number; num_params: number }

export interface Hist { counts: number[]; range: [number, number] }
export interface Pca { points: [number, number][]; color: number[]; explained: number[] }
export interface ViewStats { mean: number; std: number; min: number; max: number }

export type Norm = "channel" | "global";
export type Cmap = "seq" | "div";

export interface ImageTensorView {
  kind: "image";
  shape: number[];
  channels: number; height: number; width: number;
  channel: number;
  channel_rank: number;
  maps: { channel: number; mean: number; map: number[][]; range: [number, number] }[];
  norm: Norm;
  cmap: Cmap;
  scale: [number, number] | null; // global range used, null in per-channel mode
  dots_range: [number, number];
  dots: number[][];
  stats: ViewStats;
  peak: [number, number]; // argmax of the selected channel; the inspected position is vector.pixel
  profile: { row: number; values: number[] };
  hist: Hist;
  vector: { pixel: [number, number]; values: number[] };
  pca: Pca;
}

export interface SequenceTensorView {
  kind: "sequence";
  shape: number[];
  tokens: number; features: number;
  heatmap: number[][];
  scale: [number, number] | null;   // shared [lo, hi] the heatmap was divided by max(|lo|, |hi|) with, else per-tensor max-abs
  token_norms: number[];
  hist: Hist;
  stats: ViewStats;
  pca: Pca;
}

export interface VectorTensorView {
  kind: "vector";
  shape: number[];
  length: number;
  values: number[];
  top: { index: number; value: number }[];
  hist: Hist;
  stats: ViewStats;
}

export interface TensorOpts {
  channel?: number | null; py?: number | null; px?: number | null;
  norm?: Norm; cmap?: Cmap; lo?: number | null; hi?: number | null;
}

export type SliceValue = number | "nan" | "inf" | "-inf";
export interface SliceResult {
  op: string; expr: string; key: string;
  shape: number[]; full_shape: number[]; truncated: boolean;
  axes: number[]; indices: number[][]; values: SliceValue[];
}

export interface DiffResult {
  a: string; b: string; shape_a: number[]; shape_b: number[]; same_shape: boolean;
  diff: {
    stats: { mean: number | null; std: number | null; min: number | null; max: number | null };
    hist: Hist; max_abs: number | null; mean_abs: number | null; cosine: number | null; rel_l2: number | null;
  } | null;
}

export type TensorView = ImageTensorView | SequenceTensorView | VectorTensorView;

export type Method = "gxa" | "saliency" | "ig";

export type LayerMethod = "layer_ig" | "layer_cond";
export type Baseline = "zeros" | "mean" | "run";
/** An internal unit as the attribution objective: Σ over positions of channel `channel` (axis 1) of op `op`. */
export interface Objective { op: string; channel: number }
export type ObjectiveInfo = { kind: "logit"; index: number } | { kind: "unit"; op: string; channel: number };

export interface Attribution {
  method: Method; method_label: string; node_method: string; merge_method: string;
  objective: ObjectiveInfo; objective_label: string;
  target: number | null; target_label: string | null;
  prob: number | null; logit: number | null; value: number;
  steps: number | null;
  input: { map: number[][]; total: number; convergence_delta: number | null } | null;
  scores: Record<string, number>;
  top_units: { op: string; path?: string; channel: number; value: number }[];
  input_note?: string | null;
}

export interface LayerResult {
  method: LayerMethod; method_label: string; op: string; path: string; objective_label: string; steps: number;
  channels: number[]; top: { channel: number; value: number }[]; total: number; convergence_delta: number | null;
}

export interface NeuronResult {
  method: "neuron_cond"; method_label: string; op: string; path: string; channel: number; objective_label: string; steps: number;
  input: { map: number[][]; total: number; convergence_delta: null };
}

export interface AgreementBlock {
  basis: string; methods: string[]; labels: string[]; n: number; k: number; op?: string;
  jaccard: number[][]; overlap: number[][]; spearman: (number | null)[][];
}
export interface AgreementResult {
  objective_label: string; steps: number; input: AgreementBlock | null; units: AgreementBlock | null; units_error: string | null;
}

export interface PatchResult {
  op: string; path: string; baseline: Baseline; source_run: string | null; method_label: string;
  target: number; target_label: string | null;
  before: { logit: number; prob: number }; after: { logit: number; prob: number };
  delta_logit: number; delta_prob: number; top_after: TopK[];
  deltas: Record<string, number>; rel: Record<string, number>;   // keyed by op id (node.output_op)
}

export type JobKind = "capture" | "attribution" | "layer" | "neuron" | "agreement" | "patch";
export type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";
export interface Job<R = unknown> {
  id: string; kind: JobKind; run_id: string | null; status: JobStatus;
  progress: number; message: string; result: R | null; error: string | null; created: number;
}
export type JobRequest =
  | { kind: "attribution"; method: Method; target: number | null; objective: Objective | null; steps: number }
  | { kind: "layer"; method: LayerMethod; op: string; target: number | null; objective: Objective | null; steps: number }
  | { kind: "neuron"; op: string; channel: number; target: number | null; objective: Objective | null; steps: number }
  | { kind: "agreement"; op: string | null; target: number | null; objective: Objective | null; steps: number }
  | { kind: "patch"; op: string; baseline: Baseline; source_run: string | null; target: number | null };

export interface CreateRun {
  model: string; weights: string | null; sample: Record<string, unknown>;
  modules: string[] | null; max_capture_mb: number | null;
}
export interface Health { ok: boolean; device: string; torch: string; cuda: boolean; runs_dir: string; version: string; max_capture_mb: number | null }
