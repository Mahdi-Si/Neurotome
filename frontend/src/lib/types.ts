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

export interface TopK { index: number; prob: number | null; logit: number | null; label: string | null }  // null: non-finite

export type Backend = "torchlens" | "nnsight";

export type Unsaved = "budget" | "modules" | "no_tensor";

export interface RunMeta {
  id: string;
  created: number;
  model: {
    spec: string | null; weights: string | null; class_name: string; num_params: number;
    num_ops: number; flops: number; activation_bytes: number; truncated: boolean;
    budget_bytes?: number | null; capture_modules?: string[] | null; saved_ops?: number; // absent before M4
    backend?: Backend; granularity?: "op" | "module"; lm?: boolean; // absent before v0.2 = torchlens, op, false
  };
  sample: {
    kind: string; name: string; text?: string; token_ids?: number[]; tokens?: string[];
    ref?: string; info?: Record<string, string | number | null>;   // kind "ref": a py: module's load_sample(ref)
  };
  input_shape: number[] | null;
  output: TopK[];
  categories: string[] | null;
  reloadable: boolean;
  status: string;
  preview?: { width: number; height: number } | null; // absent on pre-M2 runs
  input?: { raw: RawInput | null; steps: string[] | null; norm: number[][] | null; blocks?: InputBlock[] | null; time?: InputTime | null }; // absent before 0.3.0
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

export type ColorMode = "plain" | "attr" | "norm" | "sparsity" | "nan" | "iv";

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
/** Over finite values; null when the tensor holds none. */
/** The uploaded image as it was, kept with the run (`GET /api/runs/{id}/input/raw`). */
export interface ImageRaw { kind?: undefined; file: string; width: number; height: number; format: string; bytes: number }
/** A dataset sample's raw signal (`load_sample` raw), kept as input_raw.npz; t in seconds, sample i at t0 + i / fs. */
export interface SignalRaw {
  kind: "signal"; file: string; fs: number; t0: number; time_label: string;
  channels: { name: string; unit: string | null; length: number }[];
}
export type RawInput = ImageRaw | SignalRaw;
/** Column group [start, stop) of a (1, T, C) input's last axis, with a label per column. */
export interface InputBlock { name: string; start: number; stop: number; labels: string[] | null; unit: string | null }
/** Input axis `axis` is time: step i sits at t0 + i · step_s on the raw signal's clock. */
export interface InputTime { axis: number; step_s: number; t0: number }
/** The Input view (`GET /api/runs/{id}/input`): the sample, its preprocessing and the model input it became. */
export interface InputInfo {
  op: string; shape: number[]; dtype: string | null; sample: RunMeta["sample"];
  raw: RawInput | null;          // null: not an uploaded image / dataset signal, or captured before 0.3.0
  steps: string[] | null;        // the preprocessing in words; null before 0.3.0 and for in-process runs
  normalised: boolean;           // the model-input picture undoes (mean, std); else it is min-max stretched
  picture: boolean;              // a (1, 1|3, H, W) input: /input/model.png draws it
  // one row per channel of an image, per block of a (1, T, C) input with blocks (named), else one row
  channels: { channel: number | null; name?: string; mean: number | null; std: number | null; min: number | null; max: number | null }[];
  blocks?: InputBlock[] | null; time?: InputTime | null; info?: Record<string, string | number | null> | null;   // absent: older server
}
/** `GET /api/runs/{id}/input/signal`: each raw channel pooled to ≤ points (stride raw samples per point). */
export interface SignalView {
  fs: number; t0: number; time_label: string; stride: number;
  channels: { name: string; unit: string | null; length: number; stride: number; min: (number | null)[]; max: (number | null)[]; mean: (number | null)[] }[];
}
/** `GET /api/runs/{id}/input/matrix`: a (1, T, C) input per block, values[column][time col] mean-pooled by stride. */
export interface MatrixView {
  time: InputTime | null; steps: number; stride: number;
  blocks: { name: string; unit: string | null; labels: string[] | null; start: number; lo: number | null; hi: number | null; values: (number | null)[][] }[];
}
/** A dataset sample of a py: module (`GET /api/samples?model=`). */
export interface SampleRef { ref: string; label: string; group: string | null }
/** `GET /api/runs/{id}/output`: every output element; above 4096 the max of each `stride` block and no labels. */
export interface OutputFull { n: number; stride: number; values: (number | null)[]; labels: (string | null)[] | null }
export type WindowKind = "target" | "context" | "horizon";
/** A py: module's `target_windows(t)`, checked: steps [start, stop) of the input's time axis, its blocks (null = all). */
export interface TargetWindow { label: string; start: number; stop: number; blocks: string[] | null; kind: WindowKind }

export interface ViewStats { mean: number | null; std: number | null; min: number | null; max: number | null }

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
  top: { index: number; value: number | null }[];   // null: a non-finite value
  hist: Hist;
  stats: ViewStats;
}

export interface TensorOpts {
  channel?: number | null; py?: number | null; px?: number | null;
  norm?: Norm; cmap?: Cmap; lo?: number | null; hi?: number | null;
  iv?: string | null;
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

export interface AttentionHead {
  head: number; entropy: number;          // mean over live (not fully masked) query rows, bits
  entropy_norm: number;                   // entropy / log2 K, 0 … 1
  sparsity: number; masked: number;       // fraction of exact zeros in live rows; fraction of all-zero (masked) rows
  argmax: number; argmax_weight: number;  // the key with the highest mean weight (key profile peak) and that weight
  // square maps only (Q = K): mean weight on q-1 (q ≥ 1), on key 0, on key q; the one > 0.5, else null
  prev?: number; first?: number; self?: number;
  pattern?: "previous" | "first" | "self" | null;
}
export interface AttentionTensorView {
  kind: "attention"; shape: number[];
  layout: "hqk" | "qhk"; square: boolean;               // qhk: heads in the middle, e.g. (time, head, lag)
  heads: number; tokens: number; head: number;          // head = selected (the `channel` param, clamped; default 0)
  queries: number; keys: number;                        // tokens = queries (older readers)
  maps: { head: number; map: number[][]; max: number }[];  // every head (≤ 64), pooled mean ≤ 64×64, map / max → [0, 1]
  full: { map: number[][]; max: number };                   // selected head, pooled ≤ 128×128, map / max
  summary: AttentionHead[];                                  // every head, from the unpooled tensor
  key_profile: number[][];                                   // every head: mean weight per key over live queries, pooled ≤ 256
  stats: ViewStats; hist: Hist;                              // stats: whole tensor; hist: selected head
}
export type TensorView = ImageTensorView | SequenceTensorView | VectorTensorView | AttentionTensorView;

export type Method = "gxa" | "saliency" | "ig";

export type LayerMethod = "layer_ig" | "layer_cond";
/** An internal unit as the attribution objective: Σ over positions of channel `channel` (axis 1) of op `op`. */
export interface Objective { op: string; channel: number }
export type ObjectiveInfo = { kind: "logit"; index: number } | { kind: "unit"; op: string; channel: number };

export interface Attribution {
  method: Method; method_label: string; node_method: string; merge_method: string;
  objective: ObjectiveInfo; objective_label: string;
  target: number | null; target_label: string | null;
  prob: number | null; logit: number | null; value: number;
  steps: number | null;
  input: InputMap | null;
  scores: Record<string, number>;
  top_units: { op: string; path?: string; channel: number; value: number }[];
  input_note?: string | null;
  tokens?: TokenAttribution | null;   // integer (token) inputs, v0.2
}

/** An input attribution map, signed in [-1, 1]: pooled to ≤ 24×24 (images: summed over channels), or a sequence's
 * channel × time map (`kind: "series"`). */
export type InputMap = { kind?: undefined; map: number[][]; total: number; convergence_delta: number | null } | SeriesMap;
/** A (1, T, C) / (1, C, T) input's attribution: map[row][col], row = `row_stride` channels (1 up to 160 channels),
 * col = Σ over `stride` time steps (≤ 128 columns); blocks in channel order with Σ a, Σ|a| and share of Σ|a|;
 * time_totals = the signed Σ of each column; time: step i at t0 + i · step_s seconds (null: steps only). */
export interface SeriesMap {
  kind: "series"; map: number[][]; total: number; convergence_delta: number | null;
  channels: number; row_stride: number; steps: number; stride: number;
  time: { t0: number; step_s: number; label: string } | null;
  blocks: (InputBlock & { sum: number; abs: number; share: number })[];
  time_totals: number[];
  scale?: number;   // map × scale = Σ a of a cell (absent before the unpooled map)
}

export interface TokenAttribution {
  scores: number[];            // one per input position (sum over the hidden dimension; saliency: L2 norm, unsigned)
  signed: boolean;             // false for saliency
  total: number; convergence_delta: number | null;   // Δ for IG only
  layer: string;               // embedding module address, e.g. "transformer.wte"
  baseline_id: number | null;  // IG baseline token id, null for gxa / saliency
  method_label: string;        // e.g. "Layer IG on transformer.wte · baseline token 50256"
}

export interface LensCell { id: number; token: string; prob: number }
export interface LensResult {
  method_label: string;        // "Logit lens · transformer.ln_f + lm_head"
  layers: string[];            // ["embed", "transformer.h.0", …] (row order, top = embed)
  pos0: number;                // first position shown (results cover the last ≤ 64 positions)
  tokens: string[];            // input token per shown position
  k: number;                   // 5
  top: LensCell[][][];         // [layer][position][rank 0..k-1]
  target: { id: number; token: string };
  target_prob: number[][];     // [layer][position]
  target_rank: number[][];     // [layer][position], 1 = top
}

export interface LayerResult {
  method: LayerMethod; method_label: string; op: string; path: string; objective_label: string; steps: number;
  channels: number[]; top: { channel: number; value: number }[]; total: number; convergence_delta: number | null;
}

export interface NeuronResult {
  method: "neuron_cond"; method_label: string; op: string; path: string; channel: number; objective_label: string; steps: number;
  input: InputMap;
}

export interface AgreementBlock {
  basis: string; methods: string[]; labels: string[]; n: number; k: number; op?: string;
  jaccard: number[][]; overlap: number[][]; spearman: (number | null)[][];
}
export interface AgreementResult {
  objective_label: string; steps: number; input: AgreementBlock | null; units: AgreementBlock | null; units_error: string | null;
}

export type EditKind = "zero" | "mean" | "resample" | "constant" | "scale" | "steer";
export type Positions = "last" | number[] | [number, number][] | null;
export type SteerDirection =
  | { kind: "contrast"; run_a: string; run_b: string; pool: "mean" | "last" }
  | { kind: "channel"; channel: number };
export interface Edit {
  op: string; kind: EditKind; channels: number[] | null; positions: Positions;
  over?: "positions" | "runs"; source_run?: string; seed?: number; value?: number; alpha?: number; direction?: SteerDirection;
}
export interface InterventionSet { edits: Edit[]; target: number | null }
export interface NodeDiff { l2: number; rel: number; cos: number | null; max_abs: number; nonfinite?: number }  // nonfinite: positions that became / stopped being NaN or inf
export interface InterventionSummary {
  hash: string; name: string; created: number; labels: string[];
  target: number; target_label: string | null; delta_logit: number | null; delta_prob: number | null; truncated: boolean;
}
export interface InterventionResult extends InterventionSummary {
  run_id: string; set: InterventionSet; method_label: string;
  before: { logit: number | null; prob: number | null }; after: { logit: number | null; prob: number | null };
  top_before: TopK[]; top_after: TopK[];
  edited: string[]; downstream: string[]; saved: string[];
  budget_bytes: number | null; bytes: number;
  nodes: Record<string, NodeDiff>; stats: Record<string, Stats>;   // keyed by op id (node.output_op)
}
export interface GenToken { id: number; token: string; prob: number | null }
export interface Continuation { ids: number[]; tokens: string[]; text: string; top: GenToken[][] }
export interface GenerationResult {
  hash: string; n: number; labels: string[]; method_label: string;
  prompt: string[]; clean: Continuation; steered: Continuation; diverge: number | null;
}

export type JobKind = "capture" | "attribution" | "layer" | "neuron" | "agreement" | "lens" | "intervene" | "generate";
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
  | { kind: "lens"; target: number | null }
  | { kind: "intervene"; interventions: InterventionSet }
  | { kind: "generate"; interventions: InterventionSet; max_new_tokens: number };

export interface CreateRun {
  model: string; weights: string | null; sample: Record<string, unknown>;
  modules: string[] | null; max_capture_mb: number | null; backend: Backend | null;   // null = auto
}
export interface ModelsInfo {
  suggested: string[]; methods: Record<Method, string>; text: string[];
  backends: { hf: boolean; nnsight: boolean }; nnsight_params: number;
}
export interface Health { ok: boolean; device: string; torch: string; cuda: boolean; runs_dir: string; version: string; max_capture_mb: number | null }
