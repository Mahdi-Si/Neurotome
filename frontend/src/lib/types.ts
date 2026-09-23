// Types mirroring the Python API (backend/dissect/api.py). Keep in sync with docs/API.md.

export interface Stats {
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  zero_frac: number | null;
  hist: number[];
  hist_range: [number, number];
  numel: number;
}

export interface TopK { index: number; prob: number; logit: number; label: string | null }

export interface RunMeta {
  id: string;
  created: number;
  model: {
    spec: string | null; weights: string | null; class_name: string; num_params: number;
    num_ops: number; flops: number; activation_bytes: number; truncated: boolean;
  };
  sample: { kind: string; name: string };
  input_shape: number[] | null;
  output: TopK[];
  categories: string[] | null;
  reloadable: boolean;
  status: string;
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
}

export interface GraphEdge { source: string; target: string; shape: number[]; share?: number; on_path?: boolean }

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

export interface ImageTensorView {
  kind: "image";
  shape: number[];
  channels: number; height: number; width: number;
  channel: number;
  channel_rank: number;
  maps: { channel: number; mean: number; map: number[][] }[];
  dots: number[][];
  stats: ViewStats;
  peak: [number, number];
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

export type TensorView = ImageTensorView | SequenceTensorView | VectorTensorView;

export type Method = "gxa" | "saliency" | "ig";

export interface Attribution {
  method: Method;
  method_label: string;
  node_method: string;
  target: number;
  target_label: string | null;
  prob: number;
  logit: number;
  steps: number | null;
  input: { map: number[][]; total: number; convergence_delta: number | null } | null;
  scores: Record<string, number>;
  top_units: { op: string; path?: string; channel: number; value: number }[];
}
