# Design language

The look comes from technical spec sheets: a warm paper ground, 1 px ink hairlines, small uppercase
monospace labels, a tight grotesque for headlines, crosshair registration marks, and dot-matrix
figures. Color is scarce and always means something.

The mockups that set this direction are in `design/mockups/`. They're canvas artboards (`.dc.html`)
made in Claude's design canvas, kept here for reference. The production UI in `frontend/` implements them.

| Mockup | Screen |
|---|---|
| `Home.dc.html` | Home: hero, primitive cards, recent runs, quickstart |
| `Main.dc.html` | Graph workspace: module tree, graph, inspector |
| `Tensor.dc.html` | Tensor inspector: channel grid, dot matrix, profile, projection |
| `Trace.dc.html` | Attribution trace: saliency, backward path, contributors |

## Tokens

Source of truth: `design/tokens.json` (`color`, `color-dark`), mirrored in `frontend/src/styles/tokens.css`.
`lib/theme.test.ts` checks that the two agree and that every theme meets the contrast floors below.

| Token | Light | Dark | Use |
|---|---|---|---|
| `paper` | `#F4F4F1` | `#141413` | App ground |
| `panel` | `#FAFAF8` | `#1C1C1A` | Nodes, cards, side panels |
| `ink` | `#111111` | `#ECEBE6` | Text, strokes, magnitude marks |
| `ink-2` | `#5E5E59` | `#A3A29B` | Secondary text |
| `line` / `line-soft` / `line-faint` | `#D6D6D1` / `#E3E3DE` / `#C4C4BE` | `#3A3A36` / `#2C2C29` / `#4D4D48` | Region borders / row rules / inactive edges |
| `muted-fill` | `#E6E6E1` | `#2C2C29` | Active nav, track backgrounds |
| `accent` | `#3B4BDB` | `#8B97FF` | Selection, positive attribution, dominant path, "high" |
| `accent-soft` | `#E6E8FB` | `#252A4D` | Selected fills |
| `negative` / `negative-text` | `#D9531E` / `#B8431A` | `#E8703F` / `#F08A5D` | Negative attribution marks / text |

The accent and negative pair passes the palette validator: CVD ΔE 30, normal-vision ΔE 38, ≥3:1 on the surface.

## Themes

Light is the default; dark follows `prefers-color-scheme` until the user picks one. Measured WCAG 2 contrast
(on `paper` / on `panel`); the floors are text ≥ 4.5 and marks ≥ 3:

| Token | Light | Dark | Floor |
|---|---|---|---|
| `ink` | 17.1 / 18.1 | 15.4 / 14.3 | text 4.5 |
| `ink-2` | 5.9 / 6.2 | 7.2 / 6.7 | text 4.5 |
| `accent` | 5.9 / 6.3 | 7.0 / 6.5 | text 4.5 on panel |
| `negative` | 3.7 / 3.9 | 6.0 / 5.5 | marks 3 |
| `negative-text` | 4.9 / 5.2 | 7.5 / 6.9 | text 4.5 |
| on `accent-soft`: `ink` / `accent` | 15.6 / 5.4 | 11.6 / 5.2 | text 4.5 |

- **CSS**: the light tokens sit on `:root` (`color-scheme: light`); the dark ones on `:root[data-theme="dark"]` and,
  identically, on `:root:not([data-theme="light"])` inside `@media (prefers-color-scheme: dark)` (`color-scheme: dark`).
- **Toggle**: `ThemeToggle`, a `.chart-export` `.lbl` text button naming the theme it switches to (`Dark` / `Light`,
  `aria-label` `Switch to dark theme` / `Switch to light theme`). It sits at the far right of the TopBar after a
  1 px `line` divider, and in the Home aside footer. The choice is stored in `localStorage` (`neurotome.theme`) only
  when it differs from the system theme, so picking the system's theme again means "follow the system". A tiny
  inline script in `index.html` sets `data-theme` before the bundle loads, so there is no flash of the wrong theme.
  Storage failures (private windows) keep the choice for the session only.
- **Canvases and exports read tokens at draw time**: the graph canvas, minimap and heatmaps take their colours from
  `useTokens()` (resolved through a probe element, so hex, `rgb()` and system colours all work) and redraw on every
  theme or forced-colours change; SVG charts use `var(--…)` directly; PNG exports resolve the variables when they
  run, so an export matches the theme on screen. No literal colour lives in `src/` outside `tokens.css` and tests.

## Motion

There are no animated transitions or pans: pan, zoom, fit and centring set the transform at once. Any future
animation is off under `prefers-reduced-motion: reduce` (a global guard in `global.css` sets `transition` and
`animation` to none and `scroll-behavior` to auto).

## Forced colours

Under `forced-colors: active` (Windows high contrast) the token block maps to system colours only (`Canvas`,
`CanvasText`, `GrayText`, `Highlight`, `LinkText` for negative) and wins over both theme blocks. Canvases, chart
SVGs, node cards and inverted controls (`.btn.solid`, pressed segments) opt out of forced adjustment so they keep
drawing with those system-colour tokens and no backplate hides their text. Selection stays visible:
`[aria-pressed="true"]` and `[aria-current="page"]` get a 2 px `Highlight` outline inset by 2 px, and the job
progress fill is `Highlight`. Separators remain 1 px borders.

## Type

- **Geist Sans**: body and headlines. Headlines use weight 500 and tracking −0.04 to −0.055 em.
- **Geist Mono**: numbers, shapes, paths; the `.lbl` label is 10.5 px, uppercase, +0.05 em tracking.
- Fonts are bundled via `@fontsource`, so the UI makes no network calls.

## Components

- **Node card**: 196×46, 1 px ink border, 2 px radius. Mono op type top-left, shape top-right, name
  below. Collapsed modules get a stacked offset shadow; the input is dashed; the output is inverted (ink fill).
  The selected node gets an accent border and soft fill. The attribution bar is a 3 px accent strip along the bottom edge.
- **Group box**: dashed 1 px ink rectangle with a mono label (`LAYER1.0 · BASICBLOCK`) and op/param count on the right.
- **Pill**: mono label in a 1 px rounded outline.
- **Segmented control**: 1 px ink frame, inverted when active.
- **Crosshair mark**: 11 px circle plus cross, top-left of cards and panels.
- **Fold toggle**: 16 px square `+`/`−` button, 1 px `ink` border on `panel`, mono 11 px. `+` sits on a
  collapsed module node, `−` in a group box label; a box you expanded by hand stays even with one child, so it can close.
- **Far-zoom nodes**: below 40 % zoom nodes are plain canvas rectangles (`panel` fill, 1 px `ink` stroke,
  `accent` when selected) without labels; above it they are the DOM node cards.
- **Minimap**: bottom-left `panel` with a 1 px `line` border; nodes as tiny blocks, the viewport as a 1 px `accent` rectangle. Click to centre.
- **Search field**: `.field` input over the canvas, border turns `ink` while open. Results are a `panel` list
  with `line-soft` row rules; the active row gets `accent-soft`. Shapes are searchable as `1x64x56x56`, `64x56x56` or `64,56,56`.
- **Colour-mode legend**: top-right `panel`, 200 px, 1 px `line` border. Mono title naming what is measured
  (and the attribution method), a 4 px `accent-soft`→`accent` ramp with its ends, and the dim note
  "Module = its output tensor". Nodes take a tint from `panel` toward `accent`.
- **Flag badge**: in NaN/Inf mode a flagged node gets a 1.5 px `negative` outline and a mono `NAN/INF` tag in
  `negative-text`. This is the only place `negative` appears on graph nodes.
- **Canvas focus ring**: `.graph-canvas:focus-visible` draws a 1.5 px `accent` outline inset by 2 px (keyboard only).
- **Patch Δ legend**: the colour-mode legend for activation patching reads
  `Patch Δ = ‖after − before‖ / ‖before‖ · <path> → <baseline>`, lo `0`, hi the largest relative change (`pct`). Tints
  are normalised by the largest change outside the patched node; the patched node itself is full `accent` with the
  readout `patched · <baseline>`, other nodes read `‖Δ‖ <value> · rel <pct>`, and nodes without a value read `‖Δ‖ —`.
- **Job progress**: a 4 px `line-soft` track with an `accent` fill at the job's progress (`.job-track`), and under it
  a mono `.lbl dim` status line (`Queued`, `42% · IG batch 3/8`, `Cancelled`) with a `Cancel` text button
  (`.chart-export` style) while the job runs. Nothing renders once the job is done; a failed job shows the server's
  message in `.err`. Used wherever a computation runs as a job (trace parameter strip, layer panel, neuron map,
  method agreement, patch panel).
- **Capture banner**: when a run's tensors were only partly saved (capture budget reached, or a module selection),
  a 32 px `panel` row with a 1 px `line` bottom border sits under the TopBar of the Graph and Tensor views: a
  `.lbl pill` tag (`Truncated` / `Selective`) and a `.lbl dim` line such as
  `Capture budget 2.0 GB reached · 412/1210 ops saved · ∅ nodes have no tensor`. No colour: it is information,
  not an error.
- **Node without a tensor**: a 1 px dotted `ink` border (selected, NaN-flag and on-path borders still win; the
  input keeps its dashed border), `∅` before the shape (`∅ 16×28×28`), and `no tensor · <reason>` appended to the
  title and hover readout. Far-zoom canvas nodes stroke with a 1.5 px dash. The inspector reads
  `Tensor op <op> · not saved` with a dim `No tensor · <reason>` line, and `Open tensor →` is disabled with the
  reason as its title. The Tensor view lists only nodes with a tensor; opening one without shows an `.empty`
  state (`No tensor saved`, `<path> · <reason>`, a `← Graph` link) instead of silently picking another node.
- **Home capture progress**: `Capture forward pass` starts a capture job; while it runs the button reads
  `Capturing…` (disabled) and a job progress bar (`capture job`, `Load model` → `Trace` → `Save tensors i/N` →
  `Summarise`) with `Cancel` sits under it. Two optional fields precede the button: `Modules` (comma separated,
  empty = every op) and `Budget · MB` with the server default dim on the right. Recent runs show `saved/total`
  in the Ops column when a capture was partial.
- **Error lines**: every `.err` line shows the server's message without the HTTP status; a model whose output
  cannot be explained shows its reason under the Trace parameter strip and disables `Run attribution`; input
  attribution skipped for a non-float input shows the server's note in the saliency panel.
- **Objective pill**: with a unit objective the Trace view's Target select becomes a `.pill`
  `Unit · <objective label>` (`relu_2_4 · ch 3 · Σ spatial` until the first result names the module path), followed
  by a `← Logits` text button that goes back to a logit target.

## Charts

- Magnitude uses ink; the selected/positive item uses accent; negative uses rust. There's never a rainbow.
- The dot matrix encodes magnitude as circle size: outline = low, ink fill = mid, accent fill = high.
- Every chart has a hover readout line underneath, so no value is conveyed by color alone.
- The graph has the same: hovering a node or edge prints a mono readout line at the bottom of the canvas
  (name · type · shape · colour-mode value, or source → target · shape · dtype · device; a dtype or device
  change along the edge is shown as `a → b` in `accent`).
- Histograms use a √count scale (labelled) so long tails stay visible.
- Baselines are 1 px ink; grids are absent or dotted.
- **Sequential vs diverging maps**. Sequential (unsigned, scaled to [0, 1]) is a `paper`→`ink` ramp on heatmaps
  and size plus outline/ink/accent on dot matrices. Diverging (signed, scaled to [-1, 1] with 0 fixed at 0) runs
  `negative` ↔ `paper` ↔ `accent`, the same look as signed attribution. Use diverging only when the sign means
  something (pre-activations, attributions, deltas); the view states which scale is on (`Scale · per channel` /
  `Scale · global lo … hi`). No third colour is added for either.
- **Crosshair on the inspected pixel**: an `accent` 1 px ring of one cell's diameter on the cell, plus a 1 px
  horizontal and vertical `accent` line through its centre across the whole chart at 0.5 opacity. Drawn in the
  chart's SVG, or as 1 px positioned divs over a canvas (a crosshair drawn into a tiny canvas blurs). Line plots
  of the same row mark the column with a 1 px `accent` vertical rule. A clickable map shows the crosshair cursor.
- **Input overlay**: a map drawn over the captured input preview (≤ 224 px, `pixelated` below 112 px). The map
  layer has a transparent ground; each cell is `accent` (v ≥ 0) or `negative` (v < 0) with alpha ∝ |v|, so weak
  cells let the image through. An `Opacity` range slider (0–100 %, `accent-color: accent`, mono percentage) sets
  the layer's opacity; the `Dots / Over input` segmented control switches it on and is disabled, with a tooltip,
  for runs without a preview.
- **PNG export**: every chart's readout line carries a `PNG` text button at its right end (`.chart-export`: no
  border, `ink-2`, `accent` on hover, the usual focus ring). The export is the chart at 2× (canvases upscaled
  nearest-neighbour to ≥ 480 px) on a `panel` ground; the file is named `run_op_cN_chart.png`.
- **Agreement matrix**: an n×n SVG grid of 40 px cells with 1 px `line` rules; row and column heads are mono
  uppercase short method names (`GXA`, `SAL`, `IG`, `LIG`, `LCOND`). A cell's fill is `accent` at alpha v for Jaccard
  and positive Spearman, `negative` at alpha |v| for negative Spearman, and its value is printed in the cell (mono
  10.5 px `ink`, 2 decimals, with a `panel` halo) so colour is never the only carrier; the diagonal is `line-soft`
  with `—`. The caption above names the basis and `top-k = k of n`; the hover readout prints every metric
  (`IG × SAL · Jaccard@58 0.41 · overlap 0.58 · Spearman 0.72`).
- **Values table**: mono `.num` 11 px cells, right-aligned, `line-soft` row rules, headers `.lbl dim` sticky on
  `panel`; negative values and `nan`/`inf` in `negative-text`; at most 320 px high, scrolls both ways, with a
  `key · shape · axes` caption above and a dim truncation note when the slice was cut.
