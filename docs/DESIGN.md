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
- **Intervention Δ legend**: the `Intervention Δ` colour mode (disabled, title `Run an intervention first`, until
  an intervention is open) reads `Intervention Δ = ‖after − before‖ / ‖before‖ · <name>`, lo `0`, hi the largest
  relative change (`pct`). Tints are normalised by the largest change among downstream nodes that were not edited;
  edited nodes are full `accent` with the readout `edited · <labels of the edits on that op>`; nodes that are not
  downstream are `v 0` (`unchanged · not downstream`); downstream nodes read
  `‖Δ‖ <l2> · rel <pct> · cos <0.000 | —> · max|Δ| <value>`, and downstream nodes without a row read `‖Δ‖ —`.
- **Intervention banner**: while an intervention is open (URL `iv`) the Graph, Tensor and Trace views show a 32 px
  `panel` row with a 1 px `line` bottom border under the capture banner: a `.lbl pill` `Intervened`, a `.lbl dim`
  line `<name> · <n> edits · Δ logit <±0.00> · <saved>/<downstream> downstream tensors saved` (+ `· budget reached`;
  the Trace view adds `· attribution runs on the clean pass`) with the Δ in the Δ colour, and on the right a
  `.chart-export` link `Clean` (aria-label `Show the clean run`) that drops `iv` (and `color=iv`). An unknown hash
  shows the server's message in `.err` in the same row. The TopBar tab links carry `iv` (not to Lens) and the draft
  `set` (to Graph and Interventions only).
- **Δ colour**: a Δ logit or Δ prob ≥ 0 is `accent` (the edit supports the target), < 0 is `negative-text` (it
  hurts it); the sign is always printed too (`+0.12` / `−3.21`).
- **Intervene section**: the inspector section after `From` / `Feeds` (every backend): `.lbl`
  `Intervene · this node's output` with a `Clear set` text button on the right while a draft exists; with an open
  intervention, kv rows `‖Δ‖`, `Rel Δ`, `Cosine`, `Max |Δ|` for the node (or `Edited · <labels>`, or `Unchanged`)
  and a `Clean vs intervened →` link (a dim reason instead when its tensor was not saved); the edit form; a dim
  `<n> edits in set` line with `Open in Interventions →`; a `.btn solid` `Run set (n)`; job progress and errors.
  Input, output and non-float nodes show a dim note instead of the form.
- **Edit form**: `.seg lbl` groups and `.field` inputs, top to bottom: `Edit kind` (`Zero` `Mean` `Resample`
  `Constant` `Scale` `Steer`, a 3×2 grid with 1 px `ink` rules); per kind `Over` (`Positions` / `Runs`), `Source run` (`Random run` first; runs with
  the same model spec and weights, this one marked `· this run`) and `Seed`, `Value`, `Alpha`, `Direction`
  (`Contrast`: `Run A`, `Run B`, `Pool` for token tensors; `Channel`: `Channel`); then the scope: `Channels`
  (placeholder `all · e.g. 0,3,7-9`, an out-of-range list shows `.err` `Channels are 0–{n-1}` and disables
  `Add to set`), the position picker for (B, C, H, W) or `Positions` (`All` / `Last` / `Selected`) with token
  selection for (B, T, D); a `.btn` `Add to set` closes it. Draft edits read like the server's labels:
  `<Kind> · <path> · <scope>` (`Scale ×2 · block.relu · ch 0–3`).
- **Interventions tab**: the TopBar tabs read `Graph · Tensors · Trace · Lens · Interventions`; `Run interventions`
  is the TopBar button. A 300 px left aside lists the saved interventions (`.lbl` `Saved` + count; each row the
  name as a button, `aria-pressed` when open, a dim `<n> edits · Δ logit` line in the Δ colour, `Rename` (the name
  becomes a `.field`; Enter saves, Escape cancels) and `Delete` → `Confirm delete` text buttons); empty:
  `No saved interventions · run a set to save one`. The main area stacks the **Set** card (`Target` select, one row
  per edit with a `×` remove button, `Clear set`), the **Add edit** card (`Node` select, then the edit form) and the
  **Result** region: `Result · <name>` with the dim method label, kv `Target`, `Logit before → after (Δ)` and
  `Prob` in the Δ colour, top-k before / after as two columns of `label · p` rows with a 4 px `ink` bar ∝ p (the
  target's label in `accent`; hover readout `before #2 "cat" 0.12 → after #1 0.41`), the edit labels, a truncation
  note, and the links `Graph · Intervention Δ →` and `Clean vs intervened →`.
- **Continuations**: for a causal LM the tab's Generation card has a number `Tokens` (1–64, default 16) and
  `Generate` with job progress, then two columns (`Clean continuation`, `Intervened continuation`): the prompt
  tokens dim, the generated tokens as mono spans with visible whitespace; from the first diverging step on, tokens
  carry a 1 px `ink` underline. Hover or focus on a step prints its top-3 per side as the readout line. Below, a
  table `Per-step top tokens` (`Step · Clean · p · Intervened · p`, top-1 per side) with rows from the divergence on
  in `accent-soft`.
- **Clean vs intervened**: with an open intervention the Tensor view's left rail gets a `.btn`
  `Clean vs intervened` (`aria-pressed`) beside `Compare`; it opens the compare layout with `A · clean` and
  `B · intervened` of the selected node on a shared scale, and `Δ = B − A` statistics. Plain compare with an open
  intervention compares two nodes both under it.
- **Job progress**: a 4 px `line-soft` track with an `accent` fill at the job's progress (`.job-track`), and under it
  a mono `.lbl dim` status line (`Queued`, `42% · IG batch 3/8`, `Cancelled`) with a `Cancel` text button
  (`.chart-export` style) while the job runs. Nothing renders once the job is done; a failed job shows the server's
  message in `.err`. Used wherever a computation runs as a job (trace parameter strip, layer panel, neuron map,
  method agreement, intervene section, Interventions tab: intervene / generate jobs, `Clean pass` → `Intervene` →
  `Save tensors i/N`, `Clean i/n` → `Intervened i/n`).
- **Capture banner**: when a run's tensors were only partly saved (capture budget reached, or a module selection),
  a 32 px `panel` row with a 1 px `line` bottom border sits under the TopBar of the Graph and Tensor views: a
  `.lbl pill` tag (`Truncated` / `Selective`) and a `.lbl dim` line such as
  `Capture budget 2.0 GB reached · 412/1210 ops saved · ∅ nodes have no tensor`. No colour: it is information,
  not an error. An otherwise complete NNsight run gets the `Modules` tag with
  `NNsight capture · module outputs only · <n> module calls` (its graph nodes are leaf-module calls, not ops).
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
- **Sample kind**: below `Weights`, the `Sample` label with a segmented control on its right (group `Sample kind`,
  `Text` / `File`). It defaults to `Text` for a text model (a known text spec, or any `hf:` spec that does not look
  like a vision model) and follows the Model field until the user picks one by hand; then it stays. `Text` shows a
  3-row `.field` `<textarea>` (`Sample text`, placeholder `The cat sat on the`; empty → form error
  `Type some text, or pick File`); `File` shows the file input (empty = random tensor / random tokens).
  A third segment `Dataset` appears when the Model field holds a complete `py:` spec whose module lists samples
  (`/api/samples`, asked 350 ms after typing stops) and is then the default: a `.field` `Filter samples`
  (placeholder `filter · ref, label or group`; every word must match), a 6-row `.field` listbox `Dataset sample`
  with one `<optgroup>` per group and the sample label per option (its ref as the title; at most 300 shown), and a
  dim line `<matched> of <n> samples · first 300 shown · <picked ref>`; the first match is picked until the user
  picks one. A module that fails to import or list shows a dim one-line `No dataset · <message>` (never `.err`).
- **Capture backend**: a `.field` select under the sample (`Capture backend`): `Auto`, `TorchLens · every op`,
  `NNsight · module outputs`. With `Auto` the label row carries the dim note `NNsight above 1 B params` on the right;
  the NNsight option is disabled, with `pip install 'neurotome[nnsight]'` as its title, when the server lacks it.
- **Lens tab**: the TopBar tabs read `Graph · Tensors · Trace · Lens · Interventions`. For a run whose model is not a causal LM the
  Lens tab is a plain `.lbl` span at 0.45 opacity with `aria-disabled="true"` and the title
  `Logit lens needs a causal language model` (not a link); opening its URL anyway shows that sentence as an `.empty`
  state with a `← Graph` link.
- **Input view**: the Tensors tab opens the model's input node as `Input` (a `Show` segmented control, `Input` /
  `Tensor`, in the left rail switches to its channel views; the tensor-only rail controls hide meanwhile). Two cards:
  `Raw input` (the uploaded image as it was, caption `<name> · <format> · W×H px · <size>`, a `Download raw ↓` text
  button; a text sample shows its text; otherwise a dim note why there is none) and `Model input` (the tensor the model
  saw, drawn at its own resolution, caption `<shape> · <dtype> · de-normalised for display` or `min–max stretched`;
  token inputs show one mono chip per token with its id under it). Pictures are 224–360 px tall with their aspect
  kept (`pixelated` below 112 px), in 1 px `line` frames, with `Axes` pixel ranges (`x · px` 0…W−1, `y · px` 0…H−1,
  row 0 at the top); token chips carry a `position` range. Below: `Preprocessing`, the steps as a numbered mono list,
  and `Model input values`, a table of mean / std / min / max per channel after preprocessing.
- **Input view of a signal** (a dataset sample with a raw signal, or any float (1, T, C) input): the two cards stack
  full width so their time axes line up. `Raw input` (caption `<name> · <n> channels · <samples> samples at <fs> Hz`)
  holds one plot per raw channel (`Raw <channel>`, 84 px): the pooled min … max envelope as an `accent-soft` band and
  the mean as a 1.25 px `accent` line over a 1 px `ink` baseline; y `hi … lo` labelled `<channel> · <unit>`; hover
  draws a 1 px `ink` rule and reads `t 10.13 min · mean 134.2 bpm · 131.0 … 137.9` (a plain value at stride 1); the
  idle readout says `2400 → 1200 points · band = min … max, line = mean`; `Download raw (.npz) ↓`. `Model input`
  holds one diverging `HeatCanvas` per block (named by the block, 48–200 px tall, 12 px a row): rows are the
  block's columns (y `start … stop − 1` labelled `<block> · <unit> · column`), a right-hand 112 px label column
  prints every k-th label (≤ 8, mono 10 px `ink-2`, ellipsis, the full label in its title), each block scaled to its
  own ± max |value| (named in a dim line under the maps, with the pooling `T → n columns, means of k steps`); the
  readout is `t 10.13 min · step 3 · "0.0625–0.125 Hz" · 0.83 z`. **One clock**: every chart's x is the union of
  the raw span and the model steps' span (step i a cell centred on `time.t0 + i · step_s`), in seconds, or in
  minutes from 10 min up (`s to delivery` → `min to delivery`); each chart's marks are placed inside that domain
  and every chart has the same y gutter (ends padded with figure spaces) and the same right-hand column, so a raw
  sample and the step computed from it sit at the same x. Without a `time` the heatmaps count `step` on their own.
  A `Sample` card (kv rows: `ref`, then the module's `info`) sits under the two cards; the values table has one
  named row per block. With a step (`t` in the Tensors URL, from the Trace view's `Show on input →`) the `Model
  input` card opens with the step's window legend, each block map carries the bands of the windows that name it
  (or name no block) and the step rule, and on the shared clock every raw plot carries every window's time span
  and the step rule (bands and rule as in the Trace map of a time series).
- **Side panels** (right of the Graph, Tensors, Trace and Lens views): 400 px (Tensors 460 px) by default, 20 px padding. Their left edge is draggable (8 px hit strip, a 2 px accent rule on hover, drag or focus; ←/→ by 24 px when focused; double-click or Home resets); the width is 320 px to 60% of the window and is remembered per view (`neurotome.side.<view>` in localStorage). Their
  content is a stack of sections, each a 1 px `line` rule, a header row (`.lbl` title left, `.lbl dim` detail right)
  and its chart or rows, 12 px apart, so charts, legends and readouts group under the header they belong to.
- **Error lines**: every `.err` line shows the server's message without the HTTP status; a model whose output
  cannot be explained shows its reason under the Trace parameter strip and disables `Run attribution`; input
  attribution skipped for a non-float input shows the server's note in the saliency panel.
- **Objective pill**: with a unit objective the Trace view's Target select becomes a `.pill`
  `Unit · <objective label>` (`relu_2_4 · ch 3 · Σ positions` until the first result names the module path), followed
  by a `← Logits` text button that goes back to a logit target.

## Charts

- Magnitude uses ink; the selected/positive item uses accent; negative uses rust. There's never a rainbow.
- The dot matrix encodes magnitude as circle size: outline = low, ink fill = mid, accent fill = high.
- Every chart has a hover readout line underneath, so no value is conveyed by color alone.
- The graph has the same: hovering a node or edge prints a mono readout line at the bottom of the canvas
  (name · type · shape · colour-mode value, or source → target · shape · dtype · device; a dtype or device
  change along the edge is shown as `a → b` in `accent`).
- Histograms use a √count scale (labelled) so long tails stay visible.
- **Every chart states its x and y range** (`<Axes>` in `charts.tsx`): the y label on its own 14 px row above the
  chart, horizontal and left-aligned (never rotated, so a short chart cannot clip it); the y numbers in a left gutter,
  `from` at the top and `to` at the bottom; x on a 16 px row below, `from` left, label centred, `to` right. Mono
  10 px `ink-2` (`.axis`), a 4 px 1 px `line` tick at each end, no gridlines. Integers (indices, counts) print as they
  are, other numbers through `num()`. The gutter width is 6 px a character of the widest end value + 8 px
  (`axisGutters`), and the PNG export draws the same ranges in the same places (SVG, canvas, overlay layers alike).
  What each chart shows:
  - histogram: x `lo … hi` of the quantity (`value`, `c6 value`, `weight`, `Δ`), y `max count … 0` on `√count`;
  - line plot: x `0 … len − 1` in unpooled positions (`col`, `token`, `layer`), y the plotted domain `hi … lo`
    (`value`, `norm`, `p` on a fixed 0 … 1);
  - diverging bars: x `0 … len − 1` (`channel`, `index`), y `+m … −m` (`value`, `attribution`);
  - PCA scatter: x and y the data values at the box's edges (`PC1 31%`, `PC2 16%`), one scale for both;
  - spatial maps (dot matrix, over-input overlay, trace input attribution): y `0 … H − 1`, x `0 … W − 1` in real pixels
    of the unpooled tensor (the trace map speaks input pixels, `input y` / `input x`; a time-series trace map is
    `channel` × time on the run's clock, see Trace map of a time series); the sequence heatmap `token`
    × `feature`; the attention map `query` × `key`;
  - small multiples (channel tiles, head tiles) get one shared line under the grid, `each tile y 0–111 · x 0–111`
    (printed under the exported grid too); the logit lens names its layer and position ranges in its caption
    (`3 layers (embed … h.1) × 22 positions (0–21)`); top-k probability bars say `bar = p 0–1`. The agreement
    matrix and the token strip are categorical / text and name every row and column already.
  The idle readout keeps what the axes do not: `24 bins · √count scale`, the line plot's `data min … max`, a pooled
  bar chart's `1000 → 512 bars · block means`, the scatter's shade.
- **Readouts print real values and source positions**: never a normalised value or a pooled index. A 1-D chart of
  block means (a 1000-long vector as 512 bars, a 224-px row as 128 points) names the span (`index 652–653`), its head
  reads `1D · 224 → 128 pooled` and the idle readout `1000 → 512 bars · block means`; highlight and click work in
  source indices. A PCA point reads as the position or token it was sampled from (`px (116, 172) · PC (…)`), not
  its sample number. The rare value whose divisor is not in the payload (a pooled sequence heatmap) reads `rel 0.42`.
- **Geometry is not distorted**: the PCA scatter uses one scale for both axes (a PC2 with a sixth of PC1's variance
  looks flat); channel tiles keep the feature map's aspect (`W / H`), in the PNG too; probabilities (the lens
  target across layers) plot on a fixed `0 … 1` axis. A line plot with one point draws a 6 px `accent` dot, no area.
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
- **PNG export**: every chart's readout line (except the token strip's, which is text) carries a `PNG` text button at its right end (`.chart-export`: no
  border, `ink-2`, `accent` on hover, the usual focus ring). The export is the chart at 2× (canvases upscaled
  nearest-neighbour to ≥ 480 px) on a `panel` ground; the file is named `run_op_cN_chart.png`.
- **Agreement matrix**: an n×n SVG grid of 40 px cells with 1 px `line` rules; row and column heads are mono
  uppercase short method names (`GXA`, `SAL`, `IG`, `LIG`, `LCOND`). A cell's fill is `accent` at alpha v for Jaccard
  and positive Spearman, `negative` at alpha |v| for negative Spearman, and its value is printed in the cell (mono
  10.5 px `ink`, 2 decimals, with a `panel` halo) so colour is never the only carrier; the diagonal is `line-soft`
  with `—`. The caption above names the basis and `top-k = k of n`; the hover readout prints every metric
  (`IG × SAL · Jaccard@58 0.41 · overlap 0.58 · Spearman 0.72`).
- **Token labels**: wherever a tensor's token axis has exactly as many positions as the run's sample tokens, the
  token names the position. Tokens are shown with visible whitespace (`·` space, `↵` newline, `→` tab, `∅` empty).
  A single position reads `#3 "cat"`; a pooled cell reads `#64–67 "a·b·c·d"` (the first 4 tokens, `…` beyond).
  The sequence heatmap readout is `#3 "cat" · feature 4–7 · 0.123`, and for `T ≤ 32` a label column (mono 10.5 px,
  `ink-2`, right aligned, ellipsis at 10 characters) runs beside the heatmap, one token per row. The per-token norm
  line plot reads `token 3 "cat" · 12.3`; the PCA scatter names the token it sampled (`token 3 "cat" · norm 12.3 · PC (…)`).
- **Attention map**: a `(1, H, T, T)` tensor whose rows are probability distributions opens as `Attention` (so do
  non-square `(1, H, Q, K)` and heads-in-the-middle `(1, Q, H, K)` maps, e.g. lag attention; see ARCHITECTURE); its
  axis boxes read `B H Q K` (`B Q H K`) with the `H` box hot (`accent` on `accent-soft`), and the left rail's `Head (0–H-1)`
  field picks the head (the `c` URL param). The main area is a grid of head tiles like the channel grid: a
  sequential `paper`→`ink` `HeatCanvas` per head, each scaled to its own max, the selected tile with a 2 px `accent`
  border; under each tile `h3`, the dim mean entropy (`2.41 b`) and the head's pattern word (`prev` / `first` /
  `self`) in `accent`. The side panel shows `h3` large, `Q×K → n×m pooled`, and the selected head's map as a square
  canvas (`Head 3 attention`, a non-square map stretched to it) with rows = queries (y `query`) and columns = keys
  (x `key`); its readout is `q #5 "cat" → k #4 "the" · 0.412` (the unscaled weight; pooled cells say `mean`; an axis
  whose length is not the token count stays numeric) and a click drops the crosshair. For a square map with
  `T ≤ 32` both axes carry token ticks: a label column for queries and a −60° rotated row for keys. The upper
  triangle of a causal map is simply `paper`. Under the map, `Key profile · h3`: a line plot of the head's mean weight
  per key over its live queries (x `key`, y `mean weight` from 0, the peak key marked; for lag attention the lag
  profile), then a dim line `peak k3 · 0.123 · zeros 71% · masked rows 3.3%`. `Value distribution` covers the
  selected head.
- **Head summary**: a `<table>` (`Head summary`) under the head grid: `.num` cells, `line-soft` row rules, a 1 px
  `ink` rule under the `.lbl dim` headers, columns `Head · Entropy (bits) · Prev · First · Self · Zeros · Peak key ·
  Pattern` (mean weight on the previous token, on the first token, on itself; the fraction of exact zeros; the key
  profile's peak `k3 · 0.123`; the pattern is the one above 0.5). A non-square map has no token patterns: its columns
  are `Head · Entropy (bits) · Entropy / log K · Masked · Zeros · Peak key` (masked = all-zero rows). The selected
  head's row is `accent-soft`; clicking a row selects its head.
- **Lens grid**: the Lens view's main chart, one SVG (`Logit lens`) so the `PNG` export works like any chart. Rows
  are the residual stream after the embedding (`embed`) and after each block, top to bottom (a 120 px mono
  uppercase head column, `transformer.h.3` shown as `h.3`); columns are positions (56 px, the token as the header,
  the selected position's in `accent`); cells are 56×22 px with 1 px `line-soft` rules. `Top-1` colour: `accent` at
  alpha = the top-1 probability, the top-1 token (6 characters) printed in the cell. `Target` colour: `accent` at
  alpha = the target's probability, its rank `#17` printed. Cell text is `ink` below alpha 0.55 and `panel` above,
  so it never disappears. The selected column gets a 1 px `accent` outline; the grid scrolls sideways inside its
  card. Readout: `h.3 · pos 5 "cat" → "dog" 0.41 · "the" 0.12 · "a" 0.05 · target "mat" #17 p 0.003`. The
  parameter strip holds `Target` (the run's top outputs or `top prediction`) and the `Lens colour` segmented
  control (`Top-1` / `Target`), with the job progress, then the dim method label, on the right; `Run lens` is the
  TopBar button. The side panel (360 px) names the position, plots the target's probability across layers (line
  plot, readout `layer 3 "h.2" · 0.410`) with a `layer · #rank · p` list, and lists the top-k tokens of the
  hovered (or last clicked) cell as rows with an `ink` probability bar. URL: `t` (target id), `mode=target`, `pos`.
- **Token strip**: token attribution for token inputs (group `Token attribution`), in the Trace view's saliency
  panel under the big `TOKENS` label with the method label on its right. Tokens are inline mono spans (13 px,
  2 px 3 px padding, 1 px `line-soft` border, wrapping) whose ground is `accent` (supports) or `negative`
  (opposes) mixed into transparent at |score| / max; unsigned saliency is always `accent` and the legend reads
  `Magnitude · unsigned` instead of `Supports / Opposes / Weak`. The hovered token gets a 1 px `ink` border and the
  readout `#3 "cat" · +0.1234 · 21% of Σ|a|`. **Exception to PNG export**: the strip has no `PNG` button; it is
  text, so copy it instead.
- **Trace map of a time series**: a sequence input's attribution (`input.kind: "series"`, input attribution or a
  neuron map) replaces the dot matrix in the saliency panel with a group `Time-series attribution`, and the left
  panel widens from 400 px to `min(760 px, 50vw)` (the channel × time map needs the room; a `Narrow →←` /
  `Widen ←→` `.chart-export` text button beside the input shape toggles it; the image overlay controls hide). Three
  charts, top to bottom: (1) a diverging `HeatCanvas` (`Input attribution` / `Neuron conductance`, 3 px a row,
  96–300 px tall, `pixelated`), rows = channels in block order (y `0 … C − 1` labelled `channel · by block`),
  x = time on the run's clock (`608 … 1192` `s from recording start`, minutes from a 10 min span like the Input
  view, else `step`), a 1 px `ink` rule at each block boundary and the block names in a 112 px right-hand column at
  their block's middle (mono 10 px `ink-2`, ellipsis); readout `fhr_st · "ψ3 · 0.887 Hz" · t=-19.93 min · rel 0.42`
  (a pooled row reads `ch 160–162`, no clock `step 3–5`). (2) `Attribution over time`: the signed Σ over channels of
  each column as a 1.5 px `accent` line on a symmetric `+m … −m` axis (`Σ over channels`) with a dashed `line` zero
  rule, on the same x and the same (padded) y gutter and right column as the map, so a column sits under its cells;
  hover draws an `ink` rule and reads `t=610.00 s · Σ +0.012`; idle it names the peak column. (3) `Block shares`:
  one 18 px row per block (drawn 1:1, ≤ 380 px wide), the name (mono 10 px `ink-2`), an 8 px `ink` bar on a
  `line-soft` track (0–100 % of Σ|attribution|, `accent` on hover) and the percentage; categorical, so no axes; readout
  `fhr_st · 44% of Σ|a| · Σ +0.2 · 36 channels`. A dim line under them names the pooling
  (`126 channels × 300 steps · colour ± max |cell|`; time pools only above 512 steps). Each chart exports its PNG.
- **Step picker**: the Target of a values run (`prob: null`, a score per step) with more than 5 outputs is a group
  `Target`: the select (`top prediction` and the top-5 as quick picks, `logit[i] · <label> · value v`; a picked
  index outside them gets its own option), a number field `Output index` (0 … n − 1, only a valid index reaches the
  URL `t`), a range slider `Output index slider` (`accent`) and a `.lbl` readout `t40 · +5:28 · value 0.8278 · of 74`
  (the CATEGORIES label, else `[i]`; `· top` while it follows the top prediction). Classifier runs keep the plain
  select.
- **Window bands** (a picked step whose module has `target_windows`): drawn over the series map, restricted to the
  rows of the window's blocks (all rows for `blocks: null`), and full height over the time line; `target` an
  `accent` 22 % fill with 1.5 px solid `accent` edges, `context` a 1 px dashed `accent` outline with no fill (a
  context window is often the whole causal past, and a fill would tint the map under it), `horizon` an `ink` 7 % fill with 1 px dotted `ink` edges, and the step itself a 1.5 px `accent` rule at its
  centre; distinct by edge style as well as colour, and `forced-color-adjust: none` so under forced colours they
  keep `Highlight` / `CanvasText`. They never take the pointer (the readout still works). A legend (`Input
  windows`, `.lbl dim`, wrapping) names each: a swatch drawn like its band, `<label> · steps 28–39 · <blocks>`, or
  why there are none (the hook's error, or no hook).
- **Zoom**: a `Zoom` segmented control `Full` / `Around step` beside `Colour` crops the map and the time line to
  the step and its `target` / `horizon` windows, plus 10 % of the input's steps (2–30) on each side; `context` windows
  are left out, since one often spans the whole causal past;
  both x axes print the cropped clock, the readouts keep full-map steps, and the dim pooling line adds
  `zoomed to steps 17–58`.
- **Where the attribution is**: a table under the legend: one row per window and `outside all windows`; columns
  `Window` (swatch + label), `Steps` (`steps 28–39 · <blocks>`), `Share Σ|a|` (the window's cells' Σ|a| over the
  whole map's, from the map in `windowShares`; overlapping windows each count their cells, so shares need not add
  up), `Σ a` (signed, `negative-text` below 0) and a `Values →` link (`Show input values · <label>`). Rules and type
  as the Input view's values table. `Show on input →` under it opens the Input view with the step.
- **Show input values**: clicking a map cell marks it (the chart's cross marks) and a dim line
  `Picked · pressure · band power · "1–2 Hz" · t=608.00 s` offers `Show input values →` and `Clear`. The link (and
  each window's `Values →`) opens the Tensors view on the input node, `Tensor` side, with the Values expression
  prefilled (`s` in the URL, e.g. `[0, 28:40, 5:10]`: the cell's or window's steps and its blocks' channels).
- **Position picker**: the edit form's `Position picker` (an img-role dot matrix of the op's channel map) marks each
  pooled cell with a selected pixel with a 1 px `accent` square inset 0.5 px; a click toggles every pixel of that
  cell; a dim `<k> px selected ·` line with a `Clear` text button sits under it; the usual readout line applies.
- **Token selection**: in selection mode (group `Token positions`) the token strip's tokens are buttons
  (`aria-pressed`); selected tokens get an `accent-soft` ground and a 1 px `accent` border, and the readout reads
  `#3 "cat" · selected`.
- **Values table**: mono `.num` 11 px cells, right-aligned, `line-soft` row rules, headers `.lbl dim` sticky on
  `panel`; negative values and `nan`/`inf` in `negative-text`; at most 320 px high, scrolls both ways, with a
  `key · shape · axes` caption above and a dim truncation note when the slice was cut.
