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

Source of truth: `design/tokens.json`, mirrored in `frontend/src/styles/tokens.css`.

| Token | Value | Use |
|---|---|---|
| `paper` | `#F4F4F1` | App ground |
| `panel` | `#FAFAF8` | Nodes, cards, side panels |
| `ink` | `#111111` | Text, strokes, magnitude marks |
| `ink-2` | `#5E5E59` | Secondary text (≈6:1 on paper) |
| `line` / `line-soft` / `line-faint` | `#D6D6D1` / `#E3E3DE` / `#C4C4BE` | Region borders / row rules / inactive edges |
| `accent` | `#3B4BDB` | Selection, positive attribution, dominant path, "high" |
| `accent-soft` | `#E6E8FB` | Selected fills |
| `negative` / `negative-text` | `#D9531E` / `#B8431A` | Negative attribution marks / text |

The accent and negative pair passes the palette validator: CVD ΔE 30, normal-vision ΔE 38, ≥3:1 on the surface.

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

## Charts

- Magnitude uses ink; the selected/positive item uses accent; negative uses rust. There's never a rainbow.
- The dot matrix encodes magnitude as circle size: outline = low, ink fill = mid, accent fill = high.
- Every chart has a hover readout line underneath, so no value is conveyed by color alone.
- The graph has the same: hovering a node or edge prints a mono readout line at the bottom of the canvas
  (name · type · shape · colour-mode value, or source → target · shape · dtype · device; a dtype or device
  change along the edge is shown as `a → b` in `accent`).
- Histograms use a √count scale (labelled) so long tails stay visible.
- Baselines are 1 px ink; grids are absent or dotted.

## Not yet done

- Dark theme: add a `[data-theme="dark"]` token block and re-validate the chart colors against the dark surface.
- Reduced-motion and high-contrast (forced-colors) passes.
