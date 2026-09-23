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

## Charts

- Magnitude uses ink; the selected/positive item uses accent; negative uses rust. There's never a rainbow.
- The dot matrix encodes magnitude as circle size: outline = low, ink fill = mid, accent fill = high.
- Every chart has a hover readout line underneath, so no value is conveyed by color alone.
- Histograms use a √count scale (labelled) so long tails stay visible.
- Baselines are 1 px ink; grids are absent or dotted.

## Not yet done

- Dark theme: add a `[data-theme="dark"]` token block and re-validate the chart colors against the dark surface.
- Reduced-motion and high-contrast (forced-colors) passes.
