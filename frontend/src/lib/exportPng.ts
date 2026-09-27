// PNG export for the charts: SVGs are rasterised with their CSS variables resolved, canvases are
// upscaled nearest-neighbour, and everything sits on the panel ground. Pure helpers first (tested).

/** Replace every var(--name) / var(--name, fallback) with lookup("--name") || fallback || "". */
export function resolveCssVars(svgText: string, lookup: (name: string) => string): string {
  return svgText.replace(/var\(\s*(--[\w-]+)\s*(?:,\s*([^()]*?))?\s*\)/g,
    (_, name: string, fallback?: string) => lookup(name).trim() || fallback?.trim() || "");
}

/** Safe file name from parts: empty parts dropped, joined by "_", anything odd → "-". */
export function fileName(parts: (string | number | null | undefined)[], ext: string): string {
  const clean = parts.map((p) => String(p ?? "").replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, ""));
  return `${clean.filter(Boolean).join("_")}.${ext}`;
}

export function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name);

function ground(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h));
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = cssVar("--panel").trim();   // the current theme's ground
  ctx.fillRect(0, 0, c.width, c.height);
  return [c, ctx];
}

const save = (c: HTMLCanvasElement, name: string) => new Promise<void>((resolve, reject) =>
  c.toBlob((b) => (b ? (download(b, name), resolve()) : reject(new Error("PNG encoding failed"))), "image/png"));

/** Axis ranges as printed (see <Axes> in charts.tsx): y.from is the top value. */
export interface AxisText { from: string; to: string; label: string }
export interface AxisPair { x?: AxisText; y?: AxisText }

/** Gutters in CSS px around a chart, shared by <Axes> and the PNG export: top = a 14 px row for the y label (horizontal,
 * above the numbers, so it is never clipped however short the chart), left = the y numbers (6 px a character) + 8 px
 * for the tick, bottom = 16 px for the x row. */
export function axisGutters(a?: AxisPair): { left: number; top: number; bottom: number } {
  return {
    left: a?.y ? 8 + 6 * Math.max(a.y.from.length, a.y.to.length, 1) : 0,
    top: a?.y ? 14 : 0,
    bottom: a?.x ? 16 : 0,
  };
}

/** The ranges drawn around a cw×ch chart placed at (L, T), at `s` output px per CSS px. */
function drawAxes(ctx: CanvasRenderingContext2D, { x, y }: AxisPair, L: number, T: number, cw: number, ch: number, s: number) {
  const ink2 = cssVar("--ink-2").trim(), tick = cssVar("--line").trim();
  ctx.save();
  ctx.font = `${10 * s}px ${cssVar("--font-mono").trim() || "monospace"}`;
  const text = (t: string, px: number, py: number, align: CanvasTextAlign, base: CanvasTextBaseline) => {
    ctx.fillStyle = ink2; ctx.textAlign = align; ctx.textBaseline = base; ctx.fillText(t, px, py);
  };
  if (y) {
    text(y.label, 0, 0, "left", "top");
    text(y.from, L - 8 * s, T, "right", "top");
    text(y.to, L - 8 * s, T + ch, "right", "bottom");
    ctx.fillStyle = tick; ctx.fillRect(L - 4 * s, T, 4 * s, s); ctx.fillRect(L - 4 * s, T + ch - s, 4 * s, s);
  }
  if (x) {
    const ty = T + ch + 5 * s;
    text(x.from, L, ty, "left", "top"); text(x.label, L + cw / 2, ty, "center", "top"); text(x.to, L + cw, ty, "right", "top");
    ctx.fillStyle = tick; ctx.fillRect(L, T + ch, s, 4 * s); ctx.fillRect(L + cw - s, T + ch, s, 4 * s);
  }
  ctx.restore();
}

/** A panel ground for a cw×ch chart plus its axis gutters; returns the chart's left and top offsets. */
function framed(cw: number, ch: number, s: number, axes?: AxisPair): [HTMLCanvasElement, CanvasRenderingContext2D, number, number] {
  const g = axisGutters(axes), L = Math.round(g.left * s), T = Math.round(g.top * s);
  const [c, ctx] = ground(cw + L, ch + T + g.bottom * s);
  if (axes) drawAxes(ctx, axes, L, T, cw, ch, s);
  return [c, ctx, L, T];
}

const size = (el: HTMLImageElement | HTMLCanvasElement): [number, number] =>
  el instanceof HTMLImageElement ? [el.naturalWidth, el.naturalHeight] : [el.width, el.height];

export async function exportSvg(svg: SVGSVGElement, name: string, scale = 2, axes?: AxisPair): Promise<void> {
  const { width, height } = svg.getBoundingClientRect();
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("x", "-10000"); bg.setAttribute("y", "-10000");
  bg.setAttribute("width", "20000"); bg.setAttribute("height", "20000");
  bg.setAttribute("fill", "var(--panel)");
  clone.insertBefore(bg, clone.firstChild);
  const text = resolveCssVars(new XMLSerializer().serializeToString(clone), cssVar);
  const img = new Image();
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(text)}`;
  await img.decode();
  const cw = Math.round(width * scale), ch = Math.round(height * scale);
  const [c, ctx, L, T] = framed(cw, ch, scale, axes);
  ctx.drawImage(img, L, T, cw, ch);
  await save(c, name);
}

/** A canvas upscaled nearest-neighbour to ≥ minWidth px, with its axes at 2× text size. */
export async function exportCanvas(canvas: HTMLCanvasElement, name: string, minWidth = 480, axes?: AxisPair): Promise<void> {
  const k = Math.max(1, Math.ceil(minWidth / Math.max(1, canvas.width)));
  const [c, ctx, L, T] = framed(canvas.width * k, canvas.height * k, 2, axes);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(canvas, L, T, canvas.width * k, canvas.height * k);
  await save(c, name);
}

export async function exportLayers(
  layers: { el: HTMLImageElement | HTMLCanvasElement; alpha: number }[], name: string, axes?: AxisPair,
): Promise<void> {
  if (!layers.length) return;
  const [w, h] = size(layers[0].el);
  const [c, ctx, L, T] = framed(w * 2, h * 2, 2, axes);
  for (const { el, alpha } of layers) {
    ctx.globalAlpha = alpha;
    ctx.drawImage(el, L, T, w * 2, h * 2);
  }
  await save(c, name);
}

/** Small multiples in a grid; `note` (the shared axis ranges of every tile) is printed centred under them. */
export async function exportTiles(canvases: HTMLCanvasElement[], cols: number, name: string, note?: string): Promise<void> {
  if (!canvases.length) return;
  const T = 96, G = 8, n = Math.max(1, Math.min(cols, canvases.length)), rows = Math.ceil(canvases.length / n);
  const H = Math.round((T * canvases[0].height) / Math.max(1, canvases[0].width));   // tiles keep the map's aspect
  const [c, ctx] = framed(n * T + (n - 1) * G, rows * H + (rows - 1) * G, 1, note ? { x: { from: "", to: "", label: note } } : undefined);
  ctx.imageSmoothingEnabled = false;
  canvases.forEach((t, i) => ctx.drawImage(t, (i % n) * (T + G), Math.floor(i / n) * (H + G), T, H));
  await save(c, name);
}
