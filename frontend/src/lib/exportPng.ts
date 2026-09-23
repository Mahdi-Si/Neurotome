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

const size = (el: HTMLImageElement | HTMLCanvasElement): [number, number] =>
  el instanceof HTMLImageElement ? [el.naturalWidth, el.naturalHeight] : [el.width, el.height];

export async function exportSvg(svg: SVGSVGElement, name: string, scale = 2): Promise<void> {
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
  const [c, ctx] = ground(width * scale, height * scale);
  ctx.drawImage(img, 0, 0, c.width, c.height);
  await save(c, name);
}

export async function exportCanvas(canvas: HTMLCanvasElement, name: string, minWidth = 480): Promise<void> {
  const k = Math.max(1, Math.ceil(minWidth / Math.max(1, canvas.width)));
  const [c, ctx] = ground(canvas.width * k, canvas.height * k);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(canvas, 0, 0, c.width, c.height);
  await save(c, name);
}

export async function exportLayers(
  layers: { el: HTMLImageElement | HTMLCanvasElement; alpha: number }[], name: string,
): Promise<void> {
  if (!layers.length) return;
  const [w, h] = size(layers[0].el);
  const [c, ctx] = ground(w * 2, h * 2);
  for (const { el, alpha } of layers) {
    ctx.globalAlpha = alpha;
    ctx.drawImage(el, 0, 0, c.width, c.height);
  }
  await save(c, name);
}

export async function exportTiles(canvases: HTMLCanvasElement[], cols: number, name: string): Promise<void> {
  if (!canvases.length) return;
  const T = 96, G = 8, n = Math.max(1, Math.min(cols, canvases.length)), rows = Math.ceil(canvases.length / n);
  const [c, ctx] = ground(n * T + (n - 1) * G, rows * T + (rows - 1) * G);
  ctx.imageSmoothingEnabled = false;
  canvases.forEach((t, i) => ctx.drawImage(t, (i % n) * (T + G), Math.floor(i / n) * (T + G), T, T));
  await save(c, name);
}
