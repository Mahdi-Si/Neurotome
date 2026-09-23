// Light / dark theme: the choice (localStorage, else the system), token colours for canvases, WCAG contrast.
import { useSyncExternalStore } from "react";

export type Theme = "light" | "dark";
export const THEME_KEY = "neurotome.theme";
const EVENT = "neurotome:theme";

/** A stored "light" / "dark" wins, anything else follows the system. */
export const resolveTheme = (stored: string | null, systemDark: boolean): Theme =>
  stored === "light" || stored === "dark" ? stored : systemDark ? "dark" : "light";

/** What to store after choosing `next`: null (follow the system again) when it equals the system theme. */
export const storedFor = (next: Theme, system: Theme): Theme | null => (next === system ? null : next);

/** "#rgb", "#rrggbb", "rgb(r, g, b)", "rgba(r, g, b, a)" (also space separated) → [r, g, b]. */
export function parseColor(s: string): [number, number, number] {
  const t = s.trim();
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(t);
  if (hex) {
    const h = hex[1].length === 3 ? hex[1].replace(/./g, "$&$&") : hex[1];
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
  }
  const rgb = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i.exec(t);
  if (rgb) return [1, 2, 3].map((i) => Math.round(Number(rgb[i]))) as [number, number, number];
  throw new Error(`Unsupported colour "${s}"`);
}

const luminance = (c: string) => {
  const [r, g, b] = parseColor(c).map((v) => { const x = v / 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

/** WCAG 2 contrast ratio, 1..21. */
export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

export type Tokens = Record<"paper" | "panel" | "ink" | "ink2" | "line" | "lineFaint" | "accent" | "accentSoft" | "negative" | "fontMono", string>;
const VARS: [keyof Tokens, string][] = [
  ["paper", "--paper"], ["panel", "--panel"], ["ink", "--ink"], ["ink2", "--ink-2"], ["line", "--line"],
  ["lineFaint", "--line-faint"], ["accent", "--accent"], ["accentSoft", "--accent-soft"], ["negative", "--negative"],
];

/** Token colours resolved through a hidden probe (forced-color-adjust: none), so hex, rgb() and system colours
 * all come back as "rgb(r, g, b)" and canvas drawing matches the DOM. */
export function readTokens(): Tokens {
  const probe = document.createElement("span");
  probe.style.cssText = "position:absolute;visibility:hidden;forced-color-adjust:none";
  document.body.appendChild(probe);
  const out = { fontMono: getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim() || "monospace" } as Tokens;
  for (const [k, v] of VARS) { probe.style.color = `var(${v})`; out[k] = getComputedStyle(probe).color; }
  probe.remove();
  return out;
}

let choice: Theme | null = (() => {
  try { const s = localStorage.getItem(THEME_KEY); return s === "light" || s === "dark" ? s : null; } catch { return null; }
})();
let mqs: MediaQueryList[] | null = null;
const queries = () => (mqs ??= ["(prefers-color-scheme: dark)", "(forced-colors: active)"].map((q) => window.matchMedia(q)));

function subscribe(cb: () => void) {
  window.addEventListener(EVENT, cb);
  queries().forEach((m) => m.addEventListener("change", cb));
  return () => { window.removeEventListener(EVENT, cb); queries().forEach((m) => m.removeEventListener("change", cb)); };
}

const currentTheme = () => resolveTheme(choice, queries()[0].matches);

/** Switch theme: data-theme mirrors the stored choice (none = follow prefers-color-scheme), like index.html's script. */
function toggle() {
  const next: Theme = currentTheme() === "dark" ? "light" : "dark";
  choice = storedFor(next, queries()[0].matches ? "dark" : "light");
  try { if (choice) localStorage.setItem(THEME_KEY, choice); else localStorage.removeItem(THEME_KEY); } catch { /* session only */ }
  if (choice) document.documentElement.dataset.theme = choice;
  else delete document.documentElement.dataset.theme;
  window.dispatchEvent(new Event(EVENT));
}

export function useTheme(): [Theme, () => void] {
  return [useSyncExternalStore(subscribe, currentTheme), toggle];
}

let cache: { key: string; t: Tokens } | null = null;
function currentTokens(): Tokens {
  const key = [document.documentElement.dataset.theme, ...queries().map((m) => m.matches)].join("|");
  if (cache?.key !== key) cache = { key, t: readTokens() };
  return cache.t;
}

/** Token colours for canvases; a new object after every theme or forced-colours change, so draw effects rerun. */
export function useTokens(): Tokens {
  return useSyncExternalStore(subscribe, currentTokens);
}
