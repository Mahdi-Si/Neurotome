export function compact(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a >= 1e9) return `${(n / 1e9).toFixed(digits)} G`;
  if (a >= 1e6) return `${(n / 1e6).toFixed(digits)} M`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(digits)} K`;
  return `${Math.round(n)}`;
}

export function bytes(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

export function num(n: number | null | undefined, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return n.toExponential(1);
  return n.toFixed(digits);
}

export function signed(n: number, digits = 3): string {
  return `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(digits)}`;
}

export const shape = (s: number[] | null | undefined) => (s && s.length ? s.join("×") : "—");

/** Drop a leading batch dim of 1 for compact display. */
export const shapeShort = (s: number[] | null | undefined) =>
  s && s.length > 1 && s[0] === 1 ? s.slice(1).join("×") : shape(s);

export function pct(n: number | null | undefined): string {
  return n == null ? "—" : `${Math.round(n * 100)}%`;
}

/** Human name for a graph node: its module path, or its op label when it has no module. */
export const displayName = (n: { id: string; path: string; label: string }) => (n.path === n.id ? n.label : n.path);
