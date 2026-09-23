const s = { fill: "none", stroke: "currentColor" } as const;

export const Logo = ({ size = 18 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 18 18" {...s} strokeWidth={1.2} aria-hidden="true">
    <circle cx="9" cy="9" r="7.4" />
    <circle cx="9" cy="9" r="2.4" fill="currentColor" />
    <path d="M9 0v4M9 14v4M0 9h4M14 9h4" />
  </svg>
);

export const Crosshair = () => (
  <svg width="11" height="11" viewBox="0 0 11 11" {...s} strokeWidth={1} aria-hidden="true">
    <circle cx="5.5" cy="5.5" r="3" />
    <path d="M5.5 0v11M0 5.5h11" />
  </svg>
);

export const IconHome = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" {...s} strokeWidth={1.1} aria-hidden="true"><path d="M2.5 7.2 8 2.8l5.5 4.4v6.3h-11z" /></svg>
);
export const IconGraph = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" {...s} strokeWidth={1.1} aria-hidden="true">
    <circle cx="4" cy="4" r="1.8" /><circle cx="12" cy="8" r="1.8" /><circle cx="4" cy="12" r="1.8" />
    <path d="M5.6 4.8l4.8 2.4M5.6 11.2l4.8-2.4" />
  </svg>
);
export const IconTensor = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" {...s} strokeWidth={1.1} aria-hidden="true">
    <rect x="2.5" y="2.5" width="4.5" height="4.5" /><rect x="9" y="2.5" width="4.5" height="4.5" />
    <rect x="2.5" y="9" width="4.5" height="4.5" /><rect x="9" y="9" width="4.5" height="4.5" />
  </svg>
);
export const IconTrace = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" {...s} strokeWidth={1.1} aria-hidden="true">
    <path d="M2.5 13.5v-4h5v-4h6v-3" /><circle cx="2.5" cy="13.5" r="1" />
  </svg>
);
export const IconArrow = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" {...s} strokeWidth={1.3} aria-hidden="true"><path d="M2 7h10M8 3l4 4-4 4" /></svg>
);
export const IconSearch = () => (
  <svg width="12" height="12" viewBox="0 0 12 12" {...s} strokeWidth={1.1} aria-hidden="true">
    <circle cx="5" cy="5" r="3.6" /><path d="M7.7 7.7 11 11" />
  </svg>
);
