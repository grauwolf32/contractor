import type { ReactNode } from "react";

/*
 * Every stroke icon of the UI, drawn on a 24 x 24 grid in the V3B mockup
 * style (docs/design/ui/mockups/v3b). Routes render <Icon name=… /> instead
 * of their own <svg>; lint rejects new inline SVG icons under src/routes.
 */
const shapes = {
  copy: (
    <>
      <rect x="8" y="8" width="12" height="13" rx="2" />
      <path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" />
    </>
  ),
  play: <path d="m7 4 14 8-14 8V4Z" />,
  pause: <path d="M8 4v16M16 4v16" />,
  stop: <rect x="5" y="5" width="14" height="14" rx="2" />,
  home: <path d="m3 10 9-7 9 7M5 9v12h5v-6h4v6h5V9" />,
  projects: (
    <path d="M3 7V5a2 2 0 0 1 2-2h4l3 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" />
  ),
  runs: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m10 8 6 4-6 4V8Z" />
    </>
  ),
  catalog: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </>
  ),
  artifacts: (
    <path d="m3 7.5 9-4.5 9 4.5v9L12 21l-9-4.5v-9Zm0 0 9 4.5 9-4.5M12 12v9M7.5 5.25l9 4.5" />
  ),
  evals: (
    <path d="M9 3h6M10 3v6l-5.6 9A2 2 0 0 0 6.1 21h11.8a2 2 0 0 0 1.7-3L14 9V3M7.5 14h9" />
  ),
  operations: (
    <>
      <rect x="3" y="3" width="18" height="7" rx="2" />
      <rect x="3" y="14" width="18" height="7" rx="2" />
      <path d="M7 6.5h.01M7 17.5h.01M12 6.5h5M12 17.5h5" />
    </>
  ),
  settings: (
    <>
      <path d="M3 6h4m4 0h10M3 12h10m4 0h4M3 18h4m4 0h10" />
      <circle cx="9" cy="6" r="2" />
      <circle cx="15" cy="12" r="2" />
      <circle cx="9" cy="18" r="2" />
    </>
  ),
  logout: (
    <path d="M10 4H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h5M9 12h12m-4-4 4 4-4 4" />
  ),
  "add-config": (
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Zm0 0v6h6M8 15h8m-4-4v8" />
  ),
  "add-credential": (
    <>
      <circle cx="7" cy="8" r="4" />
      <path d="m10 11 8 8h3v-3l-8-8M18 3v6m-3-3h6" />
    </>
  ),
  binding: (
    <path d="M10 13a5 5 0 0 0 7 .1l3-3a5 5 0 0 0-7.1-7.1l-1.7 1.7M14 11a5 5 0 0 0-7-.1l-3 3a5 5 0 0 0 7.1 7.1l1.7-1.7" />
  ),
  refresh: <path d="M20 12a8 8 0 1 1-2.35-5.65M20 4v5h-5" />,
  delete: <path d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13M10 11v5m4-5v5" />,
  // Shell rail and command palette.
  inbox: (
    <>
      <path d="M3.5 13.5h4.6l1.4 2.5h5l1.4-2.5h4.6" />
      <path d="M6 5h12l2.5 8.5V19h-17v-5.5z" />
    </>
  ),
  checks: (
    <>
      <path d="M12 3.5l7 2.8v5.2c0 4.3-2.9 7.9-7 9-4.1-1.1-7-4.7-7-9V6.3z" />
      <path d="M9.2 12l2 2 3.8-3.8" />
    </>
  ),
  issues: (
    <>
      <path d="M12 3.2l8.8 8.8-8.8 8.8L3.2 12z" />
      <path d="M12 8.2v4.6M12 15.8v.01" />
    </>
  ),
  reports: (
    <>
      <path d="M6.5 3.5h7.5l4.5 4.5v12.5h-12z" />
      <path d="M14 3.5V8h4.5" />
      <path d="M9.5 12.5h6M9.5 16h6" />
    </>
  ),
  library: (
    <path d="M4.5 4.5h4v15h-4zM10 4.5h4v15h-4zM15.6 5.4l3.5-.9 2.4 14.5-3.5.9z" />
  ),
  checklist: (
    <path d="M10 6.5h10M10 12h10M10 17.5h10M3.5 6.5l1.5 1.5 2.5-2.5M3.5 12l1.5 1.5 2.5-2.5M3.5 17.5 5 19l2.5-2.5" />
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="6.5" />
      <path d="M16 16l4 4" />
    </>
  ),
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  close: <path d="M6 6l12 12M18 6 6 18" />,
  plus: <path d="M12 5v14M5 12h14" />,
  user: (
    <>
      <circle cx="12" cy="9" r="3.5" />
      <path d="M5.5 19.5c1.2-3.2 3.6-4.8 6.5-4.8s5.3 1.6 6.5 4.8" />
    </>
  ),
  // Disclosure chevrons (rotate the right one with CSS for open states),
  // arrows and small actions.
  "chevron-right": <path d="M9.5 6l6 6-6 6" />,
  "chevron-up": <path d="M6 14.5l6-6 6 6" />,
  "chevron-down": <path d="M6 9.5l6 6 6-6" />,
  "arrow-down": <path d="M12 5v14M6.5 13.5L12 19l5.5-5.5" />,
  open: (
    <path d="M13.5 4.5h6v6M19.5 4.5L11 13M17.5 14v4.5a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1v-11a1 1 0 0 1 1-1H10" />
  ),
  upload: (
    <>
      <path d="M12 15V4.5M7.5 9l4.5-4.5L16.5 9" />
      <path d="M4.5 15.5v3a1.5 1.5 0 0 0 1.5 1.5h12a1.5 1.5 0 0 0 1.5-1.5v-3" />
    </>
  ),
  download: (
    <>
      <path d="M12 4.5V15M7.5 10.5 12 15l4.5-4.5" />
      <path d="M4.5 19.5h15" />
    </>
  ),
  history: (
    <>
      <path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3" />
      <path d="M4.5 4.5v3h3M12 8v4l2.8 1.8" />
    </>
  ),
  lock: (
    <>
      <rect x="5" y="10.5" width="14" height="9.5" rx="2" />
      <path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5" />
    </>
  ),
  git: (
    <>
      <circle cx="6" cy="5" r="2" />
      <circle cx="6" cy="19" r="2" />
      <circle cx="18" cy="6" r="2" />
      <path d="M6 7v10M18 8v2a5 5 0 0 1-5 5H6" />
    </>
  ),
  list: (
    <path d="M9 6.5h10.5M9 12h10.5M9 17.5h10.5M4.5 6.5h.01M4.5 12h.01M4.5 17.5h.01" />
  ),
  bulb: (
    <>
      <path d="M9 17.5h6M10 20.5h4" />
      <path d="M12 3.5a5.8 5.8 0 0 0-3.4 10.5c.6.5.9 1.1.9 1.8v.2h5v-.2c0-.7.3-1.3.9-1.8A5.8 5.8 0 0 0 12 3.5z" />
    </>
  ),
  // Decisions: confirm, not an issue, needs evidence.
  check: <path d="M5 12.5l4.2 4.2L19 7" />,
  "x-circle": (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M9.2 9.2l5.6 5.6M14.8 9.2l-5.6 5.6" />
    </>
  ),
  "question-circle": (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M9.7 9.6a2.4 2.4 0 0 1 4.6 1c0 1.7-2.3 2-2.3 3.4M12 16.8v.01" />
    </>
  ),
  // Material kinds (routes/artifacts/kinds.ts), a project's live target and
  // a Run's published result.
  source: <path d="M8.5 8l-4 4 4 4M15.5 8l4 4-4 4" />,
  api: (
    <path d="M8.5 4.5H7A1.5 1.5 0 0 0 5.5 6v4L4 12l1.5 2v4A1.5 1.5 0 0 0 7 19.5h1.5M15.5 4.5H17A1.5 1.5 0 0 1 18.5 6v4l1.5 2-1.5 2v4a1.5 1.5 0 0 1-1.5 1.5h-1.5" />
  ),
  architecture: (
    <>
      <rect x="3.5" y="4" width="7" height="5" rx="1" />
      <rect x="13.5" y="15" width="7" height="5" rx="1" />
      <path d="M10.5 6.5h4.5a2 2 0 0 1 2 2V15M7 9v5a3 3 0 0 0 3 3h3.5" />
    </>
  ),
  docs: (
    <>
      <path d="M6.5 3.5h7.5l4.5 4.5v12.5h-12z" />
      <path d="M14 3.5V8h4.5M9.5 12.5h6M9.5 16h6" />
    </>
  ),
  diffs: (
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="2.5" />
      <path d="M8.5 6.5v5M6 9h5M13 15.5h5" />
    </>
  ),
  other: (
    <>
      <path d="M4 7.5h6l2 2h8v10H4z" />
      <path d="M4 7.5v-3h6l2 3" />
    </>
  ),
  target: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M3.5 12h17M12 3.5c2.4 2.4 3.5 5.2 3.5 8.5s-1.1 6.1-3.5 8.5c-2.4-2.4-3.5-5.2-3.5-8.5s1.1-6.1 3.5-8.5z" />
    </>
  ),
  result: (
    <path d="M6.5 3.5h7.5l4.5 4.5v12.5h-12zM14 3.5V8h4.5M9.2 14l2 2 3.8-3.8" />
  ),
} satisfies Record<string, ReactNode>;

export type IconName = keyof typeof shapes;

/**
 * A stroke icon. Decorative by default (aria-hidden, the words next to it
 * carry the meaning); with `label` it is an image named by the label.
 */
export function Icon({
  name,
  size = 20,
  strokeWidth = 1.8,
  label,
  className,
}: {
  name: IconName;
  /** Width and height in px; default 20. */
  size?: number | undefined;
  /** Stroke width on the 24 px grid; default 1.8, chevrons use 2. */
  strokeWidth?: number | undefined;
  /** Accessible name; omit for a decorative icon. */
  label?: string | undefined;
  /** Added after "app-icon", e.g. a class that rotates a chevron. */
  className?: string | undefined;
}) {
  return (
    <svg
      className={className === undefined ? "app-icon" : `app-icon ${className}`}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      focusable="false"
      {...(label === undefined
        ? { "aria-hidden": true }
        : { role: "img", "aria-label": label })}
    >
      {shapes[name]}
    </svg>
  );
}
